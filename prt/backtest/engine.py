"""Daily backtest engine.

Convention (same as live): forecasts for exec date t only saw data known
before t's decision instant (enforced by the DataView); the position held
from close t earns the adjusted series' return from close t to close t+1;
each position change pays half-spread cost in bps of traded notional.

PnL is computed as returns x USD notional — never price points on the
(fictitious) historical levels of a ratio-adjusted series.

Per-signal subsystem PnLs (each signal traded standalone through the same
portfolio layer) are produced for the Signal Lab.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from prt.config import Config
from prt.db import Database
from prt.portfolio.construction import compute_targets
from prt.signals.base import compute_all_forecasts
from prt.signals.dataview import DataView
from prt.signals.lab import TRADING_DAYS


@dataclass
class BacktestResult:
    run_id: str
    pnl_fund: pd.Series
    pnl_by_instrument: pd.DataFrame
    pnl_by_signal: pd.DataFrame            # standalone subsystem PnLs
    stats: dict = field(default_factory=dict)


def _pnl_one_instrument(frame: pd.DataFrame, returns: pd.Series, cost_bps: float) -> pd.Series:
    """frame: positions frame (held_notional per exec date); returns: adjusted
    series returns indexed by date (close-to-close)."""
    held = frame["held_notional"].fillna(0.0)
    # position decided & executed at close t earns return close t -> close t+1
    ret_next = returns.reindex(held.index).shift(-1)
    gross = held * ret_next
    traded = held.diff().abs().fillna(held.abs())
    costs = traded * cost_bps * 1e-4
    return (gross - costs).shift(1)  # book the t->t+1 pnl on date t+1


def _portfolio_pnl(
    db: Database,
    config: Config,
    view: DataView,
    forecasts: dict[str, dict[str, pd.Series]],
    weights: dict[str, float],
) -> pd.DataFrame:
    frames = compute_targets(db, config, view, forecasts, weights=weights, persist=False)
    pnls = {}
    for inst_id, frame in frames.items():
        raw = db.series(inst_id, "adjusted")["value"]
        returns = raw.pct_change()
        pnls[inst_id] = _pnl_one_instrument(frame, returns, config.instrument(inst_id).cost_bps)
    return pd.DataFrame(pnls)


def run_backtest(
    db: Database,
    config: Config,
    signal_names: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    store: bool = True,
) -> BacktestResult:
    view = DataView(db, config)
    names = signal_names if signal_names is not None else list(config.signal_weights)
    forecasts = compute_all_forecasts(db, config, view, signal_names=names, persist=store)

    by_inst = _portfolio_pnl(db, config, view, forecasts, config.signal_weights)
    subsystems = {
        name: _portfolio_pnl(db, config, view, {name: forecasts[name]}, {name: 1.0}).sum(axis=1)
        for name in names
    }

    if start:
        by_inst = by_inst.loc[pd.Timestamp(start):]
        subsystems = {k: v.loc[pd.Timestamp(start):] for k, v in subsystems.items()}
    if end:
        by_inst = by_inst.loc[:pd.Timestamp(end)]
        subsystems = {k: v.loc[:pd.Timestamp(end)] for k, v in subsystems.items()}

    fund = by_inst.sum(axis=1)
    result = BacktestResult(
        run_id=uuid.uuid4().hex[:12],
        pnl_fund=fund,
        pnl_by_instrument=by_inst,
        pnl_by_signal=pd.DataFrame(subsystems),
        stats=compute_stats(fund, config),
    )
    if store:
        db.save_backtest(
            result.run_id,
            meta={"signals": names, "start": start, "end": end, "stats": result.stats},
            pnl_frames={
                "fund": fund.to_frame("fund"),
                "instrument": by_inst,
                "signal": result.pnl_by_signal,
            },
        )
    return result


def compute_stats(pnl: pd.Series, config: Config) -> dict:
    pnl = pnl.dropna()
    if pnl.empty:
        return {}
    capital = config.fund.capital_usd
    rets = pnl / capital
    cum = rets.cumsum()
    dd = (cum - cum.cummax()).min()
    std = rets.std()
    return {
        "ann_return": float(rets.mean() * TRADING_DAYS),
        "ann_vol": float(std * np.sqrt(TRADING_DAYS)),
        "sharpe": float(rets.mean() / std * np.sqrt(TRADING_DAYS)) if std > 0 else float("nan"),
        "max_drawdown": float(dd),
        "n_days": int(len(rets)),
    }
