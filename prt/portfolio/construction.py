"""Portfolio construction: forecasts -> notional -> contracts.

Generic pipeline, no strategy-specific code:
  1. combine per-instrument forecasts across signals (weights, IDM, cap);
  2. vol-target: notional = f/f_avg * weight_i * capital * IDM * target_vol
     / instrument vol (EWMA of adjusted returns);
  3. buffering: only move the held position when the target drifts out of a
     no-trade band around it (fraction of the average position size);
  4. translate: futures -> integer contracts (notional / price*pv*fx),
     fx -> pair notional in USD.

Prices used for contract translation are the adjusted series: its *recent*
levels equal the actual front contract price (backward adjustment leaves
the present intact), which is what matters for live sizing; historical
contract counts in backtests are an approximation (PnL uses returns x
notional and is unaffected).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prt.config import Config
from prt.db import Database
from prt.signals.dataview import DataView

TRADING_DAYS = 252


def instrument_vol(prices: pd.Series, span: int, min_periods: int = 20) -> pd.Series:
    """Annualised EWMA vol of a price series' returns."""
    vol = prices.pct_change().ewm(span=span, min_periods=min_periods).std() * np.sqrt(TRADING_DAYS)
    return vol.where(vol > 0)


def combine_forecasts(
    per_signal: dict[str, pd.Series], weights: dict[str, float], config: Config
) -> pd.Series:
    """Weighted average of forecasts x IDM, clipped to the forecast cap.

    A signal with no forecast on a date (e.g. not enough history yet)
    contributes zero; dates where NO signal has a forecast stay NaN.
    """
    active = {s: f for s, f in per_signal.items() if s in weights and weights[s] > 0}
    if not active:
        raise ValueError("no active signals to combine")
    total_w = sum(weights[s] for s in active)
    stacked = pd.concat(active, axis=1)
    combined = (stacked.fillna(0.0) * pd.Series({s: weights[s] / total_w for s in active})).sum(axis=1)
    combined[stacked.isna().all(axis=1)] = np.nan
    return (combined * config.fund.idm).clip(-config.fund.forecast_cap, config.fund.forecast_cap)


def _buffered(target: pd.Series, band: pd.Series) -> pd.Series:
    """Apply the no-trade buffer sequentially: hold until |target-held| > band."""
    held = np.zeros(len(target))
    cur = 0.0
    tgt = target.to_numpy()
    bnd = band.to_numpy()
    for i in range(len(tgt)):
        if np.isnan(tgt[i]):
            held[i] = cur
            continue
        b = bnd[i] if not np.isnan(bnd[i]) else 0.0
        if abs(tgt[i] - cur) > b:
            cur = tgt[i]
        held[i] = cur
    return pd.Series(held, index=target.index)


def positions_from_forecasts(
    combined: pd.Series,
    prices: pd.Series,
    inst_weight: float,
    config: Config,
    fx_to_usd: pd.Series | float = 1.0,
    point_value: float | None = None,
) -> pd.DataFrame:
    """Per-date target/held notional (USD) and contracts for one instrument.

    Columns: forecast, vol, target_notional, held_notional, contracts.
    """
    f = config.fund
    vol = instrument_vol(prices, f.vol_ewma_span)
    risk_budget = inst_weight * f.capital_usd * f.idm * f.target_vol
    avg_pos = risk_budget / vol                       # notional at |forecast| = f_avg
    target = (combined / f.forecast_avg) * avg_pos
    band = f.buffer_fraction * avg_pos
    held = _buffered(target, band)

    out = pd.DataFrame(
        {"forecast": combined, "vol": vol, "target_notional": target, "held_notional": held}
    )
    if point_value:
        px_usd = prices * point_value * fx_to_usd
        out["contracts"] = (held / px_usd).round()
        out["held_notional"] = out["contracts"] * px_usd   # notional of the integer position
    else:
        out["contracts"] = np.nan
    return out


def fx_to_usd_series(view: DataView, config: Config, currency: str, for_inst: str) -> pd.Series | float:
    """Point-in-time ccy->USD conversion series aligned on `for_inst`'s grid."""
    if currency == "USD":
        return 1.0
    conv = config.fx_conversion.get(currency)
    if conv is None or conv["pair"] not in config.instruments:
        raise ValueError(f"no fx conversion configured for {currency}")
    spot = view.prices(conv["pair"], "spot", for_inst=for_inst)
    return 1.0 / spot if conv.get("invert") else spot


def compute_targets(
    db: Database,
    config: Config,
    view: DataView,
    forecasts: dict[str, dict[str, pd.Series]],
    weights: dict[str, float] | None = None,
    persist: bool = False,
) -> dict[str, pd.DataFrame]:
    """Full-universe positions per instrument (all dates).

    forecasts: {signal: {instrument: forecast series}} from the signal engine.
    Returns {instrument: positions frame}; optionally persists the latest
    exec date of each instrument into the targets table.
    """
    weights = weights if weights is not None else config.signal_weights
    n = len(config.instruments)
    result: dict[str, pd.DataFrame] = {}
    latest_rows = []
    for inst_id, inst in config.instruments.items():
        per_signal = {s: f[inst_id] for s, f in forecasts.items() if inst_id in f}
        if not per_signal:
            continue
        combined = combine_forecasts(per_signal, weights, config)
        prices = view.prices(inst_id, "adjusted")
        fx = fx_to_usd_series(view, config, inst.currency, inst_id)
        frame = positions_from_forecasts(
            combined, prices, 1.0 / n, config, fx_to_usd=fx, point_value=inst.point_value
        )
        result[inst_id] = frame
        last = frame.dropna(subset=["target_notional"]).tail(1)
        if not last.empty:
            latest_rows.append(
                {
                    "instrument_id": inst_id,
                    "exec_date": last.index[-1],
                    "combined_forecast": last["forecast"].iloc[-1],
                    "target_notional": last["held_notional"].iloc[-1],
                    "target_contracts": last["contracts"].iloc[-1],
                    "price": prices.loc[last.index[-1]],
                }
            )
    if persist and latest_rows:
        db.write_targets(pd.DataFrame(latest_rows))
    return result
