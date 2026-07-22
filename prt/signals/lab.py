"""Signal Lab — the standard report to run BEFORE giving capital to a signal.

Three views of decorrelation:
  * forecast correlation: correlation of forecast values across the panel
    (instrument x date) between signals;
  * subsystem PnL correlation: each signal traded standalone through the
    same portfolio layer — the one that matters for allocation;
  * marginal Sharpe: what a candidate adds to the existing stack.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prt.db import Database

TRADING_DAYS = 252


def sharpe(pnl: pd.Series) -> float:
    pnl = pnl.dropna()
    if len(pnl) < 20 or pnl.std() == 0:
        return float("nan")
    return float(pnl.mean() / pnl.std() * np.sqrt(TRADING_DAYS))


def forecast_panel(db: Database, signal: str) -> pd.DataFrame:
    """dates x instruments forecast matrix for one signal."""
    df = db.forecasts(signal=signal)
    if df.empty:
        return pd.DataFrame()
    return df.pivot(index="exec_date", columns="instrument_id", values="value")


def forecast_correlation(db: Database, signals: list[str]) -> pd.DataFrame:
    """Signal x signal correlation of forecasts, computed on the stacked
    (instrument, date) panel restricted to common observations."""
    stacked = {}
    for s in signals:
        panel = forecast_panel(db, s)
        if not panel.empty:
            stacked[s] = panel.stack()
    if not stacked:
        return pd.DataFrame()
    return pd.DataFrame(stacked).corr()


def subsystem_correlation(subsystem_pnls: dict[str, pd.Series]) -> pd.DataFrame:
    """Correlation of standalone per-signal PnL streams (from the backtester)."""
    return pd.DataFrame(subsystem_pnls).corr()


def marginal_sharpe(existing_pnls: dict[str, pd.Series], candidate_pnl: pd.Series) -> dict:
    """Sharpe of the stack with vs without the candidate (equal-risk stack)."""
    def _norm(s: pd.Series) -> pd.Series:
        sd = s.std()
        return s / sd if sd and sd > 0 else s * 0.0

    base = pd.DataFrame({k: _norm(v) for k, v in existing_pnls.items()}).sum(axis=1)
    with_cand = base.add(_norm(candidate_pnl), fill_value=0.0)
    return {
        "sharpe_without": sharpe(base),
        "sharpe_with": sharpe(with_cand),
        "marginal": sharpe(with_cand) - sharpe(base),
        "candidate_standalone": sharpe(candidate_pnl),
    }
