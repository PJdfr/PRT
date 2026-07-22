"""Backtest block: end-to-end run, hand-checked PnL arithmetic, persistence."""

import numpy as np
import pandas as pd

from prt.backtest import run_backtest
from prt.backtest.engine import _pnl_one_instrument


def test_pnl_arithmetic_hand_check():
    """Constant long 1M USD: pnl_t+1 = 1M x ret_t+1, minus entry cost once."""
    idx = pd.bdate_range("2026-01-05", periods=5)
    prices = pd.Series([100, 101, 100, 102, 102.5], index=idx, dtype=float)
    returns = prices.pct_change()
    held = pd.DataFrame({"held_notional": [1e6] * 5}, index=idx)
    pnl = _pnl_one_instrument(held, returns, cost_bps=1.0)
    # day 2 pnl: 1M x 1% minus 1M x 1bp entry cost booked with day-1 position
    assert np.isclose(pnl.iloc[1], 1e6 * 0.01 - 1e6 * 1e-4)
    assert np.isclose(pnl.iloc[2], 1e6 * (100 / 101 - 1))


def test_run_backtest_end_to_end(ingested_db, config):
    result = run_backtest(ingested_db, config)
    pnl = result.pnl_fund.dropna()
    assert len(pnl) > 400
    assert np.isfinite(pnl).all()
    assert set(result.pnl_by_signal.columns) == set(config.signal_weights)
    assert set(result.pnl_by_instrument.columns) == set(config.universe)
    assert result.stats["n_days"] == len(pnl)
    assert 0 < result.stats["ann_vol"] < 1.0

    # persisted and re-loadable
    fund = ingested_db.backtest_pnl(result.run_id, "fund")
    assert np.isclose(fund["fund"].sum(), pnl.sum())
    sub = ingested_db.backtest_pnl(result.run_id, "signal")
    assert set(sub.columns) == set(config.signal_weights)


def test_realised_vol_in_target_ballpark(ingested_db, config):
    """10% vol target with IDM: realised fund vol should be within a loose
    band around target (synthetic data, diversification, buffering)."""
    result = run_backtest(ingested_db, config, store=False)
    vol = result.stats["ann_vol"]
    assert 0.02 < vol < 0.30, vol
