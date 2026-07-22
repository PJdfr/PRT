"""Signal Lab block: correlation matrices and marginal Sharpe."""

import numpy as np
import pandas as pd

from prt.backtest import run_backtest
from prt.signals.lab import (
    forecast_correlation,
    marginal_sharpe,
    sharpe,
    subsystem_correlation,
)


def test_forecast_correlation(ingested_db, config):
    run_backtest(ingested_db, config)  # persists forecasts
    corr = forecast_correlation(ingested_db, ["momentum", "carry"])
    assert corr.shape == (2, 2)
    assert np.allclose(np.diag(corr), 1.0)
    assert corr.loc["momentum", "carry"] == corr.loc["carry", "momentum"]
    assert abs(corr.loc["momentum", "carry"]) < 1.0


def test_subsystem_correlation_and_marginal(ingested_db, config):
    result = run_backtest(ingested_db, config, store=False)
    pnls = {c: result.pnl_by_signal[c] for c in result.pnl_by_signal.columns}
    corr = subsystem_correlation(pnls)
    assert np.allclose(np.diag(corr), 1.0)

    # a perfectly duplicated signal adds ~zero marginal sharpe
    dup = pnls["momentum"]
    report = marginal_sharpe({"momentum": pnls["momentum"]}, dup)
    assert np.isclose(report["sharpe_with"], report["sharpe_without"], atol=1e-9)
    assert set(report) == {"sharpe_without", "sharpe_with", "marginal", "candidate_standalone"}


def test_sharpe_helper():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(0.001, 0.01, 500))
    assert np.isfinite(sharpe(s))
    assert np.isnan(sharpe(pd.Series([0.0] * 30)))
