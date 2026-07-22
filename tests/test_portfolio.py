"""Portfolio block: combination, vol targeting, contracts, buffering."""

import numpy as np
import pandas as pd

from prt.portfolio.construction import (
    _buffered,
    combine_forecasts,
    compute_targets,
    instrument_vol,
    positions_from_forecasts,
)
from prt.signals.base import compute_all_forecasts
from prt.signals.dataview import DataView


def _flat_series(val, n=300):
    return pd.Series(val, index=pd.bdate_range("2025-01-01", periods=n))


def test_combine_weights_and_cap(config):
    f1, f2 = _flat_series(4.0), _flat_series(-2.0)
    combined = combine_forecasts({"a": f1, "b": f2}, {"a": 0.5, "b": 0.5}, config)
    # (0.5*4 - 0.5*2) * idm 1.5 = 1.5
    assert np.isclose(combined.iloc[-1], 1.5)
    capped = combine_forecasts({"a": f1}, {"a": 1.0}, config)
    assert capped.max() <= config.fund.forecast_cap  # 4*1.5=6 -> capped at 5


def test_vol_targeting_magnitude(config):
    """At |forecast| = forecast_avg the notional should equal the risk budget
    over the instrument vol."""
    rng = np.random.default_rng(3)
    px = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, 400)),
                   index=pd.bdate_range("2025-01-01", periods=400))
    f = pd.Series(config.fund.forecast_avg, index=px.index)
    out = positions_from_forecasts(f, px, inst_weight=0.2, config=config, point_value=1000)
    vol = instrument_vol(px, config.fund.vol_ewma_span).iloc[-1]
    expected = 0.2 * config.fund.capital_usd * config.fund.idm * config.fund.target_vol / vol
    assert abs(out["target_notional"].iloc[-1] / expected - 1) < 1e-9
    assert float(out["contracts"].iloc[-1]) == round(out["contracts"].iloc[-1])  # integer


def test_buffering_avoids_churn():
    target = pd.Series([100.0, 101.0, 99.5, 120.0, 119.0], index=pd.bdate_range("2026-01-05", periods=5))
    band = pd.Series(10.0, index=target.index)
    held = _buffered(target, band)
    assert list(held) == [100.0, 100.0, 100.0, 120.0, 120.0]


def test_fx_position_is_notional_futures_are_contracts(ingested_db, config):
    view = DataView(ingested_db, config)
    forecasts = compute_all_forecasts(ingested_db, config, view, persist=False)
    frames = compute_targets(ingested_db, config, view, forecasts, persist=True)
    ty = frames["TY"].dropna(subset=["target_notional"])
    fx = frames["EURUSD"].dropna(subset=["target_notional"])
    assert (ty["contracts"].dropna() == ty["contracts"].dropna().round()).all()
    assert fx["contracts"].isna().all()          # fx: pure notional
    latest = ingested_db.latest_targets().set_index("instrument_id")
    assert set(latest.index) == set(config.universe)
    # persisted notional of a future must equal contracts x price x pv x fx
    row = latest.loc["TY"]
    assert np.isclose(row["target_notional"], row["target_contracts"] * row["price"] * 1000, rtol=1e-9)


def test_non_usd_future_uses_fx_conversion(ingested_db, config):
    view = DataView(ingested_db, config)
    forecasts = compute_all_forecasts(ingested_db, config, view, persist=False)
    frames = compute_targets(ingested_db, config, view, forecasts)
    rx = frames["RX"].dropna(subset=["contracts"])
    px = view.prices("RX", "adjusted")
    eurusd = view.prices("EURUSD", "spot", for_inst="RX")
    t = rx.index[-1]
    implied = rx.loc[t, "contracts"] * px.loc[t] * 1000 * eurusd.loc[t]
    assert np.isclose(rx.loc[t, "held_notional"], implied, rtol=1e-9)
