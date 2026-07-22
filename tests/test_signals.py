"""Signal engine: registry/plug-in mechanics, normalisation, built-ins."""

import numpy as np
import pandas as pd

from prt.instruments.master import close_ts_frame
from prt.signals import DataView, available_signals, compute_all_forecasts, get_signal
from prt.signals.base import Signal, scale_forecast


def test_builtin_signals_registered():
    assert {"momentum", "carry"} <= set(available_signals())


def test_forecasts_bounded_and_scaled(ingested_db, config):
    view = DataView(ingested_db, config)
    for name in ("momentum", "carry"):
        sig = get_signal(name)
        for inst_id in config.universe:
            f = sig.forecasts(view, config, inst_id).dropna()
            assert not f.empty, (name, inst_id)
            assert f.abs().max() <= config.fund.forecast_cap + 1e-9
            # average strength should be in the right ballpark of forecast_avg
            assert 0.5 < f.abs().mean() < 5.0, (name, inst_id, f.abs().mean())


def test_momentum_sign_on_trend(db, provider, config):
    """A steadily rising series must give a positive momentum forecast."""
    inst = config.instrument("TY")
    dates = pd.bdate_range("2024-01-01", periods=500)
    up = pd.DataFrame(
        {"value": 100 * np.cumprod(1 + 0.001 + 0.002 * np.sin(np.arange(500)))}, index=dates
    )
    db.upsert_series("TY", "adjusted", up.join(close_ts_frame(inst, dates, config)))
    view = DataView(db, config)
    f = get_signal("momentum").forecasts(view, config, "TY").dropna()
    assert f.iloc[-1] > 0


def test_scale_forecast_uses_only_past(config):
    """Normalisation is expanding: past values of the forecast do not change
    when future raw data is appended."""
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(7)
    raw = pd.Series(rng.normal(0, 1, 300), index=idx)
    full = scale_forecast(raw, config)
    trunc = scale_forecast(raw.iloc[:200], config)
    common = trunc.dropna().index
    assert np.allclose(full.loc[common], trunc.loc[common])


def test_plugin_signal_one_class_is_enough(ingested_db, config):
    """Adding a strategy = one Signal subclass. It must flow through the
    whole engine without touching anything else."""

    class MonthEndSeaso(Signal):
        name = "test_seaso"

        def compute(self, view, config, inst_id):
            cal = view.calendar(inst_id)
            return cal["is_month_end"].astype(float)  # long into month-end, calendar-only

    try:
        assert "test_seaso" in available_signals()
        out = compute_all_forecasts(ingested_db, config, signal_names=["test_seaso"], persist=True)
        f = out["test_seaso"]["TY"].dropna()
        assert not f.empty
        stored = ingested_db.forecasts(signal="test_seaso", instrument_id="TY")
        assert not stored.empty
        # calendar-only signal: month-end forecast is ON at month-end exec dates (no lag)
        cal = DataView(ingested_db, config).calendar("TY")
        month_ends = cal[cal["is_month_end"]].index.intersection(f.index)
        assert (f.loc[month_ends] > 0).all()
    finally:
        from prt.signals import base

        base._REGISTRY.pop("test_seaso", None)
