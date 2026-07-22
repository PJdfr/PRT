"""Signal engine: registry/plug-in mechanics, normalisation, and the two
production signals (12-block momentum, calendar-day seasonality)."""

import numpy as np
import pandas as pd

from prt.db import Database
from prt.instruments.master import close_ts_frame
from prt.signals import DataView, available_signals, compute_all_forecasts, get_signal
from prt.signals.base import Signal, scale_forecast
from prt.signals.seaso import _PROFILE_CACHE, day_of_year_365


def _truncated_copy(db, inst_ids, n_drop):
    """Same DB with the last n_drop rows of every series removed."""
    db2 = Database(":memory:")
    for inst_id in inst_ids:
        for st in ("adjusted", "g1_raw", "g2_raw", "spot"):
            s = db.series(inst_id, st)
            if not s.empty:
                db2.upsert_series(inst_id, st, s.iloc[:-n_drop] if n_drop else s)
    return db2


def test_builtin_signals_registered():
    assert {"momentum", "carry", "seaso"} <= set(available_signals())


def test_every_production_signal_has_a_whitepaper(config):
    """The dashboard methodology tab is fed by Signal.whitepaper: every
    signal with capital must document itself, formulas included."""
    for name in config.signal_weights:
        wp = get_signal(name).whitepaper
        assert wp.strip(), name
        assert "$$" in wp, f"{name}: whitepaper has no LaTeX formulas"


def test_forecasts_bounded_and_scaled(ingested_db, config):
    view = DataView(ingested_db, config)
    for name in ("momentum", "carry", "seaso"):
        sig = get_signal(name)
        for inst_id in config.universe:
            f = sig.forecasts(view, config, inst_id).dropna()
            assert not f.empty, (name, inst_id)
            assert f.abs().max() <= config.fund.forecast_cap + 1e-9
            assert 0.5 < f.abs().mean() < 5.0, (name, inst_id, f.abs().mean())


def test_momentum_sign_on_trend(db, config):
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


def test_momentum_is_point_in_time(ingested_db, config):
    """Momentum at date t must not change when later data is appended."""
    full = get_signal("momentum").compute(DataView(ingested_db, config), config, "TY")
    db2 = _truncated_copy(ingested_db, ["TY"], n_drop=60)
    trunc = get_signal("momentum").compute(DataView(db2, config), config, "TY")
    pd.testing.assert_series_equal(full.reindex(trunc.index), trunc)


def test_day_of_year_365_merges_feb29():
    idx = pd.DatetimeIndex(["2024-02-28", "2024-02-29", "2024-03-01", "2023-12-31"])
    assert list(day_of_year_365(idx)) == [59, 59, 60, 365]


def test_seaso_detects_calendar_pattern(db, config):
    """Inject +1% on calendar days 100..110 every year: the seasonality
    forecast must be strongly positive inside that window vs outside."""
    inst = config.instrument("TY")
    dates = pd.bdate_range("2018-01-01", "2025-12-31")
    rng = np.random.default_rng(11)
    doy = day_of_year_365(dates)
    rets = rng.normal(0.0, 0.005, len(dates)) + np.where((doy >= 100) & (doy <= 110), 0.01, 0.0)
    px = pd.DataFrame({"value": 100 * np.cumprod(1 + rets)}, index=dates)
    db.upsert_series("TY", "adjusted", px.join(close_ts_frame(inst, dates, config)))

    _PROFILE_CACHE.clear()
    f = get_signal("seaso").forecasts(DataView(db, config), config, "TY")
    y25 = f[f.index.year == 2025].dropna()
    d = day_of_year_365(y25.index)
    inside = y25[(d >= 100) & (d <= 110)]
    outside = y25[(d < 85) | (d > 125)]
    assert inside.mean() > 1.0
    assert inside.mean() > outside.mean() + 1.0


def test_seaso_is_point_in_time(ingested_db, config):
    """Within year Y the signal only uses years < Y: truncating the current
    year's data must not change any forecast."""
    _PROFILE_CACHE.clear()
    full = get_signal("seaso").compute(DataView(ingested_db, config), config, "TY")
    db2 = _truncated_copy(ingested_db, ["TY"], n_drop=60)
    _PROFILE_CACHE.clear()
    trunc = get_signal("seaso").compute(DataView(db2, config), config, "TY")
    pd.testing.assert_series_equal(full.reindex(trunc.index), trunc)


def test_scale_forecast_uses_only_past(config):
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
        cal = DataView(ingested_db, config).calendar("TY")
        month_ends = cal[cal["is_month_end"]].index.intersection(f.index)
        assert (f.loc[month_ends] > 0).all()
    finally:
        from prt.signals import base

        base._REGISTRY.pop("test_seaso", None)
