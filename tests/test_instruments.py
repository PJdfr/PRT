"""Time semantics: settlement instants, DST, knowledge floors."""

import pandas as pd

from prt.instruments.master import close_ts, decision_ts, knowledge_ts


def test_close_ts_handles_dst(config):
    ty = config.instrument("TY")
    # Chicago is UTC-6 in winter, UTC-5 in summer: 14:00 local settle
    assert close_ts(ty, "2026-01-15").hour == 20
    assert close_ts(ty, "2026-07-15").hour == 19


def test_close_ts_timezones_differ_per_instrument(config):
    d = "2026-07-15"
    nk = close_ts(config.instrument("NK"), d)   # 15:15 Tokyo = 06:15 UTC
    ty = close_ts(config.instrument("TY"), d)   # 14:00 Chicago = 19:00 UTC
    assert nk.hour == 6 and nk.minute == 15
    assert nk < ty


def test_knowledge_floor_is_daily_batch(config):
    # Tokyo close is published early in the UTC day, but OUR system only has
    # it after the daily batch (22:30 UTC)
    nk = config.instrument("NK")
    k = knowledge_ts(nk, "2026-07-15", config)
    assert (k.hour, k.minute) == (22, 30)
    # a late close (FX NY 17:00 EDT = 21:00 UTC + lag) stays before the batch,
    # so batch is the binding constraint there too
    fx = config.instrument("EURUSD")
    assert knowledge_ts(fx, "2026-07-15", config) >= close_ts(fx, "2026-07-15")


def test_decision_is_before_close(config):
    ty = config.instrument("TY")
    d = "2026-07-15"
    assert decision_ts(ty, d, config) < close_ts(ty, d)


def test_prev_day_knowledge_before_today_decision_for_all_pairs(config):
    """The 1-day-lag execution convention must be provably safe for every
    ordered pair of instruments in the universe."""
    d_prev, d = pd.Timestamp("2026-07-14"), pd.Timestamp("2026-07-15")
    for src in config.instruments.values():
        for tgt in config.instruments.values():
            assert knowledge_ts(src, d_prev, config) < decision_ts(tgt, d, config), (
                f"{src.id} close({d_prev.date()}) not known before {tgt.id} decision({d.date()})"
            )
