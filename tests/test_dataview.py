"""DataView block: the anti-lookahead guarantee, the emergent 1-day lag,
zero-lag calendar features, live-preview overrides."""

import pandas as pd

from prt.signals.dataview import DataView


def test_no_lookahead_generic(ingested_db, config):
    """For every (instrument, series, target) actually usable, the knowledge
    timestamp of the data used must be strictly before the decision instant."""
    view = DataView(ingested_db, config)
    for inst in config.instruments.values():
        types = ["adjusted", "g1_raw", "g2_raw"] if inst.is_future else ["adjusted", "spot"]
        for st in types:
            audit = view.audit(inst.id, st).dropna(subset=["used_knowledge_ts"])
            assert (audit["used_knowledge_ts"] < audit["decision_ts"]).all(), (inst.id, st)


def test_one_day_lag_emerges_from_timestamps(ingested_db, config):
    """Same-instrument prices: value at exec date t == stored close of t-1.
    No shift() anywhere — pure timestamp comparison."""
    view = DataView(ingested_db, config)
    stored = ingested_db.series("TY", "adjusted")["value"]
    seen = view.prices("TY", "adjusted")
    common = stored.index.intersection(seen.index)[1:]
    for t in common[-50:]:
        prev_dates = stored.index[stored.index < t]
        assert seen.loc[t] == stored.loc[prev_dates[-1]]


def test_cross_instrument_lag(ingested_db, config):
    """Tokyo close of t is published before the US decision of t, but the
    22:30 UTC daily batch makes it known only after — so NK data seen by TY
    at exec t is NK's close of t-1 (honest operational knowledge)."""
    view = DataView(ingested_db, config)
    nk_stored = ingested_db.series("NK", "adjusted")["value"]
    nk_seen_by_ty = view.prices("NK", "adjusted", for_inst="TY")
    t = nk_seen_by_ty.dropna().index[-5]
    prev = nk_stored.index[nk_stored.index < t]
    assert nk_seen_by_ty.loc[t] == nk_stored.loc[prev[-1]]


def test_calendar_has_no_lag(ingested_db, config):
    """Calendar features (known in advance) are NOT shifted: month-end flag
    sits exactly on the last business day of the month."""
    view = DataView(ingested_db, config)
    cal = view.calendar("TY")
    june_end = cal[(cal.index.month == 6) & (cal.index.year == 2026) & cal["is_month_end"]]
    assert list(june_end.index) == [pd.Timestamp("2026-06-30")]


def test_exec_grid_includes_next_day(ingested_db, config):
    """After the last stored close we can already decide for the next
    business day — the grid exposes it."""
    view = DataView(ingested_db, config)
    stored = ingested_db.series("TY", "adjusted")
    dates = view.exec_dates("TY")
    assert dates[-1] == stored.index[-1] + pd.offsets.BDay(1)
    # and its price is the last stored close (known after the batch)
    assert view.prices("TY", "adjusted").loc[dates[-1]] == stored["value"].iloc[-1]


def test_override_feeds_preview(ingested_db, config):
    """Injecting a hypothetical close (live price, knowledge=now) changes only
    the extra exec date — history is untouched."""
    stored = ingested_db.series("TY", "adjusted")["value"]
    pending = stored.index[-1] + pd.offsets.BDay(1)
    # deterministic "now": after today's decision instant, before tomorrow's
    now = pending.tz_localize("UTC") + pd.Timedelta(hours=20)
    view = DataView(
        ingested_db, config, overrides={("TY", "adjusted"): (pending, 999.0, now)}
    )
    seen = view.prices("TY", "adjusted")
    assert seen.index[-1] == pending + pd.offsets.BDay(1)  # grid extends one more day
    assert seen.iloc[-1] == 999.0                          # hypothetical close visible tomorrow
    assert seen.loc[pending] == stored.iloc[-1]            # today still sees yesterday's close
    plain = DataView(ingested_db, config).prices("TY", "adjusted")
    assert (plain.dropna() == seen.reindex(plain.index).dropna()).all()
