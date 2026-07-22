"""Ingestion block: initial load, append, roll-triggered full refresh,
checksum-triggered full refresh, query logging."""

import pandas as pd

from prt.data.ingest import daily_update, ingest_instrument, live_tickers

TODAY = pd.Timestamp("2026-07-21")


def test_initial_load(ingested_db, config):
    for inst in config.instruments.values():
        s = ingested_db.series(inst.id, "adjusted")
        assert len(s) > 500
        if inst.is_future:
            assert not ingested_db.series(inst.id, "g2_raw").empty
            assert ingested_db.current_active_contract(inst.id) is not None
        else:
            assert not ingested_db.series(inst.id, "spot").empty
    refreshes = ingested_db.refreshes()
    assert (refreshes["reason"] == "initial_load").all()
    assert not ingested_db.query_log().empty


def test_rerun_same_day_appends_nothing(ingested_db, provider, config):
    summaries = daily_update(ingested_db, provider, config, as_of=TODAY)
    for s in summaries:
        assert not s["refreshed"], s
        assert all(v == 0 for v in s["appended"].values()), s


def test_append_next_days(ingested_db, provider, config):
    provider.today = TODAY + pd.Timedelta(days=2)  # advance two days, same quarter
    s = ingest_instrument(ingested_db, provider, config, config.instrument("TY"), provider.today)
    assert not s["rolled"] and not s["refreshed"]
    assert s["appended"]["adjusted"] >= 1


def test_roll_triggers_full_refresh(ingested_db, provider, config):
    before = ingested_db.series("TY", "adjusted")["value"]
    provider.advance_roll("TY1 Comdty")
    s = ingest_instrument(ingested_db, provider, config, config.instrument("TY"), TODAY)
    assert s["rolled"]
    assert any(r[0] == "adjusted" and r[1] == "roll_detected" for r in s["refreshed"])
    after = ingested_db.series("TY", "adjusted")["value"]
    # backward-ratio rewrite: whole history changed, same dates
    assert len(after) == len(before)
    assert (after / before).round(6).nunique() == 1  # uniform ratio
    assert after.iloc[0] != before.iloc[0]
    # unadjusted generics are append-only, untouched by the roll
    assert not any(r[0] == "g1_raw" for r in s["refreshed"])


def test_checksum_mismatch_triggers_full_refresh(ingested_db, provider, config):
    # simulate a silent re-adjustment: tamper one stored point
    ingested_db.conn.execute(
        "UPDATE prices_series SET value = value * 1.01 WHERE instrument_id='TY' "
        "AND series_type='adjusted' AND date=(SELECT MAX(date) FROM prices_series "
        "WHERE instrument_id='TY' AND series_type='adjusted')"
    )
    ingested_db.conn.commit()
    s = ingest_instrument(ingested_db, provider, config, config.instrument("TY"), TODAY)
    assert any(r[1] == "checksum_mismatch" for r in s["refreshed"])


def test_live_tickers_map(ingested_db, config):
    m = live_tickers(ingested_db, config)
    assert m["EURUSD Curncy"] == "EURUSD"
    assert any(v == "TY" for v in m.values())  # active contract ticker for futures
