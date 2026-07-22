"""DB block: round-trips of every table family."""

import pandas as pd

from prt.instruments.master import close_ts_frame


def _mini_frame(config, inst_id="TY", n=5, start="2026-07-06"):
    inst = config.instrument(inst_id)
    dates = pd.bdate_range(start, periods=n)
    df = pd.DataFrame({"value": [100.0 + i for i in range(n)]}, index=dates)
    return df.join(close_ts_frame(inst, dates, config))


def test_series_roundtrip(db, config):
    df = _mini_frame(config)
    assert db.upsert_series("TY", "adjusted", df) == 5
    out = db.series("TY", "adjusted")
    assert list(out["value"]) == list(df["value"])
    assert out["knowledge_ts"].dt.tz is not None  # tz-aware UTC
    assert db.last_series_date("TY", "adjusted") == df.index[-1]


def test_replace_series(db, config):
    db.upsert_series("TY", "adjusted", _mini_frame(config))
    df2 = _mini_frame(config, n=3)
    df2["value"] += 50
    db.replace_series("TY", "adjusted", df2)
    out = db.series("TY", "adjusted")
    assert len(out) == 3 and out["value"].iloc[0] == 150.0


def test_active_contract(db):
    assert db.current_active_contract("TY") is None
    db.set_active_contract("TY", "2026-07-21", "TYU6 Comdty", {"FUT_CONT_SIZE": 1000})
    ticker, info = db.current_active_contract("TY")
    assert ticker == "TYU6 Comdty" and info["FUT_CONT_SIZE"] == 1000


def test_fills_aggregate_to_positions(db):
    db.add_fill("TY", contracts=-3, notional_usd=-350_000, price=112.5)
    db.add_fill("TY", contracts=1, notional_usd=115_000, price=115.0)
    db.add_fill("EURUSD", contracts=None, notional_usd=2_000_000)
    pos = db.positions()
    assert pos.loc["TY", "contracts"] == -2
    assert pos.loc["TY", "notional_usd"] == -235_000
    assert pos.loc["EURUSD", "notional_usd"] == 2_000_000


def test_live_quotes_upsert_keeps_latest(db):
    db.upsert_live_quote("TYU6 Comdty", "TY", 112.0, event_ts=pd.Timestamp.now(tz="UTC"))
    db.upsert_live_quote("TYU6 Comdty", "TY", 112.5, event_ts=pd.Timestamp.now(tz="UTC"))
    q = db.live_quotes()
    assert len(q) == 1 and q.loc["TYU6 Comdty", "last_price"] == 112.5


def test_forecasts_and_targets(db):
    s = pd.Series([1.0, -2.0], index=pd.bdate_range("2026-07-20", periods=2))
    db.write_forecasts("momentum", "TY", s)
    out = db.forecasts(signal="momentum", instrument_id="TY")
    assert len(out) == 2
    db.write_targets(
        pd.DataFrame(
            [{"instrument_id": "TY", "exec_date": "2026-07-21", "combined_forecast": 1.5,
              "target_notional": 1e6, "target_contracts": 9.0, "price": 112.0}]
        )
    )
    assert db.latest_targets()["target_contracts"].iloc[0] == 9.0


def test_eco_releases_as_released_with_revisions(db):
    db.upsert_eco_release("NFP", "2026-06", pd.Timestamp("2026-07-03 12:30", tz="UTC"), 150.0, 140.0, 130.0)
    db.upsert_eco_release(
        "NFP", "2026-06", pd.Timestamp("2026-08-07 12:30", tz="UTC"), 145.0, None, None,
        revision_of="2026-07-03T12:30:00+00:00",
    )
    out = db.eco_releases("NFP")
    assert len(out) == 2  # revision is a NEW row, first release untouched
    assert out.iloc[0]["actual"] == 150.0
