"""Daily ingestion: provider -> DB, with roll detection and full refresh.

Policy for the roll-adjusted (backward ratio) generics:
  * every run, compare the actual front contract behind the generic
    (FUT_CUR_GEN_TICKER) with the one stored in the DB;
  * additionally compare the last `checksum_points` stored values with a
    fresh pull (catches silent re-adjustments);
  * if either differs -> FULL refresh of the adjusted series (its whole
    history was rewritten by the ratio adjustment), logged in refresh_log;
  * otherwise -> plain append of missing dates.

Unadjusted generics (g1/g2) and spot series are append-only: their history
is never rewritten.

Timestamps: every stored row carries event_ts (settlement instant, UTC) and
knowledge_ts = max(event + publish lag, daily batch time) — see
prt.instruments.master.
"""

from __future__ import annotations

import pandas as pd

from prt.config import Config, Instrument
from prt.data.provider import DataProvider
from prt.db import Database
from prt.instruments.master import close_ts_frame


def _series_map(inst: Instrument) -> dict[str, str]:
    m = {"adjusted": inst.adj_ticker}
    if inst.g1_ticker:
        m["g1_raw"] = inst.g1_ticker
    if inst.g2_ticker:
        m["g2_raw"] = inst.g2_ticker
    if inst.spot_ticker:
        m["spot"] = inst.spot_ticker
    return m


def _with_timestamps(inst: Instrument, series: pd.Series, config: Config) -> pd.DataFrame:
    df = series.to_frame("value")
    ts = close_ts_frame(inst, df.index, config)
    return df.join(ts)


def _checksum_ok(db: Database, provider: DataProvider, inst: Instrument, config: Config) -> bool:
    """Compare last N stored adjusted values against a fresh pull."""
    stored = db.series(inst.id, "adjusted")
    if stored.empty:
        return False
    tail = stored["value"].tail(config.data.checksum_points)
    fresh = provider.history(
        inst.adj_ticker,
        tail.index[0].strftime("%Y-%m-%d"),
        tail.index[-1].strftime("%Y-%m-%d"),
    )
    common = tail.index.intersection(fresh.index)
    if len(common) == 0:
        return False
    return bool((tail.loc[common] - fresh.loc[common]).abs().max() < 1e-9)


def detect_roll(db: Database, provider: DataProvider, inst: Instrument, as_of: pd.Timestamp) -> bool:
    """Update active contract; True if the front contract changed."""
    if not inst.is_future:
        return False
    generic = inst.g1_ticker or inst.adj_ticker
    current = provider.current_generic_ticker(generic)
    stored = db.current_active_contract(inst.id)
    if stored is not None and stored[0] == current:
        return False
    info = provider.contract_info(current)
    db.set_active_contract(inst.id, as_of.strftime("%Y-%m-%d"), current, info)
    return True


def ingest_instrument(
    db: Database,
    provider: DataProvider,
    config: Config,
    inst: Instrument,
    as_of: pd.Timestamp,
) -> dict:
    """Ingest all series of one instrument. Returns a summary dict."""
    summary = {"instrument": inst.id, "rolled": False, "refreshed": [], "appended": {}}
    end = as_of.strftime("%Y-%m-%d")
    start_hist = config.data.history_start

    rolled = detect_roll(db, provider, inst, as_of)
    summary["rolled"] = rolled

    for series_type, ticker in _series_map(inst).items():
        last = db.last_series_date(inst.id, series_type)
        full_refresh = last is None
        reason = "initial_load"
        if series_type == "adjusted" and not full_refresh:
            if rolled:
                full_refresh, reason = True, "roll_detected"
            elif not _checksum_ok(db, provider, inst, config):
                full_refresh, reason = True, "checksum_mismatch"

        if full_refresh:
            series = provider.history(ticker, start_hist, end)
            n = db.replace_series(inst.id, series_type, _with_timestamps(inst, series, config))
            db.log_refresh(inst.id, series_type, reason, n)
            db.log_query("bdh", [ticker], ["PX_LAST"], n)
            summary["refreshed"].append((series_type, reason, n))
        else:
            start = (last + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if start > end:
                summary["appended"][series_type] = 0
                continue
            series = provider.history(ticker, start, end)
            series = series[series.index > last]
            n = db.upsert_series(inst.id, series_type, _with_timestamps(inst, series, config))
            db.log_query("bdh", [ticker], ["PX_LAST"], n)
            summary["appended"][series_type] = n
    return summary


def daily_update(
    db: Database, provider: DataProvider, config: Config, as_of: str | pd.Timestamp | None = None
) -> list[dict]:
    """Run the daily batch over the whole universe."""
    as_of_ts = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC").tz_localize(None)
    return [
        ingest_instrument(db, provider, config, inst, as_of_ts)
        for inst in config.instruments.values()
    ]


def live_tickers(db: Database, config: Config) -> dict[str, str]:
    """ticker -> instrument_id map for the real-time stream:
    actual front contract for futures (generic as fallback), spot for fx."""
    out: dict[str, str] = {}
    for inst in config.instruments.values():
        if inst.is_future:
            stored = db.current_active_contract(inst.id)
            out[stored[0] if stored else (inst.g1_ticker or inst.adj_ticker)] = inst.id
        else:
            out[inst.spot_ticker or inst.adj_ticker] = inst.id
    return out
