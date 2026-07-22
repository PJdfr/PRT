"""SQLite persistence layer — the single meeting point between blocks.

All timestamps are stored as ISO-8601 UTC strings; dates as YYYY-MM-DD
labels (join keys only — time information always lives in *_ts columns).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices_series (
    instrument_id TEXT NOT NULL,
    series_type   TEXT NOT NULL,      -- adjusted | g1_raw | g2_raw | spot
    date          TEXT NOT NULL,
    value         REAL NOT NULL,
    event_ts      TEXT NOT NULL,
    knowledge_ts  TEXT NOT NULL,
    ingested_at   TEXT NOT NULL,
    PRIMARY KEY (instrument_id, series_type, date)
);
CREATE TABLE IF NOT EXISTS active_contracts (
    instrument_id   TEXT NOT NULL,
    date            TEXT NOT NULL,
    contract_ticker TEXT NOT NULL,
    info_json       TEXT,
    PRIMARY KEY (instrument_id, date)
);
CREATE TABLE IF NOT EXISTS live_quotes (
    ticker        TEXT PRIMARY KEY,
    instrument_id TEXT,
    last_price    REAL,
    bid           REAL,
    ask           REAL,
    event_ts      TEXT,
    received_at   TEXT
);
CREATE TABLE IF NOT EXISTS refresh_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_id TEXT,
    series_type   TEXT,
    reason        TEXT,
    n_rows        INTEGER,
    refreshed_at  TEXT
);
CREATE TABLE IF NOT EXISTS bbg_query_log (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           TEXT,
    request_type TEXT,
    tickers      TEXT,
    fields       TEXT,
    n_points     INTEGER
);
CREATE TABLE IF NOT EXISTS forecasts (
    signal        TEXT NOT NULL,
    instrument_id TEXT NOT NULL,
    exec_date     TEXT NOT NULL,
    value         REAL NOT NULL,
    computed_at   TEXT NOT NULL,
    PRIMARY KEY (signal, instrument_id, exec_date)
);
CREATE TABLE IF NOT EXISTS targets (
    instrument_id     TEXT NOT NULL,
    exec_date         TEXT NOT NULL,
    combined_forecast REAL,
    target_notional   REAL,
    target_contracts  REAL,
    price             REAL,
    computed_at       TEXT,
    PRIMARY KEY (instrument_id, exec_date)
);
CREATE TABLE IF NOT EXISTS fills (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_id TEXT NOT NULL,
    ts            TEXT NOT NULL,
    contracts     REAL,               -- signed; fx: NULL
    notional_usd  REAL,               -- signed notional at fill
    price         REAL,
    note          TEXT
);
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id     TEXT PRIMARY KEY,
    created_at TEXT,
    meta_json  TEXT
);
CREATE TABLE IF NOT EXISTS backtest_pnl (
    run_id TEXT NOT NULL,
    level  TEXT NOT NULL,             -- fund | instrument | signal
    key    TEXT NOT NULL,
    date   TEXT NOT NULL,
    value  REAL NOT NULL,
    PRIMARY KEY (run_id, level, key, date)
);
CREATE TABLE IF NOT EXISTS eco_releases (
    indicator    TEXT NOT NULL,
    period       TEXT NOT NULL,       -- reference period label (e.g. 2026-06)
    release_ts   TEXT NOT NULL,       -- publication instant, UTC
    actual       REAL,
    survey       REAL,
    prior        REAL,
    revision_of  TEXT,                -- release_ts of the release this revises
    knowledge_ts TEXT NOT NULL,
    ingested_at  TEXT NOT NULL,
    PRIMARY KEY (indicator, period, release_ts)
);
"""


def utcnow() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc))


class Database:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---------------- prices -----------------------------------------
    def upsert_series(self, instrument_id: str, series_type: str, df: pd.DataFrame) -> int:
        """df: index = dates, columns value, event_ts, knowledge_ts."""
        now = utcnow().isoformat()
        rows = [
            (
                instrument_id,
                series_type,
                pd.Timestamp(d).strftime("%Y-%m-%d"),
                float(r["value"]),
                pd.Timestamp(r["event_ts"]).isoformat(),
                pd.Timestamp(r["knowledge_ts"]).isoformat(),
                now,
            )
            for d, r in df.iterrows()
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO prices_series VALUES (?,?,?,?,?,?,?)", rows
        )
        self.conn.commit()
        return len(rows)

    def replace_series(self, instrument_id: str, series_type: str, df: pd.DataFrame) -> int:
        self.conn.execute(
            "DELETE FROM prices_series WHERE instrument_id=? AND series_type=?",
            (instrument_id, series_type),
        )
        return self.upsert_series(instrument_id, series_type, df)

    def series(self, instrument_id: str, series_type: str = "adjusted") -> pd.DataFrame:
        """Returns DataFrame indexed by date with columns value, event_ts, knowledge_ts."""
        df = pd.read_sql_query(
            "SELECT date, value, event_ts, knowledge_ts FROM prices_series "
            "WHERE instrument_id=? AND series_type=? ORDER BY date",
            self.conn,
            params=(instrument_id, series_type),
        )
        if df.empty:
            return pd.DataFrame(columns=["value", "event_ts", "knowledge_ts"])
        df["date"] = pd.to_datetime(df["date"])
        df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True)
        df["knowledge_ts"] = pd.to_datetime(df["knowledge_ts"], utc=True)
        return df.set_index("date")

    def last_series_date(self, instrument_id: str, series_type: str) -> pd.Timestamp | None:
        row = self.conn.execute(
            "SELECT MAX(date) FROM prices_series WHERE instrument_id=? AND series_type=?",
            (instrument_id, series_type),
        ).fetchone()
        return pd.Timestamp(row[0]) if row and row[0] else None

    # ---------------- active contracts --------------------------------
    def current_active_contract(self, instrument_id: str) -> tuple[str, dict] | None:
        row = self.conn.execute(
            "SELECT contract_ticker, info_json FROM active_contracts "
            "WHERE instrument_id=? ORDER BY date DESC LIMIT 1",
            (instrument_id,),
        ).fetchone()
        if row is None:
            return None
        return row[0], json.loads(row[1] or "{}")

    def set_active_contract(
        self, instrument_id: str, date: str, contract_ticker: str, info: dict | None = None
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO active_contracts VALUES (?,?,?,?)",
            (instrument_id, str(date)[:10], contract_ticker, json.dumps(info or {})),
        )
        self.conn.commit()

    # ---------------- logs --------------------------------------------
    def log_refresh(self, instrument_id: str, series_type: str, reason: str, n_rows: int) -> None:
        self.conn.execute(
            "INSERT INTO refresh_log (instrument_id, series_type, reason, n_rows, refreshed_at) "
            "VALUES (?,?,?,?,?)",
            (instrument_id, series_type, reason, n_rows, utcnow().isoformat()),
        )
        self.conn.commit()

    def refreshes(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM refresh_log ORDER BY id", self.conn)

    def log_query(self, request_type: str, tickers: list[str], fields: list[str], n_points: int) -> None:
        self.conn.execute(
            "INSERT INTO bbg_query_log (ts, request_type, tickers, fields, n_points) VALUES (?,?,?,?,?)",
            (utcnow().isoformat(), request_type, ",".join(tickers), ",".join(fields), n_points),
        )
        self.conn.commit()

    def query_log(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM bbg_query_log ORDER BY id", self.conn)

    # ---------------- live quotes --------------------------------------
    def upsert_live_quote(
        self,
        ticker: str,
        instrument_id: str | None,
        last_price: float | None,
        bid: float | None = None,
        ask: float | None = None,
        event_ts: pd.Timestamp | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO live_quotes VALUES (?,?,?,?,?,?,?)",
            (
                ticker,
                instrument_id,
                last_price,
                bid,
                ask,
                pd.Timestamp(event_ts).isoformat() if event_ts is not None else None,
                utcnow().isoformat(),
            ),
        )
        self.conn.commit()

    def live_quotes(self) -> pd.DataFrame:
        df = pd.read_sql_query("SELECT * FROM live_quotes", self.conn)
        if not df.empty:
            df["event_ts"] = pd.to_datetime(df["event_ts"], utc=True)
            df["received_at"] = pd.to_datetime(df["received_at"], utc=True)
        return df.set_index("ticker")

    # ---------------- forecasts / targets ------------------------------
    def write_forecasts(self, signal: str, instrument_id: str, forecasts: pd.Series) -> None:
        now = utcnow().isoformat()
        rows = [
            (signal, instrument_id, pd.Timestamp(d).strftime("%Y-%m-%d"), float(v), now)
            for d, v in forecasts.dropna().items()
        ]
        self.conn.executemany("INSERT OR REPLACE INTO forecasts VALUES (?,?,?,?,?)", rows)
        self.conn.commit()

    def forecasts(self, signal: str | None = None, instrument_id: str | None = None) -> pd.DataFrame:
        q = "SELECT signal, instrument_id, exec_date, value FROM forecasts WHERE 1=1"
        params: list = []
        if signal:
            q += " AND signal=?"
            params.append(signal)
        if instrument_id:
            q += " AND instrument_id=?"
            params.append(instrument_id)
        df = pd.read_sql_query(q + " ORDER BY exec_date", self.conn, params=params)
        if not df.empty:
            df["exec_date"] = pd.to_datetime(df["exec_date"])
        return df

    def write_targets(self, df: pd.DataFrame) -> None:
        """df columns: instrument_id, exec_date, combined_forecast, target_notional,
        target_contracts, price."""
        now = utcnow().isoformat()
        rows = [
            (
                r.instrument_id,
                pd.Timestamp(r.exec_date).strftime("%Y-%m-%d"),
                float(r.combined_forecast),
                float(r.target_notional),
                None if pd.isna(r.target_contracts) else float(r.target_contracts),
                None if pd.isna(r.price) else float(r.price),
                now,
            )
            for r in df.itertuples()
        ]
        self.conn.executemany("INSERT OR REPLACE INTO targets VALUES (?,?,?,?,?,?,?)", rows)
        self.conn.commit()

    def targets(self, exec_date: str | None = None) -> pd.DataFrame:
        if exec_date:
            df = pd.read_sql_query(
                "SELECT * FROM targets WHERE exec_date=?", self.conn, params=(str(exec_date)[:10],)
            )
        else:
            df = pd.read_sql_query("SELECT * FROM targets ORDER BY exec_date", self.conn)
        if not df.empty:
            df["exec_date"] = pd.to_datetime(df["exec_date"])
        return df

    def latest_targets(self) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT t.* FROM targets t JOIN (SELECT instrument_id, MAX(exec_date) d "
            "FROM targets GROUP BY instrument_id) m "
            "ON t.instrument_id=m.instrument_id AND t.exec_date=m.d",
            self.conn,
        )
        if not df.empty:
            df["exec_date"] = pd.to_datetime(df["exec_date"])
        return df

    # ---------------- fills / positions --------------------------------
    def add_fill(
        self,
        instrument_id: str,
        contracts: float | None,
        notional_usd: float,
        price: float | None = None,
        ts: pd.Timestamp | None = None,
        note: str = "",
    ) -> None:
        self.conn.execute(
            "INSERT INTO fills (instrument_id, ts, contracts, notional_usd, price, note) "
            "VALUES (?,?,?,?,?,?)",
            (
                instrument_id,
                pd.Timestamp(ts if ts is not None else utcnow()).isoformat(),
                contracts,
                notional_usd,
                price,
                note,
            ),
        )
        self.conn.commit()

    def fills(self) -> pd.DataFrame:
        df = pd.read_sql_query("SELECT * FROM fills ORDER BY ts", self.conn)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def positions(self) -> pd.DataFrame:
        """Current book aggregated from fills: signed contracts & notional per instrument."""
        df = pd.read_sql_query(
            "SELECT instrument_id, SUM(COALESCE(contracts,0)) contracts, "
            "SUM(notional_usd) notional_usd FROM fills GROUP BY instrument_id",
            self.conn,
        )
        return df.set_index("instrument_id") if not df.empty else df

    # ---------------- backtests ----------------------------------------
    def save_backtest(self, run_id: str, meta: dict, pnl_frames: dict[str, pd.DataFrame]) -> None:
        """pnl_frames: level -> DataFrame(index=dates, columns=keys, values=daily pnl usd)."""
        self.conn.execute(
            "INSERT OR REPLACE INTO backtest_runs VALUES (?,?,?)",
            (run_id, utcnow().isoformat(), json.dumps(meta)),
        )
        rows = []
        for level, frame in pnl_frames.items():
            for key in frame.columns:
                for d, v in frame[key].dropna().items():
                    rows.append((run_id, level, str(key), pd.Timestamp(d).strftime("%Y-%m-%d"), float(v)))
        self.conn.executemany("INSERT OR REPLACE INTO backtest_pnl VALUES (?,?,?,?,?)", rows)
        self.conn.commit()

    def backtest_pnl(self, run_id: str, level: str) -> pd.DataFrame:
        df = pd.read_sql_query(
            "SELECT key, date, value FROM backtest_pnl WHERE run_id=? AND level=?",
            self.conn,
            params=(run_id, level),
        )
        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"])
        return df.pivot(index="date", columns="key", values="value")

    def backtest_runs(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM backtest_runs ORDER BY created_at", self.conn)

    # ---------------- eco releases -------------------------------------
    def upsert_eco_release(
        self,
        indicator: str,
        period: str,
        release_ts: pd.Timestamp,
        actual: float | None,
        survey: float | None,
        prior: float | None,
        revision_of: str | None = None,
        knowledge_ts: pd.Timestamp | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO eco_releases VALUES (?,?,?,?,?,?,?,?,?)",
            (
                indicator,
                period,
                pd.Timestamp(release_ts).isoformat(),
                actual,
                survey,
                prior,
                revision_of,
                pd.Timestamp(knowledge_ts if knowledge_ts is not None else release_ts).isoformat(),
                utcnow().isoformat(),
            ),
        )
        self.conn.commit()

    def eco_releases(self, indicator: str | None = None) -> pd.DataFrame:
        q = "SELECT * FROM eco_releases"
        params: tuple = ()
        if indicator:
            q += " WHERE indicator=?"
            params = (indicator,)
        df = pd.read_sql_query(q + " ORDER BY release_ts", self.conn, params=params)
        if not df.empty:
            df["release_ts"] = pd.to_datetime(df["release_ts"], utc=True)
            df["knowledge_ts"] = pd.to_datetime(df["knowledge_ts"], utc=True)
        return df
