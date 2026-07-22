"""Shared fixtures: a small 5-instrument universe on synthetic data.

Everything runs against MockProvider + in-memory SQLite — no terminal,
no network, deterministic.
"""

from __future__ import annotations

import pytest

from prt.config import build_config
from prt.data.ingest import daily_update
from prt.data.mock import MockProvider
from prt.db import Database

TODAY = "2026-07-21"

RAW_CONFIG = {
    "fund": {
        "capital_usd": 100_000_000,
        "target_vol": 0.10,
        "forecast_cap": 5.0,
        "forecast_avg": 2.5,
        "idm": 1.5,
        "vol_ewma_span": 32,
        "buffer_fraction": 0.10,
        "publish_lag_minutes": 10,
        "decision_margin_minutes": 10,
    },
    "data": {"history_start": "2022-01-03", "daily_batch_utc": "22:30", "checksum_points": 20},
    "signals": {"momentum": {"weight": 0.5}, "carry": {"weight": 0.5}},
    "exchanges": {
        "us_rates": {"tz": "America/Chicago", "settle_time": "14:00"},
        "eurex_rates": {"tz": "Europe/Berlin", "settle_time": "17:15"},
        "jpx": {"tz": "Asia/Tokyo", "settle_time": "15:15"},
        "fx_ny": {"tz": "America/New_York", "settle_time": "17:00"},
    },
    "fx_conversion": {
        "EUR": {"pair": "EURUSD", "invert": False},
        "JPY": {"pair": "USDJPY", "invert": True},
    },
    "instruments": [
        {"id": "TY", "name": "US 10Y", "class": "bonds", "exchange": "us_rates",
         "adj": "TY1 R:03_0_R Comdty", "g1": "TY1 Comdty", "g2": "TY2 Comdty",
         "ccy": "USD", "pv": 1000, "cost_bps": 0.4},
        {"id": "RX", "name": "Bund", "class": "bonds", "exchange": "eurex_rates",
         "adj": "RX1 R:03_0_R Comdty", "g1": "RX1 Comdty", "g2": "RX2 Comdty",
         "ccy": "EUR", "pv": 1000, "cost_bps": 0.4},
        {"id": "NK", "name": "Nikkei", "class": "equity", "exchange": "jpx",
         "adj": "NK1 R:03_0_R Index", "g1": "NK1 Index", "g2": "NK2 Index",
         "ccy": "JPY", "pv": 1000, "cost_bps": 0.6},
        {"id": "EURUSD", "name": "EUR/USD", "class": "fx", "kind": "fx", "exchange": "fx_ny",
         "adj": "EURUSDCR Curncy", "spot": "EURUSD Curncy", "ccy": "USD", "cost_bps": 0.3},
        {"id": "USDJPY", "name": "USD/JPY", "class": "fx", "kind": "fx", "exchange": "fx_ny",
         "adj": "USDJPYCR Curncy", "spot": "USDJPY Curncy", "ccy": "USD", "cost_bps": 0.3},
    ],
}


@pytest.fixture
def config():
    return build_config(RAW_CONFIG)


@pytest.fixture
def provider():
    return MockProvider(today=TODAY)


@pytest.fixture
def db():
    d = Database(":memory:")
    yield d
    d.close()


@pytest.fixture
def ingested_db(db, provider, config):
    daily_update(db, provider, config, as_of=TODAY)
    return db
