"""Live monitor: intraday PnL since last close, signal preview, orders.

Reads the DB only (live_quotes fed by the stream daemon, closes fed by the
daily batch, positions from the fills table).  The preview re-runs the
exact same signal + portfolio machinery on a DataView where today's live
price is injected as a hypothetical close — "if today closes here,
tomorrow's target is X".
"""

from __future__ import annotations

import pandas as pd

from prt.config import Config
from prt.db import Database
from prt.portfolio.construction import compute_targets
from prt.signals.base import compute_all_forecasts
from prt.signals.dataview import DataView


def _fx_now(db: Database, config: Config, currency: str) -> float:
    """Spot ccy->USD using the freshest source available (live quote, else last close)."""
    if currency == "USD":
        return 1.0
    conv = config.fx_conversion[currency]
    pair = config.instruments[conv["pair"]]
    rate = None
    quotes = db.live_quotes()
    if not quotes.empty and pair.spot_ticker in quotes.index:
        rate = quotes.loc[pair.spot_ticker, "last_price"]
    if rate is None or pd.isna(rate):
        spot = db.series(conv["pair"], "spot")["value"]
        rate = float(spot.iloc[-1]) if not spot.empty else None
    if rate is None:
        raise ValueError(f"no fx rate available for {currency}")
    return 1.0 / float(rate) if conv.get("invert") else float(rate)


def _live_price(db: Database, inst_id: str) -> float | None:
    quotes = db.live_quotes()
    if quotes.empty:
        return None
    mine = quotes[quotes["instrument_id"] == inst_id]
    if mine.empty or pd.isna(mine["last_price"].iloc[0]):
        return None
    return float(mine["last_price"].iloc[0])


def pnl_since_close(db: Database, config: Config) -> pd.DataFrame:
    """Per-instrument PnL (USD) between the last stored close and the live quote.

    Futures: contracts x point_value x (live - prev close) x fx.
    FX: pair notional x spot return since close.
    """
    positions = db.positions()
    rows = []
    for inst_id, pos in positions.iterrows() if not positions.empty else []:
        inst = config.instruments.get(inst_id)
        if inst is None:
            continue
        live = _live_price(db, inst_id)
        if inst.is_future:
            prev = db.series(inst_id, "adjusted")["value"]  # last value = actual front price
        else:
            prev = db.series(inst_id, "spot")["value"]
        prev_close = float(prev.iloc[-1]) if not prev.empty else None
        pnl = None
        if live is not None and prev_close:
            if inst.is_future:
                fx = _fx_now(db, config, inst.currency)
                pnl = pos["contracts"] * inst.point_value * (live - prev_close) * fx
            else:
                pnl = pos["notional_usd"] * (live / prev_close - 1.0)
        rows.append(
            {
                "instrument_id": inst_id,
                "contracts": pos["contracts"],
                "notional_usd": pos["notional_usd"],
                "prev_close": prev_close,
                "live": live,
                "pnl_usd": pnl,
            }
        )
    return pd.DataFrame(rows).set_index("instrument_id") if rows else pd.DataFrame()


def preview_targets(db: Database, config: Config) -> pd.DataFrame:
    """Targets for the NEXT close assuming today closes at current live levels.

    Injects live quotes as a hypothetical close (knowledge_ts = now) into
    the DataView and re-runs signals + portfolio — same code path as the
    real thing, one extra hypothetical data point.
    """
    now = pd.Timestamp.now(tz="UTC")
    overrides: dict = {}
    for inst_id, inst in config.instruments.items():
        live = _live_price(db, inst_id)
        if live is None:
            continue
        stored = db.series(inst_id, "adjusted")["value"]
        if stored.empty:
            continue
        pending_date = stored.index[-1] + pd.offsets.BDay(1)
        overrides[(inst_id, "adjusted")] = (pending_date, live, now)

    view = DataView(db, config, overrides=overrides)
    forecasts = compute_all_forecasts(db, config, view, persist=False)
    frames = compute_targets(db, config, view, forecasts, persist=False)
    rows = []
    for inst_id, frame in frames.items():
        last = frame.dropna(subset=["target_notional"]).tail(1)
        if last.empty:
            continue
        rows.append(
            {
                "instrument_id": inst_id,
                "exec_date": last.index[-1],
                "forecast": last["forecast"].iloc[-1],
                "target_notional": last["held_notional"].iloc[-1],
                "target_contracts": last["contracts"].iloc[-1],
                "hypothetical": inst_id in {k[0] for k in overrides},
            }
        )
    return pd.DataFrame(rows).set_index("instrument_id") if rows else pd.DataFrame()


def generate_orders(db: Database, config: Config, targets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Diff latest persisted targets vs the current book -> order list."""
    tgt = targets if targets is not None else db.latest_targets()
    if tgt.empty:
        return pd.DataFrame()
    if "instrument_id" in tgt.columns:
        tgt = tgt.set_index("instrument_id")
    positions = db.positions()
    rows = []
    for inst_id, row in tgt.iterrows():
        inst = config.instruments.get(inst_id)
        if inst is None:
            continue
        cur_contracts = float(positions.loc[inst_id, "contracts"]) if not positions.empty and inst_id in positions.index else 0.0
        cur_notional = float(positions.loc[inst_id, "notional_usd"]) if not positions.empty and inst_id in positions.index else 0.0
        if inst.is_future:
            tgt_c = row.get("target_contracts")
            delta = (0.0 if pd.isna(tgt_c) else float(tgt_c)) - cur_contracts
            unit = "contracts"
        else:
            delta = float(row.get("target_notional") or 0.0) - cur_notional
            unit = "usd_notional"
        if abs(delta) < 1e-9:
            continue
        rows.append(
            {
                "instrument_id": inst_id,
                "action": "BUY" if delta > 0 else "SELL",
                "quantity": abs(delta),
                "unit": unit,
                "current": cur_contracts if inst.is_future else cur_notional,
                "target": (row.get("target_contracts") if inst.is_future else row.get("target_notional")),
                "forecast": row.get("combined_forecast", row.get("forecast")),
            }
        )
    return pd.DataFrame(rows).set_index("instrument_id") if rows else pd.DataFrame()
