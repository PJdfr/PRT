"""Time semantics of the instrument master.

Everything in the system carries full UTC timestamps, never bare dates.
This module is the ONLY place where a (instrument, date) daily bar is
turned into actual instants:

    close_ts      when the settlement print happens (event time)
    knowledge_ts  when OUR system can know it: max(close + publish lag,
                  the daily ingestion batch time of that date)
    decision_ts   when we must decide to execute at that close
                  (a margin BEFORE the close)

The DataView compares knowledge_ts vs decision_ts and nothing else —
the one-day execution lag is a consequence, not a hard-coded shift.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from prt.config import Config, Instrument

UTC = ZoneInfo("UTC")


def _parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def close_ts(inst: Instrument, date: pd.Timestamp | str) -> pd.Timestamp:
    """UTC instant of the settlement print of `inst` on `date`.

    Uses the IANA timezone so DST transitions are handled per-date.
    """
    d = pd.Timestamp(date)
    local = datetime.combine(d.date(), _parse_hhmm(inst.settle_time), tzinfo=ZoneInfo(inst.tz))
    return pd.Timestamp(local.astimezone(UTC))


def knowledge_ts(inst: Instrument, date: pd.Timestamp | str, config: Config) -> pd.Timestamp:
    """UTC instant at which OUR system knows the close of `inst` on `date`.

    max(settle + publish lag, daily ingestion batch of that date): data the
    exchange published is still unknown to us until our batch has run.
    """
    event = close_ts(inst, date) + timedelta(minutes=config.fund.publish_lag_minutes)
    d = pd.Timestamp(date)
    batch = pd.Timestamp(
        datetime.combine(d.date(), _parse_hhmm(config.data.daily_batch_utc), tzinfo=UTC)
    )
    return max(event, batch)


def decision_ts(inst: Instrument, date: pd.Timestamp | str, config: Config) -> pd.Timestamp:
    """UTC instant at which we must decide in order to execute at the close of `date`."""
    return close_ts(inst, date) - timedelta(minutes=config.fund.decision_margin_minutes)


def close_ts_frame(inst: Instrument, dates: pd.DatetimeIndex, config: Config) -> pd.DataFrame:
    """Vectorised event/knowledge timestamps for a set of dates."""
    rows = [(d, close_ts(inst, d), knowledge_ts(inst, d, config)) for d in dates]
    return pd.DataFrame(rows, columns=["date", "event_ts", "knowledge_ts"]).set_index("date")
