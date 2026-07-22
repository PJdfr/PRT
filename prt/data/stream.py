"""Real-time stream daemon: provider.subscribe() -> live_quotes table.

Keeps only the latest value per ticker (upsert).  Downstream consumers
(live PnL, preview, dashboard) read live_quotes and never know whether the
value came from a Bloomberg subscription, a bdp fallback or a mock stream.
"""

from __future__ import annotations

import logging
import time

from prt.config import Config
from prt.data.ingest import live_tickers
from prt.data.provider import DataProvider
from prt.db import Database

log = logging.getLogger(__name__)


class StreamDaemon:
    def __init__(
        self,
        db: Database,
        provider: DataProvider,
        tickers: dict[str, str],
        max_ticks: int | None = None,
        max_retries: int = 5,
        backoff_seconds: float = 2.0,
    ):
        self.db = db
        self.provider = provider
        self.tickers = tickers  # ticker -> instrument_id
        self.max_ticks = max_ticks
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds

    @classmethod
    def for_universe(cls, db: Database, provider: DataProvider, config: Config, **kw) -> "StreamDaemon":
        return cls(db, provider, live_tickers(db, config), **kw)

    def run(self) -> int:
        """Consume the stream; returns number of ticks processed."""
        n, retries = 0, 0
        while True:
            try:
                for tick in self.provider.subscribe(list(self.tickers)):
                    self.db.upsert_live_quote(
                        ticker=tick.ticker,
                        instrument_id=self.tickers.get(tick.ticker),
                        last_price=tick.last_price,
                        bid=tick.bid,
                        ask=tick.ask,
                        event_ts=tick.event_ts,
                    )
                    n += 1
                    retries = 0
                    if self.max_ticks is not None and n >= self.max_ticks:
                        return n
                return n  # stream ended cleanly
            except Exception:
                retries += 1
                if retries > self.max_retries:
                    log.exception("stream failed after %d retries", self.max_retries)
                    raise
                wait = self.backoff_seconds * 2 ** (retries - 1)
                log.warning("stream error, reconnecting in %.1fs (retry %d)", wait, retries)
                time.sleep(wait)
