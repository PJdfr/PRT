"""DataProvider interface — the only boundary to the outside world.

Two implementations:
    MockProvider       deterministic synthetic data (CI, dev, tests)
    BloombergProvider  xbbg/blpapi (only on the terminal machine, lazy import)

Nothing outside prt.data may import xbbg.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass(frozen=True)
class Tick:
    ticker: str
    last_price: float
    bid: float | None
    ask: float | None
    event_ts: pd.Timestamp


class DataProvider(ABC):
    @abstractmethod
    def history(
        self, ticker: str, start: str, end: str, field: str = "PX_LAST"
    ) -> pd.Series:
        """Daily history, indexed by date."""

    @abstractmethod
    def snapshot(self, tickers: list[str], fields: list[str]) -> pd.DataFrame:
        """Current reference snapshot (bdp-like), index=tickers, columns=fields."""

    @abstractmethod
    def current_generic_ticker(self, generic_ticker: str) -> str:
        """Actual front contract ticker behind a generic (FUT_CUR_GEN_TICKER)."""

    @abstractmethod
    def contract_info(self, contract_ticker: str) -> dict:
        """Reference data of an actual contract (point value, ccy, expiry, ticks)."""

    @abstractmethod
    def subscribe(self, tickers: list[str]) -> Iterator[Tick]:
        """Real-time market data stream."""
