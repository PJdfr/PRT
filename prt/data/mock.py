"""Deterministic synthetic data provider — CI / dev / tests, no terminal needed.

Every ticker gets a reproducible random-walk path (seeded by the ticker
name).  Futures rolls are simulated: the "active contract" behind a generic
changes quarterly (or on demand via `advance_roll`), and each roll multiplies
the *adjusted* series history by a ratio factor — reproducing Bloomberg's
backward-ratio rewrite so the ingestion refresh logic is exercised for real.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterator

import numpy as np
import pandas as pd

from prt.data.provider import DataProvider, Tick

_ROLL_SPEC_RE = re.compile(r"\s+R:\S+")
_GENERIC_RE = re.compile(r"^(.*?)(\d)( \w+)$")

EPOCH = "2010-01-04"
HORIZON = "2030-12-31"
ROLL_RATIO = 0.98

# realistic-ish starting levels so contract counts / notionals make sense
PRICE0 = {
    # bonds
    "TU1": 103, "FV1": 108, "TY1": 112, "UXY1": 115, "US1": 118, "WN1": 122,
    "DU1": 106, "OE1": 118, "RX1": 132, "UB1": 178, "OAT1": 128, "IK1": 118,
    "G 1": 98, "JB1": 145, "CN1": 122, "YM1": 96, "XM1": 95,
    # equity indices
    "ES1": 6300, "NQ1": 23000, "RTY1": 2250, "VG1": 5400, "GX1": 24000,
    "CF1": 8000, "Z 1": 8200, "SM1": 12500, "ST1": 35000, "EO1": 940,
    "NK1": 40000, "TP1": 2850, "XP1": 8300, "PT1": 1500,
    # commodities
    "CL1": 70, "CO1": 74, "NG1": 3.0, "HO1": 2.3, "XB1": 2.2, "QS1": 680,
    "GC1": 2400, "SI1": 29, "HG1": 4.2, "PL1": 950,
    "C 1": 450, "S 1": 1050, "W 1": 550, "SB1": 19, "KC1": 320, "CT1": 68,
    "CC1": 8000, "LC1": 190,
    # fx spots
    "EURUSD": 1.09, "GBPUSD": 1.27, "AUDUSD": 0.66, "NZDUSD": 0.60,
    "USDJPY": 155, "USDCHF": 0.88, "USDCAD": 1.37, "USDSEK": 10.5, "USDNOK": 10.6,
    # crypto
    "BTC1": 100_000, "ETH1": 5_000,
}


def _root(ticker: str) -> str:
    """Strip the roll spec: 'TU1 R:03_0_R Comdty' -> 'TU1 Comdty'."""
    return _ROLL_SPEC_RE.sub("", ticker)


def _seed(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


class MockProvider(DataProvider):
    def __init__(self, today: str = "2026-07-21"):
        self.today = pd.Timestamp(today)
        self._paths: dict[str, pd.Series] = {}
        self._roll_offset: dict[str, int] = {}
        self._contract_roots: dict[str, str] = {}
        self._tick_count = 0

    # ---------------- synthetic paths ---------------------------------
    def _base_path(self, root: str) -> pd.Series:
        if root not in self._paths:
            idx = pd.bdate_range(EPOCH, HORIZON)
            rng = np.random.default_rng(_seed(root))
            rets = rng.normal(0.0002, 0.01, len(idx))
            price0 = PRICE0.get(root.rsplit(" ", 1)[0], 100.0)
            self._paths[root] = pd.Series(price0 * np.exp(np.cumsum(rets)), index=idx)
        return self._paths[root]

    def _price_path(self, ticker: str) -> pd.Series:
        if ticker in self._contract_roots:
            # actual front contract: quotes at the unadjusted generic's level,
            # like the real world (front price == last level of the series)
            return self._base_path(self._contract_roots[ticker])
        root = _root(ticker)
        m = _GENERIC_RE.match(root)
        if m and m.group(2) == "2":
            # generic 2 = generic 1 in mild contango, so futures carry is well-defined
            return self._base_path(m.group(1) + "1" + m.group(3)) * 0.995
        path = self._base_path(root)
        if ticker != root:
            # Adjusted series: like real backward-ratio adjustment, only the
            # history BEFORE each roll is rescaled — the latest point stays
            # anchored at the current front price.  factor(t) = r^(rolls
            # occurred after t), so every new roll rewrites the whole past.
            quarters = (path.index.year - 2010) * 4 + (path.index.month - 1) // 3
            exponent = np.clip(self._roll_count(root) - quarters, 0, None)
            path = path * (ROLL_RATIO ** exponent)
        return path

    # ---------------- rolls -------------------------------------------
    def _roll_count(self, root: str) -> int:
        quarters = (self.today.year - 2010) * 4 + (self.today.month - 1) // 3
        return quarters + self._roll_offset.get(root, 0)

    def advance_roll(self, ticker: str) -> None:
        """Force a roll of the generic behind `ticker` (test hook)."""
        root = _root(ticker)
        self._roll_offset[root] = self._roll_offset.get(root, 0) + 1

    # ---------------- DataProvider interface ---------------------------
    def history(self, ticker: str, start: str, end: str, field: str = "PX_LAST") -> pd.Series:
        path = self._price_path(ticker)
        end_ts = min(pd.Timestamp(end), self.today)
        return path.loc[pd.Timestamp(start): end_ts].rename(ticker)

    def snapshot(self, tickers: list[str], fields: list[str]) -> pd.DataFrame:
        out = {}
        for t in tickers:
            path = self._price_path(t)
            last = float(path.loc[:self.today].iloc[-1])
            row = {}
            for f in fields:
                if f == "PX_BID":
                    row[f] = last * (1 - 1e-4)
                elif f == "PX_ASK":
                    row[f] = last * (1 + 1e-4)
                else:
                    row[f] = last
            out[t] = row
        return pd.DataFrame.from_dict(out, orient="index")[fields]

    def current_generic_ticker(self, generic_ticker: str) -> str:
        root = _root(generic_ticker)
        contract = f"{root.split()[0]}_C{self._roll_count(root)}"
        self._contract_roots[contract] = root
        return contract

    def contract_info(self, contract_ticker: str) -> dict:
        return {
            "FUT_CONT_SIZE": 1000.0,
            "CRNCY": "USD",
            "LAST_TRADEABLE_DT": (self.today + pd.Timedelta(days=60)).strftime("%Y-%m-%d"),
            "FUT_TICK_SIZE": 0.01,
        }

    def subscribe(self, tickers: list[str]) -> Iterator[Tick]:
        while True:
            for t in tickers:
                path = self._price_path(t)
                last = float(path.loc[:self.today].iloc[-1])
                self._tick_count += 1
                wiggle = 1 + 0.0005 * np.sin(self._tick_count)
                px = last * wiggle
                yield Tick(
                    ticker=t,
                    last_price=px,
                    bid=px * (1 - 1e-4),
                    ask=px * (1 + 1e-4),
                    event_ts=pd.Timestamp.now(tz="UTC"),
                )
