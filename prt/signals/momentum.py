"""Trend-following signal: EWMAC (fast minus slow EWMA, vol-normalised),
averaged over several speeds."""

from __future__ import annotations

import pandas as pd

from prt.config import Config
from prt.signals.base import Signal
from prt.signals.dataview import DataView

SPAN_PAIRS = [(8, 32), (16, 64), (32, 128)]


class Momentum(Signal):
    name = "momentum"

    def compute(self, view: DataView, config: Config, inst_id: str) -> pd.Series:
        px = view.prices(inst_id, "adjusted")
        ret = px.pct_change()
        vol_px = (ret.ewm(span=config.fund.vol_ewma_span).std() * px).where(lambda s: s > 0)
        parts = [
            (px.ewm(span=fast).mean() - px.ewm(span=slow).mean()) / vol_px
            for fast, slow in SPAN_PAIRS
        ]
        return pd.concat(parts, axis=1).mean(axis=1)
