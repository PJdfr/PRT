"""Carry signal.

Futures: annualised slope of the term structure between unadjusted generic
1 and generic 2 (quarterly spacing approximation), risk-adjusted by the
instrument's vol.

FX: the carry-adjusted (CR) series is a total-return series, so trailing
CR return minus trailing spot return isolates realised carry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prt.config import Config
from prt.signals.base import Signal
from prt.signals.dataview import DataView

QUARTERS_PER_YEAR = 4
FX_CARRY_WINDOW = 252


class Carry(Signal):
    name = "carry"

    def compute(self, view: DataView, config: Config, inst_id: str) -> pd.Series:
        inst = config.instrument(inst_id)
        px = view.prices(inst_id, "adjusted")
        vol_ann = px.pct_change().ewm(span=config.fund.vol_ewma_span).std() * np.sqrt(252)
        vol_ann = vol_ann.where(vol_ann > 0)

        if inst.is_future and inst.g2_ticker:
            g1 = view.prices(inst_id, "g1_raw")
            g2 = view.prices(inst_id, "g2_raw")
            carry_ann = (g1 - g2) / g1 * QUARTERS_PER_YEAR
        elif not inst.is_future and inst.spot_ticker:
            spot = view.prices(inst_id, "spot")
            carry_ann = px.pct_change(FX_CARRY_WINDOW) - spot.pct_change(FX_CARRY_WINDOW)
        else:
            return pd.Series(0.0, index=px.index)

        return carry_ann / vol_ann
