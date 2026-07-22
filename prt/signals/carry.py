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
    whitepaper = r"""
**Futures** — pente relative du term structure entre les deux premiers
génériques **non ajustés** $G_1, G_2$ (espacement trimestriel, annualisé
$\times 4$), risk-adjusted par la vol annualisée de l'instrument :

$$c(t)=\frac{G_1(t)-G_2(t)}{G_1(t)}\times 4
\qquad
X(t)=\frac{c(t)}{\hat\sigma_{\mathrm{ann}}(t)}$$

Un marché en backwardation ($G_1>G_2$) porte un carry long positif, un
marché en contango un carry négatif.

**FX** — la série carry-adjusted (CR) est un indice total-return ; le carry
réalisé sur un an est l'écart entre son return et celui du spot $S$ :

$$X(t)=\frac{1}{\hat\sigma_{\mathrm{ann}}(t)}\left[\frac{CR(t)}{CR(t-252)}-\frac{S(t)}{S(t-252)}\right]$$

Normalisation commune vers $[-5,+5]$. (Amélioration prévue : carry FX
*ex ante* via les points de forward 1M du terminal.)
"""

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
