"""Signal engine: base class, registry, normalisation, batch computation.

A signal is a plug-in: subclass `Signal`, set `name`, implement
`compute(view, config, inst_id) -> raw pd.Series indexed by exec date`.
The engine normalises raw values into forecasts in [-cap, +cap] (cap = 5,
expected average |forecast| = 2.5) so every signal speaks the same units —
which is what makes position sizing generic and cross-signal correlation
analysis trivial.

Dropping a new module defining a Signal subclass in prt/signals/ and
importing it in prt/signals/__init__.py is all it takes to add a strategy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from prt.config import Config
from prt.db import Database
from prt.signals.dataview import DataView

_REGISTRY: dict[str, type["Signal"]] = {}

CONVENTIONS_DOC = r"""
Tout signal produit un **forecast** $F\in[-5,+5]$, force moyenne $\pm 2.5$.
Le moteur normalise la sortie brute $X$ de chaque signal par sa force
moyenne *expanding* (uniquement le passé, aucun lookahead) :

$$F(t)=\operatorname{clip}\!\Big(2.5\;\frac{X(t)}{\tfrac{1}{|u\le t|}\sum_{u\le t}\lvert X(u)\rvert}\,,\;-5,\;+5\Big)$$

Combinaison des signaux (poids $w_s$ de la config, IDM) puis vol targeting :

$$F_{\mathrm{comb}}=\operatorname{clip}\!\Big(\mathrm{IDM}\cdot\frac{\sum_s w_s F_s}{\sum_s w_s},-5,+5\Big)
\qquad
N_i(t)=\frac{F_{\mathrm{comb}}(t)}{2.5}\cdot\frac{w_i\,K\,\mathrm{IDM}\,\sigma^{\star}}{\hat\sigma_i(t)}$$

avec $K$ le capital, $\sigma^{\star}$ la vol cible du fund, $\hat\sigma_i$
la vol EWMA de l'instrument, $w_i=1/N$. Buffering : on ne traite que si la
cible sort d'une bande de $\pm 10\%$ de la position moyenne. Les données
vues par un signal passent par la DataView point-in-time : la ligne
d'exécution $t$ ne contient que l'information connaissable avant l'instant
de décision de $t$ (prix $\le t-1$, calendrier $\le t$) — le lag d'un jour
est une conséquence des timestamps, jamais un `shift`.
"""


class Signal(ABC):
    name: str = ""
    whitepaper: str = ""  # markdown + LaTeX, rendered on the dashboard

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.name:
            _REGISTRY[cls.name] = cls

    def universe(self, config: Config) -> list[str]:
        return config.universe

    @abstractmethod
    def compute(self, view: DataView, config: Config, inst_id: str) -> pd.Series:
        """Raw (un-normalised) signal, indexed by execution date."""

    def forecasts(self, view: DataView, config: Config, inst_id: str) -> pd.Series:
        return scale_forecast(self.compute(view, config, inst_id), config)


def scale_forecast(raw: pd.Series, config: Config, min_periods: int = 60) -> pd.Series:
    """Normalise a raw signal to the common forecast units.

    forecast = forecast_avg * raw / expanding mean(|raw|), clipped to +-cap.
    The expanding window only uses the signal's own past — no lookahead.
    """
    denom = raw.abs().expanding(min_periods=min_periods).mean()
    f = config.fund.forecast_avg * raw / denom.where(denom > 0)
    return f.clip(-config.fund.forecast_cap, config.fund.forecast_cap)


def available_signals() -> list[str]:
    return sorted(_REGISTRY)


def get_signal(name: str) -> Signal:
    return _REGISTRY[name]()


def compute_all_forecasts(
    db: Database,
    config: Config,
    view: DataView | None = None,
    signal_names: list[str] | None = None,
    persist: bool = True,
) -> dict[str, dict[str, pd.Series]]:
    """Compute forecasts for every (signal, instrument); optionally persist.

    Returns {signal_name: {instrument_id: forecast series}}.
    """
    view = view or DataView(db, config)
    names = signal_names if signal_names is not None else list(config.signal_weights)
    out: dict[str, dict[str, pd.Series]] = {}
    for name in names:
        sig = get_signal(name)
        per_inst: dict[str, pd.Series] = {}
        for inst_id in sig.universe(config):
            f = sig.forecasts(view, config, inst_id)
            per_inst[inst_id] = f
            if persist:
                db.write_forecasts(name, inst_id, f)
        out[name] = per_inst
    return out
