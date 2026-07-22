"""Momentum: 12 lagged one-month Sharpe sub-signals, performance-weighted.

Spec:
  * take the trailing 252 trading days, split into 12 blocks of 21 days;
  * sub-signal j (j = 0..11) at date t = Sharpe (mean/std of daily returns)
    of the block lagged by j months;
  * each sub-signal's weight = the Sharpe of ITS OWN standalone PnL stream
    (sub-forecast x realised next-period return) over a rolling n-year
    window — computed point-in-time, clipped at 0 and normalised to sum to
    1 (equal weights as fallback when no history or all Sharpes <= 0);
  * combined raw signal = sum_j w_j(t) x sharpe_j(t), then the engine's
    scale_forecast maps it into [-5, +5].
"""

from __future__ import annotations

import pandas as pd

from prt.config import Config
from prt.signals.base import Signal
from prt.signals.dataview import DataView

BLOCK = 21
N_BLOCKS = 12
MIN_WEIGHT_OBS = 252  # need at least a year of sub-signal PnL before weighting


class Momentum(Signal):
    name = "momentum"
    whitepaper = r"""
**Univers** : toutes les séries ajustées. **Données** : returns quotidiens
$r_t$ de la série backward-ratio-adjusted, vus au travers de la DataView.

Les 252 derniers jours sont découpés en 12 blocs de 21 jours. Le
sous-signal $j$ ($j=0\ldots 11$) est le Sharpe du bloc lagué de $j$ mois :

$$S_j(t)=\frac{\operatorname{mean}\big(r_{t-21(j+1)+1},\dots,r_{t-21j}\big)}{\operatorname{std}\big(r_{t-21(j+1)+1},\dots,r_{t-21j}\big)}$$

Chaque sous-signal est pondéré par la qualité de **son propre** track
record : son PnL standalone, aligné à la date de réalisation (le forecast
émis en $t-2$ est réalisé par le return connu en $t$) :

$$\pi_j(t)=S_j(t-2)\,r_t
\qquad
\tilde w_j(t)=\max\Big(0,\ \widehat{\mathrm{Sharpe}}_{[t-252n,\ t]}\big(\pi_j\big)\Big)$$

$$w_j(t)=\frac{\tilde w_j(t)}{\sum_k \tilde w_k(t)}
\quad\text{(équipondération } 1/12 \text{ si } \textstyle\sum_k \tilde w_k=0
\text{ ou historique} < 1 \text{ an)}$$

Signal brut $M(t)=\sum_{j} w_j(t)\,S_j(t)$, puis normalisation commune vers
$[-5,+5]$. Paramètre : $n$ = fenêtre (années) du Sharpe de pondération.
"""

    def compute(self, view: DataView, config: Config, inst_id: str) -> pd.Series:
        n_years = int(config.signal_params.get(self.name, {}).get("n_years", 10))
        px = view.prices(inst_id, "adjusted")
        ret = px.pct_change()

        subs = {}
        for j in range(N_BLOCKS):
            r = ret.shift(BLOCK * j)
            subs[j] = r.rolling(BLOCK).mean() / r.rolling(BLOCK).std()
        sharpes = pd.DataFrame(subs)

        # Standalone PnL of each sub-signal, aligned at REALISATION time so a
        # rolling window ending at t only uses information available at t:
        # the forecast made at row t-2 is realised by the close-to-close
        # return that appears in the as-known series at row t.
        perf = sharpes.shift(2).mul(ret, axis=0)
        win = 252 * n_years
        mu = perf.rolling(win, min_periods=MIN_WEIGHT_OBS).mean()
        sd = perf.rolling(win, min_periods=MIN_WEIGHT_OBS).std()
        weights = (mu / sd).clip(lower=0.0)
        total = weights.sum(axis=1)
        weights = weights.div(total.where(total > 0), axis=0)
        weights = weights.fillna(1.0 / N_BLOCKS)  # early history / all-negative

        return (weights * sharpes).sum(axis=1, min_count=1)
