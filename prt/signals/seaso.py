"""Seasonality: calendar-day Sharpe profile over the past n complete years.

Spec:
  * lookback = the last n COMPLETE calendar years (Jan 1 -> Dec 31); the
    current year is excluded — per-year detrending needs the full year, so
    including it would be lookahead;
  * returns are detrended per year (each year's mean removed), then
    standardised by their GARCH(1,1) conditional volatility so they are
    not heteroskedastic;
  * every day is mapped onto a 365-day calendar (Feb 29 merged into
    Feb 28), missing days are simply absent from the sample;
  * for each calendar day d and each centred window w in
    [min_window .. max_window]: pool the n x w standardised returns around
    d across the n years and compute their Sharpe; the signal for d is the
    average of those Sharpes over all window sizes;
  * the engine's scale_forecast maps the profile into [-5, +5].

The signal is calendar-known: within year Y it only uses data of years
< Y (via DataView.event_history), so it carries no execution lag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from prt.config import Config
from prt.signals.base import Signal
from prt.signals.dataview import DataView

MIN_OBS = 400  # need at least ~2 years of returns to build a profile

# cumulative day offsets of a non-leap year, per month
_MONTH_OFFSET = np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30])

# profile cache: (db_path, inst, year, params, data fingerprint) -> np.ndarray
_PROFILE_CACHE: dict[tuple, np.ndarray] = {}


def day_of_year_365(index: pd.DatetimeIndex) -> np.ndarray:
    """1..365 ordinal in a non-leap calendar; Feb 29 maps to Feb 28."""
    month = index.month.to_numpy()
    day = np.minimum(index.day.to_numpy(), np.where(month == 2, 28, 31))
    return _MONTH_OFFSET[month - 1] + day


def garch_conditional_vol(returns: pd.Series) -> pd.Series:
    """GARCH(1,1) conditional volatility (zero-mean); EWMA fallback."""
    try:
        from arch import arch_model  # heavy import, keep local

        res = arch_model(returns * 100, mean="Zero", vol="GARCH", p=1, q=1, rescale=False).fit(
            disp="off", show_warning=False
        )
        sigma = pd.Series(res.conditional_volatility, index=returns.index) / 100
        if sigma.notna().all() and (sigma > 0).all():
            return sigma
    except Exception:
        pass
    ewma = returns.ewm(span=32, min_periods=10).std().bfill()
    return ewma.where(ewma > 0, returns.std())


def _circular_window_sum(arr: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros_like(arr)
    for offset in range(-(window // 2), window - window // 2):
        out += np.roll(arr, -offset)
    return out


class Seasonality(Signal):
    name = "seaso"

    def compute(self, view: DataView, config: Config, inst_id: str) -> pd.Series:
        p = config.signal_params.get(self.name, {})
        n_years = int(p.get("n_years", 10))
        w_min = int(p.get("min_window", 10))
        w_max = int(p.get("max_window", 30))

        grid = view.exec_dates(inst_id)
        out = pd.Series(np.nan, index=grid)
        for year in sorted(set(grid.year)):
            profile = self._profile(view, inst_id, int(year), n_years, w_min, w_max)
            if profile is None:
                continue
            mask = grid.year == year
            out[mask] = profile[day_of_year_365(grid[mask]) - 1]
        return out

    def _profile(
        self, view: DataView, inst_id: str, year: int, n_years: int, w_min: int, w_max: int
    ) -> np.ndarray | None:
        hist = view.event_history(inst_id, "adjusted", before_year=year)
        ret = hist.pct_change().dropna()
        ret = ret[ret.index.year >= year - n_years]
        if len(ret) < MIN_OBS:
            return None

        key = (
            view.db.path, inst_id, year, n_years, w_min, w_max,
            len(ret), float(ret.iloc[-1]),
        )
        if key in _PROFILE_CACHE:
            return _PROFILE_CACHE[key]

        # per-year detrend: a window's Sharpe then measures how those days do
        # relative to the rest of their year, not the year's drift
        detrended = ret - ret.groupby(ret.index.year).transform("mean")
        z = detrended / garch_conditional_vol(detrended)
        z = z.dropna()

        doy = day_of_year_365(z.index) - 1
        s1 = np.zeros(365)
        s2 = np.zeros(365)
        cnt = np.zeros(365)
        np.add.at(s1, doy, z.to_numpy())
        np.add.at(s2, doy, z.to_numpy() ** 2)
        np.add.at(cnt, doy, 1.0)

        sharpes = []
        with np.errstate(invalid="ignore", divide="ignore"):
            for w in range(w_min, w_max + 1):
                c = _circular_window_sum(cnt, w)
                m = _circular_window_sum(s1, w) / c
                var = _circular_window_sum(s2, w) / c - m**2
                sharpes.append(m / np.sqrt(var))
            profile = np.nanmean(np.vstack(sharpes), axis=0)
        profile = np.nan_to_num(profile, nan=0.0)

        _PROFILE_CACHE[key] = profile
        return profile
