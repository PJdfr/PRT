"""Point-in-time DataView — the only door signals have to the data.

A signal receives series *indexed by execution date*: the row for exec date
t contains the last value whose knowledge_ts is strictly before the
decision instant for executing at the close of t.  The one-day lag on
prices is therefore a provable consequence of timestamps, never a shift
applied downstream — and calendar features (known in advance) carry no lag
at all, so e.g. a seasonality signal is NOT spuriously delayed.

Live preview: `overrides` injects a hypothetical close (today's live price,
knowledge_ts = now) so that "if today closes here" forecasts for tomorrow
come out of the exact same machinery.
"""

from __future__ import annotations

import pandas as pd

from prt.config import Config
from prt.db import Database
from prt.instruments.master import decision_ts


class DataView:
    def __init__(
        self,
        db: Database,
        config: Config,
        overrides: dict[tuple[str, str], tuple[pd.Timestamp, float, pd.Timestamp]] | None = None,
    ):
        """overrides: (instrument_id, series_type) -> (date, value, knowledge_ts)."""
        self.db = db
        self.config = config
        self._overrides = overrides or {}
        self._source_cache: dict[tuple[str, str], pd.DataFrame] = {}
        self._decision_cache: dict[str, pd.Series] = {}
        self._price_cache: dict[tuple[str, str, str], pd.Series] = {}

    # ---------------- raw sources (internal) ---------------------------
    def _source(self, inst_id: str, series_type: str) -> pd.DataFrame:
        key = (inst_id, series_type)
        if key not in self._source_cache:
            df = self.db.series(inst_id, series_type)
            ov = self._overrides.get(key)
            if ov is not None:
                d, v, kts = ov
                row = pd.DataFrame(
                    {"value": [float(v)],
                     "event_ts": [pd.Timestamp(kts)],
                     "knowledge_ts": [pd.Timestamp(kts)]},
                    index=pd.DatetimeIndex([pd.Timestamp(d)]),
                )
                df = pd.concat([df[df.index < pd.Timestamp(d)], row])
            self._source_cache[key] = df
        return self._source_cache[key]

    # ---------------- execution grid -----------------------------------
    def exec_dates(self, inst_id: str) -> pd.DatetimeIndex:
        """Trading grid of an instrument + the next business day (the date we
        can already decide for once the last stored close is known)."""
        dates = self._source(inst_id, "adjusted").index
        if len(dates) == 0:
            return pd.DatetimeIndex([])
        return dates.append(pd.DatetimeIndex([dates[-1] + pd.offsets.BDay(1)]))

    def _decisions(self, inst_id: str) -> pd.Series:
        if inst_id not in self._decision_cache:
            inst = self.config.instrument(inst_id)
            dates = self.exec_dates(inst_id)
            dec = pd.DatetimeIndex([decision_ts(inst, d, self.config) for d in dates])
            self._decision_cache[inst_id] = pd.Series(dec, index=dates)
        return self._decision_cache[inst_id]

    # ---------------- point-in-time series ------------------------------
    def prices(self, inst_id: str, series_type: str = "adjusted", for_inst: str | None = None) -> pd.Series:
        """Series indexed by exec dates of `for_inst` (default: same instrument);
        value at t = last value of (inst_id, series_type) known strictly
        before the decision instant of `for_inst` for the close of t."""
        for_inst = for_inst or inst_id
        key = (inst_id, series_type, for_inst)
        if key not in self._price_cache:
            self._price_cache[key] = self._asof(inst_id, series_type, for_inst)["value"].rename(
                f"{inst_id}.{series_type}"
            )
        return self._price_cache[key]

    def returns(self, inst_id: str, for_inst: str | None = None) -> pd.Series:
        return self.prices(inst_id, "adjusted", for_inst).pct_change()

    def _asof(self, inst_id: str, series_type: str, for_inst: str) -> pd.DataFrame:
        dec = self._decisions(for_inst)
        left = pd.DataFrame({"exec_date": dec.index, "decision_ts": dec.values}).sort_values("decision_ts")
        src = self._source(inst_id, series_type)
        if src.empty:
            out = left.assign(value=float("nan"), used_knowledge_ts=pd.NaT)
        else:
            right = (
                src.reset_index(names="src_date")[["knowledge_ts", "value", "src_date"]]
                .sort_values("knowledge_ts")
                .rename(columns={"knowledge_ts": "used_knowledge_ts"})
            )
            out = pd.merge_asof(
                left,
                right,
                left_on="decision_ts",
                right_on="used_knowledge_ts",
                direction="backward",
                allow_exact_matches=False,
            )
        return out.set_index(pd.DatetimeIndex(out.pop("exec_date")))

    def audit(self, inst_id: str, series_type: str, for_inst: str | None = None) -> pd.DataFrame:
        """decision_ts vs the knowledge_ts actually used — for the generic
        anti-lookahead test (used_knowledge_ts must be < decision_ts)."""
        return self._asof(inst_id, series_type, for_inst or inst_id)

    # ---------------- calendar features (known in advance, no lag) ------
    def calendar(self, inst_id: str) -> pd.DataFrame:
        dates = self.exec_dates(inst_id)
        df = pd.DataFrame(index=dates)
        df["month"] = dates.month
        df["day"] = dates.day
        df["dow"] = dates.dayofweek
        nxt = dates.to_series().shift(-1)
        last_next = dates[-1] + pd.offsets.BDay(1) if len(dates) else pd.NaT
        nxt.iloc[-1] = last_next
        df["is_month_end"] = (nxt.dt.month != dates.month).values
        return df
