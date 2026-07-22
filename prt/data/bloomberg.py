"""Bloomberg provider via xbbg — only usable on the terminal machine.

xbbg/blpapi are imported lazily so the rest of the system installs and runs
anywhere.  All query volumes should be logged by the caller (ingest does).

NOTE: field names and the generic roll-spec ticker syntax (e.g.
'ES1 R:03_0_R Index') must be validated on the terminal before live use.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from prt.data.provider import DataProvider, Tick


def _blp():
    try:
        from xbbg import blp  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "xbbg is not installed - this provider only works on the Bloomberg "
            "terminal machine (pip install 'prt[bloomberg]')"
        ) from e
    return blp


class BloombergProvider(DataProvider):  # pragma: no cover - requires terminal
    def history(self, ticker: str, start: str, end: str, field: str = "PX_LAST") -> pd.Series:
        df = _blp().bdh(tickers=ticker, flds=field, start_date=start, end_date=end)
        if df.empty:
            return pd.Series(dtype=float, name=ticker)
        s = df.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return s.rename(ticker).dropna()

    def snapshot(self, tickers: list[str], fields: list[str]) -> pd.DataFrame:
        return _blp().bdp(tickers=tickers, flds=fields)

    def current_generic_ticker(self, generic_ticker: str) -> str:
        df = _blp().bdp(tickers=generic_ticker, flds="FUT_CUR_GEN_TICKER")
        raw = str(df.iloc[0, 0])
        # FUT_CUR_GEN_TICKER omits the sector suffix; re-attach it.
        suffix = generic_ticker.split()[-1]
        return raw if raw.endswith(suffix) else f"{raw} {suffix}"

    def contract_info(self, contract_ticker: str) -> dict:
        flds = ["FUT_CONT_SIZE", "CRNCY", "LAST_TRADEABLE_DT", "FUT_TICK_SIZE"]
        df = _blp().bdp(tickers=contract_ticker, flds=flds)
        return {} if df.empty else df.iloc[0].to_dict()

    def subscribe(self, tickers: list[str]) -> Iterator[Tick]:
        flds = ["LAST_PRICE", "BID", "ASK"]
        state: dict[str, dict] = {t: {} for t in tickers}
        for update in _blp().live(tickers=tickers, flds=flds):
            ticker = update.get("TICKER") or update.get("ticker")
            if ticker is None:
                continue
            st = state.setdefault(ticker, {})
            for f in flds:
                if f in update and update[f] is not None:
                    st[f] = float(update[f])
            if "LAST_PRICE" not in st:
                continue
            yield Tick(
                ticker=ticker,
                last_price=st["LAST_PRICE"],
                bid=st.get("BID"),
                ask=st.get("ASK"),
                event_ts=pd.Timestamp.now(tz="UTC"),
            )
