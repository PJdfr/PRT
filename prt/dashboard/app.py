"""Streamlit dashboard (localhost, read-only on the DB).

Run:  streamlit run prt/dashboard/app.py -- --db prt.db [--config config/config.yaml]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from prt.config import load_config
from prt.db import Database
from prt.live.monitor import generate_orders, pnl_since_close, preview_targets
from prt.signals.base import CONVENTIONS_DOC, get_signal
from prt.signals.lab import forecast_correlation


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="prt.db")
    p.add_argument("--config", default=None)
    known, _ = p.parse_known_args(sys.argv[1:])
    return known


def main() -> None:  # pragma: no cover - UI, exercised manually
    import streamlit as st

    args = _args()
    st.set_page_config(page_title="PRT", layout="wide")
    config = load_config(args.config)
    db = Database(args.db)

    tabs = st.tabs(
        ["PnL live", "Positions & risque", "Signaux", "Ordres", "Corrélations", "Santé data", "Méthodo"]
    )

    with tabs[0]:
        st.subheader("PnL depuis le dernier close")
        pnl = pnl_since_close(db, config)
        if pnl.empty:
            st.info("Aucune position (table fills vide) ou pas de quotes live.")
        else:
            total = pnl["pnl_usd"].sum()
            st.metric("PnL total (USD)", f"{total:,.0f}")
            st.dataframe(pnl.style.format(precision=2))

    with tabs[1]:
        st.subheader("Book courant")
        pos = db.positions()
        st.dataframe(pos if not pos.empty else pd.DataFrame({"info": ["book vide"]}))
        st.subheader("Derniers targets")
        st.dataframe(db.latest_targets())

    with tabs[2]:
        st.subheader("Forecasts courants + préview (si close = niveau live)")
        prev = preview_targets(db, config)
        if prev.empty:
            st.info("Pas de forecasts — lancer l'ingestion puis un backtest/compute.")
        else:
            st.dataframe(prev.style.format(precision=2))
        sig = st.selectbox("Historique du signal", sorted(db.forecasts()["signal"].unique()) if not db.forecasts().empty else [])
        if sig:
            panel = db.forecasts(signal=sig).pivot(index="exec_date", columns="instrument_id", values="value")
            st.line_chart(panel.tail(250))
            st.caption("Forecasts, 250 derniers jours")

    with tabs[3]:
        st.subheader("Ordres à exécuter (targets vs book)")
        orders = generate_orders(db, config)
        st.dataframe(orders if not orders.empty else pd.DataFrame({"info": ["rien à trader"]}))
        st.caption("Saisir les fills via prt.db.Database.add_fill (ou le CLI).")

    with tabs[4]:
        st.subheader("Corrélation des forecasts entre signaux")
        signals = sorted(db.forecasts()["signal"].unique()) if not db.forecasts().empty else []
        if signals:
            st.dataframe(forecast_correlation(db, signals).style.format(precision=2))
        runs = db.backtest_runs()
        st.subheader("Backtests stockés")
        st.dataframe(runs)
        if not runs.empty:
            run_id = st.selectbox("Run", runs["run_id"].tolist())
            fund = db.backtest_pnl(run_id, "fund")
            if not fund.empty:
                st.line_chart(fund.cumsum())
            sub = db.backtest_pnl(run_id, "signal")
            if not sub.empty:
                st.subheader("Corrélation des PnL subsystems")
                st.dataframe(sub.corr().style.format(precision=2))

    with tabs[5]:
        st.subheader("Fraîcheur des quotes live")
        quotes = db.live_quotes()
        if not quotes.empty:
            quotes["age_s"] = (pd.Timestamp.now(tz="UTC") - quotes["received_at"]).dt.total_seconds()
            st.dataframe(quotes)
        st.subheader("Derniers full-refresh (rolls / re-ajustements)")
        st.dataframe(db.refreshes().tail(30))
        st.subheader("Consommation Bloomberg")
        qlog = db.query_log()
        if not qlog.empty:
            qlog["day"] = qlog["ts"].str[:10]
            st.bar_chart(qlog.groupby("day")["n_points"].sum())

    with tabs[6]:
        st.subheader("Méthodologie des signaux")
        st.markdown("#### Conventions communes")
        st.markdown(CONVENTIONS_DOC)
        st.divider()
        for name, weight in sorted(config.signal_weights.items()):
            sig = get_signal(name)
            params = config.signal_params.get(name, {})
            suffix = f" — paramètres : {params}" if params else ""
            with st.expander(f"{name} (poids {weight:g}){suffix}", expanded=True):
                st.markdown(sig.whitepaper or "_pas encore documenté_")


if __name__ == "__main__":
    main()
