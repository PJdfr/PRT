"""Build a complete fake environment for the dashboard.

Runs the REAL pipeline on the MockProvider's synthetic prices for the full
60-instrument universe: ingestion -> signals -> backtest -> targets, then
seeds a plausible book (fills at ~50% of targets so the order blotter has
something to show) and streams fake live quotes.

    uv run python scripts/demo_env.py --db demo.db
    uv run streamlit run prt/dashboard/app.py -- --db demo.db
"""

from __future__ import annotations

import argparse

import pandas as pd

from prt.backtest import run_backtest
from prt.config import load_config
from prt.data.ingest import daily_update
from prt.data.mock import MockProvider
from prt.data.stream import StreamDaemon
from prt.db import Database
from prt.portfolio.construction import compute_targets
from prt.signals.base import compute_all_forecasts
from prt.signals.dataview import DataView


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="demo.db")
    p.add_argument("--today", default="2026-07-21")
    args = p.parse_args()

    config = load_config()
    db = Database(args.db)
    provider = MockProvider(today=args.today)

    print(f"[1/5] ingesting synthetic history for {len(config.instruments)} instruments ...")
    daily_update(db, provider, config, as_of=args.today)

    print("[2/5] running backtest (persists forecasts + subsystem PnLs) ...")
    result = run_backtest(db, config)
    print(f"      run_id={result.run_id} stats={result.stats}")

    print("[3/5] computing & persisting latest targets ...")
    view = DataView(db, config)
    forecasts = compute_all_forecasts(db, config, view, persist=False)
    compute_targets(db, config, view, forecasts, persist=True)

    print("[4/5] seeding a demo book (fills at ~50% of targets) ...")
    targets = db.latest_targets().set_index("instrument_id")
    seeded = 0
    for inst_id, row in targets.iterrows():
        inst = config.instruments.get(inst_id)
        if inst is None or seeded >= 12:
            continue
        if inst.is_future:
            contracts = row["target_contracts"]
            if pd.isna(contracts) or contracts == 0:
                continue
            fill_contracts = round(float(contracts) * 0.5)
            if fill_contracts == 0:
                continue
            frac = fill_contracts / float(contracts)
            db.add_fill(inst_id, contracts=fill_contracts,
                        notional_usd=float(row["target_notional"]) * frac,
                        price=float(row["price"]), note="demo seed")
        else:
            notional = float(row["target_notional"])
            if abs(notional) < 1e5:
                continue
            db.add_fill(inst_id, contracts=None, notional_usd=notional * 0.5,
                        price=float(row["price"]), note="demo seed")
        seeded += 1
    print(f"      {seeded} positions in the book")

    print("[5/5] streaming fake live quotes ...")
    StreamDaemon.for_universe(db, provider, config, max_ticks=200).run()
    print(f"done -> {args.db}")


if __name__ == "__main__":
    main()
