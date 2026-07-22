"""Command-line entry points.

    python -m prt.cli ingest    --db prt.db [--mock] [--as-of 2026-07-21]
    python -m prt.cli backtest  --db prt.db
    python -m prt.cli targets   --db prt.db          (compute & persist latest targets)
    python -m prt.cli stream    --db prt.db [--mock] [--max-ticks N]
    python -m prt.cli fill      --db prt.db --instrument TY --contracts -3 --notional -350000 --price 112.5
    python -m prt.cli orders    --db prt.db
"""

from __future__ import annotations

import argparse

import pandas as pd

from prt.backtest import run_backtest
from prt.config import load_config
from prt.data.stream import StreamDaemon
from prt.db import Database
from prt.live.monitor import generate_orders
from prt.portfolio.construction import compute_targets
from prt.signals.base import compute_all_forecasts
from prt.signals.dataview import DataView


def _provider(mock: bool):
    if mock:
        from prt.data.mock import MockProvider

        return MockProvider()
    from prt.data.bloomberg import BloombergProvider

    return BloombergProvider()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="prt")
    p.add_argument("command", choices=["ingest", "backtest", "targets", "stream", "fill", "orders"])
    p.add_argument("--db", default="prt.db")
    p.add_argument("--config", default=None)
    p.add_argument("--mock", action="store_true", help="use the synthetic MockProvider")
    p.add_argument("--as-of", default=None)
    p.add_argument("--max-ticks", type=int, default=None)
    p.add_argument("--instrument")
    p.add_argument("--contracts", type=float)
    p.add_argument("--notional", type=float)
    p.add_argument("--price", type=float)
    args = p.parse_args(argv)

    config = load_config(args.config)
    db = Database(args.db)

    if args.command == "ingest":
        from prt.data.ingest import daily_update

        summaries = daily_update(db, _provider(args.mock), config, as_of=args.as_of)
        for s in summaries:
            print(s)
    elif args.command == "backtest":
        result = run_backtest(db, config)
        print(f"run_id={result.run_id}")
        for k, v in result.stats.items():
            print(f"  {k}: {v}")
    elif args.command == "targets":
        view = DataView(db, config)
        forecasts = compute_all_forecasts(db, config, view)
        compute_targets(db, config, view, forecasts, persist=True)
        print(db.latest_targets().to_string())
    elif args.command == "stream":
        daemon = StreamDaemon.for_universe(db, _provider(args.mock), config, max_ticks=args.max_ticks)
        print(f"streamed {daemon.run()} ticks")
    elif args.command == "fill":
        db.add_fill(
            args.instrument,
            contracts=args.contracts,
            notional_usd=args.notional or 0.0,
            price=args.price,
            ts=pd.Timestamp.now(tz="UTC"),
        )
        print(db.positions().to_string())
    elif args.command == "orders":
        orders = generate_orders(db, config)
        print(orders.to_string() if not orders.empty else "nothing to trade")


if __name__ == "__main__":
    main()
