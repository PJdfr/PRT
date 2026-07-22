# PRT — systematic macro fund (context for agents)

One-person systematic fund on a single Bloomberg terminal. Python, SQLite,
Streamlit on localhost. Instruments: Bloomberg backward-ratio-adjusted
futures generics + FX carry-adjusted series (`USDJPYCR Curncy`), DM only.

## Commands (uv is the package manager)

```bash
uv sync --extra dev          # install (creates .venv)
uv run pytest                # full suite, mock data, NO terminal needed
uv run ruff check prt tests  # lint
uv run python -m prt.cli ingest --db prt.db --mock   # also: backtest, targets, stream, fill, orders
uv run streamlit run prt/dashboard/app.py -- --db prt.db   # needs --extra dashboard
```

## Architecture — building blocks, DB is the only meeting point

```
Bloomberg (xbbg) -> prt/data -> prt/db (SQLite) -> prt/signals -> prt/portfolio
                                        |               -> prt/backtest | prt/live -> prt/dashboard
```

- `prt/config` — YAML loader; `config/config.yaml` is the single source of
  truth (universe, capital, vol target, settle times, costs, conventions).
- `prt/db` — persistence; every table described in `database.py` SCHEMA.
- `prt/data` — the ONLY block allowed to touch Bloomberg. `provider.py`
  (interface), `bloomberg.py` (xbbg, lazy import, requires terminal),
  `mock.py` (deterministic synthetic data incl. simulated rolls),
  `ingest.py` (daily batch, roll detection, full refresh), `stream.py`
  (real-time -> live_quotes).
- `prt/signals` — point-in-time `DataView`, plug-in signal engine
  (see prt/signals/CLAUDE.md), Signal Lab (correlations, marginal Sharpe).
- `prt/portfolio` — forecasts -> vol-targeted notional -> integer contracts
  (futures) or pair notional (FX); no-trade buffering.
- `prt/backtest` — daily simulator + per-signal subsystem PnLs.
- `prt/live` — PnL since last close, live-level signal preview, orders.
- `prt/dashboard` — Streamlit, strictly read-only on the DB.

Blocks never talk laterally; everything goes through the DB and config.
Full diagram: docs/architecture.md.

## Invariants — do not break these

1. **No Bloomberg outside `prt/data`.** `xbbg` is imported lazily in
   `prt/data/bloomberg.py` and nowhere else. Everything must run without a
   terminal (CI does).
2. **Timestamps, not dates.** Every observation carries `event_ts` and
   `knowledge_ts` (UTC). `knowledge_ts = max(event + publish lag, 22:30 UTC
   daily batch)`. Dates are join labels only. Time logic lives ONLY in
   `prt/instruments/master.py`.
3. **Never `shift()` a signal.** The one-day execution lag is enforced by
   the `DataView` comparing `knowledge_ts` vs per-instrument decision
   instants. Calendar-known information (seasonality, event dates) carries
   NO lag. `tests/test_dataview.py::test_no_lookahead_generic` is the
   guarantee — keep it passing.
4. **PnL = returns x USD notional.** Historical price levels of
   ratio-adjusted series are fictitious; never compute PnL or notional from
   them (the *latest* level equals the real front price — that one is fine
   for live sizing).
5. **Adjusted series are rewritten at every roll.** Only full-refresh them
   on roll detection or checksum mismatch, and log it (`refresh_log`).
   Unadjusted generics (g1/g2) and spots are append-only.
6. **Eco releases are as-released.** Revisions are new rows, never updates.
7. **Forecasts live in [-5, +5]**, average strength ±2.5, produced by
   `scale_forecast`. Backtest and live run the exact same signal+portfolio
   code path.
8. Every building block keeps its own tests green in `tests/test_<block>.py`.

## Conventions

- Fund: $100M capital, 10% vol target, IDM 1.5, leverage unconstrained.
- Execution: signal from close t-1 -> executed at close t; cost =
  per-ticker half-spread bps on traded notional.
- FX positions are pair notional (implicit forward), futures are integer
  contracts. Non-USD exposure converts via `fx_conversion` in the config.
- Bloomberg tickers / roll-spec syntax / settle times in the config are
  best-effort and flagged for validation on the terminal (see README).
