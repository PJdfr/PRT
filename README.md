# PRT — systematic macro fund infrastructure

One-person systematic fund on a single Bloomberg terminal (via `xbbg`):
Bloomberg roll-adjusted futures + FX carry-adjusted series → SQLite →
signal engine → vol-targeted portfolio → backtest **and** live monitoring
(intraday PnL since close, signal preview, order list) on a localhost
Streamlit dashboard.

## Architecture (building blocks)

The DB is the only meeting point; Bloomberg is touched by exactly one block.

```
Bloomberg (xbbg) ──> prt/data (providers, ingestion, stream) ──> prt/db (SQLite)
                                                                    │
        ┌──────────────┬─────────────────┬──────────────────────────┤
        ▼              ▼                 ▼                          ▼
  prt/signals     prt/portfolio     prt/backtest │ prt/live    prt/dashboard
  (DataView,      (forecast →       (same signal+portfolio     (read-only)
   engine, lab)    contracts)        code path for both)
```

Full block diagram and dependency rules: see `docs/architecture.md`-style
notes in each module docstring.

## Core conventions (all in `config/config.yaml`)

- **Capital / risk**: $100M, 10% annualised vol target, leverage free,
  IDM 1.5, integer contracts for futures, pair notional (implicit forward)
  for FX.
- **Forecasts**: every signal outputs values in **[−5, +5]**, average
  strength ±2.5. Position sizing and cross-signal correlation are generic.
- **Execution**: signal from close t−1 → executed at close t; cost =
  per-ticker half-spread (bps) on every position change. Backtest and live
  use the same convention and the same code.
- **Data**: Bloomberg backward-ratio-adjusted generics are the signal &
  return series. Their history is rewritten at every roll, so ingestion
  detects rolls (`FUT_CUR_GEN_TICKER` + a trailing checksum) and does a
  logged **full refresh** only then; unadjusted generics 1–2 (futures
  carry) and FX spots are append-only. PnL is always returns × notional —
  never price points on the fictitious historical levels of an adjusted
  series.
- **Timestamps**: nothing is indexed by bare dates. Every observation
  carries `event_ts` (settlement instant, per-instrument IANA timezone +
  settle time → UTC) and `knowledge_ts` = max(event + publish lag, the
  22:30 UTC daily batch — data is only "known" once *our* system has it).
  The point-in-time `DataView` compares timestamps against per-instrument
  decision instants; the one-day lag is a provable consequence, **never a
  blanket `shift(1)`** — calendar-driven signals (seasonality, events)
  carry no spurious lag. A generic CI test asserts no-lookahead for every
  (source, target) pair.

## Quickstart (uv, no terminal needed — mock data)

```bash
uv sync --extra dev
uv run python -m prt.cli ingest   --db prt.db --mock     # synthetic history
uv run python -m prt.cli backtest --db prt.db            # stats + persisted PnLs
uv run python -m prt.cli targets  --db prt.db            # tomorrow's targets
uv run python -m prt.cli stream   --db prt.db --mock --max-ticks 200
uv run python -m prt.cli orders   --db prt.db
uv sync --extra dashboard
uv run streamlit run prt/dashboard/app.py -- --db prt.db
```

On the terminal machine: `uv sync --extra bloomberg` and drop `--mock`.

## Adding a signal

One file in `prt/signals/`:

```python
from prt.signals.base import Signal

class MySignal(Signal):
    name = "mysignal"
    def compute(self, view, config, inst_id):
        px = view.prices(inst_id, "adjusted")   # point-in-time, indexed by exec date
        cal = view.calendar(inst_id)            # known-in-advance features, no lag
        return some_series                      # raw; engine normalises to [-5, +5]
```

Import it in `prt/signals/__init__.py`, give it a weight in the config.
Then run the **Signal Lab** before allocating: forecast correlation vs the
existing stack, standalone subsystem-PnL correlation, marginal Sharpe
(`prt/signals/lab.py`, also on the dashboard).

## Tests / CI

`uv run pytest` runs the full chain (ingestion → signals → portfolio →
backtest → live) on the deterministic `MockProvider` — zero Bloomberg dependency;
`xbbg` is imported lazily in exactly one file. Tests needing a terminal are
marked `@pytest.mark.requires_terminal` and skipped by default. GitHub
Actions runs lint + tests on every push.

## To validate on the terminal before live use

- Generic roll-spec ticker syntax (`ES1 R:03_0_R Index`) and crypto generics.
- Settlement times in `config/config.yaml` (`exchanges:` section).
- Field names in `prt/data/bloomberg.py` (`FUT_CUR_GEN_TICKER`, contract
  reference fields, `blp.live` field set).
