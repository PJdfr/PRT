# prt/signals — signal engine (context for agents)

## Adding a signal (the whole point of this repo)

One file, one class:

```python
from prt.signals.base import Signal

class MySignal(Signal):
    name = "mysignal"                       # unique registry key

    def compute(self, view, config, inst_id):
        px = view.prices(inst_id, "adjusted")     # point-in-time, indexed by EXECUTION date
        g2 = view.prices(inst_id, "g2_raw")       # futures term structure
        cal = view.calendar(inst_id)              # month/day/dow/is_month_end — NO lag
        return raw_series                          # any scale; engine normalises
```

Then: import it in `prt/signals/__init__.py`, add a weight under `signals:`
in `config/config.yaml`. Done — sizing, backtest, live preview, orders and
the dashboard pick it up automatically.

Before allocating capital, run the Signal Lab (`lab.py`): forecast
correlation vs existing signals, standalone subsystem-PnL correlation,
marginal Sharpe. The dashboard "Corrélations" tab shows the same.

## Rules

- `compute` returns a RAW series indexed by execution date; `scale_forecast`
  (expanding normalisation, no lookahead) maps it to [-5, +5] with average
  strength ±2.5. Never clip/scale yourself; never persist yourself.
- NEVER call `.shift()` for execution lag and never read the DB directly —
  the `DataView` is the only data door. Row t of `view.prices(...)` already
  contains only what was knowable before t's decision instant (for prices
  that means t-1's close; for calendar features it means t itself).
- Cross-instrument data: `view.prices(other_id, "adjusted", for_inst=inst_id)`
  aligns `other_id`'s series on `inst_id`'s decision instants.
- Signals with no meaningful value for an instrument should return zeros
  (see `Carry` for futures without g2), not raise.
- Eco-event data (phase 2) must be read with `knowledge_ts` = release
  timestamp semantics, same DataView philosophy.

## Files

- `dataview.py` — point-in-time views; anti-lookahead enforced by
  merge_asof on (knowledge_ts < decision_ts). Test: test_dataview.py.
- `base.py` — Signal ABC, registry (`__init_subclass__`), `scale_forecast`,
  `compute_all_forecasts` (persists to the forecasts table).
- `momentum.py`, `carry.py` — built-ins, use them as templates.
- `lab.py` — decorrelation reports.
