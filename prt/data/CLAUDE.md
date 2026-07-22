# prt/data — Bloomberg boundary (context for agents)

This is the ONLY block allowed to talk to Bloomberg. Everything else reads
the DB. If you find yourself importing xbbg anywhere else, stop.

## Providers

- `provider.py` — `DataProvider` interface (`history`, `snapshot`,
  `current_generic_ticker`, `contract_info`, `subscribe`).
- `bloomberg.py` — xbbg implementation, **lazy import**, only works on the
  terminal machine. Field names / roll-spec ticker syntax are flagged for
  on-terminal validation.
- `mock.py` — deterministic synthetic paths (seeded by ticker). Simulates
  quarterly rolls AND the backward-ratio history rewrite (`advance_roll`
  test hook, `today` settable). CI runs entirely on this.

## Ingestion policy (`ingest.py`) — the roll logic, do not simplify it away

Adjusted generics (`ES1 R:03_0_R Index`) have their WHOLE history rewritten
by Bloomberg at every roll. Therefore:

1. every daily run compares `FUT_CUR_GEN_TICKER` with the stored active
   contract, and re-pulls the last `checksum_points` values to detect
   silent re-adjustments;
2. roll or checksum mismatch -> FULL refresh of the adjusted series, logged
   in `refresh_log` with a reason (`roll_detected` / `checksum_mismatch`);
3. otherwise plain append of missing dates;
4. unadjusted generics (g1/g2, used for carry) and FX spots are
   append-only — their history never changes;
5. every provider hit is logged in `bbg_query_log` (quota watching).

Timestamps are attached HERE at ingestion via
`prt.instruments.master.close_ts_frame` — a stored row without
event_ts/knowledge_ts is a bug.

## Stream (`stream.py`)

`StreamDaemon` consumes `provider.subscribe` (Bloomberg real-time
subscriptions in production) and upserts ONLY the latest quote per ticker
into `live_quotes` (no tick history). Reconnect with exponential backoff.
Consumers never know where a quote came from; a `bdp` snapshot fallback
repopulating `live_quotes` is a valid degraded mode.
