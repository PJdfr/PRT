"""Live block: stream daemon, PnL since close, preview, orders."""

import numpy as np
import pandas as pd
import pytest

from prt.backtest import run_backtest
from prt.data.provider import DataProvider, Tick
from prt.data.stream import StreamDaemon
from prt.live.monitor import generate_orders, pnl_since_close, preview_targets


@pytest.fixture
def streamed_db(ingested_db, provider, config):
    daemon = StreamDaemon.for_universe(ingested_db, provider, config, max_ticks=30)
    daemon.run()
    return ingested_db


def test_stream_writes_latest_quotes(streamed_db, config):
    quotes = streamed_db.live_quotes()
    assert len(quotes) == len(config.universe)  # one row per ticker, upserted
    assert quotes["last_price"].notna().all()
    assert set(quotes["instrument_id"]) == set(config.universe)


def test_stream_reconnects_after_error(ingested_db, provider, config):
    class Flaky(DataProvider):
        def __init__(self, inner):
            self.inner, self.calls = inner, 0

        def subscribe(self, tickers):
            self.calls += 1
            if self.calls == 1:
                raise ConnectionError("bloomberg session lost")
            yield from self.inner.subscribe(tickers)

        def history(self, *a, **k):
            return self.inner.history(*a, **k)

        def snapshot(self, *a, **k):
            return self.inner.snapshot(*a, **k)

        def current_generic_ticker(self, *a, **k):
            return self.inner.current_generic_ticker(*a, **k)

        def contract_info(self, *a, **k):
            return self.inner.contract_info(*a, **k)

    flaky = Flaky(provider)
    daemon = StreamDaemon.for_universe(ingested_db, flaky, config, max_ticks=5, backoff_seconds=0.0)
    assert daemon.run() == 5
    assert flaky.calls == 2


def test_pnl_since_close_arithmetic(streamed_db, config):
    streamed_db.add_fill("TY", contracts=-3, notional_usd=-350_000, price=112.5)
    streamed_db.add_fill("EURUSD", contracts=None, notional_usd=2_000_000)
    pnl = pnl_since_close(streamed_db, config)

    prev = streamed_db.series("TY", "adjusted")["value"].iloc[-1]
    quotes = streamed_db.live_quotes()
    live = float(quotes[quotes["instrument_id"] == "TY"]["last_price"].iloc[0])
    assert np.isclose(pnl.loc["TY", "pnl_usd"], -3 * 1000 * (live - prev))  # USD future, fx=1

    prev_spot = streamed_db.series("EURUSD", "spot")["value"].iloc[-1]
    live_spot = float(quotes[quotes["instrument_id"] == "EURUSD"]["last_price"].iloc[0])
    assert np.isclose(pnl.loc["EURUSD", "pnl_usd"], 2_000_000 * (live_spot / prev_spot - 1))


def test_preview_uses_live_levels(streamed_db, config):
    prev = preview_targets(streamed_db, config)
    assert set(prev.index) == set(config.universe)
    assert prev["hypothetical"].all()
    assert prev["forecast"].abs().max() <= config.fund.forecast_cap + 1e-9
    # futures previews give integer contracts, fx give notional
    assert prev.loc["TY", "target_contracts"] == round(prev.loc["TY", "target_contracts"])
    assert pd.isna(prev.loc["EURUSD", "target_contracts"])


def test_orders_from_targets_vs_book(ingested_db, config):
    run_backtest(ingested_db, config)  # persists forecasts
    from prt.portfolio.construction import compute_targets
    from prt.signals.base import compute_all_forecasts
    from prt.signals.dataview import DataView

    view = DataView(ingested_db, config)
    forecasts = compute_all_forecasts(ingested_db, config, view, persist=False)
    compute_targets(ingested_db, config, view, forecasts, persist=True)

    orders = generate_orders(ingested_db, config)
    assert not orders.empty  # empty book -> orders to reach targets

    # fill exactly to target -> no residual order for that instrument
    tgt = ingested_db.latest_targets().set_index("instrument_id").loc["TY"]
    ingested_db.add_fill("TY", contracts=float(tgt["target_contracts"]),
                         notional_usd=float(tgt["target_notional"]), price=float(tgt["price"]))
    orders2 = generate_orders(ingested_db, config)
    assert "TY" not in orders2.index


def test_tick_dataclass():
    t = Tick("X", 1.0, 0.9, 1.1, pd.Timestamp.now(tz="UTC"))
    assert t.ticker == "X"
