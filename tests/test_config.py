"""Config block: the real production config must load and be coherent."""

from prt.config import load_config


def test_production_config_loads():
    config = load_config()
    assert len(config.instruments) >= 55
    assert config.fund.capital_usd == 100_000_000
    assert config.fund.target_vol == 0.10
    assert set(config.signal_weights) == {"momentum", "carry"}


def test_every_instrument_is_complete():
    config = load_config()
    for inst in config.instruments.values():
        assert inst.adj_ticker
        assert inst.tz and inst.settle_time
        assert inst.cost_bps > 0
        if inst.is_future:
            assert inst.point_value and inst.point_value > 0
            assert inst.g1_ticker and inst.g2_ticker
        else:
            assert inst.spot_ticker


def test_every_non_usd_currency_is_convertible():
    config = load_config()
    for inst in config.instruments.values():
        if inst.currency != "USD":
            conv = config.fx_conversion.get(inst.currency)
            assert conv, f"no fx conversion for {inst.currency}"
            assert conv["pair"] in config.instruments, f"{conv['pair']} not in universe"


def test_asset_classes_present():
    config = load_config()
    classes = {i.asset_class for i in config.instruments.values()}
    assert {"bonds", "equity", "commodity", "fx", "crypto"} <= classes
