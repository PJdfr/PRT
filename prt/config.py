"""Configuration loading: fund conventions + instrument master.

The YAML file is the single source of truth for the universe and all
conventions.  Everything downstream receives a `Config` object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


@dataclass(frozen=True)
class Instrument:
    id: str
    name: str
    asset_class: str
    kind: str                   # "future" | "fx"
    exchange: str
    tz: str                     # IANA timezone of the settlement
    settle_time: str            # "HH:MM" local settlement time
    adj_ticker: str             # roll-adjusted / carry-adjusted series (signal + returns)
    g1_ticker: str | None       # unadjusted generic 1 (futures)
    g2_ticker: str | None       # unadjusted generic 2 (futures carry)
    spot_ticker: str | None     # spot series (fx)
    currency: str
    point_value: float | None   # contract multiplier (futures); None for fx
    cost_bps: float             # half-spread estimate, bps of notional

    @property
    def is_future(self) -> bool:
        return self.kind == "future"


@dataclass(frozen=True)
class FundConfig:
    capital_usd: float
    target_vol: float
    forecast_cap: float
    forecast_avg: float
    idm: float
    vol_ewma_span: int
    buffer_fraction: float
    publish_lag_minutes: int
    decision_margin_minutes: int


@dataclass(frozen=True)
class DataConfig:
    history_start: str
    daily_batch_utc: str
    checksum_points: int


@dataclass(frozen=True)
class Config:
    fund: FundConfig
    data: DataConfig
    signal_weights: dict[str, float]
    instruments: dict[str, Instrument] = field(default_factory=dict)
    fx_conversion: dict[str, dict] = field(default_factory=dict)

    def instrument(self, inst_id: str) -> Instrument:
        return self.instruments[inst_id]

    @property
    def universe(self) -> list[str]:
        return list(self.instruments)

    def futures(self) -> list[Instrument]:
        return [i for i in self.instruments.values() if i.is_future]


def load_config(path: str | Path | None = None) -> Config:
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG_PATH).read_text())
    return build_config(raw)


def build_config(raw: dict) -> Config:
    """Build a Config from a raw dict (also used by tests with small universes)."""
    exchanges = raw["exchanges"]
    instruments: dict[str, Instrument] = {}
    for spec in raw["instruments"]:
        exch = exchanges[spec["exchange"]]
        inst = Instrument(
            id=spec["id"],
            name=spec.get("name", spec["id"]),
            asset_class=spec["class"],
            kind=spec.get("kind", "future"),
            exchange=spec["exchange"],
            tz=exch["tz"],
            settle_time=str(exch["settle_time"]),
            adj_ticker=spec["adj"],
            g1_ticker=spec.get("g1"),
            g2_ticker=spec.get("g2"),
            spot_ticker=spec.get("spot"),
            currency=spec["ccy"],
            point_value=spec.get("pv"),
            cost_bps=float(spec["cost_bps"]),
        )
        if inst.id in instruments:
            raise ValueError(f"duplicate instrument id: {inst.id}")
        instruments[inst.id] = inst

    f = raw["fund"]
    fund = FundConfig(
        capital_usd=float(f["capital_usd"]),
        target_vol=float(f["target_vol"]),
        forecast_cap=float(f["forecast_cap"]),
        forecast_avg=float(f["forecast_avg"]),
        idm=float(f.get("idm", 1.0)),
        vol_ewma_span=int(f.get("vol_ewma_span", 32)),
        buffer_fraction=float(f.get("buffer_fraction", 0.10)),
        publish_lag_minutes=int(f.get("publish_lag_minutes", 10)),
        decision_margin_minutes=int(f.get("decision_margin_minutes", 10)),
    )
    d = raw.get("data", {})
    data = DataConfig(
        history_start=str(d.get("history_start", "2005-01-01")),
        daily_batch_utc=str(d.get("daily_batch_utc", "22:30")),
        checksum_points=int(d.get("checksum_points", 20)),
    )
    weights = {name: float(s["weight"]) for name, s in raw.get("signals", {}).items()}
    return Config(
        fund=fund,
        data=data,
        signal_weights=weights,
        instruments=instruments,
        fx_conversion=raw.get("fx_conversion", {}),
    )
