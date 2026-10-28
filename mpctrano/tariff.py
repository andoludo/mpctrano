"""Energy tariffs and demand-side flexibility signals for the economic MPC.

A :class:`Tariff` turns a time index into a :class:`TariffSignal`: the energy price of each step
[EUR/kWh], an optional building-level power limit [W] (demand response) and an optional
capacity charge [EUR/kW] applied to the peak power of the horizon. The MPC minimises the
resulting cost, so the heating is shifted towards cheap hours and away from peaks as far as
the comfort band allows: this is the flexibility of the building.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, field_validator, model_validator

HOURS_PER_DAY = 24


@dataclass(frozen=True)
class TariffSignal:
    """Tariff evaluated on a time index."""

    index: pd.DatetimeIndex
    prices: np.ndarray
    """Energy price of each step [EUR/kWh]."""
    power_limits: np.ndarray
    """Maximum total heating power of each step [W] (``inf`` when unconstrained)."""
    capacity_charge: float
    """Price of the peak power over the horizon [EUR/kW]."""
    export_prices: np.ndarray | None = None
    """Remuneration of the exported energy [EUR/kWh] (zeros when the tariff has none)."""
    power_floors: np.ndarray | None = None
    """Minimum billed power of each step [W] (upward activations; zeros when none)."""

    @property
    def export(self) -> np.ndarray:
        return np.zeros(len(self.index)) if self.export_prices is None else self.export_prices

    @property
    def floors(self) -> np.ndarray:
        return np.zeros(len(self.index)) if self.power_floors is None else self.power_floors

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"price": self.prices, "export_price": self.export, "power_limit": self.power_limits}, index=self.index
        )


class Tariff(BaseModel, ABC):
    """Base class of the tariffs."""

    model_config = ConfigDict(frozen=True)

    capacity_charge: float = Field(0.0, ge=0, description="Price of the peak power of the horizon [EUR/kW]")

    @abstractmethod
    def energy_price(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Price of the energy consumed during each step [EUR/kWh]."""

    def power_limit(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.full(len(index), np.inf)

    def export_price(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Remuneration of the energy exported to the grid [EUR/kWh] (none by default)."""
        return np.zeros(len(index))

    def power_floor(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Minimum billed power of each step [W] (an upward activation request)."""
        return np.zeros(len(index))

    @property
    def has_export_price(self) -> bool:
        """Whether the tariff remunerates the export (otherwise it is sold at the energy price, net metering)."""
        return False

    def signal(self, index: pd.DatetimeIndex) -> TariffSignal:
        return TariffSignal(
            index=index,
            prices=np.asarray(self.energy_price(index), dtype=float),
            power_limits=np.asarray(self.power_limit(index), dtype=float),
            capacity_charge=self.capacity_charge,
            export_prices=np.asarray(self.export_price(index), dtype=float),
            power_floors=np.asarray(self.power_floor(index), dtype=float),
        )

    def cost(self, index: pd.DatetimeIndex, total_power: np.ndarray, time_step: float) -> float:
        """Energy cost [EUR] of a total heating power profile [W] (without the capacity charge)."""
        return float(np.sum(self.energy_price(index) * np.asarray(total_power) * time_step / 3.6e6))

    def with_events(self, events: list[DemandResponseEvent]) -> DemandResponseTariff:
        return DemandResponseTariff(base=self, events=events, capacity_charge=self.capacity_charge)


class FlatTariff(Tariff):
    """Single energy price."""

    kind: Literal["flat"] = "flat"
    price: float = Field(0.25, ge=0, description="[EUR/kWh]")

    def energy_price(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.full(len(index), self.price)


class TimeOfUseTariff(Tariff):
    """Peak / off-peak prices; the weekend is off-peak unless ``weekend_peak``."""

    kind: Literal["time_of_use"] = "time_of_use"
    peak_price: float = Field(0.32, ge=0, description="[EUR/kWh]")
    off_peak_price: float = Field(0.18, ge=0, description="[EUR/kWh]")
    peak_hours: tuple[float, float] = Field((7.0, 22.0), description="Start and end hour of the peak period")
    weekend_peak: bool = False

    @field_validator("peak_hours")
    @classmethod
    def _check_hours(cls, hours: tuple[float, float]) -> tuple[float, float]:
        if not 0 <= hours[0] < hours[1] <= HOURS_PER_DAY:
            raise ValueError("peak_hours must satisfy 0 <= start < end <= 24.")
        return hours

    def is_peak(self, index: pd.DatetimeIndex) -> np.ndarray:
        hours = index.hour + index.minute / 60
        peak = (hours >= self.peak_hours[0]) & (hours < self.peak_hours[1])
        if not self.weekend_peak:
            peak &= index.dayofweek < 5
        return np.asarray(peak, dtype=bool)

    def energy_price(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.where(self.is_peak(index), self.peak_price, self.off_peak_price)


def _to_series(value: Any) -> pd.Series:  # noqa: ANN401
    """Accept a series or a ``{timestamp: price}`` mapping (JSON)."""
    series = value if isinstance(value, pd.Series) else pd.Series(dict(value))
    series.index = pd.to_datetime(series.index)
    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("The price series needs a DatetimeIndex.")
    return series.sort_index().astype(float)


PriceSeries = Annotated[
    pd.Series,
    BeforeValidator(_to_series),
    PlainSerializer(
        lambda series: {str(stamp.isoformat()): float(price) for stamp, price in series.items()}, return_type=dict
    ),
]


class DynamicTariff(Tariff):
    """Time series of prices (e.g. day-ahead market prices plus grid fees), held between samples."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    kind: Literal["dynamic"] = "dynamic"
    series: PriceSeries = Field(description="Prices [EUR/kWh] indexed by time")
    export_series: PriceSeries | None = Field(None, description="Export remuneration [EUR/kWh] indexed by time")

    @classmethod
    def from_csv(cls, path: Path | str, column: str = "price", capacity_charge: float = 0.0) -> DynamicTariff:
        frame = pd.read_csv(path)
        frame[frame.columns[0]] = pd.to_datetime(frame[frame.columns[0]])
        series = frame.set_index(frame.columns[0])[column]
        return cls(series=series, capacity_charge=capacity_charge)

    @classmethod
    def synthetic_day_ahead(
        cls,
        start: pd.Timestamp | str,
        days: int,
        base: float = 0.22,
        amplitude: float = 0.10,
        noise: float = 0.02,
        seed: int | None = 0,
        capacity_charge: float = 0.0,
    ) -> DynamicTariff:
        """Hourly prices with a morning and an evening peak, a night trough and random variations."""
        index = pd.date_range(pd.Timestamp(start), periods=days * HOURS_PER_DAY, freq="h")
        hour = index.hour.to_numpy()
        profile = (
            np.exp(-((hour - 8) ** 2) / 8) + np.exp(-((hour - 19) ** 2) / 8) - 0.6 * np.exp(-((hour - 3) ** 2) / 12)
        )
        generator = np.random.default_rng(seed)
        prices = base + amplitude * profile + noise * generator.standard_normal(len(index))
        return cls(series=pd.Series(np.clip(prices, 0.01, None), index=index), capacity_charge=capacity_charge)

    def energy_price(self, index: pd.DatetimeIndex) -> np.ndarray:
        return _held(self.series, index)

    def export_price(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.zeros(len(index)) if self.export_series is None else _held(self.export_series, index)

    @property
    def has_export_price(self) -> bool:
        return self.export_series is not None


def _held(series: pd.Series, index: pd.DatetimeIndex) -> np.ndarray:
    """Values of a series held between its samples, evaluated on ``index``."""
    if index.min() < series.index.min() or index.max() > series.index.max() + pd.Timedelta(days=1):
        raise ValueError(f"The price series does not cover {index.min()} .. {index.max()}.")
    return series.reindex(series.index.union(index)).ffill().loc[index].to_numpy(dtype=float)


class DemandResponseEvent(BaseModel):
    """A period where the grid asks for flexibility: a price multiplier, a power cap or a power floor."""

    model_config = ConfigDict(frozen=True)

    start: datetime
    end: datetime
    price_multiplier: float = Field(1.0, ge=0, description="Multiplies the base price during the event")
    power_limit: float | None = Field(None, ge=0, description="Maximum billed power during the event [W]")
    power_floor: float | None = Field(None, ge=0, description="Minimum billed power during the event [W]")
    notice_hours: float | None = Field(
        None,
        ge=0,
        description="Hours before the start from which the controller knows the cap or floor "
        "(None: over its whole horizon); the price multiplier is always known",
    )

    def active(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray((index >= pd.Timestamp(self.start)) & (index < pd.Timestamp(self.end)), dtype=bool)

    def known(self, index: pd.DatetimeIndex) -> bool:
        """Whether a controller planning from ``index[0]`` has been notified of the cap or floor."""
        if self.notice_hours is None or len(index) == 0:
            return True
        return bool(pd.Timestamp(index[0]) >= pd.Timestamp(self.start) - pd.Timedelta(hours=self.notice_hours))


class DemandResponseTariff(Tariff):
    """A base tariff with demand response events on top."""

    kind: Literal["demand_response"] = "demand_response"
    base: Annotated[FlatTariff | TimeOfUseTariff | DynamicTariff, Field(discriminator="kind")]
    events: list[DemandResponseEvent] = Field(default_factory=list)

    def energy_price(self, index: pd.DatetimeIndex) -> np.ndarray:
        prices = self.base.energy_price(index).astype(float)
        for event in self.events:
            prices = np.where(event.active(index), prices * event.price_multiplier, prices)
        return prices

    def power_limit(self, index: pd.DatetimeIndex) -> np.ndarray:
        limits = self.base.power_limit(index).astype(float)
        for event in self.events:
            if event.power_limit is not None and event.known(index):
                limits = np.where(event.active(index), np.minimum(limits, event.power_limit), limits)
        return limits

    def power_floor(self, index: pd.DatetimeIndex) -> np.ndarray:
        floors = self.base.power_floor(index).astype(float)
        for event in self.events:
            if event.power_floor is not None and event.known(index):
                floors = np.where(event.active(index), np.maximum(floors, event.power_floor), floors)
        return floors

    def export_price(self, index: pd.DatetimeIndex) -> np.ndarray:
        return self.base.export_price(index)

    @property
    def has_export_price(self) -> bool:
        return self.base.has_export_price


MONTHS_PER_YEAR = 12


class CapacityTariff(Tariff):
    """A base tariff plus a capacity charge on the monthly peak, as the Flemish ``capaciteitstarief``.

    The distribution tariff bills ``capacity_price`` [EUR/kW/year] on the average of the last twelve
    monthly peaks, each the highest quarter-hour average power of the month floored at
    ``minimum_peak_kw`` (Vlaamse Nutsregulator, in force since 2023; 2025 levels between 48.9 and
    56.9 EUR/kW/year depending on the area, see ``tutorials/references.md``). Raising the peak of
    one month by one kilowatt therefore costs ``capacity_price / 12``, which is the
    ``capacity_charge`` the controller applies to the peak of the month so far and of its horizon
    (``running_peak``).
    """

    kind: Literal["capacity"] = "capacity"
    base: Annotated[FlatTariff | TimeOfUseTariff | DynamicTariff | DemandResponseTariff, Field(discriminator="kind")]
    capacity_price: float = Field(50.0, ge=0, description="Price of the average monthly peak [EUR/kW/year]")
    minimum_peak_kw: float = Field(2.5, ge=0, description="Floor of each monthly peak [kW]")
    running_peak: bool = Field(True, description="The controller carries the peak of the month so far")

    @model_validator(mode="after")
    def _monthly_charge(self) -> CapacityTariff:
        object.__setattr__(self, "capacity_charge", self.capacity_price / MONTHS_PER_YEAR)
        return self

    def energy_price(self, index: pd.DatetimeIndex) -> np.ndarray:
        return self.base.energy_price(index)

    def power_limit(self, index: pd.DatetimeIndex) -> np.ndarray:
        return self.base.power_limit(index)

    def power_floor(self, index: pd.DatetimeIndex) -> np.ndarray:
        return self.base.power_floor(index)

    def export_price(self, index: pd.DatetimeIndex) -> np.ndarray:
        return self.base.export_price(index)

    @property
    def has_export_price(self) -> bool:
        return self.base.has_export_price

    def with_events(self, events: list[DemandResponseEvent]) -> CapacityTariff:  # type: ignore[override]
        base = (
            self.base.with_events(events)
            if not isinstance(self.base, DemandResponseTariff)
            else (self.base.model_copy(update={"events": [*self.base.events, *events]}))
        )
        return self.model_copy(update={"base": base})

    def monthly_peaks(self, index: pd.DatetimeIndex, power: np.ndarray) -> pd.Series:
        """Highest average power of each month [kW] (at the resolution of ``index``), floored."""
        series = pd.Series(np.asarray(power, dtype=float) / 1000, index=index)
        peaks = series.groupby(index.to_period("M")).max()
        return peaks.clip(lower=self.minimum_peak_kw)

    def capacity_cost(self, index: pd.DatetimeIndex, power: np.ndarray) -> float:
        """Capacity charge [EUR] of a power profile: the monthly peaks at the monthly price."""
        return float(self.monthly_peaks(index, power).sum() * self.capacity_price / MONTHS_PER_YEAR)


AnyTariff = Annotated[
    FlatTariff | TimeOfUseTariff | DynamicTariff | DemandResponseTariff | CapacityTariff, Field(discriminator="kind")
]
