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
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

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

    @property
    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"price": self.prices, "power_limit": self.power_limits}, index=self.index)


class Tariff(BaseModel, ABC):
    """Base class of the tariffs."""

    model_config = ConfigDict(frozen=True)

    capacity_charge: float = Field(0.0, ge=0, description="Price of the peak power of the horizon [EUR/kW]")

    @abstractmethod
    def energy_price(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Price of the energy consumed during each step [EUR/kWh]."""

    def power_limit(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.full(len(index), np.inf)

    def signal(self, index: pd.DatetimeIndex) -> TariffSignal:
        return TariffSignal(
            index=index,
            prices=np.asarray(self.energy_price(index), dtype=float),
            power_limits=np.asarray(self.power_limit(index), dtype=float),
            capacity_charge=self.capacity_charge,
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


class DynamicTariff(Tariff):
    """Time series of prices (e.g. day-ahead market prices plus grid fees), held between samples."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    kind: Literal["dynamic"] = "dynamic"
    series: Annotated[pd.Series, Field(description="Prices [EUR/kWh] indexed by time")]

    @field_validator("series")
    @classmethod
    def _check_series(cls, series: pd.Series) -> pd.Series:
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("The price series needs a DatetimeIndex.")
        return series.sort_index().astype(float)

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
        if index.min() < self.series.index.min() or index.max() > self.series.index.max() + pd.Timedelta(days=1):
            raise ValueError(f"The price series does not cover {index.min()} .. {index.max()}.")
        return self.series.reindex(self.series.index.union(index)).ffill().loc[index].to_numpy(dtype=float)


class DemandResponseEvent(BaseModel):
    """A period where the grid asks for flexibility: a price multiplier and/or a power cap."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    start: pd.Timestamp
    end: pd.Timestamp
    price_multiplier: float = Field(1.0, ge=0, description="Multiplies the base price during the event")
    power_limit: float | None = Field(None, ge=0, description="Maximum total heating power during the event [W]")

    def active(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray((index >= self.start) & (index < self.end), dtype=bool)


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
            if event.power_limit is not None:
                limits = np.where(event.active(index), np.minimum(limits, event.power_limit), limits)
        return limits


AnyTariff = Annotated[FlatTariff | TimeOfUseTariff | DynamicTariff | DemandResponseTariff, Field(discriminator="kind")]
