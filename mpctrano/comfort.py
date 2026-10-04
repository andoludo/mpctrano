"""Comfort requirements: time-varying temperature bands per zone.

The MPC keeps the indoor temperature of each zone inside ``[lower, upper]``; the band is a soft
constraint, violations are penalised in the objective. Widening the band during unoccupied
periods (setback) is what gives the building its flexibility towards the tariff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from mpctrano.model import KELVIN

if TYPE_CHECKING:
    from mpctrano.model import RCModel


@dataclass(frozen=True)
class ComfortBounds:
    """Lower and upper temperature bounds ``(n_zones, N)`` [K] on a time index."""

    index: pd.DatetimeIndex
    zones: list[str]
    lower: np.ndarray
    upper: np.ndarray

    def frame(self) -> pd.DataFrame:
        columns = {f"{zone}_lower": self.lower[i] for i, zone in enumerate(self.zones)}
        columns |= {f"{zone}_upper": self.upper[i] for i, zone in enumerate(self.zones)}
        return pd.DataFrame(columns, index=self.index)


class ComfortSchedule(BaseModel):
    """Occupied / unoccupied temperature band of a zone, in degrees Celsius."""

    model_config = ConfigDict(frozen=True)

    occupied_lower: float = Field(21.0, description="[degC]")
    occupied_upper: float = Field(24.0, description="[degC]")
    unoccupied_lower: float = Field(17.0, description="[degC]")
    unoccupied_upper: float = Field(27.0, description="[degC]")
    occupied_hours: tuple[float, float] = Field((7.0, 22.0), description="Start and end hour of the occupied period")
    weekend_occupied: bool = True

    @model_validator(mode="after")
    def _check(self) -> ComfortSchedule:
        if self.occupied_lower > self.occupied_upper or self.unoccupied_lower > self.unoccupied_upper:
            raise ValueError("Lower bounds must not exceed upper bounds.")
        start, end = self.occupied_hours
        if not 0 <= start < end <= 24:
            raise ValueError("occupied_hours must satisfy 0 <= start < end <= 24.")
        return self

    def occupied(self, index: pd.DatetimeIndex) -> np.ndarray:
        hours = index.hour + index.minute / 60
        occupied = (hours >= self.occupied_hours[0]) & (hours < self.occupied_hours[1])
        if not self.weekend_occupied:
            occupied &= index.dayofweek < 5
        return np.asarray(occupied, dtype=bool)

    def bounds(self, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """Lower and upper bounds [K] on the index."""
        occupied = self.occupied(index)
        lower = np.where(occupied, self.occupied_lower, self.unoccupied_lower) + KELVIN
        upper = np.where(occupied, self.occupied_upper, self.unoccupied_upper) + KELVIN
        return lower, upper

    def widen(self, delta: float) -> ComfortSchedule:
        """Schedule with a band wider by ``delta`` K on each side (more flexibility)."""
        return self.model_copy(
            update={
                "occupied_lower": self.occupied_lower - delta,
                "occupied_upper": self.occupied_upper + delta,
                "unoccupied_lower": self.unoccupied_lower - delta,
                "unoccupied_upper": self.unoccupied_upper + delta,
            }
        )

    def constant(self) -> ComfortSchedule:
        """Same band all day (no setback)."""
        return self.model_copy(
            update={"unoccupied_lower": self.occupied_lower, "unoccupied_upper": self.occupied_upper}
        )


class ComfortSpec(BaseModel):
    """Comfort schedule of every zone (``default`` unless the zone is listed in ``zones``)."""

    model_config = ConfigDict(frozen=True)

    default: ComfortSchedule = Field(default_factory=ComfortSchedule)
    zones: dict[str, ComfortSchedule] = Field(default_factory=dict)

    def schedule(self, zone: str) -> ComfortSchedule:
        return self.zones.get(zone, self.default)

    def bounds(self, model: RCModel, index: pd.DatetimeIndex) -> ComfortBounds:
        lower, upper = [], []
        for zone in model.zone_names:
            zone_lower, zone_upper = self.schedule(zone).bounds(index)
            lower.append(zone_lower)
            upper.append(zone_upper)
        return ComfortBounds(index=index, zones=list(model.zone_names), lower=np.vstack(lower), upper=np.vstack(upper))

    def map(self, transform: str, *args: float) -> ComfortSpec:
        """Apply ``ComfortSchedule.<transform>(*args)`` to every schedule (e.g. ``widen``)."""
        default = getattr(self.default, transform)(*args)
        zones = {name: getattr(schedule, transform)(*args) for name, schedule in self.zones.items()}
        return ComfortSpec(default=default, zones=zones)
