"""Electrical storage next to the building: a stationary battery and an electric vehicle at the meter.

Trano's FastSmoothBuildings describes the building and its thermal systems; the electrical storages are optimised here,
in the balance of the meter: a :class:`Battery` charges from the grid or the PV and discharges into the house, an
:class:`ElectricVehicle` charges while it is plugged in and must hold its departure energy when it leaves. Both
are linear state-of-charge models (a charging and a discharging efficiency, power bounds), exact in the closed
loop: the controller carries their state (:attr:`~mpctrano.mpc.ModelPredictiveController.storage_state`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

HOURS_PER_DAY = 24.0


class Battery(BaseModel):
    """A stationary battery behind the meter."""

    model_config = ConfigDict(frozen=True)

    name: str = "battery"
    capacity_kwh: float = Field(gt=0)
    max_charge_w: float = Field(gt=0, description="Charging power [W]")
    max_discharge_w: float = Field(gt=0, description="Discharging power [W]")
    efficiency: float = Field(0.92, gt=0, le=1, description="Round-trip efficiency (split evenly between the ways)")
    soc_min: float = Field(0.1, ge=0, le=1, description="Lowest state of charge kept [-]")
    soc_max: float = Field(1.0, ge=0, le=1, description="Highest state of charge [-]")
    initial_soc: float = Field(0.5, ge=0, le=1, description="State of charge at the start of a run [-]")
    driven_kwh: float = Field(0.0, ge=0, description="Energy leaving the storage at each departure [kWh] (a vehicle)")
    departure_energy_kwh: float = Field(0.0, ge=0, description="Energy required at a departure [kWh] (a vehicle)")

    @model_validator(mode="after")
    def _ordered(self) -> Battery:
        if not self.soc_min <= self.initial_soc <= self.soc_max:
            raise ValueError("initial_soc must lie between soc_min and soc_max.")
        return self

    @property
    def one_way_efficiency(self) -> float:
        return float(np.sqrt(self.efficiency))

    @property
    def energy_bounds_kwh(self) -> tuple[float, float]:
        return self.soc_min * self.capacity_kwh, self.soc_max * self.capacity_kwh

    @property
    def initial_energy_kwh(self) -> float:
        return self.initial_soc * self.capacity_kwh

    def plugged(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.ones(len(index))

    def departures(self, index: pd.DatetimeIndex, time_step: float) -> np.ndarray:
        return np.zeros(len(index))


class ElectricVehicle(Battery):
    """An electric vehicle charged at home: plugged in from ``arrival_hour`` to ``departure_hour`` (overnight when
    the departure comes first), it leaves with at least ``departure_energy_kwh`` (soft, at ``missing_energy_price``)
    and comes back ``driven_kwh`` lower. It does not discharge into the house (``max_discharge_w`` is ignored)."""

    name: str = "vehicle"
    max_discharge_w: float = Field(1.0, gt=0, description="Ignored: a vehicle does not feed the house")
    arrival_hour: float = Field(18.0, ge=0, lt=24)
    departure_hour: float = Field(7.0, ge=0, lt=24)
    driven_kwh: float = Field(12.0, ge=0, description="Energy of a day of driving [kWh]")
    departure_energy_kwh: float = Field(30.0, ge=0, description="Energy the vehicle must hold at departure [kWh]")
    missing_energy_price: float = Field(2.0, ge=0, description="Penalty of the energy missing at departure [EUR/kWh]")
    weekdays_only: bool = Field(False, description="The vehicle stays plugged in at the weekend")

    @model_validator(mode="after")
    def _fits(self) -> ElectricVehicle:
        if self.departure_energy_kwh > self.capacity_kwh:
            raise ValueError("departure_energy_kwh exceeds the capacity.")
        if self.driven_kwh > self.departure_energy_kwh:
            raise ValueError("driven_kwh exceeds the departure energy: the vehicle would come back empty.")
        return self

    def _away_day(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Days the vehicle leaves (every day, or the weekdays)."""
        return np.asarray(index.dayofweek < 5, dtype=bool) if self.weekdays_only else np.ones(len(index), dtype=bool)

    def plugged(self, index: pd.DatetimeIndex) -> np.ndarray:
        """1 while the vehicle is at home (plugged in) at the start of each step."""
        hour = np.asarray(index.hour + index.minute / 60, dtype=float)
        if self.arrival_hour > self.departure_hour:  # home overnight
            away = (hour >= self.departure_hour) & (hour < self.arrival_hour)
        else:  # home during the day
            away = (hour >= self.departure_hour) | (hour < self.arrival_hour)
        return np.where(away & self._away_day(index), 0.0, 1.0)

    def departures(self, index: pd.DatetimeIndex, time_step: float) -> np.ndarray:
        """1 on the step during which the vehicle leaves (the first step starting at or after the departure)."""
        hour = np.asarray(index.hour + index.minute / 60, dtype=float)
        step_hours = time_step / 3600
        leaves = (hour >= self.departure_hour) & (hour < self.departure_hour + step_hours) & self._away_day(index)
        return np.asarray(leaves, dtype=float)


def storage_schedule(
    storages: tuple[Battery, ...], index: pd.DatetimeIndex, time_step: float
) -> tuple[np.ndarray, np.ndarray]:
    """``plugged`` and ``departures`` indicators of each storage over ``index``, ``(n_storages, N)``."""
    if not storages:
        return np.zeros((0, len(index))), np.zeros((0, len(index)))
    plugged = np.vstack([storage.plugged(index) for storage in storages])
    departures = np.vstack([storage.departures(index, time_step) for storage in storages])
    return plugged, departures


__all__ = ["Battery", "ElectricVehicle", "storage_schedule"]
