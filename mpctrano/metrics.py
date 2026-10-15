"""Performance indicators: fit of a model to data and energy, cost and comfort of a control run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from mpctrano.tariff import Tariff


class FitMetrics(BaseModel):
    """Goodness of fit of simulated zone temperatures against measurements."""

    rmse: dict[str, float] = Field(description="Root mean square error per zone [K]")
    mae: dict[str, float] = Field(description="Mean absolute error per zone [K]")
    max_error: dict[str, float] = Field(description="Maximum absolute error per zone [K]")
    r2: dict[str, float] = Field(description="Coefficient of determination per zone")
    samples: int

    @property
    def rmse_mean(self) -> float:
        return float(np.mean(list(self.rmse.values())))

    @property
    def mae_mean(self) -> float:
        return float(np.mean(list(self.mae.values())))

    @classmethod
    def compute(cls, measured: np.ndarray, simulated: np.ndarray, names: list[str]) -> FitMetrics:
        measured = np.atleast_2d(measured)
        simulated = np.atleast_2d(simulated)
        error = simulated - measured
        rmse, mae, max_error, r2 = {}, {}, {}, {}
        for i, name in enumerate(names):
            rmse[name] = float(np.sqrt(np.mean(error[i] ** 2)))
            mae[name] = float(np.mean(np.abs(error[i])))
            max_error[name] = float(np.max(np.abs(error[i])))
            variance = float(np.sum((measured[i] - measured[i].mean()) ** 2))
            r2[name] = float(1 - np.sum(error[i] ** 2) / variance) if variance > 0 else float("nan")
        return cls(rmse=rmse, mae=mae, max_error=max_error, r2=r2, samples=int(measured.shape[1]))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"rmse": self.rmse, "mae": self.mae, "max_error": self.max_error, "r2": self.r2})


@dataclass(frozen=True)
class Prices:
    """Prices of the carriers over the steps of a run [EUR/kWh] and the capacity charge [EUR/kW]."""

    electricity: np.ndarray
    feed_in: np.ndarray
    fuel: np.ndarray
    capacity_charge: float = 0.0

    @classmethod
    def of(
        cls,
        tariff: Tariff,
        index: pd.DatetimeIndex,
        fuel_price: float | None = None,
        feed_in_price: float | None = None,
    ) -> Prices:
        """The prices of ``tariff`` on ``index``; the fuel and the electricity sold at the tariff price unless
        given."""
        signal = tariff.signal(index)
        steps = len(index)

        def price(value: float | None) -> np.ndarray:
            return signal.prices.copy() if value is None else np.full(steps, float(value))

        return cls(
            electricity=signal.prices,
            feed_in=price(feed_in_price),
            fuel=price(fuel_price),
            capacity_charge=signal.capacity_charge,
        )


class ControlMetrics(BaseModel):
    """Energy, cost, comfort and peak of a run, from the meters of the building."""

    fuel_kwh: float = Field(description="Fuel drawn [kWh]")
    electricity_kwh: float = Field(description="Electricity bought from the grid [kWh]")
    sold_kwh: float = Field(description="Electricity sold to the grid [kWh]")
    energy_kwh: float = Field(description="Energy bought: fuel and electricity [kWh]")
    energy_cost: float = Field(description="Cost of the fuel and of the electricity bought, less the sales [EUR]")
    capacity_cost: float = Field(0.0, description="Capacity charge of the peak electric power [EUR]")
    comfort_violation_kh: float = Field(description="Integral of the band violations over the zones [K.h]")
    max_violation: float = Field(description="Largest band violation [K]")
    peak_power_kw: float = Field(description="Peak electric power bought [kW]")
    peak_fuel_power_kw: float = Field(description="Peak fuel power [kW]")
    low_price_share: float = Field(description="Share of the energy bought below the median electricity price")
    mean_price: float = Field(description="Average price of the energy bought [EUR/kWh]")
    steps: int

    @property
    def total_cost(self) -> float:
        return self.energy_cost + self.capacity_cost

    @classmethod
    def compute(
        cls,
        energy: pd.DataFrame,
        prices: Prices,
        temperatures: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        time_step: float,
    ) -> ControlMetrics:
        """``energy`` has the ``fuel``, ``bought`` and ``sold`` energy of each step [kWh]; ``temperatures``,
        ``lower`` and ``upper`` are ``(n_zones, N)``, the temperatures at the end of each step."""
        hours = time_step / 3600
        fuel, bought, sold = (energy[name].to_numpy(dtype=float) for name in ("fuel", "bought", "sold"))
        purchased = fuel + bought
        cost_of_energy = prices.fuel * fuel + prices.electricity * bought - prices.feed_in * sold
        violation = np.maximum(lower - temperatures, 0) + np.maximum(temperatures - upper, 0)
        total = float(purchased.sum())
        steps = len(energy)
        cheap = prices.electricity < float(np.median(prices.electricity)) if steps else np.zeros(0, dtype=bool)
        peak = float(bought.max() / hours) if steps else 0.0
        return cls(
            fuel_kwh=float(fuel.sum()),
            electricity_kwh=float(bought.sum()),
            sold_kwh=float(sold.sum()),
            energy_kwh=total,
            energy_cost=float(cost_of_energy.sum()),
            capacity_cost=prices.capacity_charge * peak,
            comfort_violation_kh=float(np.sum(violation) * hours),
            max_violation=float(np.max(violation)) if violation.size else 0.0,
            peak_power_kw=peak,
            peak_fuel_power_kw=float(fuel.max() / hours) if steps else 0.0,
            low_price_share=float(purchased[cheap].sum() / total) if total > 0 else 0.0,
            mean_price=float((prices.fuel * fuel + prices.electricity * bought).sum() / total)
            if total > 0
            else float("nan"),
            steps=steps,
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"value": self.model_dump()})
