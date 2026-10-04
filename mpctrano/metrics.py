"""Performance indicators: fit of a model to data and comfort / cost of a control run."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


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


class ControlMetrics(BaseModel):
    """Energy, cost, comfort and peak of a (closed-loop) control run."""

    energy_kwh: float = Field(description="Total heating energy [kWh]")
    energy_cost: float = Field(description="Energy cost at the tariff [EUR]")
    capacity_cost: float = Field(0.0, description="Capacity charge of the peak power [EUR]")
    comfort_violation_kh: float = Field(description="Integral of the band violations over the zones [K.h]")
    max_violation: float = Field(description="Largest band violation [K]")
    peak_power_kw: float = Field(description="Peak total heating power [kW]")
    low_price_share: float = Field(description="Share of the energy consumed below the median price")
    mean_price: float = Field(description="Energy-weighted average price paid [EUR/kWh]")
    steps: int

    @property
    def total_cost(self) -> float:
        return self.energy_cost + self.capacity_cost

    @classmethod
    def compute(
        cls,
        index: pd.DatetimeIndex,
        powers: np.ndarray,
        temperatures: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        prices: np.ndarray,
        time_step: float,
        capacity_charge: float = 0.0,
    ) -> ControlMetrics:
        """``powers`` and ``temperatures`` are ``(n_zones, N)``, the temperatures at the end of each step."""
        hours = time_step / 3600
        total = np.sum(np.atleast_2d(powers), axis=0)
        energy = total * hours / 1000
        violation = np.maximum(lower - temperatures, 0) + np.maximum(temperatures - upper, 0)
        median = float(np.median(prices))
        total_energy = float(energy.sum())
        return cls(
            energy_kwh=total_energy,
            energy_cost=float(np.sum(prices * energy)),
            capacity_cost=float(capacity_charge * total.max() / 1000) if len(total) else 0.0,
            comfort_violation_kh=float(np.sum(violation) * hours),
            max_violation=float(np.max(violation)) if violation.size else 0.0,
            peak_power_kw=float(total.max() / 1000) if len(total) else 0.0,
            low_price_share=float(energy[prices < median].sum() / total_energy) if total_energy > 0 else 0.0,
            mean_price=float(np.sum(prices * energy) / total_energy) if total_energy > 0 else float("nan"),
            steps=len(index),
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"value": self.model_dump()})
