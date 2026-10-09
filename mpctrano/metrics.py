"""Performance indicators: fit of a model to data and comfort / cost of a control run."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from mpctrano.model import RCModel


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
    """Energy, cost, comfort and peak of a (closed-loop) control run.

    ``energy_kwh`` is the energy billed at the tariff: the ideal heating of the zones without
    system plus the electricity imported from the grid; the export, the gas and the PV
    generation are reported separately.
    """

    energy_kwh: float = Field(description="Energy billed at the tariff (ideal heating + grid import) [kWh]")
    energy_cost: float = Field(description="Energy cost at the tariff [EUR]")
    capacity_cost: float = Field(0.0, description="Capacity charge of the peak power [EUR]")
    comfort_violation_kh: float = Field(description="Integral of the band violations over the zones [K.h]")
    max_violation: float = Field(description="Largest band violation [K]")
    peak_power_kw: float = Field(description="Peak billed power [kW]")
    low_price_share: float = Field(description="Share of the energy consumed below the median price")
    mean_price: float = Field(description="Energy-weighted average price paid [EUR/kWh]")
    steps: int
    export_kwh: float = Field(0.0, description="Energy exported to the grid [kWh]")
    export_revenue: float = Field(0.0, description="Revenue of the exported energy [EUR]")
    gas_kwh: float = Field(0.0, description="Fuel energy of the boilers [kWh]")
    gas_cost: float = Field(0.0, description="Cost of the fuel [EUR]")
    generation_kwh: float = Field(0.0, description="PV generation [kWh]")
    self_consumption: float = Field(
        float("nan"), description="Share of the PV generation consumed on site (NaN without PV)"
    )

    @property
    def total_cost(self) -> float:
        return self.energy_cost + self.capacity_cost + self.gas_cost - self.export_revenue

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
        export: np.ndarray | None = None,
        export_prices: np.ndarray | None = None,
        gas: np.ndarray | None = None,
        gas_price: float = 0.0,
        generation: np.ndarray | None = None,
    ) -> ControlMetrics:
        """``powers`` are the billed powers ``(n, N)`` [W] (summed), ``temperatures`` the zone temperatures
        ``(n_zones, N)`` at the end of each step; ``export``, ``gas`` and ``generation`` are ``(N,)`` [W]."""
        hours = time_step / 3600
        total = np.sum(np.atleast_2d(powers), axis=0)
        energy = total * hours / 1000
        violation = np.maximum(lower - temperatures, 0) + np.maximum(temperatures - upper, 0)
        median = float(np.median(prices))
        total_energy = float(energy.sum())
        exported = np.zeros_like(total) if export is None else np.asarray(export, dtype=float) * hours / 1000
        export_prices = np.zeros_like(prices) if export_prices is None else np.asarray(export_prices, dtype=float)
        gas_energy = np.zeros_like(total) if gas is None else np.asarray(gas, dtype=float) * hours / 1000
        generated = None if generation is None else np.asarray(generation, dtype=float) * hours / 1000
        self_consumption = float("nan")
        if generated is not None and generated.sum() > 0:
            self_consumption = float(1 - exported.sum() / generated.sum())
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
            export_kwh=float(exported.sum()),
            export_revenue=float(np.sum(export_prices * exported)),
            gas_kwh=float(gas_energy.sum()),
            gas_cost=float(gas_price * gas_energy.sum()),
            generation_kwh=0.0 if generated is None else float(generated.sum()),
            self_consumption=self_consumption,
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"value": self.model_dump()})


def compressor_load(frame: pd.DataFrame, model: RCModel, margin: float = 0.02) -> pd.DataFrame:
    """Use of the shared compressors of a closed-loop or open-loop frame, one row per control group.

    ``mean_load`` is the average of the summed electrical inputs of the group over its limit,
    ``binding_share`` the share of steps within ``margin`` of the limit (where the compressor,
    not the price, decides the dispatch), ``peak_kw`` the highest summed input; the share of the
    group's electricity drawn by each input follows (``share_<input>``).
    """
    rows = {}
    for group in model.control_groups:
        names = [model.control_names[i] for i in group.indices]
        total = frame[names].sum(axis=1)
        row: dict[str, float] = {
            "limit_kw": group.max_power / 1000,
            "peak_kw": float(total.max()) / 1000,
            "mean_load": float(total.mean()) / group.max_power,
            "binding_share": float((total >= (1 - margin) * group.max_power).mean()),
        }
        energy = frame[names].sum()
        row |= {f"share_{name}": float(energy[name] / energy.sum()) if energy.sum() > 0 else 0.0 for name in names}
        rows[group.system] = row
    return pd.DataFrame(rows).T
