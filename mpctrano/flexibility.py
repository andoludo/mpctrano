"""Quantify the flexibility a building offers to a tariff: compare closed-loop runs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from mpctrano.simulation import ClosedLoopResult
from mpctrano.tariff import DemandResponseTariff, FlatTariff, Tariff

if TYPE_CHECKING:
    from mpctrano.forecast import Forecaster
    from mpctrano.model import FastModel
    from mpctrano.scenarios import Scenario
    from mpctrano.simulation import Plant


def compare(results: Mapping[str, ClosedLoopResult], baseline: str | None = None) -> pd.DataFrame:
    """One row per run: energy, cost, comfort, peak and savings relative to the baseline run."""
    frame = pd.DataFrame({name: result.summary() for name, result in results.items()}).T
    baseline = baseline or next(iter(results))
    cost = float(results[baseline].metrics.total_cost)
    energy = float(results[baseline].metrics.energy_kwh)
    frame["cost_saving_pct"] = 100 * (cost - frame["total_cost"]) / cost if cost else np.nan
    frame["energy_change_pct"] = 100 * (frame["energy_kwh"] - energy) / energy if energy else np.nan
    frame["shifted_energy_kwh"] = [shifted_energy(results[name], results[baseline]) for name in frame.index]
    return frame


def value_of_flexibility(
    model: FastModel,
    scenario: Scenario,
    plant: FastModel | Plant | None = None,
    forecaster: Forecaster | None = None,
) -> pd.DataFrame:
    """Savings of a price-aware MPC against a price-agnostic one under the same tariff.

    The price-agnostic controller minimises the cost at a flat price (no capacity charge) with the
    same comfort band, model and demand response power limits; its run is then costed with the
    tariff of the scenario. The difference is what the flexibility of the building is worth under
    that tariff. Evaluate over several days: the heat stored at the end of the period is not paid
    back within it (end effect of the receding horizon).
    """
    from mpctrano.scenarios import run_scenario

    aware = run_scenario(model, scenario, plant=plant, forecaster=forecaster)
    index = pd.DatetimeIndex(aware.frame.index)
    agnostic_scenario = scenario.model_copy(
        update={"name": f"{scenario.name} (price agnostic)", "tariff": price_agnostic_tariff(scenario.tariff, index)}
    )
    agnostic = run_scenario(model, agnostic_scenario, plant=plant, forecaster=forecaster)
    agnostic_metrics = agnostic.metrics_under(scenario.tariff)
    rows = {
        "price aware": {**aware.metrics.model_dump(), "total_cost": aware.metrics.total_cost},
        "price agnostic": {**agnostic_metrics.model_dump(), "total_cost": agnostic_metrics.total_cost},
    }
    saving = agnostic_metrics.total_cost - aware.metrics.total_cost
    rows["price aware"] |= {
        "cost_saving": saving,
        "cost_saving_pct": 100 * saving / agnostic_metrics.total_cost if agnostic_metrics.total_cost else 0.0,
        "shifted_energy_kwh": shifted_energy(aware, agnostic),
    }
    rows["price agnostic"] |= {"cost_saving": 0.0, "cost_saving_pct": 0.0, "shifted_energy_kwh": 0.0}
    return pd.DataFrame.from_dict(rows, orient="index")


def price_agnostic_tariff(tariff: Tariff, index: pd.DatetimeIndex) -> Tariff:
    """Flat tariff at the mean price of ``tariff`` over ``index``, keeping its power limits (grid constraints)."""
    flat = FlatTariff(price=float(np.mean(tariff.energy_price(index))))
    if isinstance(tariff, DemandResponseTariff):
        events = [event.model_copy(update={"price_multiplier": 1.0}) for event in tariff.events if event.power_limit]
        return flat.with_events(events) if events else flat
    return flat


def shifted_energy(result: ClosedLoopResult, reference: ClosedLoopResult) -> float:
    """Energy [kWh] moved in time with respect to the reference run (half of the L1 distance)."""
    profile = _energy_profile(result)
    other = _energy_profile(reference)
    index = profile.index.intersection(other.index)
    return float(0.5 * np.abs(profile.loc[index] - other.loc[index]).sum())


def _energy_profile(result: ClosedLoopResult) -> pd.Series:
    """Energy bought at each step, fuel and electricity [kWh], from the meters of the plant."""
    return result.energy


def price_response(result: ClosedLoopResult) -> pd.DataFrame:
    """Energy bought per price level: shows whether the heating follows the tariff."""
    energy = _energy_profile(result)
    frame = pd.DataFrame({"price": result.frame["price"], "energy_kwh": energy})
    grouped = frame.groupby("price")["energy_kwh"].agg(["sum", "count"])
    grouped["share"] = grouped["sum"] / grouped["sum"].sum()
    return grouped
