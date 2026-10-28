"""Ready-made control scenarios: tariffs, comfort and forecasts combined into named studies."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mpctrano.comfort import ComfortSchedule, ComfortSpec
from mpctrano.estimation import MovingHorizonEstimator, PerfectEstimator, StateEstimator
from mpctrano.forecast import ForecastNoise, Forecaster, WeatherForecaster
from mpctrano.model import BuildingModel
from mpctrano.mpc import ModelPredictiveController, MPCSettings
from mpctrano.simulation import ClosedLoopResult, ModelPlant, Plant, run_closed_loop, run_rule_based
from mpctrano.tariff import AnyTariff, DemandResponseEvent, DynamicTariff, FlatTariff, TimeOfUseTariff

DEFAULT_START = "2023-01-16"
"""A Monday in the middle of the heating season of the typical year."""


class Scenario(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    description: str
    tariff: AnyTariff
    comfort: ComfortSpec = Field(default_factory=ComfortSpec)
    settings: MPCSettings = Field(default_factory=MPCSettings)
    start: str = DEFAULT_START
    days: float = Field(2.0, gt=0)
    forecast_noise: ForecastNoise | None = None
    estimator: Literal["mhe", "perfect"] = Field(
        "mhe", description="Moving horizon estimation from the air temperatures, or the true state of the plant"
    )

    @property
    def steps(self) -> int:
        return round(self.days * 86400 / self.settings.time_step)

    def with_days(self, days: float) -> Scenario:
        return self.model_copy(update={"days": days})


def scenario_catalogue(start: str = DEFAULT_START, days: float = 2.0, horizon: int = 24) -> dict[str, Scenario]:
    """The scenarios shipped with the package (``flat`` is the baseline for the others)."""
    settings = MPCSettings(horizon=horizon)
    timestamp = pd.Timestamp(start)
    comfort = ComfortSpec()
    deep_setback = ComfortSpec(default=ComfortSchedule(unoccupied_lower=15.0))
    # Contrasted but realistic retail prices: the slow envelope of a building only pays back part of the
    # stored heat within the expensive hours, so a spread of a few times is needed to see it move.
    time_of_use = TimeOfUseTariff(peak_price=0.40, off_peak_price=0.10)
    dynamic = DynamicTariff.synthetic_day_ahead(
        timestamp - pd.Timedelta(days=1), int(days) + 3, base=0.18, amplitude=0.18, noise=0.02
    )
    event = DemandResponseEvent(
        start=timestamp + pd.Timedelta(hours=17),
        end=timestamp + pd.Timedelta(hours=20),
        price_multiplier=3.0,
        power_limit=800.0,
    )
    scenarios = [
        Scenario(
            name="flat",
            description="Flat tariff: the baseline, the MPC only minimises the energy within the comfort band.",
            tariff=FlatTariff(price=0.25),
            comfort=comfort,
        ),
        Scenario(
            name="time_of_use",
            description="Peak / off-peak tariff (0.40 / 0.10 EUR/kWh): the MPC pre-heats before the peak hours.",
            tariff=time_of_use,
            comfort=comfort,
        ),
        Scenario(
            name="dynamic_price",
            description="Day-ahead hourly prices: the heating follows the price signal.",
            tariff=dynamic,
            comfort=comfort,
        ),
        Scenario(
            name="demand_response",
            description="Time-of-use tariff with a demand response event (price x3 and a power cap, 17h-20h on day 1).",
            tariff=time_of_use.with_events([event]),
            comfort=comfort,
        ),
        Scenario(
            name="peak_shaving",
            description="Deep night setback and a capacity charge on the peak: the MPC spreads the morning boost.",
            tariff=FlatTariff(price=0.25, capacity_charge=8.0),
            comfort=deep_setback,
        ),
        Scenario(
            name="wide_comfort_band",
            description="Day-ahead prices with the upper bound raised by 2 K: room to pre-heat, same comfort.",
            tariff=dynamic,
            comfort=comfort.map("widen", 2.0, "upper"),
        ),
        Scenario(
            name="imperfect_forecast",
            description="Time-of-use tariff with forecast errors (2 K on the temperatures, 30 % on the irradiance).",
            tariff=time_of_use,
            comfort=comfort,
            forecast_noise=ForecastNoise(temperature_std=2.0, irradiance_relative_std=0.3, seed=1),
            estimator="mhe",
        ),
    ]
    return {
        scenario.name: scenario.model_copy(update={"start": start, "days": days, "settings": settings})
        for scenario in scenarios
    }


def build_estimator(scenario: Scenario, model: BuildingModel, plant: Plant) -> StateEstimator:
    if scenario.estimator == "perfect":
        return PerfectEstimator(plant)
    return MovingHorizonEstimator(model, scenario.settings.time_step, plant.time)


def run_scenario(
    model: BuildingModel,
    scenario: Scenario,
    plant: Plant | BuildingModel | None = None,
    forecaster: Forecaster | None = None,
    estimator: StateEstimator | None = None,
) -> ClosedLoopResult:
    """Closed loop of the scenario with ``model`` in the controller and ``plant`` as the building.

    ``plant`` defaults to ``model`` itself (perfect model); a :class:`BuildingModel` is wrapped into a
    :class:`ModelPlant` starting at the start of the scenario. The forecaster defaults to the weather file of the
    model, with the forecast errors of the scenario.
    """
    settings = scenario.settings
    if forecaster is None:
        forecaster = WeatherForecaster.from_model(model, noise=scenario.forecast_noise)
    if plant is None:
        plant = ModelPlant(model, settings.time_step, scenario.start)
    elif isinstance(plant, BuildingModel):
        plant = ModelPlant(plant, settings.time_step, scenario.start)
    if estimator is None:
        estimator = build_estimator(scenario, model, plant)
    controller = ModelPredictiveController(model, settings, scenario.comfort, scenario.tariff)
    return run_closed_loop(controller, estimator, plant, forecaster, scenario.steps, name=scenario.name)


def run_scenario_rule_based(
    closed_loop: BuildingModel, scenario: Scenario, forecaster: Forecaster | None = None
) -> ClosedLoopResult:
    """The controllers of the YAML (the closed-loop ``BuildingModel``) over the period of the scenario, costed with its
    tariff and assessed against its comfort band: the reference of the MPC runs."""
    settings = scenario.settings
    return run_rule_based(
        closed_loop,
        forecaster or WeatherForecaster.from_model(closed_loop),
        scenario.start,
        scenario.steps,
        settings.time_step,
        scenario.comfort,
        scenario.tariff,
        fuel_price=settings.fuel_price,
        feed_in_price=settings.feed_in_price,
        name=f"{scenario.name} (rule based)",
    )


__all__ = [
    "DEFAULT_START",
    "ComfortSchedule",
    "Scenario",
    "build_estimator",
    "run_scenario",
    "run_scenario_rule_based",
    "scenario_catalogue",
]
