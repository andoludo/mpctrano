"""Ready-made control scenarios: tariffs, comfort and forecasts combined into named studies."""

from __future__ import annotations

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mpctrano.comfort import ComfortSchedule, ComfortSpec
from mpctrano.estimation import KalmanFilter, MovingHorizonEstimator, StateEstimator
from mpctrano.forecast import ForecastNoise, Forecaster, WeatherForecaster
from mpctrano.model import RCModel
from mpctrano.mpc import ModelPredictiveController, MPCSettings
from mpctrano.simulation import ClosedLoopResult, ModelPlant, Plant, run_closed_loop
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
    estimator: Literal["kalman", "mhe"] = "kalman"

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
        power_limit=4000.0,
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
            description="Time-of-use tariff with forecast errors and a moving horizon estimator.",
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


def build_estimator(scenario: Scenario, model: RCModel, x0: object = None) -> StateEstimator:
    if scenario.estimator == "mhe":
        return MovingHorizonEstimator(model, scenario.settings.time_step, x0=x0)  # type: ignore[arg-type]
    return KalmanFilter(model, scenario.settings.time_step, x0=x0)  # type: ignore[arg-type]


def run_scenario(
    model: RCModel,
    scenario: Scenario,
    plant: Plant | RCModel | None = None,
    forecaster: Forecaster | None = None,
    estimator: StateEstimator | None = None,
) -> ClosedLoopResult:
    """Closed loop of the scenario with ``model`` in the controller and ``plant`` as the building.

    ``plant`` defaults to ``model`` itself (perfect model); an :class:`RCModel` is wrapped into a
    :class:`ModelPlant`. The forecaster defaults to the weather file of the model.
    """
    settings = scenario.settings
    if forecaster is None:
        forecaster = WeatherForecaster.from_model(model, noise=scenario.forecast_noise)
    if plant is None:
        plant = ModelPlant(model, settings.time_step)
    elif isinstance(plant, RCModel):
        plant = ModelPlant(plant, settings.time_step)
    if estimator is None:
        estimator = build_estimator(scenario, model, plant.state)
    controller = ModelPredictiveController(model, settings, scenario.comfort, scenario.tariff)
    return run_closed_loop(controller, estimator, plant, forecaster, scenario.start, scenario.steps, name=scenario.name)


__all__ = [
    "DEFAULT_START",
    "ComfortSchedule",
    "Scenario",
    "build_estimator",
    "run_scenario",
    "scenario_catalogue",
]
