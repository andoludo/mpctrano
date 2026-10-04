import numpy as np
import pandas as pd
import pytest

from mpctrano import (
    ComfortSchedule,
    ComfortSpec,
    DemandResponseEvent,
    FlatTariff,
    KalmanFilter,
    ModelPlant,
    ModelPredictiveController,
    MPCSettings,
    RCModel,
    TimeOfUseTariff,
    WeatherForecaster,
    run_closed_loop,
)
from mpctrano.model import KELVIN

START = pd.Timestamp("2023-01-16")
SETTINGS = MPCSettings(horizon=24, time_step=3600.0)
COMFORT = ComfortSpec(
    default=ComfortSchedule(occupied_lower=21, occupied_upper=24, unoccupied_lower=17, unoccupied_upper=27)
)


@pytest.fixture(scope="module")
def forecast(forecaster: WeatherForecaster) -> pd.DataFrame:
    return forecaster.forecast(START, 48, 3600.0)


def _cold_start(model: RCModel) -> np.ndarray:
    return np.full(model.n_states, 18.0 + KELVIN)


def test_mpc_meets_comfort_within_the_bounds(model: RCModel, forecast: pd.DataFrame) -> None:
    controller = ModelPredictiveController(model, SETTINGS, COMFORT, FlatTariff(price=0.25))
    solution = controller.solve(_cold_start(model), forecast)
    assert solution.success
    powers = solution.controls.to_numpy()
    assert np.all(powers >= -1e-6) and np.all(powers <= model.max_heating_power + 1e-6)
    temperatures = solution.comfort_temperatures[model.comfort_states].to_numpy().T
    assert np.all(temperatures >= solution.lower - 1e-3)
    assert np.all(temperatures <= solution.upper + 1e-3)
    assert solution.costs["comfort"] < 1e-3
    assert solution.costs["energy"] > 0
    assert solution.first_control.shape == (3,)
    assert len(solution.states) == 25


def test_mpc_pre_heats_when_the_night_is_cheap(model: RCModel, forecast: pd.DataFrame) -> None:
    flat = ModelPredictiveController(model, SETTINGS, COMFORT, FlatTariff(price=0.25)).solve(
        _cold_start(model), forecast
    )
    cheap_night = TimeOfUseTariff(peak_price=0.6, off_peak_price=0.02, peak_hours=(7, 22))
    tou = ModelPredictiveController(model, SETTINGS, COMFORT, cheap_night).solve(_cold_start(model), forecast)
    night = np.arange(24) < 7
    flat_night = flat.controls.to_numpy()[night].sum()
    tou_night = tou.controls.to_numpy()[night].sum()
    assert tou_night > 1.5 * flat_night  # the heating moves to the cheap hours
    assert tou.states["space_001_Ti"].iloc[7] > flat.states["space_001_Ti"].iloc[7]  # warmer when the peak starts
    assert cheap_night.cost(tou.index, tou.controls.sum(axis=1).to_numpy(), 3600) < cheap_night.cost(
        flat.index, flat.controls.sum(axis=1).to_numpy(), 3600
    )


def test_power_limit_of_a_demand_response_event(model: RCModel, forecast: pd.DataFrame) -> None:
    event = DemandResponseEvent(
        start=START + pd.Timedelta(hours=8), end=START + pd.Timedelta(hours=11), power_limit=1500.0
    )
    tariff = FlatTariff(price=0.25).with_events([event])
    solution = ModelPredictiveController(model, SETTINGS, COMFORT, tariff).solve(_cold_start(model), forecast)
    total = solution.controls.sum(axis=1).to_numpy()
    assert np.all(total[8:11] <= 1500.0 + 1e-3)
    assert total[:8].max() > 1500.0


def test_capacity_charge_flattens_the_profile(model: RCModel, forecast: pd.DataFrame) -> None:
    x0 = model.initial_state.copy()
    flat = ModelPredictiveController(model, SETTINGS, COMFORT, FlatTariff(price=0.25)).solve(x0, forecast)
    shaved = ModelPredictiveController(model, SETTINGS, COMFORT, FlatTariff(price=0.25, capacity_charge=20.0)).solve(
        x0, forecast
    )
    assert shaved.controls.sum(axis=1).max() < flat.controls.sum(axis=1).max()
    assert shaved.costs["capacity"] > 0


def test_custom_power_bounds_and_cooling(model: RCModel, forecast: pd.DataFrame) -> None:
    settings = MPCSettings(horizon=12, max_power=dict.fromkeys(model.zone_names, 500.0), min_power=-500.0)
    controller = ModelPredictiveController(model, settings, COMFORT, FlatTariff())
    solution = controller.solve(_cold_start(model), forecast)
    powers = solution.controls.to_numpy()
    assert powers.max() <= 500.0 + 1e-6 and powers.min() >= -500.0 - 1e-6
    assert solution.costs["comfort"] > 0  # 500 W per zone is not enough to reach 21 degC
    with pytest.raises(ValueError, match="missing the zones"):
        ModelPredictiveController(model, MPCSettings(max_power={"space_001": 1.0}), COMFORT, FlatTariff())


def test_short_forecast_is_rejected(model: RCModel, forecast: pd.DataFrame) -> None:
    controller = ModelPredictiveController(model, SETTINGS, COMFORT, FlatTariff())
    with pytest.raises(ValueError, match="horizon"):
        controller.solve(_cold_start(model), forecast.iloc[:10])


def test_warm_start_keeps_solving(model: RCModel, forecast: pd.DataFrame) -> None:
    controller = ModelPredictiveController(model, SETTINGS, COMFORT, TimeOfUseTariff())
    first = controller.solve(_cold_start(model), forecast)
    second = controller.solve(first.states.iloc[1].to_numpy(), forecast.iloc[1:], first.first_control)
    assert first.success and second.success
    assert second.iterations <= first.iterations + 20


def test_closed_loop_with_a_perfect_model(model: RCModel, forecaster: WeatherForecaster) -> None:
    controller = ModelPredictiveController(model, SETTINGS, COMFORT, TimeOfUseTariff())
    plant = ModelPlant(model, 3600.0, measurement_std=0.0)
    estimator = KalmanFilter(model, 3600.0, x0=plant.state)
    result = run_closed_loop(controller, estimator, plant, forecaster, START, steps=12)
    assert len(result.frame) == 12
    assert result.failures == 0
    assert result.metrics.comfort_violation_kh < 0.05
    assert result.metrics.energy_kwh > 0
    assert result.zones == model.zone_names
    assert {"price", "solve_time", "space_001_Ti_end", "space_001_lower", "TOut"} <= set(result.frame.columns)
    assert result.summary()["total_cost"] == pytest.approx(result.metrics.total_cost)
