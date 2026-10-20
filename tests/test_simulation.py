"""Forecasts, closed loop and rule-based baseline of Trano's single-zone hydronic house as a FastModel."""

import numpy as np
import pandas as pd
import pytest

from mpctrano import (
    ComfortSchedule,
    ComfortSpec,
    FastModel,
    ModelPredictiveController,
    MPCSettings,
    WeatherForecaster,
    FlatTariff,
    ForecastNoise,
    TimeOfUseTariff,
)
from mpctrano.estimation import PerfectEstimator
from mpctrano.simulation import ModelPlant, run_closed_loop, run_rule_based
from mpctrano.units import KELVIN
from tests.conftest import BOILER_HOUSE, START, WEATHER

HOUSE = BOILER_HOUSE
STEP = 3600.0
# the band of the emission control of the YAML: 22 degC from 10h to 20h, 18 degC otherwise
COMFORT = ComfortSpec(
    default=ComfortSchedule(
        occupied_lower=22.0, occupied_upper=26.0, unoccupied_lower=18.0, unoccupied_upper=28.0, occupied_hours=(10, 20)
    )
)


@pytest.fixture(scope="module")
def house() -> FastModel:
    return FastModel.from_yaml(HOUSE)


@pytest.fixture(scope="module")
def closed_loop_house() -> FastModel:
    return FastModel.from_yaml(HOUSE, library="FastModel")


def test_the_model_knows_its_weather_file_and_occupancy(house: FastModel) -> None:
    assert house.weather_file is not None
    assert house.weather_file.endswith(WEATHER.name)
    assert house.weather_path == WEATHER.resolve()
    schedule = house.occupancy["occupancy_1"]
    assert (schedule.start_hour, schedule.end_hour) == (9.0, 17.0)
    assert schedule.gain_per_area == pytest.approx((40 + 75) / 21)


def test_forecaster_gives_every_disturbance(house: FastModel) -> None:
    forecaster = WeatherForecaster.from_model(house)
    frame = forecaster.realisation(pd.Timestamp(START), 24, STEP)
    assert frame.columns.tolist() == house.disturbance_names
    assert frame.notna().all().all()
    occupied = (frame.index.hour >= 9) & (frame.index.hour < 17)
    np.testing.assert_allclose(frame["occupancy_1_gai.QRad"], np.where(occupied, 40 / 21, 0.0))
    np.testing.assert_allclose(frame["occupancy_1_gai.QCon"], np.where(occupied, 75 / 21, 0.0))
    assert np.all(frame["weather_inputs.TBlaSky"] < frame["weather_inputs.TDryBul"])
    assert frame["weather_inputs.TDryBul"].between(KELVIN - 20, KELVIN + 20).all()

    noisy = WeatherForecaster.from_model(house, noise=ForecastNoise(temperature_std=2.0, seed=0))
    forecast = noisy.forecast(pd.Timestamp(START), 24, STEP)
    assert not np.allclose(forecast["weather_inputs.TDryBul"], frame["weather_inputs.TDryBul"])
    pd.testing.assert_series_equal(forecast["weather_inputs.winSpe"], frame["weather_inputs.winSpe"])


def test_closed_loop_meters_the_energy_of_the_plant(house: FastModel) -> None:
    steps = 6
    plant = ModelPlant(house, STEP, START)
    controller = ModelPredictiveController(
        house, MPCSettings(horizon=12, fuel_price=0.08), COMFORT, FlatTariff(price=0.3)
    )
    result = run_closed_loop(controller, PerfectEstimator(plant), plant, WeatherForecaster.from_model(house), steps)
    assert result.failures == 0
    assert len(result.frame) == steps
    assert plant.time == pd.Timestamp(START) + pd.Timedelta(hours=steps)
    meter = house.state_names.index("boiler_001.EFue")
    assert result.metrics.fuel_kwh == pytest.approx(plant.state[meter] - house.initial_state[meter])
    assert result.metrics.energy_cost == pytest.approx(0.08 * result.metrics.fuel_kwh)
    assert result.metrics.electricity_kwh == 0.0
    assert result.metrics.comfort_violation_kh < 0.1


def test_rule_based_baseline_runs_the_controllers_of_the_yaml(closed_loop_house: FastModel, house: FastModel) -> None:
    assert closed_loop_house.control_names == []
    result = run_rule_based(
        closed_loop_house,
        WeatherForecaster.from_model(closed_loop_house),
        START,
        24,
        STEP,
        COMFORT,
        FlatTariff(price=0.08),
    )
    temperatures = result.frame["space_001.TAir_end"] - KELVIN
    assert temperatures.iloc[13:19].between(21.5, 22.5).all()  # 22 degC set point from 10h to 20h
    assert result.metrics.fuel_kwh > 5.0
    assert result.metrics.energy_cost == pytest.approx(0.08 * result.metrics.fuel_kwh)
    with pytest.raises(ValueError, match="closed-loop"):
        run_rule_based(house, WeatherForecaster.from_model(house), START, 1, STEP, COMFORT, FlatTariff())


@pytest.mark.slow
def test_mpc_beats_the_rule_based_controllers_under_a_time_of_use_tariff(
    house: FastModel, closed_loop_house: FastModel
) -> None:
    """The MPC keeps the band of the rule-based controllers for less, buying more energy off-peak."""
    tariff, steps = TimeOfUseTariff(peak_price=0.12, off_peak_price=0.05), 24
    baseline = run_rule_based(
        closed_loop_house, WeatherForecaster.from_model(closed_loop_house), START, steps, STEP, COMFORT, tariff
    )
    plant = ModelPlant(house, STEP, START)
    controller = ModelPredictiveController(house, MPCSettings(horizon=24), COMFORT, tariff)
    mpc = run_closed_loop(controller, PerfectEstimator(plant), plant, WeatherForecaster.from_model(house), steps)
    assert mpc.failures == 0
    assert mpc.metrics.total_cost < 0.95 * baseline.metrics.total_cost
    assert mpc.metrics.comfort_violation_kh < baseline.metrics.comfort_violation_kh
    assert mpc.metrics.low_price_share > baseline.metrics.low_price_share
