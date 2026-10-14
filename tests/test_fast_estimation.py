"""Moving horizon estimation of Trano's single-zone hydronic house as a FastModel.

Skipped unless rumoca exports connected Modelica models to CasADi (``make rumoca-fastmodel``).
"""

import numpy as np
import pandas as pd
import pytest

from mpctrano import (
    ComfortSchedule,
    ComfortSpec,
    FastModel,
    FastModelPredictiveController,
    FastMPCSettings,
    FastWeatherForecaster,
    TimeOfUseTariff,
)
from mpctrano.fast_estimation import FastEstimatorSettings, FastMovingHorizonEstimator
from mpctrano.fast_model import rumoca_exports_fast_models
from mpctrano.fast_simulation import FastModelPlant, run_fast_closed_loop
from tests.conftest import MODELS, START

pytestmark = pytest.mark.skipif(
    not rumoca_exports_fast_models(),
    reason="the CasADi export of FastModel models needs rumoca main at c74cddf (make rumoca-fastmodel)",
)

HOUSE = MODELS / "single_zone_hydronic.yaml"
STEP = 3600.0


@pytest.fixture(scope="module")
def house() -> FastModel:
    return FastModel.from_yaml(HOUSE)


def temperatures(model: FastModel) -> list[int]:
    return [i for i, state in enumerate(model.model.description.states) if state.unit == "K"]


def colder_building(model: FastModel, offset: float = 2.0) -> np.ndarray:
    """A true initial state far from the default one: every temperature ``offset`` K lower."""
    state = model.initial_state.copy()
    state[temperatures(model)] -= offset
    return state


def test_the_temperature_states_have_their_unit(house: FastModel) -> None:
    """rumoca drops the unit of array states (wall layers, radiator water); FastModel restores it."""
    names = [house.state_names[i] for i in temperatures(house)]
    assert "externalwall_0.T[3]" in names
    assert "radiator_003.T[1]" in names
    assert "boiler_001.EFue" not in names
    assert len(names) == house.n_states - len(house.meters)


def test_estimation_recovers_the_unmeasured_states(house: FastModel) -> None:
    plant = FastModelPlant(house, STEP, START, x0=colder_building(house), seed=1)
    estimator = FastMovingHorizonEstimator(house, STEP, START, settings=FastEstimatorSettings(window=6))
    open_loop = FastModelPlant(house, STEP, START, measurement_std=0.0)  # the prior, never corrected
    forecaster = FastWeatherForecaster.from_model(house)
    rows = temperatures(house)
    for k in range(12):
        estimate = estimator.correct(plant.measure())
        control = np.ones(3) if (k // 3) % 2 == 0 else np.zeros(3)  # heating on and off every 3 hours
        disturbance = forecaster.realisation(plant.time, 1, STEP).iloc[0].to_numpy()
        plant.step(control, disturbance)
        estimator.predict(control, disturbance)
        open_loop.step(control, disturbance)
    estimate = estimator.correct(plant.measure())
    error = np.abs(estimate[rows] - plant.state[rows])
    prediction_error = np.abs(open_loop.state[rows] - plant.state[rows])
    assert all(status == "Solve_Succeeded" for status in estimator.statuses)
    assert len(estimator.statuses) == 12
    assert error.mean() < 0.2
    assert error.mean() < 0.2 * prediction_error.mean()


def test_the_cycle_starts_with_a_correction(house: FastModel) -> None:
    estimator = FastMovingHorizonEstimator(house, STEP, START)
    with pytest.raises(RuntimeError, match="correct"):
        estimator.predict(np.zeros(3), np.zeros(house.n_disturbances))
    estimate = estimator.correct(np.array([290.0]))
    assert estimate[house.comfort_indices[0]] == 290.0  # before any input: the prior with the measured air
    assert estimator.statuses == []


@pytest.mark.slow
def test_closed_loop_with_estimation_keeps_comfort(house: FastModel) -> None:
    comfort = ComfortSpec(
        default=ComfortSchedule(
            occupied_lower=22.0,
            occupied_upper=26.0,
            unoccupied_lower=18.0,
            unoccupied_upper=28.0,
            occupied_hours=(10, 20),
        )
    )
    plant = FastModelPlant(house, STEP, START, x0=colder_building(house))
    estimator = FastMovingHorizonEstimator(house, STEP, START, settings=FastEstimatorSettings(window=6))
    controller = FastModelPredictiveController(
        house, FastMPCSettings(horizon=24), comfort, TimeOfUseTariff(peak_price=0.12, off_peak_price=0.05)
    )
    result = run_fast_closed_loop(controller, estimator, plant, FastWeatherForecaster.from_model(house), 24)
    assert result.failures == 0
    assert result.metrics.comfort_violation_kh < 0.5
    rows = [house.state_names[i] for i in temperatures(house)]
    last = result.frame.iloc[-1]
    error = np.abs(
        last[[f"{name}_estimate" for name in rows]].to_numpy() - last[[f"{name}_true" for name in rows]].to_numpy()
    )
    assert error.mean() < 0.2
    assert pd.Timestamp(result.frame.index[-1]) == pd.Timestamp(START) + pd.Timedelta(hours=23)
