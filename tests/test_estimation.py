import numpy as np
import pandas as pd
import pytest

from mpctrano import KalmanFilter, MovingHorizonEstimator, RCModel, SyntheticData


def _run(estimator, model: RCModel, synthetic: SyntheticData, steps: int) -> np.ndarray:
    data = synthetic.data.frame
    estimates = []
    for k in range(steps):
        measurement = data[model.comfort_states].iloc[k].to_numpy()
        estimates.append(estimator.correct(measurement))
        control = data[model.control_names].iloc[k].to_numpy()
        disturbance = data[model.disturbance_names].iloc[k].to_numpy()
        estimator.predict(control, disturbance)
    return np.array(estimates)


@pytest.mark.parametrize("estimator_class", [KalmanFilter, MovingHorizonEstimator])
def test_estimators_recover_the_unmeasured_states(
    estimator_class, true_building: RCModel, synthetic: SyntheticData
) -> None:
    steps = 72
    x0 = true_building.initial_state + np.array([0, 3, 0, -3, 0, 2])  # wrong envelope temperatures
    estimator = estimator_class(true_building, 3600.0, x0=x0)
    estimates = _run(estimator, true_building, synthetic, steps)
    truth = synthetic.states.iloc[:steps].to_numpy()
    envelope = [true_building.state_names.index(f"space_00{i}_Te") for i in (1, 2, 3)]
    initial_error = np.abs(estimates[0, envelope] - truth[0, envelope]).max()
    final_error = np.abs(estimates[-12:, envelope] - truth[-12:, envelope]).max()
    assert initial_error > 1.0
    assert final_error < 0.5
    indoor = true_building.comfort_indices
    assert np.abs(estimates[-24:, indoor] - truth[-24:, indoor]).max() < 0.2


def test_kalman_filter_handles_missing_sensors(true_building: RCModel) -> None:
    estimator = KalmanFilter(true_building, 3600.0)
    before = estimator.state
    state = estimator.correct(np.array([np.nan, np.nan, np.nan]))
    assert state == pytest.approx(before)
    state = estimator.correct(np.array([295.0, np.nan, np.nan]))
    assert state[0] != pytest.approx(before[0])
    assert estimator.covariance.shape == (6, 6)


def test_mhe_window_is_bounded(true_building: RCModel, synthetic: SyntheticData) -> None:
    estimator = MovingHorizonEstimator(true_building, 3600.0, window=6)
    _run(estimator, true_building, synthetic, 20)
    assert len(estimator._measurements) == 7
    assert len(estimator._problems) <= 8


def test_estimates_are_consistent_with_the_closed_loop_timing(true_building: RCModel) -> None:
    """correct() then predict() reproduces a noise-free simulation exactly."""
    dynamics = true_building.discrete(3600.0)
    estimator = KalmanFilter(true_building, 3600.0, measurement_std=1e-3, process_std=1e-6)
    x = true_building.initial_state.copy()
    u = np.array([800.0, 500.0, 300.0])
    d = np.array([275.0, 100.0, 50.0, 20.0, 400.0, 300.0, 200.0])
    for _ in range(5):
        estimator.correct(x[true_building.comfort_indices])
        estimator.predict(u, d)
        x = np.array(dynamics(x, u, d, true_building.parameters)).reshape(-1)
    assert estimator.state == pytest.approx(x, abs=1e-3)
    assert isinstance(pd.Timestamp("2023-01-01"), pd.Timestamp)
