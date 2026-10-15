"""Synthetic buildings and monitoring data, for tests, tutorials and controller studies.

A *true* building is the Trano model with the parameter groups of the calibration scaled by random factors
(:func:`perturb`); monitoring data is recorded on it while a thermostat with a random set point operates it
(:func:`generate_monitoring_data`), which excites the dynamics enough for the calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mpctrano.calibration import DEFAULT_GROUPS, Calibrator, CalibrationSettings, ParameterGroup
from mpctrano.data import MonitoringData
from mpctrano.forecast import Forecaster, WeatherForecaster, forecast_index, seconds_of_year
from mpctrano.model import FastModel
from mpctrano.units import KELVIN


@dataclass(frozen=True)
class SyntheticData:
    data: MonitoringData
    """What a monitoring system would record."""
    states: pd.DataFrame
    """The true states (not available in practice)."""
    setpoints: pd.Series
    """The set point of the thermostat that operated the building [K]."""


def random_factors(
    relative_std: float = 0.3, seed: int | None = 0, groups: tuple[ParameterGroup, ...] = DEFAULT_GROUPS
) -> dict[str, float]:
    """Random values of the groups: a log-normal factor of standard deviation ``relative_std`` for a scale group
    (within its bounds), a uniform value over the first unit of its range for a value group."""
    generator = np.random.default_rng(seed)
    factors = {}
    for group in groups:
        if group.mode == "scale":
            factors[group.name] = float(np.clip(np.exp(generator.normal(0, relative_std)), group.lower, group.upper))
        else:
            factors[group.name] = float(generator.uniform(group.lower, min(group.upper, group.lower + 1.0)))
    return factors


def perturb(
    model: FastModel,
    relative_std: float = 0.3,
    seed: int | None = 0,
    groups: tuple[ParameterGroup, ...] = DEFAULT_GROUPS,
    factors: dict[str, float] | None = None,
) -> FastModel:
    """A copy of the model with the parameter groups at random values (:func:`random_factors`) or at ``factors``."""
    calibrator = Calibrator(model, CalibrationSettings(groups=groups))
    values = factors or random_factors(relative_std, seed, groups)
    return calibrator.calibrated({group.name: values[group.name] for group in calibrator.groups})


def generate_monitoring_data(
    building: FastModel,
    start: pd.Timestamp | str,
    days: float,
    time_step: float = 3600.0,
    forecaster: Forecaster | None = None,
    setpoints: tuple[float, float] = (17.0, 23.0),
    hold_hours: tuple[float, float] = (3.0, 12.0),
    gain: float = 0.5,
    measurement_std: float = 0.05,
    seed: int | None = 0,
    x0: np.ndarray | None = None,
) -> SyntheticData:
    """Monitoring data recorded on ``building`` (a FastModel playing the real building) and its true states.

    A thermostat with a random set point [degC] held for random durations drives the building: every control gets
    the same command, proportional (``gain`` per K) to the gap between the set point and the mean air temperature,
    so that the production only runs while the water flows (random controls overheat a boiler with closed valves).
    The data has the controls, the disturbances, the measured air temperatures and the meter readings.
    """
    generator = np.random.default_rng(seed)
    steps = round(days * 86400 / time_step)
    index = forecast_index(pd.Timestamp(start), steps + 1, time_step)
    forecaster = forecaster or WeatherForecaster.from_model(building)
    disturbances = forecaster.realisation(index[0], steps + 1, time_step)[building.disturbance_names].to_numpy()
    setpoint = np.zeros(steps + 1)
    k = 0
    while k <= steps:
        hold = max(1, round(generator.uniform(*hold_hours) * 3600 / time_step))
        setpoint[k : k + hold] = generator.uniform(*setpoints) + KELVIN
        k += hold
    lower, upper = building.control_bounds
    seconds = seconds_of_year(index)
    states = [building.initial_state if x0 is None else np.asarray(x0, dtype=float)]
    controls = np.zeros((building.n_controls, steps + 1))
    for k in range(steps + 1):
        command = float(np.clip(gain * (setpoint[k] - np.mean(states[-1][building.comfort_indices])), 0.0, 1.0))
        controls[:, k] = lower + (upper - lower) * command
        if k < steps:
            trajectory = building.simulate(
                states[-1], controls[:, k : k + 1], disturbances[k : k + 1].T, time_step, start_time=float(seconds[k])
            )
            states.append(trajectory[:, -1])
    truth = pd.DataFrame(np.vstack(states), index=index, columns=building.state_names)
    data = pd.DataFrame(controls.T, index=index, columns=building.control_names)
    data[building.disturbance_names] = disturbances
    for name in building.comfort_states:
        data[name] = truth[name] + generator.normal(0, measurement_std, len(index))
    for name in building.meters:
        data[name] = truth[name]
    return SyntheticData(
        data=MonitoringData(data), states=truth, setpoints=pd.Series(setpoint, index=index, name="setpoint")
    )
