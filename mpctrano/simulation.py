"""Closed-loop simulation: estimator -> MPC -> plant, step after step.

The plant is usually the *true* building: the same RC structure with other parameters
(:func:`mpctrano.synthetic.perturb`) or the calibrated model itself with measurement noise.
The controller only sees the measured zone temperatures and the (possibly imperfect) forecasts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from mpctrano.estimation import StateEstimator
from mpctrano.forecast import Forecaster
from mpctrano.metrics import ControlMetrics
from mpctrano.model import RCModel
from mpctrano.mpc import ModelPredictiveController


class Plant(Protocol):
    """The system being controlled."""

    @property
    def state(self) -> np.ndarray: ...

    def measure(self) -> np.ndarray:
        """Measured zone temperatures [K]."""

    def step(self, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        """Apply the inputs during one control interval; returns the new state."""


@dataclass
class ModelPlant:
    """An RC model used as the real building, with sensor noise."""

    model: RCModel
    time_step: float
    x0: np.ndarray | None = None
    measurement_std: float = 0.05
    seed: int | None = 0
    substeps: int | None = None
    _x: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.dynamics = self.model.discrete(self.time_step, self.substeps)
        self._x = self.model.initial_state.copy() if self.x0 is None else np.asarray(self.x0, dtype=float).copy()
        self._generator = np.random.default_rng(self.seed)

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    def measure(self) -> np.ndarray:
        temperatures = self._x[self.model.comfort_indices]
        return temperatures + self._generator.normal(0, self.measurement_std, temperatures.shape)

    def step(self, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        self._x = np.asarray(self.dynamics(self._x, control, disturbance, self.model.parameters), dtype=float).reshape(
            -1
        )
        return self.state


@dataclass
class ClosedLoopResult:
    """Trajectories of a closed-loop run and its performance indicators."""

    name: str
    frame: pd.DataFrame
    metrics: ControlMetrics
    time_step: float
    failures: int = 0
    elapsed: float = 0.0

    @property
    def zones(self) -> list[str]:
        return [column.removesuffix("_QHea") for column in self.frame.columns if column.endswith("_QHea")]

    def summary(self) -> pd.Series:
        values = self.metrics.model_dump()
        values["total_cost"] = self.metrics.total_cost
        values["failures"] = self.failures
        values["mean_solve_time"] = float(self.frame["solve_time"].mean())
        return pd.Series(values, name=self.name)


def run_closed_loop(
    controller: ModelPredictiveController,
    estimator: StateEstimator,
    plant: Plant,
    forecaster: Forecaster,
    start: pd.Timestamp | str,
    steps: int,
    name: str = "mpc",
) -> ClosedLoopResult:
    """Run ``steps`` control intervals from ``start``."""
    model, settings = controller.model, controller.settings
    dt = settings.time_step
    start = pd.Timestamp(start)
    rows: list[dict[str, object]] = []
    previous_control: np.ndarray | None = None
    failures = 0
    started = time.perf_counter()
    for k in range(steps):
        now = start + pd.Timedelta(seconds=k * dt)
        measurement = plant.measure()
        estimate = estimator.correct(measurement)
        forecast = forecaster.forecast(now, settings.horizon, dt)
        solution = controller.solve(estimate, forecast, previous_control)
        failures += not solution.success
        control = solution.first_control
        disturbance = forecaster.realisation(now, 1, dt).iloc[0].to_numpy(dtype=float)
        true_before = plant.state
        true_after = plant.step(control, disturbance)
        estimator.predict(control, disturbance)
        previous_control = control
        row: dict[str, object] = {"time": now}
        row |= dict(zip(model.control_names, control.tolist(), strict=True))
        row |= dict(zip(model.disturbance_names, disturbance.tolist(), strict=True))
        row |= {
            f"{name_}_measured": value for name_, value in zip(model.comfort_states, measurement.tolist(), strict=True)
        }
        row |= {f"{name_}_estimate": value for name_, value in zip(model.state_names, estimate.tolist(), strict=True)}
        row |= {f"{name_}_true": value for name_, value in zip(model.state_names, true_before.tolist(), strict=True)}
        row |= {
            f"{name_}_end": value
            for name_, value in zip(model.comfort_states, true_after[model.comfort_indices].tolist(), strict=True)
        }
        row |= {f"{zone}_lower": solution.lower[i, 0] for i, zone in enumerate(model.zone_names)}
        row |= {f"{zone}_upper": solution.upper[i, 0] for i, zone in enumerate(model.zone_names)}
        row |= {
            "price": float(solution.prices[0]),
            "planned_cost": solution.total_cost,
            "solve_time": solution.solve_time,
            "iterations": solution.iterations,
            "success": float(solution.success),
        }
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("time")
    index = pd.DatetimeIndex(frame.index)
    powers = frame[model.control_names].to_numpy(dtype=float).T
    temperatures = frame[[f"{name_}_end" for name_ in model.comfort_states]].to_numpy(dtype=float).T
    lower = frame[[f"{zone}_lower" for zone in model.zone_names]].to_numpy(dtype=float).T
    upper = frame[[f"{zone}_upper" for zone in model.zone_names]].to_numpy(dtype=float).T
    metrics = ControlMetrics.compute(
        index,
        powers,
        temperatures,
        lower,
        upper,
        frame["price"].to_numpy(dtype=float),
        dt,
        capacity_charge=controller.tariff.capacity_charge,
    )
    return ClosedLoopResult(
        name=name, frame=frame, metrics=metrics, time_step=dt, failures=failures, elapsed=time.perf_counter() - started
    )
