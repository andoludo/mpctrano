"""Closed-loop simulation: estimator -> MPC -> plant, step after step.

The plant is usually the *true* building: the same RC structure with other parameters
(:func:`mpctrano.synthetic.perturb`), a higher-order structure generated from the same YAML
(an R4C3 plant against an R3C2 controller) or the calibrated model itself with measurement
noise. The controller only sees the measured states (zone temperatures, storages) and the
(possibly imperfect) forecasts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from mpctrano.estimation import StateEstimator
from mpctrano.forecast import Forecaster
from mpctrano.meter import Meter
from mpctrano.metrics import ControlMetrics
from mpctrano.model import RCModel
from mpctrano.mpc import ModelPredictiveController
from mpctrano.tariff import Tariff


class Plant(Protocol):
    """The system being controlled."""

    @property
    def state(self) -> np.ndarray: ...

    def align(self, model: RCModel) -> None:
        """Adopt the signal order of the control model (measured states, controls, disturbances)."""

    def measure(self) -> np.ndarray:
        """Measured states in the order of ``model.measured_states`` of the control model."""

    def step(self, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        """Apply the inputs (control model order) during one control interval; returns the new state."""


@dataclass
class ModelPlant:
    """An RC model used as the real building, with sensor noise.

    The plant may have another RC structure than the controller (e.g. R4C3 against R3C2): the
    signals are matched by name after :meth:`align`, so the plant needs every measured state,
    control and disturbance of the control model.
    """

    model: RCModel
    time_step: float
    x0: np.ndarray | None = None
    measurement_std: float = 0.05
    """Sensor noise of the temperatures [K]."""
    energy_measurement_relative_std: float = 0.0
    """Sensor noise of the stored energies, relative to the measured value."""
    seed: int | None = 0
    substeps: int | None = None
    _x: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.dynamics = self.model.discrete(self.time_step, self.substeps)
        self._x = self.model.initial_state.copy() if self.x0 is None else np.asarray(self.x0, dtype=float).copy()
        self._generator = np.random.default_rng(self.seed)
        self.align(self.model)

    def align(self, model: RCModel) -> None:
        try:
            self._measured = [self.model.state_names.index(name) for name in model.measured_states]
            self._control_order = [model.control_names.index(name) for name in self.model.control_names]
            self._disturbance_order = [model.disturbance_names.index(name) for name in self.model.disturbance_names]
        except ValueError as error:
            raise ValueError(f"The plant {self.model.name} lacks a signal of the control model: {error}") from error
        self._energy = np.array([self.model.state_spec(name).unit == "J" for name in model.measured_states], dtype=bool)

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    def measure(self) -> np.ndarray:
        values = self._x[self._measured]
        noise = self._generator.normal(0, self.measurement_std, values.shape)
        if self._energy.any():
            relative = self._generator.normal(0, self.energy_measurement_relative_std, values.shape)
            noise = np.where(self._energy, values * relative, noise)
        return values + noise

    def step(self, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        u = np.asarray(control, dtype=float)[self._control_order]
        d = np.asarray(disturbance, dtype=float)[self._disturbance_order]
        self._x = np.asarray(self.dynamics(self._x, u, d, self.model.parameters), dtype=float).reshape(-1)
        return self.state


@dataclass
class ClosedLoopResult:
    """Trajectories of a closed-loop run and its performance indicators.

    ``frame`` holds, per step: the controls, the disturbances, the measured states
    (``<state>_measured``), the estimates, the true states, the zone temperatures and storage
    states at the end of the step (``<state>_end``), the comfort bounds, the price and the meter
    columns (``P_import``, ``P_export``, ``P_meter``, ``P_gas``, ...).
    """

    name: str
    frame: pd.DataFrame
    metrics: ControlMetrics
    time_step: float
    failures: int = 0
    elapsed: float = 0.0
    heating_columns: list[str] = field(default_factory=list)
    """Heating input of each zone (``<zone>_QHea`` or ``<zone>_PHea``)."""
    gas_price: float = 0.0

    @property
    def zones(self) -> list[str]:
        return [column.removesuffix("_Ti_end") for column in self.frame.columns if column.endswith("_Ti_end")]

    def metrics_under(self, tariff: Tariff) -> ControlMetrics:
        """The indicators of this run costed with another tariff (same trajectories)."""
        index = pd.DatetimeIndex(self.frame.index)
        signal = tariff.signal(index)
        zones = self.zones
        return ControlMetrics.compute(
            index,
            self.frame["P_meter"].to_numpy(dtype=float),
            self.frame[[f"{zone}_Ti_end" for zone in zones]].to_numpy(dtype=float).T,
            self.frame[[f"{zone}_lower" for zone in zones]].to_numpy(dtype=float).T,
            self.frame[[f"{zone}_upper" for zone in zones]].to_numpy(dtype=float).T,
            signal.prices,
            self.time_step,
            capacity_charge=signal.capacity_charge,
            export=self.frame["P_export"].to_numpy(dtype=float),
            export_prices=signal.export,
            gas=self.frame["P_gas"].to_numpy(dtype=float),
            gas_price=self.gas_price,
            generation=self.frame["P_generation"].to_numpy(dtype=float),
        )

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
    plant.align(model)
    rows: list[dict[str, object]] = []
    previous_control: np.ndarray | None = None
    failures = 0
    started = time.perf_counter()
    end_states = model.measured_states
    end_indices = model.measured_indices
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
        row |= {f"{name_}_measured": value for name_, value in zip(end_states, measurement.tolist(), strict=True)}
        row |= {f"{name_}_estimate": value for name_, value in zip(model.state_names, estimate.tolist(), strict=True)}
        row |= _true_states(plant, model, true_before)
        ends = _select(plant, model, true_after, end_indices)
        row |= {f"{name_}_end": value for name_, value in zip(end_states, ends, strict=True)}
        row |= {f"{zone}_lower": solution.lower[i, 0] for i, zone in enumerate(model.zone_names)}
        row |= {f"{zone}_upper": solution.upper[i, 0] for i, zone in enumerate(model.zone_names)}
        row |= {
            "price": float(solution.prices[0]),
            "export_price": float(controller.tariff.signal(pd.DatetimeIndex([now])).export[0]),
            "planned_cost": solution.total_cost,
            "solve_time": solution.solve_time,
            "iterations": solution.iterations,
            "success": float(solution.success),
            "status": solution.status,
        }
        rows.append(row)
    frame = pd.DataFrame(rows).set_index("time")
    meter = Meter(model).evaluate(frame)
    frame = pd.concat([frame, meter], axis=1)
    frame["P_meter"] = frame["P_heat"] + frame["P_import"]
    index = pd.DatetimeIndex(frame.index)
    temperatures = frame[[f"{name_}_end" for name_ in model.comfort_states]].to_numpy(dtype=float).T
    lower = frame[[f"{zone}_lower" for zone in model.zone_names]].to_numpy(dtype=float).T
    upper = frame[[f"{zone}_upper" for zone in model.zone_names]].to_numpy(dtype=float).T
    metrics = ControlMetrics.compute(
        index,
        frame["P_meter"].to_numpy(dtype=float),
        temperatures,
        lower,
        upper,
        frame["price"].to_numpy(dtype=float),
        dt,
        capacity_charge=controller.tariff.capacity_charge,
        export=frame["P_export"].to_numpy(dtype=float),
        export_prices=frame["export_price"].to_numpy(dtype=float),
        gas=frame["P_gas"].to_numpy(dtype=float),
        gas_price=settings.gas_price,
        generation=frame["P_generation"].to_numpy(dtype=float),
    )
    return ClosedLoopResult(
        name=name,
        frame=frame,
        metrics=metrics,
        time_step=dt,
        failures=failures,
        elapsed=time.perf_counter() - started,
        heating_columns=[model.control_names[index] for index in model.zone_heating_indices],
        gas_price=settings.gas_price,
    )


def _plant_model(plant: Plant) -> RCModel | None:
    return getattr(plant, "model", None)


def initial_state_of(plant: Plant, model: RCModel) -> np.ndarray:
    """The plant state as seen by the control model (matched by name, unknown states at their initial value)."""
    plant_model = _plant_model(plant)
    state = plant.state
    if plant_model is None or plant_model.state_names == model.state_names:
        return np.asarray(state, dtype=float)
    values = model.initial_state.copy()
    for index, name in enumerate(model.state_names):
        if name in plant_model.state_names:
            values[index] = state[plant_model.state_names.index(name)]
    return values


def _select(plant: Plant, model: RCModel, state: np.ndarray, indices: list[int]) -> list[float]:
    """Values of the control model states ``indices`` in a plant state vector (matched by name)."""
    plant_model = _plant_model(plant)
    if plant_model is None or plant_model.state_names == model.state_names:
        return [float(value) for value in state[indices]]
    positions = [plant_model.state_names.index(model.state_names[index]) for index in indices]
    return [float(value) for value in state[positions]]


def _true_states(plant: Plant, model: RCModel, state: np.ndarray) -> dict[str, object]:
    plant_model = _plant_model(plant)
    names = plant_model.state_names if plant_model is not None else model.state_names
    return {f"{name}_true": float(value) for name, value in zip(names, state.tolist(), strict=True)}
