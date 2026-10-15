"""Closed loop of a FastModel: estimator -> MPC -> plant, and the rule-based reference.

The plant is a FastModel simulated with CVODES, keeping its own clock (the model depends on the time of the year).
The controls are those of the systems (valves, compressor speed, boiler load), not the heat delivered: the energy and
its cost are read from the meters of the plant (fuel, electricity, production) at the price of each carrier. The
same accounting applies to :func:`run_rule_based`, which simulates the closed-loop ``FastModel`` of the same YAML,
with the controllers it describes, as the reference the MPC is compared with.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from mpctrano.comfort import ComfortSpec
from mpctrano.estimation import StateEstimator
from mpctrano.forecast import Forecaster, seconds_of_year
from mpctrano.metrics import ControlMetrics, Prices
from mpctrano.model import FastModel
from mpctrano.mpc import WATT_HOUR_TO_JOULE, ModelPredictiveController
from mpctrano.tariff import Tariff


class Plant(Protocol):
    """The system being controlled."""

    @property
    def state(self) -> np.ndarray: ...

    @property
    def time(self) -> pd.Timestamp: ...

    def measure(self) -> np.ndarray:
        """Measured zone air temperatures [K]."""

    def step(self, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        """Apply the inputs during one control interval; returns the new state."""


@dataclass
class ModelPlant:
    """A FastModel playing the real building, with sensor noise on the zone air temperatures."""

    model: FastModel
    time_step: float
    start: pd.Timestamp | str
    x0: np.ndarray | None = None
    measurement_std: float = 0.05
    seed: int | None = 0
    _x: np.ndarray = field(init=False, repr=False)
    _steps: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self.start = pd.Timestamp(self.start)
        self._x = self.model.initial_state.copy() if self.x0 is None else np.asarray(self.x0, dtype=float).copy()
        self._generator = np.random.default_rng(self.seed)

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    @property
    def time(self) -> pd.Timestamp:
        return pd.Timestamp(self.start) + pd.Timedelta(seconds=self._steps * self.time_step)

    def measure(self) -> np.ndarray:
        temperatures = self._x[self.model.comfort_indices]
        return temperatures + self._generator.normal(0, self.measurement_std, temperatures.shape)

    def step(self, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        start_time = float(seconds_of_year(pd.DatetimeIndex([self.time]))[0])
        trajectory = self.model.simulate(
            self._x,
            np.asarray(control, dtype=float).reshape(-1, 1),
            np.asarray(disturbance, dtype=float).reshape(-1, 1),
            self.time_step,
            start_time=start_time,
        )
        self._x = trajectory[:, -1].copy()
        self._steps += 1
        return self.state


def metered_energy(
    model: FastModel, before: np.ndarray, after: np.ndarray, control: np.ndarray, time_step: float
) -> dict[str, float]:
    """Energy of one step [kWh], read from the meters: fuel, electricity bought and sold (the electricity meters and
    the heat of the ideal emitters, netted with the production)."""
    drawn = {"fuel": 0.0, "electricity": 0.0, "electricity_production": 0.0}
    for name, carrier in model.meters.items():
        index = model.state_names.index(name)
        drawn[carrier] += float(after[index] - before[index])
    heat = (
        sum(float(control[model.control_names.index(name)]) for name in model.heat_controls)
        * time_step
        / WATT_HOUR_TO_JOULE
    )
    net = drawn["electricity"] + heat - drawn["electricity_production"]
    return {"fuel": drawn["fuel"], "bought": max(net, 0.0), "sold": max(-net, 0.0)}


@dataclass
class ClosedLoopResult:
    """Trajectories of a run and its indicators.

    ``frame`` has one row per step: the controls, the disturbances, the true states at the start of the step
    (``<state>_true``), the zone air temperatures at its end (``<state>_end``), the comfort band (``<zone>_lower``,
    ``<zone>_upper``), the energy of the step (``fuel``, ``bought``, ``sold`` [kWh]), the prices and, for an MPC
    run, the measurements, the estimates and the solver statistics.
    """

    name: str
    frame: pd.DataFrame
    metrics: ControlMetrics
    time_step: float
    zones: list[str]
    comfort_states: list[str]
    fuel_price: float | None = None
    feed_in_price: float | None = None
    failures: int = 0
    elapsed: float = 0.0

    @property
    def energy(self) -> pd.Series:
        """Energy bought at each step, fuel and electricity [kWh]."""
        return (self.frame["fuel"] + self.frame["bought"]).rename("energy_kwh")

    def metrics_under(self, tariff: Tariff) -> ControlMetrics:
        """The indicators of this run costed with another tariff (same trajectories)."""
        index = pd.DatetimeIndex(self.frame.index)
        prices = Prices.of(tariff, index, self.fuel_price, self.feed_in_price)
        return ControlMetrics.compute(
            self.frame,
            prices,
            self.frame[[f"{state}_end" for state in self.comfort_states]].to_numpy(dtype=float).T,
            self.frame[[f"{zone}_lower" for zone in self.zones]].to_numpy(dtype=float).T,
            self.frame[[f"{zone}_upper" for zone in self.zones]].to_numpy(dtype=float).T,
            self.time_step,
        )

    def summary(self) -> pd.Series:
        values: dict[str, float] = {**self.metrics.model_dump()}
        values["total_cost"] = self.metrics.total_cost
        values["failures"] = self.failures
        if "solve_time" in self.frame:
            values["mean_solve_time"] = float(self.frame["solve_time"].mean())
        return pd.Series(values, name=self.name)


def _record(
    model: FastModel,
    now: pd.Timestamp,
    control: np.ndarray,
    disturbance: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    energy: dict[str, float],
) -> dict[str, object]:
    row: dict[str, object] = {"time": now}
    row |= dict(zip(model.control_names, control.tolist(), strict=True))
    row |= dict(zip(model.disturbance_names, disturbance.tolist(), strict=True))
    row |= {f"{name}_true": value for name, value in zip(model.state_names, before.tolist(), strict=True)}
    row |= {
        f"{name}_end": value
        for name, value in zip(model.comfort_states, after[model.comfort_indices].tolist(), strict=True)
    }
    row |= energy
    return row


def _result(
    name: str,
    model: FastModel,
    rows: list[dict[str, object]],
    comfort: ComfortSpec,
    tariff: Tariff,
    fuel_price: float | None,
    feed_in_price: float | None,
    time_step: float,
    failures: int,
    started: float,
) -> ClosedLoopResult:
    frame = pd.DataFrame(rows).set_index("time")
    index = pd.DatetimeIndex(frame.index)
    bounds = comfort.bounds(model, index + pd.Timedelta(seconds=time_step))
    for i, zone in enumerate(model.zone_names):
        frame[f"{zone}_lower"] = bounds.lower[i]
        frame[f"{zone}_upper"] = bounds.upper[i]
    prices = Prices.of(tariff, index, fuel_price, feed_in_price)
    frame["price"] = prices.electricity
    frame["fuel_price"] = prices.fuel
    temperatures = frame[[f"{state}_end" for state in model.comfort_states]].to_numpy(dtype=float).T
    return ClosedLoopResult(
        name=name,
        frame=frame,
        metrics=ControlMetrics.compute(frame, prices, temperatures, bounds.lower, bounds.upper, time_step),
        time_step=time_step,
        zones=list(model.zone_names),
        comfort_states=list(model.comfort_states),
        fuel_price=fuel_price,
        feed_in_price=feed_in_price,
        failures=failures,
        elapsed=time.perf_counter() - started,
    )


def run_closed_loop(
    controller: ModelPredictiveController,
    estimator: StateEstimator,
    plant: Plant,
    forecaster: Forecaster,
    steps: int,
    name: str = "mpc",
) -> ClosedLoopResult:
    """Run ``steps`` control intervals from the time of the plant: measure, estimate, forecast, solve, apply."""
    model, settings = controller.model, controller.settings
    dt = settings.time_step
    rows: list[dict[str, object]] = []
    previous: np.ndarray | None = None
    failures = 0
    started = time.perf_counter()
    for _ in range(steps):
        now = plant.time
        measurement = plant.measure()
        estimate = estimator.correct(measurement)
        solution = controller.solve(estimate, forecaster.forecast(now, settings.horizon, dt), previous)
        failures += not solution.success
        control = solution.first_control
        disturbance = forecaster.realisation(now, 1, dt).iloc[0].to_numpy(dtype=float)
        before = plant.state
        after = plant.step(control, disturbance)
        estimator.predict(control, disturbance)
        previous = control
        row = _record(
            model, now, control, disturbance, before, after, metered_energy(model, before, after, control, dt)
        )
        row |= {
            f"{state}_measured": value for state, value in zip(model.comfort_states, measurement.tolist(), strict=True)
        }
        row |= {f"{state}_estimate": value for state, value in zip(model.state_names, estimate.tolist(), strict=True)}
        row |= {
            "planned_cost": solution.total_cost,
            "solve_time": solution.solve_time,
            "iterations": solution.iterations,
            "success": float(solution.success),
        }
        rows.append(row)
    return _result(
        name,
        model,
        rows,
        controller.comfort,
        controller.tariff,
        settings.fuel_price,
        settings.feed_in_price,
        dt,
        failures,
        started,
    )


def run_rule_based(
    model: FastModel,
    forecaster: Forecaster,
    start: pd.Timestamp | str,
    steps: int,
    time_step: float,
    comfort: ComfortSpec,
    tariff: Tariff,
    fuel_price: float | None = None,
    feed_in_price: float | None = None,
    x0: np.ndarray | None = None,
    name: str = "rule_based",
) -> ClosedLoopResult:
    """The closed-loop ``FastModel`` (``FastModel.from_yaml(path, library="FastModel")``), run with the controllers
    of its YAML over the same weather, and costed and assessed like an MPC run."""
    if model.control_names:
        raise ValueError(f"{model.name} has controls {model.control_names}: give the closed-loop FastModel.")
    plant = ModelPlant(model, time_step, start, x0=x0, measurement_std=0.0)
    rows = []
    started = time.perf_counter()
    none = np.zeros(0)
    for _ in range(steps):
        now = plant.time
        disturbance = forecaster.realisation(now, 1, time_step).iloc[0].to_numpy(dtype=float)
        before = plant.state
        after = plant.step(none, disturbance)
        rows.append(
            _record(model, now, none, disturbance, before, after, metered_energy(model, before, after, none, time_step))
        )
    return _result(name, model, rows, comfort, tariff, fuel_price, feed_in_price, time_step, 0, started)
