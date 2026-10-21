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
from mpctrano.tariff import CapacityTariff, Tariff


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
    model: FastModel,
    before: np.ndarray,
    after: np.ndarray,
    control: np.ndarray,
    time_step: float,
    heat_cop: float = 1.0,
    cooling_eer: float = 3.0,
    storage_kwh: float = 0.0,
) -> dict[str, float]:
    """Energy of one step [kWh], read from the meters: fuel, electricity bought and sold (the electricity meters and
    the heat of the ideal emitters through their COP, or their cooling through the EER, plus what the storages
    took from the meter, netted with the production)."""
    drawn = {"fuel": 0.0, "electricity": 0.0, "electricity_production": 0.0}
    for name, carrier in model.meters.items():
        index = model.state_names.index(name)
        drawn[carrier] += float(after[index] - before[index])
    flows = [float(control[model.control_names.index(name)]) for name in model.heat_controls]
    heat = (
        sum(max(flow, 0.0) / heat_cop + max(-flow, 0.0) / cooling_eer for flow in flows)
        * time_step
        / WATT_HOUR_TO_JOULE
    )
    net = drawn["electricity"] + heat + storage_kwh - drawn["electricity_production"]
    return {
        "fuel": drawn["fuel"],
        "bought": max(net, 0.0),
        "sold": max(-net, 0.0),
        "generated": drawn["electricity_production"],
    }


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
    frame["feed_in_price"] = prices.feed_in
    frame["fuel_price"] = prices.fuel
    frame["P_meter"] = frame["bought"] * WATT_HOUR_TO_JOULE / time_step  # billed electric power [W]
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
    """Run ``steps`` control intervals from the time of the plant: measure, estimate, forecast, solve, apply.

    Under a :class:`~mpctrano.tariff.CapacityTariff` with ``running_peak``, the controller carries the peak of the
    month so far (from its floor ``minimum_peak_kw``, reset when the month changes).
    """
    model, settings = controller.model, controller.settings
    dt = settings.time_step
    rows: list[dict[str, object]] = []
    previous: np.ndarray | None = None
    failures = 0
    tariff = controller.tariff
    carries_peak = isinstance(tariff, CapacityTariff) and tariff.running_peak
    peak_floor = 1000 * tariff.minimum_peak_kw if isinstance(tariff, CapacityTariff) else 0.0
    peak_so_far, month = peak_floor, plant.time.to_period("M")
    started = time.perf_counter()
    for _ in range(steps):
        now = plant.time
        if now.to_period("M") != month:
            peak_so_far, month = peak_floor, now.to_period("M")
        measurement = plant.measure()
        estimate = estimator.correct(measurement)
        solution = controller.solve(
            estimate,
            forecaster.forecast(now, settings.horizon, dt),
            previous,
            peak_so_far=peak_so_far if carries_peak else 0.0,
        )
        failures += not solution.success
        control = solution.first_control
        disturbance = forecaster.realisation(now, 1, dt).iloc[0].to_numpy(dtype=float)
        before = plant.state
        after = plant.step(control, disturbance)
        estimator.predict(control, disturbance)
        previous = control
        exchange = controller.advance(solution)
        energy = metered_energy(
            model, before, after, control, dt, settings.heat_cop, settings.cooling_eer, storage_kwh=exchange
        )
        peak_so_far = max(peak_so_far, energy["bought"] * WATT_HOUR_TO_JOULE / dt)
        row = _record(model, now, control, disturbance, before, after, energy)
        if solution.storage is not None:
            row |= {str(name): float(value) for name, value in solution.storage.iloc[0].items()}
        row |= {
            f"{state}_measured": value for state, value in zip(model.comfort_states, measurement.tolist(), strict=True)
        }
        row |= {f"{state}_estimate": value for state, value in zip(model.state_names, estimate.tolist(), strict=True)}
        row |= {
            "planned_cost": solution.total_cost,
            "solve_time": solution.solve_time,
            "iterations": solution.iterations,
            "success": float(solution.success),
            "peak_so_far": peak_so_far,
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


def spin_up(
    model: FastModel,
    closed_loop: FastModel,
    start: pd.Timestamp | str,
    days: float = 3.0,
    forecaster: Forecaster | None = None,
    time_step: float = 3600.0,
) -> np.ndarray:
    """A state of ``model`` at ``start`` after ``days`` of operation by the controllers of the YAML.

    A FastModel starts with every temperature at its start value (the walls at the set point, the buffer at its
    supply temperature): a closed loop from there spends its first day emptying heat the building would not hold.
    The closed-loop ``FastModel`` of the same YAML runs ``days`` before ``start``; the states the two models share
    (by name) are copied into the initial state of ``model``.
    """
    from mpctrano.forecast import WeatherForecaster

    forecaster = forecaster or WeatherForecaster.from_model(closed_loop)
    plant = ModelPlant(closed_loop, time_step, pd.Timestamp(start) - pd.Timedelta(days=days))
    none = np.zeros(0)
    for _ in range(round(days * 86400 / time_step)):
        plant.step(none, forecaster.realisation(plant.time, 1, time_step).iloc[0].to_numpy(dtype=float))
    reached = dict(zip(closed_loop.state_names, plant.state.tolist(), strict=True))
    return np.array(
        [reached.get(name, value) for name, value in zip(model.state_names, model.initial_state, strict=True)]
    )
