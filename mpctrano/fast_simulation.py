"""Closed loop of a FastModel: estimator -> MPC -> plant, and the rule-based baseline.

The plant is a FastModel simulated with CVODES, keeping its own clock (the model depends on the time of the
year). Unlike the RC models, the controls are not the heat delivered: the energy and its cost are read from the
meters of the plant (fuel, electricity, production) at the prices of each carrier. The same accounting applies to
:func:`run_rule_based`, which simulates the closed-loop ``FastModel`` of the same YAML, with the controllers it
describes, as the reference the MPC is compared with.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from mpctrano.comfort import ComfortSpec
from mpctrano.fast_model import FastModel
from mpctrano.fast_mpc import FastModelPredictiveController
from mpctrano.forecast import Forecaster, forecast_index, seconds_of_year
from mpctrano.mpc import WATT_HOUR_TO_JOULE
from mpctrano.tariff import Tariff


@dataclass
class FastModelPlant:
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


@dataclass
class PerfectEstimator:
    """The true state of the plant (no estimation): to study the controller alone."""

    plant: FastModelPlant

    @property
    def state(self) -> np.ndarray:
        return self.plant.state

    def correct(self, measurement: np.ndarray) -> np.ndarray:
        return self.plant.state

    def predict(self, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        return self.plant.state


@dataclass(frozen=True)
class Prices:
    """Prices of the carriers over the steps of a run [EUR/kWh]."""

    electricity: np.ndarray
    feed_in: np.ndarray
    fuel: np.ndarray
    capacity_charge: float = 0.0

    @classmethod
    def of(
        cls, tariff: Tariff, index: pd.DatetimeIndex, fuel_price: float | None, feed_in_price: float | None
    ) -> Prices:
        """The prices of ``tariff``; the fuel and the electricity sold at the tariff price unless given."""
        signal = tariff.signal(index)
        steps = len(index)

        def price(value: float | None) -> np.ndarray:
            return signal.prices.copy() if value is None else np.full(steps, float(value))

        return cls(
            electricity=signal.prices,
            feed_in=price(feed_in_price),
            fuel=price(fuel_price),
            capacity_charge=signal.capacity_charge,
        )


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


class FastControlMetrics(BaseModel):
    """Energy, cost, comfort and peak of a run of a FastModel, from its meters."""

    fuel_kwh: float = Field(description="Fuel drawn [kWh]")
    electricity_kwh: float = Field(description="Electricity bought from the grid [kWh]")
    sold_kwh: float = Field(description="Electricity sold to the grid [kWh]")
    energy_kwh: float = Field(description="Energy bought: fuel and electricity [kWh]")
    energy_cost: float = Field(description="Cost of the fuel and of the electricity bought, less the sales [EUR]")
    capacity_cost: float = Field(0.0, description="Capacity charge of the peak electric power [EUR]")
    comfort_violation_kh: float = Field(description="Integral of the band violations over the zones [K.h]")
    max_violation: float = Field(description="Largest band violation [K]")
    peak_power_kw: float = Field(description="Peak electric power bought [kW]")
    peak_fuel_power_kw: float = Field(description="Peak fuel power [kW]")
    low_price_share: float = Field(description="Share of the energy bought below the median electricity price")
    mean_price: float = Field(description="Average price of the energy bought [EUR/kWh]")
    steps: int

    @property
    def total_cost(self) -> float:
        return self.energy_cost + self.capacity_cost

    @classmethod
    def compute(
        cls,
        energy: pd.DataFrame,
        prices: Prices,
        temperatures: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        time_step: float,
    ) -> FastControlMetrics:
        """``energy`` has the ``fuel``, ``bought`` and ``sold`` energy of each step [kWh]; ``temperatures``,
        ``lower`` and ``upper`` are ``(n_zones, N)``, the temperatures at the end of each step."""
        hours = time_step / 3600
        fuel, bought, sold = (energy[name].to_numpy(dtype=float) for name in ("fuel", "bought", "sold"))
        purchased = fuel + bought
        cost_of_energy = prices.fuel * fuel + prices.electricity * bought - prices.feed_in * sold
        violation = np.maximum(lower - temperatures, 0) + np.maximum(temperatures - upper, 0)
        total = float(purchased.sum())
        cheap = prices.electricity < float(np.median(prices.electricity)) if len(energy) else np.array([], bool)
        peak = float(bought.max() / hours) if len(energy) else 0.0
        return cls(
            fuel_kwh=float(fuel.sum()),
            electricity_kwh=float(bought.sum()),
            sold_kwh=float(sold.sum()),
            energy_kwh=total,
            energy_cost=float(cost_of_energy.sum()),
            capacity_cost=prices.capacity_charge * peak,
            comfort_violation_kh=float(np.sum(violation) * hours),
            max_violation=float(np.max(violation)) if violation.size else 0.0,
            peak_power_kw=peak,
            peak_fuel_power_kw=float(fuel.max() / hours) if len(energy) else 0.0,
            low_price_share=float(purchased[cheap].sum() / total) if total > 0 else 0.0,
            mean_price=float((prices.fuel * fuel + prices.electricity * bought).sum() / total)
            if total > 0
            else float("nan"),
            steps=len(energy),
        )


@dataclass
class FastClosedLoopResult:
    """Trajectories of a run of a FastModel and its indicators."""

    name: str
    frame: pd.DataFrame
    metrics: FastControlMetrics
    time_step: float
    failures: int = 0
    elapsed: float = 0.0

    def summary(self) -> pd.Series:
        values = self.metrics.model_dump()
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
    prices: Prices,
    time_step: float,
    failures: int,
    started: float,
) -> FastClosedLoopResult:
    frame = pd.DataFrame(rows).set_index("time")
    index = pd.DatetimeIndex(frame.index)
    bounds = comfort.bounds(model, index + pd.Timedelta(seconds=time_step))
    for i, zone in enumerate(model.zone_names):
        frame[f"{zone}_lower"] = bounds.lower[i]
        frame[f"{zone}_upper"] = bounds.upper[i]
    frame["price"] = prices.electricity
    frame["fuel_price"] = prices.fuel
    temperatures = frame[[f"{state}_end" for state in model.comfort_states]].to_numpy(dtype=float).T
    metrics = FastControlMetrics.compute(frame, prices, temperatures, bounds.lower, bounds.upper, time_step)
    return FastClosedLoopResult(
        name=name,
        frame=frame,
        metrics=metrics,
        time_step=time_step,
        failures=failures,
        elapsed=time.perf_counter() - started,
    )


def run_fast_closed_loop(
    controller: FastModelPredictiveController,
    estimator: PerfectEstimator,
    plant: FastModelPlant,
    forecaster: Forecaster,
    steps: int,
    name: str = "mpc",
) -> FastClosedLoopResult:
    """Run ``steps`` control intervals from the time of the plant."""
    model, settings = controller.model, controller.settings
    dt = settings.time_step
    start = plant.time
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
        row |= {
            "planned_cost": solution.total_cost,
            "solve_time": solution.solve_time,
            "iterations": solution.iterations,
            "success": float(solution.success),
        }
        rows.append(row)
    index = forecast_index(start, steps, dt)
    prices = Prices.of(controller.tariff, index, settings.fuel_price, settings.feed_in_price)
    return _result(name, model, rows, controller.comfort, prices, dt, failures, started)


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
) -> FastClosedLoopResult:
    """The closed-loop ``FastModel`` (``FastModel.from_yaml(path, library="FastModel")``), run with the controllers
    of its YAML over the same weather, and costed and assessed like an MPC run."""
    if model.control_names:
        raise ValueError(f"{model.name} has controls {model.control_names}: give the closed-loop FastModel.")
    plant = FastModelPlant(model, time_step, start, x0=x0, measurement_std=0.0)
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
    index = forecast_index(pd.Timestamp(start), steps, time_step)
    prices = Prices.of(tariff, index, fuel_price, feed_in_price)
    return _result(name, model, rows, comfort, prices, time_step, 0, started)
