"""Economic model predictive control of the heating powers of the zones.

Every step, the controller solves with IPOPT, over a receding horizon of ``N`` steps:

.. math::

    \\min_{u, x, s}\\; \\sum_k \\pi_k \\, \\frac{\\sum_z u_{z,k}\\,\\Delta t}{3.6\\cdot 10^6}
    + c_\\mathrm{cap} \\frac{P_\\mathrm{peak}}{1000}
    + \\sum_k \\left(w_1 (s^{lo}_k + s^{hi}_k) + w_2 (s^{lo\\,2}_k + s^{hi\\,2}_k)\\right)\\frac{\\Delta t}{3600}
    + \\text{(power limit and ramp penalties)}

subject to the RC dynamics ``x[k+1] = F(x[k], u[k], d[k])``, the heating power bounds
``0 <= u <= u_max``, the soft comfort band ``T_lower - s_lo <= T_zone <= T_upper + s_hi`` and
the soft building-level power limit of the demand response events. ``pi_k`` is the energy price
of the tariff: with a time-varying price the optimiser pre-heats the thermal mass when the
energy is cheap and lets the temperature float inside the comfort band when it is expensive.

The optimisation problem is built once; each call to :meth:`ModelPredictiveController.solve`
only updates the parameters (initial state, forecasts, prices, bounds) and warm starts IPOPT.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mpctrano.calibration import ipopt_options
from mpctrano.comfort import ComfortSpec
from mpctrano.forecast import forecast_index
from mpctrano.model import STATE_BOUNDS, RCModel
from mpctrano.tariff import Tariff

WATT_HOUR_TO_JOULE = 3.6e6


class MPCSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizon: int = Field(24, ge=1, description="Prediction horizon [steps]")
    time_step: float = Field(3600.0, gt=0, description="Control interval [s]")
    comfort_penalty: float = Field(5.0, ge=0, description="Linear penalty of comfort violations [EUR/(K.h)]")
    comfort_quadratic: float = Field(1.0, ge=0, description="Quadratic penalty of comfort violations [EUR/(K2.h)]")
    power_limit_penalty: float = Field(
        50.0, ge=0, description="Penalty of the energy above a demand response power limit [EUR/kWh] (soft cap)"
    )
    ramp_weight: float = Field(0.0, ge=0, description="Penalty of control changes [EUR/kW2]")
    max_power: dict[str, float] | None = Field(None, description="Maximum heating power per zone [W]")
    min_power: float = Field(0.0, description="Minimum heating power [W] (negative allows cooling)")
    substeps: int | None = None
    max_iterations: int = Field(300, ge=1)
    tolerance: float = Field(1e-6, gt=0)
    verbose: bool = False


@dataclass(frozen=True)
class MPCSolution:
    """Optimal plan over the horizon (``controls`` has ``horizon`` rows, ``states`` one more)."""

    index: pd.DatetimeIndex
    controls: pd.DataFrame
    states: pd.DataFrame
    lower: np.ndarray
    upper: np.ndarray
    prices: np.ndarray
    costs: dict[str, float]
    success: bool
    status: str
    iterations: int
    solve_time: float

    @property
    def first_control(self) -> np.ndarray:
        return self.controls.iloc[0].to_numpy(dtype=float)

    @property
    def total_cost(self) -> float:
        return float(sum(self.costs.values()))

    @property
    def comfort_temperatures(self) -> pd.DataFrame:
        """Zone temperatures at the end of each step (the ones bounded by the comfort band)."""
        return self.states.iloc[1:]


class ModelPredictiveController:
    """Economic MPC of a Trano RC model, see the module documentation for the problem."""

    def __init__(self, model: RCModel, settings: MPCSettings, comfort: ComfortSpec, tariff: Tariff) -> None:
        self.model = model
        self.settings = settings
        self.comfort = comfort
        self.tariff = tariff
        self.max_power = self._max_power()
        self.dynamics = model.discrete(settings.time_step, settings.substeps)
        self._previous: dict[str, np.ndarray] | None = None
        self._build()

    def _max_power(self) -> np.ndarray:
        if self.settings.max_power is None:
            return self.model.max_heating_power
        missing = set(self.model.zone_names) - set(self.settings.max_power)
        if missing:
            raise ValueError(f"max_power is missing the zones {sorted(missing)}.")
        return np.array([self.settings.max_power[zone] for zone in self.model.zone_names], dtype=float)

    def _build(self) -> None:
        model, settings = self.model, self.settings
        N, dt = settings.horizon, settings.time_step
        hours = dt / 3600
        opti = ca.Opti()
        X = opti.variable(model.n_states, N + 1)
        U = opti.variable(model.n_controls, N)
        S_lo = opti.variable(model.n_zones, N)
        S_hi = opti.variable(model.n_zones, N)
        S_lim = opti.variable(1, N)
        peak = opti.variable()

        x0 = opti.parameter(model.n_states)
        D = opti.parameter(model.n_disturbances, N)
        price = opti.parameter(1, N)
        lower = opti.parameter(model.n_zones, N)
        upper = opti.parameter(model.n_zones, N)
        limit = opti.parameter(1, N)
        capacity = opti.parameter()
        u_previous = opti.parameter(model.n_controls)

        opti.subject_to(X[:, 0] == x0)
        for k in range(N):
            opti.subject_to(X[:, k + 1] == self.dynamics(X[:, k], U[:, k], D[:, k], model.parameters))
        opti.subject_to(opti.bounded(STATE_BOUNDS[0], ca.vec(X), STATE_BOUNDS[1]))
        opti.subject_to(opti.bounded(settings.min_power, ca.vec(U), np.tile(self.max_power, N)))
        opti.subject_to(ca.vec(S_lo) >= 0)
        opti.subject_to(ca.vec(S_hi) >= 0)
        opti.subject_to(ca.vec(S_lim) >= 0)
        comfort = X[model.comfort_indices, 1:]
        opti.subject_to(ca.vec(comfort + S_lo) >= ca.vec(lower))
        opti.subject_to(ca.vec(comfort - S_hi) <= ca.vec(upper))
        total = ca.sum1(U)
        opti.subject_to(ca.vec(total - S_lim) <= ca.vec(limit))
        opti.subject_to(ca.vec(total) <= peak)

        energy_cost = ca.sum2(price * total) * dt / WATT_HOUR_TO_JOULE
        capacity_cost = capacity * peak / 1000
        comfort_cost = (
            settings.comfort_penalty * (ca.sum1(ca.sum2(S_lo)) + ca.sum1(ca.sum2(S_hi)))
            + settings.comfort_quadratic * (ca.sumsqr(S_lo) + ca.sumsqr(S_hi))
        ) * hours
        limit_cost = settings.power_limit_penalty * ca.sum2(S_lim) * dt / WATT_HOUR_TO_JOULE
        ramp = ca.horzcat(U[:, 0] - u_previous, U[:, 1:] - U[:, :-1]) / 1000
        ramp_cost = settings.ramp_weight * ca.sumsqr(ramp)
        objective = energy_cost + capacity_cost + comfort_cost + limit_cost + ramp_cost
        opti.minimize(objective)
        opti.solver("ipopt", ipopt_options(settings.max_iterations, settings.tolerance, settings.verbose))

        self._opti = opti
        self._variables = {"X": X, "U": U, "S_lo": S_lo, "S_hi": S_hi, "S_lim": S_lim, "peak": peak}
        self._parameters = {
            "x0": x0, "D": D, "price": price, "lower": lower, "upper": upper, "limit": limit,
            "capacity": capacity, "u_previous": u_previous,
        }  # fmt: skip
        self._costs = {
            "energy": energy_cost, "capacity": capacity_cost, "comfort": comfort_cost,
            "power_limit": limit_cost, "ramp": ramp_cost,
        }  # fmt: skip

    def solve(self, x0: np.ndarray, forecast: pd.DataFrame, u_previous: np.ndarray | None = None) -> MPCSolution:
        """Optimal plan from the state ``x0`` with the disturbance ``forecast`` (``horizon`` rows)."""
        settings, model = self.settings, self.model
        N, dt = settings.horizon, settings.time_step
        if len(forecast) < N:
            raise ValueError(f"The forecast has {len(forecast)} rows, the horizon needs {N}.")
        forecast = forecast.iloc[:N]
        index = pd.DatetimeIndex(forecast.index)
        end_of_steps = forecast_index(index[0] + pd.Timedelta(seconds=dt), N, dt)
        signal = self.tariff.signal(index)
        bounds = self.comfort.bounds(model, end_of_steps)
        limits = np.where(np.isfinite(signal.power_limits), signal.power_limits, 10 * self.max_power.sum())

        opti, parameters = self._opti, self._parameters
        opti.set_value(parameters["x0"], np.asarray(x0, dtype=float))
        opti.set_value(parameters["D"], forecast[model.disturbance_names].to_numpy(dtype=float).T)
        opti.set_value(parameters["price"], signal.prices.reshape(1, -1))
        opti.set_value(parameters["lower"], bounds.lower)
        opti.set_value(parameters["upper"], bounds.upper)
        opti.set_value(parameters["limit"], limits.reshape(1, -1))
        opti.set_value(parameters["capacity"], signal.capacity_charge)
        opti.set_value(parameters["u_previous"], np.zeros(model.n_controls) if u_previous is None else u_previous)
        self._warm_start(np.asarray(x0, dtype=float))

        started = time.perf_counter()
        solution, success, status = _solve(opti)
        solve_time = time.perf_counter() - started
        variables = self._variables
        states = np.asarray(solution.value(variables["X"]), dtype=float).reshape(model.n_states, N + 1)
        controls = np.asarray(solution.value(variables["U"]), dtype=float).reshape(model.n_controls, N)
        self._previous = {
            name: np.asarray(solution.value(variable), dtype=float) for name, variable in variables.items()
        }
        costs = {name: float(solution.value(expression)) for name, expression in self._costs.items()}
        return MPCSolution(
            index=index,
            controls=pd.DataFrame(
                np.clip(controls.T, settings.min_power, self.max_power), index=index, columns=model.control_names
            ),
            states=pd.DataFrame(states.T, index=index.append(end_of_steps[-1:]), columns=model.state_names),
            lower=bounds.lower,
            upper=bounds.upper,
            prices=signal.prices,
            costs=costs,
            success=success,
            status=status,
            iterations=int(solution.stats().get("iter_count", 0)),
            solve_time=solve_time,
        )

    def _warm_start(self, x0: np.ndarray) -> None:
        variables = self._variables
        N = self.settings.horizon
        if self._previous is None:
            opti = self._opti
            opti.set_initial(variables["X"], np.tile(x0.reshape(-1, 1), (1, N + 1)))
            opti.set_initial(variables["U"], np.tile(0.3 * self.max_power.reshape(-1, 1), (1, N)))
            return
        previous = self._previous
        X = np.asarray(previous["X"]).reshape(self.model.n_states, N + 1)
        U = np.asarray(previous["U"]).reshape(self.model.n_controls, N)
        shifted_x = np.hstack([X[:, 1:], X[:, -1:]])
        shifted_x[:, 0] = x0
        self._opti.set_initial(variables["X"], shifted_x)
        self._opti.set_initial(variables["U"], np.hstack([U[:, 1:], U[:, -1:]]))
        for name in ("S_lo", "S_hi"):
            S = np.asarray(previous[name]).reshape(self.model.n_zones, N)
            self._opti.set_initial(variables[name], np.hstack([S[:, 1:], S[:, -1:]]))
        self._opti.set_initial(variables["peak"], float(previous["peak"]))


def _solve(opti: Any) -> tuple[Any, bool, str]:  # noqa: ANN401
    try:
        solution = opti.solve()
    except RuntimeError as error:
        return opti.debug, False, str(error).splitlines()[-1][:120]
    return solution, bool(solution.stats()["success"]), str(solution.stats()["return_status"])
