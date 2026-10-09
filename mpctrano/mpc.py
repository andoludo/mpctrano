"""Economic model predictive control of the inputs of a Trano building model.

Every step, the controller solves with IPOPT, over a receding horizon of ``N`` steps:

.. math::

    \\min\\; \\sum_k \\pi_k \\, \\frac{P^{meter}_k \\Delta t}{3.6\\cdot 10^6}
    - \\sum_k \\pi^{exp}_k \\, \\frac{P^{exp}_k \\Delta t}{3.6\\cdot 10^6}
    + \\pi^{gas} \\sum_k \\frac{P^{gas}_k \\Delta t}{3.6\\cdot 10^6}
    + c_\\mathrm{cap} \\frac{P_\\mathrm{peak}}{1000}
    + \\text{comfort, power limit, ramp and vehicle penalties}

where ``P_meter = P_heat + P_import`` is what the tariff bills: the ideal thermal inputs of the
zones without system (``P_heat``, costed at the tariff price as before) plus the electricity
imported from the grid. The import and the export are split from the net power at the meter
(``consumption - battery discharge - PV generation``) with two non-negative variables, which is
exact as long as the import price is not lower than the export price; the export is
bounded by the on-site generation and discharge (the split is skipped when there is none). Gas
boilers are costed with their efficiency at ``gas_price``.

Constraints: the ODE of the model, the bounds of every control input (design heating power of a
thermal input, compressor, charger and tank limits from the interface), the compressor limit
shared by the zones of one heat pump, the state bounds (temperatures, state of charge), the soft
comfort band of each zone, the soft temperature band of each hot water tank, the soft
building-level power limit of the demand response events, the plug-in windows and the departure
energy targets of the electric vehicles (soft).

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
from mpctrano.meter import Meter
from mpctrano.model import RCModel
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
    max_power: dict[str, float] | None = Field(None, description="Maximum heating input per zone [W]")
    min_power: float = Field(0.0, description="Minimum thermal heating power [W] (negative allows ideal cooling)")
    gas_price: float = Field(0.08, ge=0, description="Price of the fuel of the boilers [EUR/kWh]")
    vehicle_penalty: float = Field(
        2.0, ge=0, description="Penalty of the energy missing in a vehicle battery at departure [EUR/kWh]"
    )
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
    meter: pd.DataFrame | None = None
    """Planned power at the meter per step [W]: ``P_import``, ``P_export``, ``P_meter``, ``P_gas``."""

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
    """Economic MPC of a Trano building model, see the module documentation for the problem."""

    def __init__(self, model: RCModel, settings: MPCSettings, comfort: ComfortSpec, tariff: Tariff) -> None:
        self.model = model
        self.settings = settings
        self.comfort = comfort
        self.tariff = tariff
        self.meter = Meter(model)
        self.lower_bounds, self.upper_bounds = self._control_bounds()
        self.dynamics = model.discrete(settings.time_step, settings.substeps)
        self._previous: dict[str, np.ndarray] | None = None
        self._build()

    @property
    def max_power(self) -> np.ndarray:
        """Upper bound of the heating input of each zone [W] (zone order)."""
        return np.array([self.upper_bounds[index] for index in self.model.zone_heating_indices], dtype=float)

    def _control_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        model, settings = self.model, self.settings
        lower, upper = model.control_bounds()
        if settings.max_power is not None:
            missing = set(model.zone_names) - set(settings.max_power)
            if missing:
                raise ValueError(f"max_power is missing the zones {sorted(missing)}.")
            for zone, index in zip(model.zone_names, model.zone_heating_indices, strict=True):
                upper[index] = settings.max_power[zone]
        for index in model.heat_control_indices:
            lower[index] = settings.min_power
        return lower, upper

    def _build(self) -> None:  # noqa: PLR0915
        model, settings = self.model, self.settings
        N, dt = settings.horizon, settings.time_step
        hours = dt / 3600
        opti = ca.Opti()
        X = opti.variable(model.n_states, N + 1)
        U = opti.variable(model.n_controls, N)
        S_lo = opti.variable(model.n_zones, N)
        S_hi = opti.variable(model.n_zones, N)
        S_lim = opti.variable(1, N)
        S_floor = opti.variable(1, N)
        peak = opti.variable()
        variables: dict[str, Any] = {
            "X": X, "U": U, "S_lo": S_lo, "S_hi": S_hi, "S_lim": S_lim, "S_floor": S_floor, "peak": peak,
        }  # fmt: skip

        x0 = opti.parameter(model.n_states)
        D = opti.parameter(model.n_disturbances, N)
        price = opti.parameter(1, N)
        export_price = opti.parameter(1, N)
        lower = opti.parameter(model.n_zones, N)
        upper = opti.parameter(model.n_zones, N)
        limit = opti.parameter(1, N)
        floor = opti.parameter(1, N)
        capacity = opti.parameter()
        peak_so_far = opti.parameter()
        u_previous = opti.parameter(model.n_controls)
        parameters: dict[str, Any] = {
            "x0": x0, "D": D, "price": price, "export_price": export_price, "lower": lower, "upper": upper,
            "limit": limit, "floor": floor, "capacity": capacity, "peak_so_far": peak_so_far,
            "u_previous": u_previous,
        }  # fmt: skip

        # ---- dynamics and bounds
        opti.subject_to(X[:, 0] == x0)
        for k in range(N):
            opti.subject_to(X[:, k + 1] == self.dynamics(X[:, k], U[:, k], D[:, k], model.parameters))
        state_lower, state_upper = model.state_bounds()
        opti.subject_to(opti.bounded(np.tile(state_lower, N), ca.vec(X[:, 1:]), np.tile(state_upper, N)))
        opti.subject_to(opti.bounded(np.tile(self.lower_bounds, N), ca.vec(U), np.tile(self.upper_bounds, N)))
        for group in model.control_groups:
            opti.subject_to(ca.vec(ca.sum1(U[group.indices, :])) <= group.max_power)
        opti.subject_to(ca.vec(S_lo) >= 0)
        opti.subject_to(ca.vec(S_hi) >= 0)
        opti.subject_to(ca.vec(S_lim) >= 0)
        opti.subject_to(ca.vec(S_floor) >= 0)

        # ---- comfort of the zones and of the hot water tanks
        comfort = X[model.comfort_indices, 1:]
        opti.subject_to(ca.vec(comfort + S_lo) >= ca.vec(lower))
        opti.subject_to(ca.vec(comfort - S_hi) <= ca.vec(upper))
        comfort_cost = (
            settings.comfort_penalty * (ca.sum1(ca.sum2(S_lo)) + ca.sum1(ca.sum2(S_hi)))
            + settings.comfort_quadratic * (ca.sumsqr(S_lo) + ca.sumsqr(S_hi))
        ) * hours
        if model.storage_bands:
            T_lo = opti.variable(len(model.storage_bands), N)
            T_hi = opti.variable(len(model.storage_bands), N)
            variables |= {"T_lo": T_lo, "T_hi": T_hi}
            opti.subject_to(ca.vec(T_lo) >= 0)
            opti.subject_to(ca.vec(T_hi) >= 0)
            for row, band in enumerate(model.storage_bands):
                tank = X[band.state_index, 1:]
                opti.subject_to(ca.vec(tank + T_lo[row, :]) >= band.lower)
                opti.subject_to(ca.vec(tank - T_hi[row, :]) <= band.upper)
            comfort_cost += (
                settings.comfort_penalty * (ca.sum1(ca.sum2(T_lo)) + ca.sum1(ca.sum2(T_hi)))
                + settings.comfort_quadratic * (ca.sumsqr(T_lo) + ca.sumsqr(T_hi))
            ) * hours

        # ---- the meter: import / export split, metered power, gas
        heat = self.meter.heat(U)
        export_cost: Any = ca.DM(0)
        exchange: dict[str, Any] = {}
        if model.has_meter and model.has_sources:
            # Import / export split, the export bounded by the on-site sources (PV, battery discharge):
            # without the bound a power floor could be met by importing and exporting at the same time.
            P_imp = opti.variable(1, N)
            P_exp = opti.variable(1, N)
            variables |= {"P_imp": P_imp, "P_exp": P_exp}
            opti.subject_to(ca.vec(P_imp) >= 0)
            opti.subject_to(ca.vec(P_exp) >= 0)
            opti.subject_to(ca.vec(P_imp - P_exp) == ca.vec(self.meter.net(U, D)))
            opti.subject_to(ca.vec(P_exp) <= ca.vec(self.meter.discharge(U) + self.meter.generation(D)))
            metered = heat + P_imp
            export_cost = -ca.sum2(export_price * P_exp) * dt / WATT_HOUR_TO_JOULE
            exchange = {"P_import": P_imp, "P_export": P_exp}
        elif model.has_meter:
            # Nothing can feed the grid: the net power is the import (the inputs are non-negative), and a
            # split would only add a degenerate pair of variables pinned at zero.
            metered = heat + self.meter.net(U, D)
            exchange = {"P_import": self.meter.net(U, D)}
        else:
            metered = heat
        self._exchange = exchange
        gas_cost = settings.gas_price * ca.sum2(self.meter.gas(U)) * dt / WATT_HOUR_TO_JOULE
        opti.subject_to(ca.vec(metered - S_lim) <= ca.vec(limit))
        opti.subject_to(ca.vec(metered + S_floor) >= ca.vec(floor))
        opti.subject_to(ca.vec(metered) <= peak)
        opti.subject_to(peak >= peak_so_far)

        # ---- electric vehicles: plug-in windows and departure targets
        vehicle_cost: Any = ca.DM(0)
        if model.vehicles:
            plugged = opti.parameter(len(model.vehicles), N)
            departure = opti.parameter(len(model.vehicles), N)
            V = opti.variable(len(model.vehicles), N)
            parameters |= {"plugged": plugged, "departure": departure}
            variables["V"] = V
            opti.subject_to(ca.vec(V) >= 0)
            for row, vehicle in enumerate(model.vehicles):
                charging = U[vehicle.control_index, :]
                opti.subject_to(ca.vec(charging) <= ca.vec(plugged[row, :] * self.upper_bounds[vehicle.control_index]))
                energy = X[vehicle.state_index, 1:]
                opti.subject_to(ca.vec(energy + V[row, :]) >= ca.vec(departure[row, :] * vehicle.target_energy))
            vehicle_cost = settings.vehicle_penalty * ca.sum1(ca.sum2(V)) / WATT_HOUR_TO_JOULE

        energy_cost = ca.sum2(price * metered) * dt / WATT_HOUR_TO_JOULE
        capacity_cost = capacity * peak / 1000
        limit_cost = settings.power_limit_penalty * (ca.sum2(S_lim) + ca.sum2(S_floor)) * dt / WATT_HOUR_TO_JOULE
        ramp = ca.horzcat(U[:, 0] - u_previous, U[:, 1:] - U[:, :-1]) / 1000
        ramp_cost = settings.ramp_weight * ca.sumsqr(ramp)
        objective = (
            energy_cost + export_cost + gas_cost + capacity_cost + comfort_cost + limit_cost + ramp_cost + vehicle_cost
        )
        opti.minimize(objective)
        opti.solver("ipopt", ipopt_options(settings.max_iterations, settings.tolerance, settings.verbose))

        self._opti = opti
        self._variables = variables
        self._parameters = parameters
        self._metered = metered
        self._costs = {
            "energy": energy_cost, "export": export_cost, "gas": gas_cost, "capacity": capacity_cost,
            "comfort": comfort_cost, "power_limit": limit_cost, "ramp": ramp_cost, "vehicle": vehicle_cost,
        }  # fmt: skip

    def solve(
        self,
        x0: np.ndarray,
        forecast: pd.DataFrame,
        u_previous: np.ndarray | None = None,
        *,
        power_limits: np.ndarray | None = None,
        power_floors: np.ndarray | None = None,
        peak_so_far: float = 0.0,
    ) -> MPCSolution:
        """Optimal plan from the state ``x0`` with the disturbance ``forecast`` (``horizon`` rows).

        ``power_limits`` and ``power_floors`` [W] (``horizon`` values) override the demand response
        events of the tariff for this solve (flexibility envelopes); ``peak_so_far`` [W] is the
        billed peak of the current month, which a capacity charge applies to together with the
        peak of the horizon.
        """
        settings, model = self.settings, self.model
        N, dt = settings.horizon, settings.time_step
        if len(forecast) < N:
            raise ValueError(f"The forecast has {len(forecast)} rows, the horizon needs {N}.")
        forecast = forecast.iloc[:N]
        index = pd.DatetimeIndex(forecast.index)
        end_of_steps = forecast_index(index[0] + pd.Timedelta(seconds=dt), N, dt)
        signal = self.tariff.signal(index)
        bounds = self.comfort.bounds(model, end_of_steps)
        unbounded = 10 * self.upper_bounds.sum()
        raw_limits = signal.power_limits if power_limits is None else np.asarray(power_limits, dtype=float)
        limits = np.where(np.isfinite(raw_limits), raw_limits, unbounded)
        floors = signal.floors if power_floors is None else np.asarray(power_floors, dtype=float)

        opti, parameters = self._opti, self._parameters
        opti.set_value(parameters["x0"], np.asarray(x0, dtype=float))
        opti.set_value(parameters["D"], forecast[model.disturbance_names].to_numpy(dtype=float).T)
        opti.set_value(parameters["price"], signal.prices.reshape(1, -1))
        opti.set_value(parameters["export_price"], signal.export.reshape(1, -1))
        opti.set_value(parameters["lower"], bounds.lower)
        opti.set_value(parameters["upper"], bounds.upper)
        opti.set_value(parameters["limit"], limits.reshape(1, -1))
        opti.set_value(parameters["floor"], floors.reshape(1, -1))
        opti.set_value(parameters["capacity"], signal.capacity_charge)
        opti.set_value(parameters["peak_so_far"], float(peak_so_far))
        opti.set_value(parameters["u_previous"], np.zeros(model.n_controls) if u_previous is None else u_previous)
        if model.vehicles:
            opti.set_value(
                parameters["plugged"], np.vstack([vehicle.plugged_in(index) for vehicle in model.vehicles]) * 1.0
            )
            opti.set_value(
                parameters["departure"],
                np.vstack([vehicle.departures(index, dt) for vehicle in model.vehicles]) * 1.0,
            )
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
        meter = pd.DataFrame(
            {
                "P_meter": np.asarray(solution.value(self._metered), dtype=float).reshape(-1),
                "P_import": np.asarray(solution.value(self._exchange["P_import"]), dtype=float).reshape(-1)
                if "P_import" in self._exchange
                else np.zeros(N),
                "P_export": np.asarray(solution.value(self._exchange["P_export"]), dtype=float).reshape(-1)
                if "P_export" in self._exchange
                else np.zeros(N),
                "P_gas": np.asarray(solution.value(self.meter.gas(variables["U"])), dtype=float).reshape(-1),
            },
            index=index,
        )
        return MPCSolution(
            index=index,
            controls=pd.DataFrame(
                np.clip(controls.T, self.lower_bounds, self.upper_bounds), index=index, columns=model.control_names
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
            meter=meter,
        )

    def _warm_start(self, x0: np.ndarray) -> None:
        variables = self._variables
        N = self.settings.horizon
        opti = self._opti
        if self._previous is None:
            opti.set_initial(variables["X"], np.tile(x0.reshape(-1, 1), (1, N + 1)))
            opti.set_initial(variables["U"], np.tile(0.3 * self.upper_bounds.reshape(-1, 1), (1, N)))
            return
        for name, previous in self._previous.items():
            variable = variables[name]
            if name == "peak":
                opti.set_initial(variable, float(previous))
                continue
            rows, columns = variable.shape
            values = np.asarray(previous).reshape(rows, columns)
            shifted = np.hstack([values[:, 1:], values[:, -1:]])
            if name == "X":
                shifted[:, 0] = x0
            opti.set_initial(variable, shifted)


def _solve(opti: Any) -> tuple[Any, bool, str]:  # noqa: ANN401
    try:
        solution = opti.solve()
    except RuntimeError as error:
        return opti.debug, False, str(error).splitlines()[-1][:120]
    return solution, bool(solution.stats()["success"]), str(solution.stats()["return_status"])
