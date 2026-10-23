"""Economic model predictive control of a Trano ``FastSmoothBuildingsMPC`` building.

The controls are those of the systems of the building (valve positions, mixing valves, boiler part load ratio, heat
pump speed, or the heat of ideal emitters) and the energy is read from the **meters** of the model, states in kWh, so
that the efficiency of the systems (boiler, COP of the heat pump, fans, PV) is part of the optimisation. Over a
horizon of ``N`` steps from the time ``t0``, the controller solves with IPOPT

.. math::

    \\min\\; \\sum_k \\left(\\pi_k\\, b_k - \\pi^{in}_k (b_k - e_k) + \\pi^{fuel}_k f_k\\right)
    + c_\\mathrm{cap} \\frac{P_\\mathrm{peak}}{1000}
    + \\sum_k \\left(w_1 (s^{lo}_k + s^{hi}_k) + w_2 (s^{lo\\,2}_k + s^{hi\\,2}_k)\\right)\\frac{\\Delta t}{3600}
    + \\text{(power limit, move and effort penalties)}

where ``e_k`` is the net electricity of step ``k`` [kWh] (the electricity meters and the heat of the ideal
emitters, minus the production), ``b_k`` its smooth positive part (bought, the rest is sold at ``pi_in``) and
``f_k`` the fuel drawn. The ODE is collocated on Radau IIA points in each step
(:class:`trano_optimization.collocation.Collocated`, stable for the stiff water nodes), the comfort band
``T_lower - s_lo <= T_zone <= T_upper + s_hi`` is soft, and the demand response power limits and the capacity charge
apply to the electric power bought from the grid.

The problem is built once with the time, the forecasts, the prices and the bounds as parameters; each call to
:meth:`ModelPredictiveController.solve` updates them and warm starts IPOPT from the previous plan.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from trano_optimization.collocation import SOLVED, Collocated

from mpctrano.comfort import ComfortSpec
from mpctrano.forecast import forecast_index, seconds_of_year
from mpctrano.model import FastModel
from mpctrano.storage import Battery, ElectricVehicle, storage_schedule
from mpctrano.tariff import Tariff

WATT_HOUR_TO_JOULE = 3.6e6
NET_SMOOTHING = 1e-3
"""Smoothing [kWh] of the positive part of the net electricity of a step."""
HEAT_SMOOTHING = 1.0
"""Smoothing [W] of the heating and cooling parts of the heat flow of an ideal emitter."""


class MPCSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    horizon: int = Field(24, ge=1, description="Prediction horizon [steps]")
    time_step: float = Field(3600.0, gt=0, description="Control interval [s]")
    degree: int = Field(3, ge=1, le=9, description="Radau collocation points per step")
    comfort_penalty: float = Field(5.0, ge=0, description="Linear penalty of comfort violations [EUR/(K.h)]")
    comfort_quadratic: float = Field(1.0, ge=0, description="Quadratic penalty of comfort violations [EUR/(K2.h)]")
    power_limit_penalty: float = Field(
        50.0,
        ge=0,
        description="Penalty of the electricity above a demand response power limit, or below a power floor [EUR/kWh]",
    )
    move_weight: float = Field(
        1e-2, ge=0, description="Penalty of a full-range change of a control between steps [EUR]"
    )
    effort_weight: float = Field(
        1e-3,
        ge=0,
        description="Penalty of a control at full range over a step [EUR]: separates the controls that deliver the "
        "same heat (a valve and the boiler), whose flat valley IPOPT would otherwise crawl along",
    )
    fuel_price: float | None = Field(
        None, ge=0, description="Price of the fuel [EUR/kWh]; None: the energy price of the tariff"
    )
    feed_in_price: float | None = Field(
        None,
        description="Price of the electricity sold [EUR/kWh]; None: the export price of the tariff when it has one, "
        "its energy price otherwise",
    )
    warm_mu_init: float | None = Field(
        1e-3,
        gt=0,
        description="Initial barrier parameter of IPOPT once the controller has a previous plan: the warm start is "
        "close to the optimum, IPOPT's default (0.1) pushes it back into the interior and costs iterations; "
        "None: IPOPT's default",
    )
    default_heat_bound: float = Field(1e4, gt=0, description="Upper bound of a heat control [W] without a max")
    heat_cop: float = Field(
        1.0, gt=0, description="Heat per electricity of the ideal emitters (1: resistive, a heat pump: its COP)"
    )
    cooling_eer: float = Field(
        3.0, gt=0, description="Cooling per electricity of the ideal emitters used to cool (a chiller: its EER)"
    )
    ipopt: dict[str, float | int | str] = Field(
        default_factory=dict, description="IPOPT options added to trano's defaults (max_iter, tol, ...)"
    )
    batteries: tuple[Battery, ...] = Field((), description="Stationary batteries at the meter")
    vehicles: tuple[ElectricVehicle, ...] = Field((), description="Electric vehicles charged at the meter")

    @property
    def storages(self) -> tuple[Battery, ...]:
        return (*self.batteries, *self.vehicles)


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
    """Planned power of each step [W]: ``P_meter`` (electricity bought, what the tariff bills), ``P_export``
    (sold) and ``P_fuel``."""
    storage: pd.DataFrame | None = None
    """Planned storages, on the index of the states: ``<name>_soc_kwh`` at the start of each step (and after the
    last), ``<name>_charge_w`` and ``<name>_discharge_w`` over each step (NaN on the last row)."""

    @property
    def first_control(self) -> np.ndarray:
        return self.controls.iloc[0].to_numpy(dtype=float)

    @property
    def total_cost(self) -> float:
        return float(sum(self.costs.values()))

    @property
    def economic_cost(self) -> float:
        """The money of the plan [EUR]: electricity, fuel and capacity, without the penalties."""
        return self.costs["electricity"] + self.costs["fuel"] + self.costs["capacity"]

    @property
    def comfort_temperatures(self) -> pd.DataFrame:
        """Zone temperatures at the end of each step (the ones bounded by the comfort band)."""
        return self.states.iloc[1:]


class ModelPredictiveController:
    """Economic MPC of a ``FastSmoothBuildingsMPC`` model, see the module documentation for the problem."""

    def __init__(self, model: FastModel, settings: MPCSettings, comfort: ComfortSpec, tariff: Tariff) -> None:
        self.model = model
        self.settings = settings
        self.comfort = comfort
        self.tariff = tariff
        collocation = model.collocation_settings(
            settings.time_step,
            degree=settings.degree,
            default_heat_bound=settings.default_heat_bound,
            ipopt=settings.ipopt,
        )
        self.collocated = Collocated(model.model, model.roles, collocation, model.parameters)
        self._cold_settings = collocation
        self._warm_settings = (
            collocation
            if settings.warm_mu_init is None
            else collocation.model_copy(update={"ipopt": {"mu_init": settings.warm_mu_init, **settings.ipopt}})
        )
        self._previous: dict[str, np.ndarray] | None = None
        self.storage_state: dict[str, float] = {
            storage.name: storage.initial_energy_kwh for storage in settings.storages
        }
        """Energy in each storage [kWh], carried from one solve to the next by :meth:`advance`."""
        self.energy: pd.DataFrame | None = None
        """Electricity bought and net, fuel [kWh] and electric power [W] of each step of the last plan."""
        self._build()

    @property
    def lower(self) -> np.ndarray:
        return np.asarray(self.collocated.lower, dtype=float)

    @property
    def upper(self) -> np.ndarray:
        return np.asarray(self.collocated.upper, dtype=float)

    def _meter_rows(self, carrier: str) -> list[int]:
        return [self.model.state_names.index(name) for name, kind in self.model.meters.items() if kind == carrier]

    def _build(self) -> None:  # noqa: PLR0915 - one optimisation problem, written in one place
        model, settings, collocated = self.model, self.settings, self.collocated
        N, dt, degree = settings.horizon, settings.time_step, settings.degree
        n_x, n_u, n_d, n_c = model.n_states, model.n_controls, model.n_disturbances, model.n_zones
        hours = dt / 3600
        opti = ca.Opti()
        z = opti.variable(n_x, N + 1)  # states shifted by the offsets of collocated (temperatures in degC)
        zc = opti.variable(n_x * degree, N)
        v = opti.variable(n_u, N)  # controls scaled to [0, 1]
        s_lo = opti.variable(n_c, N)
        s_hi = opti.variable(n_c, N)
        s_lim = opti.variable(1, N)
        s_floor = opti.variable(1, N)
        peak = opti.variable()

        t0 = opti.parameter()
        x0 = opti.parameter(n_x)
        disturbances = opti.parameter(n_d, N)
        price = opti.parameter(1, N)
        feed_in = opti.parameter(1, N)
        fuel_price = opti.parameter(1, N)
        lower = opti.parameter(n_c, N)
        upper = opti.parameter(n_c, N)
        limit = opti.parameter(1, N)
        floor = opti.parameter(1, N)
        capacity = opti.parameter()
        peak_so_far = opti.parameter()  # the peak already billed this month [W]
        v_previous = opti.parameter(n_u)
        first_move = opti.parameter()  # 1 when v_previous is the control applied before the horizon, else 0

        offset = ca.DM(collocated.offset)
        for k in range(N):
            residual, end = collocated._interval(t0 + k * dt, z[:, k], zc[:, k], v[:, k], disturbances[:, k])
            opti.subject_to(residual == 0)
            opti.subject_to(z[:, k + 1] == end)
        opti.subject_to(z[:, 0] == x0 - offset)
        opti.subject_to(opti.bounded(0, ca.vec(v), 1))
        collocated.bound_temperatures(opti, z[:, 1:])
        for slack in (s_lo, s_hi, s_lim, s_floor):
            opti.subject_to(ca.vec(slack) >= 0)
        comfort = z[model.comfort_indices, 1:] + offset[model.comfort_indices] @ ca.DM.ones(1, N)
        opti.subject_to(ca.vec(comfort + s_lo) >= ca.vec(lower))
        opti.subject_to(ca.vec(comfort - s_hi) <= ca.vec(upper))

        def drawn(rows: list[int]) -> ca.MX:
            """Energy metered over each step [kWh], (1, N)."""
            return ca.sum1(z[rows, 1:] - z[rows, :-1]) if rows else ca.MX.zeros(1, N)

        heat_rows = [i for i, name in enumerate(model.control_names) if name in model.heat_controls]
        heat_energy: Any = ca.MX.zeros(1, N)
        for i in heat_rows:  # the heat of an ideal emitter [W], bought at the electricity price through the COP / EER
            flow = self.lower[i] + (self.upper[i] - self.lower[i]) * v[i, :]
            heating = 0.5 * (flow + ca.sqrt(flow**2 + HEAT_SMOOTHING**2))
            heat_energy = heat_energy + (heating / settings.heat_cop + (heating - flow) / settings.cooling_eer) * (
                dt / WATT_HOUR_TO_JOULE
            )
        net = drawn(self._meter_rows("electricity")) + heat_energy - drawn(self._meter_rows("electricity_production"))
        # ---- electrical storages at the meter: batteries charge and discharge, vehicles charge while plugged in
        storages = settings.storages
        n_s = len(storages)
        storage_cost: Any = ca.MX(0)
        price_mean = opti.parameter()
        if n_s:
            p_ch, p_dis, s_ev = (opti.variable(n_s, N) for _ in range(3))
            soc = opti.variable(n_s, N + 1)
            soc0, plugged, departs = opti.parameter(n_s), opti.parameter(n_s, N), opti.parameter(n_s, N)
            per_kwh = dt / WATT_HOUR_TO_JOULE
            for i, storage in enumerate(storages):
                eta = storage.one_way_efficiency
                low, high = storage.energy_bounds_kwh
                discharge = 0.0 if isinstance(storage, ElectricVehicle) else storage.max_discharge_w
                opti.subject_to(ca.vec(p_ch[i, :]) >= 0)
                opti.subject_to(ca.vec(p_ch[i, :]) <= storage.max_charge_w * ca.vec(plugged[i, :]))
                opti.subject_to(opti.bounded(0, ca.vec(p_dis[i, :]), discharge))
                opti.subject_to(opti.bounded(low, ca.vec(soc[i, 1:]), high))
                opti.subject_to(soc[i, 0] == soc0[i])
                opti.subject_to(
                    soc[i, 1:]
                    == soc[i, :-1]
                    + (eta * p_ch[i, :] - p_dis[i, :] / eta) * per_kwh
                    - departs[i, :] * storage.driven_kwh
                )
                opti.subject_to(ca.vec(s_ev[i, :]) >= 0)
                if isinstance(storage, ElectricVehicle):
                    target = departs[i, :] * storage.departure_energy_kwh
                    opti.subject_to(ca.vec(soc[i, :-1] + s_ev[i, :]) >= ca.vec(target))
                    storage_cost = storage_cost + storage.missing_energy_price * ca.sum2(s_ev[i, :])
                else:  # the energy stored at the end is worth the mean price (no emptying at the horizon end)
                    storage_cost = storage_cost - price_mean * eta * (soc[i, N] - soc[i, 0])
                net = net + (p_ch[i, :] - p_dis[i, :]) * per_kwh
        bought = 0.5 * (net + ca.sqrt(net**2 + NET_SMOOTHING**2))
        fuel = drawn(self._meter_rows("fuel"))
        power = bought * WATT_HOUR_TO_JOULE / dt  # electric power bought from the grid [W]
        opti.subject_to(ca.vec(power - s_lim) <= ca.vec(limit))
        opti.subject_to(ca.vec(power + s_floor) >= ca.vec(floor))
        opti.subject_to(ca.vec(power) <= peak)
        opti.subject_to(peak >= peak_so_far)

        electricity_cost = ca.sum2(price * bought - feed_in * (bought - net))
        fuel_cost = ca.sum2(fuel_price * fuel)
        capacity_cost = capacity * peak / 1000
        comfort_cost = (
            settings.comfort_penalty * (ca.sum1(ca.sum2(s_lo)) + ca.sum1(ca.sum2(s_hi)))
            + settings.comfort_quadratic * (ca.sumsqr(s_lo) + ca.sumsqr(s_hi))
        ) * hours
        limit_cost = settings.power_limit_penalty * ca.sum2(s_lim) * dt / WATT_HOUR_TO_JOULE
        floor_cost = settings.power_limit_penalty * ca.sum2(s_floor) * dt / WATT_HOUR_TO_JOULE
        moves = ca.sumsqr(v[:, 1:] - v[:, :-1]) if N > 1 else ca.MX(0)
        move_cost = settings.move_weight * (moves + first_move * ca.sumsqr(v[:, 0] - v_previous))
        effort_cost = settings.effort_weight * ca.sumsqr(v)
        opti.minimize(
            electricity_cost
            + fuel_cost
            + capacity_cost
            + comfort_cost
            + limit_cost
            + floor_cost
            + move_cost
            + effort_cost
            + storage_cost
        )

        self._opti = opti
        self._variables = {
            "z": z, "zc": zc, "v": v, "s_lo": s_lo, "s_hi": s_hi, "s_lim": s_lim, "s_floor": s_floor, "peak": peak,
        }  # fmt: skip
        self._parameters = {
            "t0": t0, "x0": x0, "D": disturbances, "price": price, "feed_in": feed_in, "fuel_price": fuel_price,
            "lower": lower, "upper": upper, "limit": limit, "floor": floor, "capacity": capacity,
            "peak_so_far": peak_so_far, "v_previous": v_previous, "first_move": first_move,
            "price_mean": price_mean,
        }  # fmt: skip
        if n_s:
            self._variables |= {"p_ch": p_ch, "p_dis": p_dis, "soc": soc, "s_ev": s_ev}
            self._parameters |= {"soc0": soc0, "plugged": plugged, "departs": departs}
        self._costs = {
            "electricity": electricity_cost, "fuel": fuel_cost, "capacity": capacity_cost, "comfort": comfort_cost,
            "power_limit": limit_cost, "power_floor": floor_cost, "moves": move_cost, "effort": effort_cost,
            "storage": storage_cost,
        }  # fmt: skip
        self._energy = {"electricity_bought": bought, "electricity_net": net, "fuel": fuel, "power": power}

    def solve(
        self,
        x0: np.ndarray,
        forecast: pd.DataFrame,
        u_previous: np.ndarray | None = None,
        power_limits: np.ndarray | None = None,
        power_floors: np.ndarray | None = None,
        peak_so_far: float = 0.0,
    ) -> MPCSolution:
        """Optimal plan from the state ``x0`` with the disturbance ``forecast`` (``horizon`` rows from ``t0``).

        ``u_previous`` is the control applied before the horizon (in the units of the model), penalised as a move.
        ``power_limits`` and ``power_floors`` (``horizon`` values [W]) cap and floor the electricity bought on top of
        the demand response events of the tariff, softly; ``peak_so_far`` [W] is the peak already billed by a
        capacity tariff, below which the peak of the plan costs nothing more.
        """
        settings, model, collocated = self.settings, self.model, self.collocated
        N, dt = settings.horizon, settings.time_step
        if len(forecast) < N:
            raise ValueError(f"The forecast has {len(forecast)} rows, the horizon needs {N}.")
        forecast = forecast.iloc[:N]
        index = pd.DatetimeIndex(forecast.index)
        end_of_steps = forecast_index(index[0] + pd.Timedelta(seconds=dt), N, dt)
        start_time = float(seconds_of_year(index[:1])[0])
        signal = self.tariff.signal(index)
        bounds = self.comfort.bounds(model, end_of_steps)
        limits = signal.power_limits if power_limits is None else np.minimum(signal.power_limits, power_limits)
        limits = np.where(np.isfinite(limits), limits, 1e9)
        floors = signal.floors if power_floors is None else np.maximum(signal.floors, power_floors)
        if settings.feed_in_price is not None:
            feed_in = np.full(N, settings.feed_in_price)
        else:
            feed_in = signal.export if self.tariff.has_export_price else signal.prices
        disturbances = forecast[model.disturbance_names].to_numpy(dtype=float).T
        x0 = np.asarray(x0, dtype=float)

        def row(values: np.ndarray | float | None) -> np.ndarray:
            return np.broadcast_to(signal.prices if values is None else values, (N,)).reshape(1, -1)

        parameter_values: dict[str, float | np.ndarray] = {
            "t0": start_time,
            "x0": x0,
            "D": disturbances,
            "price": row(signal.prices),
            "feed_in": row(feed_in),
            "fuel_price": row(settings.fuel_price),
            "lower": bounds.lower,
            "upper": bounds.upper,
            "limit": limits.reshape(1, -1),
            "floor": floors.reshape(1, -1),
            "capacity": signal.capacity_charge,
            "peak_so_far": float(peak_so_far),
            "price_mean": float(np.mean(signal.prices)),
            "v_previous": np.zeros(model.n_controls)
            if u_previous is None
            else collocated.scaled(np.asarray(u_previous, dtype=float).reshape(-1, 1)).ravel(),
            "first_move": 0.0 if u_previous is None else 1.0,
        }
        parameter_values |= self._storage_parameters(index)
        for name, parameter_value in parameter_values.items():
            self._opti.set_value(self._parameters[name], parameter_value)
        self._warm_start(x0, disturbances, start_time)

        collocated.settings = self._cold_settings if self._previous is None else self._warm_settings
        started = time.perf_counter()
        value, status = collocated.solve_nlp(self._opti)
        solve_time = time.perf_counter() - started
        stats = self._opti.stats()
        solution = {name: np.asarray(value(variable), dtype=float) for name, variable in self._variables.items()}
        success = status in SOLVED
        self._previous = solution if success else None  # a failed plan is a poor start: start cold next time
        states = solution["z"].reshape(model.n_states, N + 1) + collocated.offset[:, None]
        controls = collocated.unscaled(np.clip(solution["v"].reshape(model.n_controls, N), 0.0, 1.0))
        costs = {name: float(value(expression)) for name, expression in self._costs.items()}
        self.energy = pd.DataFrame(
            {name: np.asarray(value(expression), dtype=float).ravel() for name, expression in self._energy.items()},
            index=index,
        )
        per_kwh = WATT_HOUR_TO_JOULE / dt
        meter = pd.DataFrame(
            {
                "P_meter": self.energy["power"],
                "P_export": (self.energy["electricity_bought"] - self.energy["electricity_net"]).clip(lower=0)
                * per_kwh,
                "P_fuel": self.energy["fuel"] * per_kwh,
            },
            index=index,
        )
        storage = self._storage_frame(solution, pd.DatetimeIndex(index.append(end_of_steps[-1:])))
        return MPCSolution(
            index=index,
            controls=pd.DataFrame(controls.T, index=index, columns=model.control_names),
            states=pd.DataFrame(states.T, index=index.append(end_of_steps[-1:]), columns=model.state_names),
            lower=bounds.lower,
            upper=bounds.upper,
            prices=signal.prices,
            costs=costs,
            success=success,
            status=status,
            iterations=int(stats.get("iter_count", 0)),
            solve_time=solve_time,
            meter=meter,
            storage=storage,
        )

    def _storage_parameters(self, index: pd.DatetimeIndex) -> dict[str, float | np.ndarray]:
        """The state of the storages, clipped to their bounds, and the schedule of the vehicles over ``index``; the
        state of charge variables start at the current state."""
        storages = self.settings.storages
        if not storages:
            return {}
        soc0 = np.array([np.clip(self.storage_state[s.name], *s.energy_bounds_kwh) for s in storages], dtype=float)
        plugged, departs = storage_schedule(storages, index, self.settings.time_step)
        self._opti.set_initial(self._variables["soc"], np.repeat(soc0[:, None], len(index) + 1, axis=1))
        return {"soc0": soc0, "plugged": plugged, "departs": departs}

    def _storage_frame(self, solution: dict[str, np.ndarray], index: pd.DatetimeIndex) -> pd.DataFrame | None:
        storages = self.settings.storages
        if not storages:
            return None
        N = len(index) - 1
        columns: dict[str, np.ndarray] = {}
        for i, storage in enumerate(storages):
            columns[f"{storage.name}_soc_kwh"] = solution["soc"].reshape(len(storages), N + 1)[i]
            for kind, name in (("ch", "charge"), ("dis", "discharge")):
                powers = solution[f"p_{kind}"].reshape(len(storages), N)[i]
                columns[f"{storage.name}_{name}_w"] = np.append(powers, np.nan)
        return pd.DataFrame(columns, index=index)

    def advance(self, solution: MPCSolution) -> float:
        """Apply the first step of ``solution`` to the storages (their model is exact): their state becomes the one
        after that step; returns the energy they took from the meter over it [kWh] (negative when they fed it)."""
        if solution.storage is None:
            return 0.0
        exchange = 0.0
        for storage in self.settings.storages:
            self.storage_state[storage.name] = float(solution.storage[f"{storage.name}_soc_kwh"].iloc[1])
            first = solution.storage.iloc[0]
            exchange += float(first[f"{storage.name}_charge_w"] - first[f"{storage.name}_discharge_w"])
        return exchange * self.settings.time_step / WATT_HOUR_TO_JOULE

    def full_load_power(self, x0: np.ndarray, forecast: pd.DataFrame) -> float:
        """Largest electric power [W] the building draws over the horizon with every control at its upper bound: the
        bound of what it can absorb (a simulation, no optimisation)."""
        settings, model, collocated = self.settings, self.model, self.collocated
        N, dt = settings.horizon, settings.time_step
        index = pd.DatetimeIndex(forecast.index[:N])
        disturbances = forecast.iloc[:N][model.disturbance_names].to_numpy(dtype=float).T
        controls = np.tile(collocated.unscaled(np.ones((model.n_controls, 1))), (1, N))
        states = model.simulate(x0, controls, disturbances, dt, float(seconds_of_year(index[:1])[0]))
        rows = self._meter_rows("electricity")
        drawn = np.diff(states[rows], axis=1).sum(axis=0) if rows else np.zeros(N)
        heat = sum(controls[model.control_names.index(name)] for name in model.heat_controls) * dt / WATT_HOUR_TO_JOULE
        return float(np.max((drawn + heat / settings.heat_cop) * WATT_HOUR_TO_JOULE / dt))

    def _warm_start(self, x0: np.ndarray, disturbances: np.ndarray, start_time: float) -> None:
        """Initial guess: a simulation of the model (CVODES) from ``x0``, consistent with the dynamics, under the
        previous plan shifted by one step (mid-range controls at the first solve).

        Shifting the previous states instead starts IPOPT off the dynamics at the first step (the measured state
        differs from the predicted one) and makes it slower than a cold start on the stiff water nodes.
        """
        collocated, variables, opti = self.collocated, self._variables, self._opti
        N, degree, n_u = self.settings.horizon, self.settings.degree, self.model.n_controls
        if self._previous is None:
            v = np.full((n_u, N), 0.5)
        else:
            v = _shift(self._previous["v"].reshape(n_u, N))
            for name in ("s_lo", "s_hi"):
                opti.set_initial(variables[name], _shift(self._previous[name].reshape(self.model.n_zones, N)))
            opti.set_initial(variables["peak"], float(self._previous["peak"]))
        z = collocated.simulate(x0, collocated.unscaled(v), disturbances, start_time) - collocated.offset[:, None]
        opti.set_initial(variables["z"], z)
        opti.set_initial(variables["zc"], np.tile(z[:, 1:], (degree, 1)))
        opti.set_initial(variables["v"], v)


def _shift(values: Any) -> np.ndarray:  # noqa: ANN401
    """Drop the first column and repeat the last one."""
    values = np.asarray(values, dtype=float)
    return np.hstack([values[:, 1:], values[:, -1:]])
