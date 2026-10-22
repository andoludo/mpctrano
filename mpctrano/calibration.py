"""Calibration of a FastModel on monitoring data.

A FastModel has hundreds of parameters, most of them settled by the YAML (areas, orientations, optical fits) or
not identifiable from a few zone temperatures. :class:`Calibrator` adjusts **groups** of parameters instead,
each by one decision variable:

* a ``scale`` group multiplies the values Trano computed by a common factor ``exp(z)`` (all the conductances of the
  envelope, all its capacities, ...), within ``[lower, upper]`` and regularised towards 1;
* a ``value`` group gives every parameter it matches one common value within ``[lower, upper]``, regularised towards
  their mean Trano value (the infiltration rate, which Trano leaves at 0).

The problem is solved with IPOPT, by one of two methods (:attr:`CalibrationSettings.method`):

* **collocation** (models with control inputs): the Radau collocation of the model over the whole data set, the
  parameters symbolic; the states of every step are decision variables bound by the dynamics, so that a long period
  and poor initial parameters stay well conditioned, and the unmeasured states (walls, water) are free at the start;
* **shooting** (closed-loop models, their controllers inside): the groups are the only decision variables, the
  model is simulated with CVODES inside the objective from the state its own controllers reach before the data
  (:func:`~mpctrano.simulation.spin_up`) and IPOPT gets the gradient from central finite differences of the
  objective (a quasi-Newton Hessian). The switching controllers of such a model (thermostats, hysteresis) make its
  collocated dynamics non-smooth, where a simulation follows them.

The objective weighs the errors of the measured zone air temperatures, of the energy metered over each step (fuel,
electricity) when the data has the meters, and the deviation of each group from its prior.

Only the parameters the equations use count: the layer data of a construction (``constructionType.k``) are not,
the conductances and capacities computed from them (``G[i]``, ``C[i]``) are. The heat output law of the radiators is
settled when rumoca translates the model and cannot be calibrated.

A third kind of group comes from the YAML of the building (:mod:`mpctrano.yaml_parameters`): a :class:`YamlGroup`
is one field of the YAML declared as a calibration parameter (a conductivity, a thickness, an infiltration rate),
adjusted within its bounds; it moves the parameters of the model that depend on it, each with the exponent measured
by regenerating the model (``p = p0 * (field / field0) ** e``). The model of a closed-loop ``FastModel`` (the
controllers of the YAML inside, no control input) calibrates alike: the data then holds the disturbances, the
measured air temperatures and the meters only.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import casadi as ca
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from trano_optimization.collocation import SOLVED, collocation_matrices, default_ipopt_options

from mpctrano.data import MonitoringData, time_step_of
from mpctrano.forecast import Forecaster, WeatherForecaster, seconds_of_year
from mpctrano.metrics import FitMetrics
from mpctrano.model import FastModel
from mpctrano.units import KELVIN, STATE_BOUNDS

Symbol = TypeVar("Symbol", ca.SX, ca.MX)
FD_STEP = 0.05
"""Step of the central differences of the shooting objective, in the decision variables (log of a factor): 5 %
of a field. The switching controllers make the objective jumpy at the scale of a tenth of a percent (a thermostat
flips one integration step earlier); a wide step reads the slope of the valley through the jumps."""
"""The kind of CasADi expression a problem is written in: SX inside an integrator, MX in an Opti problem."""


class ParameterGroup(BaseModel):
    """Parameters adjusted together by one decision variable."""

    model_config = ConfigDict(frozen=True)

    name: str
    patterns: tuple[str, ...] = Field(description="Regular expressions searched in the parameter names")
    mode: Literal["scale", "value"] = "scale"
    lower: float = Field(0.25, description="Lowest factor (scale) or value")
    upper: float = Field(4.0, description="Highest factor (scale) or value")
    prior_std: float = Field(
        1.0, gt=0, description="Standard deviation of the prior: of log(factor) (scale), of the value (value)"
    )

    def matches(self, name: str) -> bool:
        return any(re.search(pattern, name) for pattern in self.patterns)


class YamlGroup(BaseModel):
    """A field of the YAML declared as a calibration parameter, and the parameters of the model it moves."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(description="The path of the field (``material[ROCKWOOL:CAL].thermal_conductivity``)")
    value: float = Field(gt=0, description="The nominal value of the field (the YAML)")
    lower: float = Field(gt=0)
    upper: float = Field(gt=0)
    exponents: dict[str, float] = Field(description="Exponent of each parameter of the model in the field")
    prior_std: float = Field(0.5, gt=0, description="Standard deviation of the prior of log(field / value)")
    mode: Literal["field"] = "field"

    def matches(self, name: str) -> bool:
        return name in self.exponents


AnyGroup = ParameterGroup | YamlGroup

DEFAULT_GROUPS = (
    ParameterGroup(
        name="envelope_conductance",
        patterns=(r"(wall|ground|floor|roof|ceiling)\w*_\d+\.G\[\d+\]$", r"window_\d+\.(GGap\[\d+\]|GFra)$"),
    ),
    ParameterGroup(
        name="envelope_capacity",
        patterns=(r"(wall|ground|floor|roof|ceiling)\w*_\d+\.C\[\d+\]$", r"window_\d+\.CPan\[\d+\]$"),
    ),
    ParameterGroup(name="air_capacity", patterns=(r"(^|\.)space_\d+\.mSenFac$",), lower=1.0, upper=10.0),
    ParameterGroup(name="infiltration", patterns=(r"(^|\.)space_\d+\.ACH$",), mode="value", lower=0.0, upper=3.0),
    ParameterGroup(
        name="boiler_efficiency", patterns=(r"(^|\.)boiler_\d+\.eta_nominal$",), lower=0.7, upper=1.1, prior_std=0.2
    ),
)
"""Groups of the envelope, the air and the boilers of a FastModel: identifiable from the air temperatures and the
fuel meter (the efficiency of the boilers needs the meter; without it, its prior keeps it at the Trano value)."""

OPTIONAL_GROUPS = (
    ParameterGroup(
        name="solar_transmittance",
        patterns=(r"window_\d+\.tauCoe\[\d+\]$", r"window_\d+\.glazing\.tau\[\d+\]$"),
        lower=0.5,
        upper=1.5,
        prior_std=0.3,
    ),
    ParameterGroup(name="emitter_capacity", patterns=(r"(^|\.)radiator_\d+\.C$",)),
)
"""Groups that the air temperatures hardly identify on their own (the transmittance of the windows trades with the
gains, the capacity of the radiators with the air): add them to the settings with data that excites them."""


class CalibrationSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    groups: tuple[AnyGroup, ...] = DEFAULT_GROUPS
    temperature_std: float = Field(0.1, gt=0, description="Standard deviation of a temperature measurement [K]")
    meter_std: float = Field(0.05, gt=0, description="Standard deviation of the energy metered over a step [kWh]")
    regularization: float = Field(1.0, ge=0, description="Weight of the priors of the groups")
    degree: int = Field(2, ge=1, le=9, description="Radau collocation points per step")
    max_iterations: int = Field(500, ge=1)
    method: Literal["auto", "collocation", "shooting"] = Field(
        "auto", description="How the dynamics enter the problem; auto: shooting for a closed-loop model"
    )
    spin_up_days: float = Field(
        2.0, gt=0, description="Days a closed-loop model runs before the data, with the candidate parameters"
    )
    verbose: bool = False

    def method_for(self, model: FastModel) -> Literal["collocation", "shooting"]:
        if self.method == "auto":
            return "shooting" if not model.control_names else "collocation"
        return self.method


class CalibrationResult(BaseModel):
    """Calibrated groups and fit before and after."""

    factors: dict[str, float] = Field(
        description="Factor (scale group), common value (value group) or value of the field (YAML group) of each group"
    )
    parameters: dict[str, float] = Field(description="Calibrated values of the parameters of the groups")
    initial_state: list[float] = Field(description="Estimated state at the start of the data")
    training_before: FitMetrics
    training_after: FitMetrics
    validation_before: FitMetrics | None = None
    validation_after: FitMetrics | None = None
    success: bool
    status: str
    iterations: int
    solve_time: float
    method: Literal["collocation", "shooting"] = "collocation"
    variables: int = Field(0, description="Decision variables of the nonlinear programme (states, points, groups)")
    constraints: int = Field(0, description="Constraints of the nonlinear programme (the collocated dynamics)")

    def metrics(self) -> pd.DataFrame:
        rows = {
            "training (Trano)": self.training_before,
            "training (calibrated)": self.training_after,
            "validation (Trano)": self.validation_before,
            "validation (calibrated)": self.validation_after,
        }
        return pd.DataFrame(
            {name: {"rmse": fit.rmse_mean, "mae": fit.mae_mean} for name, fit in rows.items() if fit is not None}
        ).T


@dataclass(frozen=True)
class _Data:
    """Arrays of a monitoring period: inputs held over each step, measurements at the step boundaries."""

    index: pd.DatetimeIndex
    time_step: float
    controls: np.ndarray  # (controls, N)
    disturbances: np.ndarray  # (disturbances, N)
    temperatures: np.ndarray  # (zones, N + 1), NaN when missing
    meters: np.ndarray  # (meters, N) energy over each step [kWh], NaN when missing
    meter_names: list[str]

    @property
    def start_time(self) -> float:
        return float(seconds_of_year(self.index[:1])[0])


def prepare(model: FastModel, data: MonitoringData | pd.DataFrame, forecaster: Forecaster | None = None) -> _Data:
    """Read monitoring data named after the model: the controls applied (held from each row to the next), the
    measured air temperatures ``<zone>.TAir`` [K] and, optionally, the readings of the meters [kWh] and the
    disturbances (from the weather file of the model when missing)."""
    data = data.frame if isinstance(data, MonitoringData) else data
    index = pd.DatetimeIndex(data.index)
    time_step = time_step_of(index)
    missing = [name for name in [*model.control_names, *model.comfort_states] if name not in data]
    if missing:
        raise ValueError(f"The data has no column {missing}.")
    steps = len(index) - 1
    if all(name in data for name in model.disturbance_names):
        disturbances = data[model.disturbance_names].to_numpy(dtype=float).T[:, :-1]
    else:
        forecaster = forecaster or WeatherForecaster.from_model(model)
        disturbances = forecaster.realisation(index[0], steps, time_step)[model.disturbance_names].to_numpy().T
    meter_names = [name for name in model.meters if name in data]
    readings = data[meter_names].to_numpy(dtype=float).T if meter_names else np.zeros((0, steps + 1))
    return _Data(
        index=index,
        time_step=time_step,
        controls=data[model.control_names].to_numpy(dtype=float).T[:, :-1],
        disturbances=np.asarray(disturbances, dtype=float),
        temperatures=data[model.comfort_states].to_numpy(dtype=float).T,
        meters=np.diff(readings, axis=1),
        meter_names=meter_names,
    )


def _initial_state(model: FastModel, data: _Data, spin_up_days: float = 2.0) -> np.ndarray:
    """The initial state of the model with the measured air temperatures.

    The other states of a model with controls are its temperatures at the mean of the measurements (the water and
    the walls are unknown); a closed-loop model (its controllers inside, no control) starts from ``spin_up_days`` of
    its own operation before the data (:func:`~mpctrano.simulation.spin_up`): its water, buffer and controller
    states are then consistent, where a buffer at the air temperature sends the controllers into a transient the
    integrator cannot follow.
    """
    measured = data.temperatures[:, 0]
    if model.control_names:
        state = model.initial_state.copy()
        mean = float(np.nanmean(measured)) if np.any(~np.isnan(measured)) else None
        for i, variable in enumerate(model.model.description.states):
            if variable.unit == "K" and mean is not None:
                state[i] = mean
    else:
        from mpctrano.simulation import spin_up

        state = spin_up(model, model, data.index[0], days=spin_up_days, time_step=data.time_step)
    for row, index in enumerate(model.comfort_indices):
        if not np.isnan(measured[row]):
            state[index] = measured[row]
    return state


class _FiniteDifferences(ca.Callback):
    """A scalar CasADi function with its gradient by central differences of a fixed step."""

    def __init__(self, function: ca.Function, step: float) -> None:
        ca.Callback.__init__(self)
        self.function, self.step = function, step
        self.n = int(function.size1_in(0))
        self.construct("objective", {})

    def get_n_in(self) -> int:
        return 1

    def get_n_out(self) -> int:
        return 1

    def get_sparsity_in(self, index: int) -> ca.Sparsity:
        return ca.Sparsity.dense(self.n, 1)

    def get_sparsity_out(self, index: int) -> ca.Sparsity:
        return ca.Sparsity.dense(1, 1)

    def eval(self, arg: Sequence[ca.DM]) -> list[ca.DM]:
        return [self.function(arg[0])]

    def has_jacobian(self) -> bool:
        return True

    def get_jacobian(
        self, name: str, inames: Sequence[str], onames: Sequence[str], opts: Mapping[str, Any]
    ) -> ca.Function:
        theta, nominal = ca.MX.sym("theta", self.n), ca.MX.sym("f")
        columns = []
        for i in range(self.n):
            e = ca.DM.zeros(self.n)
            e[i] = self.step
            columns.append((self.function(theta + e) - self.function(theta - e)) / (2 * self.step))
        return ca.Function(name, [theta, nominal], [ca.hcat(columns)], list(inames), list(onames), dict(opts))


def evaluate(
    model: FastModel, data: MonitoringData | pd.DataFrame | _Data, x0: np.ndarray | None = None
) -> tuple[FitMetrics, np.ndarray]:
    """Fit of an open-loop simulation (CVODES) to the measured air temperatures, from ``x0`` (by default the air
    temperatures measured at the start and every other temperature at their mean); returns the states too."""
    arrays = data if isinstance(data, _Data) else prepare(model, data)
    start = _initial_state(model, arrays) if x0 is None else np.asarray(x0, dtype=float)
    states = model.simulate(start, arrays.controls, arrays.disturbances, arrays.time_step, start_time=arrays.start_time)
    simulated = states[model.comfort_indices]
    observed = ~np.isnan(arrays.temperatures).any(axis=0)
    metrics = FitMetrics.compute(arrays.temperatures[:, observed], simulated[:, observed], model.zone_names)
    return metrics, states


class Calibrator:
    """Least-squares calibration of the parameter groups of a FastModel (see the module documentation)."""

    def __init__(self, model: FastModel, settings: CalibrationSettings | None = None) -> None:
        self.model = model
        self.settings = settings or CalibrationSettings()
        used = self._used_parameters()
        self.members: dict[str, list[int]] = {}
        for group in self.settings.groups:
            members = [
                i
                for i, name in enumerate(model.parameter_names)
                if i in used and group.matches(name) and np.isfinite(model.parameters[i])
            ]
            if group.mode in ("scale", "field"):
                members = [i for i in members if model.parameters[i] != 0]
            if members:
                self.members[group.name] = members
        self.groups = [group for group in self.settings.groups if group.name in self.members]
        if not self.groups:
            raise ValueError(
                f"No parameter of {model.name} matches the groups {[g.name for g in self.settings.groups]}."
            )

    def _used_parameters(self) -> set[int]:
        """The parameters the equations depend on."""
        model = self.model
        t = ca.SX.sym("t")
        x = ca.SX.sym("x", model.n_states)
        u = ca.SX.sym("u", len(model.input_names))
        p = ca.SX.sym("p", model.n_parameters)
        rhs = model.model.rhs.expand()(t, x, u, p)
        return set(ca.jacobian(rhs, p).sparsity().get_col())

    def parameter_vector(self, values: dict[str, float]) -> np.ndarray:
        """The parameters of the model with each group at its factor (scale), value (value) or field value (YAML)."""
        vector = self.model.parameters.copy()
        names = self.model.parameter_names
        for group in self.groups:  # the factors of the groups sharing a parameter compound
            members = self.members[group.name]
            if isinstance(group, YamlGroup):
                factor = values[group.name] / group.value
                for i in members:
                    vector[i] *= factor ** group.exponents[names[i]]
            elif group.mode == "scale":
                vector[members] *= values[group.name]
            else:
                vector[members] = values[group.name]
        return vector

    def calibrated(self, values: dict[str, float]) -> FastModel:
        return self.model.with_parameters(self.parameter_vector(values))

    def _prior_value(self, group: AnyGroup) -> float:
        """The decision variable of a group at the Trano values: log(1) for a scale or YAML group, the mean value
        otherwise."""
        return (
            0.0
            if group.mode in ("scale", "field")
            else float(np.mean(np.nan_to_num(self.model.parameters)[self.members[group.name]]))
        )

    def _bounds(self, group: AnyGroup) -> tuple[float, float]:
        if isinstance(group, YamlGroup):
            return float(np.log(group.lower / group.value)), float(np.log(group.upper / group.value))
        if group.mode == "scale":
            return float(np.log(group.lower)), float(np.log(group.upper))
        return group.lower, group.upper

    def _symbolic_parameters(self, theta: Symbol) -> tuple[Symbol, dict[str, Symbol]]:
        """The parameter vector of the model as expressions (of the kind of ``theta``) of the decision variables of
        the groups; the factors of the groups sharing a parameter compound."""
        symbol = type(theta)
        base = np.nan_to_num(self.model.parameters)
        names = self.model.parameter_names
        parameters: list[Symbol] = [symbol(float(value)) for value in base]
        values: dict[str, Symbol] = {}
        for position, group in enumerate(self.groups):
            if isinstance(group, YamlGroup):
                values[group.name] = group.value * ca.exp(theta[position])
                for i in self.members[group.name]:
                    parameters[i] = parameters[i] * ca.exp(group.exponents[names[i]] * theta[position])
                continue
            values[group.name] = ca.exp(theta[position]) if group.mode == "scale" else theta[position]
            for i in self.members[group.name]:
                parameters[i] = parameters[i] * values[group.name] if group.mode == "scale" else theta[position]
        return ca.vertcat(*parameters), values

    def _collocate(self, opti: ca.Opti, z: ca.MX, zc: ca.MX, p: ca.MX, arrays: _Data) -> None:
        """The dynamics on the Radau points of every step, with the inputs of the data and symbolic parameters."""
        model, degree, dt = self.model, self.settings.degree, arrays.time_step
        n_x = model.n_states
        tau, c, d = collocation_matrices(degree)
        collocated = model.collocated(dt)
        offset = ca.DM(collocated.offset)
        positions = {name: row for row, name in enumerate(model.control_names)}
        disturbances = {name: row for row, name in enumerate(model.disturbance_names)}
        for k in range(arrays.controls.shape[1]):
            inputs = ca.DM(
                [
                    arrays.controls[positions[name], k]
                    if name in positions
                    else arrays.disturbances[disturbances[name], k]
                    for name in model.input_names
                ]
            )
            points = [z[:, k]] + [zc[j * n_x : (j + 1) * n_x, k] for j in range(degree)]
            t0 = arrays.start_time + k * dt
            for j in range(1, degree + 1):
                slope = sum(c[r, j] * points[r] for r in range(degree + 1))
                derivative = collocated.rhs(t0 + tau[j] * dt, points[j] + offset, inputs, p)
                opti.subject_to(dt * derivative - slope == 0)
            opti.subject_to(z[:, k + 1] == sum(d[r] * points[r] for r in range(degree + 1)))

    def _objective(self, z: Symbol, theta: Symbol, arrays: _Data) -> Symbol:
        """Weighted squares of the temperature and meter errors and of the deviations of the groups from their prior."""
        model, settings = self.model, self.settings
        observed = ~np.isnan(arrays.temperatures)
        measured = np.where(observed, arrays.temperatures - KELVIN, 0.0)
        errors = (z[model.comfort_indices, :] - ca.DM(measured)) * ca.DM(observed.astype(float))
        objective = ca.sumsqr(errors) / settings.temperature_std**2
        for row, name in enumerate(arrays.meter_names):
            index = model.state_names.index(name)
            seen = ~np.isnan(arrays.meters[row])
            drawn = z[index, 1:] - z[index, :-1]
            residual = (drawn - ca.DM(np.where(seen, arrays.meters[row], 0.0)).T) * ca.DM(seen.astype(float)).T
            objective += ca.sumsqr(residual) / settings.meter_std**2
        priors = [
            (theta[position] - self._prior_value(group)) / group.prior_std for position, group in enumerate(self.groups)
        ]
        return objective + settings.regularization * ca.sumsqr(ca.vertcat(*priors))

    def fit(
        self, data: MonitoringData | pd.DataFrame, validation: MonitoringData | pd.DataFrame | None = None
    ) -> tuple[FastModel, CalibrationResult]:
        """Calibrate on ``data`` (see :func:`prepare`) by the method of the settings; ``validation`` follows it in
        time and is only simulated."""
        arrays = prepare(self.model, data)
        if self.settings.method_for(self.model) == "shooting":
            return self._fit_shooting(arrays, validation)
        return self._fit_collocation(arrays, validation)

    def _fit_shooting(
        self, arrays: _Data, validation: MonitoringData | pd.DataFrame | None
    ) -> tuple[FastModel, CalibrationResult]:
        """The groups as the only decision variables, the model simulated with CVODES inside the objective.

        A closed-loop model starts ``spin_up_days`` before the data from the start values of the Modelica model and
        runs with its own controllers and the candidate parameters: the walls, the water and the controller states at
        the start of the data are then those of the candidate model, not of the nominal one (a fixed start, spun up
        with the nominal parameters, biases the estimates towards it).
        """
        model, settings = self.model, self.settings
        steps, dt = arrays.controls.shape[1], arrays.time_step
        collocated = model.collocated(dt)
        x = ca.SX.sym("x", model.n_states)
        u = ca.SX.sym("u", len(model.control_names))
        w = ca.SX.sym("w", len(model.disturbance_names))
        t0, s = ca.SX.sym("t0"), ca.SX.sym("s")
        theta_sx = ca.SX.sym("theta", len(self.groups))
        p, _ = self._symbolic_parameters(theta_sx)
        ode = {
            "x": x,
            "t": s,
            "p": ca.vertcat(t0, u, w, theta_sx),
            "ode": collocated.rhs(t0 + s, x, collocated._inputs(u, w), p),
        }
        # tight tolerances: the objective is differentiated by finite differences
        step = ca.integrator("step", "cvodes", ode, 0.0, dt, {"abstol": 1e-8, "reltol": 1e-8})
        theta = ca.MX.sym("theta", len(self.groups))
        _, values = self._symbolic_parameters(theta)
        if model.control_names:
            state = ca.MX(_initial_state(model, arrays))
        else:
            state = ca.MX(model.initial_state)
            warm_up = self._spin_up_disturbances(arrays)
            for k in range(warm_up.shape[1]):
                t_k = arrays.start_time - (warm_up.shape[1] - k) * dt
                state = step(x0=state, p=ca.vertcat(t_k, np.zeros(0), warm_up[:, k], theta))["xf"]
        states = [state]
        for k in range(steps):
            inputs = ca.vertcat(arrays.start_time + k * dt, arrays.controls[:, k], arrays.disturbances[:, k], theta)
            state = step(x0=state, p=inputs)["xf"]
            states.append(state)
        z = ca.hcat(states) - ca.DM(collocated.offset)
        # the switching controllers (thermostats, hysteresis) make the adjoint gradient of the chain noisy and the
        # objective jumpy at a small scale (IPOPT then stalls short of the optimum): the gradient comes from wide
        # central differences of the objective, 2n + 1 simulations of a second each for n groups
        objective_function = _FiniteDifferences(
            ca.Function("objective", [theta], [self._objective(z, theta, arrays)]), FD_STEP
        )
        objective = objective_function(theta)
        bounds = np.array([self._bounds(group) for group in self.groups])
        options = {
            **default_ipopt_options(),
            "max_iter": settings.max_iterations,
            "print_level": 5 if settings.verbose else 0,
            "hessian_approximation": "limited-memory",
            "tol": 1e-4,
            # the small-scale jumps of the objective (a thermostat flipping one integration step earlier) defeat a
            # sufficient-decrease test and keep the objective changing by a fraction of a percent: full quasi-Newton
            # steps, and the solve is over once the objective settles within a percent
            "acceptable_tol": 1.0,
            "acceptable_obj_change_tol": 1e-2,
            "acceptable_iter": 3,
            "accept_every_trial_step": "yes",
        }
        solver = ca.nlpsol(
            "calibration", "ipopt", {"x": theta, "f": objective}, {"ipopt": options, "print_time": False}
        )
        started = time.perf_counter()
        solution = solver(x0=[self._prior_value(group) for group in self.groups], lbx=bounds[:, 0], ubx=bounds[:, 1])
        solve_time = time.perf_counter() - started
        stats = solver.stats()
        at_solution = ca.Function("solution", [theta], [ca.vertcat(*list(values.values())), z])
        estimated_values, trajectory = at_solution(solution["x"])
        estimated = {group.name: float(estimated_values[position]) for position, group in enumerate(self.groups)}
        states_array = np.asarray(trajectory, dtype=float) + collocated.offset[:, None]
        result = self._result(
            estimated, states_array, arrays, validation, str(stats["return_status"]), stats, solve_time
        )
        return result[0], result[1].model_copy(update={"method": "shooting", "variables": len(self.groups)})

    def _spin_up_disturbances(self, arrays: _Data) -> np.ndarray:
        """The disturbances of the ``spin_up_days`` before the data, from the weather file of the model."""
        model, dt = self.model, arrays.time_step
        steps = round(self.settings.spin_up_days * 86400 / dt)
        start = arrays.index[0] - pd.Timedelta(seconds=steps * dt)
        realisation = WeatherForecaster.from_model(model).realisation(start, steps, dt)
        return realisation[model.disturbance_names].to_numpy(dtype=float).T

    def _fit_collocation(
        self, arrays: _Data, validation: MonitoringData | pd.DataFrame | None
    ) -> tuple[FastModel, CalibrationResult]:
        """The states on the collocation points as decision variables, the dynamics as constraints."""
        model, settings = self.model, self.settings
        n_x, steps, degree, dt = model.n_states, arrays.controls.shape[1], settings.degree, arrays.time_step
        collocated = model.collocated(dt)

        opti = ca.Opti()
        z = opti.variable(n_x, steps + 1)  # states shifted by the offsets (temperatures in degC)
        zc = opti.variable(n_x * degree, steps)
        theta = opti.variable(len(self.groups))  # log(factor) of a scale group, value of a value group
        p, values = self._symbolic_parameters(theta)
        self._collocate(opti, z, zc, p, arrays)
        temperatures = [i for i, variable in enumerate(model.model.description.states) if variable.unit == "K"]
        low, high = STATE_BOUNDS
        opti.subject_to(opti.bounded(low - KELVIN, ca.vec(z[temperatures, :]), high - KELVIN))
        start = _initial_state(model, arrays)
        if not model.control_names:
            # a closed-loop model: its controller states (integrators, hysteresis flags, pressures) at the start are
            # not identifiable from the data and left free they make the simulation chatter; the spin-up sets them
            others = [i for i in range(n_x) if i not in temperatures]
            if others:
                opti.subject_to(z[others, 0] == ca.DM(start[others] - collocated.offset[others]))
        bounds = [self._bounds(group) for group in self.groups]
        opti.subject_to(opti.bounded(ca.DM([b[0] for b in bounds]), theta, ca.DM([b[1] for b in bounds])))
        opti.minimize(self._objective(z, theta, arrays))

        guess = (
            model.simulate(start, arrays.controls, arrays.disturbances, dt, start_time=arrays.start_time)
            - collocated.offset[:, None]
        )
        opti.set_initial(z, guess)
        opti.set_initial(zc, np.tile(guess[:, 1:], (degree, 1)))
        opti.set_initial(theta, [self._prior_value(group) for group in self.groups])
        options = {
            **default_ipopt_options(),
            "max_iter": settings.max_iterations,
            "print_level": 5 if settings.verbose else 0,
        }
        opti.solver("ipopt", {"print_time": False, "expand": True}, options)
        started = time.perf_counter()
        try:
            value = opti.solve().value
        except RuntimeError:
            value = opti.debug.value
        solve_time = time.perf_counter() - started
        stats = opti.stats()

        estimated = {group.name: float(value(values[group.name])) for group in self.groups}
        trajectory = np.asarray(value(z), dtype=float).reshape(n_x, steps + 1) + collocated.offset[:, None]
        result = self._result(estimated, trajectory, arrays, validation, str(stats["return_status"]), stats, solve_time)
        return result[0], result[1].model_copy(
            update={"method": "collocation", "variables": int(opti.nx), "constraints": int(opti.ng)}
        )

    def _result(
        self,
        estimated: dict[str, float],
        trajectory: np.ndarray,
        arrays: _Data,
        validation: MonitoringData | pd.DataFrame | None,
        status: str,
        stats: dict[str, object],
        solve_time: float,
    ) -> tuple[FastModel, CalibrationResult]:
        """The calibrated model and its fit, before and after, on the training and the validation data (simulated
        from where the training ends). ``trajectory`` is the collocated solution: the fit after is that of a
        simulation from its first state, or of the trajectory itself when the integrator cannot follow that state."""
        model = self.model
        calibrated = self.calibrated(estimated)
        x0 = trajectory[:, 0]
        before, before_states = evaluate(model, arrays)
        try:
            after, after_states = evaluate(calibrated, arrays, x0)
        except RuntimeError:  # CVODES: the collocated states stand in for the simulation
            after_states = trajectory
            observed = ~np.isnan(arrays.temperatures).any(axis=0)
            after = FitMetrics.compute(
                arrays.temperatures[:, observed], trajectory[model.comfort_indices][:, observed], model.zone_names
            )
        validation_before = validation_after = None
        if validation is not None:
            validation_arrays = prepare(model, validation)
            validation_before, _ = evaluate(model, validation_arrays, before_states[:, -1])
            try:
                validation_after, _ = evaluate(calibrated, validation_arrays, after_states[:, -1])
            except RuntimeError:
                validation_after = None
        changed = sorted({i for group in self.groups for i in self.members[group.name]})
        iterations = stats.get("iter_count", 0)
        return calibrated, CalibrationResult(
            factors=estimated,
            parameters={model.parameter_names[i]: float(calibrated.parameters[i]) for i in changed},
            initial_state=x0.tolist(),
            training_before=before,
            training_after=after,
            validation_before=validation_before,
            validation_after=validation_after,
            success=status in SOLVED,
            status=status,
            iterations=int(iterations) if isinstance(iterations, int | float) else 0,
            solve_time=solve_time,
        )


def simulate_fit(model: FastModel, data: MonitoringData | pd.DataFrame, x0: np.ndarray | None = None) -> pd.DataFrame:
    """Measured and simulated zone air temperatures (``<zone>.TAir_measured`` and ``<zone>.TAir_simulated`` [K]) and
    meters (``<meter>_measured``, ``<meter>_simulated`` [kWh] since the start) of an open-loop simulation from ``x0``
    (see :func:`evaluate`), on the index of the data."""
    arrays = prepare(model, data)
    _, states = evaluate(model, arrays, x0)
    columns = {}
    for row, (state, index) in enumerate(zip(model.comfort_states, model.comfort_indices, strict=True)):
        columns[f"{state}_measured"] = arrays.temperatures[row]
        columns[f"{state}_simulated"] = states[index]
    for row, name in enumerate(arrays.meter_names):  # the meters [kWh], from the first reading
        index = model.state_names.index(name)
        measured = np.concatenate([[0.0], np.nancumsum(arrays.meters[row])])
        columns[f"{name}_measured"] = measured
        columns[f"{name}_simulated"] = states[index] - states[index, 0]
    return pd.DataFrame(columns, index=arrays.index)
