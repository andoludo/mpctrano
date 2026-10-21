"""Calibration of a FastModel on monitoring data.

A FastModel has hundreds of parameters, most of them settled by the YAML (areas, orientations, optical fits) or
not identifiable from a few zone temperatures. :class:`Calibrator` adjusts **groups** of parameters instead,
each by one decision variable:

* a ``scale`` group multiplies the values Trano computed by a common factor ``exp(z)`` (all the conductances of the
  envelope, all its capacities, ...), within ``[lower, upper]`` and regularised towards 1;
* a ``value`` group gives every parameter it matches one common value within ``[lower, upper]``, regularised towards
  their mean Trano value (the infiltration rate, which Trano leaves at 0).

The problem is solved with IPOPT on the Radau collocation of the model over the whole data set (the parameters
symbolic): the states of every step are decision variables bound by the dynamics, so that a long period and poor
initial parameters stay well conditioned. The objective weighs the errors of the measured zone air temperatures, of
the energy metered over each step (fuel, electricity) when the data has the meters, and the deviation of each group
from its prior. The unmeasured states (walls, water) are free at the start of the data.

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
from dataclasses import dataclass
from typing import Literal

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
    verbose: bool = False


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


def _initial_state(model: FastModel, data: _Data) -> np.ndarray:
    """The initial state of the model with the measured air temperatures.

    The other states of a model with controls are its temperatures at the mean of the measurements (the water and
    the walls are unknown); a closed-loop model (its controllers inside, no control) starts from two days of its own
    operation before the data (:func:`~mpctrano.simulation.spin_up`): its water, buffer and controller states are
    then consistent, where a buffer at the air temperature sends the controllers into a transient the integrator
    cannot follow.
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

        state = spin_up(model, model, data.index[0], days=2.0, time_step=data.time_step)
    for row, index in enumerate(model.comfort_indices):
        if not np.isnan(measured[row]):
            state[index] = measured[row]
    return state


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
        for group in self.groups:
            members = self.members[group.name]
            if isinstance(group, YamlGroup):
                factor = values[group.name] / group.value
                for i in members:
                    vector[i] = self.model.parameters[i] * factor ** group.exponents[names[i]]
            elif group.mode == "scale":
                vector[members] = self.model.parameters[members] * values[group.name]
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

    def _symbolic_parameters(self, theta: ca.MX) -> tuple[ca.MX, dict[str, ca.MX]]:
        """The parameter vector of the model as expressions of the decision variables of the groups."""
        base = np.nan_to_num(self.model.parameters)
        names = self.model.parameter_names
        parameters: list[ca.MX] = [ca.MX(float(value)) for value in base]
        values: dict[str, ca.MX] = {}
        for position, group in enumerate(self.groups):
            if isinstance(group, YamlGroup):
                values[group.name] = group.value * ca.exp(theta[position])
                for i in self.members[group.name]:
                    parameters[i] = float(base[i]) * ca.exp(group.exponents[names[i]] * theta[position])
                continue
            values[group.name] = ca.exp(theta[position]) if group.mode == "scale" else theta[position]
            for i in self.members[group.name]:
                parameters[i] = float(base[i]) * values[group.name] if group.mode == "scale" else theta[position]
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

    def _objective(self, z: ca.MX, theta: ca.MX, arrays: _Data) -> ca.MX:
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
        """Calibrate on ``data`` (see :func:`prepare`); ``validation`` follows it in time and is only simulated."""
        model, settings = self.model, self.settings
        arrays = prepare(model, data)
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
        bounds = [self._bounds(group) for group in self.groups]
        opti.subject_to(opti.bounded(ca.DM([b[0] for b in bounds]), theta, ca.DM([b[1] for b in bounds])))
        opti.minimize(self._objective(z, theta, arrays))

        guess = (
            model.simulate(
                _initial_state(model, arrays), arrays.controls, arrays.disturbances, dt, start_time=arrays.start_time
            )
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
        x0 = np.asarray(value(z[:, 0]), dtype=float).ravel() + collocated.offset
        result = self._result(estimated, x0, arrays, validation, str(stats["return_status"]), stats, solve_time)
        return result[0], result[1].model_copy(update={"variables": int(opti.nx), "constraints": int(opti.ng)})

    def _result(
        self,
        estimated: dict[str, float],
        x0: np.ndarray,
        arrays: _Data,
        validation: MonitoringData | pd.DataFrame | None,
        status: str,
        stats: dict[str, object],
        solve_time: float,
    ) -> tuple[FastModel, CalibrationResult]:
        """The calibrated model and its fit, before and after, on the training and the validation data (simulated
        from where the training ends)."""
        model = self.model
        calibrated = self.calibrated(estimated)
        before, before_states = evaluate(model, arrays)
        after, after_states = evaluate(calibrated, arrays, x0)
        validation_before = validation_after = None
        if validation is not None:
            validation_arrays = prepare(model, validation)
            validation_before, _ = evaluate(model, validation_arrays, before_states[:, -1])
            validation_after, _ = evaluate(calibrated, validation_arrays, after_states[:, -1])
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
