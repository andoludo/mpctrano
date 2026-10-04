"""Calibration of the RC parameters on monitoring data.

The parameters of the Trano model are physically consistent initial guesses derived from the
geometry and the constructions. :class:`Calibrator` identifies them on measurements by solving a
least-squares problem with IPOPT:

* **multiple shooting**: the state trajectory is a decision variable constrained by the discrete
  dynamics ``x[k+1] = F(x[k], u[k], d[k], p)``, which keeps the problem well conditioned even
  for long periods and poor initial parameters;
* **log-parameterisation**: each calibrated parameter is ``p = p0 * exp(z)`` with ``z`` bounded,
  so the parameters stay positive and within a factor range of the initial guess;
* **regularisation** towards the initial guess: zone temperatures alone do not identify every
  parameter of a multi-zone RC network (e.g. the envelope capacity and the ground resistance
  are correlated); the Tikhonov term keeps the non-identifiable directions at their physical
  prior while the data drives the identifiable ones;
* **free unmeasured states**: the envelope, mass or emitter temperatures are not measured, their
  trajectories are estimated together with the parameters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import casadi as ca
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mpctrano.metrics import FitMetrics
from mpctrano.model import STATE_BOUNDS, RCModel

if TYPE_CHECKING:
    from mpctrano.data import MonitoringData, Signals

FIXED_PARAMETERS = ("TGro",)
FRACTION_UNIT = "1"


def ipopt_options(
    max_iterations: int = 500,
    tolerance: float = 1e-6,
    verbose: bool = False,
    hessian: str = "exact",
    expand: bool = False,
) -> dict[str, Any]:
    """Options shared by the IPOPT problems of the package."""
    return {
        "expand": expand,
        "print_time": verbose,
        "ipopt.max_iter": max_iterations,
        "ipopt.tol": tolerance,
        "ipopt.print_level": 5 if verbose else 0,
        "ipopt.sb": "yes",
        "ipopt.mu_strategy": "adaptive",
        "ipopt.hessian_approximation": hessian,
    }


class CalibrationSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    parameters: list[str] | None = Field(
        None,
        description="fnmatch patterns of the parameters to calibrate (default: all but the ground temperature)",
    )
    lower_factor: float = Field(0.2, gt=0, le=1, description="Lower bound relative to the initial guess")
    upper_factor: float = Field(5.0, ge=1, description="Upper bound relative to the initial guess")
    regularisation: float = Field(1e-2, ge=0, description="Weight of the log-deviation from the initial guess")
    measurement_std: float = Field(0.1, gt=0, description="Expected measurement noise, normalises the residuals [K]")
    free_initial_states: bool = Field(True, description="Estimate the initial unmeasured states")
    substeps: int | None = Field(None, description="RK4 sub-steps per sample (default: stable choice)")
    max_iterations: int = Field(500, ge=1)
    tolerance: float = Field(1e-6, gt=0)
    hessian: Literal["exact", "limited-memory"] = Field("exact", description="IPOPT Hessian of the Lagrangian")
    expand: bool = Field(False, description="Expand the problem into an SX graph before solving")
    verbose: bool = False


class CalibrationResult(BaseModel):
    """Calibrated parameters with the fit before and after calibration."""

    parameters: dict[str, float]
    initial_parameters: dict[str, float]
    calibrated: list[str]
    training: FitMetrics
    initial_training: FitMetrics
    validation: FitMetrics | None = None
    initial_validation: FitMetrics | None = None
    objective: float
    iterations: int
    status: str
    success: bool
    solve_time: float

    @property
    def ratios(self) -> dict[str, float]:
        """Calibrated / initial value of each calibrated parameter."""
        return {name: self.parameters[name] / self.initial_parameters[name] for name in self.calibrated}

    def summary(self, lower_factor: float | None = None, upper_factor: float | None = None) -> pd.DataFrame:
        frame = pd.DataFrame(
            {
                "initial": {name: self.initial_parameters[name] for name in self.calibrated},
                "calibrated": {name: self.parameters[name] for name in self.calibrated},
                "ratio": self.ratios,
            }
        )
        if lower_factor is not None and upper_factor is not None:
            frame["at_bound"] = (frame["ratio"] < lower_factor * 1.01) | (frame["ratio"] > upper_factor / 1.01)
        return frame

    def metrics(self) -> pd.DataFrame:
        rows = {
            "training (initial)": self.initial_training,
            "training (calibrated)": self.training,
            "validation (initial)": self.initial_validation,
            "validation (calibrated)": self.validation,
        }
        return pd.DataFrame(
            {label: {"rmse": fit.rmse_mean, "mae": fit.mae_mean} for label, fit in rows.items() if fit is not None}
        ).T


@dataclass
class Calibrator:
    """Identify the RC parameters of a model on monitoring data."""

    model: RCModel
    settings: CalibrationSettings = field(default_factory=CalibrationSettings)

    def calibrated_names(self) -> list[str]:
        patterns = self.settings.parameters
        names = self.model.select_parameters(patterns) if patterns else list(self.model.parameter_names)
        names = [name for name in names if name not in FIXED_PARAMETERS or (patterns and name in patterns)]
        if not names:
            raise ValueError(f"No parameter matches {patterns}.")
        return names

    def fit(self, data: MonitoringData, validation: MonitoringData | None = None) -> tuple[RCModel, CalibrationResult]:
        """Calibrate on ``data``; the fit is also reported on the ``validation`` period if given."""
        signals = data.signals(self.model)
        names = self.calibrated_names()
        indices = [self.model.parameter_names.index(name) for name in names]
        initial = self.model.parameters
        settings = self.settings
        dynamics = self.model.discrete(signals.time_step, settings.substeps)
        steps = signals.steps
        n_states = self.model.n_states

        opti = ca.Opti()
        X = opti.variable(n_states, steps + 1)
        z = opti.variable(len(names))
        p = _parameter_vector(self.model, initial, indices, z, opti, settings)
        for k in range(steps):
            opti.subject_to(X[:, k + 1] == dynamics(X[:, k], signals.controls[:, k], signals.disturbances[:, k], p))
        opti.subject_to(opti.bounded(STATE_BOUNDS[0], ca.vec(X), STATE_BOUNDS[1]))
        residuals = (X[signals.measured_states, :] - signals.measurements) / settings.measurement_std
        objective = ca.sumsqr(residuals) / (steps + 1) + settings.regularisation * ca.sumsqr(z)
        opti.minimize(objective)

        guess = _state_guess(self.model, data, signals)
        if not settings.free_initial_states:
            opti.subject_to(X[:, 0] == guess[:, 0])
        opti.set_initial(X, guess)
        opti.set_initial(z, np.zeros(len(names)))
        opti.solver(
            "ipopt",
            ipopt_options(
                settings.max_iterations, settings.tolerance, settings.verbose, settings.hessian, settings.expand
            ),
        )

        started = time.perf_counter()
        solution, success, status = _solve(opti)
        solve_time = time.perf_counter() - started
        parameters = initial.copy()
        parameters[indices] = initial[indices] * np.exp(np.asarray(solution.value(z), dtype=float).reshape(-1))
        calibrated = self.model.with_parameters(parameters)

        result = CalibrationResult(
            parameters=calibrated.parameter_values,
            initial_parameters=self.model.parameter_values,
            calibrated=names,
            training=evaluate(calibrated, data),
            initial_training=evaluate(self.model, data),
            validation=evaluate(calibrated, validation) if validation is not None else None,
            initial_validation=evaluate(self.model, validation) if validation is not None else None,
            objective=float(solution.value(objective)),
            iterations=int(solution.stats()["iter_count"]),
            status=status,
            success=success,
            solve_time=solve_time,
        )
        return calibrated, result


def _parameter_vector(
    model: RCModel,
    initial: np.ndarray,
    indices: list[int],
    z: Any,  # noqa: ANN401
    opti: Any,  # noqa: ANN401
    settings: CalibrationSettings,
) -> Any:  # noqa: ANN401
    """Full parameter vector with ``p_i = p0_i * exp(z_j)`` for the calibrated entries, others fixed."""
    entries: list[Any] = [ca.DM(value) for value in initial]
    for j, index in enumerate(indices):
        entries[index] = initial[index] * ca.exp(z[j])
        upper = settings.upper_factor
        if model.interface.parameters[index].unit == FRACTION_UNIT and initial[index] > 0:
            upper = min(upper, 1.0 / initial[index])  # fractions stay below one
        opti.subject_to(opti.bounded(np.log(settings.lower_factor), z[j], np.log(max(upper, 1.0 + 1e-9))))
    return ca.vertcat(*entries)


def _state_guess(model: RCModel, data: MonitoringData, signals: Signals) -> np.ndarray:
    """Dynamically consistent initial trajectory: a simulation with the initial parameters.

    Starting IPOPT from a feasible point (the shooting constraints hold) avoids the restoration
    phase and the local minima it may lead to.
    """
    return model.simulate(data.initial_state(model), signals.controls, signals.disturbances, signals.time_step)


def _solve(opti: Any) -> tuple[Any, bool, str]:  # noqa: ANN401
    """Solve and fall back on the last iterate when IPOPT does not converge."""
    try:
        solution = opti.solve()
    except RuntimeError as error:
        solution = opti.debug
        return solution, False, str(error).splitlines()[-1][:120]
    return solution, bool(solution.stats()["success"]), str(solution.stats()["return_status"])


def evaluate(model: RCModel, data: MonitoringData, fit_initial_state: bool = True) -> FitMetrics:
    """Open-loop simulation of the model over the data and comparison with the measured temperatures.

    The unmeasured initial states are unknown: with ``fit_initial_state`` they are estimated by
    least squares (the parameters stay fixed), which is the fair way to validate a model on a
    period it was not calibrated on.
    """
    signals = data.signals(model)
    x0 = data.initial_state(model)
    if fit_initial_state:
        x0 = _fit_initial_state(model, signals, x0)
    states = model.simulate(x0, signals.controls, signals.disturbances, signals.time_step)
    names = [model.state_names[index] for index in signals.measured_states]
    return FitMetrics.compute(signals.measurements, states[signals.measured_states], names)


def _fit_initial_state(model: RCModel, signals: Signals, guess: np.ndarray) -> np.ndarray:
    """Least-squares initial state: the measured trajectory is affine in ``x0`` (linear model)."""
    affine = model.affine(signals.time_step)
    reference = model.simulate(guess, signals.controls, signals.disturbances, signals.time_step)
    n_measured = len(signals.measured_states)
    sensitivity = np.zeros((n_measured * (signals.steps + 1), model.n_states))
    power = np.eye(model.n_states)
    for k in range(signals.steps + 1):
        sensitivity[k * n_measured : (k + 1) * n_measured] = power[signals.measured_states]
        power = affine.A @ power
    residual = (signals.measurements - reference[signals.measured_states]).T.reshape(-1)
    delta, *_ = np.linalg.lstsq(sensitivity, residual, rcond=None)
    return np.asarray(np.clip(guess + delta, *STATE_BOUNDS), dtype=float)


def simulate_fit(model: RCModel, data: MonitoringData, fit_initial_state: bool = True) -> pd.DataFrame:
    """Measured and simulated zone temperatures side by side (for plots)."""
    signals = data.signals(model)
    x0 = data.initial_state(model)
    if fit_initial_state:
        x0 = _fit_initial_state(model, signals, x0)
    states = model.simulate(x0, signals.controls, signals.disturbances, signals.time_step)
    frame = pd.DataFrame(index=signals.index)
    for position, index in enumerate(signals.measured_states):
        name = model.state_names[index]
        frame[f"{name}_measured"] = signals.measurements[position]
        frame[f"{name}_simulated"] = states[index]
    return frame
