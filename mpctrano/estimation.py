"""State estimation: the MPC needs the full state (including the unmeasured envelope, mass or
emitter temperatures) at the start of each horizon.

* :class:`KalmanFilter`: an extended Kalman filter on the discrete-time model, linearised at
  the current estimate and inputs (exact for the linear envelope models, where it is a
  standard Kalman filter; the heat pump terms are bilinear).
* :class:`MovingHorizonEstimator`: least-squares estimation over a sliding window solved with
  IPOPT; it handles state bounds and is the natural counterpart of the MPC problem.

Both follow the same cycle: :meth:`correct` with the measurement of the current step, use the
state, then :meth:`predict` with the inputs applied during the step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import casadi as ca
import numpy as np

from mpctrano.calibration import ipopt_options
from mpctrano.model import STATE_BOUNDS, RCModel


class StateEstimator(Protocol):
    @property
    def state(self) -> np.ndarray: ...

    def correct(self, measurement: np.ndarray) -> np.ndarray:
        """Incorporate the measured states (zone temperatures, storages; NaN when a sensor is missing)."""

    def predict(self, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        """Propagate the estimate over one step with the applied inputs."""


def _measurement_matrix(model: RCModel) -> np.ndarray:
    """Selects the measured states: the zone air temperatures and the storage states."""
    matrix = np.zeros((model.n_measured, model.n_states))
    for row, index in enumerate(model.measured_indices):
        matrix[row, index] = 1.0
    return matrix


def _initial_state(model: RCModel, x0: np.ndarray | None) -> np.ndarray:
    return model.initial_state.copy() if x0 is None else np.asarray(x0, dtype=float).reshape(-1).copy()


@dataclass
class KalmanFilter:
    """(Extended) Kalman filter on the discrete-time model, linearised at the current estimate.

    ``process_std`` is the standard deviation of the model error per step for each state [K]
    (a scalar applies to all), ``measurement_std`` the sensor noise [K]. The storage states
    (energies [J], tank temperature [K]) use the same values unless an array is given.
    """

    model: RCModel
    time_step: float
    process_std: float | np.ndarray = 0.05
    measurement_std: float = 0.1
    x0: np.ndarray | None = None
    initial_std: float = 2.0
    substeps: int | None = None
    _x: np.ndarray = field(init=False, repr=False)
    _P: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.dynamics = self.model.linearization(self.time_step, self.substeps)
        self.C = _measurement_matrix(self.model)
        process = np.broadcast_to(np.asarray(self.process_std, dtype=float), (self.model.n_states,))
        self.Q = np.diag(process**2)
        self.R = np.eye(self.model.n_measured) * self.measurement_std**2
        self._x = _initial_state(self.model, self.x0)
        self._P = np.eye(self.model.n_states) * self.initial_std**2

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._P.copy()

    def correct(self, measurement: np.ndarray) -> np.ndarray:
        y = np.asarray(measurement, dtype=float).reshape(-1)
        available = ~np.isnan(y)
        if available.any():
            C = self.C[available]
            R = self.R[np.ix_(available, available)]
            innovation = y[available] - C @ self._x
            S = C @ self._P @ C.T + R
            K = self._P @ C.T @ np.linalg.inv(S)
            self._x = self._x + K @ innovation
            identity = np.eye(self.model.n_states)
            self._P = (identity - K @ C) @ self._P @ (identity - K @ C).T + K @ R @ K.T
        return self.state

    def predict(self, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        x_next, A = self.dynamics(self._x, np.asarray(control, dtype=float), np.asarray(disturbance, dtype=float))
        self._x = np.asarray(x_next, dtype=float).reshape(-1)
        A = np.asarray(A, dtype=float)
        self._P = A @ self._P @ A.T + self.Q
        return self.state


@dataclass
class MovingHorizonEstimator:
    """Least-squares state estimation over the last ``window`` steps, solved with IPOPT.

    The cost is ``|x[0] - prior|^2 / prior_std^2 + sum |w[k]|^2 / process_std^2
    + sum |y[k] - C x[k]|^2 / measurement_std^2`` with ``x[k+1] = F(x[k], u[k], d[k]) + w[k]``.
    The prior of the next window is the second state of the current solution (shifted arrival cost).
    """

    model: RCModel
    time_step: float
    window: int = 24
    process_std: float = 0.05
    measurement_std: float = 0.1
    prior_std: float = 2.0
    x0: np.ndarray | None = None
    substeps: int | None = None
    _measurements: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _controls: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _disturbances: list[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _problems: dict[int, tuple[Any, dict[str, Any]]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.dynamics = self.model.discrete(self.time_step, self.substeps)
        self.C = _measurement_matrix(self.model)
        self._prior = _initial_state(self.model, self.x0)
        self._x = self._prior.copy()
        self._trajectory: np.ndarray | None = None

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    def correct(self, measurement: np.ndarray) -> np.ndarray:
        self._measurements.append(np.asarray(measurement, dtype=float).reshape(-1))
        self._trim()
        self._solve()
        return self.state

    def predict(self, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        self._controls.append(np.asarray(control, dtype=float).reshape(-1))
        self._disturbances.append(np.asarray(disturbance, dtype=float).reshape(-1))
        self._x = np.asarray(
            self.dynamics(self._x, self._controls[-1], self._disturbances[-1], self.model.parameters), dtype=float
        ).reshape(-1)
        return self.state

    def _trim(self) -> None:
        while len(self._measurements) > self.window + 1:
            self._measurements.pop(0)
            self._controls.pop(0)
            self._disturbances.pop(0)
            if self._trajectory is not None:
                self._prior = self._trajectory[:, 1]
                self._trajectory = self._trajectory[:, 1:]

    def _problem(self, steps: int) -> tuple[Any, dict[str, Any]]:
        if steps in self._problems:
            return self._problems[steps]
        model = self.model
        opti = ca.Opti()
        X = opti.variable(model.n_states, steps + 1)
        W = opti.variable(model.n_states, max(steps, 1))
        Y = opti.parameter(model.n_measured, steps + 1)
        mask = opti.parameter(model.n_measured, steps + 1)
        U = opti.parameter(model.n_controls, max(steps, 1))
        D = opti.parameter(model.n_disturbances, max(steps, 1))
        prior = opti.parameter(model.n_states)
        cost = ca.sumsqr(X[:, 0] - prior) / self.prior_std**2
        for k in range(steps):
            opti.subject_to(X[:, k + 1] == self.dynamics(X[:, k], U[:, k], D[:, k], model.parameters) + W[:, k])
            cost += ca.sumsqr(W[:, k]) / self.process_std**2
        residuals = (ca.DM(self.C) @ X - Y) * mask
        cost += ca.sumsqr(residuals) / self.measurement_std**2
        opti.subject_to(opti.bounded(STATE_BOUNDS[0], ca.vec(X), STATE_BOUNDS[1]))
        opti.minimize(cost)
        opti.solver("ipopt", ipopt_options(200, 1e-8))
        problem = (opti, {"X": X, "W": W, "Y": Y, "mask": mask, "U": U, "D": D, "prior": prior})
        self._problems[steps] = problem
        return problem

    def _solve(self) -> None:
        steps = len(self._measurements) - 1
        opti, variables = self._problem(steps)
        measurements = np.array(self._measurements).T
        mask = (~np.isnan(measurements)).astype(float)
        opti.set_value(variables["Y"], np.nan_to_num(measurements))
        opti.set_value(variables["mask"], mask)
        if steps:
            opti.set_value(variables["U"], np.array(self._controls).T)
            opti.set_value(variables["D"], np.array(self._disturbances).T)
        else:
            opti.set_value(variables["U"], np.zeros((self.model.n_controls, 1)))
            opti.set_value(variables["D"], np.zeros((self.model.n_disturbances, 1)))
        opti.set_value(variables["prior"], self._prior)
        guess = self._guess(steps)
        opti.set_initial(variables["X"], guess)
        opti.set_initial(variables["W"], np.zeros((self.model.n_states, max(steps, 1))))
        try:
            solution = opti.solve()
        except RuntimeError:
            solution = opti.debug
        self._trajectory = np.asarray(solution.value(variables["X"]), dtype=float).reshape(self.model.n_states, -1)
        self._x = self._trajectory[:, -1].copy()

    def _guess(self, steps: int) -> np.ndarray:
        if self._trajectory is not None and self._trajectory.shape[1] == steps + 1:
            return self._trajectory
        if self._trajectory is not None and self._trajectory.shape[1] == steps:
            return np.hstack([self._trajectory, self._x.reshape(-1, 1)])
        return np.tile(self._prior.reshape(-1, 1), (1, steps + 1))
