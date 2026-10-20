"""State estimation of a FastModel: moving horizon estimation over a sliding window of the past.

The FastModel is nonlinear (water flows, boiler and heat pump maps) and only the zone air temperatures are measured:
the walls, the water of the circuits and the buffers are not. :class:`MovingHorizonEstimator` keeps the inputs
applied and the measurements of the last ``window`` steps and, at each :meth:`~MovingHorizonEstimator.correct`,
finds the trajectory of the model closest to them with :class:`trano_optimization.MovingHorizonEstimation`
(Radau collocation, IPOPT): the measured air temperatures, the model errors over each step (process noise) and the
deviation of the first state of the window from its prior (arrival cost) are weighted by their standard deviations.
When the window slides, the estimate of its second state becomes the prior of the next window.

Every estimator follows the same cycle (:class:`StateEstimator`): :meth:`correct` with the measurement of the current
step, use the state, then :meth:`predict` with the inputs applied during the step. :class:`PerfectEstimator` returns
the true state of a simulated plant, to study the controller alone.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from trano_optimization import EstimationSettings, MovingHorizonEstimation

from mpctrano.forecast import seconds_of_year
from mpctrano.model import FastModel
from mpctrano.units import STATE_BOUNDS


class StateEstimator(Protocol):
    @property
    def state(self) -> np.ndarray: ...

    def correct(self, measurement: np.ndarray) -> np.ndarray:
        """Incorporate the measured zone air temperatures (NaN for a missing sensor); returns the estimate."""

    def predict(self, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        """Propagate the estimate over one step with the applied inputs."""


class _HasState(Protocol):
    @property
    def state(self) -> np.ndarray: ...


@dataclass
class PerfectEstimator:
    """The true state of a simulated plant (no estimation): to study the controller alone."""

    plant: _HasState

    @property
    def state(self) -> np.ndarray:
        return self.plant.state

    def correct(self, measurement: np.ndarray) -> np.ndarray:
        return self.plant.state

    def predict(self, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        return self.plant.state


class EstimatorSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    window: int = Field(12, ge=1, description="Steps of the past window")
    measurement_noise: float = Field(0.1, gt=0, description="Standard deviation of a measurement [K]")
    process_noise: float = Field(0.05, gt=0, description="Standard deviation of the model error over a step [K]")
    arrival_noise: float = Field(2.0, gt=0, description="Standard deviation of the prior of the first state [K]")
    degree: int = Field(3, ge=1, le=9, description="Radau collocation points per step")


@dataclass
class MovingHorizonEstimator:
    """Moving horizon estimation of the full state of a FastModel from its zone air temperatures."""

    model: FastModel
    time_step: float
    start: pd.Timestamp | str
    x0: np.ndarray | None = None
    settings: EstimatorSettings = field(default_factory=EstimatorSettings)
    _x: np.ndarray = field(init=False, repr=False)
    _prior: np.ndarray = field(init=False, repr=False)
    _window_start: pd.Timestamp = field(init=False, repr=False)
    _controls: deque[np.ndarray] = field(init=False, repr=False)
    _disturbances: deque[np.ndarray] = field(init=False, repr=False)
    _measurements: deque[np.ndarray] = field(init=False, repr=False)
    statuses: list[str] = field(init=False, default_factory=list)
    """IPOPT status of each estimation."""

    def __post_init__(self) -> None:
        self.start = pd.Timestamp(self.start)
        self._x = self.model.initial_state.copy() if self.x0 is None else np.asarray(self.x0, dtype=float).copy()
        self._prior = self._x.copy()
        self._window_start = self.start
        window = self.settings.window
        self._controls = deque(maxlen=window)
        self._disturbances = deque(maxlen=window)
        self._measurements = deque(maxlen=window)  # the current measurement is added in correct()
        self._estimates: np.ndarray | None = None  # the states of the last window, from its estimation
        self._pending: np.ndarray | None = None  # measurement of the current step, before predict
        settings = self.settings
        self.estimator = MovingHorizonEstimation(
            self.model.model,
            self.model.roles,
            EstimationSettings(
                time_step=self.time_step,
                degree=settings.degree,
                temperature_bounds=STATE_BOUNDS,
                measurement_noise=settings.measurement_noise,
                process_noise=settings.process_noise,
                arrival_noise=settings.arrival_noise,
            ),
            self.model.parameters,
        )

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    def correct(self, measurement: np.ndarray) -> np.ndarray:
        """The estimate of the current state given the measured zone air temperatures (NaN when missing)."""
        measurement = np.asarray(measurement, dtype=float).reshape(-1)
        self._pending = measurement
        if not self._controls:  # nothing applied yet: the prior, with the air temperatures measured
            measured = ~np.isnan(measurement)
            indices = np.asarray(self.model.comfort_indices)[measured]
            self._x[indices] = measurement[measured]
            return self.state
        measurements = np.column_stack([*self._measurements, measurement])
        start_time = float(seconds_of_year(pd.DatetimeIndex([self._window_start]))[0])
        result = self.estimator.estimate(
            self._prior,
            np.column_stack(self._controls),
            np.column_stack(self._disturbances),
            measurements,
            start_time=start_time,
        )
        self.statuses.append(result.status)
        self._estimates = result.states if result.success else None
        if result.success:
            self._x = result.current.copy()
        return self.state

    def predict(self, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        """Record the inputs applied over the step and propagate the estimate (CVODES)."""
        control = np.asarray(control, dtype=float).reshape(-1)
        disturbance = np.asarray(disturbance, dtype=float).reshape(-1)
        if self._pending is None:
            raise RuntimeError("correct() must be called with the measurement of the step before predict().")
        now = self._window_start + pd.Timedelta(seconds=len(self._controls) * self.time_step)
        if len(self._controls) == self.settings.window:  # the window slides: its second state is the next prior
            if self._estimates is not None:
                self._prior = self._estimates[:, 1].copy()
            else:  # the last estimation failed: propagate the prior over the step that leaves the window
                seconds = float(seconds_of_year(pd.DatetimeIndex([self._window_start]))[0])
                self._prior = self.model.simulate(
                    self._prior,
                    self._controls[0].reshape(-1, 1),
                    self._disturbances[0].reshape(-1, 1),
                    self.time_step,
                    start_time=seconds,
                )[:, -1]
            self._window_start += pd.Timedelta(seconds=self.time_step)
        if not self._measurements:
            self._prior = self._x.copy()  # the first window starts from the corrected prior
        self._measurements.append(self._pending)
        self._controls.append(control)
        self._disturbances.append(disturbance)
        self._pending = None
        start_time = float(seconds_of_year(pd.DatetimeIndex([now]))[0])
        trajectory = self.model.simulate(
            self._x, control.reshape(-1, 1), disturbance.reshape(-1, 1), self.time_step, start_time=start_time
        )
        self._x = trajectory[:, -1].copy()
        return self.state
