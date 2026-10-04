"""Monitoring data of a building: the measurements used for calibration and state estimation.

The data is a regularly sampled, time-indexed :class:`pandas.DataFrame` whose columns are named
after the signals of the Trano MPC interface:

* measured zone temperatures ``<zone>_Ti`` [K] (at least one zone),
* heating powers ``<zone>_QHea`` [W] (the control inputs),
* disturbances ``TOut`` [K], ``HSol_<orientation>`` [W/m2] and ``<zone>_QInt`` [W]. The
  disturbances that are not measured can be filled with a forecaster
  (:meth:`MonitoringData.fill_disturbances`), e.g. the solar irradiance from the weather file and
  the internal gains from the occupancy schedule.

Temperatures are stored in Kelvin; :meth:`MonitoringData.from_csv` converts from Celsius.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from mpctrano.model import KELVIN

if TYPE_CHECKING:
    from mpctrano.forecast import Forecaster
    from mpctrano.model import RCModel

TEMPERATURE_SUFFIXES = ("_Ti", "_Te", "_Th", "_Tm")


class DataError(ValueError):
    """The monitoring data cannot be used with the model."""


def time_step_of(index: pd.Index, tolerance: float = 1e-6) -> float:
    """Sampling period [s] of a regular :class:`pandas.DatetimeIndex`."""
    if not isinstance(index, pd.DatetimeIndex):
        raise DataError("The data must have a DatetimeIndex.")
    if len(index) < 2:
        raise DataError("At least two samples are needed.")
    steps = np.diff(index.to_numpy(dtype="datetime64[ns]")).astype("timedelta64[ns]").astype(float) / 1e9
    if np.max(np.abs(steps - steps[0])) > tolerance * max(steps[0], 1.0):
        raise DataError("The data must be regularly sampled; resample it first.")
    return float(steps[0])


def is_temperature(column: str) -> bool:
    return column == "TOut" or column.endswith(TEMPERATURE_SUFFIXES)


class DataReport(BaseModel):
    """Result of :meth:`MonitoringData.validate` for a model."""

    samples: int
    time_step: float
    measured_states: list[str]
    controls: list[str]
    disturbances: list[str]
    missing_controls: list[str] = Field(default_factory=list)
    missing_disturbances: list[str] = Field(default_factory=list)
    columns_with_gaps: list[str] = Field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not (self.missing_controls or self.missing_disturbances or self.columns_with_gaps)

    @property
    def usable(self) -> bool:
        """Enough for calibration once the missing disturbances are filled."""
        return bool(self.measured_states) and not self.missing_controls and not self.columns_with_gaps


@dataclass(frozen=True)
class Signals:
    """Arrays extracted from monitoring data for a model (inputs act between two samples)."""

    index: pd.DatetimeIndex
    time_step: float
    measurements: np.ndarray
    """``(n_measured, N + 1)`` measured states."""
    measured_states: list[int]
    """Indices of the measured states in the state vector."""
    controls: np.ndarray
    """``(n_controls, N)``."""
    disturbances: np.ndarray
    """``(n_disturbances, N)``."""

    @property
    def steps(self) -> int:
        return int(self.controls.shape[1])


@dataclass(frozen=True)
class MonitoringData:
    """Regularly sampled measurements, see the module documentation for the column names."""

    frame: pd.DataFrame

    def __post_init__(self) -> None:
        if not isinstance(self.frame.index, pd.DatetimeIndex):
            raise DataError("Monitoring data needs a DatetimeIndex.")
        if not self.frame.index.is_monotonic_increasing:
            raise DataError("Monitoring data must be sorted by time.")
        time_step_of(self.frame.index)

    # ----------------------------------------------------------------- IO
    @classmethod
    def from_csv(
        cls,
        path: Path | str,
        time_column: str | None = None,
        temperature_unit: str = "K",
        rename: dict[str, str] | None = None,
    ) -> MonitoringData:
        """Read a CSV file; ``time_column`` defaults to the first column.

        ``temperature_unit="C"`` converts the temperature columns (``TOut``, ``*_Ti``...) to Kelvin,
        ``rename`` maps the CSV headers to the interface names.
        """
        frame = pd.read_csv(path)
        time_column = time_column or str(frame.columns[0])
        frame[time_column] = pd.to_datetime(frame[time_column])
        frame = frame.set_index(time_column).sort_index()
        frame.index.name = "time"
        if rename:
            frame = frame.rename(columns=rename)
        if temperature_unit.upper() == "C":
            for column in frame.columns:
                if is_temperature(str(column)):
                    frame[column] = frame[column] + KELVIN
        elif temperature_unit.upper() != "K":
            raise DataError("temperature_unit must be 'K' or 'C'.")
        return cls(frame.astype(float))

    def to_csv(self, path: Path | str, temperature_unit: str = "K") -> Path:
        frame = self.frame.copy()
        if temperature_unit.upper() == "C":
            for column in frame.columns:
                if is_temperature(str(column)):
                    frame[column] = frame[column] - KELVIN
        path = Path(path)
        frame.to_csv(path, index_label="time")
        return path

    # ----------------------------------------------------------------- properties
    @property
    def index(self) -> pd.DatetimeIndex:
        return self.frame.index  # type: ignore[return-value]

    @property
    def time_step(self) -> float:
        return time_step_of(self.frame.index)

    @property
    def columns(self) -> list[str]:
        return [str(column) for column in self.frame.columns]

    def __len__(self) -> int:
        return len(self.frame)

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.frame.index[0])

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.frame.index[-1])

    # ----------------------------------------------------------------- transformations
    def resample(self, time_step: float) -> MonitoringData:
        """Average the data on a new sampling period [s] (missing values are interpolated)."""
        frame = self.frame.resample(pd.Timedelta(seconds=time_step)).mean().interpolate(limit_direction="both")
        return MonitoringData(frame)

    def split(
        self, fraction: float | None = None, at: pd.Timestamp | str | None = None
    ) -> tuple[MonitoringData, MonitoringData]:
        """Split into a training and a validation period, by fraction of samples or at a timestamp."""
        if (fraction is None) == (at is None):
            raise DataError("Give either a fraction or a timestamp.")
        if fraction is not None:
            if not 0 < fraction < 1:
                raise DataError("The fraction must be in (0, 1).")
            cut = round(len(self.frame) * fraction)
        else:
            cut = int(self.frame.index.searchsorted(pd.Timestamp(at)))  # type: ignore[arg-type]
        if cut < 2 or len(self.frame) - cut < 2:
            raise DataError("Both periods need at least two samples.")
        # The validation period starts at the last training sample (the state is continuous).
        return MonitoringData(self.frame.iloc[:cut]), MonitoringData(self.frame.iloc[cut - 1 :])

    def window(self, start: pd.Timestamp | str, end: pd.Timestamp | str) -> MonitoringData:
        return MonitoringData(self.frame.loc[pd.Timestamp(start) : pd.Timestamp(end)])

    def with_columns(self, columns: pd.DataFrame | dict[str, np.ndarray]) -> MonitoringData:
        frame = self.frame.copy()
        for name, values in columns.items():
            frame[name] = np.asarray(values, dtype=float)
        return MonitoringData(frame)

    def fill_disturbances(self, model: RCModel, forecaster: Forecaster, overwrite: bool = False) -> MonitoringData:
        """Add the disturbance columns the data lacks with the realisation of a forecaster."""
        missing = [name for name in model.disturbance_names if overwrite or name not in self.frame.columns]
        if not missing:
            return self
        realisation = forecaster.realisation(self.start, len(self.frame), self.time_step)
        return self.with_columns(realisation[missing])

    # ----------------------------------------------------------------- model binding
    def validate(self, model: RCModel) -> DataReport:
        columns = set(self.columns)
        measured_states = [name for name in model.comfort_states if name in columns]
        controls = [name for name in model.control_names if name in columns]
        disturbances = [name for name in model.disturbance_names if name in columns]
        return DataReport(
            samples=len(self.frame),
            time_step=self.time_step,
            measured_states=measured_states,
            controls=controls,
            disturbances=disturbances,
            missing_controls=[name for name in model.control_names if name not in columns],
            missing_disturbances=[name for name in model.disturbance_names if name not in columns],
            columns_with_gaps=[column for column in self.columns if self.frame[column].isna().any()],
        )

    def signals(self, model: RCModel) -> Signals:
        """Arrays for the model; raises :class:`DataError` when a control or disturbance is missing."""
        report = self.validate(model)
        if not report.measured_states:
            raise DataError(f"No measured zone temperature among {model.comfort_states}.")
        if report.missing_controls or report.missing_disturbances:
            raise DataError(
                f"Missing columns: {report.missing_controls + report.missing_disturbances}. "
                "Use fill_disturbances() for the disturbances."
            )
        if report.columns_with_gaps:
            raise DataError(f"Columns with missing values: {report.columns_with_gaps}.")
        measured = [name for name in model.state_names if name in self.frame.columns]
        return Signals(
            index=self.index,
            time_step=report.time_step,
            measurements=self.frame[measured].to_numpy(dtype=float).T,
            measured_states=[model.state_names.index(name) for name in measured],
            controls=self.frame[model.control_names].to_numpy(dtype=float).T[:, :-1],
            disturbances=self.frame[model.disturbance_names].to_numpy(dtype=float).T[:, :-1],
        )

    def initial_state(self, model: RCModel) -> np.ndarray:
        """Initial state guess: every state of a zone starts at the measured zone temperature."""
        x0 = model.initial_state.copy()
        for zone in model.interface.zones:
            if zone.indoor_temperature in self.frame.columns:
                temperature = float(self.frame[zone.indoor_temperature].iloc[0])
                for index, state in enumerate(model.interface.states):
                    if state.zone == zone.name:
                        x0[index] = temperature
        return x0
