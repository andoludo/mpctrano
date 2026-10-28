"""Monitoring data of a building: the measurements used for calibration and state estimation.

The data is a regularly sampled, time-indexed :class:`pandas.DataFrame` whose columns are named after the variables of
the :class:`~mpctrano.model.BuildingModel` of the building:

* the measured zone air temperatures ``<zone>.TAir`` [K] (at least one zone);
* the controls applied, held from each sample to the next (``valve_001_y.y``, ``boiler_001_y.y``, ...);
* optionally, the readings of the energy meters [kWh] (``boiler_001.EEle``, ``boiler_001.EFue``, ...), cumulative;
* optionally, the disturbances (``weather_inputs.TDryBul``, ``occupancy_1_gai.QCon``, ...); the missing ones are
  filled with a forecaster (:meth:`MonitoringData.fill_disturbances`), e.g. from the weather file of the model.

Temperatures are stored in Kelvin; :meth:`MonitoringData.from_csv` converts from Celsius.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from mpctrano.units import KELVIN

if TYPE_CHECKING:
    from mpctrano.forecast import Forecaster
    from mpctrano.model import BuildingModel

TEMPERATURE_SUFFIXES = (".TAir", ".TDryBul", ".TBlaSky")


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
    return column.endswith(TEMPERATURE_SUFFIXES)


class DataReport(BaseModel):
    """What the data offers to a model."""

    samples: int
    time_step: float
    measured_zones: list[str] = Field(description="Zone air temperatures measured")
    controls: list[str]
    meters: list[str] = Field(description="Energy meters read")
    disturbances: list[str]
    missing_controls: list[str]
    missing_disturbances: list[str]
    columns_with_gaps: list[str]

    @property
    def complete(self) -> bool:
        return not (self.missing_controls or self.missing_disturbances or self.columns_with_gaps)

    @property
    def usable(self) -> bool:
        """Enough to calibrate: an air temperature and every control (the disturbances can be filled)."""
        return bool(self.measured_zones) and not self.missing_controls


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

        ``temperature_unit="C"`` converts the temperature columns (``*.TAir``, ``*.TDryBul``, ``*.TBlaSky``) to
        Kelvin, ``rename`` maps the CSV headers to the names of the model.
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

    def fill_disturbances(
        self, model: BuildingModel, forecaster: Forecaster, overwrite: bool = False
    ) -> MonitoringData:
        """Add the disturbance columns the data lacks with the realisation of a forecaster."""
        missing = [name for name in model.disturbance_names if overwrite or name not in self.frame.columns]
        if not missing:
            return self
        realisation = forecaster.realisation(self.start, len(self.frame), self.time_step)
        return self.with_columns(realisation[missing])

    # ----------------------------------------------------------------- model binding
    def validate(self, model: BuildingModel) -> DataReport:
        columns = set(self.columns)
        return DataReport(
            samples=len(self.frame),
            time_step=self.time_step,
            measured_zones=[name for name in model.comfort_states if name in columns],
            controls=[name for name in model.control_names if name in columns],
            meters=[name for name in model.meters if name in columns],
            disturbances=[name for name in model.disturbance_names if name in columns],
            missing_controls=[name for name in model.control_names if name not in columns],
            missing_disturbances=[name for name in model.disturbance_names if name not in columns],
            columns_with_gaps=[column for column in self.columns if self.frame[column].isna().any()],
        )
