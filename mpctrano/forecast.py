"""Disturbance forecasts of a :class:`~mpctrano.model.FastModel`.

A FastModel computes the solar position and the irradiance on its surfaces itself, so its disturbances are the raw
channels of the weather file, as the Buildings weather reader gives them to the runnable model, and the internal
gains per floor area of each occupancy model:

* ``weather_inputs.TDryBul``, ``.TBlaSky`` (black-body sky temperature, from the dew point and the opaque sky
  cover), ``.winSpe``, ``.winDir`` [rad], ``.HDirNor``, ``.HDifHor``, ``.HGloHor``;
* ``<occupancy>_gai.QRad`` and ``.QCon`` [W/m2], from the schedule of the occupancy model of the YAML.

:class:`WeatherForecaster` builds them from the TMY3 ``.mos`` weather file of the model and its occupancy
schedules, :class:`DataForecaster` replays recorded disturbances; :class:`ForecastNoise` adds forecast errors
growing with the lead time.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from mpctrano.units import KELVIN

if TYPE_CHECKING:
    from mpctrano.model import FastModel

DEFAULT_YEAR = 2023
"""Year used to give the typical meteorological year a calendar (2023 starts on a Sunday, like TMY3 files)."""
SECONDS_PER_YEAR = 365 * 86400
STEFAN_BOLTZMANN = 5.670374419e-8


def forecast_index(start: pd.Timestamp, steps: int, time_step: float) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start), periods=steps, freq=pd.Timedelta(seconds=time_step))


def seconds_of_year(index: pd.DatetimeIndex) -> np.ndarray:
    """Seconds since the start of the year of each timestamp (Modelica/Buildings convention)."""
    year_start = index.to_period("Y").start_time
    return np.asarray((index - year_start).total_seconds(), dtype=float)


class Forecaster(Protocol):
    """Provides the disturbances of a model over a horizon."""

    def forecast(self, start: pd.Timestamp, steps: int, time_step: float) -> pd.DataFrame:
        """Forecast available to the controller (possibly imperfect)."""

    def realisation(self, start: pd.Timestamp, steps: int, time_step: float) -> pd.DataFrame:
        """What actually happens (used by the plant in closed-loop simulations)."""


# --------------------------------------------------------------------------- weather
@dataclass(frozen=True)
class Location:
    latitude: float
    longitude: float
    time_zone: float
    """Hours from UTC."""


@dataclass
class WeatherFile:
    """TMY3 weather file in the Modelica ``.mos`` format read by ``Buildings.BoundaryConditions``.

    The hourly columns used here are the dry-bulb temperature, the global horizontal, direct
    normal and diffuse horizontal irradiance and, when the file has them, the dew point
    temperature, the wind direction and speed and the opaque sky cover (for the sky temperature).
    Values are linearly interpolated and wrap around the year, like the Buildings reader.
    """

    path: Path
    location: Location
    table: pd.DataFrame = field(repr=False)

    @classmethod
    def read(cls, path: Path | str) -> WeatherFile:
        path = Path(path)
        location: Location | None = None
        rows: list[list[float]] = []
        with path.open() as stream:
            for line in stream:
                if line.startswith("#LOCATION"):
                    fields = line.strip().split(",")
                    location = Location(
                        latitude=float(fields[6]), longitude=float(fields[7]), time_zone=float(fields[8])
                    )
                elif line.startswith(("#", "double")) or not line.strip():
                    continue
                else:
                    rows.append([float(value) for value in line.split()[:19]])
        if location is None:
            raise ValueError(f"No #LOCATION line in the weather file {path}.")
        width = min(len(row) for row in rows)
        table = pd.DataFrame([row[:width] for row in rows], columns=[f"c{i}" for i in range(1, width + 1)])
        table = table.set_index("c1").rename(columns=_WEATHER_COLUMNS)
        table["TDryBul"] += KELVIN
        if "TDewPoi" in table:
            table["TDewPoi"] += KELVIN
        if "winDir" in table:
            table["winDir"] = np.radians(table["winDir"])
        if "nOpa" in table:
            table["nOpa"] = table["nOpa"] / 10
        return cls(
            path=path, location=location, table=table[[name for name in _WEATHER_COLUMNS.values() if name in table]]
        )

    def values(self, variable: str, seconds: np.ndarray) -> np.ndarray:
        """Linear interpolation of a variable at seconds of the year (periodic)."""
        seconds = np.mod(np.asarray(seconds, dtype=float), SECONDS_PER_YEAR)
        times = np.append(self.table.index.to_numpy(dtype=float), SECONDS_PER_YEAR)
        series = self.table[variable].to_numpy(dtype=float)
        return np.asarray(np.interp(seconds, times, np.append(series, series[0])), dtype=float)

    def outdoor_temperature(self, seconds: np.ndarray) -> np.ndarray:
        return self.values("TDryBul", seconds)

    def sky_temperature(self, seconds: np.ndarray) -> np.ndarray:
        """Black-body sky temperature [K], as ``Buildings.BoundaryConditions.SkyTemperature.BlackBody`` computes it
        by default in the weather reader (from the dry-bulb and dew point temperatures and the opaque sky cover)."""
        missing = {"TDewPoi", "nOpa"} - set(self.table.columns)
        if missing:
            raise ValueError(f"The weather file {self.path} has no {sorted(missing)} column for the sky temperature.")
        dry_bulb = self.values("TDryBul", seconds)
        dew_point = np.minimum(self.values("TDewPoi", seconds), dry_bulb)
        cover = 10 * np.clip(self.values("nOpa", seconds), 0.0, 1.0)
        emissivity = (0.787 + 0.764 * np.log(dew_point / KELVIN)) * (
            1 + 0.0224 * cover - 0.0035 * cover**2 + 0.00028 * cover**3
        )
        return np.asarray(dry_bulb * np.clip(emissivity, 1e-3, None) ** 0.25, dtype=float)


_WEATHER_COLUMNS = {
    "c2": "TDryBul",
    "c3": "TDewPoi",
    "c9": "HGloHor",
    "c10": "HDirNor",
    "c11": "HDifHor",
    "c16": "winDir",
    "c17": "winSpe",
    "c19": "nOpa",
}
"""Columns of the TMY3 ``.mos`` files: name in the Buildings weather bus (``nOpa`` the opaque sky cover)."""


# --------------------------------------------------------------------------- occupancy
class OccupancySchedule(BaseModel):
    """Sensible internal gains of a zone, following Trano's ``SimpleOccupancy`` model.

    When occupied, the gains per floor area are ``gain_per_area``, the sum of the radiant and convective gains of
    the Trano model (``(35 + 70)/24 W/m2`` by default), split by ``radiant_fraction``.
    """

    start_hour: float = Field(7.0, ge=0, le=24, description="Start of the occupied period")
    end_hour: float = Field(19.0, ge=0, le=24, description="End of the occupied period")
    gain_per_area: float = Field((35 + 70) / 24, ge=0, description="Sensible gains when occupied [W/m2]")
    unoccupied_gain_per_area: float = Field(0.0, ge=0, description="Sensible gains when unoccupied [W/m2]")
    radiant_fraction: float = Field(35 / 105, ge=0, le=1, description="Radiant share of the sensible gains")
    weekend_occupied: bool = True

    @classmethod
    def from_parameters(cls, parameters: Mapping[str, str | float]) -> OccupancySchedule:
        """Schedule of a Trano ``SimpleOccupancy`` from its Modelica parameters (``occupancy``, ``gain``, ``k``)."""
        values = {key: str(value) for key, value in parameters.items()}
        start, end = cls._hours(values.get("occupancy", "3600 * {7, 19}"))
        gains = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+", values.get("gain", "[35; 70; 30]"))]
        fraction = _evaluate_fraction(values.get("k", "1/6/4"))
        if len(gains) < 2 or sum(gains[:2]) <= 0:
            return cls(start_hour=start, end_hour=end)
        return cls(
            start_hour=start,
            end_hour=end,
            gain_per_area=sum(gains[:2]) * fraction,
            radiant_fraction=gains[0] / sum(gains[:2]),
        )

    @staticmethod
    def _hours(expression: str) -> tuple[float, float]:
        numbers = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+", expression.rsplit("{", maxsplit=1)[-1])]
        if len(numbers) < 2:
            return 7.0, 19.0
        return numbers[0], numbers[1]

    def occupied(self, index: pd.DatetimeIndex) -> np.ndarray:
        hours = index.hour + index.minute / 60
        occupied = (hours >= self.start_hour) & (hours < self.end_hour)
        if not self.weekend_occupied:
            occupied &= index.dayofweek < 5
        return np.asarray(occupied, dtype=bool)

    def gains(self, index: pd.DatetimeIndex, floor_area: float) -> np.ndarray:
        """Sensible internal gains [W]."""
        return floor_area * self.gains_per_area(index)

    def gains_per_area(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Sensible internal gains per floor area [W/m2]."""
        return np.where(self.occupied(index), self.gain_per_area, self.unoccupied_gain_per_area)

    def radiant_and_convective(self, index: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        """Radiant and convective internal gains per floor area [W/m2] (the ``QRad`` and ``QCon`` of FastModel)."""
        gains = self.gains_per_area(index)
        return gains * self.radiant_fraction, gains * (1 - self.radiant_fraction)


def _evaluate_fraction(expression: str) -> float:
    numbers = [float(value) for value in re.findall(r"\d*\.?\d+", expression)]
    if not numbers:
        return 1 / 24
    value = numbers[0]
    for divisor in numbers[1:]:
        value /= divisor
    return value


# --------------------------------------------------------------------------- forecasters
class ForecastNoise(BaseModel):
    """Imperfect forecast: Gaussian errors added to the realisation, growing with the lead time."""

    temperature_std: float = Field(0.0, ge=0, description="Outdoor temperature error at the horizon end [K]")
    irradiance_relative_std: float = Field(0.0, ge=0, description="Relative irradiance error at the horizon end")
    gains_relative_std: float = Field(0.0, ge=0, description="Relative internal gains error at the horizon end")
    seed: int | None = None

    def perturb(self, realisation: pd.DataFrame, kinds: Mapping[str, str]) -> pd.DataFrame:
        """Forecast of the columns of ``realisation`` named in ``kinds`` (``temperature``, ``irradiance`` or
        ``gains``, in this order of draws); the other columns are kept."""
        steps = len(realisation)
        if steps == 0:
            return realisation
        generator = np.random.default_rng(self.seed)
        lead = np.linspace(1 / steps, 1.0, steps)
        forecast = realisation.copy()
        for name, kind in kinds.items():
            values = forecast[name].to_numpy(dtype=float)
            if kind == "temperature":
                values = values + generator.normal(0, 1, steps) * self.temperature_std * lead
            elif kind == "irradiance":
                values = np.clip(
                    values * (1 + generator.normal(0, 1, steps) * self.irradiance_relative_std * lead), 0, None
                )
            elif kind == "gains":
                values = np.clip(values * (1 + generator.normal(0, 1, steps) * self.gains_relative_std * lead), 0, None)
            else:
                raise ValueError(f"Unknown kind of disturbance {kind!r} for {name}.")
            forecast[name] = values
        return forecast


# --------------------------------------------------------------------------- forecasters
WEATHER_CHANNELS = ("TDryBul", "TBlaSky", "winSpe", "winDir", "HDirNor", "HDifHor", "HGloHor")
GAIN_CHANNELS = ("QRad", "QCon")
NOISE_KINDS = {
    "TDryBul": "temperature",
    "TBlaSky": "temperature",
    "HDirNor": "irradiance",
    "HDifHor": "irradiance",
    "HGloHor": "irradiance",
    "QRad": "gains",
    "QCon": "gains",
}
"""Kind of forecast error of each channel (the wind is kept)."""


def channel(name: str) -> tuple[str, str]:
    """``(owner, channel)`` of a disturbance: ``weather_inputs.TDryBul`` -> ``("weather_inputs", "TDryBul")``,
    ``occupancy_1_gai.QCon`` -> ``("occupancy_1", "QCon")`` (``occupancy_1.gai.QCon`` before Trano 0.57)."""
    owner, _, name_ = name.rpartition(".")
    return owner.removesuffix(".gai").removesuffix("_gai"), name_


def noise_kinds(names: list[str]) -> dict[str, str]:
    """Kind of forecast error of each disturbance named."""
    return {name: NOISE_KINDS[channel(name)[1]] for name in names if channel(name)[1] in NOISE_KINDS}


@dataclass
class WeatherForecaster:
    """Disturbances of a FastModel from a weather file and the occupancy schedules of the model."""

    model: FastModel
    weather: WeatherFile
    occupancy: dict[str, OccupancySchedule] = field(default_factory=dict)
    noise: ForecastNoise | None = None

    def __post_init__(self) -> None:
        unknown = [
            name for name in self.model.disturbance_names if channel(name)[1] not in (*WEATHER_CHANNELS, *GAIN_CHANNELS)
        ]
        if unknown:
            raise ValueError(f"Disturbances of {self.model.name} without a forecast: {unknown}")

    @classmethod
    def from_model(
        cls,
        model: FastModel,
        weather: WeatherFile | Path | str | None = None,
        occupancy: OccupancySchedule | dict[str, OccupancySchedule] | None = None,
        noise: ForecastNoise | None = None,
    ) -> WeatherForecaster:
        """Weather file and occupancy schedules of the model unless given (``occupancy`` one schedule for every
        occupancy model, or schedules by occupancy model name)."""
        if weather is None:
            if model.weather_path is None:
                raise ValueError(f"The weather file of the model ({model.weather_file}) is not found; give one.")
            weather = model.weather_path
        weather_file = weather if isinstance(weather, WeatherFile) else WeatherFile.read(weather)
        owners = {channel(name)[0] for name in model.disturbance_names if channel(name)[1] in GAIN_CHANNELS}
        schedules = {}
        for owner in sorted(owners):
            if isinstance(occupancy, OccupancySchedule):
                schedules[owner] = occupancy
            elif isinstance(occupancy, dict) and owner in occupancy:
                schedules[owner] = occupancy[owner]
            else:
                schedules[owner] = model.occupancy.get(owner, OccupancySchedule(gain_per_area=0.0))
        return cls(model=model, weather=weather_file, occupancy=schedules, noise=noise)

    def with_noise(self, noise: ForecastNoise | None) -> WeatherForecaster:
        return WeatherForecaster(model=self.model, weather=self.weather, occupancy=self.occupancy, noise=noise)

    def realisation(self, start: pd.Timestamp, steps: int, time_step: float) -> pd.DataFrame:
        """What happens: the weather at the middle of each step (the inputs are held over the step) and the gains
        at its start."""
        index = forecast_index(start, steps, time_step)
        seconds = seconds_of_year(index) + time_step / 2
        weather = {
            name: self.weather.sky_temperature(seconds) if name == "TBlaSky" else self.weather.values(name, seconds)
            for name in WEATHER_CHANNELS
        }
        gains = {
            owner: dict(zip(GAIN_CHANNELS, schedule.radiant_and_convective(index), strict=True))
            for owner, schedule in self.occupancy.items()
        }
        columns = {}
        for name in self.model.disturbance_names:
            owner, name_ = channel(name)
            columns[name] = gains[owner][name_] if name_ in GAIN_CHANNELS else weather[name_]
        return pd.DataFrame(columns, index=index)

    def forecast(self, start: pd.Timestamp, steps: int, time_step: float) -> pd.DataFrame:
        """What the controller expects: the realisation with the errors of :attr:`noise`."""
        realisation = self.realisation(start, steps, time_step)
        if self.noise is None:
            return realisation
        return self.noise.perturb(realisation, noise_kinds(self.model.disturbance_names))


@dataclass
class DataForecaster:
    """Replays recorded disturbances (e.g. the columns of the monitoring data)."""

    frame: pd.DataFrame
    noise: ForecastNoise | None = None

    def realisation(self, start: pd.Timestamp, steps: int, time_step: float) -> pd.DataFrame:
        index = forecast_index(start, steps, time_step)
        missing = index.difference(pd.DatetimeIndex(self.frame.index))
        if len(missing):
            raise ValueError(f"The recorded data does not cover {missing[0]} .. {missing[-1]}.")
        return self.frame.loc[index]

    def forecast(self, start: pd.Timestamp, steps: int, time_step: float) -> pd.DataFrame:
        realisation = self.realisation(start, steps, time_step)
        if self.noise is None:
            return realisation
        return self.noise.perturb(realisation, noise_kinds([str(column) for column in realisation.columns]))
