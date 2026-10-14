"""Disturbance forecasts of a :class:`~mpctrano.fast_model.FastModel`.

A FastModel computes the solar position and the irradiance on its surfaces itself, so its disturbances are the raw
channels of the weather file, as the Buildings weather reader gives them to the runnable model, and the internal
gains per floor area of each occupancy model:

* ``weather_inputs.TDryBul``, ``.TBlaSky`` (black-body sky temperature, from the dew point and the opaque sky
  cover), ``.winSpe``, ``.winDir`` [rad], ``.HDirNor``, ``.HDifHor``, ``.HGloHor``;
* ``<occupancy>.gai.QRad`` and ``.QCon`` [W/m2], from the schedule of the occupancy model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from mpctrano.fast_model import FastModel
from mpctrano.forecast import ForecastNoise, OccupancySchedule, WeatherFile, forecast_index, seconds_of_year

WEATHER_CHANNELS = ("TDryBul", "TBlaSky", "winSpe", "winDir", "HDirNor", "HDifHor", "HGloHor")
GAIN_CHANNELS = ("QRad", "QCon")
_NOISE_KINDS = {
    "TDryBul": "temperature",
    "TBlaSky": "temperature",
    "HDirNor": "irradiance",
    "HDifHor": "irradiance",
    "HGloHor": "irradiance",
    "QRad": "gains",
    "QCon": "gains",
}


def _channel(name: str) -> tuple[str, str]:
    """``(owner, channel)`` of a disturbance: ``weather_inputs.TDryBul`` -> ``("weather_inputs", "TDryBul")``,
    ``occupancy_1.gai.QCon`` -> ``("occupancy_1", "QCon")``."""
    owner, _, channel = name.rpartition(".")
    return owner.removesuffix(".gai"), channel


@dataclass
class FastWeatherForecaster:
    """Disturbances of a FastModel from a weather file and the occupancy schedules of the model."""

    model: FastModel
    weather: WeatherFile
    occupancy: dict[str, OccupancySchedule] = field(default_factory=dict)
    noise: ForecastNoise | None = None

    def __post_init__(self) -> None:
        unknown = [
            name
            for name in self.model.disturbance_names
            if _channel(name)[1] not in (*WEATHER_CHANNELS, *GAIN_CHANNELS)
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
    ) -> FastWeatherForecaster:
        """Weather file and occupancy schedules of the model unless given (``occupancy`` one schedule for every
        occupancy model, or schedules by occupancy model name)."""
        if weather is None:
            if model.weather_path is None:
                raise ValueError(f"The weather file of the model ({model.weather_file}) is not found; give one.")
            weather = model.weather_path
        weather_file = weather if isinstance(weather, WeatherFile) else WeatherFile.read(weather)
        owners = {_channel(name)[0] for name in model.disturbance_names if _channel(name)[1] in GAIN_CHANNELS}
        schedules = {}
        for owner in sorted(owners):
            if isinstance(occupancy, OccupancySchedule):
                schedules[owner] = occupancy
            elif isinstance(occupancy, dict) and owner in occupancy:
                schedules[owner] = occupancy[owner]
            else:
                schedules[owner] = model.occupancy.get(owner, OccupancySchedule(gain_per_area=0.0))
        return cls(model=model, weather=weather_file, occupancy=schedules, noise=noise)

    def realisation(self, start: pd.Timestamp, steps: int, time_step: float) -> pd.DataFrame:
        """What happens: the weather at the middle of each step (the inputs are held over the step) and the gains
        at its start, as for the RC models."""
        index = forecast_index(start, steps, time_step)
        seconds = seconds_of_year(index) + time_step / 2
        weather = {
            channel: self.weather.sky_temperature(seconds)
            if channel == "TBlaSky"
            else self.weather.values(channel, seconds)
            for channel in WEATHER_CHANNELS
        }
        gains = {
            owner: dict(zip(GAIN_CHANNELS, schedule.radiant_and_convective(index), strict=True))
            for owner, schedule in self.occupancy.items()
        }
        columns = {}
        for name in self.model.disturbance_names:
            owner, channel = _channel(name)
            columns[name] = gains[owner][channel] if channel in GAIN_CHANNELS else weather[channel]
        return pd.DataFrame(columns, index=index)

    def forecast(self, start: pd.Timestamp, steps: int, time_step: float) -> pd.DataFrame:
        """What the controller expects: the realisation with the errors of :attr:`noise`."""
        realisation = self.realisation(start, steps, time_step)
        if self.noise is None:
            return realisation
        kinds = {
            name: _NOISE_KINDS[_channel(name)[1]]
            for name in self.model.disturbance_names
            if _channel(name)[1] in _NOISE_KINDS
        }
        return self.noise.perturb(realisation, kinds)
