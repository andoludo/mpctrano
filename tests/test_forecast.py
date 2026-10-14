import math

import numpy as np
import pandas as pd
import pytest

from mpctrano import DataForecaster, ForecastNoise, OccupancySchedule, RCModel, WeatherFile, WeatherForecaster
from mpctrano.forecast import SolarGeometry, seconds_of_year
from mpctrano.model import KELVIN
from tests.conftest import WEATHER


@pytest.fixture(scope="module")
def weather() -> WeatherFile:
    return WeatherFile.read(WEATHER)


def test_weather_file_location_and_table(weather: WeatherFile) -> None:
    assert weather.location.latitude == pytest.approx(50.7969)
    assert weather.location.time_zone == 1.0
    assert len(weather.table) == 8760
    assert weather.outdoor_temperature(np.array([0.0])) == pytest.approx(273.15 + 7.1)
    assert weather.outdoor_temperature(np.array([365 * 86400.0 + 1800])) == pytest.approx(273.15 + 7.1)  # wraps


def test_solar_geometry_noon_in_june(weather: WeatherFile) -> None:
    geometry = SolarGeometry(weather.location)
    noon = pd.Timestamp("2023-06-21 12:30")
    zenith, azimuth = geometry.position(seconds_of_year(pd.DatetimeIndex([noon])))
    assert math.degrees(zenith[0]) == pytest.approx(90 - (90 - 50.8 + 23.4), abs=3)  # solar elevation
    assert abs(math.degrees(azimuth[0])) < 15  # the sun is south at solar noon
    midnight = geometry.position(seconds_of_year(pd.DatetimeIndex([pd.Timestamp("2023-06-21 00:30")])))[0]
    assert math.degrees(midnight[0]) > 90


def test_tilted_irradiance_orientations(weather: WeatherFile) -> None:
    seconds = seconds_of_year(pd.date_range("2023-03-20", periods=24, freq="h")) + 1800
    south = weather.tilted_irradiance(seconds, azimuth=0, tilt=90)
    north = weather.tilted_irradiance(seconds, azimuth=180, tilt=90)
    roof = weather.tilted_irradiance(seconds, azimuth=0, tilt=0)
    assert np.all(south >= 0) and np.all(north >= 0)
    assert south[:5].max() == 0  # night
    assert south.sum() > north.sum()
    assert roof.sum() == pytest.approx(weather.values("HGloHor", seconds).sum(), rel=0.3)


def test_occupancy_schedule_from_the_interface(model: RCModel) -> None:
    source = next(signal.source for signal in model.interface.disturbances if signal.name == "space_001_QInt")
    schedule = OccupancySchedule.from_source(source)
    assert (schedule.start_hour, schedule.end_hour) == (7, 19)
    assert schedule.gain_per_area == pytest.approx(105 / 24)
    index = pd.date_range("2023-01-16", periods=24, freq="h")
    gains = schedule.gains(index, 100.0)
    assert gains[6] == 0 and gains[7] == pytest.approx(437.5) and gains[19] == 0
    weekday_only = OccupancySchedule(weekend_occupied=False)
    assert weekday_only.gains(pd.date_range("2023-01-15 12:00", periods=1, freq="h"), 100.0)[0] == 0  # Sunday


def test_weather_forecaster_columns_and_noise(model: RCModel) -> None:
    perfect = WeatherForecaster.from_model(model)
    start = pd.Timestamp("2023-01-16")
    realisation = perfect.realisation(start, 48, 3600)
    assert list(realisation.columns) == model.disturbance_names
    assert len(realisation) == 48
    assert realisation["TOut"].between(250, 310).all()
    assert realisation["space_002_QInt"].max() == pytest.approx(70 * 105 / 24)
    noisy = WeatherForecaster.from_model(model, noise=ForecastNoise(temperature_std=2.0, seed=0))
    forecast = noisy.forecast(start, 48, 3600)
    assert forecast["TOut"].to_numpy() != pytest.approx(realisation["TOut"].to_numpy())
    assert noisy.realisation(start, 48, 3600).equals(realisation)
    assert (forecast["HSol_azi0_til90"] >= 0).all()


def test_weather_forecaster_with_custom_occupancy(model: RCModel) -> None:
    forecaster = WeatherForecaster.from_model(model, occupancy=OccupancySchedule(gain_per_area=0.0))
    realisation = forecaster.realisation(pd.Timestamp("2023-01-16"), 24, 3600)
    assert realisation["space_001_QInt"].abs().max() == 0


def test_data_forecaster_replays_and_checks_coverage(model: RCModel) -> None:
    index = pd.date_range("2023-01-16", periods=30, freq="h")
    frame = pd.DataFrame(1.0, index=index, columns=model.disturbance_names)
    forecaster = DataForecaster(frame)
    assert len(forecaster.forecast(index[0], 24, 3600)) == 24
    with pytest.raises(ValueError, match="does not cover"):
        forecaster.forecast(index[10], 24, 3600)


def test_weather_file_reads_the_channels_of_the_sky_and_the_wind():
    weather = WeatherFile.read(WEATHER)
    assert {"TDewPoi", "winDir", "winSpe", "nOpa"} <= set(weather.table.columns)
    seconds = np.arange(0, 48) * 3600.0
    assert np.all((weather.values("winDir", seconds) >= 0) & (weather.values("winDir", seconds) <= 2 * np.pi))
    assert np.all((weather.values("nOpa", seconds) >= 0) & (weather.values("nOpa", seconds) <= 1))
    sky = weather.sky_temperature(seconds)
    assert np.all(sky < weather.outdoor_temperature(seconds))
    assert np.all(sky > weather.outdoor_temperature(seconds) - 30)


def test_sky_temperature_follows_the_black_body_model_of_buildings():
    table = pd.DataFrame(
        {"TDryBul": [KELVIN, KELVIN], "TDewPoi": [KELVIN, KELVIN], "nOpa": [0.0, 1.0]}, index=[0.0, 3600.0]
    )
    weather = WeatherFile(path=WEATHER, location=WeatherFile.read(WEATHER).location, table=table)
    clear, overcast = weather.sky_temperature(np.array([0.0, 3600.0]))
    assert clear == pytest.approx(KELVIN * 0.787**0.25)
    assert overcast == pytest.approx(KELVIN * (0.787 * (1 + 0.224 - 0.35 + 0.28)) ** 0.25)


def test_occupancy_schedule_splits_the_radiant_and_convective_gains():
    schedule = OccupancySchedule.from_parameters({"gain": "[40; 75; 40]", "k": "1/7/3", "occupancy": "3600*{9, 17}"})
    assert (schedule.start_hour, schedule.end_hour) == (9.0, 17.0)
    index = pd.date_range("2023-01-16", periods=24, freq="1h")
    radiant, convective = schedule.radiant_and_convective(index)
    occupied = (index.hour >= 9) & (index.hour < 17)
    np.testing.assert_allclose(radiant, np.where(occupied, 40 / 21, 0.0))
    np.testing.assert_allclose(convective, np.where(occupied, 75 / 21, 0.0))
    np.testing.assert_allclose(schedule.gains(index, 10.0), 10 * (radiant + convective))


def test_forecast_noise_perturbs_the_named_channels_only():
    index = pd.date_range("2023-01-16", periods=12, freq="1h")
    frame = pd.DataFrame({"T": np.full(12, 280.0), "H": np.full(12, 100.0), "wind": np.full(12, 3.0)}, index=index)
    noise = ForecastNoise(temperature_std=1.0, irradiance_relative_std=0.2, seed=0)
    forecast = noise.perturb(frame, {"T": "temperature", "H": "irradiance"})
    assert not np.allclose(forecast["T"], frame["T"])
    assert not np.allclose(forecast["H"], frame["H"])
    pd.testing.assert_series_equal(forecast["wind"], frame["wind"])
    with pytest.raises(ValueError, match="Unknown kind"):
        noise.perturb(frame, {"wind": "wind"})
