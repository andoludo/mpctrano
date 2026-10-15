import numpy as np
import pandas as pd
import pytest

from mpctrano import DataForecaster, FastModel, ForecastNoise, OccupancySchedule, WeatherFile, WeatherForecaster
from mpctrano.units import KELVIN
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


def test_weather_forecaster_gives_every_disturbance(heat_pump: FastModel, forecaster: WeatherForecaster) -> None:
    frame = forecaster.realisation(pd.Timestamp("2023-01-16"), 24, 3600.0)
    assert frame.columns.tolist() == heat_pump.disturbance_names
    assert frame.notna().all().all()
    occupied = (frame.index.hour >= 7) & (frame.index.hour < 22)
    np.testing.assert_allclose(frame["occupancy_1.gai.QRad"], np.where(occupied, 40 / 21, 0.0))
    np.testing.assert_allclose(frame["occupancy_1.gai.QCon"], np.where(occupied, 75 / 21, 0.0))
    assert np.all(frame["weather_inputs.TBlaSky"] < frame["weather_inputs.TDryBul"])


def test_weather_forecaster_noise_and_custom_occupancy(heat_pump: FastModel) -> None:
    start = pd.Timestamp("2023-01-16")
    perfect = WeatherForecaster.from_model(heat_pump).realisation(start, 24, 3600.0)
    noisy = WeatherForecaster.from_model(heat_pump, noise=ForecastNoise(temperature_std=2.0, seed=0))
    forecast = noisy.forecast(start, 24, 3600.0)
    assert not np.allclose(forecast["weather_inputs.TDryBul"], perfect["weather_inputs.TDryBul"])
    pd.testing.assert_series_equal(forecast["weather_inputs.winSpe"], perfect["weather_inputs.winSpe"])
    pd.testing.assert_frame_equal(noisy.realisation(start, 24, 3600.0), perfect)
    empty = WeatherForecaster.from_model(heat_pump, occupancy=OccupancySchedule(gain_per_area=0.0))
    assert empty.realisation(start, 24, 3600.0)["occupancy_1.gai.QCon"].sum() == 0.0


def test_data_forecaster_replays_and_checks_coverage(heat_pump: FastModel) -> None:
    index = pd.date_range("2023-01-16", periods=30, freq="h")
    frame = pd.DataFrame(280.0, index=index, columns=heat_pump.disturbance_names)
    forecaster = DataForecaster(frame, noise=ForecastNoise(temperature_std=1.0, seed=0))
    assert forecaster.realisation(index[0], 24, 3600).equals(frame.iloc[:24])
    forecast = forecaster.forecast(index[0], 24, 3600)
    assert not np.allclose(forecast["weather_inputs.TDryBul"], 280.0)
    np.testing.assert_allclose(forecast["weather_inputs.winDir"], 280.0)
    with pytest.raises(ValueError, match="does not cover"):
        forecaster.forecast(index[10], 24, 3600)
