"""Monitoring data named after the variables of the FastModel."""

import numpy as np
import pandas as pd
import pytest

from mpctrano import FastModel, MonitoringData, SyntheticData, WeatherForecaster
from mpctrano.data import DataError
from mpctrano.units import KELVIN

AIR = "space_001.TAir"


def test_synthetic_data_has_the_columns_of_the_model(heat_pump: FastModel, synthetic: SyntheticData) -> None:
    columns = set(synthetic.data.columns)
    assert {AIR, *heat_pump.control_names, *heat_pump.meters, *heat_pump.disturbance_names} <= columns
    assert len(synthetic.data) == 3 * 24 + 1
    assert synthetic.states["boiler_001.T"].max() < KELVIN + 100  # the thermostat runs the heat pump with the flow
    assert np.all(np.diff(synthetic.data.frame["boiler_001.EEle"]) >= -1e-9)
    assert synthetic.setpoints.between(KELVIN + 17, KELVIN + 23).all()


def test_csv_round_trip_with_celsius(synthetic: SyntheticData, tmp_path) -> None:
    path = synthetic.data.to_csv(tmp_path / "data.csv", temperature_unit="C")
    raw = pd.read_csv(path)
    assert raw[AIR].mean() < 40  # written in degrees Celsius
    assert raw["boiler_001.EEle"].iloc[-1] == pytest.approx(synthetic.data.frame["boiler_001.EEle"].iloc[-1])
    read = MonitoringData.from_csv(path, temperature_unit="C")
    pd.testing.assert_frame_equal(read.frame, synthetic.data.frame, check_freq=False, check_names=False, atol=1e-9)
    with pytest.raises(DataError):
        MonitoringData.from_csv(path, temperature_unit="F")


def test_irregular_sampling_is_rejected() -> None:
    index = pd.DatetimeIndex(["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 03:00"])
    with pytest.raises(DataError, match="regularly"):
        MonitoringData(pd.DataFrame({AIR: [290.0, 291.0, 292.0]}, index=index))
    with pytest.raises(DataError, match="DatetimeIndex"):
        MonitoringData(pd.DataFrame({AIR: [290.0, 291.0]}))


def test_validation_report(heat_pump: FastModel, synthetic: SyntheticData) -> None:
    report = synthetic.data.validate(heat_pump)
    assert report.complete and report.usable
    assert report.measured_zones == [AIR]
    assert report.meters == ["boiler_001.EEle"]
    partial = MonitoringData(synthetic.data.frame[[AIR, "valve_001_y.y"]])
    report = partial.validate(heat_pump)
    assert not report.usable
    assert report.missing_controls == ["boiler_001_y.y", "three_way_valve_001_y.y"]
    assert report.missing_disturbances == heat_pump.disturbance_names


def test_fill_disturbances_from_the_forecaster(
    heat_pump: FastModel, synthetic: SyntheticData, forecaster: WeatherForecaster
) -> None:
    partial = MonitoringData(synthetic.data.frame[[AIR, *heat_pump.control_names]])
    complete = partial.fill_disturbances(heat_pump, forecaster)
    assert complete.validate(heat_pump).complete
    np.testing.assert_allclose(
        complete.frame[heat_pump.disturbance_names].to_numpy(),
        synthetic.data.frame[heat_pump.disturbance_names].to_numpy(),
    )
    assert complete.fill_disturbances(heat_pump, forecaster) is complete


def test_split_window_and_resample(synthetic: SyntheticData) -> None:
    training, validation = synthetic.data.split(fraction=0.75)
    assert training.end == validation.start  # the validation starts where the training ends
    assert len(training) + len(validation) == len(synthetic.data) + 1
    training, validation = synthetic.data.split(at="2023-01-04")
    assert validation.start == pd.Timestamp("2023-01-03 23:00")
    with pytest.raises(DataError):
        synthetic.data.split(fraction=1.5)
    window = synthetic.data.window("2023-01-02 06:00", "2023-01-02 12:00")
    assert len(window) == 7
    coarse = synthetic.data.resample(3 * 3600)
    assert coarse.time_step == 3 * 3600
    assert coarse.frame[AIR].iloc[0] == pytest.approx(synthetic.data.frame[AIR].iloc[:3].mean())
