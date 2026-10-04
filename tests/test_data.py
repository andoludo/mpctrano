import numpy as np
import pandas as pd
import pytest

from mpctrano import MonitoringData, RCModel, SyntheticData, WeatherForecaster
from mpctrano.data import DataError, time_step_of


def test_csv_round_trip_with_celsius(synthetic: SyntheticData, tmp_path) -> None:
    path = synthetic.data.to_csv(tmp_path / "data.csv", temperature_unit="C")
    raw = pd.read_csv(path)
    assert raw["space_001_Ti"].max() < 40  # written in Celsius
    loaded = MonitoringData.from_csv(path, temperature_unit="C")
    assert loaded.frame["space_001_Ti"].to_numpy() == pytest.approx(synthetic.data.frame["space_001_Ti"].to_numpy())
    assert loaded.time_step == 3600


def test_irregular_sampling_is_rejected() -> None:
    index = pd.DatetimeIndex(["2023-01-01 00:00", "2023-01-01 01:00", "2023-01-01 03:00"])
    with pytest.raises(DataError, match="regularly"):
        MonitoringData(pd.DataFrame({"space_001_Ti": [293.0, 293.0, 293.0]}, index=index))
    with pytest.raises(DataError):
        time_step_of(pd.Index([1, 2, 3]))


def test_validation_report(model: RCModel, synthetic: SyntheticData) -> None:
    report = synthetic.data.validate(model)
    assert report.complete and report.usable
    assert report.measured_states == model.comfort_states
    partial = MonitoringData(synthetic.data.frame[["space_001_Ti", *model.control_names, "TOut"]])
    report = partial.validate(model)
    assert not report.complete and report.usable
    assert report.missing_disturbances == [name for name in model.disturbance_names if name != "TOut"]
    with pytest.raises(DataError, match="Missing columns"):
        partial.signals(model)


def test_fill_disturbances_from_forecaster(
    model: RCModel, synthetic: SyntheticData, forecaster: WeatherForecaster
) -> None:
    partial = MonitoringData(synthetic.data.frame[["space_001_Ti", *model.control_names]])
    complete = partial.fill_disturbances(model, forecaster)
    assert complete.validate(model).complete
    signals = complete.signals(model)
    assert signals.measured_states == [0]
    assert signals.controls.shape == (3, len(complete) - 1)


def test_split_and_window(synthetic: SyntheticData) -> None:
    train, validation = synthetic.data.split(fraction=0.75)
    assert len(train) + len(validation) == len(synthetic.data) + 1  # one shared sample
    assert validation.start == train.end
    at = synthetic.data.start + pd.Timedelta(days=2)
    train, validation = synthetic.data.split(at=at)
    assert validation.start <= at <= validation.index[1]
    window = synthetic.data.window(at, at + pd.Timedelta(hours=5))
    assert len(window) == 6
    with pytest.raises(DataError):
        synthetic.data.split()


def test_resample_averages(synthetic: SyntheticData) -> None:
    coarse = synthetic.data.resample(7200)
    assert coarse.time_step == 7200
    assert coarse.frame["TOut"].mean() == pytest.approx(synthetic.data.frame["TOut"].mean(), abs=0.1)


def test_initial_state_uses_the_measured_temperatures(model: RCModel, synthetic: SyntheticData) -> None:
    x0 = synthetic.data.initial_state(model)
    first = synthetic.data.frame.iloc[0]
    assert x0[model.state_names.index("space_002_Te")] == first["space_002_Ti"]
    assert np.all(np.isfinite(x0))
