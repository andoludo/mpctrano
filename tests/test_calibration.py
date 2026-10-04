import numpy as np
import pytest

from mpctrano import CalibrationSettings, Calibrator, MonitoringData, RCModel, SyntheticData, evaluate, simulate_fit


@pytest.fixture(scope="module")
def calibration(model: RCModel, synthetic: SyntheticData):
    train, validation = synthetic.data.split(fraction=0.75)
    calibrated, result = Calibrator(model).fit(train, validation)
    return calibrated, result


def test_calibration_reaches_the_noise_level(calibration, true_building: RCModel) -> None:
    _, result = calibration
    assert result.success
    assert result.training.rmse_mean < 0.1  # measurement noise is 0.05 K
    assert result.validation is not None and result.validation.rmse_mean < 0.15
    assert result.training.rmse_mean < 0.3 * result.initial_training.rmse_mean
    assert result.validation.rmse_mean < 0.5 * result.initial_validation.rmse_mean
    assert set(result.calibrated) == set(true_building.parameter_names) - {"TGro"}
    assert result.metrics().shape == (4, 2)


def test_identifiable_parameters_are_recovered(calibration, true_building: RCModel) -> None:
    calibrated, _ = calibration
    # The fast air-node parameters are well identified from the zone temperatures.
    for zone in ("space_001", "space_002", "space_003"):
        for name in ("Ci", "Ria"):
            parameter = f"{zone}_{name}"
            assert calibrated.parameter(parameter) == pytest.approx(true_building.parameter(parameter), rel=0.25)


def test_summary_flags_bounds(calibration) -> None:
    _, result = calibration
    summary = result.summary(0.2, 5.0)
    assert set(summary.columns) == {"initial", "calibrated", "ratio", "at_bound"}
    assert summary["ratio"].between(0.2, 5.0).all()
    assert len(result.ratios) == len(result.calibrated)


def test_subset_of_parameters(model: RCModel, synthetic: SyntheticData) -> None:
    settings = CalibrationSettings(parameters=["*_Ci", "*_Ria"], max_iterations=50)
    calibrator = Calibrator(model, settings)
    assert calibrator.calibrated_names() == model.select_parameters(["*_Ci", "*_Ria"])
    calibrated, result = calibrator.fit(
        synthetic.data.window(synthetic.data.start, synthetic.data.start + np.timedelta64(2, "D"))
    )
    untouched = [name for name in model.parameter_names if name not in result.calibrated]
    assert [calibrated.parameter(name) for name in untouched] == pytest.approx(
        [model.parameter(name) for name in untouched]
    )
    assert result.training.rmse_mean <= result.initial_training.rmse_mean


def test_bounds_are_respected(model: RCModel, synthetic: SyntheticData) -> None:
    settings = CalibrationSettings(lower_factor=0.9, upper_factor=1.1, parameters=["*_Ci"], max_iterations=30)
    calibrated, result = Calibrator(model, settings).fit(synthetic.data)
    for name in result.calibrated:
        assert 0.9 - 1e-6 <= calibrated.parameter(name) / model.parameter(name) <= 1.1 + 1e-6


def test_unknown_pattern_raises(model: RCModel) -> None:
    with pytest.raises(ValueError, match="No parameter"):
        Calibrator(model, CalibrationSettings(parameters=["nothing_*"])).calibrated_names()


def test_evaluate_with_initial_state_fit_is_better(true_building: RCModel, synthetic: SyntheticData) -> None:
    data = MonitoringData(synthetic.data.frame.iloc[:48])
    fitted = evaluate(true_building, data, fit_initial_state=True)
    raw = evaluate(true_building, data, fit_initial_state=False)
    assert fitted.rmse_mean <= raw.rmse_mean + 1e-9
    assert fitted.rmse_mean < 0.1


def test_simulate_fit_frame(calibration, synthetic: SyntheticData) -> None:
    calibrated, _ = calibration
    frame = simulate_fit(calibrated, synthetic.data)
    assert {"space_001_Ti_measured", "space_001_Ti_simulated"} <= set(frame.columns)
    assert len(frame) == len(synthetic.data)


@pytest.mark.parametrize("hessian", ["exact", "limited-memory"])
def test_hessian_options(model: RCModel, synthetic: SyntheticData, hessian: str) -> None:
    short = MonitoringData(synthetic.data.frame.iloc[:49])
    settings = CalibrationSettings(hessian=hessian, parameters=["*_Ci"], max_iterations=100)
    _, result = Calibrator(model, settings).fit(short)
    assert result.training.rmse_mean <= result.initial_training.rmse_mean + 1e-6
