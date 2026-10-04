import numpy as np
import pandas as pd
import pytest
from trano.mpc import EstimationSettings, MPCModelInterface, RCModelType

from mpctrano import RCModel
from mpctrano.model import ModelError
from tests.conftest import THREE_ZONES


def test_model_from_yaml_matches_the_trano_interface(model: RCModel) -> None:
    assert model.name == "three_zones.building_mpc"
    assert model.n_zones == 3
    assert model.n_states == 6
    assert model.control_names == ["space_001_QHea", "space_002_QHea", "space_003_QHea"]
    assert model.disturbance_names[0] == "TOut"
    assert model.comfort_states == ["space_001_Ti", "space_002_Ti", "space_003_Ti"]
    assert model.n_parameters == 32
    assert model.weather_path is not None and model.weather_path.exists()
    assert np.all(model.max_heating_power > 0)


@pytest.mark.parametrize("model_type", list(RCModelType))
def test_every_rc_structure_loads(models: dict[RCModelType, RCModel], model_type: RCModelType) -> None:
    model = models[model_type]
    states_per_zone = {RCModelType.r1c1: 1, RCModelType.r4c3: 3}.get(model_type, 2)
    assert model.n_states == states_per_zone * 3
    xdot = model.continuous(model.initial_state, np.zeros(3), np.zeros(model.n_disturbances), model.parameters)
    assert np.all(np.isfinite(np.array(xdot)))


def test_from_files_round_trip(model: RCModel, tmp_path) -> None:
    modelica, interface = model.write(tmp_path, "house")
    loaded = RCModel.from_files(modelica)
    assert loaded.state_names == model.state_names
    assert loaded.parameters == pytest.approx(model.parameters)
    assert loaded.base_dir == tmp_path
    assert MPCModelInterface.read(interface).model == model.interface.model


def test_interface_mismatch_is_detected(model: RCModel) -> None:
    broken = model.interface.model_copy(update={"states": model.interface.states[:-1]})
    with pytest.raises(ModelError, match="states"):
        RCModel.from_source(model.source, broken)


def test_estimation_settings_change_the_initial_parameters() -> None:
    tuned = RCModel.from_yaml(THREE_ZONES, settings=EstimationSettings(air_capacity_multiplier=1.0))
    default = RCModel.from_yaml(THREE_ZONES)
    assert tuned.parameter("space_001_Ci") == pytest.approx(default.parameter("space_001_Ci") / 5)
    assert tuned.weather_path == default.weather_path


def test_discrete_step_converges_to_the_ode(model: RCModel) -> None:
    x = model.initial_state + np.arange(model.n_states)
    u = np.array([1000.0, 500.0, 200.0])
    d = np.array([275.0, 100.0, 50.0, 20.0, 400.0, 300.0, 200.0])
    coarse = np.array(model.discrete(60.0, substeps=1)(x, u, d, model.parameters)).reshape(-1)
    fine = np.array(model.discrete(60.0, substeps=20)(x, u, d, model.parameters)).reshape(-1)
    euler = x + 60.0 * np.array(model.continuous(x, u, d, model.parameters)).reshape(-1)
    assert coarse == pytest.approx(fine, abs=1e-6)
    assert np.abs(euler - fine).max() < 1e-2


def test_affine_form_is_exact(model: RCModel) -> None:
    affine = model.affine(3600.0)
    dynamics = model.discrete(3600.0)
    rng = np.random.default_rng(0)
    for _ in range(3):
        x = 290 + 5 * rng.standard_normal(model.n_states)
        u = 2000 * rng.random(model.n_controls)
        d = np.array([275.0, 100.0, 50.0, 20.0, 400.0, 300.0, 200.0]) * rng.random(model.n_disturbances)
        expected = np.array(dynamics(x, u, d, model.parameters)).reshape(-1)
        assert affine(x, u, d) == pytest.approx(expected, abs=1e-8)
    assert np.all(np.abs(np.linalg.eigvals(affine.A)) < 1)  # stable


def test_stable_substeps_follow_the_time_step(model: RCModel) -> None:
    assert model.stable_substeps(60.0) == 1
    assert model.stable_substeps(3600.0) >= model.stable_substeps(900.0)


def test_with_parameters_and_calibrated_interface(model: RCModel) -> None:
    changed = model.with_parameters({"space_001_Ci": 2e6})
    assert changed.parameter("space_001_Ci") == 2e6
    assert model.parameter("space_001_Ci") != 2e6
    interface = changed.calibrated_interface()
    assert interface.parameter_values[model.parameter_names.index("space_001_Ci")] == 2e6
    with pytest.raises(KeyError):
        model.with_parameters({"unknown": 1.0})
    with pytest.raises(ModelError):
        model.with_parameters(np.ones(3))


def test_select_parameters_with_patterns(model: RCModel) -> None:
    assert model.select_parameters(["H_*"]) == ["H_space_001_space_002", "H_space_002_space_003"]
    assert len(model.select_parameters(["*_Ci"])) == 3


def test_simulate_and_simulate_frame_agree(model: RCModel) -> None:
    steps = 5
    index = pd.date_range("2023-01-01", periods=steps + 1, freq="h")
    inputs = pd.DataFrame(0.0, index=index, columns=model.control_names + model.disturbance_names)
    inputs["TOut"] = 275.0
    inputs["space_001_QHea"] = 1000.0
    frame = model.simulate_frame(model.initial_state, inputs)
    arrays = model.simulate(
        model.initial_state,
        inputs[model.control_names].to_numpy().T[:, :-1],
        inputs[model.disturbance_names].to_numpy().T[:, :-1],
        3600.0,
    )
    assert frame.shape == (steps + 1, model.n_states)
    assert pytest.approx(arrays) == frame.to_numpy().T
    assert frame["space_001_Ti"].iloc[-1] > frame["space_003_Ti"].iloc[-1]  # the heated zone is warmer
