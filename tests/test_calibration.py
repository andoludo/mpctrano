"""Calibration of Trano's single-zone hydronic house as a FastModel, on data of a perturbed copy."""

import casadi as ca
import numpy as np
import pandas as pd
import pytest

from mpctrano import FastModel
from mpctrano.synthetic import generate_monitoring_data
from mpctrano.calibration import (
    DEFAULT_GROUPS,
    OPTIONAL_GROUPS,
    CalibrationSettings,
    Calibrator,
    ParameterGroup,
    YamlGroup,
    evaluate,
)
from mpctrano.units import KELVIN
from tests.conftest import BOILER_HOUSE

HOUSE = BOILER_HOUSE
TRUTH = {
    "envelope_conductance": 1.3,
    "envelope_capacity": 0.8,
    "air_capacity": 3.0,
    "infiltration": 0.5,
    "boiler_efficiency": 0.9,
}


@pytest.fixture(scope="module")
def house() -> FastModel:
    return FastModel.from_yaml(HOUSE)


@pytest.fixture(scope="module")
def recorded(house: FastModel) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Six days of a building whose groups differ from Trano's, operated by a thermostat."""
    synthetic = generate_monitoring_data(Calibrator(house).calibrated(TRUTH), "2023-01-02", 6, seed=0)
    return synthetic.data.frame, synthetic.states


def test_groups_hold_the_parameters_the_equations_use(house: FastModel) -> None:
    calibrator = Calibrator(house, CalibrationSettings(groups=DEFAULT_GROUPS + OPTIONAL_GROUPS))
    members = {name: [house.parameter_names[i] for i in indices] for name, indices in calibrator.members.items()}
    assert "externalwall_0.G[2]" in members["envelope_conductance"]
    assert "window_0.GGap[1]" in members["envelope_conductance"]
    assert "flooronground_0.C[2]" in members["envelope_capacity"]
    assert members["air_capacity"] == ["space_001.mSenFac"]
    assert members["infiltration"] == ["space_001.ACH"]
    assert members["boiler_efficiency"] == ["boiler_001.eta_nominal"]
    assert members["emitter_capacity"] == ["radiator_003.C"]
    every = {name for names in members.values() for name in names}
    assert not any("constructionType.k" in name for name in every)  # layer data: not in the equations
    assert "boiler_001.a[1]" not in every
    with pytest.raises(ValueError, match="No parameter"):
        Calibrator(house, CalibrationSettings(groups=(ParameterGroup(name="none", patterns=("^nothing$",)),)))


def test_groups_scale_or_set_their_parameters(house: FastModel) -> None:
    calibrator = Calibrator(house)
    model = calibrator.calibrated({**dict.fromkeys(TRUTH, 1.0), "envelope_conductance": 2.0, "infiltration": 0.7})
    assert model.parameter("externalwall_0.G[2]") == pytest.approx(2 * house.parameter("externalwall_0.G[2]"))
    assert model.parameter("space_001.ACH") == 0.7
    assert model.parameter("space_001.mSenFac") == house.parameter("space_001.mSenFac")


def test_the_factors_of_groups_sharing_a_parameter_compound(house: FastModel) -> None:
    """A conductivity and a thickness both move a layer conductance: doubling both leaves it unchanged."""
    conductivity = YamlGroup(name="k", value=0.04, lower=0.01, upper=0.1, exponents={"externalwall_0.G[2]": 1.0})
    thickness = YamlGroup(
        name="x", value=0.1, lower=0.02, upper=0.3, exponents={"externalwall_0.G[2]": -1.0, "externalwall_0.C[2]": 1.0}
    )
    calibrator = Calibrator(house, CalibrationSettings(groups=(conductivity, thickness)))
    model = calibrator.calibrated({"k": 0.08, "x": 0.2})
    assert model.parameter("externalwall_0.G[2]") == pytest.approx(house.parameter("externalwall_0.G[2]"))
    assert model.parameter("externalwall_0.C[2]") == pytest.approx(2 * house.parameter("externalwall_0.C[2]"))
    theta = ca.SX.sym("theta", 2)
    symbolic, values = calibrator._symbolic_parameters(theta)
    at = ca.Function("p", [theta], [symbolic, values["k"], values["x"]])
    vector, k, x = at(np.log([2.0, 2.0]))
    assert np.allclose(np.asarray(vector).ravel(), np.nan_to_num(model.parameters))  # NaN parameters are zeroed
    assert (float(k), float(x)) == pytest.approx((0.08, 0.2))


def test_synthetic_data_is_plausible(house: FastModel, recorded: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    data, truth = recorded
    assert len(data) == 6 * 24 + 1
    assert {*house.control_names, *house.disturbance_names, *house.comfort_states, *house.meters} <= set(data)
    assert truth["boiler_001.T"].max() < KELVIN + 100  # the thermostat only fires the boiler while water flows
    assert data["space_001.TAir"].std() > 1.0  # the set point excites the dynamics
    assert np.all(np.diff(data["boiler_001.EFue"]) >= -1e-9)


def test_calibration_recovers_the_building(house: FastModel, recorded: tuple[pd.DataFrame, pd.DataFrame]) -> None:
    data, _ = recorded
    training, validation = data.iloc[: 4 * 24 + 1], data.iloc[4 * 24 :]
    calibrated, result = Calibrator(house).fit(training, validation)
    assert result.success, result.status
    assert result.factors["envelope_conductance"] == pytest.approx(1.3, rel=0.1)
    assert result.factors["envelope_capacity"] == pytest.approx(0.8, rel=0.1)
    assert result.factors["air_capacity"] == pytest.approx(3.0, rel=0.15)
    assert result.factors["infiltration"] == pytest.approx(0.5, abs=0.1)
    assert result.factors["boiler_efficiency"] == pytest.approx(0.9, rel=0.03)  # from the fuel meter
    assert result.training_before.rmse_mean > 2.0
    assert result.training_after.rmse_mean < 0.1
    assert result.validation_after is not None
    assert result.validation_after.rmse_mean < 0.1
    assert calibrated.parameter("space_001.ACH") == pytest.approx(result.factors["infiltration"])
    assert result.metrics().shape == (4, 2)

    fit, states = evaluate(calibrated, training, np.array(result.initial_state))
    assert fit.rmse_mean == pytest.approx(result.training_after.rmse_mean)
    assert states.shape == (house.n_states, len(training))
