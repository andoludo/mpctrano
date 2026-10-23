"""Calibration parameters declared in the YAML: parsing, flattening, sensitivities and a closed-loop calibration."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from mpctrano import CalibrationSettings, CalibrationSpec, Calibrator, generate_closed_loop_data, spin_up
from mpctrano.model import CLOSED_LOOP_LIBRARY
from tests.conftest import MODELS

ANNOTATED = MODELS / "single_zone_calibration.yaml"


def _mappings(node):
    """Every mapping of a YAML tree."""
    if isinstance(node, dict):
        yield node
        for child in node.values():
            yield from _mappings(child)
    elif isinstance(node, list):
        for item in node:
            yield from _mappings(item)


def test_the_spec_collects_the_annotated_fields_and_writes_a_flat_yaml(tmp_path: Path) -> None:
    spec = CalibrationSpec.from_yaml(ANNOTATED)
    assert spec.paths == [
        "material[ROCKWOOL:CAL].thermal_conductivity",
        "constructions[CAVITYWALL:CAL].layers[1].thickness",
        "spaces[SPACE:001].parameters.ach",
    ]
    conductivity = spec["material[ROCKWOOL:CAL].thermal_conductivity"]
    assert (conductivity.value, conductivity.lower, conductivity.upper) == (0.04, 0.025, 0.08)
    flat = yaml.safe_load(spec.write(directory=tmp_path).read_text().replace("!include_default ''", "null"))
    assert flat["material"][0]["thermal_conductivity"] == 0.04
    assert flat["constructions"][0]["layers"][1]["thickness"] == 0.1
    assert not [node for node in _mappings(flat) if "parameter" in node]
    raised = spec.write({"material[ROCKWOOL:CAL].thermal_conductivity": 0.06}, directory=tmp_path)
    assert (
        yaml.safe_load(raised.read_text().replace("!include_default ''", "null"))["material"][0]["thermal_conductivity"]
        == 0.06
    )
    assert "!include_default" in raised.read_text()
    with pytest.raises(KeyError):
        spec.write({"nope": 1.0})
    with pytest.raises(KeyError):
        spec["nope"]


def test_the_multizone_script_yaml_declares_three_fields_and_finds_its_weather() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = CalibrationSpec.from_yaml(root / "scripts" / "multizone_heat_pump.yaml", base_dir=MODELS)
    assert spec.paths == [
        "material[MINERAL_WOOL:001].thermal_conductivity",
        "spaces[LIVINGROOM:001].parameters.ach",
        "spaces[BEDROOM_2:001].parameters.ach",
    ]
    assert spec.base_dir == MODELS.resolve()
    assert any(path.suffix == ".mos" for path in MODELS.iterdir())


def test_a_value_outside_its_bounds_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "material:\n  - id: M:1\n    thermal_conductivity: {value: 1.0, parameter: true, min: 2.0, max: 3.0}\n"
    )
    with pytest.raises(ValueError, match="outside"):
        CalibrationSpec.from_yaml(path)


@pytest.mark.slow
def test_the_sensitivities_are_measured_on_the_model() -> None:
    spec = CalibrationSpec.from_yaml(ANNOTATED)
    model = spec.model(CLOSED_LOOP_LIBRARY)
    assert model.control_names == []
    groups = spec.groups(model, CLOSED_LOOP_LIBRARY)
    by_name = {group.name: group for group in groups}
    # the closed-loop model splits the rockwool layer of each of the three walls into sublayers: the conductivity
    # moves the conductance of each of them, the thickness divides it and adds to their capacities
    conductivity = by_name["material[ROCKWOOL:CAL].thermal_conductivity"].exponents
    conductances = {name: exponent for name, exponent in conductivity.items() if re.search(r"\.G\[\d+\]$", name)}
    assert {name.split(".")[0] for name in conductances} == {f"externalwall_{i}" for i in range(3)}
    assert all(exponent == pytest.approx(1.0) for exponent in conductances.values())
    thickness = by_name["constructions[CAVITYWALL:CAL].layers[1].thickness"].exponents
    assert all(thickness[name] == pytest.approx(-1.0) for name in conductances)
    capacities = [exponent for name, exponent in thickness.items() if re.search(r"^externalwall_0\.C\[\d+\]$", name)]
    assert any(0 < exponent < 0.2 for exponent in capacities)  # the node shared with the next layer
    infiltration = by_name["spaces[SPACE:001].parameters.ach"].exponents
    assert infiltration == {"space_001.ACH": pytest.approx(1.0)}


@pytest.mark.slow
def test_the_closed_loop_model_is_calibrated_on_noisy_data_from_the_yaml_truth() -> None:
    """The true building is the YAML with a worse insulation, run by its own controllers; the infiltration is
    measured (the model is generated with it), the conductivity is calibrated by single shooting."""
    spec = CalibrationSpec.from_yaml(ANNOTATED)
    conductivity, infiltration = "material[ROCKWOOL:CAL].thermal_conductivity", "spaces[SPACE:001].parameters.ach"
    truth = spec.model(CLOSED_LOOP_LIBRARY, {conductivity: 0.06, infiltration: 0.6})
    model = spec.model(CLOSED_LOOP_LIBRARY, {infiltration: 0.6})
    assert model.control_names == []
    start = pd.Timestamp("2023-01-09")
    x0 = spin_up(truth, truth, start, days=2.0)
    recorded = generate_closed_loop_data(truth, start, days=3.0, measurement_std=0.1, meter_std=0.02, x0=x0)
    assert recorded.data.frame.columns.tolist()[: len(model.disturbance_names)] == model.disturbance_names
    assert "space_001.TAir" in recorded.data.frame and "boiler_001.EEle" in recorded.data.frame
    groups = spec.groups(model, CLOSED_LOOP_LIBRARY, values={infiltration: 0.6}, only=[conductivity])
    assert [group.name for group in groups] == [conductivity]
    assert "space_001.ACH" not in groups[0].exponents  # measured around the model's own values
    settings = CalibrationSettings(groups=groups, max_iterations=60)
    assert settings.method_for(model) == "shooting"
    calibrated, result = Calibrator(model, settings).fit(recorded.data)
    assert result.success, result.status
    assert result.method == "shooting" and result.variables == 1
    assert result.factors[conductivity] == pytest.approx(0.06, rel=0.1)
    assert result.training_after.rmse_mean < 0.6 * result.training_before.rmse_mean
    conductances = [name for name in groups[0].exponents if re.search(r"\.G\[\d+\]$", name)]
    assert conductances
    for name in conductances:  # every sublayer of the rockwool scaled by the calibrated conductivity
        assert calibrated.parameter(name) == pytest.approx(
            model.parameter(name) * result.factors[conductivity] / 0.04, rel=1e-6
        )
    assert not np.allclose(calibrated.parameters, model.parameters, equal_nan=True)
