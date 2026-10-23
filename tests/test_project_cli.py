"""The project facade and the command line on the heat pump house."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from mpctrano import CalibrationSettings, FastModel, MonitoringData, Project, SyntheticData, scenario_catalogue
from mpctrano.cli import app
from tests.conftest import HEAT_PUMP_HOUSE, WEATHER

QUICK = scenario_catalogue(days=2 / 24, horizon=6)


def test_project_workflow(
    project: Project, heat_pump: FastModel, synthetic: SyntheticData, true_building: FastModel, tmp_path: Path
) -> None:
    fresh = Project.from_model(heat_pump, source=HEAT_PUMP_HOUSE)
    partial = MonitoringData(synthetic.data.frame[[*heat_pump.comfort_states, *heat_pump.control_names]])
    result = fresh.calibrate(partial, settings=CalibrationSettings(max_iterations=100))
    assert fresh.calibrated is not None and fresh.control_model is fresh.calibrated
    assert fresh.validation is not None and result.validation_after is not None
    assert set(result.factors) == {"envelope_conductance", "envelope_capacity", "air_capacity", "infiltration"}

    run = fresh.run(QUICK["flat"], plant=true_building)
    assert len(run.frame) == 2 and run.failures == 0
    reference = fresh.run_rule_based(QUICK["flat"])
    assert reference.name == "flat (rule based)"
    assert len(reference.frame) == 2
    assert fresh.closed_loop().control_names == []

    written = fresh.save(tmp_path, "house")
    assert written["calibration"].exists() and written["run_flat"].exists() and written["modelica"].exists()
    again = Project.from_yaml(HEAT_PUMP_HOUSE)
    again.load_calibrated(written["calibration"])
    assert again.calibrated is not None
    np.testing.assert_allclose(again.calibrated.parameters, fresh.calibrated.parameters, equal_nan=True)


def _resistance(model: FastModel, wall: str) -> float:
    """Resistance of a wall through its layers [K/W]: the same whether the layers are lumped or split into sublayers
    (the closed-loop FastModel of trano >= 0.59 discretises them as Buildings, the MPC model does not)."""
    return sum(1 / model.parameter(name) for name in model.select_parameters([f"{wall}.G[[]*[]]"]))


def test_true_closed_loop_has_the_groups_of_the_true_building(project: Project, true_building: FastModel) -> None:
    closed_loop = project.true_closed_loop(seed=0)
    for name in ("space_001.mSenFac", "space_001.ACH"):
        assert closed_loop.parameter(name) == true_building.parameter(name)
    assert _resistance(closed_loop, "externalwall_0") == pytest.approx(_resistance(true_building, "externalwall_0"))


def _copy_building(tmp_path: Path) -> Path:
    (tmp_path / "house.yaml").write_text(HEAT_PUMP_HOUSE.read_text())
    (tmp_path / WEATHER.name).write_bytes(WEATHER.read_bytes())
    return tmp_path / "house.yaml"


def test_cli_generate_synthetic_calibrate_run(tmp_path: Path) -> None:
    runner = CliRunner()
    building = _copy_building(tmp_path)

    result = runner.invoke(app, ["generate", str(building), "--closed-loop"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "house.mo").exists() and (tmp_path / "house.closed_loop.mo").exists()

    data_path = tmp_path / "monitoring.csv"
    result = runner.invoke(app, ["synthetic-data", str(building), "--output", str(data_path), "--days", "2"])
    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(data_path)) == 49

    result = runner.invoke(app, ["calibrate", str(building), str(data_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "house.calibration.json").exists()
    assert "training (calibrated)" in result.output

    output = tmp_path / "run.csv"
    result = runner.invoke(
        app,
        ["run", str(building), "--scenario", "time_of_use", "--days", str(2 / 24), "--horizon", "6",
         "--output", str(output), "--calibrated", str(tmp_path / "house.calibration.json"), "--seed", "1",
         "--rule-based"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(output)) == 2
    assert "time_of_use (rule based)" in result.output

    result = runner.invoke(app, ["run", str(building), "--scenario", "unknown"])
    assert result.exit_code != 0

    result = runner.invoke(app, ["scenarios"])
    assert result.exit_code == 0 and "demand_response" in result.output
