from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from mpctrano import CalibrationSettings, MonitoringData, Project, RCModel, SyntheticData, scenario_catalogue
from mpctrano.cli import app
from tests.conftest import THREE_ZONES, WEATHER


def test_project_workflow(model: RCModel, synthetic: SyntheticData, true_building: RCModel, tmp_path: Path) -> None:
    project = Project.from_model(model)
    partial = MonitoringData(synthetic.data.frame[[*model.comfort_states, *model.control_names]])
    result = project.calibrate(partial, settings=CalibrationSettings(parameters=["*_Ci", "*_Ria"], max_iterations=40))
    assert project.calibrated is not None and project.control_model is project.calibrated
    assert project.validation is not None and result.validation is not None
    run = project.run(scenario_catalogue(days=0.25)["flat"], plant=true_building)
    assert len(run.frame) == 6
    written = project.save(tmp_path, "house")
    assert written["calibrated_interface"].exists() and written["run_flat"].exists()
    fresh = Project.from_files(written["modelica"])
    fresh.load_calibrated(written["calibrated_interface"])
    assert fresh.calibrated is not None
    assert fresh.calibrated.parameters.tolist() == project.calibrated.parameters.tolist()


def _copy_model(tmp_path: Path) -> Path:
    (tmp_path / "house.yaml").write_text(THREE_ZONES.read_text())
    (tmp_path / WEATHER.name).write_bytes(WEATHER.read_bytes())
    return tmp_path / "house.yaml"


def test_cli_generate_synthetic_calibrate_run(tmp_path: Path) -> None:
    runner = CliRunner()
    model_path = _copy_model(tmp_path)

    result = runner.invoke(app, ["generate", str(model_path)])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "house.mo").exists() and (tmp_path / "house.mpc.json").exists()

    data_path = tmp_path / "monitoring.csv"
    result = runner.invoke(app, ["synthetic-data", str(model_path), "--output", str(data_path), "--days", "3"])
    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(data_path)) == 73

    result = runner.invoke(
        app, ["calibrate", str(tmp_path / "house.mo"), str(data_path), "--parameters", "*_Ci", "--parameters", "*_Ria"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "house.calibrated.mpc.json").exists()
    assert "training (calibrated)" in result.output

    output = tmp_path / "run.csv"
    result = runner.invoke(
        app,
        ["run", str(model_path), "--scenario", "time_of_use", "--days", "0.25", "--output", str(output),
         "--calibrated", str(tmp_path / "house.calibrated.mpc.json"), "--seed", "1"],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert len(pd.read_csv(output)) == 6

    result = runner.invoke(app, ["run", str(model_path), "--scenario", "unknown"])
    assert result.exit_code != 0

    result = runner.invoke(app, ["scenarios"])
    assert result.exit_code == 0 and "demand_response" in result.output
