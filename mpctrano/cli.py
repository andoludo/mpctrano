"""Command line interface: ``mpctrano --help``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from trano.mpc import RCModelType

from mpctrano.calibration import CalibrationSettings
from mpctrano.data import MonitoringData
from mpctrano.project import Project
from mpctrano.scenarios import scenario_catalogue

app = typer.Typer(
    help="Calibration, state estimation and economic MPC of Trano RC building models.", no_args_is_help=True
)
CHECK = "✔"


def _project(model: Path, model_type: RCModelType, calibrated: Path | None) -> Project:
    project = Project.from_files(model) if model.suffix == ".mo" else Project.from_yaml(model, model_type=model_type)
    if calibrated is not None:
        project.load_calibrated(calibrated)
    return project


ModelArgument = Annotated[
    Path, typer.Argument(help="Trano building (.yaml/.json) or Modelica package (.mo) with its .mpc.json")
]
ModelTypeOption = Annotated[RCModelType, typer.Option(help="RC zone model when generating from a YAML file")]


@app.command()
def generate(
    model: ModelArgument,
    model_type: ModelTypeOption = RCModelType.r3c2,
    output: Annotated[Path | None, typer.Option(help="Output directory (default: next to the model)")] = None,
) -> None:
    """Generate the Modelica package and the MPC interface with Trano."""
    project = _project(model, model_type, None)
    modelica, interface = project.model.write(output or model.parent)
    size = f"{project.model.n_states} states, {project.model.n_parameters} parameters"
    typer.echo(f"{CHECK} {modelica} and {interface} written ({size})")


@app.command("synthetic-data")
def synthetic_data(
    model: ModelArgument,
    output: Annotated[Path, typer.Option(help="CSV file to write")] = Path("monitoring.csv"),
    days: Annotated[float, typer.Option(help="Length of the data")] = 14,
    start: Annotated[str, typer.Option(help="First timestamp")] = "2023-01-02",
    model_type: ModelTypeOption = RCModelType.r3c2,
    seed: Annotated[int, typer.Option(help="Seed of the perturbed 'true' building")] = 0,
) -> None:
    """Generate monitoring data from a perturbed copy of the model (to try the package without a building)."""
    project = _project(model, model_type, None)
    synthetic = project.synthetic_data(days=days, start=start, building=project.true_building(seed=seed))
    synthetic.data.to_csv(output)
    typer.echo(f"{CHECK} {len(synthetic.data)} samples written to {output}")


@app.command()
def calibrate(
    model: ModelArgument,
    data: Annotated[Path, typer.Argument(help="Monitoring data CSV (time, <zone>_Ti, <zone>_QHea, TOut...)")],
    output: Annotated[Path | None, typer.Option(help="Directory of the calibrated interface")] = None,
    model_type: ModelTypeOption = RCModelType.r3c2,
    validation_fraction: Annotated[float, typer.Option(min=0, max=0.9)] = 0.25,
    temperature_unit: Annotated[str, typer.Option(help="Unit of the temperatures in the CSV: K or C")] = "K",
    regularisation: Annotated[float, typer.Option(min=0)] = 1e-2,
    parameters: Annotated[list[str] | None, typer.Option(help="Patterns of the parameters to calibrate")] = None,
) -> None:
    """Calibrate the RC parameters on monitoring data and write <model>.calibrated.mpc.json."""
    project = _project(model, model_type, None)
    monitoring = MonitoringData.from_csv(data, temperature_unit=temperature_unit)
    settings = CalibrationSettings(regularisation=regularisation, parameters=parameters or None)
    result = project.calibrate(monitoring, settings=settings, validation_fraction=validation_fraction or None)
    written = project.save(output or model.parent)
    typer.echo(result.metrics().round(3).to_string())
    typer.echo(f"{CHECK} calibrated interface written to {written['calibrated_interface']}")


@app.command()
def run(
    model: ModelArgument,
    scenario: Annotated[str, typer.Option(help="Scenario name, see `mpctrano scenarios`")] = "time_of_use",
    calibrated: Annotated[Path | None, typer.Option(help="Calibrated .mpc.json to control with")] = None,
    output: Annotated[Path | None, typer.Option(help="CSV file of the closed-loop trajectories")] = None,
    days: Annotated[float, typer.Option()] = 2,
    start: Annotated[str, typer.Option()] = "2023-01-16",
    model_type: ModelTypeOption = RCModelType.r3c2,
    seed: Annotated[int | None, typer.Option(help="Seed of the perturbed plant (default: the model itself)")] = None,
) -> None:
    """Closed-loop MPC run of a scenario (state estimation + MPC against a simulated building)."""
    catalogue = scenario_catalogue(start=start, days=days)
    if scenario not in catalogue:
        raise typer.BadParameter(f"Unknown scenario {scenario!r}, choose among {sorted(catalogue)}.")
    project = _project(model, model_type, calibrated)
    plant = project.true_building(seed=seed) if seed is not None else project.control_model
    result = project.run(catalogue[scenario], plant=plant)
    typer.echo(result.summary().round(3).to_string())
    if output is not None:
        result.frame.to_csv(output)
        typer.echo(f"{CHECK} trajectories written to {output}")


@app.command()
def scenarios() -> None:
    """List the scenarios shipped with the package."""
    for name, scenario in scenario_catalogue().items():
        typer.echo(f"{name:20s} {scenario.description}")


if __name__ == "__main__":
    app()
