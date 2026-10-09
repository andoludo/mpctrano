"""Generate the Modelica models and interfaces of the tutorial buildings and their variants.

Each typology has a base YAML in ``tutorials/buildings/`` and a table of variants: named
overrides applied to the loaded YAML (envelope constructions, heat pump size, PV...). The
script writes ``tutorials/buildings/<typology>/<variant>.yaml`` and, for every RC structure
needed by the notebooks (R3C2 for the controller, R4C3 for the plant), the Modelica package and
the interface JSON in ``tutorials/models/<typology>/``. Every generated model is translated with
rumoca, so a broken variant fails here rather than in a notebook.

    uv run python tutorials/scripts/generate_models.py             # all typologies
    uv run python tutorials/scripts/generate_models.py house       # one typology
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import rumoca
import typer
import yaml
from trano.mpc import RCModelType

from mpctrano import RCModel

TUTORIALS = Path(__file__).resolve().parents[1]
BUILDINGS = TUTORIALS / "buildings"
MODELS = TUTORIALS / "models"
STRUCTURES = (RCModelType.r3c2, RCModelType.r4c3)
Override = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class Variant:
    name: str
    description: str
    overrides: list[Override] = field(default_factory=list)


def set_constructions(walls: str | None = None, windows: str | None = None, floor: str | None = None) -> Override:
    def apply(data: dict[str, Any]) -> None:
        for space in data["spaces"]:
            boundaries = space["external_boundaries"]
            if walls:
                for wall in boundaries.get("external_walls", []):
                    wall["construction"] = walls
            if windows:
                for window in boundaries.get("windows", []):
                    window["construction"] = windows
            if floor:
                for element in boundaries.get("floor_on_grounds", []):
                    element["construction"] = floor

    return apply


def set_heat_pump(**parameters: float) -> Override:
    def apply(data: dict[str, Any]) -> None:
        for system in data["systems"]:
            if "boiler" in system:
                system["boiler"]["parameters"].update(parameters)

    return apply


def set_emitter(supply: float, return_: float) -> Override:
    def apply(data: dict[str, Any]) -> None:
        for space in data["spaces"]:
            for emission in space.get("emissions", []):
                radiator = emission.get("radiator")
                if radiator:
                    radiator["parameters"]["water_inlet_temperature_at_nominal_condition"] = supply
                    radiator["parameters"]["water_outlet_temperature_at_nominal_condition"] = return_

    return apply


def set_tank(**parameters: float) -> Override:
    def apply(data: dict[str, Any]) -> None:
        for system in data["systems"]:
            if "dhw_tank" in system:
                system["dhw_tank"]["parameters"].update(parameters)

    return apply


def add_pv(area: float, efficiency: float = 0.19, azimuth: float = 0.0, tilt: float = 35.0) -> Override:
    def apply(data: dict[str, Any]) -> None:
        parameters = {"area": area, "efficiency": efficiency, "azimuth": azimuth, "tilt": tilt}
        data.setdefault("solar", []).append({"photovoltaic": {"id": "PV:001", "parameters": parameters}})

    return apply


def add_storage(battery_kwh: float, ev_kwh: float) -> Override:
    def apply(data: dict[str, Any]) -> None:
        battery = {"capacity": battery_kwh, "max_charge_power": 3000, "max_discharge_power": 3000}
        charger = {"battery_capacity": ev_kwh, "max_charge_power": 7400, "energy_per_day": 12}
        data["systems"] += [
            {"battery": {"id": "BATTERY:001", "parameters": battery}},
            {"ev_charger": {"id": "EV:001", "parameters": charger}},
        ]

    return apply


TYPOLOGIES: dict[str, list[Variant]] = {
    "apartments": [
        Variant("base", "six apartments, radiators at 55/45 degC, 24 kW central heat pump, 1 m3 DHW tank"),
        Variant("floor_heating", "floor heating at 35/30 degC in every apartment", [set_emitter(308.15, 303.15)]),
        Variant("high_temperature", "old radiators at 70/55 degC", [set_emitter(343.15, 328.15)]),
        Variant("big_tank", "2 m3 DHW tank (more storage for the same demand)", [set_tank(volume=2.0)]),
        Variant(
            "small_heat_pump",
            "16 kW heat pump at 5.5 kW electrical (the compressor binds on cold days)",
            [set_heat_pump(nominal_heating_power=16000, max_electrical_power=5500)],
        ),
    ],
    "house": [
        Variant("base", "cavity walls with 10 cm of insulation, floor heating, 7 kW heat pump, DHW tank"),
        Variant(
            "old",
            "partially filled cavity walls and single glazing (pre-insulation envelope), 10 kW heat pump",
            [
                set_constructions(walls="CAVITYWALLPARTIALFILL:001", windows="EPCSINGLE:001"),
                set_heat_pump(nominal_heating_power=10000, max_electrical_power=3200),
            ],
        ),
        Variant(
            "radiators",
            "high temperature radiators (70/55 degC) instead of floor heating",
            [set_emitter(343.15, 328.15)],
        ),
        Variant(
            "big_heat_pump",
            "oversized 12 kW heat pump (more power to shift)",
            [set_heat_pump(nominal_heating_power=12000, max_electrical_power=3800)],
        ),
        Variant("pv", "base house with 30 m2 of PV", [add_pv(30.0)]),
        Variant(
            "pv_battery_ev",
            "base house with PV, an 8 kWh battery and an EV charger",
            [add_pv(30.0), add_storage(8.0, 50.0)],
        ),
    ],
}

app = typer.Typer(add_completion=False, help=__doc__)


def _load(path: Path) -> dict[str, Any]:
    """The YAML as a plain mapping (the `!include_default` tag is kept as a marker)."""

    class Loader(yaml.SafeLoader):
        pass

    Loader.add_constructor("!include_default", lambda loader, node: "!include_default")
    return yaml.load(path.read_text(), Loader=Loader)


def _dump(data: dict[str, Any], path: Path) -> None:
    text = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    path.write_text(text.replace("default: '!include_default'", "default: !include_default"))


def generate(typology: str) -> list[Path]:
    base = BUILDINGS / f"{typology}.yaml"
    if not base.exists():
        raise FileNotFoundError(base)
    variants_dir = BUILDINGS / typology
    output = MODELS / typology
    variants_dir.mkdir(exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    data = _load(base)
    weather = data["weather"]["parameters"]["path"]
    written = []
    for variant in TYPOLOGIES[typology]:
        variant_data = copy.deepcopy(data)
        for override in variant.overrides:
            override(variant_data)
        variant_data["weather"]["parameters"]["path"] = f"../{weather}"
        variant_path = variants_dir / f"{variant.name}.yaml"
        _dump(variant_data, variant_path)
        for structure in STRUCTURES:
            model = RCModel.from_yaml(variant_path, model_type=structure, name=f"{typology}_{variant.name}")
            stem = f"{variant.name}_{structure.value.lower()}"
            modelica, interface = model.write(output, stem, relative_weather=True)
            rumoca.Session().loads(modelica.read_text(), model=model.interface.model).to_casadi()
            written += [modelica, interface]
            typer.echo(
                f"{stem}: {model.n_states} states, {model.n_controls} controls, {model.n_disturbances} disturbances"
            )
    lines = [
        f"# {typology}: generated models",
        "",
        f"Generated by `tutorials/scripts/generate_models.py {typology}`; do not edit.",
        "",
        *(f"- `{variant.name}`: {variant.description}" for variant in TYPOLOGIES[typology]),
        "",
        "Each variant exists as `<variant>_r3c2` (control model) and `<variant>_r4c3` (plant).",
        "",
    ]
    (output / "README.md").write_text("\n".join(lines))
    return written


@app.command()
def main(typology: str | None = typer.Argument(None, help="Typology to generate (all when omitted)")) -> None:
    for name in [typology] if typology else list(TYPOLOGIES):
        generate(name)


if __name__ == "__main__":
    app()
