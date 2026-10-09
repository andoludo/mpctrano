"""The pre-generated tutorial models stay loadable, translatable and consistent with their interfaces."""

from pathlib import Path

import pytest

from mpctrano import RCModel

MODELS = sorted((Path(__file__).parent.parent / "tutorials" / "models").glob("*/*.mpc.json"))


@pytest.mark.parametrize("interface", MODELS, ids=[f"{path.parent.name}/{path.stem}" for path in MODELS])
def test_tutorial_model_loads(interface: Path) -> None:
    model = RCModel.from_files(interface.with_suffix("").with_suffix(".mo"), interface)
    assert model.interface.version == "2"
    assert model.n_states == len(model.initial_state)
    lower, upper = model.control_bounds()
    assert len(lower) == model.n_controls and (upper > 0).all()
    assert model.weather_path is not None and model.weather_path.exists()
    heat_pumps = [system for system in model.systems if system.kind == "heat_pump"]
    assert len(heat_pumps) == 1
    assert set(heat_pumps[0].zones) == set(model.zone_names)


def test_house_variants_cover_the_notebook_grid() -> None:
    names = {path.name.removesuffix(".mpc.json") for path in MODELS if path.parent.name == "house"}
    for variant in ("base", "old", "radiators", "big_heat_pump", "pv", "pv_battery_ev"):
        assert {f"{variant}_r3c2", f"{variant}_r4c3"} <= names


def test_apartment_variants_cover_the_notebook_grid() -> None:
    names = {path.name.removesuffix(".mpc.json") for path in MODELS if path.parent.name == "apartments"}
    for variant in ("base", "floor_heating", "high_temperature", "big_tank", "small_heat_pump"):
        assert {f"{variant}_r3c2", f"{variant}_r4c3"} <= names
    block = RCModel.from_files(
        MODELS[0].parent.parent / "apartments" / "base_r3c2.mo",
        MODELS[0].parent.parent / "apartments" / "base_r3c2.mpc.json",
    )
    assert len(block.zone_names) == 6
    assert [band.system for band in block.storage_bands] == ["tank_001"]
    assert len(block.control_groups) == 1 and len(block.control_groups[0].indices) == 7
