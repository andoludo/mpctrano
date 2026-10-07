"""Notebook helpers: pre-generated models, the result cache, price options and the widgets (headless)."""

from pathlib import Path

import pandas as pd
import pytest
from trano.mpc import RCModelType

from mpctrano import FlatTariff, Mismatch
from mpctrano.notebook import (
    ResultCache,
    TutorialBuilding,
    closed_loop_explorer,
    comfort_spec,
    days,
    kpi_table,
    plan_explorer,
    price_options,
    scenario,
)
from mpctrano.scenarios import run_scenario

HOUSE = Path(__file__).parent.parent / "tutorials" / "models" / "house"
START = "2023-01-16"


@pytest.fixture(scope="module")
def building() -> TutorialBuilding:
    return TutorialBuilding(HOUSE, "base")


def test_tutorial_building_loads_both_structures(building: TutorialBuilding) -> None:
    assert "base" in TutorialBuilding.variants(HOUSE)
    assert building.control.interface.zones[0].model_type == "R3C2"
    assert building.plant_model.interface.zones[0].model_type == "R4C3"
    project = building.project()
    assert project.plant_model(RCModelType.r4c3) is building.plant_model
    plant = building.plant(Mismatch(parameter_std=0.1))
    assert plant.model.interface.zones[0].model_type == "R4C3"


def test_days_follow_quick_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MPCTRANO_QUICK", "1")
    assert days(7) == 1.0
    monkeypatch.delenv("MPCTRANO_QUICK")
    assert days(7) == 7


def test_price_options_cover_the_period() -> None:
    options = price_options(START, 2, sample=None)
    assert set(options) == {
        "flat",
        "time of use",
        "day-ahead",
        "day-ahead + evening spike",
        "day-ahead + negative midday",
    }
    index = pd.date_range(START, periods=48, freq="h")
    spike = options["day-ahead + evening spike"].energy_price(index)
    assert spike[17] > 2 * options["day-ahead"].energy_price(index)[17]
    assert (options["day-ahead + negative midday"].export_price(index)[11:15] < 0).all()


def test_comfort_spec_from_widget_values() -> None:
    spec = comfort_spec(21.0, 24.0, 3.0, widen_upper=1.0)
    assert spec.default.unoccupied_lower == 18.0
    assert spec.default.occupied_upper == 25.0


def test_cache_round_trip(tmp_path: Path, building: TutorialBuilding) -> None:
    cache = ResultCache(tmp_path / "cache")
    run = scenario("flat", FlatTariff(price=0.25), comfort_spec(21, 24, 3), START, 0.25)
    calls = []

    def compute():
        calls.append(1)
        return run_scenario(building.control, run, plant=building.plant(Mismatch(parameter_std=0.1)))

    first = cache.run(compute, variant="base", price="flat", band=3.0)
    second = cache.run(compute, variant="base", price="flat", band=3.0)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first.frame, second.frame)
    assert second.metrics.model_dump(exclude={"self_consumption"}) == first.metrics.model_dump(
        exclude={"self_consumption"}
    )
    assert second.heating_columns == first.heating_columns
    index = cache.index()
    assert list(index["variant"]) == ["base"] and index["total_cost"].iloc[0] == pytest.approx(first.metrics.total_cost)
    table = kpi_table({"first": first})
    assert "total_cost" in table.columns


def test_widgets_build_headless(tmp_path: Path, building: TutorialBuilding) -> None:
    prices = price_options(START, 1, sample=None)
    explorer = plan_explorer(building, prices, START)
    assert len(explorer.children) == 3
    cache = ResultCache(tmp_path / "cache")
    runs = {"flat": {"price": "flat"}}

    def compute(price: str):
        run = scenario(price, prices[price], comfort_spec(21, 24, 3), START, 0.25)
        return run_scenario(building.control, run, plant=building.plant(Mismatch(parameter_std=0.1)))

    widget = closed_loop_explorer(cache, runs, compute, {"price": ["flat"]})
    assert len(widget.children) == 2
    assert cache.has(cache.key(price="flat"))
