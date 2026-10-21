"""Notebook helpers: the tutorial building, the result cache, price options and the widgets (headless)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mpctrano import FlatTariff
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
from tests.conftest import HEAT_PUMP_HOUSE, START


@pytest.fixture(scope="module")
def building() -> TutorialBuilding:
    return TutorialBuilding(HEAT_PUMP_HOUSE, start=pd.Timestamp(START), spin_up_days=1.0)


def test_tutorial_building_has_its_models_and_spun_up_states(building: TutorialBuilding) -> None:
    assert building.closed_loop.control_names == []
    assert building.truth.state_names == building.model.state_names
    assert not np.allclose(building.truth.parameters, building.model.parameters, equal_nan=True)
    perfect, mismatch = building.initial_state(), building.initial_state(mismatch=True)
    assert not np.allclose(perfect, building.model.initial_state)
    assert not np.allclose(perfect, mismatch)
    plant = building.plant(mismatch=True)
    assert plant.model is building.truth and plant.measurement_std == building.measurement_std
    np.testing.assert_allclose(plant.state, mismatch)
    assert building.plant().measurement_std == 0.0


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
    assert options["day-ahead"].has_export_price


def test_comfort_spec_from_widget_values() -> None:
    spec = comfort_spec(21.0, 24.0, 3.0, widen_upper=1.0)
    assert spec.default.unoccupied_lower == 18.0
    assert spec.default.occupied_upper == 25.0


def test_cache_round_trip(tmp_path: Path, building: TutorialBuilding) -> None:
    cache = ResultCache(tmp_path / "cache")
    run = scenario("flat", FlatTariff(price=0.25), comfort_spec(21, 24, 3), START, 0.25, horizon=6)
    calls = []

    def compute():
        calls.append(1)
        return run_scenario(building.model, run, plant=building.plant(mismatch=True))

    first = cache.run(compute, variant="base", price="flat", band=3.0)
    second = cache.run(compute, variant="base", price="flat", band=3.0)
    assert len(calls) == 1
    pd.testing.assert_frame_equal(first.frame, second.frame, check_freq=False)
    assert second.metrics.model_dump(exclude={"self_consumption"}) == first.metrics.model_dump(
        exclude={"self_consumption"}
    )
    assert second.comfort_states == first.comfort_states
    index = cache.index()
    assert list(index["variant"]) == ["base"]
    assert index["total_cost"].iloc[0] == pytest.approx(first.metrics.total_cost)
    assert "total_cost" in kpi_table({"first": first}).columns


def test_widgets_build_headless(tmp_path: Path, building: TutorialBuilding) -> None:
    prices = price_options(START, 1, sample=None)
    explorer = plan_explorer(building, prices)
    assert len(explorer.children) == 3
    cache = ResultCache(tmp_path / "cache")

    def compute(price: str):
        run = scenario(price, prices[price], comfort_spec(21, 24, 3), START, 0.25, horizon=6)
        return run_scenario(building.model, run, plant=building.plant(mismatch=True))

    widget = closed_loop_explorer(cache, {"flat": {"price": "flat"}}, compute, {"price": ["flat"]})
    assert len(widget.children) == 2
    assert cache.has(cache.key(price="flat"))
