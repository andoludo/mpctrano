"""Scenarios, closed-loop runs and flexibility indicators on the heat pump house."""

import numpy as np
import pandas as pd
import pytest

from mpctrano import (
    ClosedLoopResult,
    ControlMetrics,
    FlatTariff,
    Prices,
    BuildingModel,
    MovingHorizonEstimator,
    PerfectEstimator,
    Project,
    compare,
    price_response,
    run_scenario,
    scenario_catalogue,
    value_of_flexibility,
)
from mpctrano.scenarios import build_estimator
from mpctrano.simulation import ModelPlant

QUICK = scenario_catalogue(days=2 / 24, horizon=6)


def test_catalogue_contents() -> None:
    catalogue = scenario_catalogue(days=0.5)
    assert set(catalogue) == {
        "flat",
        "time_of_use",
        "dynamic_price",
        "demand_response",
        "peak_shaving",
        "wide_comfort_band",
        "imperfect_forecast",
    }
    scenario = catalogue["demand_response"]
    assert scenario.steps == 12
    assert scenario.with_days(2).steps == 48
    assert scenario.tariff.kind == "demand_response"
    assert all(scenario.estimator == "mhe" for scenario in catalogue.values())


def test_estimator_choice(heat_pump: BuildingModel) -> None:
    plant = ModelPlant(heat_pump, 3600.0, "2023-01-16")
    assert isinstance(build_estimator(QUICK["flat"], heat_pump, plant), MovingHorizonEstimator)
    perfect = QUICK["flat"].model_copy(update={"estimator": "perfect"})
    assert isinstance(build_estimator(perfect, heat_pump, plant), PerfectEstimator)


def test_a_scenario_runs_against_the_true_building(heat_pump: BuildingModel, true_building: BuildingModel) -> None:
    result = run_scenario(heat_pump, QUICK["time_of_use"], plant=true_building)
    assert isinstance(result, ClosedLoopResult)
    assert result.name == "time_of_use"
    assert result.failures == 0
    assert len(result.frame) == 2
    assert result.frame.index[0] == pd.Timestamp("2023-01-16")
    assert result.metrics.electricity_kwh == pytest.approx(result.frame["bought"].sum())
    assert result.metrics.fuel_kwh == 0.0
    assert {"space_001.TAir_estimate", "space_001.TAir_true", "price"} <= set(result.frame.columns)
    again = result.metrics_under(result_tariff := QUICK["flat"].tariff)
    assert again.energy_kwh == pytest.approx(result.metrics.energy_kwh)
    assert again.energy_cost == pytest.approx(result_tariff.price * result.metrics.electricity_kwh)  # type: ignore[attr-defined]
    response = price_response(result)
    assert response["share"].sum() == pytest.approx(1.0) or response["sum"].sum() == 0


def _result(name: str, bought: float) -> ClosedLoopResult:
    index = pd.DatetimeIndex(["2023-01-16 00:00", "2023-01-16 01:00"])
    frame = pd.DataFrame(
        {
            "fuel": 0.0,
            "bought": [bought, 0.0],
            "sold": 0.0,
            "space_001.TAir_end": 294.0,
            "space_001_lower": 290.0,
            "space_001_upper": 300.0,
            "price": 0.3,
        },
        index=index,
    )
    prices = Prices.of(FlatTariff(price=0.3), index)
    metrics = ControlMetrics.compute(
        frame, prices, np.full((1, 2), 294.0), np.full((1, 2), 290.0), np.full((1, 2), 300.0), 3600.0
    )
    return ClosedLoopResult(name, frame, metrics, 3600.0, ["space_001"], ["space_001.TAir"])


def test_compare_guards_a_reference_without_energy() -> None:
    comparison = compare({"reference": _result("reference", 0.0), "other": _result("other", 1.0)}, baseline="reference")
    assert np.isnan(comparison.loc["other", "cost_saving_pct"])
    assert comparison.loc["other", "total_cost"] == pytest.approx(0.3)
    assert comparison.loc["other", "shifted_energy_kwh"] == pytest.approx(0.5)


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(QUICK))
def test_every_scenario_runs(heat_pump: BuildingModel, true_building: BuildingModel, name: str) -> None:
    result = run_scenario(heat_pump, QUICK[name], plant=true_building)
    assert result.failures == 0
    assert result.metrics.max_violation < 1.0


@pytest.mark.slow
def test_flexibility_under_a_time_of_use_tariff(project: Project, true_building: BuildingModel) -> None:
    scenario = scenario_catalogue(days=1, horizon=12)["time_of_use"]
    table = value_of_flexibility(project.model, scenario, plant=true_building)
    assert table.loc["price aware", "total_cost"] <= table.loc["price agnostic", "total_cost"] + 1e-6
    assert table.loc["price aware", "low_price_share"] >= table.loc["price agnostic", "low_price_share"]
