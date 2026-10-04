import pandas as pd
import pytest

from mpctrano import (
    ClosedLoopResult,
    FlatTariff,
    Project,
    RCModel,
    compare,
    price_response,
    run_scenario,
    scenario_catalogue,
    value_of_flexibility,
)
from mpctrano.scenarios import Scenario, build_estimator

CATALOGUE = scenario_catalogue(days=0.5)


def test_catalogue_contents() -> None:
    assert set(CATALOGUE) == {
        "flat",
        "time_of_use",
        "dynamic_price",
        "demand_response",
        "peak_shaving",
        "wide_comfort_band",
        "imperfect_forecast",
    }
    scenario = CATALOGUE["demand_response"]
    assert scenario.steps == 12
    assert scenario.with_days(2).steps == 48
    assert scenario.tariff.kind == "demand_response"


@pytest.mark.parametrize("name", sorted(CATALOGUE))
def test_every_scenario_runs(model: RCModel, true_building: RCModel, name: str) -> None:
    result = run_scenario(model, CATALOGUE[name], plant=true_building)
    assert isinstance(result, ClosedLoopResult)
    assert result.name == name
    assert result.failures == 0
    assert result.metrics.max_violation < 1.0  # model mismatch only
    assert result.metrics.energy_kwh > 0


def test_estimator_choice(model: RCModel) -> None:
    from mpctrano import KalmanFilter, MovingHorizonEstimator

    assert isinstance(build_estimator(CATALOGUE["flat"], model), KalmanFilter)
    assert isinstance(build_estimator(CATALOGUE["imperfect_forecast"], model), MovingHorizonEstimator)


@pytest.mark.slow
def test_flexibility_comparison(project: Project, true_building: RCModel) -> None:
    catalogue = scenario_catalogue(days=1)
    names = ("flat", "dynamic_price", "wide_comfort_band")
    results = {name: project.run(catalogue[name], plant=true_building) for name in names}
    comparison = compare(results, baseline="flat")
    assert comparison.loc["flat", "cost_saving_pct"] == 0
    assert comparison.loc["flat", "shifted_energy_kwh"] == 0
    # Same tariff (day-ahead prices): more room to pre-heat buys a larger share of cheap energy.
    assert comparison.loc["wide_comfort_band", "low_price_share"] >= comparison.loc["dynamic_price", "low_price_share"]
    assert comparison.loc["dynamic_price", "low_price_share"] > comparison.loc["flat", "low_price_share"]
    assert comparison["shifted_energy_kwh"].ge(0).all()
    response = price_response(results["dynamic_price"])
    assert response["share"].sum() == pytest.approx(1.0)
    assert set(project.runs) >= set(names)


@pytest.mark.parametrize("name", sorted(CATALOGUE))
def test_scenario_serialisation_round_trip(name: str) -> None:
    scenario = CATALOGUE[name]
    restored = Scenario.model_validate_json(scenario.model_dump_json())
    assert restored.comfort == scenario.comfort
    assert restored.tariff.kind == scenario.tariff.kind
    index = pd.date_range(scenario.start, periods=24, freq="h")
    assert restored.tariff.energy_price(index) == pytest.approx(scenario.tariff.energy_price(index))


def test_metrics_under_another_tariff(model: RCModel) -> None:
    result = run_scenario(model, CATALOGUE["flat"])
    doubled = result.metrics_under(FlatTariff(price=0.5))
    assert doubled.energy_cost == pytest.approx(2 * result.metrics.energy_cost)
    assert doubled.energy_kwh == pytest.approx(result.metrics.energy_kwh)


@pytest.mark.slow
def test_value_of_flexibility(model: RCModel) -> None:
    scenario = scenario_catalogue(days=2)["time_of_use"]
    frame = value_of_flexibility(model, scenario)  # perfect model: the plant is the model itself
    assert list(frame.index) == ["price aware", "price agnostic"]
    aware, agnostic = frame.loc["price aware"], frame.loc["price agnostic"]
    assert aware["total_cost"] < agnostic["total_cost"]
    assert aware["cost_saving_pct"] > 1.0
    assert aware["low_price_share"] > agnostic["low_price_share"]
    assert aware["shifted_energy_kwh"] > 0
    assert aware["comfort_violation_kh"] < 2.0


@pytest.mark.slow
def test_capacity_charge_shaves_the_morning_peak(model: RCModel) -> None:
    scenario = scenario_catalogue(days=1.5)["peak_shaving"]
    frame = value_of_flexibility(model, scenario)
    assert frame.loc["price aware", "peak_power_kw"] < 0.9 * frame.loc["price agnostic", "peak_power_kw"]
    assert frame.loc["price aware", "total_cost"] < frame.loc["price agnostic", "total_cost"]
