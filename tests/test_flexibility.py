"""Flexibility quantification: activation requests with rebound, flexibility function, envelope, comparisons."""

import numpy as np
import pandas as pd
import pytest

from mpctrano import (
    ActivationRequest,
    ActivationResult,
    ComfortSpec,
    FastModel,
    FlatTariff,
    FlexibilityEnvelope,
    ModelPlant,
    ModelPredictiveController,
    MPCSettings,
    Scenario,
    TimeOfUseTariff,
    WeatherForecaster,
    flexibility_function,
    respond,
    spin_up,
)
from mpctrano.flexibility import compare, plan_comfort_violation, price_agnostic_tariff
from mpctrano.metrics import ControlMetrics, Prices
from mpctrano.simulation import ClosedLoopResult
from mpctrano.tariff import CapacityTariff, DemandResponseEvent
from tests.conftest import HEAT_PUMP_HOUSE, START

STEP = 3600.0


def toy_result(power: np.ndarray, name: str) -> ClosedLoopResult:
    """A run with a given billed power [W] and the comfort band always met."""
    index = pd.date_range(START, periods=len(power), freq="h")
    frame = pd.DataFrame({"fuel": 0.0, "bought": power / 1000, "sold": 0.0, "P_meter": power, "price": 0.25}, index)
    band = np.full((1, len(power)), 294.15)
    metrics = ControlMetrics.compute(frame, Prices.of(FlatTariff(), index), band, band - 1, band + 3, STEP)
    return ClosedLoopResult(
        name=name, frame=frame, metrics=metrics, time_step=STEP, zones=["z"], comfort_states=["z.TAir"]
    )


def test_activation_metrics_on_a_toy_profile() -> None:
    baseline = np.full(24, 2000.0)
    response = baseline.copy()
    response[8:10] = 500.0  # shed 1.5 kW for 2 h
    response[10:13] = [3000.0, 2500.0, 2000.0]  # rebound 1.5 kWh
    start = pd.Timestamp(START)
    request = ActivationRequest(start=start + pd.Timedelta(hours=8), duration_hours=2, direction="down", power=1500.0)
    metrics = ActivationResult(request, toy_result(baseline, "base"), toy_result(response, "resp")).metrics
    assert metrics["delivered_energy_kwh"] == pytest.approx(3.0)
    assert metrics["delivery_ratio"] == pytest.approx(1.0)
    assert metrics["rebound_energy_kwh"] == pytest.approx(1.5)
    assert metrics["rebound_peak_kw"] == pytest.approx(1.0)
    assert metrics["rebound_ratio"] == pytest.approx(0.5)
    assert metrics["rebound_duration_h"] == pytest.approx(2.0)
    assert metrics["net_energy_change_kwh"] == pytest.approx(-1.5)
    assert request.event(2000.0).power_limit == pytest.approx(500.0)
    up = ActivationRequest(start=start, duration_hours=1, direction="up", power=1000.0).event(2000.0)
    assert up.power_floor == pytest.approx(3000.0)


def test_compare_reports_the_peak_reduction() -> None:
    flat = np.full(24, 2000.0)
    peaky = flat.copy()
    peaky[7] = 4000.0
    frame = compare({"peaky": toy_result(peaky, "peaky"), "flat": toy_result(flat, "flat")}, baseline="peaky")
    assert frame.loc["flat", "peak_reduction_pct"] == pytest.approx(50.0)
    assert frame.loc["flat", "shifted_energy_kwh"] == pytest.approx(1.0)


def test_the_price_agnostic_tariff_keeps_the_grid_constraints() -> None:
    index = pd.date_range(START, periods=24, freq="h")
    event = DemandResponseEvent(start=index[17], end=index[19], price_multiplier=3.0, power_floor=1500.0)
    tariff = CapacityTariff(base=TimeOfUseTariff().with_events([event]))
    agnostic = price_agnostic_tariff(tariff, index)
    assert np.ptp(agnostic.energy_price(index)) == 0.0
    assert agnostic.power_floor(index)[17] == 1500.0


@pytest.fixture(scope="module")
def spun_up(heat_pump: FastModel) -> np.ndarray:
    return spin_up(heat_pump, FastModel.from_yaml(HEAT_PUMP_HOUSE, library="FastModel"), START, days=2.0)


@pytest.mark.slow
def test_respond_sheds_and_rebounds(heat_pump: FastModel, spun_up: np.ndarray) -> None:
    scenario = Scenario(
        name="tou",
        description="",
        tariff=TimeOfUseTariff(),
        settings=MPCSettings(horizon=12),
        start=START,
        days=0.5,
        estimator="perfect",
    )
    start = pd.Timestamp(START)
    request = ActivationRequest(start=start + pd.Timedelta(hours=5), duration_hours=2, power=400.0, notice_hours=1)
    result = respond(heat_pump, scenario, request, plant=lambda: ModelPlant(heat_pump, STEP, START, x0=spun_up))
    window = request.window(pd.DatetimeIndex(result.frame.index))
    assert result.frame.loc[window, "response"].mean() < result.frame.loc[window, "baseline"].mean()
    assert result.metrics["delivered_energy_kwh"] > 0.1


@pytest.mark.slow
def test_flexibility_function_responds_to_a_price_step(heat_pump: FastModel, spun_up: np.ndarray) -> None:
    scenario = Scenario(
        name="flat",
        description="",
        tariff=FlatTariff(),
        settings=MPCSettings(horizon=12),
        start=START,
        days=0.5,
        estimator="perfect",
    )
    response = flexibility_function(
        heat_pump,
        scenario,
        pd.Timestamp(START) + pd.Timedelta(hours=5),
        2.0,
        price_factor=4.0,
        plant=lambda: ModelPlant(heat_pump, STEP, START, x0=spun_up),
    )
    assert response.metrics["delivered_energy_kwh"] > 0.0


@pytest.mark.slow
def test_flexibility_envelope(heat_pump: FastModel, spun_up: np.ndarray) -> None:
    controller = ModelPredictiveController(heat_pump, MPCSettings(horizon=12), ComfortSpec(), FlatTariff())
    forecast = WeatherForecaster.from_model(heat_pump).forecast(pd.Timestamp(START) + pd.Timedelta(hours=6), 12, STEP)
    envelope = FlexibilityEnvelope.compute(controller, spun_up, forecast, durations=(1.0, 2.0), iterations=3)
    frame = envelope.frame
    assert list(frame.index) == [1.0, 2.0]
    assert (frame[["down_kw", "up_kw"]] >= 0).all().all()
    assert frame["up_kw"].iloc[0] > 0.0  # the heat pump can always run harder for an hour
    assert frame["up_kw"].iloc[1] <= frame["up_kw"].iloc[0] + 1e-6  # holding longer is not easier
    plan = controller.solve(spun_up, forecast)
    assert plan_comfort_violation(plan, heat_pump) >= 0.0
