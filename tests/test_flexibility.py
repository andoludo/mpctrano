"""Flexibility quantification: capacity tariff, activation requests with rebound, flexibility function, envelope."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mpctrano import (
    ActivationRequest,
    ActivationResult,
    CapacityTariff,
    ComfortSpec,
    DemandResponseEvent,
    FlatTariff,
    FlexibilityEnvelope,
    Mismatch,
    ModelPredictiveController,
    MPCSettings,
    Scenario,
    TimeOfUseTariff,
    flexibility_function,
    respond,
    run_scenario,
)
from mpctrano.flexibility import compare, plan_comfort_violation, price_agnostic_tariff
from mpctrano.notebook import TutorialBuilding
from mpctrano.scenarios import build_plant
from mpctrano.simulation import ClosedLoopResult

HOUSE = Path(__file__).parent.parent / "tutorials" / "models" / "house"
START = pd.Timestamp("2023-01-16")


@pytest.fixture(scope="module")
def building() -> TutorialBuilding:
    return TutorialBuilding(HOUSE, "base")


# --------------------------------------------------------------------------- capacity tariff
def test_capacity_tariff_monthly_charge_and_peaks() -> None:
    tariff = CapacityTariff(base=FlatTariff(price=0.25), capacity_price=48.0)
    assert tariff.capacity_charge == pytest.approx(4.0)
    index = pd.date_range("2023-01-30", periods=72, freq="h")
    power = np.zeros(72)
    power[10] = 6000.0  # January peak 6 kW
    power[60] = 1000.0  # February peak below the 2.5 kW floor
    peaks = tariff.monthly_peaks(index, power)
    assert list(peaks.round(2)) == [6.0, 2.5]
    assert tariff.capacity_cost(index, power) == pytest.approx((6.0 + 2.5) * 4.0)
    signal = tariff.signal(index)
    assert signal.capacity_charge == pytest.approx(4.0)
    np.testing.assert_allclose(signal.prices, 0.25)
    with_event = tariff.with_events(
        [DemandResponseEvent(start="2023-01-30 10:00", end="2023-01-30 12:00", power_floor=2000.0)]
    )
    assert isinstance(with_event, CapacityTariff)
    assert with_event.power_floor(index)[10] == 2000.0 and with_event.power_floor(index)[13] == 0.0
    assert isinstance(price_agnostic_tariff(with_event, index), FlatTariff | type(with_event.base))


def test_running_peak_is_carried_in_the_closed_loop(building: TutorialBuilding) -> None:
    tariff = CapacityTariff(base=TimeOfUseTariff(peak_price=0.35, off_peak_price=0.15), capacity_price=60.0)
    run = Scenario(
        name="capacity",
        description="",
        tariff=tariff,
        days=1.0,
        mismatch=Mismatch(plant_model_type=None, parameter_std=0.1),
    )
    result = run_scenario(building.control, run)
    frame = result.frame
    assert "running_peak_kw" in frame
    assert frame["running_peak_kw"].is_monotonic_increasing
    assert frame["running_peak_kw"].iloc[-1] == pytest.approx(frame["P_meter"].iloc[:-1].max() / 1000, rel=1e-6)
    assert result.metrics.capacity_cost == pytest.approx(5.0 * result.metrics.peak_power_kw)


def test_capacity_charge_lowers_the_peak(building: TutorialBuilding) -> None:
    model = building.control
    forecast = building.forecaster.forecast(START, 24, 3600.0)
    comfort = ComfortSpec()
    plain = ModelPredictiveController(model, MPCSettings(horizon=24), comfort, FlatTariff(price=0.25)).solve(
        model.initial_state, forecast
    )
    charged = ModelPredictiveController(
        model, MPCSettings(horizon=24), comfort, CapacityTariff(base=FlatTariff(price=0.25), capacity_price=120.0)
    ).solve(model.initial_state, forecast, peak_so_far=0.0)
    assert plain.meter is not None and charged.meter is not None
    assert charged.meter["P_meter"].max() <= plain.meter["P_meter"].max() + 1.0
    assert charged.costs["capacity"] > 0
    with_history = ModelPredictiveController(
        model, MPCSettings(horizon=24), comfort, CapacityTariff(base=FlatTariff(price=0.25), capacity_price=120.0)
    ).solve(model.initial_state, forecast, peak_so_far=3000.0)
    assert with_history.costs["capacity"] >= 10.0 * 3.0 - 1e-3  # 120/12 EUR/kW x 3 kW at least


# --------------------------------------------------------------------------- activation and rebound
def _toy_result(power: np.ndarray, name: str) -> ClosedLoopResult:
    from mpctrano.metrics import ControlMetrics

    index = pd.date_range(START, periods=len(power), freq="h")
    frame = pd.DataFrame(
        {"P_meter": power, "P_export": 0.0, "P_gas": 0.0, "P_generation": 0.0, "price": 0.25}, index=index
    )
    frame["z_Ti_end"], frame["z_lower"], frame["z_upper"] = 294.15, 293.15, 297.15
    metrics = ControlMetrics.compute(
        index,
        power,
        np.full((1, len(power)), 294.15),
        np.full((1, len(power)), 293.15),
        np.full((1, len(power)), 297.15),
        np.full(len(power), 0.25),
        3600.0,
    )
    return ClosedLoopResult(name=name, frame=frame, metrics=metrics, time_step=3600.0)


def test_activation_metrics_on_a_toy_profile() -> None:
    baseline = np.full(24, 2000.0)
    response = baseline.copy()
    response[8:10] = 500.0  # shed 1.5 kW for 2 h
    response[10:13] = [3000.0, 2500.0, 2000.0]  # rebound 1.5 kWh
    request = ActivationRequest(start=START + pd.Timedelta(hours=8), duration_hours=2, direction="down", power=1500.0)
    result = ActivationResult(request, _toy_result(baseline, "base"), _toy_result(response, "resp"))
    metrics = result.metrics
    assert metrics["delivered_energy_kwh"] == pytest.approx(3.0)
    assert metrics["delivery_ratio"] == pytest.approx(1.0)
    assert metrics["rebound_energy_kwh"] == pytest.approx(1.5)
    assert metrics["rebound_peak_kw"] == pytest.approx(1.0)
    assert metrics["rebound_ratio"] == pytest.approx(0.5)
    assert metrics["rebound_duration_h"] == pytest.approx(2.0)
    assert metrics["net_energy_change_kwh"] == pytest.approx(-1.5)
    event = request.event(2000.0)
    assert event.power_limit == pytest.approx(500.0)
    up = ActivationRequest(start=START, duration_hours=1, direction="up", power=1000.0).event(2000.0)
    assert up.power_floor == pytest.approx(3000.0)


def test_respond_sheds_and_rebounds(building: TutorialBuilding) -> None:
    model = building.control
    run = Scenario(name="tou", description="", tariff=TimeOfUseTariff(peak_price=0.35, off_peak_price=0.15), days=1.5)
    baseline = run_scenario(model, run, plant=build_plant(model, Mismatch.none(), 3600.0))
    # Ask for half of the billed power of the busiest two hours of the first day.
    billed = baseline.frame["P_meter"].iloc[:22]
    busiest = billed.rolling(2).mean().idxmax() - pd.Timedelta(hours=1)
    power = 0.5 * float(billed.loc[busiest : busiest + pd.Timedelta(hours=1)].mean())
    request = ActivationRequest(start=busiest, duration_hours=2, direction="down", power=power)
    result = respond(model, run, request, plant=lambda: build_plant(model, Mismatch.none(), 3600.0), baseline=baseline)
    metrics = result.metrics
    window = request.window(pd.DatetimeIndex(result.frame.index))
    baseline_power = result.frame.loc[window, "baseline"].mean()
    assert result.response.frame.loc[window, "P_meter"].max() <= baseline_power - power + 0.1 * power
    assert metrics["delivered_energy_kwh"] > 0.8 * request.requested_energy_kwh
    assert metrics["rebound_energy_kwh"] >= 0
    assert "comfort_change_kh" in metrics


def test_flexibility_function_responds_to_a_price_step(building: TutorialBuilding) -> None:
    model = building.control
    run = Scenario(name="flat", description="", tariff=FlatTariff(price=0.25), days=1.5)
    result = flexibility_function(
        model,
        run,
        START + pd.Timedelta(hours=10),
        3.0,
        price_factor=4.0,
        plant=lambda: build_plant(model, Mismatch.none(), 3600.0),
    )
    assert result.request.direction == "down"
    assert result.metrics["delivered_energy_kwh"] > 0  # the billed power drops during the expensive hours


# --------------------------------------------------------------------------- envelope
def test_flexibility_envelope(building: TutorialBuilding) -> None:
    model = building.control
    controller = ModelPredictiveController(model, MPCSettings(horizon=24), ComfortSpec(), FlatTariff(price=0.25))
    forecast = building.forecaster.forecast(START + pd.Timedelta(hours=6), 24, 3600.0)
    envelope = FlexibilityEnvelope.compute(
        controller, model.initial_state, forecast, durations=(1.0, 4.0), iterations=5
    )
    frame = envelope.frame
    assert list(frame.index) == [1.0, 4.0]
    assert (frame[["down_kw", "up_kw"]] >= 0).all().all()
    assert frame.loc[1.0, "up_kw"] > 0
    assert frame.loc[1.0, "down_kw"] >= frame.loc[4.0, "down_kw"] - 1e-6  # shorter windows allow at least as much power
    assert frame.loc[4.0, "up_energy_kwh"] >= frame.loc[1.0, "up_energy_kwh"] - 1e-6
    solution = controller.solve(model.initial_state, forecast)
    assert plan_comfort_violation(solution, model) < 0.5


APARTMENTS = HOUSE.parent / "apartments"


def test_power_floor_cannot_be_met_by_exporting(building: TutorialBuilding) -> None:
    """Without PV or a battery the meter never exports, so a floor above the compressor capacity is infeasible."""
    block = TutorialBuilding(APARTMENTS, "base")
    model = block.control
    controller = ModelPredictiveController(model, MPCSettings(horizon=24), ComfortSpec(), FlatTariff(price=0.25))
    forecast = block.forecaster.forecast(START + pd.Timedelta(hours=6), 24, 3600.0)
    capacity = model.consumption_capacity(controller.upper_bounds)
    assert capacity == pytest.approx(8000.0)  # one compressor shared by six apartments and the tank
    floors = np.zeros(24)
    floors[:2] = 50_000.0
    solution = controller.solve(model.initial_state, forecast, power_floors=floors)
    assert solution.meter is not None
    assert solution.meter["P_export"].max() < 1.0
    assert solution.meter["P_meter"].max() <= capacity + 1.0
    envelope = FlexibilityEnvelope.compute(controller, model.initial_state, forecast, durations=(2.0,), iterations=3)
    assert envelope.frame.loc[2.0, "up_kw"] <= capacity / 1000 + 1e-6


def test_compare_reports_peak_reduction(building: TutorialBuilding) -> None:
    model = building.control
    flat = run_scenario(model, Scenario(name="flat", description="", tariff=FlatTariff(price=0.25), days=0.5))
    capped = run_scenario(
        model,
        Scenario(
            name="capped",
            description="",
            tariff=CapacityTariff(base=FlatTariff(price=0.25), capacity_price=200.0),
            days=0.5,
        ),
    )
    table = compare({"flat": flat, "capped": capped}, baseline="flat")
    assert "peak_reduction_pct" in table.columns and table.loc["flat", "peak_reduction_pct"] == 0.0
