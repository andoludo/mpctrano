"""Interface version 2: systems, meter, bounds, forecasts and the MPC with a heat pump, tank, battery, EV and PV."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mpctrano import (
    ComfortSpec,
    DynamicTariff,
    FlatTariff,
    Meter,
    Mismatch,
    ModelPlant,
    ModelPredictiveController,
    MPCSettings,
    PriceSignal,
    Project,
    RCModel,
    Scenario,
    WeatherForecaster,
    perturb,
    run_scenario,
)

MODELS = Path(__file__).parent / "models"
HOUSE = MODELS / "house_systems.yaml"
START = pd.Timestamp("2023-01-16")


@pytest.fixture(scope="module")
def house() -> RCModel:
    return RCModel.from_yaml(HOUSE)


@pytest.fixture(scope="module")
def forecaster(house: RCModel) -> WeatherForecaster:
    return WeatherForecaster.from_model(house)


# --------------------------------------------------------------------------- model and interface
def test_model_exposes_the_systems(house: RCModel) -> None:
    assert house.interface.version == "2"
    assert [system.kind for system in house.systems] == [
        "chiller", "heat_pump", "battery", "ev_charger", "photovoltaic", "dhw_tank",
    ]  # fmt: skip
    assert house.state_names[-3:] == ["battery_001_E", "ev_001_E", "tank_001_T"]
    assert house.measured_states == ["space_001_Ti", "space_002_Ti", "battery_001_E", "ev_001_E", "tank_001_T"]
    names = house.control_names
    assert [names[i] for i in house.zone_heating_indices] == ["space_001_PHea", "space_002_PHea"]
    consumption = [names[i] for i in house.consumption_control_indices]
    assert consumption[:4] == ["space_001_PHea", "space_001_PCoo", "space_002_PHea", "space_002_PCoo"]
    assert consumption[4:] == ["battery_001_PCha", "ev_001_PCha", "tank_001_PHea"]
    assert [names[i] for i in house.discharge_control_indices] == ["battery_001_PDis"]
    assert house.heat_control_indices == [] and house.gas_control_indices == []
    assert [house.disturbance_names[i] for i in house.generation_indices] == ["pv_001_P"]
    assert house.has_meter
    groups = {group.system: group for group in house.control_groups}
    assert [names[i] for i in groups["heatpump_001"].indices] == ["space_001_PHea", "space_002_PHea", "tank_001_PHea"]
    assert groups["heatpump_001"].max_power == 2500.0
    lower, upper = house.control_bounds()
    assert dict(zip(names, upper, strict=True))["battery_001_PCha"] == 3000.0
    assert dict(zip(names, upper, strict=True))["space_001_PHea"] == 2500.0
    assert np.all(lower == 0)
    state_lower, state_upper = house.state_bounds()
    battery = house.state_names.index("battery_001_E")
    assert (state_lower[battery], state_upper[battery]) == pytest.approx((0.1 * 8 * 3.6e6, 0.9 * 8 * 3.6e6))
    assert len(house.storage_bands) == 1 and len(house.vehicles) == 1


def test_perturb_leaves_the_systems_alone(house: RCModel) -> None:
    perturbed = perturb(house, relative_std=0.5, seed=1)
    for name, before, after in zip(house.parameter_names, house.parameters, perturbed.parameters, strict=True):
        if name.startswith(("heatpump_", "chiller_", "battery_", "ev_", "tank_")):
            assert before == after, name
    assert perturbed.parameter("space_001_Ci") != house.parameter("space_001_Ci")


def test_forecaster_builds_pv_draw_off_and_driving(house: RCModel, forecaster: WeatherForecaster) -> None:
    frame = forecaster.realisation(START, 48, 3600.0)
    assert set(house.disturbance_names) <= set(frame.columns)
    pv = frame["pv_001_P"]
    assert pv.min() >= 0 and pv.max() > 0
    assert pv[frame.index.hour == 2].max() == 0.0
    draw = frame["tank_001_QDraw"]
    assert draw.sum() * 3600 / 3.6e6 == pytest.approx(2 * 6.0)  # 6 kWh per day
    driving = frame["ev_001_PDri"]
    assert driving[(frame.index.hour >= 7) & (frame.index.hour < 18)].min() > 0
    assert driving[frame.index.hour >= 18].max() == 0.0
    assert driving.sum() * 3600 / 3.6e6 == pytest.approx(2 * 12.0)


def test_meter_evaluates_a_frame(house: RCModel, forecaster: WeatherForecaster) -> None:
    frame = forecaster.realisation(START, 4, 3600.0)
    for name in house.control_names:
        frame[name] = 0.0
    frame["space_001_PHea"] = 1000.0
    frame["battery_001_PDis"] = 400.0
    meter = Meter(house).evaluate(frame)
    np.testing.assert_allclose(meter["P_consumption"], 1000.0)
    np.testing.assert_allclose(meter["P_net"], 600.0 - frame["pv_001_P"])
    assert np.all(meter["P_import"] - meter["P_export"] == meter["P_net"])
    assert np.all(meter["P_gas"] == 0)


# --------------------------------------------------------------------------- MPC
def _controller(house: RCModel, tariff: FlatTariff | DynamicTariff, **settings: object) -> ModelPredictiveController:
    return ModelPredictiveController(house, MPCSettings(horizon=24, **settings), ComfortSpec(), tariff)  # type: ignore[arg-type]


def test_mpc_runs_on_electricity_with_the_shared_compressor_limit(
    house: RCModel, forecaster: WeatherForecaster
) -> None:
    controller = _controller(house, FlatTariff(price=0.25))
    solution = controller.solve(house.initial_state, forecaster.forecast(START, 24, 3600.0))
    assert solution.success, solution.status
    controls = solution.controls
    compressor = controls[["space_001_PHea", "space_002_PHea", "tank_001_PHea"]].sum(axis=1)
    assert compressor.max() <= 2500.0 + 1e-3
    assert solution.meter is not None
    assert (solution.meter["P_import"] >= -1e-6).all() and (solution.meter["P_export"] >= -1e-6).all()
    assert solution.costs["energy"] > 0 and solution.costs["gas"] == 0
    tank = solution.states["tank_001_T"].iloc[1:]
    assert tank.min() >= 318.15 - 0.5  # the tank band is kept
    temperatures = solution.comfort_temperatures[["space_001_Ti", "space_002_Ti"]].to_numpy().T
    assert np.all(temperatures >= solution.lower - 0.5)  # the comfort band is met (soft constraint)


def test_battery_and_vehicle_follow_the_prices(house: RCModel, forecaster: WeatherForecaster) -> None:
    prices = PriceSignal.synthetic(START - pd.Timedelta(days=1), days=3, base=100, amplitude=150, noise=0.0)
    controller = _controller(house, prices.tariff())
    solution = controller.solve(house.initial_state, forecaster.forecast(START, 24, 3600.0))
    assert solution.success, solution.status
    controls = solution.controls
    hourly_price = pd.Series(solution.prices, index=solution.index)
    cheap = hourly_price <= hourly_price.quantile(0.3)
    expensive = hourly_price >= hourly_price.quantile(0.7)
    assert controls.loc[cheap, "battery_001_PCha"].sum() > controls.loc[expensive, "battery_001_PCha"].sum()
    assert controls.loc[expensive, "battery_001_PDis"].sum() > 0
    # The vehicle leaves at 7h: charged to 80 % by then, not charged while away.
    energy = solution.states["ev_001_E"]
    departure = START + pd.Timedelta(hours=7)
    assert energy.loc[departure] >= 0.8 * 50 * 3.6e6 - 1e3
    away = (solution.index.hour >= 7) & (solution.index.hour < 18)
    assert controls.loc[away, "ev_001_PCha"].abs().max() < 1e-6


def test_closed_loop_with_an_r4c3_plant_against_an_r3c2_controller() -> None:
    project = Project.from_yaml(HOUSE)
    assert project.model.interface.zones[0].model_type == "R3C2"
    plant = project.plant(Mismatch(parameter_std=0.2, seed=2), time_step=3600.0)
    assert isinstance(plant, ModelPlant)
    assert plant.model.interface.zones[0].model_type == "R4C3"
    scenario = Scenario(name="hp", description="", tariff=FlatTariff(price=0.25), days=1.0)
    result = project.run(scenario, plant=plant)
    assert result.failures == 0
    frame = result.frame
    assert {"P_import", "P_export", "P_generation", "P_meter", "tank_001_T_end", "ev_001_E_measured"} <= set(
        frame.columns
    )
    assert result.heating_columns == ["space_001_PHea", "space_002_PHea"]
    assert result.zones == ["space_001", "space_002"]
    assert result.metrics.generation_kwh > 0
    assert 0 <= result.metrics.self_consumption <= 1
    assert result.metrics.total_cost == pytest.approx(
        result.metrics.energy_cost
        + result.metrics.capacity_cost
        + result.metrics.gas_cost
        - result.metrics.export_revenue
    )
    assert "space_001_Th_true" in frame.columns  # the plant has the emitter state


def test_run_scenario_with_mismatch_uses_a_perturbed_plant(house: RCModel) -> None:
    scenario = Scenario(
        name="m", description="", tariff=FlatTariff(price=0.25), days=0.5, mismatch=Mismatch(plant_model_type=None)
    )
    result = run_scenario(house, scenario)
    assert result.failures == 0
    assert result.metrics.energy_kwh > 0
