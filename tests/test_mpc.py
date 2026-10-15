"""Economic MPC of the FastModel houses: the fuel of the boiler house, the electricity of the heat pump house."""

import numpy as np
import pandas as pd
import pytest

from mpctrano import (
    ComfortSpec,
    DemandResponseEvent,
    FastModel,
    FlatTariff,
    ModelPredictiveController,
    MPCSettings,
    TimeOfUseTariff,
    WeatherForecaster,
)
from tests.conftest import BOILER_HOUSE, START

CONTROLS = ["valve_003_y.y", "boiler_001_y.y", "three_way_valve_001_y.y"]
FUEL_PRICE = 0.08


@pytest.fixture(scope="module")
def house() -> FastModel:
    return FastModel.from_yaml(BOILER_HOUSE)


def disturbances(model: FastModel, start: str, steps: int) -> pd.DataFrame:
    return WeatherForecaster.from_model(model).realisation(pd.Timestamp(start), steps, 3600.0)


def test_mpc_keeps_comfort_at_the_cost_of_the_metered_fuel(house: FastModel) -> None:
    settings = MPCSettings(horizon=24, fuel_price=FUEL_PRICE)
    controller = ModelPredictiveController(house, settings, ComfortSpec(), FlatTariff(price=0.3))
    forecast = disturbances(house, START, 25)
    plan = controller.solve(house.initial_state, forecast)
    assert plan.success, plan.status
    assert plan.controls.columns.tolist() == CONTROLS
    assert plan.controls.to_numpy().min() >= 0.0
    assert plan.controls.to_numpy().max() <= 1.0
    temperatures = plan.states[house.comfort_states[0]].to_numpy()[1:]
    assert np.all(temperatures >= plan.lower[0] - 0.1)
    meter = plan.states["boiler_001.EFue"].to_numpy()
    assert plan.costs["fuel"] == pytest.approx(FUEL_PRICE * (meter[-1] - meter[0]), rel=1e-6)
    assert plan.costs["electricity"] == pytest.approx(0.0, abs=1e-6)
    assert controller.energy is not None
    assert controller.energy["fuel"].sum() == pytest.approx(meter[-1] - meter[0], rel=1e-6)

    # receding horizon: the next step starts from the first predicted state, warm started from this plan
    following = controller.solve(plan.states.iloc[1].to_numpy(), forecast.iloc[1:], plan.controls.iloc[0].to_numpy())
    assert following.success, following.status
    assert following.index[0] == forecast.index[1]


def test_mpc_moves_the_fuel_out_of_the_peak_hours(house: FastModel) -> None:
    forecast = disturbances(house, START, 24)  # a Monday: 7h to 22h peak
    fuel = {}
    for name, tariff in {"flat": FlatTariff(price=0.25), "time_of_use": TimeOfUseTariff()}.items():
        controller = ModelPredictiveController(house, MPCSettings(horizon=24), ComfortSpec(), tariff)
        assert controller.solve(house.initial_state, forecast).success
        assert controller.energy is not None
        fuel[name] = controller.energy["fuel"]
    peak = TimeOfUseTariff().is_peak(pd.DatetimeIndex(forecast.index))
    assert fuel["time_of_use"][~peak].sum() > fuel["flat"][~peak].sum() + 0.1
    assert fuel["time_of_use"][peak].sum() < fuel["flat"][peak].sum()


def test_heat_pump_mpc_costs_the_metered_electricity(heat_pump: FastModel) -> None:
    controller = ModelPredictiveController(heat_pump, MPCSettings(horizon=24), ComfortSpec(), FlatTariff(price=0.3))
    plan = controller.solve(heat_pump.initial_state, disturbances(heat_pump, START, 24))
    assert plan.success, plan.status
    meter = plan.states["boiler_001.EEle"].to_numpy()
    assert meter[-1] - meter[0] > 0.5  # kWh: the heat pump runs
    assert plan.costs["fuel"] == pytest.approx(0.0, abs=1e-9)
    assert plan.costs["electricity"] == pytest.approx(0.3 * (meter[-1] - meter[0]), abs=0.01)
    temperatures = plan.states["space_001.TAir"].to_numpy()[1:]
    assert np.all(temperatures >= plan.lower[0] - 0.1)


def test_heat_pump_mpc_follows_the_grid_signals(heat_pump: FastModel) -> None:
    """A demand response power cap and a capacity charge lower the electric power the heat pump draws."""
    forecast = disturbances(heat_pump, START, 24)
    start = pd.Timestamp(START)
    event = DemandResponseEvent(
        start=start + pd.Timedelta(hours=16), end=start + pd.Timedelta(hours=20), power_limit=300
    )
    tariffs = {
        "free": TimeOfUseTariff(),
        "capped": TimeOfUseTariff().with_events([event]),
        "capacity": TimeOfUseTariff(capacity_charge=20.0),
    }
    power = {}
    for name, tariff in tariffs.items():
        controller = ModelPredictiveController(heat_pump, MPCSettings(horizon=24), ComfortSpec(), tariff)
        assert controller.solve(heat_pump.initial_state, forecast).success
        assert controller.energy is not None
        power[name] = controller.energy["power"].to_numpy()
    window = slice(16, 20)
    assert power["capped"][window].max() <= 300 + 20  # W
    assert power["capped"][window].sum() <= power["free"][window].sum() + 1.0
    assert power["capacity"].max() <= power["free"].max() + 1.0
