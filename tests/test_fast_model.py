"""FastModel: the ``FastModelMPC`` model of Trano's single-zone hydronic house (boiler, mixing valve, radiator).

Skipped unless rumoca exports connected Modelica models to CasADi (``make rumoca-fastmodel``).
"""

import numpy as np
import pandas as pd
import pytest

from mpctrano import (
    ComfortSpec,
    FastModel,
    FastModelPredictiveController,
    FastMPCSettings,
    FlatTariff,
    FastWeatherForecaster,
    TimeOfUseTariff,
)
from mpctrano.fast_model import rumoca_exports_fast_models
from mpctrano.model import ModelError
from tests.conftest import MODELS, START

pytestmark = pytest.mark.skipif(
    not rumoca_exports_fast_models(),
    reason="the CasADi export of FastModel models needs rumoca main at c74cddf (make rumoca-fastmodel)",
)

HOUSE = MODELS / "single_zone_hydronic.yaml"
CONTROLS = ["valve_003_y.y", "boiler_001_y.y", "three_way_valve_001_y.y"]
FUEL_PRICE = 0.08


@pytest.fixture(scope="module")
def house() -> FastModel:
    return FastModel.from_yaml(HOUSE)


def disturbances(model: FastModel, start: str, steps: int) -> pd.DataFrame:
    return FastWeatherForecaster.from_model(model).realisation(pd.Timestamp(start), steps, 3600.0)


def test_roles_are_read_from_the_modelica_model(house: FastModel) -> None:
    assert house.control_names == CONTROLS
    assert len(house.disturbance_names) == 9
    assert house.zone_names == ["space_001"]
    assert house.comfort_states == ["space_001.TAir"]
    assert house.meters == {"boiler_001.EFue": "fuel"}
    lower, upper = house.control_bounds
    np.testing.assert_array_equal(lower, 0.0)
    np.testing.assert_array_equal(upper, 1.0)
    assert house.heat_controls == []


def test_parameters_can_be_replaced(house: FastModel) -> None:
    name = house.parameter_names[0]
    changed = house.with_parameters({name: 2 * house.parameter(name) + 1})
    assert changed.parameter(name) == 2 * house.parameter(name) + 1
    np.testing.assert_array_equal(house.with_parameters(house.parameters).parameters, house.parameters)
    with pytest.raises(KeyError):
        house.with_parameters({"unknown": 1.0})
    with pytest.raises(ModelError):
        house.with_parameters([1.0])


def test_simulation_heats_the_zone_and_meters_the_fuel(house: FastModel) -> None:
    steps = 12
    forecast = disturbances(house, START, steps).to_numpy().T
    zone, meter = house.comfort_indices[0], house.state_names.index("boiler_001.EFue")
    heated = house.simulate(house.initial_state, np.ones((3, steps)), forecast, 3600.0)
    cold = house.simulate(house.initial_state, np.zeros((3, steps)), forecast, 3600.0)
    assert heated.shape == (house.n_states, steps + 1)
    assert heated[zone, -1] > cold[zone, -1] + 1.0
    assert heated[meter, -1] - heated[meter, 0] > 1.0  # kWh
    assert abs(cold[meter, -1] - cold[meter, 0]) < 1e-3


def test_mpc_keeps_comfort_at_the_cost_of_the_metered_fuel(house: FastModel) -> None:
    settings = FastMPCSettings(horizon=24, fuel_price=FUEL_PRICE)
    controller = FastModelPredictiveController(house, settings, ComfortSpec(), FlatTariff(price=0.3))
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
        controller = FastModelPredictiveController(house, FastMPCSettings(horizon=24), ComfortSpec(), tariff)
        assert controller.solve(house.initial_state, forecast).success
        assert controller.energy is not None
        fuel[name] = controller.energy["fuel"]
    peak = TimeOfUseTariff().is_peak(pd.DatetimeIndex(forecast.index))
    assert fuel["time_of_use"][~peak].sum() > fuel["flat"][~peak].sum() + 0.1
    assert fuel["time_of_use"][peak].sum() < fuel["flat"][peak].sum()
