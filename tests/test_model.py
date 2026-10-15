"""FastModel: the ``FastModelMPC`` models of Trano's single-zone houses, with a gas boiler or a heat pump."""

import numpy as np
import pandas as pd
import pytest

from mpctrano import (
    FastModel,
    WeatherForecaster,
)
from mpctrano.model import ModelError
from tests.conftest import BOILER_HOUSE, START, WEATHER

HOUSE = BOILER_HOUSE
CONTROLS = ["valve_003_y.y", "boiler_001_y.y", "three_way_valve_001_y.y"]
FUEL_PRICE = 0.08


@pytest.fixture(scope="module")
def house() -> FastModel:
    return FastModel.from_yaml(HOUSE)


def disturbances(model: FastModel, start: str, steps: int) -> pd.DataFrame:
    return WeatherForecaster.from_model(model).realisation(pd.Timestamp(start), steps, 3600.0)


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


def test_heat_pump_house_meters_its_electricity(heat_pump: FastModel) -> None:
    assert heat_pump.control_names == ["valve_001_y.y", "boiler_001_y.y", "three_way_valve_001_y.y"]
    assert heat_pump.meters == {"boiler_001.EEle": "electricity"}
    assert heat_pump.weather_path == WEATHER.resolve()
    assert heat_pump.occupancy["occupancy_1"].start_hour == 7.0


def test_the_temperature_states_have_their_unit(heat_pump: FastModel) -> None:
    """rumoca drops the unit of array states (wall layers, radiator water); FastModel restores it."""
    units = {state.name: state.unit for state in heat_pump.model.description.states}
    assert units["externalwall_0.T[3]"] == "K"
    assert units["radiator_001.T[1]"] == "K"
    assert units["boiler_001.T"] == "K"
    assert units["boiler_001.EEle"] == "kWh"
