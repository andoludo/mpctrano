"""BuildingModel: the ``FastSmoothBuildingsMPC`` models of Trano's single-zone houses, with a gas boiler or a heat
pump."""

import numpy as np
import pandas as pd
import pytest

import mpctrano.model
from mpctrano import (
    BuildingModel,
    WeatherForecaster,
)
from mpctrano.model import ModelError
from tests.conftest import BOILER_HOUSE, START, WEATHER

HOUSE = BOILER_HOUSE
CONTROLS = ["valve_003_y.y", "boiler_001_y.y", "three_way_valve_001_y.y"]
FUEL_PRICE = 0.08


@pytest.fixture(scope="module")
def house() -> BuildingModel:
    return BuildingModel.from_yaml(HOUSE)


def disturbances(model: BuildingModel, start: str, steps: int) -> pd.DataFrame:
    return WeatherForecaster.from_model(model).realisation(pd.Timestamp(start), steps, 3600.0)


def test_roles_are_read_from_the_modelica_model(house: BuildingModel) -> None:
    assert sorted(house.control_names) == sorted(CONTROLS)
    assert len(house.disturbance_names) == 9
    assert house.zone_names == ["space_001"]
    assert house.comfort_states == ["space_001.TAir"]
    assert house.meters == {"boiler_001.EFue": "fuel"}
    lower, upper = house.control_bounds
    np.testing.assert_array_equal(lower, 0.0)
    np.testing.assert_array_equal(upper, 1.0)
    assert house.heat_controls == []


def test_parameters_can_be_replaced(house: BuildingModel) -> None:
    name = house.parameter_names[0]
    changed = house.with_parameters({name: 2 * house.parameter(name) + 1})
    assert changed.parameter(name) == 2 * house.parameter(name) + 1
    np.testing.assert_array_equal(house.with_parameters(house.parameters).parameters, house.parameters)
    with pytest.raises(KeyError):
        house.with_parameters({"unknown": 1.0})
    with pytest.raises(ModelError):
        house.with_parameters([1.0])


def test_simulation_heats_the_zone_and_meters_the_fuel(house: BuildingModel) -> None:
    steps = 12
    forecast = disturbances(house, START, steps).to_numpy().T
    zone, meter = house.comfort_indices[0], house.state_names.index("boiler_001.EFue")
    heated = house.simulate(house.initial_state, np.ones((3, steps)), forecast, 3600.0)
    cold = house.simulate(house.initial_state, np.zeros((3, steps)), forecast, 3600.0)
    assert heated.shape == (house.n_states, steps + 1)
    assert heated[zone, -1] > cold[zone, -1] + 1.0
    assert heated[meter, -1] - heated[meter, 0] > 1.0  # kWh
    assert abs(cold[meter, -1] - cold[meter, 0]) < 1e-3


def test_heat_pump_house_meters_its_electricity(heat_pump: BuildingModel) -> None:
    assert sorted(heat_pump.control_names) == ["boiler_001_y.y", "three_way_valve_001_y.y", "valve_001_y.y"]
    assert heat_pump.meters == {"boiler_001.EEle": "electricity"}
    assert heat_pump.weather_path == WEATHER.resolve()
    assert heat_pump.occupancy["occupancy_1"].start_hour == 7.0


def test_the_temperature_states_have_their_unit(heat_pump: BuildingModel) -> None:
    """rumoca drops the unit of array states (wall layers, radiator water); BuildingModel restores it."""
    units = {state.name: state.unit for state in heat_pump.model.description.states}
    assert units["externalwall_0.T[3]"] == "K"
    assert units["radiator_001.T[1]"] == "K"
    assert units["boiler_001.T"] == "K"
    assert units["boiler_001.EEle"] == "kWh"


def test_smooth_controllers_widens_the_transitions_and_slows_the_switches() -> None:
    from mpctrano import CalibrationSpec, spin_up
    from mpctrano.model import CLOSED_LOOP_LIBRARY
    from tests.conftest import MODELS

    spec = CalibrationSpec.from_yaml(MODELS / "single_zone_calibration.yaml")
    closed_loop = spec.model(CLOSED_LOOP_LIBRARY)
    widths = tuple(n for n in closed_loop.parameter_names if n.endswith((".dU", ".dY", ".dMax", ".dT", ".dtSet")))
    switches = tuple(n for n in closed_loop.parameter_names if n.endswith(".tau"))
    if not widths or not switches:
        pytest.skip("The installed trano does not expose the smoothing of its controllers as parameters.")

    smooth = closed_loop.smooth_controllers(width_factor=5.0, switching_time=600.0)

    before = dict(zip(closed_loop.parameter_names, closed_loop.parameters, strict=True))
    after = dict(zip(smooth.parameter_names, smooth.parameters, strict=True))
    assert smooth is not closed_loop
    assert smooth.parameter_names == closed_loop.parameter_names
    for name in widths:
        assert after[name] == pytest.approx(5.0 * before[name])
    for name in switches:
        assert after[name] == 600.0
    untouched = set(closed_loop.parameter_names) - set(widths) - set(switches)
    np.testing.assert_array_equal([after[n] for n in untouched], [before[n] for n in untouched])  # NaN-safe
    # the sharp model is left as it was, and the gentler copy still runs its own closed loop
    np.testing.assert_array_equal(closed_loop.parameters, spec.model(CLOSED_LOOP_LIBRARY).parameters)
    state = spin_up(smooth, smooth, START, days=0.25)
    assert np.all(np.isfinite(state))


def test_smooth_controllers_leaves_a_model_without_controllers_alone(house: BuildingModel) -> None:
    assert house.smooth_controllers() is house


def test_a_source_that_uses_fast_smooth_buildings_without_holding_it_reads_it_from_its_package() -> None:
    import fastsmoothbuildings

    from mpctrano.model import used_libraries

    library = fastsmoothbuildings.LIBRARY
    uses = 'package house\n  annotation (uses(FastSmoothBuildings(version="1")));\nend house;'
    assert used_libraries(uses) == [library]
    assert used_libraries(uses, roots=[library]) == []
    assert used_libraries("package house\n  package FastSmoothBuildings\n  end FastSmoothBuildings;\nend house;") == []
    assert used_libraries('package house\n  annotation (uses(Buildings(version="12.0.0")));\nend house;') == []


def test_the_installed_rumoca_exports_the_building_models() -> None:
    assert mpctrano.model.rumoca_exports_building_models()


def test_a_rumoca_that_cannot_export_the_library_gets_the_install_command(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: object, **__: object) -> None:
        raise RuntimeError(
            "Target rendering error: unsupported-feature:residual_equations: Target 'casadi-ode' does not support "
            "feature 'residual_equations'"
        )

    monkeypatch.setattr(mpctrano.model.CasadiModel, "from_modelica", refuse)
    monkeypatch.setattr(mpctrano.model, "rumoca_exports_building_models", lambda: False)
    with pytest.raises(ModelError, match="cannot export FastSmoothBuildings models") as raised:
        BuildingModel.from_source("package house end house;", "house.building_mpc")
    assert mpctrano.model.RUMOCA_REQUIREMENT in str(raised.value)
    assert isinstance(raised.value.__cause__, RuntimeError)


def test_a_model_error_unrelated_to_rumoca_is_not_masked(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_: object, **__: object) -> None:
        raise RuntimeError("unresolved component")

    monkeypatch.setattr(mpctrano.model.CasadiModel, "from_modelica", fail)
    with pytest.raises(RuntimeError, match="unresolved component"):
        BuildingModel.from_source("package house end house;", "house.building_mpc")
