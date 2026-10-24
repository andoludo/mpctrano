"""What mpctrano needs from the models Trano generates, checked against the installed Trano.

Trano does not install rumoca, so its own tests of the CasADi export of BuildingModel are skipped: these tests take
their place. They fail when a change of Trano breaks the translation by rumoca, the names mpctrano relies on, or
the flat model (``include_container=False``) that :meth:`BuildingModel.from_network` generates.

The buildings are those of ``tests/models``: the two single-zone houses of mpctrano and two test buildings copied
from Trano (BESTEST 600FF, free floating, and a six-zone house with an air-to-water heat pump).
"""

from __future__ import annotations

import warnings
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from trano.data_models.conversion import convert_network
from trano.elements.library.library import Library
from trano_optimization import CasadiModel, Roles

from mpctrano import BuildingModel, ModelError, WeatherForecaster
from mpctrano.calibration import DEFAULT_GROUPS
from mpctrano.model import CLOSED_LOOP_LIBRARY, MPC_LIBRARY, used_libraries, without_containers
from tests.conftest import BOILER_HOUSE, HEAT_PUMP_HOUSE, MODELS

BESTEST_600FF = MODELS / "case_600FF.yaml"
SIX_ZONE_HOUSE = MODELS / "house_infiltration_air_water_heat_pump.yaml"
LIBRARIES = [MPC_LIBRARY, CLOSED_LOOP_LIBRARY]
WEATHER = {
    f"weather_inputs.{name}" for name in ("TDryBul", "TBlaSky", "winSpe", "winDir", "HDirNor", "HDifHor", "HGloHor")
}
BUILDINGS = [
    pytest.param(HEAT_PUMP_HOUSE, id="heat_pump_house"),
    pytest.param(BOILER_HOUSE, id="boiler_house"),
    pytest.param(BESTEST_600FF, id="600FF"),
    pytest.param(SIX_ZONE_HOUSE, id="six_zone_house", marks=pytest.mark.slow),
]


@cache
def source(path: Path, library: str, include_container: bool) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        network = convert_network(path.stem, path, library=Library.from_configuration(library))
        return network.model(include_container=include_container)


@cache
def translated(path: Path, library: str, include_container: bool) -> CasadiModel:
    text = source(path, library, include_container)
    return CasadiModel.from_modelica(text, f"{path.stem}.building_mpc", roots=used_libraries(text))


@pytest.mark.parametrize("library", LIBRARIES)
@pytest.mark.parametrize("path", BUILDINGS)
def test_rumoca_translates_the_model(path: Path, library: str) -> None:
    description = translated(path, library, include_container=False).description
    roles = Roles.infer(description)
    roles.check(description)
    assert set(description.input_names) >= WEATHER
    assert all(np.isfinite(description.initial_state))
    assert roles.comfort
    assert all(name.endswith("_y.y") for name in roles.controls)
    if library == CLOSED_LOOP_LIBRARY:
        assert not roles.controls
    for name in roles.controls:
        variable = description.input(name)
        assert variable.minimum is not None
        assert variable.maximum is not None


@pytest.mark.parametrize("library", LIBRARIES)
@pytest.mark.parametrize("path", BUILDINGS)
def test_the_containers_leave_the_model_unchanged(path: Path, library: str) -> None:
    """The containerised model, once its names are stripped, is the flat model: same names, initial state,
    parameters and right-hand side."""
    flat = translated(path, library, include_container=False)
    grouped = without_containers(translated(path, library, include_container=True))
    for kind in ("state_names", "input_names", "parameter_names"):
        assert sorted(getattr(grouped.description, kind)) == sorted(getattr(flat.description, kind))
    states = [grouped.description.state_names.index(name) for name in flat.description.state_names]
    inputs = [grouped.description.input_names.index(name) for name in flat.description.input_names]
    parameters = [grouped.description.parameter_names.index(name) for name in flat.description.parameter_names]
    np.testing.assert_allclose(grouped.description.initial_state[states], flat.description.initial_state)
    np.testing.assert_allclose(
        grouped.description.parameter_values[parameters], flat.description.parameter_values, equal_nan=True
    )
    generator = np.random.default_rng(0)
    for _ in range(3):
        x = flat.description.initial_state + generator.normal(0.0, 1.0, len(states))
        u = np.array(
            [
                generator.uniform(260.0, 290.0) if name.endswith(("TDryBul", "TBlaSky")) else generator.uniform(0, 1)
                for name in flat.description.input_names
            ]
        )
        time = generator.uniform(0.0, 3.15e7)
        x_grouped, u_grouped = np.empty_like(x), np.empty_like(u)
        x_grouped[states], u_grouped[inputs] = x, u
        p_grouped = np.empty(len(parameters))
        p_grouped[parameters] = flat.description.parameter_values
        expected = np.asarray(flat.rhs(time, x, u, flat.description.parameter_values)).ravel()
        actual = np.asarray(grouped.rhs(time, x_grouped, u_grouped, p_grouped)).ravel()[states]
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)


def test_the_heat_pump_house_has_the_names_mpctrano_relies_on(heat_pump: BuildingModel) -> None:
    """The names of the monitoring data, the comfort specifications, the calibration and the tutorials."""
    assert heat_pump.zone_names == ["space_001"]
    assert heat_pump.comfort_states == ["space_001.TAir"]
    assert heat_pump.meters == {"boiler_001.EEle": "electricity"}
    assert sorted(heat_pump.control_names) == ["boiler_001_y.y", "three_way_valve_001_y.y", "valve_001_y.y"]
    assert set(heat_pump.disturbance_names) == WEATHER | {"occupancy_1_gai.QCon", "occupancy_1_gai.QRad"}
    assert heat_pump.weather_path is not None
    assert {"occupancy_1"} == set(heat_pump.occupancy)
    for group in DEFAULT_GROUPS:
        if group.name != "boiler_efficiency":  # a heat pump has no boiler efficiency
            assert [name for name in heat_pump.parameter_names if group.matches(name)], group.name


def test_the_forecaster_gives_the_gains_of_the_occupancy(heat_pump: BuildingModel) -> None:
    forecaster = WeatherForecaster.from_model(heat_pump)
    frame = forecaster.realisation(pd.Timestamp("2023-01-16"), 24, 3600.0)
    assert frame.columns.tolist() == heat_pump.disturbance_names
    assert frame["occupancy_1_gai.QCon"].max() > 0.0


def test_a_containerised_model_read_from_a_file_has_the_names_of_the_flat_model(
    heat_pump: BuildingModel, tmp_path: Path
) -> None:
    """``trano create-model`` writes the containerised model: read from its file, it is the model of the YAML."""
    path = tmp_path / f"{HEAT_PUMP_HOUSE.stem}.mo"
    path.write_text(source(HEAT_PUMP_HOUSE, MPC_LIBRARY, include_container=True))
    from_file = BuildingModel.from_file(path)
    assert sorted(from_file.state_names) == sorted(heat_pump.state_names)
    assert from_file.roles.comfort == heat_pump.roles.comfort
    assert from_file.meters == heat_pump.meters
    assert sorted(from_file.control_names) == sorted(heat_pump.control_names)


def test_names_that_collide_without_their_containers_are_an_error() -> None:
    model = translated(BESTEST_600FF, MPC_LIBRARY, include_container=False)
    state = model.description.states[0]
    states = (
        state.model_copy(update={"name": f"envelope1.{state.name}"}),
        state.model_copy(update={"name": f"emission1.{state.name}"}),
        *model.description.states[1:],
    )
    clashing = CasadiModel(rhs=model.rhs, description=model.description.model_copy(update={"states": states}))
    with pytest.raises(ModelError, match="same names"):
        without_containers(clashing)


def test_a_flat_model_is_left_as_it_is() -> None:
    model = translated(BOILER_HOUSE, MPC_LIBRARY, include_container=False)
    assert without_containers(model) is model


@pytest.mark.parametrize("library", LIBRARIES)
def test_a_network_that_embeds_the_library_gives_the_same_model(library: str) -> None:
    """``trano create-model --embed-libraries``: the package holds a copy of FastSmoothBuildings, and the model is the
    one of the package that uses the installed library (trano's default, as Buildings and IDEAS)."""
    library_ = Library.from_configuration(library).model_copy(update={"embed_libraries": True})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        network = convert_network(HEAT_PUMP_HOUSE.stem, HEAT_PUMP_HOUSE, library=library_)
    assert not network.libraries

    embedded = BuildingModel.from_network(network).model.description
    beside = without_containers(translated(HEAT_PUMP_HOUSE, library, include_container=False)).description

    assert beside.state_names == embedded.state_names
    assert beside.input_names == embedded.input_names
    np.testing.assert_array_equal(beside.initial_state, embedded.initial_state)
