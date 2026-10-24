"""Shared fixtures: the heat pump house of the Trano test suite, a perturbed copy and its monitoring data.

Two buildings of Trano's FastSmoothBuildings tests serve the suite: a single-zone house heated by an air-to-water heat
pump (the electricity metered, the reference building of the tutorials) and the same house with a gas boiler (the fuel
metered), in ``tests/models`` next to the weather file both use.
"""

from pathlib import Path

import pytest

from mpctrano import BuildingModel, Project, SyntheticData, WeatherForecaster

MODELS = Path(__file__).parent / "models"
HEAT_PUMP_HOUSE = MODELS / "single_zone_air_water_heat_pump.yaml"
BOILER_HOUSE = MODELS / "single_zone_hydronic.yaml"
WEATHER = MODELS / "BEL_VLG_Uccle.064470_TMYx.2007-2021.mos"
START = "2023-01-16"


@pytest.fixture(scope="session")
def project() -> Project:
    return Project.from_yaml(HEAT_PUMP_HOUSE)


@pytest.fixture(scope="session")
def heat_pump(project: Project) -> BuildingModel:
    return project.model


@pytest.fixture(scope="session")
def forecaster(project: Project) -> WeatherForecaster:
    return project.forecaster


@pytest.fixture(scope="session")
def true_building(project: Project) -> BuildingModel:
    return project.true_building(seed=0)


@pytest.fixture(scope="session")
def synthetic(project: Project, true_building: BuildingModel) -> SyntheticData:
    return project.synthetic_data(days=3, start="2023-01-02", building=true_building)
