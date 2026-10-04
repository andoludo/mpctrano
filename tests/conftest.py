"""Shared fixtures: the three-zone test building of Trano and synthetic data of a perturbed copy."""

from pathlib import Path

import pytest
from trano.mpc import RCModelType

from mpctrano import Project, RCModel, SyntheticData, WeatherForecaster

MODELS = Path(__file__).parent / "models"
THREE_ZONES = MODELS / "three_zones.yaml"
WEATHER = MODELS / "BEL_VLG_Uccle.064470_TMYx.2007-2021.mos"
START = "2023-01-16"


@pytest.fixture(scope="session")
def model() -> RCModel:
    return RCModel.from_yaml(THREE_ZONES)


@pytest.fixture(scope="session")
def models() -> dict[RCModelType, RCModel]:
    return {model_type: RCModel.from_yaml(THREE_ZONES, model_type=model_type) for model_type in RCModelType}


@pytest.fixture(scope="session")
def project(model: RCModel) -> Project:
    return Project.from_model(model)


@pytest.fixture(scope="session")
def true_building(project: Project) -> RCModel:
    return project.true_building(seed=0)


@pytest.fixture(scope="session")
def forecaster(model: RCModel) -> WeatherForecaster:
    return WeatherForecaster.from_model(model)


@pytest.fixture(scope="session")
def synthetic(project: Project, true_building: RCModel) -> SyntheticData:
    return project.synthetic_data(days=6, start="2023-01-02", building=true_building)
