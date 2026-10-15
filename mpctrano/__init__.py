"""mpctrano: calibration, state estimation and economic MPC of the FastModel building models of Trano.

Workflow::

    from mpctrano import MonitoringData, Project, scenario_catalogue

    project = Project.from_yaml("house.yaml")                  # Trano generates the FastModelMPC model
    project.calibrate(MonitoringData.from_csv("monitoring.csv"))  # IPOPT least squares on the parameter groups
    scenario = scenario_catalogue()["time_of_use"]
    mpc = project.run(scenario)                                # state estimation + MPC
    reference = project.run_rule_based(scenario)               # the controllers of the YAML
"""

from importlib.metadata import PackageNotFoundError, version

from mpctrano.calibration import (
    DEFAULT_GROUPS,
    OPTIONAL_GROUPS,
    CalibrationResult,
    CalibrationSettings,
    Calibrator,
    ParameterGroup,
    evaluate,
    simulate_fit,
)
from mpctrano.comfort import ComfortSchedule, ComfortSpec
from mpctrano.data import DataReport, MonitoringData
from mpctrano.estimation import EstimatorSettings, MovingHorizonEstimator, PerfectEstimator, StateEstimator
from mpctrano.flexibility import compare, price_response, shifted_energy, value_of_flexibility
from mpctrano.forecast import (
    DataForecaster,
    Forecaster,
    ForecastNoise,
    OccupancySchedule,
    WeatherFile,
    WeatherForecaster,
)
from mpctrano.metrics import ControlMetrics, FitMetrics, Prices
from mpctrano.model import FastModel, ModelError
from mpctrano.mpc import ModelPredictiveController, MPCSettings, MPCSolution
from mpctrano.prices import PriceSignal, RetailComponents
from mpctrano.project import Project
from mpctrano.scenarios import Scenario, run_scenario, run_scenario_rule_based, scenario_catalogue
from mpctrano.simulation import ClosedLoopResult, ModelPlant, Plant, run_closed_loop, run_rule_based
from mpctrano.synthetic import SyntheticData, generate_monitoring_data, perturb, random_factors
from mpctrano.tariff import (
    DemandResponseEvent,
    DemandResponseTariff,
    DynamicTariff,
    FlatTariff,
    Tariff,
    TariffSignal,
    TimeOfUseTariff,
)

try:
    __version__ = version("mpctrano")
except PackageNotFoundError:  # pragma: no cover - source checkout without installation
    __version__ = "0.0.0"

__all__ = [
    "DEFAULT_GROUPS",
    "OPTIONAL_GROUPS",
    "CalibrationResult",
    "CalibrationSettings",
    "Calibrator",
    "ClosedLoopResult",
    "ComfortSchedule",
    "ComfortSpec",
    "ControlMetrics",
    "DataForecaster",
    "DataReport",
    "DemandResponseEvent",
    "DemandResponseTariff",
    "DynamicTariff",
    "EstimatorSettings",
    "FastModel",
    "FitMetrics",
    "FlatTariff",
    "ForecastNoise",
    "Forecaster",
    "MPCSettings",
    "MPCSolution",
    "ModelError",
    "ModelPlant",
    "ModelPredictiveController",
    "MonitoringData",
    "MovingHorizonEstimator",
    "OccupancySchedule",
    "ParameterGroup",
    "PerfectEstimator",
    "Plant",
    "PriceSignal",
    "Prices",
    "Project",
    "RetailComponents",
    "Scenario",
    "StateEstimator",
    "SyntheticData",
    "Tariff",
    "TariffSignal",
    "TimeOfUseTariff",
    "WeatherFile",
    "WeatherForecaster",
    "__version__",
    "compare",
    "evaluate",
    "generate_monitoring_data",
    "perturb",
    "price_response",
    "random_factors",
    "run_closed_loop",
    "run_rule_based",
    "run_scenario",
    "run_scenario_rule_based",
    "scenario_catalogue",
    "shifted_energy",
    "simulate_fit",
    "value_of_flexibility",
]
