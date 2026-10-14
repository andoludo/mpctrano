"""mpctrano: calibration, state estimation and economic MPC of Trano RC building models.

Workflow::

    from mpctrano import Project, MonitoringData, scenario_catalogue

    project = Project.from_yaml("house.yaml")                 # Trano generates the RC model
    data = MonitoringData.from_csv("monitoring.csv")          # measurements of the real building
    project.calibrate(data)                                   # IPOPT least squares
    result = project.run(scenario_catalogue()["time_of_use"])  # state estimation + MPC
"""

from importlib.metadata import PackageNotFoundError, version

from mpctrano.calibration import CalibrationResult, CalibrationSettings, Calibrator, evaluate, simulate_fit
from mpctrano.comfort import ComfortSchedule, ComfortSpec
from mpctrano.data import DataReport, MonitoringData
from mpctrano.estimation import KalmanFilter, MovingHorizonEstimator, StateEstimator
from mpctrano.fast_forecast import FastWeatherForecaster
from mpctrano.fast_model import FastModel
from mpctrano.fast_mpc import FastModelPredictiveController, FastMPCSettings
from mpctrano.flexibility import compare, price_response, shifted_energy, value_of_flexibility
from mpctrano.forecast import (
    DataForecaster,
    Forecaster,
    ForecastNoise,
    OccupancySchedule,
    WeatherFile,
    WeatherForecaster,
)
from mpctrano.metrics import ControlMetrics, FitMetrics
from mpctrano.model import RCModel
from mpctrano.mpc import ModelPredictiveController, MPCSettings, MPCSolution
from mpctrano.prices import PriceSignal, RetailComponents
from mpctrano.project import Project
from mpctrano.scenarios import Scenario, run_scenario, scenario_catalogue
from mpctrano.simulation import ClosedLoopResult, ModelPlant, run_closed_loop
from mpctrano.synthetic import ExcitationSettings, SyntheticData, generate_monitoring_data, perturb
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
    "ExcitationSettings",
    "FastMPCSettings",
    "FastModel",
    "FastModelPredictiveController",
    "FastWeatherForecaster",
    "FitMetrics",
    "FlatTariff",
    "ForecastNoise",
    "Forecaster",
    "KalmanFilter",
    "MPCSettings",
    "MPCSolution",
    "ModelPlant",
    "ModelPredictiveController",
    "MonitoringData",
    "MovingHorizonEstimator",
    "OccupancySchedule",
    "PriceSignal",
    "Project",
    "RCModel",
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
    "run_closed_loop",
    "run_scenario",
    "scenario_catalogue",
    "shifted_energy",
    "simulate_fit",
    "value_of_flexibility",
]
