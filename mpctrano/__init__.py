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
from mpctrano.flexibility import (
    ActivationRequest,
    ActivationResult,
    FlexibilityEnvelope,
    compare,
    flexibility_function,
    price_response,
    respond,
    shifted_energy,
    value_of_flexibility,
)
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
from mpctrano.meter import Meter
from mpctrano.scenarios import Mismatch, Scenario, build_plant, run_scenario, scenario_catalogue
from mpctrano.simulation import ClosedLoopResult, ModelPlant, run_closed_loop
from mpctrano.synthetic import ExcitationSettings, SyntheticData, generate_monitoring_data, perturb
from mpctrano.tariff import (
    CapacityTariff,
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
    "ActivationRequest",
    "ActivationResult",
    "CalibrationResult",
    "CalibrationSettings",
    "Calibrator",
    "CapacityTariff",
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
    "FitMetrics",
    "FlatTariff",
    "FlexibilityEnvelope",
    "ForecastNoise",
    "Forecaster",
    "KalmanFilter",
    "MPCSettings",
    "MPCSolution",
    "Meter",
    "Mismatch",
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
    "build_plant",
    "compare",
    "evaluate",
    "flexibility_function",
    "generate_monitoring_data",
    "perturb",
    "price_response",
    "respond",
    "run_closed_loop",
    "run_scenario",
    "scenario_catalogue",
    "shifted_energy",
    "simulate_fit",
    "value_of_flexibility",
]
