"""Tutorial 4 - Model predictive control for comfort.

The controller plans the heating power of each zone over 24 hours so that the zone temperatures
stay inside the comfort band at the lowest energy cost. It is first used open loop (one plan from
a cold building), then in closed loop against the "real" building with state estimation.
"""

import pandas as pd

from mpctrano import (
    ComfortSchedule,
    ComfortSpec,
    FlatTariff,
    KalmanFilter,
    ModelPlant,
    ModelPredictiveController,
    MPCSettings,
    Project,
    WeatherForecaster,
    run_closed_loop,
)
from mpctrano.plot import plot_closed_loop
from tutorials._common import MODEL, days, save_figure

project = Project.from_yaml(MODEL)
building = project.true_building()
# A real project would calibrate here (tutorial 2); the Trano model is used directly to keep this short.
model = project.model

comfort = ComfortSpec(
    default=ComfortSchedule(occupied_lower=21, occupied_upper=24, unoccupied_lower=17, unoccupied_upper=27)
)
settings = MPCSettings(horizon=24, time_step=3600.0, comfort_penalty=5.0)
controller = ModelPredictiveController(model, settings, comfort, FlatTariff(price=0.25))
forecaster = WeatherForecaster.from_model(model)

# --- 1. One optimal plan from a cold building (18 degC everywhere) on a Monday.
start = pd.Timestamp("2023-01-16")
cold = model.initial_state * 0 + 273.15 + 18
plan = controller.solve(cold, forecaster.forecast(start, 24, 3600.0))
print(f"IPOPT {plan.status}: cost breakdown {plan.costs}")
table = plan.controls.copy() / 1000
table["space_001_Ti [degC]"] = plan.comfort_temperatures["space_001_Ti"].to_numpy() - 273.15
table["lower [degC]"] = plan.lower[0] - 273.15
print(table.round(2).to_string())

# --- 2. Closed loop: every hour, estimate the state, re-plan, apply the first hour of the plan.
plant = ModelPlant(building, 3600.0, measurement_std=0.05)
estimator = KalmanFilter(model, 3600.0, x0=plant.state)
result = run_closed_loop(controller, estimator, plant, forecaster, start, steps=int(days(2) * 24), name="comfort")
print(result.summary().round(3).to_string())
save_figure(plot_closed_loop(result), "04_closed_loop_comfort")
