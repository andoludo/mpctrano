"""Tutorial 4 - Model predictive control for comfort, against the controllers of the YAML.

The controller plans the valve, the heat pump speed and the mixing valve over 24 hours so that the air temperature
stays inside the comfort band at the lowest cost of the electricity the heat pump draws (read from its meter, so the
COP of the heat pump is part of the plan). It is first used open loop (one plan), then in closed loop against the
"real" building with state estimation, and compared with the controllers the YAML describes (the closed-loop
FastModel of the same building: a thermostat on the valve, the heat pump holding its supply temperature).
"""

import pandas as pd

from mpctrano import (
    ComfortSchedule,
    ComfortSpec,
    FlatTariff,
    ModelPlant,
    ModelPredictiveController,
    MovingHorizonEstimator,
    MPCSettings,
    Project,
    WeatherForecaster,
    compare,
    run_closed_loop,
    run_rule_based,
)
from mpctrano.plot import plot_closed_loop
from tutorials._common import MODEL, days, horizon, save_figure

project = Project.from_yaml(MODEL)
model = project.model  # a real project would calibrate it first (tutorial 2)
building = project.true_building(seed=4)

# The band of the thermostat of the YAML: 21 degC from 7h to 22h, 17 degC otherwise.
comfort = ComfortSpec(
    default=ComfortSchedule(occupied_lower=21, occupied_upper=24, unoccupied_lower=17, unoccupied_upper=27)
)
tariff = FlatTariff(price=0.30)
settings = MPCSettings(horizon=horizon(24))
controller = ModelPredictiveController(model, settings, comfort, tariff)
forecaster = project.forecaster

# --- 1. One optimal plan from the initial state on a Monday.
start = pd.Timestamp("2023-01-16")
plan = controller.solve(model.initial_state, forecaster.forecast(start, settings.horizon, 3600.0))
print(f"IPOPT {plan.status} in {plan.solve_time:.1f} s: cost breakdown {plan.costs}")
table = plan.controls.copy()
table["air [degC]"] = plan.comfort_temperatures["space_001.TAir"].to_numpy() - 273.15
table["lower [degC]"] = plan.lower[0] - 273.15
table["electricity [kWh]"] = controller.energy["electricity_bought"].to_numpy()  # type: ignore[index]
print(table.round(2).to_string())

# --- 2. Closed loop: every hour, estimate the state, re-plan, apply the first hour of the plan.
steps = int(days(2) * 24)
plant = ModelPlant(building, 3600.0, start, measurement_std=0.05)
estimator = MovingHorizonEstimator(model, 3600.0, start)
mpc = run_closed_loop(controller, estimator, plant, forecaster, steps, name="mpc")
save_figure(plot_closed_loop(mpc), "04_closed_loop_comfort")

# --- 3. The controllers of the YAML on the same building, the same weather and the same tariff.
closed_loop = project.true_closed_loop(seed=4)
rule_forecaster = WeatherForecaster.from_model(closed_loop, weather=forecaster.weather)
reference = run_rule_based(closed_loop, rule_forecaster, start, steps, 3600.0, comfort, tariff)
columns = ["energy_kwh", "total_cost", "cost_saving_pct", "comfort_violation_kh", "peak_power_kw"]
print(compare({"rule based": reference, "mpc": mpc}, baseline="rule based")[columns].round(3).to_string())
