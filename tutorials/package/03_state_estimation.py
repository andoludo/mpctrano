"""Tutorial 3 - State estimation.

The MPC needs the full state of the FastModel (the air, the layers of the walls, the radiator and buffer water) but
only the zone air temperatures are measured. The moving horizon estimator finds the trajectory of the model closest
to the measurements over a sliding window of the past; here it starts 3 K off on every temperature and converges on
the states of the synthetic building, where an uncorrected simulation keeps the initial error for days.
"""

import numpy as np
import pandas as pd

from mpctrano import EstimatorSettings, ModelPlant, MovingHorizonEstimator, Project
from tutorials._common import MODEL, data_days, save_figure

project = Project.from_yaml(MODEL)
model = project.model
building = project.true_building(seed=4)
temperatures = [i for i, state in enumerate(model.model.description.states) if state.unit == "K"]
true_start = model.initial_state.copy()
true_start[temperatures] -= 3.0  # the building is colder than the estimator believes

start = pd.Timestamp("2023-01-16")
steps = int(data_days(2) * 24)
plant = ModelPlant(building, 3600.0, start, x0=true_start, measurement_std=0.05)
open_loop = ModelPlant(model, 3600.0, start, measurement_std=0.0)  # the model alone, never corrected
estimator = MovingHorizonEstimator(model, 3600.0, start, settings=EstimatorSettings(window=12))
disturbances = project.forecaster.realisation(start, steps, 3600.0).to_numpy()
rows = []
for k in range(steps):
    estimate = estimator.correct(plant.measure())  # use the state here (MPC)
    errors = np.abs(estimate[temperatures] - plant.state[temperatures])
    open_errors = np.abs(open_loop.state[temperatures] - plant.state[temperatures])
    rows.append({"time": plant.time, "estimation": errors.mean(), "open loop": open_errors.mean()})
    control = np.array([1.0, 0.5, 1.0]) if (k // 4) % 2 == 0 else np.zeros(3)  # heating on and off every 4 h
    plant.step(control, disturbances[k])
    estimator.predict(control, disturbances[k])  # then propagate with the applied inputs
    open_loop.step(control, disturbances[k])

errors = pd.DataFrame(rows).set_index("time")
print("mean absolute error over the temperature states [K]:")
print(errors.iloc[:: max(1, steps // 8)].round(3).to_string())
print("estimations:", pd.Series(estimator.statuses).value_counts().to_dict())

try:
    import plotly.graph_objects as go

    figure = go.Figure([go.Scatter(x=errors.index, y=errors[column], name=column) for column in errors.columns])
    figure.update_layout(title="Mean error of the temperature states", yaxis_title="K")
    save_figure(figure, "03_state_estimation")
except ImportError:
    pass
