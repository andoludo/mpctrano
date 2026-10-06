"""Tutorial 3 - State estimation.

The MPC needs the full state of the RC model but only the zone air temperatures are measured.
The envelope (or mass, or emitter) temperatures are estimated with a Kalman filter (the model is
linear) or a moving horizon estimator (IPOPT, with bounds). Both start from a wrong guess and
converge on the true envelope temperatures of the synthetic building.
"""

import numpy as np
import pandas as pd

from mpctrano import KalmanFilter, MovingHorizonEstimator, Project
from tutorials._common import MODEL, days, save_figure

project = Project.from_yaml(MODEL)
building = project.true_building()
synthetic = project.synthetic_data(days=days(4), start="2023-01-02", building=building)
frame = synthetic.data.frame
steps = len(frame) - 1

wrong_start = building.initial_state + np.array([0, 4, 0, -4, 0, 3])  # 4 K off on the envelope nodes
estimators = {
    "kalman": KalmanFilter(building, 3600.0, x0=wrong_start, process_std=0.05, measurement_std=0.1),
    "mhe": MovingHorizonEstimator(building, 3600.0, x0=wrong_start, window=24),
}
estimates = {name: [] for name in estimators}
for k in range(steps):
    measurement = frame[building.comfort_states].iloc[k].to_numpy()
    control = frame[building.control_names].iloc[k].to_numpy()
    disturbance = frame[building.disturbance_names].iloc[k].to_numpy()
    for name, estimator in estimators.items():
        estimates[name].append(estimator.correct(measurement))  # use the state here (MPC)
        estimator.predict(control, disturbance)  # then propagate with the applied inputs

truth = synthetic.states.iloc[:steps]
comparison = pd.DataFrame({"true": truth["space_001_Te"]})
for name, values in estimates.items():
    comparison[name] = np.array(values)[:, building.state_names.index("space_001_Te")]
errors = (comparison[["kalman", "mhe"]].sub(comparison["true"], axis=0)).abs()
print("absolute error on space_001_Te [K]:")
print(pd.DataFrame({"first 6 h": errors.iloc[:6].mean(), "last 24 h": errors.iloc[-24:].mean()}).round(3).to_string())

try:
    import plotly.graph_objects as go

    figure = go.Figure()
    for column in comparison.columns:
        figure.add_trace(go.Scatter(x=comparison.index, y=comparison[column] - 273.15, name=column))
    figure.update_layout(title="Envelope temperature of space_001: truth vs estimates", yaxis_title="degC")
    save_figure(figure, "03_state_estimation")
except ImportError:
    pass
