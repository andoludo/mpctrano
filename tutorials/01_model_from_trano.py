"""Tutorial 1 - From a Trano building description to a CasADi model.

Trano turns a YAML description of a building into Modelica models. With the ``mpc`` library it
generates a flat RC model (``building_mpc``) plus a JSON interface describing its states, inputs
and parameters. mpctrano loads both with rumoca into CasADi functions ready for optimisation.
"""

import numpy as np
import pandas as pd
from trano.mpc import RCModelType

from mpctrano import RCModel, WeatherForecaster
from tutorials._common import MODEL, OUTPUT

# --- 1. Generate the model with Trano (equivalent to `trano create-model house.yaml mpc`).
model = RCModel.from_yaml(MODEL, model_type=RCModelType.r3c2)
print(f"Model {model.name}: {model.n_zones} zones, {model.n_states} states, {model.n_parameters} parameters")
print("states      :", model.state_names)
print("controls    :", model.control_names)
print("disturbances:", model.disturbance_names)
print("weather file:", model.weather_path)

# --- 2. The interface tells where each disturbance comes from: this is what the forecaster uses.
for signal in model.interface.disturbances:
    print(f"  {signal.name:20s} <- {signal.source}")

# --- 3. The parameters are Trano's physics-based initial guesses (they will be calibrated in tutorial 2).
print(pd.Series(model.parameter_values).head(12).to_string())

# --- 4. The same building with another RC structure: the heat emitter dynamics of R4C3.
r4c3 = RCModel.from_yaml(MODEL, model_type=RCModelType.r4c3)
print("R4C3 states:", r4c3.state_names)

# --- 5. Open-loop simulation: two days with 1 kW of heating in every zone and the weather of mid-January.
forecaster = WeatherForecaster.from_model(model)
start = pd.Timestamp("2023-01-16")
disturbances = forecaster.realisation(start, 48, 3600.0)
controls = np.full((model.n_controls, 48), 1000.0)
states = model.simulate(model.initial_state, controls, disturbances.to_numpy().T, 3600.0)
trajectory = (
    pd.DataFrame(states.T, index=pd.date_range(start, periods=49, freq="h"), columns=model.state_names) - 273.15
)
print(trajectory.iloc[::12].round(2).to_string())

# --- 6. Write the Modelica package and the interface, e.g. to share them with another MPC runtime.
modelica, interface = model.write(OUTPUT, "house")
print(f"written {modelica} and {interface}")
section = model.source[model.source.index("model building_mpc") : model.source.index("end building_mpc;")]
print(section[:600], "...")
