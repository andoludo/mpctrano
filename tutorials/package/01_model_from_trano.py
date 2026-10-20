"""Tutorial 1 - From a Trano building description to a CasADi model.

Trano turns a YAML description of a building into Modelica models. With its ``FastModelMPC`` library it generates
a model close to the Buildings library (layered walls, windows, the hydronic circuit with its heat pump, buffer,
mixing valve and radiator) whose controls are the inputs of the systems. rumoca translates it into a CasADi ODE
``xdot = f(t, x, u, p)``; mpctrano reads the roles of the variables from the Modelica model itself.
"""

import numpy as np
import pandas as pd

from mpctrano import FastModel, WeatherForecaster
from tutorials._common import MODEL, OUTPUT

# --- 1. Generate the model with Trano (equivalent to `trano create-model house.yaml FastModelMPC`).
model = FastModel.from_yaml(MODEL)
print(f"Model {model.name}: {model.n_zones} zone(s), {model.n_states} states, {model.n_parameters} parameters")
print("controls    :", model.control_names)  # valve, heat pump speed, mixing valve: between 0 and 1
print("disturbances:", model.disturbance_names)  # the weather channels and the internal gains per floor area
print("comfort     :", model.comfort_states)
print("meters      :", model.meters)  # the electricity of the heat pump [kWh]
print("weather file:", model.weather_path)

# --- 2. The states: the air, the layers of the walls, the panes of the windows, the radiator and buffer water.
print(pd.Series({state.name: state.unit for state in model.model.description.states}).to_string())

# --- 3. The parameters are Trano's values from the YAML (some of them are calibrated in tutorial 2).
print(pd.Series(model.parameter_values).filter(regex=r"^(space_001|externalwall_0\.G|boiler_001)").to_string())

# --- 4. Open-loop simulation: one day of the weather of mid-January with the valves open and the heat pump at
#        half speed, then with everything off.
forecaster = WeatherForecaster.from_model(model)
start = pd.Timestamp("2023-01-16")
disturbances = forecaster.realisation(start, 24, 3600.0).to_numpy().T
seconds = float((start - pd.Timestamp("2023-01-01")).total_seconds())  # the model time: seconds of the year
index = pd.date_range(start, periods=25, freq="h")
for name, controls in {"heating": np.tile([[1.0], [0.5], [1.0]], 24), "off": np.zeros((3, 24))}.items():
    states = model.simulate(model.initial_state, controls, disturbances, 3600.0, start_time=seconds)
    frame = pd.DataFrame(states.T, index=index, columns=model.state_names)
    print(f"\n{name}: air [degC], buffer [degC], electricity [kWh]")
    print(
        pd.DataFrame(
            {
                "air": frame["space_001.TAir"] - 273.15,
                "buffer": frame["boiler_001.T"] - 273.15,
                "electricity": frame["boiler_001.EEle"],
            }
        )
        .iloc[::6]
        .round(2)
        .to_string()
    )

# --- 5. Write the Modelica package, e.g. to simulate the runnable model with OpenModelica.
path = model.write(OUTPUT, "house")
print(f"written {path}")
section = model.source[model.source.index("model building_mpc") : model.source.index("end building_mpc;")]
print(section[:600], "...")
