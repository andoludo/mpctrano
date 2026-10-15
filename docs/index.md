# Workflow

mpctrano sits downstream of Trano. The building description (`house.yaml`) stays the single source
of truth: Trano renders it into Modelica models with its `FastModel` library, mpctrano calibrates
them on monitoring data, estimates their state and controls the systems of the building.

```
 house.yaml ──trano (FastModelMPC)──▶ building_mpc ──rumoca──▶ CasADi ODE  xdot = f(t, x, u, p)
            ──trano (FastModel)─────▶ the same building with the controllers of the YAML (rule-based reference)
```

## 1. Model

```python
from mpctrano import FastModel

model = FastModel.from_yaml("house.yaml")                       # calls Trano (FastModelMPC library)
model = FastModel.from_file("house.mo")                         # or the package of `trano create-model house.yaml FastModelMPC`
closed_loop = FastModel.from_yaml("house.yaml", library="FastModel")  # the controllers of the YAML
```

The `FastModelMPC` model is close to the Buildings library: layered walls and floors, windows with
the angular optics of their glazing, convection and long-wave exchanges in the room, infiltration,
the hydronic circuits of the YAML (radiators, valves, mixing valves, pumps, boilers, air-to-water
heat pumps and their buffer), ventilation units with heat recovery and PV arrays. The controls of
the systems are inputs. `FastModel` reads everything from the Modelica model:

* `control_names` - the inputs with a `min` and a `max`: valve positions, mixing valves, boiler part
  load ratio or heat pump speed (between 0 and 1), the heat of ideal emitters [W];
* `disturbance_names` - the weather channels (`weather_inputs.TDryBul`, `.TBlaSky`, `.winSpe`,
  `.winDir`, `.HDirNor`, `.HDifHor`, `.HGloHor`) and the internal gains per floor area of each
  occupancy model (`<occupancy>.gai.QRad`, `.QCon`);
* `comfort_states` - the air temperature of each zone (`<zone>.TAir`);
* `meters` - the energy the systems drew or produced since the start, states in kWh: `EFue` (fuel),
  `EEle` (electricity of a heat pump), `EFan` (fans), `EPro` (PV);
* `simulate(x0, controls, disturbances, dt, start_time)` - CVODES simulation, controls held over each
  step; the model depends on the time of the year (the solar position is computed inside it);
* `with_parameters({...})`, `select_parameters(patterns)`, `write(directory)`.

The model is nonlinear and stiff (water nodes): the optimisation problems collocate it on Radau
points (`trano.optimization`), which keeps them stable whatever the step. rumoca does not report the
unit of array states (wall layers, radiator water): `FastModel` restores the kelvin on them so that
the collocation scales and bounds them as temperatures.

The CasADi export of these models needs rumoca 0.10.3 (rumoca main up to commit `c74cddf`); `uv sync`
builds it from source, see the installation section of the README.

## 2. Disturbances

`WeatherForecaster.from_model(model)` reads the TMY3 `.mos` weather file of the runnable model (a
file of the same name next to the YAML) and the occupancy schedules of the YAML: the weather channels
at the middle of each step, the black-body sky temperature computed as the Buildings weather reader
does (from the dew point and the opaque sky cover), and the radiant and convective gains of each
occupancy model. `ForecastNoise` adds lead-time dependent errors to the temperatures, the irradiance
and the gains; `DataForecaster` replays recorded disturbances.

## 3. Calibration, estimation, control

See [monitoring data](data.md), [calibration](calibration.md), [estimation](estimation.md) and
[MPC](mpc.md).

## 4. Closed loop, scenarios and the rule-based reference

`run_closed_loop(controller, estimator, plant, forecaster, steps)` runs the cycle
*measure → correct → forecast → solve → apply → predict* against a `ModelPlant` (a FastModel playing
the real building, with sensor noise) and returns a `ClosedLoopResult` with the trajectories and the
`ControlMetrics`, read from the meters of the plant: fuel and electricity bought and sold, cost at
the price of each carrier, capacity charge, comfort violation in K·h, peak electric power, share of
the energy bought below the median price.

`run_rule_based(closed_loop, forecaster, start, steps, dt, comfort, tariff)` simulates the closed-loop
`FastModel` of the same YAML, with the controllers it describes (thermostats on the valves, the boiler
or heat pump holding its supply temperature, the mixing valve its set point), costed and assessed the
same way: the reference the MPC is compared with.

`Scenario` bundles a tariff, a comfort specification, the MPC settings, the forecast noise and the
estimator; `run_scenario`, `run_scenario_rule_based` and `Project.run` / `Project.run_rule_based`
execute it; `compare` puts several runs side by side.
