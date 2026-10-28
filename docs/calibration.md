# Calibration

A BuildingModel has hundreds of parameters, most of them settled by the YAML (areas, orientations, the
optical fits of the glazing) or not identifiable from a few air temperatures. `Calibrator` adjusts
**groups** of parameters instead, each by one decision variable:

| Group                   | Parameters                                                    | Mode, range        |
|-------------------------|---------------------------------------------------------------|--------------------|
| `envelope_conductance`  | conductances of the wall, floor and roof layers (`G[i]`), of the glazing gaps and frames | factor 0.25-4 |
| `envelope_capacity`     | capacities of the layers (`C[i]`) and of the panes            | factor 0.25-4      |
| `air_capacity`          | the heat capacity factor of the air of each zone (`mSenFac`, furniture) | factor 1-10 |
| `infiltration`          | the air changes per hour of each zone (`ACH`, 0 in Trano)     | value 0-3 1/h      |
| `boiler_efficiency`     | the nominal efficiency of the boilers (from the fuel meter)   | factor 0.7-1.1     |

`OPTIONAL_GROUPS` adds the solar transmittance of the windows and the capacity of the radiators,
which the air temperatures hardly identify on their own. A `ParameterGroup` is a set of regular
expressions on the parameter names, a mode (`scale`: the Trano values times a common factor; `value`:
one common value) and bounds; only the parameters the equations depend on are adjusted (the
conductances and capacities trano computed from the layers, not the layer data). The heat output of
the radiators and the performance maps of the heat pumps are settled when rumoca translates the model
and cannot be calibrated.

```python
from mpctrano import Calibrator, CalibrationSettings

calibrator = Calibrator(model, CalibrationSettings(temperature_std=0.1, meter_std=0.05))
calibrated, result = calibrator.fit(training, validation)
result.factors       # the value of each group
result.metrics()     # RMSE and MAE of the air temperatures, training and validation, before and after
```

The problem is solved with IPOPT by one of two methods (`CalibrationSettings.method`, `auto` by
default):

* **collocation**, for a model with control inputs: the Radau collocation of the model over the whole
  period, the parameters symbolic. The states of every step are decision variables bound by the
  dynamics, so that a long period and poor initial parameters stay well conditioned, and the
  unmeasured states (walls, water) are free at the start.
* **shooting**, for a closed-loop model (the controllers of the YAML inside, no control input): the
  groups are the only decision variables; the model is simulated with CVODES inside the objective from
  the state its own controllers reach before the data (`spin_up`), and IPOPT gets the gradient from the
  adjoint sensitivities of the integrator, with a quasi-Newton (L-BFGS) Hessian. The thermostats and
  hysteresis of such a model make its collocated dynamics non-smooth (IPOPT then crawls without
  converging), where a simulation follows them in a second.

The objective weighs the errors of the air temperatures (`temperature_std`), of the energy metered
over each step (`meter_std`) and the deviation of each group from the Trano value (`prior_std`,
`regularization`). The validation period is simulated from where the training ends.

On the boiler house of the tests, with the conductances x1.3, the capacities x0.8, the air capacity
x3, 0.5 air changes per hour and the boiler efficiency x0.9, four days recover x1.35, x0.80, x2.74,
0.47 and x0.904 in about 30 s; the air temperature error falls from 4.5 K to 0.06 K (0.07 K on two
validation days). With a single day, the conductances and the infiltration trade with each other:
both are losses to the outdoor air.

`evaluate(model, data, x0)` and `simulate_fit(model, data, x0)` give the open-loop fit of any model;
`Project.save()` writes the calibration to `<stem>.calibration.json` and `Project.load_calibrated()`
reads it back.

## Smoothing the controllers

The thermostats, hysteresis and limiters of a BuildingModel are steep smooth switches (trano >= 0.59
exposes their widths `dU`, `dY`, `dMax`, `dT`, `dtSet` and the switching time `tau` as parameters).
CVODES follows them, but each switch that moves by a sample under a parameter change makes the
objective jumpy at the 0.1 % scale, which is what forces the wide finite differences of the shooting
method. `BuildingModel.smooth_controllers(width_factor=5.0, switching_time=600.0)` returns a copy whose
transitions are five times wider and whose hysteresis switches over ten minutes; calibrate that copy
and apply the factors to the sharp model.

On the single-zone house of the tests (truth k = 0.06 W/(m.K), known infiltration, three days of data
with 0.1 K of noise, four cores):

| Controllers       | Smoothed truth vs sharp data | Shooting                  | Collocation                         |
|-------------------|------------------------------|---------------------------|-------------------------------------|
| sharp (trano)     | 0.09 K (the noise)           | 211 s, 17 it, k = 0.0594  | no convergence                      |
| x1, 600 s         | 0.23 K                       | 22 s, 8 it, k = 0.0531    | no convergence                      |
| **x5, 600 s**     | 0.14 K                       | 14 s, 7 it, k = 0.0594    | no convergence                      |
| x10, 600 s        | 0.27 K                       | 11 s, 7 it, k = 0.0448    | 72 s, 48 it, k = 0.0448             |
| x20, 600 s        | 0.42 K                       | 9 s, 5 it, k = 0.025      | 130 s, 37 it, k = 0.025 (the bound) |

The switching time is what smooths the objective (the sharp model and x1 differ only by it); the width
is what costs fidelity, and from x10 on it smears the thermostat band enough to bias the conductances
low. Collocation converges only on controllers too smooth to be faithful, so the shooting method on the
x5, 600 s copy is the setting to use.

## Parameters declared in the YAML

The YAML of the building can declare the parameters to calibrate itself, so that it stays the single
source of truth: a numeric field written as a mapping with its nominal value and its bounds, instead
of the bare number Trano expects.

```yaml
material:
  - id: ROCKWOOL:CAL
    thermal_conductivity: {value: 0.04, parameter: true, min: 0.025, max: 0.08}
constructions:
  - id: CAVITYWALL:CAL
    layers:
      - material: ROCKWOOL:CAL
        thickness: {value: 0.1, parameter: true, min: 0.04, max: 0.2}
spaces:
  - id: SPACE:001
    variant: infiltration
    parameters:
      ach: {value: 0.3, parameter: true, min: 0.05, max: 1.5}
```

`CalibrationSpec` reads the file, addresses the fields by path (the convention of `trano.runtime`:
`material[ROCKWOOL:CAL].thermal_conductivity`, `spaces[SPACE:001].parameters.ach`) and writes the
flat copy Trano reads. Which parameters of the CasADi model a field moves is measured, not assumed:
the model is regenerated with the field raised by 20 % and the exponent of every parameter that
moved is read (a conductivity moves the conductances of its layers with the exponent 1, a thickness
moves them with -1 and the node capacities with a small exponent). The calibration then adjusts the
field within its bounds and every dependent parameter with it (`YamlGroup`); the factors of fields
that move the same parameter compound (a conductivity and a thickness both move the conductance of
the insulation layer).

```python
from mpctrano import CalibrationSettings, CalibrationSpec, Calibrator, generate_closed_loop_data, spin_up

spec = CalibrationSpec.from_yaml("house.yaml")
model = spec.model("FastSmoothBuildings")                     # the closed loop: the controllers of the YAML, no control input
groups = spec.groups(model, "FastSmoothBuildings")            # one regeneration per declared field
calibrated, result = Calibrator(model, CalibrationSettings(groups=groups)).fit(data)
result.factors                                      # {"material[ROCKWOOL:CAL].thermal_conductivity": 0.058, ...}
```

The data of a closed-loop model holds no control: the disturbances, the measured air temperatures and
the meter readings are enough, as a monitoring system that logs the sensors and the meters but not the
valves records. `generate_closed_loop_data` records such data on a *true* building, for instance the
YAML with other values of its parameters (`spec.model("FastSmoothBuildings", {"...ach": 0.6})`), with sensor
and meter noise, which is how the tests check that the calibration recovers them.

Trano's own tools do not read the annotations (its validation rejects unknown keys); mpctrano writes
the flat copy before calling it. A `value / parameter / min / max` convention in Trano's schema would
make the annotated file valid everywhere.
