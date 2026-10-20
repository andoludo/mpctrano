# Calibration

A FastModel has hundreds of parameters, most of them settled by the YAML (areas, orientations, the
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

The problem is solved with IPOPT on the Radau collocation of the model over the whole period, the
parameters symbolic: the states of every step are decision variables bound by the dynamics, so that a
long period and poor initial parameters stay well conditioned, and the unmeasured states (walls,
water) are free at the start. The objective weighs the errors of the air temperatures
(`temperature_std`), of the energy metered over each step (`meter_std`) and the deviation of each
group from the Trano value (`prior_std`, `regularization`). The validation period is simulated from
where the training ends.

On the boiler house of the tests, with the conductances x1.3, the capacities x0.8, the air capacity
x3, 0.5 air changes per hour and the boiler efficiency x0.9, four days recover x1.35, x0.80, x2.74,
0.47 and x0.904 in about 30 s; the air temperature error falls from 4.5 K to 0.06 K (0.07 K on two
validation days). With a single day, the conductances and the infiltration trade with each other:
both are losses to the outdoor air.

`evaluate(model, data, x0)` and `simulate_fit(model, data, x0)` give the open-loop fit of any model;
`Project.save()` writes the calibration to `<stem>.calibration.json` and `Project.load_calibrated()`
reads it back.
