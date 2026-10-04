# Workflow

mpctrano sits downstream of Trano. The building description (`house.yaml`) stays the single source
of truth: Trano derives the RC structure and the physics-based initial parameters, mpctrano
calibrates them, estimates the states and controls the heating.

## 1. Model

```python
from mpctrano import RCModel
from trano.mpc import RCModelType

model = RCModel.from_yaml("house.yaml", model_type=RCModelType.r3c2)   # calls Trano
model = RCModel.from_files("house.mo", "house.mpc.json")               # or the files of `trano create-model house.yaml mpc`
```

`RCModel` keeps the Modelica source, the interface and the current parameter vector. It exposes:

* `continuous` - CasADi function `xdot = f(x, u, d, p)` with the controls `u` (`<zone>_QHea`) and
  the disturbances `d` (`TOut`, `HSol_*`, `<zone>_QInt`) separated;
* `discrete(dt)` - RK4 zero-order-hold step `x_next = F(x, u, d, p)` with automatically chosen
  stable sub-steps; `p` stays symbolic so the same function serves the calibration;
* `affine(dt)` - the exact matrices `A, Bu, Bd, c` of the discrete step at the current parameters
  (the RC models are linear), used by the Kalman filter;
* `simulate(x0, controls, disturbances, dt)` - open-loop simulation (`mapaccum`);
* `with_parameters({...})` / `calibrated_interface()` / `write(directory)` - parameter management.

The interface order is the CasADi vector order, which mpctrano checks against the rumoca export.

## 2. Disturbances

`WeatherForecaster.from_model(model)` reads the TMY3 `.mos` weather file referenced by the Trano
model, computes the outdoor temperature and the irradiance on every orientation of the envelope
(direct + isotropic diffuse + ground reflected, as the Buildings library blocks used by Trano's
runnable model) and the internal gains of each zone from the occupancy schedule declared in the
interface (`SimpleOccupancy`: 7h-19h, (35+70)/24 W/m²). `ForecastNoise` adds lead-time dependent
errors to study imperfect forecasts; `DataForecaster` replays recorded disturbances.

## 3. Calibration, estimation, control

See [calibration](calibration.md), [estimation](estimation.md) and [mpc](mpc.md).

## 4. Closed loop and scenarios

`run_closed_loop(controller, estimator, plant, forecaster, start, steps)` runs the cycle
*measure → correct → forecast → solve → apply → predict* and returns a `ClosedLoopResult` with
the trajectories and the `ControlMetrics` (energy, cost, comfort violation in K·h, peak power,
share of energy bought below the median price). `Scenario` bundles a tariff, a comfort
specification, the MPC settings, the forecast noise and the estimator type; `run_scenario`
and `Project.run` execute it; `compare` puts several runs side by side.
