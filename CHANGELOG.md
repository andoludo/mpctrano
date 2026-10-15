# Changelog

All notable changes to mpctrano are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the tags created by the
release pipeline.

## [Unreleased]

### Added

- Notebook tooling: `notebook` extra (`plotly`, `ipywidgets`), jupytext pairing of the tutorial
  notebooks with percent scripts (`tutorials/NN_*.py` <-> `.ipynb`), `nbmake` execution of the
  notebooks in CI (`make notebooks`), a `pre-commit` configuration and this changelog.
- `tutorials/00_setup_check` notebook: checks the installation (trano, rumoca, CasADi, IPOPT) and
  the generation of a CasADi model from a Trano YAML.
- `tutorials/PLAN.md`: the plan of the interactive MPC and energy-flexibility tutorials;
  `tutorials/references.md`: the literature and regulatory sources of the tutorials, with their
  verification status; `tutorials/README.md`.
- `mpctrano.prices`: `PriceSignal` and `RetailComponents` (wholesale day-ahead series, retail
  stack, 15-minute and hourly resampling, price spikes, negative hours, relocation in time,
  cached samples) and the parsers of the Energy-Charts and ENTSO-E responses;
  `tutorials/scripts/fetch_prices.py` downloads the cached samples.
- `FastModel`: the building models of Trano's `FastModelMPC` library (zones close to Buildings,
  hydronic circuits, boilers, heat pumps, ventilation, PV) as CasADi functions; the roles of the
  variables (controls, disturbances, comfort temperatures, energy meters) are read from the Modelica
  model, the simulation runs with CVODES, and the unit of the temperature states rumoca drops (wall
  layers, radiator water) is restored. `FastModel.from_yaml(..., library="FastModel")` builds the
  closed-loop model with the controllers of the YAML.
- `WeatherForecaster` on the weather channels of the model: `WeatherFile` reads the dew point, wind
  and opaque sky cover and computes the sky temperature as the Buildings weather reader;
  `OccupancySchedule` splits the radiant and convective gains (`from_parameters`,
  `radiant_and_convective`); `ForecastNoise.perturb` applies the forecast errors to named channels.
- `MovingHorizonEstimator` and `EstimatorSettings`: the full state of a `FastModel` from its zone air
  temperatures by moving horizon estimation (`trano.optimization`) over a sliding window;
  `PerfectEstimator` gives the true state of the plant.
- `Calibrator`, `CalibrationSettings`, `ParameterGroup` and `CalibrationResult`: least-squares
  calibration of groups of parameters (envelope conductances and capacities, air capacity,
  infiltration, boiler efficiency; solar transmittance and emitter capacity in `OPTIONAL_GROUPS`) on
  the air temperatures and the meters, on the Radau collocation of the data with IPOPT; `evaluate`
  and `simulate_fit` give the open-loop fit; calibrations are saved to `<stem>.calibration.json`.
- `run_rule_based`, `run_scenario_rule_based` and `Project.run_rule_based`: the controllers of the
  YAML (the closed-loop `FastModel`) on the same building, costed and assessed like an MPC run;
  `Project.true_closed_loop` gives the closed loop of the perturbed building.
- `random_factors` and `perturb`: a "true" building with its parameter groups at random values;
  `generate_monitoring_data` operates it with a thermostat with a random set point.
- `units` module (`KELVIN`, `STATE_BOUNDS`).

### Changed

- **mpctrano is built on Trano's `FastModel` library.** `ModelPredictiveController` decides the
  controls of the systems (valves, mixing valves, heat pump speed, boiler load) on the Radau
  collocation of the model (`trano.optimization`), and prices the energy read from the meters (the
  electricity bought and sold, the fuel, each at its price: `MPCSettings.fuel_price` and
  `feed_in_price`); the capacity charge and the demand response power caps apply to the electric
  power bought from the grid. It is warm started from a CVODES simulation of the shifted plan.
- `MonitoringData`, `ControlMetrics`, `ClosedLoopResult`, the flexibility indicators, the scenarios,
  `Project`, the CLI and the plots work on the variables of the FastModel: `<zone>.TAir`, the
  controls of the systems, the meter readings and the energy bought (`fuel`, `bought`, `sold`).
- The scenarios estimate the state by moving horizon estimation (`estimator="mhe"`, or
  `"perfect"`); the demand response event of the catalogue caps the electric power at 800 W.
- The tests, the tutorials and the setup-check notebook use the single-zone air-to-water heat pump
  house of the Trano test suite (`tests/models/single_zone_air_water_heat_pump.yaml`) and its boiler
  variant.
- trano >= 0.48; rumoca 0.10.3 built from the commit `c74cddf` of its main branch (pinned in
  `[tool.uv.sources]`; the CasADi export of the FastModel models is not released on PyPI).
- `ComfortSpec.bounds` takes any model with `zone_names`.
- The script tutorials of the package moved from `tutorials/` to `tutorials/package/`; the
  top-level `tutorials/` directory is reserved for the numbered interactive notebooks.
- Ruff no longer lints the generated `.ipynb` files (the paired `.py` scripts are linted).

### Removed

- The RC models of Trano's `mpc` library: `RCModel`, its `.mpc.json` interface and RC model types,
  the RK4 discretisation, `KalmanFilter`, the per-orientation irradiance and the solar geometry of
  the forecaster, the RC calibration of individual parameters and the three-zone RC test building.
