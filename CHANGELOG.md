# Changelog

All notable changes to mpctrano are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the tags created by the
release pipeline.

## [Unreleased]

### Fixed

- rumoca 0.10.3 on PyPI (27 October 2026) cannot export the FastSmoothBuildings models to CasADi ("Target
  'casadi-ode' does not support feature 'residual_equations'"), yet reports the same version as the build of commit
  `c74cddf` that can. A plain `pip install mpctrano` picks it, and every model failed with that message.
  `rumoca_exports_building_models` reported `True` for it: its probe now also holds the two function shapes of the
  library that this rumoca refuses, so the translation error names the build to install
  (`RUMOCA_REQUIREMENT`), the setup-check notebook stops early, and the Colab setup replaces it with the wheel of
  the `rumoca-0.10.3-c74cddf` release (`--force-reinstall`: pip otherwise keeps the build of the same version).
  The README gives the pip commands.

### Changed

- Locked and tested with trano 0.66.0, FastSmoothBuildings 0.5.0 and CasADi 3.8.2; `fastsmoothbuildings` is a
  declared dependency (mpctrano imports it to find the Modelica library).

### Changed

- **Breaking**: the class `FastModel` is renamed `BuildingModel` (the library it models is FastSmoothBuildings);
  `FAST_MODEL_LIBRARY` is `MPC_LIBRARY` and `rumoca_exports_fast_models` is `rumoca_exports_building_models`. The
  former names are removed.

### Added

- `BuildingModel.from_source` and `BuildingModel.from_file` take `roots`, the folders of the Modelica libraries a
  package uses without embedding them; `BuildingModel.from_network` passes the folders the network names
  (`Network.libraries`).
- A package that declares `uses(FastSmoothBuildings(...))` without holding the library (trano's default from 0.65,
  as Buildings and IDEAS) is translated with the installed `fastsmoothbuildings` package (`used_libraries`): no
  `roots` to pass. Requires trano >= 0.65.

### Changed

- Trano's FastModel library is now FastSmoothBuildings (its own repository and PyPI package): the models are
  generated with `FastSmoothBuildingsMPC` and the closed loop with `FastSmoothBuildings`
  (`library="FastSmoothBuildings"`). Trano still accepts `FastModel` and `FastModelMPC`, with a deprecation
  warning. The Python class `FastModel` keeps its name.

### Added

- Flexibility quantification on FastModel (the Days 2 to 6 tutorial work ported): activation
  requests with the delivered energy and the rebound (`respond`, `ActivationResult`), the
  flexibility function (`flexibility_function`), the flexibility envelope (`FlexibilityEnvelope`),
  the peak reduction in `compare`.
- Tariffs: export price series, power floors and notice time of the demand response events,
  `CapacityTariff` (monthly peaks); the controller takes per-solve caps and floors and the peak
  already billed (`peak_so_far`), and reports the planned meter powers (`MPCSolution.meter`).
- `spin_up`: a state after a few days of the controllers of the YAML; the running monthly peak
  and the billed power in the closed-loop frame; PV generation and self-consumption in the metrics.
- Cooling with ideal emitters: `FastModel.with_control_bounds` (a negative lower bound),
  `MPCSettings.heat_cop` and `cooling_eer`.
- `mpctrano.notebook` (`TutorialBuilding`, `ResultCache`, price options, widgets) and the plots of
  plans, closed loops, states, activations and envelopes with a shared palette.
- Tutorials: `tutorials/buildings/` (the house and its variants, the office), the studies
  `tutorials/_house.py`, `_flexibility.py`, `_office.py` with their cached runs
  (`make cache`), the notebooks `01_house_heat_pump`, `02_flexibility_use_cases` and
  `04_office_cooling`; the docs site renders the executed notebooks (mkdocs-jupyter, `make docs`).
- `scripts/calibrate_multizone.py` and `scripts/multizone_heat_pump.yaml`: a ready-to-run calibration of
  a six-zone heat pump house on two weeks of monitoring data generated with Trano (`CalibrationSpec`
  takes a `base_dir` for the weather file).
- `FastModel.smooth_controllers(width_factor, switching_time)`: a copy of a closed-loop model whose
  thermostats, hysteresis and limiters switch over wider transitions and a set switching time (the
  `dU`, `dY`, `dMax`, `dT`, `dtSet` and `tau` parameters of trano >= 0.59). The gentler copy keeps the
  data within 0.15 K of the sharp model and calibrates by shooting in a tenth of the time.
- Calibration of closed-loop models by single shooting (`CalibrationSettings.method`, automatic for a
  model without control inputs): CVODES simulates the candidate model, spun up with its own
  controllers, inside the objective; IPOPT takes wide central differences and full steps.
  `CalibrationSpec.groups(values=..., only=...)` measures the sensitivities around given values and
  selects fields; `CalibrationResult.method`.
- `tutorials/06_your_building`: the building is a YAML cell; the notebook regenerates the models,
  plans with the widget and compares the MPC with the controllers of the YAML for two versions.
- Every notebook opens in Google Colab (badge and a first cell that clones the repository and runs
  `tutorials/_colab.setup()`); the `rumoca wheel` workflow builds the manylinux wheel of the pinned
  rumoca commit and attaches it to the release `rumoca-0.10.3-c74cddf` that the Colab setup installs.

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

### Fixed

- Calibration groups that move the same parameter of the model (a conductivity and a thickness both
  move a layer conductance) compound their factors instead of overwriting each other.
- `CalibrationSpec.sensitivities` perturbed the nominal YAML even when the base model was generated
  at other values, so the exponents picked up every other difference.

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
- trano >= 0.57 and trano-optimization >= 0.2 (`trano.optimization` moved to its own package); the
  published requirement on rumoca is >= 0.10.2 (its last release on PyPI) and `[tool.uv.sources]`
  builds 0.10.3 from the commit `c74cddf` of its main branch, which the CasADi export of the FastModel
  models needs.
- `FastModel.from_network` generates the model without Trano's containers
  (`include_container=False`), so the variables keep the names of the elements of the YAML;
  `FastModel.from_source` and `from_file` strip the container prefixes (`envelope1.`,
  `production1.`, ...) of a containerised model (`without_containers`).
- The internal gains of an occupancy model are the inputs `<occupancy>_gai.QCon` and `.QRad`
  (Trano 0.57); the forecaster still reads `<occupancy>.gai.*`. The parameter groups of the
  calibration match their elements with or without a container prefix.
- `tests/test_trano_contract.py`: the translation by rumoca of the FastModel and FastModelMPC models
  of the installed Trano (the two houses of mpctrano, BESTEST 600FF and a six-zone heat pump house
  copied from Trano), the equality of the flat and containerised models, and the names mpctrano
  relies on. Trano no longer runs these checks: it does not install rumoca.
- `ComfortSpec.bounds` takes any model with `zone_names`.
- The script tutorials of the package moved from `tutorials/` to `tutorials/package/`; the
  top-level `tutorials/` directory is reserved for the numbered interactive notebooks.
- Ruff no longer lints the generated `.ipynb` files (the paired `.py` scripts are linted).

### Removed

- The RC models of Trano's `mpc` library: `RCModel`, its `.mpc.json` interface and RC model types,
  the RK4 discretisation, `KalmanFilter`, the per-orientation irradiance and the solar geometry of
  the forecaster, the RC calibration of individual parameters and the three-zone RC test building.
