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
  model, the simulation runs with CVODES.
- `FastModelPredictiveController` and `FastMPCSettings`: economic MPC of a `FastModel` on Radau
  collocation (`trano.optimization`), with the tariff, capacity charge, comfort band and demand
  response limits of the RC controller applied to the metered electricity and fuel.
- `FastWeatherForecaster`: the disturbances of a `FastModel` from the weather file of its runnable model
  and the occupancy schedules of its YAML; `WeatherFile` reads the dew point, wind and opaque sky
  cover and computes the sky temperature as the Buildings weather reader; `OccupancySchedule` splits
  the radiant and convective gains (`from_parameters`, `radiant_and_convective`);
  `ForecastNoise.perturb` applies the forecast errors to named channels.
- `mpctrano.fast_simulation`: `FastModelPlant`, `PerfectEstimator`, `run_fast_closed_loop` and
  `run_rule_based` (the closed-loop `FastModel` with the controllers of the YAML as the baseline),
  costed from the meters of the plant (`FastControlMetrics`). `FastModel.from_yaml(..., library=
  "FastModel")` builds the closed-loop model.
- `make rumoca-fastmodel`: installs the rumoca commit that exports the FastModel models (`c74cddf`,
  not released on PyPI); the FastModel tests are skipped without it.

### Changed

- trano >= 0.48. The azimuths of the test building are in radians, as trano now requires.
- `ComfortSpec.bounds` takes any model with `zone_names` (`RCModel` or `FastModel`).

- The script tutorials of the package moved from `tutorials/` to `tutorials/package/`; the
  top-level `tutorials/` directory is reserved for the numbered interactive notebooks.
- Ruff no longer lints the generated `.ipynb` files (the paired `.py` scripts are linted).
