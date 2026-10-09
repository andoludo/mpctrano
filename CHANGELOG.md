# Changelog

All notable changes to mpctrano are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the tags created by the
release pipeline.

## [Unreleased]

### Added

- Tutorial `03_apartment_block` with its study (`tutorials/_apartments.py`, cache in
  `tutorials/cache/apartments/`): six apartments on one heat pump with radiators and a 1 m3
  domestic hot water tank; the dispatch of the shared compressor, emitter temperature and heat
  pump size (`floor_heating`, `high_temperature`, `small_heat_pump`, `big_tank` variants), the
  tank free or locked during an evening shed and in the envelope.
- Apartment block typology: `tutorials/buildings/apartments.yaml` and its variants, generated as
  R3C2 and R4C3 models in `tutorials/models/apartments/`.
- `compressor_load`: use of the shared compressors of a run (mean load, share of the steps at the
  limit, share of the electricity per input); `plot_dispatch` (stacked electrical inputs against
  the compressor limit, storage temperatures with their band, zone temperatures, price).
- `RCModel.with_storage_band` (a copy of the model with another band for a tank, to lock it or
  widen it) and `RCModel.consumption_capacity` (largest draw the bounds and the compressor
  limits allow).

### Changed

- `ActivationRequest.notice_hours` is now honoured: the controller plans with the cap or floor of
  the request from the notification on (one hour before the window by default, `None` for the
  whole horizon) instead of from the moment the window entered its horizon. The activation runs
  of tutorial 02 were recomputed; `DemandResponseEvent.notice_hours` carries it.

### Fixed

- The export of the meter is bounded by the on-site sources (PV generation and battery
  discharge), and the import / export split is skipped on a building without any source: a power
  floor (upward activation, envelope) could previously be met by importing and exporting at the
  same time. The upward bound of `FlexibilityEnvelope` respects the shared compressor limits.

- Explicit flexibility: `ActivationRequest` (shed or absorb a power during a window, expressed
  as a soft cap or floor on the billed power against the baseline), `respond` (baseline and
  response from the same plant state) and `ActivationResult.metrics` (delivered energy and
  power, delivery ratio, rebound energy, peak, duration and ratio, cost and comfort changes);
  `flexibility_function` (response to a price step, Junker et al.).
- `FlexibilityEnvelope`: the power the building can shed or absorb for a duration without
  leaving the comfort band, and the cost of that plan, by bisection on constrained MPC solves;
  `plan_comfort_violation`.
- `CapacityTariff`: a capacity charge on the monthly peak as the Flemish *capaciteitstarief*
  (price per kW and year on the average of the monthly peaks floored at 2.5 kW); the controller
  carries the peak of the month so far (`ModelPredictiveController.solve(peak_so_far=...)`, the
  closed loop tracks `running_peak_kw`). `DemandResponseEvent.power_floor` for upward
  activations; `solve()` accepts per-call `power_limits` and `power_floors`.
- `compare` reports `peak_reduction_pct`; `plot_activation`, `plot_envelope`.
- Tutorial `02_flexibility_use_cases` with its study (`tutorials/_flexibility.py`, cache in
  `tutorials/cache/flexibility/`): implicit flexibility at hourly and 15-minute resolution, peak
  shaving under the capacity tariff, activations with rebound, envelope and flexibility function,
  negative prices with PV, battery and EV.
- `mpctrano.notebook`: helpers of the interactive tutorials: `TutorialBuilding` (pre-generated
  control and plant models), `ResultCache` (closed-loop results as parquet, keyed by the run
  parameters), `price_options`, `comfort_spec`, `scenario`, `kpi_table`, the `plan_explorer`
  (live open-loop MPC) and `closed_loop_explorer` (cached runs, recompute button) widgets,
  `quick_mode`, `days` and `timed`.
- `mpctrano.plot`: one palette for every figure, `plot_plan`, `plot_states`, `plot_cop`;
  `plot_closed_loop` shows the billed power and the export.
- Tutorial `01_house_heat_pump`: pipeline walk-through, live MPC plan, closed loop with plant
  mismatch from the cache (`tutorials/cache/house/`, built by `tutorials/scripts/build_cache.py`),
  state estimation on/off, R3C2 vs R4C3 controller.
- Interface version 2 of trano 0.36 (energy systems): `RCModel` exposes the systems, the energy
  carrier of each control input, the compressor limits shared by the inputs of a heat pump
  (`control_groups`), the tank temperature bands, the electric vehicles and the bounds of every
  control and state (`control_bounds`, `state_bounds`).
- `mpctrano.meter.Meter`: electrical balance at the meter (consumption, battery discharge, PV
  generation, import and export), gas of the boilers and ideal heating, symbolic and on frames.
- The MPC bills the imported electricity at the tariff and the export at `export_price`, costs the
  gas with `MPCSettings.gas_price`, enforces the compressor limits, the state of charge bounds,
  the tank bands (soft), the plug-in windows and the departure targets of the vehicles (soft,
  `vehicle_penalty`). `MPCSolution.meter` holds the planned import, export and gas.
- `Mismatch`: how the plant differs from the control model (RC structure, perturbed envelope
  parameters, sensor noise, forecast noise); `Project.plant()` builds an R4C3 plant from the same
  YAML against the R3C2 controller; `build_plant`, `Scenario.mismatch`.
- `ControlMetrics`: export, gas, PV generation and self-consumption; `ClosedLoopResult.frame`
  carries the meter columns (`P_import`, `P_export`, `P_meter`, `P_gas`, ...) and the solver status.
- Tariffs carry an export price (`Tariff.export_price`, `DynamicTariff.export_series`,
  `PriceSignal.tariff()`).
- The forecaster builds the PV generation, the hot water draw-off and the driving discharge of
  the vehicles from the sources of the interface.
- Tutorial house: `tutorials/buildings/house.yaml` and six variants (`old`, `radiators`,
  `big_heat_pump`, `pv`, `pv_battery_ev`), generated as R3C2 and R4C3 models in
  `tutorials/models/house/` by `tutorials/scripts/generate_models.py`.
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

### Changed

- Requires trano >= 0.36 (interface version 2; version 1 files are still read).
- `KalmanFilter` is an extended Kalman filter linearised at the current estimate (the heat pump
  terms are bilinear); `RCModel.linearization()` gives the discrete map and its Jacobian.
- `ControlMetrics.energy_kwh` and `peak_power_kw` are the energy and power billed at the tariff
  (ideal heating plus grid import), no longer the thermal heating; `total_cost` subtracts the
  export revenue and adds the gas cost.
- `ModelPlant` matches the signals of the control model by name (`Plant.align`), so a plant of
  another RC structure can be controlled; `measure()` returns the measured states (zone
  temperatures and storages), `ClosedLoopResult.zones` and the plots follow the heating input of
  each zone (`<zone>_QHea` or `<zone>_PHea`).
- `perturb` leaves the parameters of the energy systems untouched unless selected.
- The script tutorials of the package moved from `tutorials/` to `tutorials/package/`; the
  top-level `tutorials/` directory is reserved for the numbered interactive notebooks.
- Ruff no longer lints the generated `.ipynb` files (the paired `.py` scripts are linted).
