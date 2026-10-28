# Plan: interactive MPC and energy-flexibility tutorials in mpctrano

Status: **re-planned on 2026-10-15 on Trano's `FastSmoothBuildings` library; phases A and B and Day 7 done on
2026-10-21** (notebooks 01 to 06, the flexibility library, the calibration from the YAML, the cached
studies, the docs site and the Colab launch; the caches and the executed notebooks finished on
2026-10-22).
The YAML stays the single source of truth: Trano renders it into the `FastSmoothBuildingsMPC` model (the
controls of the systems as inputs) and the closed-loop `BuildingModel` (the controllers of the YAML),
rumoca translates them into CasADi, and mpctrano contains no building or system physics, with one
exception decided on 2026-10-21 and open to review (section 5): the electrical storages (battery,
vehicle) are linear models in the controller, at the meter, since Trano removed its battery and
charger elements with the `mpc` library.

Execution: one phase per working session, each ending with a pushed commit and a short report.
Every step lists what it touches and a "done when" check, so a session can stop after any step and
the next one resumes from this file. Building models that the tutorials need and Trano lacks are
added to Trano (test models and FastSmoothBuildings components) on its feature branch.

Trano version: mpctrano requires trano >= 0.65 (packages that reference FastSmoothBuildings instead of embedding
it); every test, notebook and package tutorial passed with trano 0.66.0 on 2026-10-28. The migration of
`TRANO_COMPATIBILITY_PLAN.md` is done.

---

## 0. Where things stand

### 0.1 The pipeline

```
trano YAML --(FastSmoothBuildingsMPC)--> building_mpc (controls of the systems as inputs)
           --(BuildingModel)-----> the same building with the controllers of the YAML (rule-based reference)
           --(rumoca c74cddf)--> CasADi ODE xdot = f(t, x, u, p)
           --> mpctrano: Radau collocation (trano_optimization), economic MPC on the meters,
               moving horizon estimation, calibration of parameter groups, closed loop, scenarios
```

Done (mpctrano):

* `BuildingModel` (roles read from the Modelica model, CVODES simulation, temperature units restored),
  `WeatherForecaster` (weather channels, sky temperature, occupancy of the YAML), `ForecastNoise`.
* `ModelPredictiveController`: tariff on the metered electricity and fuel, capacity charge and
  demand response power caps on the electric power, soft comfort band, warm start.
* `MovingHorizonEstimator`, `Calibrator` (envelope conductances and capacities, air capacity,
  infiltration, boiler efficiency), synthetic data from a thermostat with a random set point.
* `run_closed_loop`, `run_rule_based` (the controllers of the YAML), metered `ControlMetrics`,
  `Scenario` catalogue, `compare`, `value_of_flexibility`, `Project`, CLI.
* Day 1 of the previous plan: price signals (`mpctrano.prices`), notebook tooling, the setup-check
  notebook, `references.md`.

Done (Trano): `tests/models/single_zone_air_water_heat_pump.yaml`, the reference building of the
tutorials (27 states, a day-ahead plan in 5 to 20 s).

### 0.2 What BuildingModel covers, and the gaps

| Needed by the tutorials | BuildingModel today |
|---|---|
| Heat pump + radiators or floor heating, buffer | yes (air-to-water heat pump with its buffer, radiators) |
| Boiler, mixing circuits, several zones | yes (the six-zone heat pump house: a plan takes over a minute) |
| Ventilation with heat recovery, PV | yes |
| Domestic hot water tank | **gap**: to add to BuildingModel (Trano) |
| Chiller, AHU cooling | **gap**: BuildingModel heats only |
| Battery, EV charger | **gap**: to add to BuildingModel (Trano) |
| District (several buildings, feeder) | **gap**: one network per model |

### 0.3 Measured runtimes (this container, heat pump house, hourly steps)

| Task | Time |
|---|---|
| Trano + rumoca translation | 9 s |
| MPC plan, 24 h horizon | 5-20 s (cold), 4-8 s (warm) |
| Moving horizon estimation, 12 h window | < 0.5 s |
| Calibration, 4 days | 30 s |
| Rule-based reference, 1 day | 0.1 s |

A closed-loop day takes 3 to 5 minutes: the notebooks cache their closed loops and run open-loop
plans live; CI runs them in quick mode (3 h, 6 h horizon).

---

## 1. Notebooks

| Notebook | Topic | Status |
|---|---|---|
| `00_setup_check` | toolchain check, first BuildingModel from a YAML, one MPC plan | ready |
| `01_house_heat_pump` | the heat pump house: pipeline, plan widget, closed loop vs rule-based, estimation | ready (2026-10-21) |
| `02_flexibility_use_cases` | implicit / explicit flexibility, capacity tariff, envelope, negative prices with PV | ready (2026-10-21) |
| `03_calibration_from_yaml` | parameters declared in the YAML, calibration of the closed-loop model on noisy data, IPOPT, before / after | ready (2026-10-21) |
| `09_apartment_block` | central heat pump, radiators, DHW tank | needs the DHW tank in FastSmoothBuildings (decision 1) |
| `04_office_cooling` | ideal cooling at an EER, occupancy, capacity charge, summer day | ready (2026-10-21); a chiller in FastSmoothBuildings would add the part load (decision 2) |
| `05_pv_battery_ev` | PV, battery and vehicle at the meter next to the thermal storage | ready (2026-10-21); storages in the controller (decision 3) |
| `06_your_building` | edit the YAML in a cell, regenerate, plan, MPC vs the controllers of the YAML, two versions side by side | ready (2026-10-21), everything live |
| `07_district_coordination` | 3 to 5 buildings behind one feeder: independent, centralised, ADMM | needs district models |
| `08_grid_perspective` | value of the flexibility for the TSO and the DSO, market and regulation | planned |

The studies behind the notebooks (`tutorials/_house.py`, `_flexibility.py`, `_office.py`, `_pv.py`) define
the cached closed loops (`tutorials/cache/`, `make cache`, about four hours on four cores); the notebooks
read them and solve their plans live. The buildings are in `tutorials/buildings/` (the single-zone heat pump
house of the Trano test suite and its `old` and `pv` variants, a two-zone office): the two-zone house of the
first plan took 25 to 60 s per plan, the single-zone one 5 to 20 s.

## 2. Phases

### Phase A - reference notebook (`01_house_heat_pump`) - done 2026-10-21

1. `mpctrano.notebook`: `TutorialBuilding` (model, closed-loop reference, true building, spun-up
   states), `ResultCache` (parquet, keyed by the run parameters), price options, widgets.
2. `tutorials/01_house_heat_pump`: pipeline, plan widget, closed loop against the true building vs
   the controllers of the YAML, the price of the model error, estimation (true state, moving
   horizon, none). Widgets read the cache in milliseconds; the plan widget solves in 5 to 20 s.

### Phase B - flexibility use cases (`02_flexibility_use_cases`) - done 2026-10-21

3. `mpctrano.flexibility`: activation requests and rebound (`respond`), flexibility function,
   envelope (bisection on caps and floors of the electricity bought), peak reduction.
4. `mpctrano.tariff`: `CapacityTariff` (monthly peaks, floor), export price series, power floors,
   notice time; the controller carries the running peak. A 15-minute plan solves in one to two
   minutes: the 15-minute market time unit is shown on plans, not closed loops.

### Phase C - systems BuildingModel lacks - Day 7 done 2026-10-21 with two workarounds

5. Cooling: the ideal emitter of FastSmoothBuildingsMPC is a heat flow input bounded to heating by Trano;
   `BuildingModel.with_control_bounds` lets it go negative and `MPCSettings.heat_cop` / `cooling_eer`
   cost it (`04_office_cooling`). Battery and vehicle: linear state-of-charge models at the meter
   in the controller (`mpctrano.storage`, `05_pv_battery_ev`). Still missing in FastSmoothBuildings: the DHW
   tank with its draw-off (notebook 03), a chiller with part load, a battery and a charger as
   Modelica components (decisions 1 to 3).

### Phase D - district and grid

6. Several buildings behind a feeder (one model per building, coordination by ADMM in mpctrano on
   the shared feeder constraint), then the grid perspective notebook.

## 3. Acceptance checks at the end of every phase

* mpctrano: `make linting`, `make tests` green; Trano (when touched): its linting and tests green.
* `make notebooks` green with `MPCTRANO_QUICK=1`.
* Deterministic seeds, fixed IPOPT tolerances, reproducible caches.
* CHANGELOG updated; every new public symbol typed, tested, documented.

## 4. Decisions left to the author (2026-10-21)

1. **DHW tank in FastSmoothBuildings (Trano)**: notebook 09 (apartment block, the tank as the cheapest
   flexibility) needs a tank with a draw-off disturbance in Trano's FastSmoothBuildings; the apartment YAMLs
   and the old notebook text are on the branch `ccr-c0c43f9d-7vxewf` for reference.
2. **Chiller in FastSmoothBuildings (Trano)** or keep the constant EER of notebook 04.
3. **Battery and vehicle**: keep them in the controller (`mpctrano.storage`, exact linear models,
   no plant mismatch) or add them to BuildingModel in Trano and read them from the meters.
4. **Price samples**: `tutorials/scripts/fetch_prices.py` needs network access to Energy-Charts /
   ENTSO-E (blocked in the container); the studies use the synthetic day-ahead signal until a
   sample is committed under `mpctrano/data/prices/`.
5. **Pages and Colab**: the docs site publishes the executed notebooks (mkdocs-jupyter); the widgets
   of the live sections are static there. Every notebook carries an *Open in Colab* badge: its first
   cell clones the repository and installs mpctrano with the rumoca wheel of the GitHub release
   `rumoca-0.10.3-c74cddf` (built by `.github/workflows/rumoca-wheel.yaml` on the tag; without a
   wheel for the runtime's Python it builds rumoca from source, about ten minutes). Colab is the
   interactive deployment unless a Binder or JupyterLite one is preferred. The wheel workflow has to
   be run by hand once from `main` (Actions > rumoca wheel) to create the release.
6. **Calibration of a closed-loop model** (notebook 03): the collocated problem does not converge on
   a model with thermostats and hysteresis inside, so such a model is fitted by single shooting
   (`CalibrationSettings.method`, automatic): IPOPT with wide central differences and full steps on
   the simulated objective. Two things follow for the author to weigh: (a) the parameters a
   thermostat-controlled house can identify are few - the walls and the infiltration trade along a
   flat valley, a conductivity and a thickness only enter as `k / x` - so the YAML should declare
   what a measurement cannot give and nothing more; (b) the fit of such data lands at about 0.17 K
   RMSE rather than the 0.1 K of the sensor, because a switching time one integration step off
   moves the hourly sample: finer data (15 minutes) or a smoothed thermostat in Trano's FastSmoothBuildings
   would close that gap.

## 5. Risks and mitigations

* rumoca: the BuildingModel export needs rumoca main at `c74cddf`; later commits, and the 0.10.3 release on PyPI
  built from them, stop at residual equations (section 6 of `TRANO_COMPATIBILITY_PLAN.md`). The commit stays
  pinned in `pyproject.toml` until a release exports these models or FastSmoothBuildings avoids the two shapes.
* Solve times grow with the number of zones and circuits (over a minute per plan for six zones):
  cache the closed loops, keep the live parts open loop, use the single-zone house for widgets.
* Parameters settled at translation (the radiator output law, the heat pump maps) cannot be
  calibrated: the calibration adjusts the envelope, the air and the boilers.
* Regulatory facts move: every claim carries a verification date and a status flag.
