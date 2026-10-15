# Plan: interactive MPC and energy-flexibility tutorials in mpctrano

Status: **re-planned on 2026-10-15 on Trano's `FastModel` library.** The YAML stays the single
source of truth: Trano renders it into the `FastModelMPC` model (the controls of the systems as
inputs) and the closed-loop `FastModel` (the controllers of the YAML), rumoca translates them into
CasADi, and mpctrano contains no building or system physics.

Execution: one phase per working session, each ending with a pushed commit and a short report.
Every step lists what it touches and a "done when" check, so a session can stop after any step and
the next one resumes from this file. Building models that the tutorials need and Trano lacks are
added to Trano (test models and FastModel components) on its feature branch.

---

## 0. Where things stand

### 0.1 The pipeline

```
trano YAML --(FastModelMPC)--> building_mpc (controls of the systems as inputs)
           --(FastModel)-----> the same building with the controllers of the YAML (rule-based reference)
           --(rumoca c74cddf)--> CasADi ODE xdot = f(t, x, u, p)
           --> mpctrano: Radau collocation (trano.optimization), economic MPC on the meters,
               moving horizon estimation, calibration of parameter groups, closed loop, scenarios
```

Done (mpctrano):

* `FastModel` (roles read from the Modelica model, CVODES simulation, temperature units restored),
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

### 0.2 What FastModel covers, and the gaps

| Needed by the tutorials | FastModel today |
|---|---|
| Heat pump + radiators or floor heating, buffer | yes (air-to-water heat pump with its buffer, radiators) |
| Boiler, mixing circuits, several zones | yes (the six-zone heat pump house: a plan takes over a minute) |
| Ventilation with heat recovery, PV | yes |
| Domestic hot water tank | **gap**: to add to FastModel (Trano) |
| Chiller, AHU cooling | **gap**: FastModel heats only |
| Battery, EV charger | **gap**: to add to FastModel (Trano) |
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
| `00_setup_check` | toolchain check, first FastModel from a YAML, one MPC plan | ready |
| `01_house_heat_pump` | the heat pump house: pipeline, plan widget, closed loop vs rule-based, estimation | next |
| `02_flexibility_use_cases` | implicit / explicit flexibility, capacity tariff, envelope, negative prices | planned |
| `03_apartment_block` | central heat pump, radiators, DHW tank | needs the DHW tank in FastModel |
| `04_office_cooling` | AHU + chiller, occupancy schedule, summer week | needs cooling in FastModel |
| `05_pv_battery_ev` | PV, battery and EV charger next to the thermal storage | needs battery and EV in FastModel |
| `06_district_coordination` | 3 to 5 buildings behind one feeder: independent, centralised, ADMM | needs district models |
| `07_grid_perspective` | value of the flexibility for the TSO and the DSO, market and regulation | planned |

## 2. Phases

### Phase A - reference notebook (`01_house_heat_pump`)

1. `mpctrano.notebook`: `ResultCache` (parquet, keyed by scenario hash), widget factories (comfort
   band, price signal, horizon / weights), runtime guard.
2. `tutorials/01_house_heat_pump.py` -> `.ipynb`: YAML -> Modelica -> CasADi (states, controls,
   meters, the COP in the heat pump component); open-loop plan widget; closed loop against a
   perturbed building vs the controllers of the YAML; estimation on/off. Done when: the notebook
   runs top to bottom in CI and every widget updates in < 5 s from the cache.

### Phase B - flexibility use cases (`02_flexibility_use_cases`)

3. `mpctrano.flexibility`: activation requests and rebound, flexibility function (step-price
   response), envelope (sweep of constrained MPC solves), KPI set with references.
4. `mpctrano.tariff`: capacity tariff (monthly quarter-hour peak), negative prices and export
   remuneration. Done when: KPIs reproduce with fixed seeds and tests cover each metric.

### Phase C - systems FastModel lacks (Trano branch)

5. DHW tank with a draw-off disturbance, battery and EV charger with their meters, cooling (chiller,
   AHU): FastModel components and templates, translation and validation tests in Trano, then the
   apartment, office and PV/battery/EV notebooks.

### Phase D - district and grid

6. Several buildings behind a feeder (one model per building, coordination by ADMM in mpctrano on
   the shared feeder constraint), then the grid perspective notebook.

## 3. Acceptance checks at the end of every phase

* mpctrano: `make linting`, `make tests` green; Trano (when touched): its linting and tests green.
* `make notebooks` green with `MPCTRANO_QUICK=1`.
* Deterministic seeds, fixed IPOPT tolerances, reproducible caches.
* CHANGELOG updated; every new public symbol typed, tested, documented.

## 4. Risks and mitigations

* rumoca: the FastModel export needs rumoca main up to `c74cddf` (later commits stop at residual
  equations); the commit is pinned in `pyproject.toml` until a release exports these models.
* Solve times grow with the number of zones and circuits (over a minute per plan for six zones):
  cache the closed loops, keep the live parts open loop, use the single-zone house for widgets.
* Parameters settled at translation (the radiator output law, the heat pump maps) cannot be
  calibrated: the calibration adjusts the envelope, the air and the boilers.
* Regulatory facts move: every claim carries a verification date and a status flag.
