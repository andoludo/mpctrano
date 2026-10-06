# Plan: interactive MPC and energy-flexibility tutorials in mpctrano

Status: **approved on 2026-10-06 (all open questions resolved with the proposed defaults). Day 1 in progress.**
Decision taken on 2026-10-06: **the energy systems are described in the trano `mpc` Modelica
library, generated from the YAML, so that the YAML is the single source of truth.** mpctrano
contains no building or system physics; it consumes the Modelica model (via Rumoca) and the
interface JSON that trano writes next to it.

Execution: one phase per working session (a "day" below), each ending with a pushed commit and
a short report. Every step lists the files it touches and a "done when" check, so a session can
stop after any step and the next one resumes from this file.

Branches: mpctrano on `ccr-c0c43f9d-7vxewf`; trano on new feature branches
(`feature/mpc-systems` for Day 2, `feature/district-models` for Day 8), approved on
2026-10-06. trano `master` is never touched directly.

---

## 0. What the exploration found

### 0.1 The pipeline already exists, for the envelope only

```
trano YAML --(trano "mpc" library)--> flat RC Modelica (building_mpc) + <stem>.mpc.json interface
          --(rumoca, casadi-ode target)--> CasADi ODE xdot = f(x, u, d, p), p symbolic
          --> mpctrano: RK4 discretisation, Opti/IPOPT economic MPC, Kalman/MHE, closed loop
```

* trano: `trano/mpc/` (RCBuilding, RCZone, parameter estimation from the YAML geometry and
  constructions, `MPCModelInterface`), templates `trano/templates/mpc.jinja2` + `mpc_base.jinja2`.
  Zone equations are Jinja macros, written once and inlined both in the library models
  `Trano.MPC.Zones.*` and in the flat `building_mpc`. Zone types R1C1 / R3C2 / R4C3 / ISO13790.
  `tests/test_mpc_rumoca.py` guards the "translatable by Rumoca" contract.
* mpctrano (3.4 k lines): `RCModel` (Rumoca export, RK4 `discrete()`, exact affine form),
  `ModelPredictiveController` (one `casadi.Opti`, warm start; cost = price x thermal energy +
  capacity charge x horizon peak + soft comfort band + soft power cap + ramp), `ModelPlant` +
  `run_closed_loop`, `perturb` (plant mismatch), `ForecastNoise`, tariffs (flat, time-of-use,
  dynamic series, demand-response events), `Scenario` catalogue, `compare`,
  `value_of_flexibility`, `shifted_energy`, Kalman filter and MHE, calibration.
* Tooling: uv, ruff, mypy strict, pytest (`slow` marker), mkdocs, CI (lint -> tests -> minor
  bump + tag on main -> PyPI). Tutorials are scripts under `tutorials/`, run by the tests.

### 0.2 Rumoca constraints, verified on rumoca 0.10.2 (they shape the Modelica design)

| Modelica construct | `casadi-ode` target |
|---|---|
| Explicit `der(x) = expr` with smooth `expr` (`exp`, `sqrt`, polynomials) | OK |
| Extra states (tank temperature, battery energy) | OK |
| `output Real y = expr` or any algebraic variable | **rejected** ("residual_equations" unsupported), even when it is a plain assignment |
| `if`, `max`, `min`, `abs` | **rejected** (events) |
| Sub-model instances with connectors / modifiers | flattened into algebraic variables, **rejected** |
| Inlined equations in one flat model | OK |

No other built-in target helps (`jax-ode`, `c-ode`, FMI targets share the same capabilities).
Consequences:

1. Everything the controller needs must be a **state**, an **input** or a **parameter** of the
   flat ODE. Electrical power is therefore the **control input** of each converter and the heat
   is derived inside the ODE: `QHea = COP(TOut, Th) * Pel`. There is no output variable to
   export; the interface JSON tells mpctrano which inputs are electrical and how to meter them.
2. System equations are written as Jinja macros next to the zone macros and inlined into
   `building_mpc`, exactly like the zones today. The library package `Trano.MPC.Systems.*`
   keeps one documented model per system (same macros) for reading and simulation.
3. Part-load / on-off behaviour, mixing valves and hydraulics are not modelled (continuous
   modulation); COP and EER are smooth polynomials in the outdoor and emitter temperatures.

### 0.3 Gaps against the task

| Needed by the task | Current state |
|---|---|
| Systems in the `mpc` library (heat pump, chiller, DHW tank, PV, battery, EV, emitter type) | None: the only control input is thermal `<zone>_QHea`. The YAML schema has `air_water_heat_pump` / `water_water_heat_pump` boiler variants (TSet, TSouSet, tank_volume, use_storage_tank), `photovoltaic`, `air_handling_unit`, radiator emissions; no battery, EV charger, chiller or explicit DHW element. |
| Electricity-side objective (import/export, running monthly peak, negative prices) | Thermal only. |
| Explicit flexibility (activation request, baseline, rebound), flexibility function / envelope, Annex 67 and BOPTEST KPIs | Only `DemandResponseEvent`, `shifted_energy`, `low_price_share`, comfort Kh, horizon peak. |
| Real price data (ENTSO-E / Elia) | Only `synthetic_day_ahead`. |
| District (several buildings behind a feeder, coordination) | One `RCBuilding` per network; nothing multi-building in trano or mpctrano. |
| Notebooks, widgets, jupytext, nbmake, pre-commit, CHANGELOG, Windows notes | None. |

### 0.4 Measured runtimes (this container, three-zone test building, R3C2)

| Step | Time |
|---|---|
| trano generate + Rumoca translate (first call) | 4.6 s |
| Rumoca translate only (R4C3) | 0.4 s |
| One MPC solve, horizon 24 h, step 1 h | ~110 ms |
| One MPC solve, horizon 24 h, step 15 min | ~240 ms |
| Closed loop, 2 days, step 1 h | 6 s |
| Closed loop, 2 days, step 15 min | 49 s |

Live widgets therefore re-solve a **single open-loop MPC plan** (instant); **closed-loop**
results come from a pre-computed cache over the widget grid with a "recompute" button and a
documented runtime.

---

## 1. Design decisions

1. **YAML -> trano `mpc` library -> Modelica -> Rumoca -> CasADi, for systems too.**
   trano gets `Trano.MPC.Systems` (Modelica, inlined into `building_mpc`), a `systems` section
   in `RCBuilding`, the mapping from YAML elements to these systems in `trano/mpc/estimation.py`
   (sizing from the design heating power, COP defaults, tank sizing from `tank_volume`), and a
   `systems` block in the interface JSON (`MPCModelInterface` version 2). The runnable `building`
   model wraps the new inputs like today (PV from the weather bus, EV/draw-off from external data).
2. **Formulation rules** (all smooth, all inline):
   * Heat pump: control input `<hp>_Pel` [W], bounds `[0, Pel_max]`; `QHea = COP * Pel` with
     `COP = c0 + c1*(TOut - 273.15) + c2*(Tsup - 273.15) + c3*(TOut - 273.15)*(Tsup - 273.15)`;
     `Tsup` is the emitter state `Th` (R4C3) or a parameter per emitter type (R3C2); the heat is
     split over the served zones with fixed fractions (parameters). Defaults fitted on EN 14825
     rating points (A-7/W35, A2/W35, A7/W35, A7/W55), verified in Day 1.
   * Chiller: control input `<ch>_Pel`, `QCoo = -EER(TOut) * Pel` into the served zones.
   * DHW tank: state `<tank>_T`, `der(T) = (COP_dhw * Pel_dhw - QDraw - UA (T - Ti_zone)) / C`,
     `QDraw` a disturbance input (draw-off profile from the interface), comfort as a minimum
     temperature band like a zone.
   * PV: disturbance input `PPV` [W] computed by the forecaster from the irradiance on the panel
     orientation and the YAML area / efficiency (the interface carries the source, like `HSol_*`).
   * Battery: state `E` [J], inputs `Pch`, `Pdis` >= 0, `der(E) = eta_ch Pch - Pdis / eta_dis`,
     bounds in the interface. EV: same with an availability window and an energy deadline carried
     by the interface (`ev_sessions`), enforced by the MPC as terminal constraints.
   * Backward compatibility: a building without systems keeps `<zone>_QHea` as today.
3. **mpctrano consumes the interface only.** `RCModel` learns the system states and inputs from
   the JSON; a new `mpctrano.meter` builds the electrical balance (sum of `*_Pel` and battery
   flows minus PV) and the import/export split as Opti variables; the MPC cost is written on
   grid import/export. No COP, tank or battery equation is ever written in Python.
4. **Control model vs plant, always separate objects.** `Scenario.mismatch` config (parameter
   perturbation std + seed, forecast noise, measurement noise, optional plant RC type). Default:
   R4C3 perturbed plant against an R3C2 controller, both generated from the same YAML. This is
   the reduced-order vs higher-order discussion without an FMU.
5. **Prices**: `PriceSignal` Pydantic model (day-ahead hourly, 15-min, synthetic spike, negative
   hours, grid fees and taxes stacked) backed by cached CSV samples of Belgian day-ahead prices.
6. **Interactivity**: `ipywidgets` + plotly `FigureWidget`; glue in `mpctrano.notebook`
   (optional extra `notebook`); notebooks jupytext-paired `.py` (percent format), executed by
   `nbmake` in CI with `MPCTRANO_QUICK=1`.
7. **Phase order**: flexibility metrics and use cases (task phase 4) right after the reference
   typology so the other typologies reuse them.

---

## 2. Typologies (4, confirmed on 2026-10-06)

| # | Typology | Why it contrasts | Model and YAML needs |
|---|---|---|---|
| 1 | Single-family house, air-water heat pump + floor heating | Reference. High mass, slow emitter, COP tied to supply temperature: textbook shifting, pre-heating and rebound. | 2 zones (day / night), R4C3 plant, floor-heating emitter settings, `air_water_heat_pump`. Variants: envelope (old / renovated / passive), mass, HP size, + PV. Existing YAML elements suffice. |
| 2 | Apartment block, central heat pump + radiators + DHW tank | Lower mass, fast emitters, a second storage with a hard daily demand: storage competition, DHW as the cheapest flexibility. | 6 zones (3 floors x 2 orientations), `use_storage_tank` + `tank_volume` mapped to the DHW tank, draw-off profile from external data. |
| 3 | Office, AHU + cooling, weekday occupancy | Cooling, large gains, occupancy window, summer week: pre-cooling against afternoon peaks, summer value for TSO/DSO. | 4 zones (core + 3 facades), new `chiller` system (or a cooling variant of the AHU), ventilation gains. |
| 4 | House of #1 + battery + EV charger | Electric next to thermal storage: dispatch order, negative prices, PV self-consumption, EV deadlines. | New `battery` and `ev_charger` YAML elements. |

Rule agreed with the author: a typology is in scope as long as it can be described in trano
YAML; the YAML schema is extended on the trano branch where an element is missing (chiller,
battery, ev_charger, DHW draw-off profile). YAMLs in `tutorials/buildings/`, generated `.mo` +
`.mpc.json` in `tutorials/models/`, both committed (text, tens of kB each).

---

## 3. Target layout

```
trano (branch)
  trano/mpc/systems.py        HeatPump, Chiller, DHWTank, Photovoltaic, Battery, EVCharger (Pydantic, Modelica symbols)
  trano/mpc/building.py       RCBuilding.systems, zone/system wiring, state and input ordering
  trano/mpc/estimation.py     YAML elements -> systems (sizing, COP defaults, tank from tank_volume)
  trano/mpc/interface.py      interface v2: systems block, electrical inputs, storage states, sources (pv, draw-off, ev)
  trano/mpc/district.py       RCDistrict (Day 7)
  trano/templates/mpc.jinja2  system macros, Trano.MPC.Systems library, inlined building_mpc, runnable wiring
  trano/data_models/*.yaml    new elements: chiller, battery, ev_charger (+ district)
  tests/test_mpc_systems.py, test_mpc_rumoca.py (extended), goldens

mpctrano
  mpctrano/model.py           (extended) system states/inputs from interface v2, bounds, storage comfort bands
  mpctrano/meter.py           electrical balance, import/export split, PV source in the forecaster
  mpctrano/mpc.py             (extended) cost on grid import/export, running monthly peak, EV terminal constraints
  mpctrano/prices.py          PriceSignal, fee stack, cached ENTSO-E/Elia loaders, synthetic shapes
  mpctrano/flexibility.py     (extended) ActivationRequest, baseline + rebound, FlexibilityFunction, FlexibilityEnvelope, KPIs
  mpctrano/tariff.py          (extended) CapacityTariff (monthly quarter-hour peak), negative prices
  mpctrano/scenarios.py       (extended) Mismatch, ScenarioStudy (grid of scenarios -> cache)
  mpctrano/district.py        Feeder, IndependentMPC, CentralisedMPC, ADMMCoordinator, Aggregator
  mpctrano/notebook.py        widget factories, ResultCache, runtime guard (extra "notebook")
  tutorials/README.md, PLAN.md, references.md, buildings/, models/, data/prices/, cache/, scripts/
  tutorials/01_house_heat_pump ... 07_grid_perspective (.py jupytext + .ipynb)
  tests/test_meter.py, test_prices.py, test_flexibility.py, test_district.py, notebooks via nbmake
  CHANGELOG.md, .pre-commit-config.yaml, .devcontainer/ (optional)
```

---

## 4. Phases, step by step

### Day 1 - Phase 1: scan, data, tooling (no modelling yet)

1. **Literature and regulation scan** -> `tutorials/references.md`. Each entry: full citation,
   DOI/URL, one line on its use, "verified on <date>". Items: IEA EBC Annex 67 and Annex 82;
   Junker et al. 2018 (flexibility function); Stinner et al. 2016; Reynders et al. 2017; SRI
   flexibility domain; BOPTEST KPIs (Blum et al. 2021); EU Electricity Market Design reform
   (Reg. 2024/1747, Dir. 2024/1711); network code on demand response (status); SDAC 15-min MTU
   go-live date; Flemish capacity tariff rules (Fluvius/VREG); Elia aFRR/mFRR product sheets;
   GOPACS, NODES, Piclo; EN 14825 rating points and typical air-water COP values.
   Anything not verifiable online is marked `[unverified]`.
2. **Price data** -> `tutorials/scripts/fetch_prices.py` (ENTSO-E with optional token, Elia open
   data without token) and committed samples in `tutorials/data/prices/` (BE day-ahead hourly
   for a year, one 15-min MTU week, a negative-price spring week) with `provenance.md`.
   Done when: `PriceSignal.from_cached("be_dayahead_2024")` works offline.
3. **Tooling**: dev deps (`jupyterlab`, `ipywidgets`, `jupytext`, `nbmake`, `pre-commit`,
   `pyarrow`), `notebook` extra, `.pre-commit-config.yaml`, `CHANGELOG.md`, CI job `notebooks`,
   jupytext config. Done when: `make linting`, `make tests`, `make notebooks` pass on a
   placeholder notebook.

### Day 2 - Phase 2a: trano `mpc` library systems (trano branch)

4. `trano/mpc/systems.py`: Pydantic models with Modelica symbols (same `rc_field` pattern as the
   zone parameters): `HeatPump` (served zones + fractions, COP coefficients, `Pel_max`, supply
   temperature source), `Chiller`, `DHWTank`, `Photovoltaic` (area, efficiency, orientation),
   `Battery`, `EVCharger`. `RCBuilding.systems`, ordering rules for states and inputs.
5. `mpc.jinja2`: system macros, `Trano.MPC.Systems` library models, inlined equations in
   `building_mpc` (the zone heat term becomes the sum of the converter contributions), runnable
   wiring (PV from the weather bus, draw-off / EV from external data, `Pel` real inputs).
6. `estimation.py`: YAML -> systems mapping (heat pump variants of the boiler element, emissions
   radiator vs floor heating -> emitter settings, `photovoltaic`, tank from `tank_volume`);
   schema additions `chiller`, `battery`, `ev_charger` in `trano/data_models/*.yaml` and the
   converter.
7. `interface.py` v2 with the `systems` block; tests: every system translates with Rumoca
   (extend `test_mpc_rumoca.py`), energy balance checks by simulation, goldens, interface
   round-trip. Done when: `trano create-model house.yaml mpc` on the tutorial house YAML gives a
   `building_mpc` with `hp_Pel` as input and Rumoca exports it; trano lint and tests green.
   Report the diff and open the trano PR (open question 4).

### Day 3 - Phase 2b: mpctrano consumes interface v2

8. `RCModel`: system states and inputs, bounds from the interface, storage comfort bands; the
   forecaster gains the PV and draw-off sources. `mpctrano.meter`: electrical balance and
   import/export split. `ModelPredictiveController`: cost on import/export, running monthly peak
   term, EV terminal constraints; thermal-only path unchanged (CHANGELOG entry). `Mismatch` in
   `scenarios` + plant builder (R4C3 plant from the same YAML).
9. House YAML + variants, `scripts/generate_models.py` (trano -> Modelica -> interface, checks
   the Rumoca translation), models committed. Done when: existing tests green, new tests for the
   meter and the electrical objective, `generate_models.py --typology house` is reproducible.

### Day 4 - Phase 2c: reference notebook

10. `mpctrano.notebook`: `ResultCache` (parquet, keyed by scenario hash), widget factories
    (comfort bounds, price signal, horizon / step / weights, building variant), runtime guard.
11. `tutorials/01_house_heat_pump.py` -> `.ipynb`: objectives; pipeline walk-through (YAML ->
    Modelica -> CasADi, inspect states / inputs / parameters, show the inlined COP equation);
    open-loop plan widget (live); closed loop with mismatch (cache + recompute); state estimation
    on/off; R3C2 vs R4C3 controller (solve time vs accuracy); takeaways; further reading.
12. `scripts/build_cache.py`; nbmake in CI. Done when: notebook runs top to bottom in CI, every
    widget updates in < 5 s from cache.

### Day 5 - Phase 4 (moved up): flexibility use cases and metrics

13. `mpctrano.flexibility`: `ActivationRequest` (direction, power, start, duration, notice),
    baseline run and `rebound` metrics, `FlexibilityFunction` (step-price response, Junker),
    `FlexibilityEnvelope` (max up/down power over duration at a given comfort / cost penalty via
    a sweep of constrained MPC solves), KPI set with references (shiftable energy, shifting
    efficiency, Kh, peak reduction, cost saving, storage efficiency).
14. `mpctrano.tariff`: `CapacityTariff` (monthly quarter-hour peak average, Flemish defaults as
    verified in step 1), negative prices and export remuneration in `PriceSignal`.
15. `tutorials/02_flexibility_use_cases.py`: implicit (day-ahead vs 15-min), peak shaving under
    the capacity tariff, explicit activation with rebound, envelope / function, negative prices
    + PV self-consumption; widgets and a KPI table. Done when: KPIs reproduce with fixed seeds,
    tests cover each metric on a toy profile.

### Day 6 - Phase 3a: apartment block

16. Block YAML (6 zones, heat pump, radiators, DHW tank, draw-off profile) + variants;
    `tutorials/03_apartment_block.py` (storage competition, DHW as cheapest flexibility,
    radiators vs floor heating).

### Day 7 - Phase 3b: office and house with battery + EV

17. Office YAML (chiller, AHU gains, weekday occupancy), summer scenarios,
    `tutorials/04_office_cooling.py`.
18. `tutorials/05_pv_battery_ev.py`: house of 01 with `photovoltaic`, `battery` and `ev_charger`
    (dispatch order thermal vs electric storage, negative prices, self-consumption, EV deadline).

### Day 8 - Phase 5: district

19. trano: `trano/mpc/district.py` (`RCDistrict` from several YAMLs or one district YAML, feeder
    / transformer capacity, rendered as one flat `district_mpc` with prefixed names, interface
    with building grouping), CLI `trano create-district`, Rumoca tests on a 5-building district.
20. `mpctrano.district`: `Feeder`, `IndependentMPC`, `CentralisedMPC`, `ADMMCoordinator`
    (shared feeder constraint, residual history), `Aggregator` (portfolio envelope).
21. `tutorials/06_district_coordination.py`: 5 house variants behind one feeder; price-only
    synchronisation peak, feeder limit, centralised vs ADMM convergence and solve time.

### Day 9 - Phase 6: grid perspective, docs, CI polish

22. `tutorials/07_grid_perspective.py`: TSO value (balancing, imbalance exposure, aFRR/mFRR
    activation from 02) vs DSO value (congestion, peak, capacity tariff, feeder from 06); market
    vs local conflict; regulatory context from `references.md` with status flags.
23. `tutorials/README.md` (prerequisites, `uv sync`, Rumoca install, regenerate models, launch,
    trano branch usage, troubleshooting Linux + Windows), mkdocs pages, optional
    `.devcontainer`, documented runtimes, final CHANGELOG, CI green on everything.

---

## 5. Acceptance checks at the end of every day

* mpctrano: `make linting`, `make tests` green; trano (when touched): `make linting`, tests green.
* `uv run pytest --nbmake tutorials/*.ipynb` green with `MPCTRANO_QUICK=1`.
* Deterministic seeds, fixed IPOPT tolerances, reproducible caches.
* CHANGELOG updated; every new public symbol typed, tested, documented.

---

## 6. Open questions

1. ~~Energy systems in mpctrano or trano?~~ **Resolved: trano `mpc` library (Modelica).**
2. ~~Typologies~~ **Resolved: the four typologies of section 2, each modelled in trano YAML,
   with the schema extended on the trano branch where needed.**
3. ~~Dependencies~~ **Resolved: dev group `jupyterlab`, `ipywidgets`, `jupytext`, `nbmake`,
   `pre-commit`, `pyarrow`; runtime extra `notebook = [plotly, ipywidgets]`.**
4. ~~trano release path~~ **Resolved: one trano PR for the systems after Day 2 and one for the
   district after Day 8, merged and released by the author; until then mpctrano's lock points
   at the trano branch via a git source.**
5. ~~trano branch names~~ **Resolved: new trano feature branches are allowed**
   (`feature/mpc-systems`, `feature/district-models`).
6. ~~Price data~~ **Resolved: committed Belgian CSV samples of a few hundred kB, ENTSO-E as the
   documented download path, committed caches up to ~10 MB, larger via a download script.**
7. ~~Plant model~~ **Resolved: R4C3 perturbed plant against an R3C2 controller by default, a
   switch to "same structure, perturbed parameters"; the tutorials keep the Trano prior and show
   the cost of mismatch (calibration stays an optional pointer to the existing tutorial 02).**
8. ~~Period and location~~ **Resolved: Uccle TMY weather, January week for heating, July week
   for the office.**
9. ~~YAML naming~~ **Resolved: `chiller`, `battery`, `ev_charger` under `systems`; a
   `district:` top-level list of building files for Day 8.**

## 7. Risks and mitigations

* Rumoca accepts only flat explicit ODEs: all system equations are inlined macros, guarded by
  the extended `test_mpc_rumoca.py`; the COP polynomial is kept in a bounded region by the
  input and state bounds of the MPC.
* Bilinear terms (`COP(Th) * Pel`) make the MPC non-convex for R4C3 controllers: IPOPT handles
  it with the warm start; the district ADMM uses the R3C2 controller (COP in `TOut` only, linear
  dynamics, convex problem) and says so.
* 15-min closed loops are slow (49 s / 2 days): cache them, run 1 h steps live.
* Regulatory facts move: every claim carries a verification date and a status flag.
