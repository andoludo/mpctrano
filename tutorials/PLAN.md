# Plan: interactive MPC and energy-flexibility tutorials in mpctrano

Status: **draft, waiting for approval** (no code written yet).
Execution: one phase per working session (a "day" below), each ending with a pushed commit and
a short report. Every step lists the files it touches and a "done when" check, so a session can
stop after any step and the next one resumes from this file.

Branches: mpctrano on `ccr-c0c43f9d-7vxewf`; trano on `ccr-c0c43f9d-7vxewf` (district work only,
see open question 6). trano `master` is never touched.

---

## 0. What the exploration found

### 0.1 The pipeline already exists, for the envelope only

```
trano YAML --(trano "mpc" library)--> flat RC Modelica (building_mpc) + <stem>.mpc.json interface
          --(rumoca)--> CasADi ODE xdot = f(x, u, d, p), p symbolic
          --> mpctrano: RK4 discretisation, Opti/IPOPT economic MPC, Kalman/MHE, closed loop
```

* `mpctrano.model.RCModel`: the Rumoca export with symbolic parameters, RK4 `discrete()`, exact
  affine form for the Kalman filter. Zone models R1C1 / R3C2 / R4C3 (emitter state) / ISO13790.
* `mpctrano.mpc.ModelPredictiveController`: single `casadi.Opti` built once, warm started.
  Objective = price x thermal energy + capacity charge x horizon peak + soft comfort band
  (L1 + L2) + soft power cap + optional ramp penalty.
* `mpctrano.simulation`: `ModelPlant` (same RC structure, other parameters, sensor noise) and
  `run_closed_loop` (estimator -> MPC -> plant). `synthetic.perturb` gives the plant mismatch,
  `forecast.ForecastNoise` gives lead-time-growing weather/gains forecast errors.
* `mpctrano.tariff`: flat, time-of-use, dynamic series (hourly or 15-min, held between samples),
  demand-response events (price multiplier and/or power cap), capacity charge on the horizon peak.
* `mpctrano.flexibility`: `compare`, `value_of_flexibility` (price-aware vs price-agnostic),
  `shifted_energy` (half L1 distance), `price_response`.
* `mpctrano.scenarios`: Pydantic `Scenario` + catalogue (flat, time_of_use, dynamic_price,
  demand_response, peak_shaving, wide_comfort_band, imperfect_forecast).
* Tooling in place: uv, ruff, mypy strict, pytest (`slow` marker), mkdocs, CI (lint -> tests ->
  minor bump + tag on main -> PyPI). Tutorials are plain scripts under `tutorials/`, executed by
  `tests/test_tutorials.py` in quick mode.

### 0.2 Gaps against the task

| Needed by the task | Current state |
|---|---|
| Electricity at the meter (heat pump COP, cooling EER, PV, battery, EV, DHW tank) | None. The only control input is thermal heat `<zone>_QHea` [W]; costs and peaks are thermal. The trano YAML schema has heat pumps, PV and AHUs for the Buildings/IDEAS libraries, but the `mpc` library ignores systems entirely. |
| Explicit flexibility (activation request, baseline, rebound) | Only `DemandResponseEvent` (price multiplier / power cap). No baseline, no rebound metric. |
| Flexibility function / envelope, Annex 67 / BOPTEST KPIs, shifting efficiency | Only `shifted_energy`, `low_price_share`, comfort Kh, horizon peak. |
| Capacity tariff on the monthly quarter-hour peak (Flemish) | Capacity charge applies to the horizon peak only (no memory of the running monthly peak). |
| Real price data (ENTSO-E / Elia) | Only `synthetic_day_ahead`. |
| District (several buildings, feeder limit, coordination) | One `RCBuilding` per trano network; nothing multi-building in trano or mpctrano. |
| Notebooks, widgets, jupytext, nbmake, pre-commit, CHANGELOG, Windows notes | None. |
| Reduced-order vs higher-order control model | Available for free: R3C2 controller against an R4C3 / ISO13790 plant (different structure, not only perturbed parameters). |

### 0.3 Measured runtimes (this container, three-zone test building, R3C2)

| Step | Time |
|---|---|
| trano generate + Rumoca translate (first call) | 4.6 s |
| Rumoca translate only (R4C3) | 0.4 s |
| One MPC solve, horizon 24 h, step 1 h | ~110 ms |
| One MPC solve, horizon 24 h, step 15 min (96 steps) | ~240 ms |
| Closed loop, 2 days, step 1 h | 6 s |
| Closed loop, 2 days, step 15 min | 49 s |

Consequence for the "widgets update in < 5 s" criterion: live widgets re-solve a **single
open-loop MPC plan** (instant), while **closed-loop** results come from a pre-computed cache over
the widget grid, with an explicit "recompute" button (progress bar, documented runtime).

---

## 1. Design decisions (recommendations; confirm or change in section 6)

1. **Energy-system layer in CasADi inside mpctrano, envelope stays in trano/Rumoca.**
   New module `mpctrano.systems`: typed components composed with the RC model into a
   `BuildingEnergyModel` (thermal RC states from Rumoca + extra linear states for the DHW tank
   and the battery, plus the electrical balance at the meter as algebra). The MPC objective is
   then written on grid import/export, which is what every flexibility KPI needs.
   Why not render these in the trano `mpc` Modelica library: Rumoca exports `xdot = f(x,u,p)`
   only (no outputs, no algebraic variables, no events), so COP curves, PV balance, import/export
   split and the capacity-tariff memory could not live there anyway; only the two storage states
   could, and splitting one system across the two repos is not worth it. The YAML -> Modelica ->
   Rumoca path stays the single source of truth for the building physics and its calibration.
2. **Control model vs plant, always separate objects**: `Scenario` gains a `Mismatch` config
   (parameter perturbation std + seed, forecast noise, measurement noise, optional plant RC type).
   Default plant for every tutorial = R4C3 structure with perturbed parameters; controller = R3C2.
   This is the "reduced-order vs full model" discussion without an FMU.
3. **Prices**: a `PriceSignal` Pydantic model (day-ahead hourly, 15-min, synthetic spike,
   negative hours, grid fees + taxes added on top) backed by a cached CSV sample of real BE
   day-ahead prices; no API key needed to run anything.
4. **Interactivity**: `ipywidgets` + `plotly` `FigureWidget`; widget glue lives in a new
   `mpctrano.notebook` module (optional extra `notebook`), cells only wire widgets to library
   calls. Notebooks are jupytext-paired `.py` (percent format, diffable) and executed by `nbmake`
   in CI with `MPCTRANO_QUICK=1`.
5. **Phase order change**: flexibility metrics and use cases (task phase 4) are done right after
   the reference typology (phase 2) so the other typologies reuse them.

---

## 2. Typologies (3 + 1 optional)

| # | Typology | Why it contrasts | Model |
|---|---|---|---|
| 1 | Single-family house, air-water heat pump + floor heating | Reference case. High thermal mass, slow emitter, COP strongly tied to supply temperature: textbook load shifting and pre-heating, visible rebound. | 2 zones (day / night), R4C3 plant, floor-heating emitter settings, HP with COP(TOut, Tsup). Variants: envelope (old / renovated / passive), mass, HP size, + PV. |
| 2 | Apartment block, central heat pump + radiators + DHW tank | Lower mass, fast emitters, a second storage (tank) with a hard end-of-day energy demand: shows storage competition and why DHW is the cheapest flexibility. 6 zones, one meter. | 6 zones (3 floors x 2 orientations), R3C2/R4C3, DHW tank 1C model with draw-off profile. |
| 3 | Office, AHU + cooling, weekday occupancy | Cooling instead of heating, strong internal/solar gains, occupancy-driven comfort window, a summer week: pre-cooling against afternoon peaks, shows the TSO/DSO value in summer. | 4 zones (core + 3 facades), chiller EER, AHU ventilation gain; summer period. |
| 4 (optional) | House of #1 + battery + EV charger | Electric storage next to thermal storage: dispatch order, negative prices, PV self-consumption, EV deadline constraints. | Battery and EV as linear SOC states in `mpctrano.systems`. |

All YAMLs live in `tutorials/buildings/`, generated models (`.mo` + `.mpc.json`) in
`tutorials/models/`, both committed (text, tens of kB each).

---

## 3. Target layout

```
mpctrano/
  systems.py        HeatPump, Chiller, Emitter, DHWTank, PV, Battery, EVCharger, Meter, BuildingEnergyModel
  prices.py         PriceSignal, grid-fee/tax stack, loaders for the cached ENTSO-E/Elia samples, synthetic shapes
  flexibility.py    (extended) ActivationRequest, baseline + rebound, FlexibilityFunction, FlexibilityEnvelope, KPI set
  tariff.py         (extended) CapacityTariff with running monthly peak, negative prices
  scenarios.py      (extended) Mismatch, ScenarioStudy (grid of scenarios -> cache)
  district.py       Feeder, IndependentMPC, CentralisedMPC, ADMMCoordinator, Aggregator
  notebook.py       widget factories, ResultCache, runtime guard (optional extra "notebook")
tutorials/
  README.md, PLAN.md, references.md (verified literature + regulation, with URLs and dates)
  buildings/<typology>.yaml (+ variants), models/<typology>/<variant>.{mo,mpc.json}
  data/prices/*.csv (cached samples + provenance.md), cache/*.parquet
  scripts/generate_models.py, scripts/fetch_prices.py, scripts/build_cache.py
  01_house_heat_pump.py/.ipynb ... 07_grid_perspective.py/.ipynb
tests/test_systems.py, test_prices.py, test_flexibility.py, test_district.py, test_notebooks (nbmake)
CHANGELOG.md, .pre-commit-config.yaml, .devcontainer/ (optional)
```

---

## 4. Phases, step by step

### Day 1 - Phase 1: scan, data, tooling (no modelling yet)

1. **Literature and regulation scan** -> `tutorials/references.md`. Each entry: full citation,
   DOI/URL, one line on what it is used for, "verified on <date>". Items: IEA EBC Annex 67 and
   Annex 82 reports; Junker et al. 2018 flexibility function; Stinner et al. 2016 and Reynders
   et al. 2017 flexibility quantification; SRI flexibility domain; BOPTEST KPIs (Blum et al.
   2021); EU Electricity Market Design reform (Reg. 2024/1747, Dir. 2024/1711); network code on
   demand response (status); SDAC 15-min MTU go-live date; Flemish capacity tariff rules
   (Fluvius/VREG); Elia aFRR/mFRR product sheets; GOPACS, NODES, Piclo. Anything not verifiable
   online is marked `[unverified]`.
   Done when: file exists, every line has a working source, open points listed.
2. **Price data** -> `tutorials/scripts/fetch_prices.py` (ENTSO-E with optional token, Elia
   open data without token) and committed samples in `tutorials/data/prices/` (BE day-ahead
   hourly for a full year, one 15-min MTU week, a negative-price spring week) with
   `provenance.md`. Done when: `PriceSignal.from_cached("be_dayahead_2024")` works offline.
3. **Tooling**: add dev deps (`jupyterlab`, `ipywidgets`, `jupytext`, `nbmake`, `pre-commit`,
   `pyarrow`), `notebook` extra, `.pre-commit-config.yaml` (ruff, mypy, jupytext sync),
   `CHANGELOG.md` (Keep a Changelog), CI job `notebooks` (nbmake, quick mode), jupytext config
   in `pyproject.toml`. Done when: `make linting`, `make tests`, `make notebooks` pass on an
   empty placeholder notebook.

### Day 2 - Phase 2a: library work for the reference typology

4. `mpctrano.systems`: `HeatPump` (COP = f(TOut, Tsup), smooth polynomial fitted on a
   manufacturer-style map, max electrical power), `Emitter` (supply temperature as a function of
   load, floor heating vs radiator), `Meter` (import/export split with non-negative variables),
   `BuildingEnergyModel` = RC model + systems, exposing the same `discrete()` contract plus the
   electrical power expression. Tests: COP monotonicity, energy balance, Rumoca model untouched.
5. `ModelPredictiveController` refactor: objective on electrical import/export and optional
   running-peak capacity term; keep the thermal-only path backward compatible (CHANGELOG entry).
   Tests: existing suite green, new tests for the electrical objective.
6. `Mismatch` in `mpctrano.scenarios` + plant builder (R4C3 plant from the same YAML).
7. House YAML + variants (`tutorials/buildings/house*.yaml`), `scripts/generate_models.py`
   (trano -> Modelica -> interface, checks Rumoca translation), models committed.
   Done when: `uv run python tutorials/scripts/generate_models.py --typology house` regenerates
   identical files.

### Day 3 - Phase 2b: reference notebook

8. `mpctrano.notebook`: `ResultCache` (parquet, keyed by scenario hash), widget factories
   (comfort bounds, price signal, horizon / step / weights, building variant), runtime guard.
9. `tutorials/01_house_heat_pump.py` (jupytext) -> `.ipynb`: objectives; pipeline walk-through
   (YAML -> Modelica -> CasADi, inspect states/parameters); open-loop plan widget (live);
   closed loop with mismatch (cache + recompute); state estimation on/off; R3C2 vs R4C3
   controller (solve time vs accuracy); key takeaways; further reading.
10. `scripts/build_cache.py` for the widget grid; nbmake run in CI (quick mode).
    Done when: notebook runs top to bottom in CI, every widget updates in < 5 s from cache.

### Day 4 - Phase 4 (moved up): flexibility use cases and metrics

11. `mpctrano.flexibility` extension: `ActivationRequest` (direction, power, start, duration,
    notice), baseline run and `rebound` metrics (energy, peak, duration), `FlexibilityFunction`
    (step-price response, Junker), `FlexibilityEnvelope` (max up/down power over duration at a
    given comfort/cost penalty, via a sweep of constrained MPC solves), KPI set with references
    (shiftable energy, shifting efficiency, Kh, peak reduction, cost saving, storage efficiency).
12. `mpctrano.tariff`: `CapacityTariff` (monthly quarter-hour peak average, Flemish parameters
    as defaults, verified in step 1), negative prices and export remuneration in `PriceSignal`.
13. `tutorials/02_flexibility_use_cases.py`: implicit (day-ahead vs 15-min), peak shaving under
    capacity tariff, explicit activation with rebound, envelope / function, negative prices + PV
    self-consumption; all with widgets and a KPI table.
    Done when: KPIs reproduce with fixed seeds; tests cover each metric on a toy profile.

### Day 5 - Phase 3a: apartment block

14. `DHWTank` (1C, draw-off profile, legionella/minimum temperature constraint) in
    `mpctrano.systems`; block YAML (6 zones) + variants; `tutorials/03_apartment_block.py`
    (storage competition, DHW as cheapest flexibility, radiators vs floor heating comparison).

### Day 6 - Phase 3b: office (+ optional typology 4)

15. `Chiller` (EER), cooling input (negative heat with its own bounds), ventilation gains,
    weekday occupancy; office YAML; summer-week scenarios; `tutorials/04_office_cooling.py`.
16. Optional: `Battery`, `EVCharger` (SOC states, deadline); `tutorials/05_pv_battery_ev.py`.

### Day 7 - Phase 5: district

17. trano branch: `trano.mpc.district` (`RCDistrict`: list of buildings from several YAMLs or
    one district YAML, feeder/transformer capacity, rendered as one flat `district_mpc` package
    with prefixed zone names, one interface with building grouping), CLI
    `trano create-district`, tests (Rumoca translation of a 5-building district). Report
    the diff. Single-building tutorials keep working with trano main (lazy import, clear error).
18. `mpctrano.district`: `Feeder` constraint, `IndependentMPC` (N controllers, same price),
    `CentralisedMPC` (one Opti), `ADMMCoordinator` (shared feeder constraint, residual history),
    `Aggregator` view (portfolio flexibility envelope); synchronized rebound demonstration.
19. `tutorials/06_district_coordination.py`: 5 house variants behind one feeder; price-only
    synchronisation peak, feeder limit, centralised vs ADMM convergence and solve time.

### Day 8 - Phase 6: grid perspective, docs, CI polish

20. `tutorials/07_grid_perspective.py`: TSO value (balancing, imbalance price exposure, aFRR/mFRR
    activation from tutorial 02) vs DSO value (congestion, peak, capacity tariff, feeder from 06);
    market vs local conflict; regulatory context from `references.md` with status flags.
21. `tutorials/README.md` (prerequisites, `uv sync`, Rumoca install, regenerate models, launch,
    district branch, troubleshooting Linux + Windows), mkdocs pages, optional
    `.devcontainer`/Dockerfile, documented runtimes, final CHANGELOG, CI green on everything.

---

## 5. Acceptance checks run at the end of every day

* `make linting` (ruff format, ruff check, mypy strict) and `make tests` green.
* `uv run pytest --nbmake tutorials/*.ipynb` green with `MPCTRANO_QUICK=1`.
* Deterministic: seeds fixed in configs, IPOPT tolerances fixed, caches reproducible.
* CHANGELOG updated; every new public symbol typed, tested and documented.

---

## 6. Open questions (answers needed before Day 2)

1. **Energy systems as CasADi in mpctrano** (recommended, section 1.1) or rendered in the trano
   `mpc` Modelica library? The second keeps everything in Modelica but needs a Rumoca feature
   check for outputs and splits one system across two repos.
2. **Typologies**: confirm the table of section 2. Should typology 4 (battery + EV) be in scope
   from the start or stay optional? PV alone is folded into the house variants either way.
3. **Dependencies to add** (dev group): `jupyterlab`, `ipywidgets`, `jupytext`, `nbmake`,
   `pre-commit`, `pyarrow`; runtime extra `notebook = [plotly, ipywidgets]`. OK?
4. **Price data**: committed CSV samples of a few hundred kB are fine? Preferred market area BE
   (Elia/EPEX) with ENTSO-E as the documented download path?
5. **Cache size**: committed parquet caches up to ~10 MB total, larger grids via a download
   script from a GitHub release asset. OK?
6. **trano branch name**: the session's designated branch `ccr-c0c43f9d-7vxewf` or a
   `feature/district-models` branch as in your brief? (Pushing elsewhere needs your explicit OK.)
7. **Plant model**: R4C3 perturbed plant against an R3C2 controller as the default mismatch, with
   a widget to switch to "same structure, perturbed parameters"? Any preference for a
   calibration step in the tutorials (the package supports it) or keep the Trano prior and show
   the mismatch cost instead?
8. **Period and location**: Uccle TMY weather (already in the repo), January week for heating,
   July week for the office. OK?

## 7. Risks and mitigations

* Rumoca accepts only flat ODEs without algebra or events: keep all system equations in CasADi
  (decision 1) and guard with the existing `test_mpc_rumoca.py` contract in trano.
* 15-min closed loops are slow (49 s / 2 days): cache them, run 1 h steps live.
* ADMM convergence on a non-convex problem is not guaranteed: the problem is linear dynamics with
  convex costs, so keep COP linearised around the forecast outdoor temperature in the district
  coordination and state it in the notebook.
* Regulatory facts move: every claim carries a verification date and a status flag.
