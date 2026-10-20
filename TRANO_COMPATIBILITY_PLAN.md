# Plan: make mpctrano work with the current trano

Status: **proposed on 2026-10-20**, against trano 0.57.0 (master `afe8447`), trano-optimization 0.2.0 (PyPI) and
rumoca main at `c74cddf`. The base of the work is the FastModel branch (`ccr-7b65a72f-3ybdrr`, the migration of
mpctrano to trano's FastModel library); `main` still holds the RC-model version, which trano no longer generates.

Execution: one phase per working session, each ending with a pushed commit, so a session can stop after any phase
and the next one resumes from this file. Every step names the files it touches and a "done when" check.

---

## 1. What changed in trano, and what it means for mpctrano

Measured on the single-zone heat pump house of `tests/models`, generated with the trano mpctrano was built against
(branch `ccr-7b65a72f-3ybdrr`, 0.48) and with trano master (0.57), both translated with rumoca `c74cddf`.

| trano change | Effect on mpctrano |
|---|---|
| PR #57: `trano.optimization` moved to the **trano-optimization** package (0.2.0 on PyPI); trano depends on neither it nor rumoca | `ImportError`. The package is the same code (plus an optional `fuel_price` in `receding_horizon`): a rename of the imports. |
| PR #57: the `mpc` library (RC models, `.mpc.json`), the chiller, DHW tank, battery and EV charger elements are removed | Nothing on the FastModel branch uses them. `main` and the tutorial branch (`ccr-c0c43f9d-7vxewf`) do: they cannot be made compatible. |
| PR #54: FastModel is built with the **containers** (`envelope1`, `emission1`, `distribution1`, `production1`) and a data bus | Every state and parameter is prefixed: `space_001.TAir` -> `envelope1.space_001.TAir`, `boiler_001.EEle` -> `production1.boiler_001.EEle`. The order of the states and inputs changes. |
| PR #54: occupancy inputs renamed `occupancy_1.gai.QCon` -> `occupancy_1_gai.QCon` | The forecaster finds no gains (`KeyError`). |
| PR #57: the rumoca export of FastModel is no longer tested in trano's CI (its tests skip without trano-optimization) | Nothing protects mpctrano from the next change of the generated model. |

What did **not** change:

* the physics: the old and new models have the same states, inputs and parameters (27, 12 and 581 for the MPC
  model), the same initial state and parameter values, and the same right-hand side at random points (zero
  difference), for `FastModelMPC` and the closed-loop `FastModel`;
* the trano API mpctrano calls: `convert_network`, `Library.from_configuration("FastModelMPC" | "FastModel")`,
  `Space.occupancy.processed_parameters`, `network.model()`, the `weather_inputs.*` and `<control>_y.y` inputs, the
  weather file line of the runnable model, the role conventions (controls with bounds, `TAir`, meters in kWh);
* rumoca: `c74cddf` still translates the new models, with and without containers.

**`network.model(include_container=False)` still generates a flat FastModel**, numerically identical to the old one
and with the old names of the states and parameters (`space_001.TAir`, `boiler_001.EEle`, `externalwall_0.G[2]`);
only the occupancy inputs keep the new `_gai` name. This is the key decision of the plan (section 2).

Test suite of the FastModel branch against trano master (fast tests, `-m "not slow"`):

| Change | Result |
|---|---|
| imports renamed only | 24 failed, 31 passed |
| + `_gai` in `forecast.channel` and `(^|\.)` in the calibration patterns | 23 failed, 32 passed: every remaining failure is a test with a hard-coded name or control order |

## 2. Decisions

1. **Generate the flat model** (`network.model(include_container=False)`) in `FastModel.from_network`: the names
   of the states, parameters, meters and zones stay those of the elements of the YAML, which the monitoring data,
   the calibration files, the comfort specifications and the tutorials use. Containers add nothing to an ODE.
2. **Also accept containerised models** (`trano create-model house.yaml FastModelMPC` writes them, and trano may
   drop the flat path one day): `FastModel` strips the container prefix `<container>1.` from the names it exposes,
   so both forms give the same names. The flat model is the default; the normalisation is the safety net.
3. **Depend on PyPI packages where they exist**: `trano>=0.57`, `trano-optimization>=0.2`. rumoca 0.10.3 is not on
   PyPI: the published requirement is `rumoca>=0.10.2,<0.11` (as trano-optimization does, so that `pip install
   mpctrano` resolves) and `[tool.uv.sources]` keeps the git commit that exports FastModel.
4. **One rumoca commit everywhere**: `c74cddf` (verified on trano master, flat and containerised); trano-optimization
   pins `33c2114` (an ancestor). Align trano-optimization on `c74cddf` at its next release.
5. **No backward compatibility with the RC models**: the FastModel branch already removed them; `main` gets them
   replaced when this work merges (a breaking release).

## 3. Phases

### Phase 0 - Branch and baseline (short)

1. Work on a branch from `ccr-7b65a72f-3ybdrr` (it holds this plan). `main` is its ancestor: no merge needed.
2. The FastModel branch never ran CI (no workflow run for it). The first push of this branch must start the
   pipeline; if it does not, check the Actions settings of the repository before going further.

Done when: the pipeline runs on the branch (red is expected until phase 4).

### Phase 1 - Dependencies

1. `pyproject.toml`: `trano>=0.57,<1`, add `trano-optimization>=0.2,<1`; `rumoca>=0.10.2,<0.11` with a comment;
   `[tool.uv.sources]` rumoca at `c74cddf` (unchanged).
2. `uv lock`; check that trano 0.57 pulls trano-simulate 0.2 and nothing that needs OpenModelica at import.
3. `mpctrano/model.py`: `RUMOCA_COMMIT` and the error message unchanged; `rumoca_exports_fast_models` unchanged.

Done when: `uv sync` builds, `python -c "import trano, trano_optimization, rumoca"` works, `uv lock --check` passes.

### Phase 2 - trano-optimization imports

1. `trano.optimization` -> `trano_optimization` in `mpctrano/model.py`, `mpc.py`, `estimation.py`,
   `calibration.py` (imports and the cross-references of the docstrings).
2. mypy: add `trano_optimization` to the `ignore_missing_imports` overrides only if it is not typed (it ships
   `py.typed`: it should not be needed).

Done when: `make linting` passes and the test modules import.

### Phase 3 - Model generation and names

1. `FastModel.from_network`: `network.model(include_container=False)`. Keep `from_yaml` and its `library`
   argument as they are.
2. Name normalisation for containerised sources (`from_source` / `from_file`):
   * a function `strip_containers(name)` removing a leading `<container>1.` where `<container>` is one of trano's
     `ContainerTypes` (`envelope`, `emission`, `distribution`, `production`, `ventilation`, `bus`, `solar`);
     trano builds these paths as `f"{container_type}1.{element.name}"` (`trano/elements/containers.py`);
   * applied to the state, input and parameter names of the `ModelDescription` (as `with_temperature_units`
     already rewrites the states), before `Roles.infer`; fail with a `ModelError` if two names collide;
   * the CasADi function is positional, so only the description changes.
3. `forecast.channel`: accept `<occupancy>_gai.<channel>` (new) besides `<occupancy>.gai.<channel>` (old files).
4. `calibration.DEFAULT_GROUPS` / `OPTIONAL_GROUPS`: `(^|\.)` instead of `^`, so the groups match with or without
   a prefix (defensive, as the normalisation should make it unnecessary).
5. `data.py`, `forecast.py`, `model.py`: docstrings with the new occupancy name.

Done when: for the heat pump house, `FastModel.from_yaml` and `FastModel.from_file` of the containerised package
written by trano give the same `state_names`, `input_names`, `parameter_names`, roles and `simulate` trajectory.

### Phase 4 - Tests

1. Hard-coded names (files by number of occurrences): `test_calibration.py` (15), `test_model.py` (11),
   `test_data.py` (7), `test_scenarios.py` (5), `test_tariff_comfort.py`, `test_simulation.py`, `test_mpc.py` (4
   each), `test_forecast.py`, `test_estimation.py` (3 each), `test_project_cli.py` (1). With the flat model the
   state and parameter names do not change: only `occupancy_1.gai.*` and the **order** of the controls do. Compare
   sets, or look names up, instead of lists in a fixed order.
2. New tests:
   * the flat and containerised models of the heat pump house have the same names and the same right-hand side;
   * `channel` on both occupancy names;
   * a model built with the current trano exposes `space_001.TAir`, `boiler_001.EEle` and the three controls (a
     contract test that fails when trano renames again).
3. Run the slow tests (`make tests-all`): the closed loops and the calibration over several days.

Done when: `make linting`, `make tests-all` pass locally.

### Phase 5 - Tutorials, notebook and documentation

1. `tutorials/package/01..06`: column names (`occupancy_1_gai.*`), outputs checked by running `make tutorials`.
2. `tutorials/00_setup_check.py` / `.ipynb`: the toolchain check names trano-optimization; `make sync-notebooks`.
3. `docs/data.md`, `README.md`, `tutorials/README.md`: the occupancy column name, the install instructions (PyPI
   packages, rumoca from git), the trano version.
4. `tutorials/PLAN.md`: section 0 points to this plan for the trano version; `CHANGELOG.md`: the breaking changes
   (FastModel only, trano >= 0.57, trano-optimization, occupancy column names).

Done when: `make tutorials` and `make notebooks` (quick mode) pass.

### Phase 6 - CI

1. `pipeline.yaml`: `uv sync` builds rumoca from git, which needs Rust (rumoca pins its nightly toolchain, which
   rustup installs from its `rust-toolchain` file; `ubuntu-latest` has rustup). Cache the uv build directory so the
   rumoca build (several minutes) runs once per lockfile.
2. The jobs: linting -> tests (`make tests-all`) -> notebooks (`make notebooks`), as on `main`.

Done when: the pipeline is green on the branch.

### Phase 7 - Merge and release

1. PR from the branch to `main` (breaking: the RC models disappear, FastModel only). Its description lists the
   breaking changes and the checks of the phases above.
2. Before merging: `PYPI_API_TOKEN` and `ACCESS_TOKEN` are set; the version bump will be 0.1.0 -> 0.2.0.
3. After the release: `pip install mpctrano` in a clean environment resolves (rumoca 0.10.2 from PyPI), and the
   README tells how to get rumoca 0.10.3 for FastModel.

Done when: mpctrano 0.2.0 is on PyPI and installs.

## 4. Changes asked of trano and trano-optimization (in parallel)

| # | Repository | Change | Why |
|---|---|---|---|
| T1 | trano | A CI job installing trano-optimization and rumoca `c74cddf` that runs the FastModel export tests (`test_rumoca_translates_the_model`, `..._closed_loop_model`), with and without containers | Since PR #57 they always skip: PR #54 renamed every variable without a failing test. |
| T2 | trano | Keep `network.model(include_container=False)` for FastModel, covered by a test (same states as the containerised model, as PR #54 checked by hand) | mpctrano relies on it (decision 1). |
| T3 | trano | Merge the single-zone air-to-water heat pump house (`tests/models/single_zone_air_water_heat_pump.yaml`, branch `ccr-7b65a72f-3ybdrr`, one commit on top of an old master) | The reference building of mpctrano; mpctrano keeps a copy, but trano's tests should cover it. |
| T4 | trano-optimization | Pin rumoca `c74cddf` (as mpctrano), release 0.3.0 | One rumoca commit across the packages. |
| T5 | trano (later) | Write the roles of the MPC model next to it (zones and their comfort state, controls, meters and carriers, occupancy and gain inputs), e.g. in `<name>.description.json` | mpctrano would read them instead of inferring them from naming conventions. |

## 5. Risks

* **trano drops the flat FastModel**: phase 3 step 2 keeps mpctrano working on containerised models; T2 makes the
  dependency explicit.
* **rumoca**: commits after `c74cddf` refuse the export of FastModel (`residual_equations`); a rumoca release with
  the export would let the git pin go. Watch rumoca releases.
* **Next renames in trano**: T1 and the contract test of phase 4 catch them.
* **Out of scope here**: the tutorial work of Days 2 to 6 (`ccr-c0c43f9d-7vxewf`: notebooks 01 to 03, flexibility
  envelope, capacity tariff, apartment block) is built on the RC models; it is ported to FastModel by phases A and
  B of `tutorials/PLAN.md` once this plan is done. The DHW tank, battery, EV charger and cooling no longer exist in
  trano: their notebooks wait for FastModel components (phase C of `tutorials/PLAN.md`).
