# Interactive tutorials: MPC and energy flexibility of buildings

Hands-on notebooks on demand-side flexibility with economic model predictive control (MPC), from
a Trano YAML building description to flexibility indicators. The plan and the status of the
series are in [`PLAN.md`](PLAN.md); the verified literature and regulatory sources are in
[`references.md`](references.md).

| Notebook | Topic | Status |
|---|---|---|
| `00_setup_check` | toolchain check, first CasADi model from a YAML | ready |
| `01_house_heat_pump` | single-family house, heat pump + floor heating (reference implementation) | ready |
| `02_flexibility_use_cases` | implicit / explicit flexibility, capacity tariff, envelope, negative prices | ready |
| `03_apartment_block` | central heat pump, radiators, DHW tank | planned |
| `04_office_cooling` | AHU + chiller, occupancy schedule, summer week | planned |
| `05_pv_battery_ev` | PV, battery and EV charger next to the thermal storage | planned |
| `06_district_coordination` | 3 to 5 buildings behind one feeder: independent, centralised, ADMM | planned |
| `07_grid_perspective` | value of the flexibility for the TSO and the DSO, market and regulation | planned |

The script tutorials of the package itself (calibration, state estimation, tariffs) are in
[`package/`](package/).

## Prerequisites

* Python 3.11 to 3.13 and [uv](https://docs.astral.sh/uv/).
* Everything else is a Python package: `trano` (YAML to Modelica), `rumoca` (Modelica to CasADi,
  binary wheels for Linux, macOS and Windows), `casadi` (bundles IPOPT), the notebook stack.
  No Modelica compiler, no FMU and no Docker is needed for the notebooks.

```bash
git clone https://github.com/andoludo/mpctrano && cd mpctrano
uv sync                      # dev and docs groups included (jupyterlab, ipywidgets, jupytext, nbmake)
uv run pre-commit install    # optional: ruff, mypy and jupytext sync on every commit
```

## Launch the notebooks

```bash
uv run jupyter lab tutorials/
```

Open `00_setup_check.ipynb` first. Each notebook is paired with a percent-format script
(`NN_name.py`, the reviewed and linted source) through [jupytext](https://jupytext.readthedocs.io):
edit either file and run `make sync-notebooks` (or let the pre-commit hook do it). Widgets need
`ipywidgets` (in the `notebook` extra and the dev group); plots use plotly.

Execute every notebook top to bottom, as the CI does (quick mode, cached results):

```bash
make notebooks
```

## Regenerate the models from the YAML files

The building descriptions are in `buildings/` and the generated Modelica models and interfaces in
`models/` (committed, so that the notebooks never need to call Trano):

```bash
uv run python tutorials/scripts/generate_models.py          # all typologies
uv run python tutorials/scripts/generate_models.py house    # one typology
```

## Cached results

The closed-loop runs the notebooks show are pre-computed and committed in `tutorials/cache/`
(parquet frames keyed by the run parameters, see `mpctrano.notebook.ResultCache`). The widgets read
them in milliseconds; the **Recompute** button runs the closed loop again. To rebuild the cache after
a change of the models or of the study definitions (`tutorials/_house.py`, `tutorials/_flexibility.py`):

```bash
uv run python tutorials/scripts/build_cache.py                 # only the missing runs of every study
uv run python tutorials/scripts/build_cache.py flexibility     # one study
uv run python tutorials/scripts/build_cache.py --force         # everything (about 20 min per study)
```

## Price data

Real day-ahead prices are cached in `mpctrano/data/prices/` (see `provenance.md` there) and loaded
with `PriceSignal.from_cached(name)`; no API key is needed to run the notebooks. To refresh or
extend the samples:

```bash
uv run python tutorials/scripts/fetch_prices.py energy-charts 2024-01-01 2025-01-01 --name be_day_ahead_2024
ENTSOE_TOKEN=... uv run python tutorials/scripts/fetch_prices.py entsoe 2025-10-01 2025-11-01 --name be_day_ahead_2025_10
```

## Troubleshooting

* **Linux and macOS**: `uv sync` is enough. If `import rumoca` fails, check that the platform has a
  wheel (`uv pip install rumoca` prints the candidates); rumoca 0.10 ships manylinux x86_64 and
  aarch64, macOS arm64 and x86_64 and Windows x86_64 wheels.
* **Windows**: use PowerShell or the Windows Terminal; `make` is not available by default, run the
  commands of the `makefile` directly (`uv run pytest --nbmake ...`). Long paths: enable
  `git config core.longpaths true` before cloning.
* **IPOPT is slow or fails**: the notebooks fix the IPOPT tolerances and seeds; reduce the horizon
  or the number of days with the widgets, or set `MPCTRANO_QUICK=1` to run the short versions.
* **No internet**: everything runs offline from the cached models and prices; only
  `fetch_prices.py` and the model regeneration need Trano (local) and the network (prices).
