# Interactive tutorials: MPC and energy flexibility of buildings

Hands-on notebooks on demand-side flexibility with economic model predictive control (MPC), from
a Trano YAML building description to flexibility indicators. The plan and the status of the
series are in [`PLAN.md`](PLAN.md); the verified literature and regulatory sources are in
[`references.md`](references.md).

| Notebook | Topic | Status |
|---|---|---|
| `00_setup_check` | toolchain check, first BuildingModel from a YAML, one MPC plan | ready |
| `01_house_heat_pump` | a house with a heat pump: pipeline, plan widget, closed loop vs the controllers of the YAML, estimation | ready |
| `02_flexibility_use_cases` | implicit / explicit flexibility, capacity tariff, envelope, negative prices with PV | ready |
| `03_calibration_from_yaml` | parameters declared in the YAML, calibration of the closed-loop model on noisy data, how IPOPT solves it, before and after | ready |
| `04_office_cooling` | an office in summer: ideal cooling at an EER, occupancy, capacity charge | ready |
| `05_pv_battery_ev` | PV, a battery and a vehicle at the meter next to the thermal storage | ready |
| `06_your_building` | edit the YAML in a cell, regenerate the models, plan, MPC vs the controllers of the YAML | ready |
| `07_district_coordination` | 3 to 5 buildings behind one feeder: independent, centralised, ADMM | planned |
| `08_grid_perspective` | value of the flexibility for the TSO and the DSO, market and regulation | planned |
| `09_apartment_block` | central heat pump, radiators, DHW tank | waiting for the DHW tank in FastSmoothBuildings (see `PLAN.md`) |

The executed notebooks are published on the documentation site
([andoludo.github.io/mpctrano](https://andoludo.github.io/mpctrano/), section *Tutorials*): the figures are
interactive there, the widgets of the live sections are not (they need a kernel). To play with them without
installing anything, open a notebook in Google Colab with its *Open in Colab* badge: the first cell clones the
repository and installs mpctrano and the rumoca wheel (a minute; `06_your_building` is the one to edit).

The script tutorials of the package itself (calibration, state estimation, tariffs) are in
[`package/`](package/).

## Prerequisites

* Python 3.11 to 3.13 and [uv](https://docs.astral.sh/uv/).
* Everything else is a Python package: `trano` (YAML to Modelica, its `FastSmoothBuildings` library),
  `rumoca` (Modelica to CasADi), `casadi` (bundles IPOPT), the notebook stack. No Modelica
  compiler, no FMU and no Docker is needed for the notebooks.

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

## The buildings and the cached runs

The buildings are YAML files in `buildings/`: the single-zone heat pump house of the Trano test suite
(`house/base.yaml`), its `old` (uninsulated, 55 degC radiator) and `pv` (20 m2 of PV) variants, and a
two-zone office with ideal emitters (`office/base.yaml`), with the Uccle weather file next to them.
Trano generates the models when a notebook starts (10 to 30 s per building).

The closed loops the notebooks read are cached in `cache/` (parquet frames keyed by the parameters
of the run, committed): the studies `_house.py`, `_flexibility.py`, `_office.py` and `_pv.py`
define them and `scripts/build_cache.py` computes the missing ones.

```bash
uv run python tutorials/scripts/build_cache.py              # every study, three runs at a time (hours)
uv run python tutorials/scripts/build_cache.py office       # one study
uv run python tutorials/scripts/build_cache.py --force      # recompute
```

The runs are deterministic (fixed seeds and solver tolerances), so a rebuilt cache reproduces the
committed one up to the solver's tolerance. `06_your_building` reads no cache: it writes the YAML of its cell to
`tutorials/output/your_building/` and computes everything live.

## Price data

The studies use the synthetic day-ahead signal of `PriceSignal.synthetic` unless a real sample is
cached in `mpctrano/data/prices/` (see `provenance.md` there; none is committed yet, the data
providers were not reachable from the machine that built the cache). To download one, with
network access:

```bash
uv run python tutorials/scripts/fetch_prices.py energy-charts 2024-01-01 2025-01-01 --name be_day_ahead_2024
ENTSOE_TOKEN=... uv run python tutorials/scripts/fetch_prices.py entsoe 2025-10-01 2025-11-01 --name be_day_ahead_2025_10
```

`price_options` picks `be_day_ahead_2024` up automatically; rebuild the cache afterwards.

## Troubleshooting

* **Linux, macOS and Windows**: `uv sync` (or `pip install mpctrano`) installs everything as wheels. The
  setup-check notebook stops with the upgrade command when the installed rumoca cannot export the
  FastSmoothBuildings models (rumoca before 0.10.3, or FastSmoothBuildings before 0.6.0).
* **Windows**: `make` is not available by default, run the commands of the `makefile` directly
  (`uv run pytest --nbmake ...`). Long paths: enable `git config core.longpaths true` before cloning.
* **IPOPT is slow or fails**: the notebooks fix the IPOPT tolerances and seeds; reduce the horizon
  or the number of days with the widgets, or set `MPCTRANO_QUICK=1` to run the short versions.
* **No internet**: once installed, everything runs offline (Trano generates the models locally, the
  prices are cached); only `fetch_prices.py` needs the network.
