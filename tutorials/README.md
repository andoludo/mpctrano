# Interactive tutorials: MPC and energy flexibility of buildings

Hands-on notebooks on demand-side flexibility with economic model predictive control (MPC), from
a Trano YAML building description to flexibility indicators. The plan and the status of the
series are in [`PLAN.md`](PLAN.md); the verified literature and regulatory sources are in
[`references.md`](references.md).

| Notebook | Topic | Status |
|---|---|---|
| `00_setup_check` | toolchain check, first FastModel from a YAML, one MPC plan | ready |
| `01_house_heat_pump` | single-family house, heat pump + floor heating (reference implementation) | planned |
| `02_flexibility_use_cases` | implicit / explicit flexibility, capacity tariff, envelope, negative prices | planned |
| `03_apartment_block` | central heat pump, radiators, DHW tank | planned (needs the DHW tank in FastModel) |
| `04_office_cooling` | AHU + chiller, occupancy schedule, summer week | planned (needs cooling in FastModel) |
| `05_pv_battery_ev` | PV, battery and EV charger next to the thermal storage | planned (needs battery and EV in FastModel) |
| `06_district_coordination` | 3 to 5 buildings behind one feeder: independent, centralised, ADMM | planned |
| `07_grid_perspective` | value of the flexibility for the TSO and the DSO, market and regulation | planned |

The script tutorials of the package itself (calibration, state estimation, tariffs) are in
[`package/`](package/).

## Prerequisites

* Python 3.11 to 3.13 and [uv](https://docs.astral.sh/uv/).
* Everything else is a Python package: `trano` (YAML to Modelica, its `FastModel` library),
  `rumoca` (Modelica to CasADi), `casadi` (bundles IPOPT), the notebook stack. No Modelica
  compiler, no FMU and no Docker is needed for the notebooks.
* A Rust toolchain ([rustup](https://rustup.rs)): the FastModel models need rumoca 0.10.3 (rumoca
  main up to commit `c74cddf`), not on PyPI yet, which `uv sync` builds from source (about
  10 minutes the first time, cached afterwards).

```bash
git clone https://github.com/andoludo/mpctrano && cd mpctrano
uv sync                      # rumoca built from source; dev and docs groups (jupyterlab, ipywidgets, jupytext, nbmake)
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

## The building of the tutorials

The tutorials use `tests/models/single_zone_air_water_heat_pump.yaml` (from the Trano test suite: a
single-zone house heated by an air-to-water heat pump with its buffer, a mixing valve and a
radiator) and the Uccle weather file next to it. Trano generates its models when a notebook starts
(about 10 s); `mpctrano generate house.yaml --closed-loop` writes them to disk.

## Price data

Real day-ahead prices are cached in `mpctrano/data/prices/` (see `provenance.md` there) and loaded
with `PriceSignal.from_cached(name)`; no API key is needed to run the notebooks. To refresh or
extend the samples:

```bash
uv run python tutorials/scripts/fetch_prices.py energy-charts 2024-01-01 2025-01-01 --name be_day_ahead_2024
ENTSOE_TOKEN=... uv run python tutorials/scripts/fetch_prices.py entsoe 2025-10-01 2025-11-01 --name be_day_ahead_2025_10
```

## Troubleshooting

* **Linux and macOS**: `uv sync` is enough once Rust is installed. If the rumoca build fails,
  check `cargo --version`; rumoca pins its toolchain in `rust-toolchain.toml` and rustup installs
  it on the first build. The setup-check notebook stops with a message when the installed rumoca
  cannot export the FastModel models.
* **Windows**: use PowerShell or the Windows Terminal with the MSVC build tools for Rust; `make` is
  not available by default, run the commands of the `makefile` directly (`uv run pytest --nbmake
  ...`). Long paths: enable `git config core.longpaths true` before cloning.
* **IPOPT is slow or fails**: the notebooks fix the IPOPT tolerances and seeds; reduce the horizon
  or the number of days with the widgets, or set `MPCTRANO_QUICK=1` to run the short versions.
* **No internet**: once installed, everything runs offline (Trano generates the models locally, the
  prices are cached); only `fetch_prices.py` needs the network.
