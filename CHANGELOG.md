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

### Changed

- The script tutorials of the package moved from `tutorials/` to `tutorials/package/`; the
  top-level `tutorials/` directory is reserved for the numbered interactive notebooks.
- Ruff no longer lints the generated `.ipynb` files (the paired `.py` scripts are linted).
