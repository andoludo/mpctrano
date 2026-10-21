.PHONY: install linting tests tests-all tutorials sync-notebooks notebooks cache docs

install:
	uv sync --group dev --group docs
	uv run pre-commit install

linting:
	uv run ruff format .
	uv run ruff check --fix --show-fixes --exit-non-zero-on-fix
	uv run mypy

tests:
	uv run pytest -m "not slow"

tests-all:
	uv run pytest

tutorials:
	for script in tutorials/package/0*.py; do uv run python $$script; done

# The tutorial notebooks are jupytext pairs: tutorials/NN_name.py (percent script, the source
# that is reviewed and linted) <-> tutorials/NN_name.ipynb (what Jupyter opens).
sync-notebooks:
	uv run jupytext --sync tutorials/[0-9][0-9]_*.py

# Execute every notebook top to bottom (quick mode: short periods, cached results).
notebooks: sync-notebooks
	MPCTRANO_QUICK=1 uv run pytest --nbmake --nbmake-timeout=1800 -p no:cacheprovider tutorials/[0-9][0-9]_*.ipynb

# Build the documentation site, with the executed tutorial notebooks (their outputs are committed).
docs:
	mkdir -p docs/tutorials
	cp tutorials/[0-9][0-9]_*.ipynb docs/tutorials/
	uv run mkdocs build

# Rebuild the cached closed-loop runs of the notebooks (hours: run it in the background).
cache:
	uv run python tutorials/scripts/build_cache.py
