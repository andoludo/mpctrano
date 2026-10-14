.PHONY: install rumoca-fastmodel linting tests tests-all tutorials sync-notebooks notebooks

install:
	uv sync --group dev --group docs
	uv run pre-commit install

# FastModel (trano's FastModelMPC library) needs a rumoca that exports connected Modelica models to CasADi:
# rumoca main up to c74cddf (later commits stop at residual equations), not released on PyPI yet. This builds it
# from source (Rust toolchain needed, about 10 minutes). `uv sync` puts the PyPI release back: run the tests with
# `uv run --no-sync pytest` afterwards. The FastModel tests are skipped without it.
RUMOCA_FASTMODEL = git+https://github.com/CogniPilot/rumoca@c74cddff6d7669fc2c3603d43529ab62e0964072\#subdirectory=crates/rumoca-bind-python

rumoca-fastmodel:
	uv pip install --reinstall-package rumoca "rumoca @ $(RUMOCA_FASTMODEL)"

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
