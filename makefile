.PHONY: install linting tests tests-all tutorials

install:
	uv sync

linting:
	uv run ruff format .
	uv run ruff check --fix --show-fixes --exit-non-zero-on-fix
	uv run mypy

tests:
	uv run pytest -m "not slow"

tests-all:
	uv run pytest

tutorials:
	for script in tutorials/0*.py; do uv run python $$script; done
