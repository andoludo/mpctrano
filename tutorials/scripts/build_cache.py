"""Pre-compute the closed-loop runs the notebooks read from ``tutorials/cache/``.

    uv run python tutorials/scripts/build_cache.py            # every study
    uv run python tutorials/scripts/build_cache.py --force    # recompute

The runs are deterministic (fixed seeds and solver tolerances); the cache is committed so that
the notebooks and the CI never pay the closed-loop cost. Runtime: about 5 s per 3-day run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tutorials import _house

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(force: bool = typer.Option(False, help="Recompute the runs already cached")) -> None:
    started = time.perf_counter()
    for label, parts in _house.runs().items():
        key = _house.CACHE.key(**parts)
        if _house.CACHE.has(key) and not force:
            continue
        run_started = time.perf_counter()
        result = _house.CACHE.run(lambda: _house.compute(**parts), force=force, **parts)  # noqa: B023
        typer.echo(f"{label}: {time.perf_counter() - run_started:.1f} s, cost {result.metrics.total_cost:.2f} EUR")
    typer.echo(f"house: {len(_house.runs())} runs cached in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    app()
