"""Pre-compute the closed-loop runs the notebooks read from ``tutorials/cache/``.

    uv run python tutorials/scripts/build_cache.py                 # every study
    uv run python tutorials/scripts/build_cache.py flexibility     # one study
    uv run python tutorials/scripts/build_cache.py --force         # recompute

The runs are deterministic (fixed seeds and solver tolerances); the cache is committed so that
the notebooks and the CI never pay the closed-loop cost. Runtime: about 18 s per 3-day hourly
run, about 2 min per 3-day 15-minute run.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tutorials import _apartments, _flexibility, _house

STUDIES = {"house": _house, "flexibility": _flexibility, "apartments": _apartments}
app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    study: str | None = typer.Argument(None, help="Study to build (all when omitted): " + ", ".join(STUDIES)),
    force: bool = typer.Option(False, help="Recompute the runs already cached"),
) -> None:
    for name in [study] if study else list(STUDIES):
        module = STUDIES[name]
        started = time.perf_counter()
        for label, parts in module.runs().items():
            if module.CACHE.has(module.CACHE.key(**parts)) and not force:
                continue
            run_started = time.perf_counter()
            result = module.CACHE.run(lambda: module.compute(**parts), force=force, **parts)  # noqa: B023
            typer.echo(f"{label}: {time.perf_counter() - run_started:.1f} s, cost {result.metrics.total_cost:.2f} EUR")
        typer.echo(f"{name}: {len(module.runs())} runs cached in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    app()
