"""Pre-compute the closed-loop runs the notebooks read from ``tutorials/cache/``.

    uv run python tutorials/scripts/build_cache.py                  # every study, three runs at a time
    uv run python tutorials/scripts/build_cache.py flexibility      # one study
    uv run python tutorials/scripts/build_cache.py --workers 1      # one run at a time
    uv run python tutorials/scripts/build_cache.py --force          # recompute

The runs are deterministic (fixed seeds and solver tolerances); the cache is committed so that the notebooks and
the CI never pay the closed-loop cost. Runtime: about ten minutes per two-day hourly run of the house (48 MPC solves
and moving horizon estimations), about an hour and a half for the house and flexibility studies on three workers;
the office closed loops take about an hour each, the PV house ones 10 to 30 minutes.
"""

from __future__ import annotations

import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import typer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

STUDIES = ("house", "flexibility", "office", "pv", "calibration")
app = typer.Typer(add_completion=False, help=__doc__)


def _module(name: str) -> Any:
    sys.path.insert(0, str(ROOT))
    from tutorials import _calibration, _flexibility, _house, _office, _pv

    return {"house": _house, "flexibility": _flexibility, "office": _office, "pv": _pv, "calibration": _calibration}[
        name
    ]


def _build(study: str, label: str, parts: dict[str, Any], force: bool) -> str:
    """Compute and cache one run (in a worker process)."""
    module = _module(study)
    started = time.perf_counter()
    result = module.CACHE.run(lambda: module.compute(**parts), force=force, **parts)
    return (
        f"{study} | {label}: {time.perf_counter() - started:.0f} s, cost {result.metrics.total_cost:.2f} EUR, "
        f"{result.metrics.comfort_violation_kh:.1f} K.h, {result.failures} failed solves"
    )


def _calibration(force: bool) -> str:
    started = time.perf_counter()
    module = _module("calibration")
    module.groups(force=force)
    module.known_infiltration_groups(force=force)
    module.recorded(force=force)
    module.calibrate(force=force, known_infiltration=True)
    _, result = module.calibrate(force=force)
    return f"calibration: {time.perf_counter() - started:.0f} s, {result.status} in {result.iterations} iterations"


def _envelope(at: str, force: bool) -> str:
    started = time.perf_counter()
    _module("flexibility").envelope(at, force=force)
    return f"flexibility | envelope at {at}: {time.perf_counter() - started:.0f} s"


@app.command()
def main(
    study: str | None = typer.Argument(None, help="Study to build (all when omitted): " + ", ".join(STUDIES)),
    workers: int = typer.Option(3, min=1, help="Runs computed at the same time (one process each)"),
    force: bool = typer.Option(False, help="Recompute the runs already cached"),
) -> None:
    names = [study] if study else list(STUDIES)
    pending: list[list[tuple[str, str, dict[str, Any]]]] = [[], []]
    for name in names:
        module = _module(name)
        if not hasattr(module, "runs"):
            continue
        for label, parts in module.runs().items():
            if force or not module.CACHE.has(module.CACHE.key(**parts)):
                # an activation response starts from the cached baseline: second phase
                late = parts.get("study") == "activation" and parts.get("activation") != "baseline"
                pending[late].append((name, label, parts))
    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for phase, jobs in enumerate(pending):
            futures = [pool.submit(_build, *job, force) for job in jobs]
            if phase == 0 and "flexibility" in names:
                futures += [pool.submit(_envelope, at, force) for at in _module("flexibility").ENVELOPE_TIMES]
            if phase == 0 and "calibration" in names:
                futures.append(pool.submit(_calibration, force))
            for future in as_completed(futures):
                typer.echo(future.result(), err=False)
    typer.echo(f"done in {time.perf_counter() - started:.0f} s")


if __name__ == "__main__":
    app()
