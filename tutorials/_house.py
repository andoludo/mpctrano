"""Study definition of the house tutorials: variants, price signals, comfort bands and the cached runs.

Shared by `01_house_heat_pump` and `tutorials/scripts/build_cache.py`, so that the notebook reads
exactly the runs the script pre-computes.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from mpctrano import Mismatch, run_scenario
from mpctrano.notebook import ResultCache, TutorialBuilding, comfort_spec, days, price_options, scenario
from mpctrano.simulation import ClosedLoopResult

HERE = Path(__file__).parent
MODELS = HERE / "models" / "house"
CACHE = ResultCache(HERE / "cache" / "house")
START = pd.Timestamp("2023-01-16")
"""A Monday of the typical meteorological year, in the middle of the heating season."""
DAYS = 3.0
"""Period of the cached runs (always 3 days: the cache is read in quick mode too)."""
LIVE_DAYS = days(3.0)
"""Period of the runs the notebooks compute live (one day in quick mode)."""
VARIANTS = ("base", "old", "radiators", "big_heat_pump")
PRICES = ("flat", "day-ahead", "day-ahead + evening spike")
BANDS: dict[str, tuple[float, float, float]] = {
    "tight": (21.0, 23.0, 1.0),
    "standard": (21.0, 24.0, 3.0),
    "wide": (20.0, 25.0, 4.0),
}
MISMATCH = Mismatch(parameter_std=0.2, seed=1, measurement_std=0.1)
PERFECT = Mismatch.none()


def tariffs(start: pd.Timestamp = START, days_: float = DAYS) -> Mapping[str, Any]:
    return price_options(start, days_)


def runs() -> dict[str, dict[str, Any]]:
    """Every cached closed-loop run of the house tutorial, keyed by a label."""
    table: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        for price in PRICES:
            for band in BANDS:
                for mismatch in ("mismatch", "perfect"):
                    label = f"{variant} | {price} | {band} | {mismatch}"
                    table[label] = {
                        "variant": variant,
                        "price": price,
                        "band": band,
                        "mismatch": mismatch,
                        "days": DAYS,
                    }
    return table


def compute(variant: str, price: str, band: str, mismatch: str, days: float) -> ClosedLoopResult:
    """One closed-loop run of the study (3 days hourly: a few seconds)."""
    building = TutorialBuilding(MODELS, variant)
    lower, upper, setback = BANDS[band]
    run = scenario(
        f"{variant} {price} {band} {mismatch}",
        tariffs(START, days)[price],
        comfort_spec(lower, upper, setback),
        START,
        days,
    )
    plant = building.plant(MISMATCH if mismatch == "mismatch" else PERFECT)
    return run_scenario(building.control, run, plant=plant)


def cached(**parts: Any) -> ClosedLoopResult:
    """The cached run of ``parts`` (computed and stored when missing)."""
    return CACHE.run(lambda: compute(**parts), **parts)
