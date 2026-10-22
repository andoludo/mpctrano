"""Study definition of the house tutorial (01): variants, price signals, comfort bands, estimators, cached runs.

Shared by ``01_house_heat_pump`` and ``tutorials/scripts/build_cache.py``, so that the notebook reads exactly
the runs the script pre-computes in ``tutorials/cache/house/``.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mpctrano import FastModel, MovingHorizonEstimator, run_closed_loop, run_scenario
from mpctrano.mpc import ModelPredictiveController
from mpctrano.notebook import ResultCache, TutorialBuilding, comfort_spec, days, price_options, scenario
from mpctrano.scenarios import Scenario
from mpctrano.simulation import ClosedLoopResult
from mpctrano.tariff import Tariff

HERE = Path(__file__).parent
BUILDINGS = HERE / "buildings" / "house"
CACHE = ResultCache(HERE / "cache" / "house")
START = pd.Timestamp("2023-01-16")
"""A Monday of the typical meteorological year, in the middle of the heating season."""
DAYS = 2.0
"""Period of the cached runs (always two days: the cache is read in quick mode too)."""
LIVE_DAYS = days(1.0, quick=0.25)
"""Period of the runs the notebooks compute live (six hours in quick mode)."""
VARIANTS = ("base", "old")
PRICES = ("flat", "day-ahead")
BANDS: dict[str, tuple[float, float, float]] = {
    "standard": (21.0, 24.0, 3.0),
    "wide": (20.0, 25.0, 4.0),
}
"""Occupied minimum and maximum [degC] and night setback [K] of each comfort band."""
BAND_DEFAULT = BANDS["standard"]


@cache
def building(variant: str = "base") -> TutorialBuilding:
    """The tutorial building of a variant (``tutorials/buildings/house/<variant>.yaml``)."""
    return TutorialBuilding(BUILDINGS / f"{variant}.yaml", start=START)


def tariffs(start: pd.Timestamp = START, days_: float = DAYS) -> Mapping[str, Tariff]:
    return price_options(start, days_)


def study_scenario(variant: str, price: str, band: str, estimator: str, days_: float = DAYS) -> Scenario:
    lower, upper, setback = BANDS[band]
    return scenario(
        f"{variant} | {price} | {band}", tariffs(START, days_)[price], comfort_spec(lower, upper, setback), START, days_
    ).model_copy(update={"estimator": "perfect" if estimator == "perfect" else "mhe"})


class HeldEstimator:
    """No estimation: the measured air temperatures, the other states held at the spun-up state of the model."""

    def __init__(self, model: FastModel, x0: np.ndarray) -> None:
        self.model = model
        self._x = np.asarray(x0, dtype=float).copy()

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    def correct(self, measurement: np.ndarray) -> np.ndarray:
        self._x[self.model.comfort_indices] = measurement
        return self.state

    def predict(self, control: np.ndarray, disturbance: np.ndarray) -> np.ndarray:
        return self.state


def runs() -> dict[str, dict[str, Any]]:
    """Every cached closed-loop run of the house tutorial, keyed by a label.

    Against the true building (perturbed parameter groups, sensor noise) with moving horizon estimation: both
    variants, two price signals, two comfort bands, and the evening price spike. Against the model itself: the
    perfect-model references. On the base house: the true state and no estimation, to weigh the estimator.
    """
    table: dict[str, dict[str, Any]] = {}

    def add(variant: str, price: str, band: str, mismatch: str, estimator: str) -> None:
        parts = {"variant": variant, "price": price, "band": band, "mismatch": mismatch, "estimator": estimator}
        table[" | ".join(parts.values())] = parts | {"days": DAYS}

    for variant in VARIANTS:
        for price in PRICES:
            for band in BANDS:
                add(variant, price, band, "mismatch", "mhe")
    add("base", "day-ahead + evening spike", "standard", "mismatch", "mhe")
    for price in PRICES:
        add("base", price, "standard", "perfect", "perfect")
    for estimator in ("perfect", "none"):
        add("base", "day-ahead", "standard", "mismatch", estimator)
    return table


def compute(variant: str, price: str, band: str, mismatch: str, estimator: str, days: float) -> ClosedLoopResult:
    """One closed-loop run of the study (two days hourly: about ten minutes)."""
    house = building(variant)
    run = study_scenario(variant, price, band, estimator, days)
    run = run.model_copy(update={"name": f"{run.name} | {mismatch} | {estimator}"})
    plant = house.plant(mismatch == "mismatch")
    if estimator == "perfect":
        return run_scenario(house.model, run, plant=plant, forecaster=house.forecaster)
    # the estimators start from the state the model expects (the spun-up state of the model, not of the plant)
    if estimator == "none":
        held = HeldEstimator(house.model, house.initial_state())
        controller = ModelPredictiveController(house.model, run.settings, run.comfort, run.tariff)
        return run_closed_loop(controller, held, plant, house.forecaster, run.steps, name=run.name)
    mhe = MovingHorizonEstimator(house.model, run.settings.time_step, START, x0=house.initial_state())
    return run_scenario(house.model, run, plant=plant, forecaster=house.forecaster, estimator=mhe)


def cached(**parts: Any) -> ClosedLoopResult:
    """The cached run of ``parts`` (computed and stored when missing)."""
    return CACHE.run(lambda: compute(**parts), **parts)


def reference(variant: str, price: str, band: str, mismatch: str, days_: float = DAYS) -> ClosedLoopResult:
    """The controllers of the YAML on the same building, period, tariff and band (a second, computed live)."""
    run = study_scenario(variant, price, band, "mhe", days_)
    return building(variant).rule_based(run, mismatch=mismatch == "mismatch")
