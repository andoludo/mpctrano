"""Study definition of the flexibility tutorial (02): cached runs shared with ``scripts/build_cache.py``.

Use cases on the house variants: implicit flexibility under hourly and 15-minute day-ahead prices,
peak shaving under the Flemish-style capacity tariff, explicit activations with their rebound,
negative prices with PV. The runs are keyed by their parameters in ``tutorials/cache/flexibility/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mpctrano import ActivationRequest, CapacityTariff, respond, run_scenario
from mpctrano.flexibility import ActivationResult
from mpctrano.notebook import ResultCache, TutorialBuilding, comfort_spec, days, price_options, scenario
from mpctrano.prices import RetailComponents
from mpctrano.simulation import ClosedLoopResult
from tutorials._house import MISMATCH, MODELS, PERFECT, START

HERE = Path(__file__).parent
CACHE = ResultCache(HERE / "cache" / "flexibility")
DAYS = 3.0
"""Period of the cached runs."""
LIVE_DAYS = days(DAYS)
"""Period of the runs the notebook computes live (one day in quick mode)."""
COMPONENTS = RetailComponents(grid_fee=0.10, taxes=0.03, vat=0.06, export_share=0.4)
BAND = (21.0, 24.0, 3.0)
CAPACITY_PRICES = (0.0, 50.0, 150.0)
"""EUR/kW/year: none, the Flemish 2025 level (48.9 to 56.9) and a steep one."""
ACTIVATIONS: dict[str, dict[str, Any]] = {
    "shed 1 kW, 2 h at 7h": {"direction": "down", "power": 1000.0, "hour": 7.0, "duration": 2.0},
    "shed 1 kW, 2 h at 17h": {"direction": "down", "power": 1000.0, "hour": 17.0, "duration": 2.0},
    "shed 0.5 kW, 4 h at 17h": {"direction": "down", "power": 500.0, "hour": 17.0, "duration": 4.0},
    "absorb 1 kW, 2 h at 13h": {"direction": "up", "power": 1000.0, "hour": 13.0, "duration": 2.0},
    "absorb 2 kW, 1 h at 13h": {"direction": "up", "power": 2000.0, "hour": 13.0, "duration": 1.0},
}


def building(variant: str = "base") -> TutorialBuilding:
    return TutorialBuilding(MODELS, variant)


def plant_factory(variant: str, mismatch: str, time_step: float = 3600.0):
    return lambda: building(variant).plant(MISMATCH if mismatch == "mismatch" else PERFECT, time_step)


# --------------------------------------------------------------------------- implicit: hourly vs 15 min
def implicit_runs() -> dict[str, dict[str, Any]]:
    return {
        f"{variant} | {resolution}": {"study": "implicit", "variant": variant, "resolution": resolution, "days": DAYS}
        for variant in ("base", "old", "big_heat_pump")
        for resolution in ("1h", "15min")
    }


def compute_implicit(variant: str, resolution: str, days: float, **_: Any) -> ClosedLoopResult:
    """Price-aware closed loop at the resolution of the price signal (15 min: 96 steps a day)."""
    house = building(variant)
    tariffs = price_options(START, days, COMPONENTS)
    time_step = 3600.0 if resolution == "1h" else 900.0
    horizon = 24 if resolution == "1h" else 96
    run = scenario(
        f"{variant} {resolution}",
        tariffs["day-ahead"],
        comfort_spec(*BAND),
        START,
        days,
        horizon=horizon,
        time_step=time_step,
    )
    return run_scenario(house.control, run, plant=house.plant(MISMATCH, time_step))


# --------------------------------------------------------------------------- peak shaving
def capacity_runs() -> dict[str, dict[str, Any]]:
    return {
        f"{variant} | {price:g} EUR/kW/year": {
            "study": "capacity",
            "variant": variant,
            "capacity_price": price,
            "days": DAYS,
        }
        for variant in ("base", "old", "radiators")
        for price in CAPACITY_PRICES
    }


def compute_capacity(variant: str, capacity_price: float, days: float, **_: Any) -> ClosedLoopResult:
    house = building(variant)
    base = price_options(START, days, COMPONENTS)["time of use"]
    tariff = CapacityTariff(base=base, capacity_price=capacity_price) if capacity_price > 0 else base  # type: ignore[arg-type]
    run = scenario(f"{variant} capacity {capacity_price:g}", tariff, comfort_spec(20.0, 24.0, 4.0), START, days)
    return run_scenario(house.control, run, plant=house.plant(MISMATCH))


# --------------------------------------------------------------------------- explicit activations
def activation_runs() -> dict[str, dict[str, Any]]:
    return {
        f"{variant} | {label}": {"study": "activation", "variant": variant, "activation": label, "days": DAYS}
        for variant in ("base", "big_heat_pump")
        for label in ACTIVATIONS
    }


def _activation(variant: str, label: str, days: float = DAYS) -> ActivationResult:
    """Baseline and response of an activation on the second day of the study (both runs cached)."""
    house = building(variant)
    spec = ACTIVATIONS[label]
    tariffs = price_options(START, days, COMPONENTS)
    run = scenario(f"{variant} activation", tariffs["day-ahead"], comfort_spec(*BAND), START, days)
    request = ActivationRequest(
        start=START + pd.Timedelta(days=1, hours=spec["hour"]),
        duration_hours=spec["duration"],
        direction=spec["direction"],
        power=spec["power"],
    )
    baseline = CACHE.run(
        lambda: run_scenario(house.control, run, plant=plant_factory(variant, "mismatch")()),
        study="activation", variant=variant, activation="baseline", days=days,
    )  # fmt: skip
    factory = plant_factory(variant, "mismatch")
    response = CACHE.run(
        lambda: respond(house.control, run, request, plant=factory, baseline=baseline).response,
        study="activation", variant=variant, activation=label, days=days,
    )  # fmt: skip
    return ActivationResult(request=request, baseline=baseline, response=response)


def activation(variant: str, label: str, days: float = DAYS) -> ActivationResult:
    """Baseline and response of an activation of the study (both runs cached)."""
    return _activation(variant, label, days)


def compute_activation(variant: str, activation: str, days: float, **_: Any) -> ClosedLoopResult:
    """Entry point of the cache builder: computes (and caches) the baseline and the response."""
    if activation == "baseline":
        house = building(variant)
        tariffs = price_options(START, days, COMPONENTS)
        run = scenario(f"{variant} activation", tariffs["day-ahead"], comfort_spec(*BAND), START, days)
        return run_scenario(house.control, run, plant=plant_factory(variant, "mismatch")())
    return _activation(variant, activation, days).response


# --------------------------------------------------------------------------- negative prices and PV
def pv_runs() -> dict[str, dict[str, Any]]:
    return {
        f"{variant} | {price}": {"study": "pv", "variant": variant, "price": price, "days": DAYS}
        for variant in ("pv", "pv_battery_ev")
        for price in ("day-ahead", "day-ahead + negative midday")
    }


def compute_pv(variant: str, price: str, days: float, **_: Any) -> ClosedLoopResult:
    house = building(variant)
    tariffs = price_options(START, days, COMPONENTS)
    run = scenario(f"{variant} {price}", tariffs[price], comfort_spec(*BAND), START, days)
    return run_scenario(house.control, run, plant=house.plant(MISMATCH))


STUDIES = {
    "implicit": (implicit_runs, compute_implicit),
    "capacity": (capacity_runs, compute_capacity),
    "activation": (activation_runs, compute_activation),
    "pv": (pv_runs, compute_pv),
}


def runs() -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for builder, _ in STUDIES.values():
        table |= builder()
    return table


def compute(**parts: Any) -> ClosedLoopResult:
    return STUDIES[parts["study"]][1](**parts)


def cached(**parts: Any) -> ClosedLoopResult:
    return CACHE.run(lambda: compute(**parts), **parts)
