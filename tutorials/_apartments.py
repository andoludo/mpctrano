"""Study definition of the apartment block tutorial (03): cached runs shared with ``scripts/build_cache.py``.

Six apartments on one central heat pump with a 1 m3 domestic hot water tank: the compressor and
the tank are shared storages, which is what the runs explore (variants of emitters, tank size and
heat pump size; the tank locked or free during an activation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mpctrano import ActivationRequest, respond, run_scenario
from mpctrano.flexibility import ActivationResult
from mpctrano.model import RCModel
from mpctrano.notebook import ResultCache, TutorialBuilding, comfort_spec, days, price_options, scenario
from mpctrano.simulation import ClosedLoopResult
from tutorials._flexibility import COMPONENTS
from tutorials._house import MISMATCH, START

HERE = Path(__file__).parent
MODELS = HERE / "models" / "apartments"
CACHE = ResultCache(HERE / "cache" / "apartments")
DAYS = 3.0
LIVE_DAYS = days(DAYS)
BAND = (21.0, 24.0, 2.0)
VARIANTS = ("base", "floor_heating", "high_temperature", "big_tank", "small_heat_pump")
PRICES = ("day-ahead", "day-ahead + evening spike")
TANK = "tank_001"
LOCKED_BAND = (327.15, 328.15)
"""A one-kelvin band at the set temperature: the tank is kept full and offers no flexibility."""
REQUEST = {"hour": 18.0, "duration": 2.0, "power": 3000.0}
"""The evening shed of the activation study: 3 kW for two hours on the second day."""


def building(variant: str = "base") -> TutorialBuilding:
    return TutorialBuilding(MODELS, variant)


def tariffs(days_: float = DAYS) -> dict[str, Any]:
    return price_options(START, days_, COMPONENTS)


def control_model(variant: str, tank: str = "free") -> RCModel:
    model = building(variant).control
    return model.with_storage_band(TANK, *LOCKED_BAND) if tank == "locked" else model


# --------------------------------------------------------------------------- variants and prices
def block_runs() -> dict[str, dict[str, Any]]:
    return {
        f"{variant} | {price}": {"study": "block", "variant": variant, "price": price, "days": DAYS}
        for variant in VARIANTS
        for price in PRICES
    }


def compute_block(variant: str, price: str, days: float, **_: Any) -> ClosedLoopResult:
    house = building(variant)
    run = scenario(f"{variant} {price}", tariffs(days)[price], comfort_spec(*BAND), START, days)
    return run_scenario(house.control, run, plant=house.plant(MISMATCH))


# --------------------------------------------------------------------------- the tank in an activation
def activation_runs() -> dict[str, dict[str, Any]]:
    return {
        f"activation | tank {tank}": {"study": "activation", "tank": tank, "days": DAYS} for tank in ("free", "locked")
    }


def _request(days_: float) -> ActivationRequest:
    return ActivationRequest(
        start=START + pd.Timedelta(days=min(1.0, days_ - 1.0) if days_ > 1 else 0, hours=REQUEST["hour"]),
        duration_hours=REQUEST["duration"],
        direction="down",
        power=REQUEST["power"],
    )


def activation(tank: str, days: float = DAYS) -> ActivationResult:
    """Baseline and response of the evening shed with the tank free or locked (both runs cached)."""
    house = building("base")
    model = control_model("base", tank)
    run = scenario(f"base activation tank {tank}", tariffs(days)["day-ahead"], comfort_spec(*BAND), START, days)
    request = _request(days)
    baseline = CACHE.run(
        lambda: run_scenario(model, run, plant=house.plant(MISMATCH)),
        study="activation", tank=tank, role="baseline", days=days,
    )  # fmt: skip
    response = CACHE.run(
        lambda: respond(model, run, request, plant=lambda: house.plant(MISMATCH), baseline=baseline).response,
        study="activation", tank=tank, role="response", days=days,
    )  # fmt: skip
    return ActivationResult(request=request, baseline=baseline, response=response)


def compute_activation(tank: str, days: float, **_: Any) -> ClosedLoopResult:
    return activation(tank, days).response


STUDIES = {"block": (block_runs, compute_block), "activation": (activation_runs, compute_activation)}


def runs() -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for builder, _ in STUDIES.values():
        table |= builder()
    return table


def compute(**parts: Any) -> ClosedLoopResult:
    return STUDIES[parts["study"]][1](**parts)


def cached(**parts: Any) -> ClosedLoopResult:
    return CACHE.run(lambda: compute(**parts), **parts)
