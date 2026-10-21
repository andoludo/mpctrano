"""Study definition of the PV, battery and vehicle tutorial (05): cached runs shared with ``scripts/build_cache.py``.

The house of tutorial 01 with 20 m2 of PV on the roof (``tutorials/buildings/house/pv.yaml``), a stationary battery
and an electric vehicle charged at home overnight, optimised in the balance of the meter next to the thermal
storage of the building. Runs: day-ahead prices with and without negative midday prices, with and without the
electrical storages, against the true building with moving horizon estimation.
"""

from __future__ import annotations

from typing import Any


from mpctrano import Battery, ElectricVehicle, MovingHorizonEstimator, run_scenario
from mpctrano.notebook import ResultCache, comfort_spec, days, price_options, scenario
from mpctrano.scenarios import Scenario
from mpctrano.simulation import ClosedLoopResult
from tutorials._house import BAND_DEFAULT, HERE, START, building

CACHE = ResultCache(HERE / "cache" / "pv")
DAYS = 1.0
"""Period of the cached runs: one day (the PV house solves in 10 to 50 s)."""
LIVE_DAYS = days(1.0, quick=0.25)
BATTERY = Battery(capacity_kwh=8.0, max_charge_w=3000.0, max_discharge_w=3000.0, efficiency=0.92, initial_soc=0.5)
VEHICLE = ElectricVehicle(
    capacity_kwh=50.0,
    max_charge_w=7400.0,
    arrival_hour=18.0,
    departure_hour=7.0,
    driven_kwh=12.0,
    departure_energy_kwh=30.0,
    initial_soc=0.5,
)
STORAGES: dict[str, dict[str, Any]] = {
    "none": {},
    "battery": {"batteries": (BATTERY,)},
    "battery + vehicle": {"batteries": (BATTERY,), "vehicles": (VEHICLE,)},
}
PRICES = ("day-ahead", "day-ahead + negative midday")


def tariffs(days_: float = DAYS) -> dict[str, Any]:
    return dict(price_options(START, days_))


def study_scenario(price: str, storages: str, days_: float = DAYS) -> Scenario:
    return scenario(
        f"pv | {price} | {storages}",
        tariffs(days_)[price],
        comfort_spec(*BAND_DEFAULT),
        START,
        days_,
        **STORAGES[storages],
    )


def runs() -> dict[str, dict[str, Any]]:
    return {
        f"pv | {price} | {storages}": {"price": price, "storages": storages, "days": DAYS}
        for price in PRICES
        for storages in STORAGES
    }


def compute(price: str, storages: str, days: float) -> ClosedLoopResult:
    """One day of the PV house against its true building, moving horizon estimation (10 to 30 minutes)."""
    house = building("pv")
    run = study_scenario(price, storages, days)
    mhe = MovingHorizonEstimator(house.model, run.settings.time_step, run.start, x0=house.initial_state())
    return run_scenario(house.model, run, plant=house.plant(mismatch=True), forecaster=house.forecaster, estimator=mhe)


def cached(**parts: Any) -> ClosedLoopResult:
    return CACHE.run(lambda: compute(**parts), **parts)


def reference(price: str, days_: float = DAYS) -> ClosedLoopResult:
    """The controllers of the YAML on the PV house (no storage: the PV exports what the house does not use)."""
    return building("pv").rule_based(study_scenario(price, "none", days_), mismatch=True)
