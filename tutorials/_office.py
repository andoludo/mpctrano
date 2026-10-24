"""Study definition of the office cooling tutorial (04): a summer week, plans and the cached closed loops.

The office of ``tutorials/buildings/office/base.yaml`` has an ideal emitter per zone: a heat flow the controller
decides, which Trano writes as heating only (``min=0``); the study lets it go negative (cooling) and costs the
cooling at the EER of a chiller and the heating at the COP of a heat pump. The closed-loop ``BuildingModel`` of the
YAML has the thermostats of the heating only: in summer it is the office without cooling, the reference of the
comfort the MPC buys.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import pandas as pd

from mpctrano import ComfortSchedule, ComfortSpec, MovingHorizonEstimator, run_scenario
from mpctrano.notebook import ResultCache, TutorialBuilding, days, price_options, scenario
from mpctrano.scenarios import Scenario
from mpctrano.simulation import ClosedLoopResult
from tutorials._house import HERE

BUILDINGS = HERE / "buildings" / "office"
CACHE = ResultCache(HERE / "cache" / "office")
START = pd.Timestamp("2023-08-02")
"""The hottest day of the typical meteorological year (a Wednesday: 33.6 degC in the afternoon, after two hot days)."""
DAYS = 1.0
"""Period of the cached runs: one working day (a plan of this two-zone office takes one to two minutes)."""
LIVE_DAYS = days(1.0, quick=0.25)
COOLING = {"emitter_001_Q.Q_flow": (-12000.0, None), "emitter_002_Q.Q_flow": (-5000.0, None)}
"""Cooling capacity of the emitters [W]: the lower bound of their heat flow."""
EFFICIENCIES = {"heat_cop": 3.5, "cooling_eer": 3.0}
"""A reversible heat pump: 3.5 kWh of heat and 3 kWh of cooling per kWh of electricity."""
BAND = ComfortSpec(
    default=ComfortSchedule(
        occupied_lower=21.0, occupied_upper=25.0, unoccupied_lower=15.0, unoccupied_upper=30.0, occupied_hours=(8, 18)
    )
)
"""Occupied from 8h to 18h between 21 and 25 degC, free floating between 15 and 30 degC otherwise."""
PRICES = ("flat", "day-ahead")


@cache
def building() -> TutorialBuilding:
    return TutorialBuilding(BUILDINGS / "base.yaml", start=START, control_bounds=COOLING)


def tariffs(days_: float = DAYS) -> dict[str, Any]:
    return dict(price_options(START, days_))


def study_scenario(price: str, days_: float = DAYS, capacity_charge: float = 0.0) -> Scenario:
    tariff = tariffs(days_)[price]
    if capacity_charge:
        tariff = tariff.model_copy(update={"capacity_charge": capacity_charge})
    return scenario(f"office | {price}", tariff, BAND, START, days_, **EFFICIENCIES)


def runs() -> dict[str, dict[str, Any]]:
    return {f"office | {price}": {"price": price, "days": DAYS} for price in PRICES}


def compute(price: str, days: float) -> ClosedLoopResult:
    """One working day of the office against its true building, moving horizon estimation (about an hour)."""
    office = building()
    run = study_scenario(price, days)
    mhe = MovingHorizonEstimator(office.model, run.settings.time_step, run.start, x0=office.initial_state())
    return run_scenario(
        office.model, run, plant=office.plant(mismatch=True), forecaster=office.forecaster, estimator=mhe
    )


def cached(**parts: Any) -> ClosedLoopResult:
    return CACHE.run(lambda: compute(**parts), **parts)


def reference(price: str, days_: float = DAYS) -> ClosedLoopResult:
    """The office without cooling (the closed-loop model of the YAML in summer), costed and assessed alike."""
    return building().rule_based(study_scenario(price, days_), mismatch=True)
