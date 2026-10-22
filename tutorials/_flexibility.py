"""Study definition of the flexibility tutorial (02): cached runs shared with ``scripts/build_cache.py``.

Use cases on the house variants: implicit flexibility under day-ahead prices (price aware against price agnostic),
peak shaving under a capacity tariff, explicit activations with their rebound, the flexibility function, the
flexibility envelope, and negative prices with PV. Every run is a closed loop against the true building of the
variant with moving horizon estimation, from the spun-up state; the runs are keyed by their parameters in
``tutorials/cache/flexibility/``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from mpctrano import (
    ActivationRequest,
    CapacityTariff,
    ComfortSpec,
    FlexibilityEnvelope,
    ModelPredictiveController,
    MovingHorizonEstimator,
    MPCSettings,
    flexibility_function,
    respond,
    run_scenario,
)
from mpctrano.flexibility import ActivationResult, price_agnostic_tariff
from mpctrano.notebook import ResultCache, comfort_spec, days, price_options, scenario
from mpctrano.scenarios import Scenario
from mpctrano.simulation import ClosedLoopResult
from tutorials._house import HERE, START, building

CACHE = ResultCache(HERE / "cache" / "flexibility")
DAYS = 2.0
"""Period of the cached runs."""
LIVE_DAYS = days(1.0, quick=0.25)
"""Period of the runs the notebook computes live (six hours in quick mode)."""
BAND = (21.0, 24.0, 3.0)
CAPACITY: dict[str, tuple[float, float] | None] = {
    "none": None,
    "50 EUR/kW/year, 2.5 kW floor": (50.0, 2.5),
    "150 EUR/kW/year, no floor": (150.0, 0.0),
}
"""Capacity tariffs: none, the Flemish one (2025 level, the peak floored at 2.5 kW) and a steep one without floor."""
ACTIVATIONS: dict[str, dict[str, Any]] = {
    "shed 0.3 kW, 2 h at 17h": {"direction": "down", "power": 300.0, "hour": 17.0, "duration": 2.0},
    "shed 0.3 kW, 4 h at 17h": {"direction": "down", "power": 300.0, "hour": 17.0, "duration": 4.0},
    "absorb 0.5 kW, 2 h at 13h": {"direction": "up", "power": 500.0, "hour": 13.0, "duration": 2.0},
    "absorb 1 kW, 1 h at 13h": {"direction": "up", "power": 1000.0, "hour": 13.0, "duration": 1.0},
}
"""Requests sized to the house: it draws 0.3 to 0.6 kW in the evening and its compressor about 1.2 kW at most."""
PRICE_STEP = {"hour": 17.0, "duration": 3.0, "factor": 3.0}
ENVELOPE_TIMES = {"13h": START + pd.Timedelta(hours=13), "19h": START + pd.Timedelta(hours=19)}
"""When the envelope is computed on the first day: at midday the house is in its band and the compressor idle
(room to absorb, nothing to shed); in the evening the heat pump runs (room to shed). At 6h the morning recovery
saturates the heat pump and any shed breaks the 7h set point."""
ENVELOPE_DURATIONS = (1.0, 2.0, 4.0)


def tariffs(days_: float = DAYS) -> dict[str, Any]:
    return dict(price_options(START, days_))


def _run(variant: str, name: str, tariff: Any, days_: float, band: tuple[float, float, float] = BAND) -> Scenario:
    return scenario(f"{variant} | {name}", tariff, comfort_spec(*band), START, days_)


def estimator(variant: str, run: Scenario) -> MovingHorizonEstimator:
    """A moving horizon estimator started from the spun-up state the model expects (not the plant's)."""
    house = building(variant)
    return MovingHorizonEstimator(house.model, run.settings.time_step, run.start, x0=house.initial_state())


def _closed_loop(variant: str, run: Scenario) -> ClosedLoopResult:
    """The closed loop of ``run`` against the true building of the variant, with moving horizon estimation."""
    house = building(variant)
    return run_scenario(
        house.model,
        run,
        plant=house.plant(mismatch=True),
        forecaster=house.forecaster,
        estimator=estimator(variant, run),
    )


# --------------------------------------------------------------------------- implicit flexibility
def implicit_runs() -> dict[str, dict[str, Any]]:
    return {
        f"{variant} | {pricing}": {"study": "implicit", "variant": variant, "pricing": pricing, "days": DAYS}
        for variant in ("base", "old")
        for pricing in ("aware", "agnostic")
    }


def compute_implicit(variant: str, pricing: str, days: float, **_: Any) -> ClosedLoopResult:
    """The price-aware run follows the day-ahead prices; the price-agnostic one plans at their mean (cost it with
    ``metrics_under`` the day-ahead tariff)."""
    tariff = tariffs(days)["day-ahead"]
    if pricing == "agnostic":
        index = pd.date_range(START, periods=round(days * 24), freq="h")
        tariff = price_agnostic_tariff(tariff, index)
    return _closed_loop(variant, _run(variant, f"day-ahead, price {pricing}", tariff, days))


# --------------------------------------------------------------------------- peak shaving
def capacity_runs() -> dict[str, dict[str, Any]]:
    return {
        f"{variant} | {capacity}": {"study": "capacity", "variant": variant, "capacity": capacity, "days": DAYS}
        for variant in ("base", "old")
        for capacity in CAPACITY
    }


def capacity_tariff(capacity: str, days_: float = DAYS) -> Any:
    base = tariffs(days_)["time of use"]
    spec = CAPACITY[capacity]
    if spec is None:
        return base
    price, floor = spec
    return CapacityTariff(base=base, capacity_price=price, minimum_peak_kw=floor)


def compute_capacity(variant: str, capacity: str, days: float, **_: Any) -> ClosedLoopResult:
    """Time-of-use prices and a deep night setback: the morning recovery makes the peak."""
    run = _run(variant, f"capacity {capacity}", capacity_tariff(capacity, days), days, band=(21.0, 24.0, 5.0))
    return _closed_loop(variant, run)


# --------------------------------------------------------------------------- explicit activations
def activation_runs() -> dict[str, dict[str, Any]]:
    return {
        f"base | {label}": {"study": "activation", "variant": "base", "activation": label, "days": DAYS}
        for label in ("baseline", *ACTIVATIONS, "price step")
    }


def _activation_run(days_: float) -> Scenario:
    return _run("base", "day-ahead", tariffs(days_)["day-ahead"], days_)


def _request(label: str) -> ActivationRequest:
    spec = ACTIVATIONS[label]
    return ActivationRequest(
        start=START + pd.Timedelta(hours=spec["hour"]),
        duration_hours=spec["duration"],
        direction=spec["direction"],
        power=spec["power"],
    )


def activation(label: str, days_: float = DAYS) -> ActivationResult:
    """Baseline and response of an activation (both runs cached), or of the price step of the flexibility
    function (``label="price step"``)."""
    baseline = cached(study="activation", variant="base", activation="baseline", days=days_)
    response = cached(study="activation", variant="base", activation=label, days=days_)
    if label == "price step":
        in_step = (baseline.frame.index >= _step_start()) & (
            baseline.frame.index < _step_start() + pd.Timedelta(hours=PRICE_STEP["duration"])
        )
        request = ActivationRequest(
            start=_step_start(),
            duration_hours=PRICE_STEP["duration"],
            direction="down",
            power=max(float(baseline.frame.loc[in_step, "P_meter"].mean()), 1.0),
        )
    else:
        request = _request(label)
    return ActivationResult(request=request, baseline=baseline, response=response)


def _step_start() -> pd.Timestamp:
    return START + pd.Timedelta(hours=PRICE_STEP["hour"])


def compute_activation(activation: str, days: float, **_: Any) -> ClosedLoopResult:
    """The baseline, the response to an activation request or to the price step (from the cached baseline)."""
    run = _activation_run(days)
    if activation == "baseline":
        return _closed_loop("base", run)
    house = building("base")
    baseline = cached(study="activation", variant="base", activation="baseline", days=days)

    def plant() -> Any:
        return house.plant(mismatch=True)

    def fresh_estimator() -> MovingHorizonEstimator:
        return estimator("base", run)

    if activation == "price step":
        return flexibility_function(
            house.model,
            run,
            _step_start(),
            PRICE_STEP["duration"],
            price_factor=PRICE_STEP["factor"],
            plant=plant,
            forecaster=house.forecaster,
            baseline=baseline,
            estimator=fresh_estimator,
        ).response
    return respond(
        house.model,
        run,
        _request(activation),
        plant=plant,
        forecaster=house.forecaster,
        baseline=baseline,
        estimator=fresh_estimator,
    ).response


# --------------------------------------------------------------------------- negative prices and PV
def pv_runs() -> dict[str, dict[str, Any]]:
    return {
        f"pv | {price}": {"study": "pv", "variant": "pv", "price": price, "days": DAYS}
        for price in ("day-ahead", "day-ahead + negative midday")
    }


def compute_pv(variant: str, price: str, days: float, **_: Any) -> ClosedLoopResult:
    return _closed_loop(variant, _run(variant, price, tariffs(days)[price], days))


# --------------------------------------------------------------------------- the study
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


def envelope(time: str = "13h", force: bool = False) -> pd.DataFrame:
    """The flexibility envelope of the base house at ``time`` on the first day (:data:`ENVELOPE_TIMES`), day-ahead
    prices (cached as a CSV: about thirty open-loop solves)."""
    path = CACHE.directory / f"envelope_base_{time}.csv"
    if path.exists() and not force:
        return pd.read_csv(path, index_col="duration_h")
    at = ENVELOPE_TIMES[time]
    house = building("base")
    controller = ModelPredictiveController(house.model, MPCSettings(horizon=24), ComfortSpec(), tariffs()["day-ahead"])
    frame = FlexibilityEnvelope.compute(
        controller,
        house.initial_state(start=at),
        house.forecaster.forecast(at, 24, 3600.0),
        durations=ENVELOPE_DURATIONS,
        iterations=5,
    ).frame
    frame.to_csv(path)
    return frame
