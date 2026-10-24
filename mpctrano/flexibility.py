"""Quantify the flexibility a building offers to the grid.

Three complementary views, with the definitions of the literature the tutorials cite
(``tutorials/references.md``):

* **implicit flexibility**: what a price-aware controller saves against a price-agnostic one under
  the same tariff (:func:`value_of_flexibility`), how much energy it moves in time
  (:func:`shifted_energy`) and at which prices it buys (:func:`price_response`);
* **explicit flexibility**: the response to an activation request of an aggregator or a grid
  operator (:class:`ActivationRequest`, :func:`respond`): the energy delivered during the window,
  the rebound afterwards and its ratio, in the spirit of the available storage capacity, storage
  efficiency and power shifting capability of Reynders, Diriken and Saelens (2017) and of the
  Annex 67 characterisation; the response to a price step is the flexibility function of Junker
  et al. (2018) (:func:`flexibility_function`);
* **the flexibility envelope**: how much power can be shed or absorbed for how long from the
  current state without leaving the comfort band, and at what cost (:class:`FlexibilityEnvelope`,
  a sweep of constrained MPC solves).

The power is the electricity bought from the grid (``P_meter``, what the tariff bills), read from
the meters of the plant in a closed loop and from the meters of the model in a plan.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mpctrano.estimation import StateEstimator
from mpctrano.simulation import ClosedLoopResult, Plant
from mpctrano.tariff import CapacityTariff, DemandResponseEvent, DemandResponseTariff, FlatTariff, Tariff

if TYPE_CHECKING:
    from mpctrano.forecast import Forecaster
    from mpctrano.model import BuildingModel
    from mpctrano.mpc import ModelPredictiveController, MPCSolution
    from mpctrano.scenarios import Scenario

JOULE_PER_KWH = 3.6e6


def _plant(plant: BuildingModel | Plant | Callable[[], Plant] | None) -> BuildingModel | Plant | None:
    """A plant, a model to wrap into one, or a fresh plant from a factory (the compared runs start alike)."""
    return plant() if callable(plant) and not hasattr(plant, "state") else plant  # type: ignore[return-value]


def _estimator(estimator: Callable[[], StateEstimator] | None) -> StateEstimator | None:
    """A fresh estimator from a factory (an estimator is stateful: each run needs its own), or the default of the
    scenario."""
    return None if estimator is None else estimator()


# --------------------------------------------------------------------------- implicit flexibility
def compare(results: Mapping[str, ClosedLoopResult], baseline: str | None = None) -> pd.DataFrame:
    """One row per run: energy, cost, comfort, peak and the changes relative to the baseline run."""
    frame = pd.DataFrame({name: result.summary() for name, result in results.items()}).T
    baseline = baseline or next(iter(results))
    reference = results[baseline].metrics
    cost, energy, peak = reference.total_cost, reference.energy_kwh, reference.peak_power_kw
    frame["cost_saving_pct"] = 100 * (cost - frame["total_cost"]) / cost if cost else np.nan
    frame["energy_change_pct"] = 100 * (frame["energy_kwh"] - energy) / energy if energy else np.nan
    frame["peak_reduction_pct"] = 100 * (peak - frame["peak_power_kw"]) / peak if peak else np.nan
    frame["shifted_energy_kwh"] = [shifted_energy(results[name], results[baseline]) for name in frame.index]
    return frame


def value_of_flexibility(
    model: BuildingModel,
    scenario: Scenario,
    plant: BuildingModel | Plant | Callable[[], Plant] | None = None,
    forecaster: Forecaster | None = None,
    estimator: Callable[[], StateEstimator] | None = None,
) -> pd.DataFrame:
    """Savings of a price-aware MPC against a price-agnostic one under the same tariff.

    The price-agnostic controller minimises the cost at a flat price (no capacity charge) with the
    same comfort band, model and demand response power limits; its run is then costed with the
    tariff of the scenario. The difference is what the flexibility of the building is worth under
    that tariff. Evaluate over several days: the heat stored at the end of the period is not paid
    back within it (end effect of the receding horizon). ``plant`` may be a factory so that both
    runs start from the same plant state.
    """
    from mpctrano.scenarios import run_scenario

    aware = run_scenario(model, scenario, plant=_plant(plant), forecaster=forecaster, estimator=_estimator(estimator))
    index = pd.DatetimeIndex(aware.frame.index)
    agnostic_scenario = scenario.model_copy(
        update={"name": f"{scenario.name} (price agnostic)", "tariff": price_agnostic_tariff(scenario.tariff, index)}
    )
    agnostic = run_scenario(
        model, agnostic_scenario, plant=_plant(plant), forecaster=forecaster, estimator=_estimator(estimator)
    )
    agnostic_metrics = agnostic.metrics_under(scenario.tariff)
    rows = {
        "price aware": {**aware.metrics.model_dump(), "total_cost": aware.metrics.total_cost},
        "price agnostic": {**agnostic_metrics.model_dump(), "total_cost": agnostic_metrics.total_cost},
    }
    saving = agnostic_metrics.total_cost - aware.metrics.total_cost
    rows["price aware"] |= {
        "cost_saving": saving,
        "cost_saving_pct": 100 * saving / agnostic_metrics.total_cost if agnostic_metrics.total_cost else 0.0,
        "shifted_energy_kwh": shifted_energy(aware, agnostic),
    }
    rows["price agnostic"] |= {"cost_saving": 0.0, "cost_saving_pct": 0.0, "shifted_energy_kwh": 0.0}
    return pd.DataFrame.from_dict(rows, orient="index")


def price_agnostic_tariff(tariff: Tariff, index: pd.DatetimeIndex) -> Tariff:
    """Flat tariff at the mean price of ``tariff`` over ``index``, keeping its power limits and floors (grid
    constraints)."""
    flat = FlatTariff(price=float(np.mean(tariff.energy_price(index))))
    base = tariff.base if isinstance(tariff, CapacityTariff) else tariff
    if isinstance(base, DemandResponseTariff):
        events = [
            event.model_copy(update={"price_multiplier": 1.0})
            for event in base.events
            if event.power_limit is not None or event.power_floor is not None
        ]
        return flat.with_events(events) if events else flat
    return flat


def shifted_energy(result: ClosedLoopResult, reference: ClosedLoopResult) -> float:
    """Energy [kWh] moved in time with respect to the reference run (half of the L1 distance)."""
    profile = result.energy
    other = reference.energy
    index = profile.index.intersection(other.index)
    return float(0.5 * np.abs(profile.loc[index] - other.loc[index]).sum())


def price_response(result: ClosedLoopResult) -> pd.DataFrame:
    """Energy bought per price level: shows whether the heating follows the tariff."""
    frame = pd.DataFrame({"price": result.frame["price"], "energy_kwh": result.energy})
    grouped = frame.groupby("price")["energy_kwh"].agg(["sum", "count"])
    grouped["share"] = grouped["sum"] / grouped["sum"].sum()
    return grouped


# --------------------------------------------------------------------------- explicit flexibility
Direction = Literal["down", "up"]


class ActivationRequest(BaseModel):
    """A request to change the billed power against the baseline during a window.

    ``down`` sheds ``power`` [W] (a DSO congestion signal, an mFRR/aFRR upward activation seen from
    the grid), ``up`` absorbs it (a negative-price or downward-balancing activation). The request
    is expressed to the controller as a soft cap or floor on the billed power relative to the
    baseline (the plan of the same controller without the request), as an aggregator with a
    baseline-based settlement would.
    """

    model_config = ConfigDict(frozen=True)

    start: datetime
    duration_hours: float = Field(gt=0)
    direction: Direction = "down"
    power: float = Field(gt=0, description="Requested change of the billed power [W]")
    notice_hours: float | None = Field(
        1.0,
        ge=0,
        description="Time between the notification and the start [h]: the controller plans with the cap or "
        "floor from then on (None: from the moment the window enters its horizon)",
    )
    non_delivery_price: float = Field(
        1.0,
        ge=0,
        description="What the energy not delivered costs the building [EUR/kWh] (the activation price it forgoes or "
        "pays back): the controller weighs it against the comfort penalty",
    )

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp(self.start) + pd.Timedelta(hours=self.duration_hours)

    @property
    def requested_energy_kwh(self) -> float:
        return self.power * self.duration_hours / 1000

    def event(self, baseline_power: float) -> DemandResponseEvent:
        """The demand response event implementing the request around ``baseline_power`` [W]."""
        notice = {"notice_hours": self.notice_hours}
        if self.direction == "down":
            return DemandResponseEvent(
                start=self.start, end=self.end, power_limit=max(baseline_power - self.power, 0.0), **notice
            )
        return DemandResponseEvent(start=self.start, end=self.end, power_floor=baseline_power + self.power, **notice)

    def window(self, index: pd.DatetimeIndex) -> np.ndarray:
        return np.asarray((index >= pd.Timestamp(self.start)) & (index < self.end), dtype=bool)


@dataclass(frozen=True)
class ActivationResult:
    """Baseline and response of an activation, with the delivered energy and the rebound."""

    request: ActivationRequest
    baseline: ClosedLoopResult
    response: ClosedLoopResult
    rebound_window_hours: float = 12.0

    @cached_property
    def frame(self) -> pd.DataFrame:
        """Billed power of the baseline and of the response [W], and the change in the requested direction."""
        index = self.baseline.frame.index.intersection(self.response.frame.index)
        base = self.baseline.frame.loc[index, "P_meter"]
        resp = self.response.frame.loc[index, "P_meter"]
        sign = 1.0 if self.request.direction == "down" else -1.0
        return pd.DataFrame({"baseline": base, "response": resp, "delta": sign * (base - resp)}, index=index)

    @cached_property
    def metrics(self) -> dict[str, float]:
        """Delivered energy and power, rebound energy, peak and ratio, cost and comfort changes.

        ``delta`` counts positive in the requested direction: during the window it is the delivered
        change, after the window the negative part is the rebound (the energy paid back).
        """
        frame = self.frame
        hours = self.baseline.time_step / 3600
        index = pd.DatetimeIndex(frame.index)
        window = self.request.window(index)
        after = (index >= self.request.end) & (index < self.request.end + pd.Timedelta(hours=self.rebound_window_hours))
        delivered = float(frame.loc[window, "delta"].sum() * hours / 1000)
        rebound = float(np.clip(-frame.loc[after, "delta"], 0, None).sum() * hours / 1000)
        rebound_peak = float(np.clip(-frame.loc[after, "delta"], 0, None).max() / 1000) if after.any() else 0.0
        settled = frame.loc[after, "delta"].abs() < 0.1 * self.request.power
        rebound_duration = float((~settled).sum() * hours) if after.any() else 0.0
        requested = self.request.requested_energy_kwh
        return {
            "delivered_energy_kwh": delivered,
            "delivered_power_kw": delivered / self.request.duration_hours,
            "delivery_ratio": delivered / requested if requested else float("nan"),
            "rebound_energy_kwh": rebound,
            "rebound_peak_kw": rebound_peak,
            "rebound_duration_h": rebound_duration,
            "rebound_ratio": rebound / delivered if delivered > 0 else float("nan"),
            "net_energy_change_kwh": self.response.metrics.energy_kwh - self.baseline.metrics.energy_kwh,
            "cost_change": self.response.metrics.total_cost - self.baseline.metrics.total_cost,
            "comfort_change_kh": self.response.metrics.comfort_violation_kh
            - self.baseline.metrics.comfort_violation_kh,
        }


def respond(
    model: BuildingModel,
    scenario: Scenario,
    request: ActivationRequest,
    plant: Callable[[], Plant] | None = None,
    forecaster: Forecaster | None = None,
    baseline: ClosedLoopResult | None = None,
    rebound_window_hours: float = 12.0,
    estimator: Callable[[], StateEstimator] | None = None,
) -> ActivationResult:
    """Run the baseline (without the request) and the response, both from the same plant state.

    ``plant`` and ``estimator`` are factories returning a fresh plant and estimator for each run; the baseline power
    of the window is the mean billed power of the baseline run over it. The response weighs the energy not
    delivered at ``request.non_delivery_price`` (the ``power_limit_penalty`` of the controller).
    """
    from mpctrano.scenarios import run_scenario

    if baseline is None:
        baseline = run_scenario(
            model, scenario, plant=_plant(plant), forecaster=forecaster, estimator=_estimator(estimator)
        )
    index = pd.DatetimeIndex(baseline.frame.index)
    window = request.window(index)
    if not window.any():
        raise ValueError(f"The request window {request.start} .. {request.end} is outside the run.")
    baseline_power = float(baseline.frame.loc[window, "P_meter"].mean())
    tariff = scenario.tariff.with_events([request.event(baseline_power)])
    responding = scenario.model_copy(
        update={
            "name": f"{scenario.name} ({request.direction} {request.power / 1000:g} kW)",
            "tariff": tariff,
            "settings": scenario.settings.model_copy(update={"power_limit_penalty": request.non_delivery_price}),
        }
    )
    response = run_scenario(
        model, responding, plant=_plant(plant), forecaster=forecaster, estimator=_estimator(estimator)
    )
    return ActivationResult(
        request=request, baseline=baseline, response=response, rebound_window_hours=rebound_window_hours
    )


def flexibility_function(
    model: BuildingModel,
    scenario: Scenario,
    step_start: datetime,
    step_hours: float,
    price_factor: float = 3.0,
    plant: Callable[[], Plant] | None = None,
    forecaster: Forecaster | None = None,
    baseline: ClosedLoopResult | None = None,
    estimator: Callable[[], StateEstimator] | None = None,
) -> ActivationResult:
    """The response of the billed power to a price step (the flexibility function of Junker et al.).

    The price is multiplied by ``price_factor`` during the step; the result reads like an
    activation: ``delta`` is the power reduction, the metrics give the energy shed, the rebound
    and its duration (see :meth:`ActivationResult.metrics`).
    """
    from mpctrano.scenarios import run_scenario

    if baseline is None:
        baseline = run_scenario(
            model, scenario, plant=_plant(plant), forecaster=forecaster, estimator=_estimator(estimator)
        )
    end = pd.Timestamp(step_start) + pd.Timedelta(hours=step_hours)
    event = DemandResponseEvent(start=step_start, end=end, price_multiplier=price_factor)
    stepped = scenario.model_copy(
        update={"name": f"{scenario.name} (price x{price_factor:g})", "tariff": scenario.tariff.with_events([event])}
    )
    response = run_scenario(model, stepped, plant=_plant(plant), forecaster=forecaster, estimator=_estimator(estimator))
    in_step = (baseline.frame.index >= pd.Timestamp(step_start)) & (baseline.frame.index < end)
    window = baseline.frame.loc[in_step, "P_meter"]
    if window.empty:
        raise ValueError(f"The price step {step_start} .. {end} is outside the baseline run.")
    request = ActivationRequest(
        start=step_start, duration_hours=step_hours, direction="down", power=max(float(window.mean()), 1.0)
    )
    return ActivationResult(request=request, baseline=baseline, response=response)


# --------------------------------------------------------------------------- flexibility envelope
def plan_comfort_violation(solution: MPCSolution, model: BuildingModel) -> float:
    """Comfort violations of an open-loop plan [K.h] over the zones."""
    temperatures = solution.comfort_temperatures[model.comfort_states].to_numpy(dtype=float).T
    violation = np.maximum(solution.lower - temperatures, 0) + np.maximum(temperatures - solution.upper, 0)
    index = solution.index
    step = (index[1] - index[0]).total_seconds() if len(index) > 1 else 3600.0
    return float(violation.sum() * step / 3600)


@dataclass(frozen=True)
class FlexibilityEnvelope:
    """How much billed power the building can shed (``down``) or absorb (``up``) for a duration.

    For each duration, the largest constant change of the billed power from the start of the window that the
    controller can hold while adding at most ``tolerance_kh`` of comfort violations to those of its unconstrained
    plan, found by bisection on the cap (or floor) of the billed power; the energy and the extra cost of that plan
    against the unconstrained one come with it. Absorbing is bounded by the full-load power of the building
    (:meth:`~mpctrano.mpc.ModelPredictiveController.full_load_power`). The end effect of the horizon matters: the
    plan must have room after the window to recover.
    """

    frame: pd.DataFrame
    """Indexed by the duration [h]: ``down_kw``, ``down_energy_kwh``, ``down_cost``, ``up_kw``, ..."""
    baseline_cost: float

    @classmethod
    def compute(
        cls,
        controller: ModelPredictiveController,
        x0: np.ndarray,
        forecast: pd.DataFrame,
        durations: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0),
        offset_hours: float = 0.0,
        tolerance_kh: float = 0.05,
        iterations: int = 7,
    ) -> FlexibilityEnvelope:
        model, dt = controller.model, controller.settings.time_step
        baseline = controller.solve(x0, forecast)
        if baseline.meter is None or not baseline.success:
            raise ValueError(f"The unconstrained plan failed: {baseline.status}.")
        billed = baseline.meter["P_meter"].to_numpy(dtype=float)
        n = len(billed)
        allowed = plan_comfort_violation(baseline, model) + tolerance_kh
        capacity = controller.full_load_power(x0, forecast)
        first = round(offset_hours * 3600 / dt)
        if first >= n:
            raise ValueError("The window starts after the horizon.")
        rows = []
        for duration in durations:
            last = min(n, first + max(1, round(duration * 3600 / dt)))
            window = slice(first, last)
            row: dict[str, float] = {"duration_h": duration}
            for direction in ("down", "up"):
                bound = (
                    float(billed[window].mean())
                    if direction == "down"
                    else max(capacity - float(billed[window].max()), 0.0)
                )
                low, high, best, best_solution = 0.0, bound, 0.0, baseline
                for _ in range(iterations):
                    power = 0.5 * (low + high)
                    limits = np.full(n, np.inf)
                    floors = np.zeros(n)
                    if direction == "down":
                        limits[window] = np.maximum(billed[window] - power, 0.0)
                    else:
                        floors[window] = billed[window] + power
                    solution = controller.solve(x0, forecast, power_limits=limits, power_floors=floors)
                    feasible = solution.success and plan_comfort_violation(solution, model) <= allowed
                    if feasible and solution.meter is not None:
                        delivered = solution.meter["P_meter"].to_numpy(dtype=float)[window]
                        held = (billed[window] - delivered if direction == "down" else delivered - billed[window]).min()
                        feasible = held >= 0.9 * power
                    if feasible:
                        low, best, best_solution = power, power, solution
                    else:
                        high = power
                row[f"{direction}_kw"] = best / 1000
                row[f"{direction}_energy_kwh"] = best * (last - first) * dt / JOULE_PER_KWH
                row[f"{direction}_cost"] = best_solution.economic_cost - baseline.economic_cost
            rows.append(row)
        return cls(frame=pd.DataFrame(rows).set_index("duration_h"), baseline_cost=baseline.economic_cost)


__all__ = [
    "ActivationRequest",
    "ActivationResult",
    "FlexibilityEnvelope",
    "compare",
    "flexibility_function",
    "plan_comfort_violation",
    "price_agnostic_tariff",
    "price_response",
    "respond",
    "shifted_energy",
    "value_of_flexibility",
]
