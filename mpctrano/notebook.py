"""Helpers of the interactive tutorials: pre-generated models, cached closed-loop results, widgets.

The notebooks only wire these functions to ``ipywidgets`` controls:

* :class:`TutorialBuilding` loads the control model (R3C2) and the plant model (R4C3) of a
  pre-generated variant from ``tutorials/models/<typology>/``;
* :class:`ResultCache` stores closed-loop results as parquet frames keyed by the parameters of
  the run, so that a widget reads a cached run in milliseconds and only a "recompute" button
  pays the closed-loop cost;
* :func:`plan_explorer` and :func:`closed_loop_explorer` build the widgets; the live widgets
  re-solve a single open-loop MPC plan (a fraction of a second), the closed-loop ones read
  the cache;
* :func:`quick_mode` and :func:`days` shorten the periods when ``MPCTRANO_QUICK=1`` (CI).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import pandas as pd
from trano.mpc import RCModelType

from mpctrano.comfort import ComfortSchedule, ComfortSpec
from mpctrano.forecast import WeatherForecaster
from mpctrano.metrics import ControlMetrics
from mpctrano.model import RCModel
from mpctrano.mpc import ModelPredictiveController, MPCSettings, MPCSolution
from mpctrano.prices import PriceSignal, RetailComponents
from mpctrano.project import Project
from mpctrano.scenarios import Mismatch, Scenario, build_plant
from mpctrano.simulation import ClosedLoopResult, ModelPlant
from mpctrano.tariff import FlatTariff, Tariff, TimeOfUseTariff

QUICK_VARIABLE = "MPCTRANO_QUICK"


def quick_mode() -> bool:
    """Whether the notebooks run in quick mode (short periods, cached results only)."""
    return os.environ.get(QUICK_VARIABLE) == "1"


def days(value: float, quick: float = 1.0) -> float:
    return min(value, quick) if quick_mode() else value


@contextmanager
def timed(label: str) -> Any:  # noqa: ANN401
    """Print the wall-clock time of a block (the notebooks document their runtimes)."""
    started = time.perf_counter()
    yield
    print(f"{label}: {time.perf_counter() - started:.1f} s")


# --------------------------------------------------------------------------- pre-generated models
@dataclass
class TutorialBuilding:
    """A pre-generated variant of a tutorial typology: control model (R3C2) and plant model (R4C3)."""

    directory: Path
    variant: str = "base"
    controller: RCModelType = RCModelType.r3c2
    plant_structure: RCModelType = RCModelType.r4c3

    @classmethod
    def variants(cls, directory: Path) -> list[str]:
        stems = {path.name.removesuffix("_r3c2.mpc.json") for path in Path(directory).glob("*_r3c2.mpc.json")}
        return sorted(stems)

    def model(self, structure: RCModelType) -> RCModel:
        stem = f"{self.variant}_{structure.value.lower()}"
        return RCModel.from_files(self.directory / f"{stem}.mo", self.directory / f"{stem}.mpc.json")

    @cached_property
    def control(self) -> RCModel:
        return self.model(self.controller)

    @cached_property
    def plant_model(self) -> RCModel:
        return self.model(self.plant_structure)

    @cached_property
    def forecaster(self) -> WeatherForecaster:
        return WeatherForecaster.from_model(self.control)

    def project(self) -> Project:
        """A project on the control model whose plants of the other structure are pre-loaded."""
        project = Project.from_model(self.control)
        project.plants[self.plant_structure] = self.plant_model
        project.plants[self.controller] = self.control
        return project

    def plant(self, mismatch: Mismatch | None = None, time_step: float = 3600.0) -> ModelPlant:
        mismatch = mismatch or Mismatch()
        plant_model = self.plant_model if mismatch.plant_model_type == self.plant_structure else None
        return build_plant(self.control, mismatch, time_step, plant_model=plant_model)


# --------------------------------------------------------------------------- cache
@dataclass
class ResultCache:
    """Closed-loop results stored as ``<key>.parquet`` + ``<key>.json`` under ``directory``."""

    directory: Path
    _index: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(**parts: Any) -> str:  # noqa: ANN401
        """A stable key of the parameters of a run (JSON-serialised, sorted)."""
        payload = json.dumps(parts, sort_keys=True, default=_serialise)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def paths(self, key: str) -> tuple[Path, Path]:
        return self.directory / f"{key}.parquet", self.directory / f"{key}.json"

    def has(self, key: str) -> bool:
        return all(path.exists() for path in self.paths(key))

    def load(self, key: str) -> ClosedLoopResult:
        frame_path, meta_path = self.paths(key)
        meta = json.loads(meta_path.read_text())
        frame = pd.read_parquet(frame_path)
        return ClosedLoopResult(
            name=meta["name"],
            frame=frame,
            metrics=ControlMetrics.model_validate(meta["metrics"]),
            time_step=meta["time_step"],
            failures=meta["failures"],
            elapsed=meta["elapsed"],
            heating_columns=meta["heating_columns"],
            gas_price=meta["gas_price"],
        )

    def store(self, key: str, result: ClosedLoopResult, **parts: Any) -> None:  # noqa: ANN401
        frame_path, meta_path = self.paths(key)
        result.frame.to_parquet(frame_path)
        meta = {
            "name": result.name,
            "metrics": result.metrics.model_dump(),
            "time_step": result.time_step,
            "failures": result.failures,
            "elapsed": result.elapsed,
            "heating_columns": result.heating_columns,
            "gas_price": result.gas_price,
            "parts": json.loads(json.dumps(parts, default=_serialise)),
        }
        meta_path.write_text(json.dumps(meta, indent=1))

    def run(
        self,
        compute: Callable[[], ClosedLoopResult],
        force: bool = False,
        **parts: Any,  # noqa: ANN401
    ) -> ClosedLoopResult:
        """The cached result of ``parts``, or ``compute()`` stored under their key."""
        key = self.key(**parts)
        if not force and self.has(key):
            return self.load(key)
        result = compute()
        self.store(key, result, **parts)
        return result

    def index(self) -> pd.DataFrame:
        """One row per cached run with its parts and headline indicators."""
        rows = []
        for meta_path in sorted(self.directory.glob("*.json")):
            meta = json.loads(meta_path.read_text())
            rows.append(
                {
                    "key": meta_path.stem,
                    **meta.get("parts", {}),
                    "total_cost": ControlMetrics.model_validate(meta["metrics"]).total_cost,
                    "comfort_violation_kh": meta["metrics"]["comfort_violation_kh"],
                    "elapsed": meta["elapsed"],
                }
            )
        return pd.DataFrame(rows)


def _serialise(value: Any) -> Any:  # noqa: ANN401
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


# --------------------------------------------------------------------------- scenario building blocks
def price_options(
    start: pd.Timestamp | str,
    days_: float,
    components: RetailComponents | None = None,
    sample: str | None = "be_day_ahead_2024",
) -> dict[str, Tariff]:
    """Ready-made price signals over the period: flat, time-of-use, day-ahead, a scarcity spike, negative hours.

    The day-ahead prices come from the cached sample ``sample`` relocated onto the period when it is
    available, from the synthetic day-ahead generator otherwise.
    """
    start = pd.Timestamp(start)
    components = components or RetailComponents(grid_fee=0.10, taxes=0.03, vat=0.06, export_share=0.4)
    period = int(days_) + 3
    try:
        signal = PriceSignal.from_cached(sample, components) if sample else None
    except FileNotFoundError:
        signal = None
    if signal is not None:
        signal = signal.slice(signal.start, days=period).relocate(start - pd.Timedelta(days=1))
    else:
        signal = PriceSignal.synthetic(
            start - pd.Timedelta(days=1), period, base=90, amplitude=70, noise=8, components=components
        )
    retail_mean = float(signal.retail().mean())
    spike = signal.with_spike(start + pd.Timedelta(hours=17), hours=3, price=600.0)
    negative = signal.with_negative_hours(start + pd.Timedelta(hours=11), hours=4, price=-20.0)
    return {
        "flat": FlatTariff(price=round(retail_mean, 3)),
        "time of use": TimeOfUseTariff(
            peak_price=round(1.5 * retail_mean, 3), off_peak_price=round(0.6 * retail_mean, 3)
        ),
        "day-ahead": signal.tariff(),
        "day-ahead + evening spike": spike.tariff(),
        "day-ahead + negative midday": negative.tariff(),
    }


def comfort_spec(
    occupied_lower: float, occupied_upper: float, night_setback: float, widen_upper: float = 0.0
) -> ComfortSpec:
    """A comfort band from the widget values: occupied band, night setback below it, optional extra room above."""
    schedule = ComfortSchedule(
        occupied_lower=occupied_lower,
        occupied_upper=occupied_upper + widen_upper,
        unoccupied_lower=occupied_lower - night_setback,
        unoccupied_upper=occupied_upper + widen_upper + 2.0,
    )
    return ComfortSpec(default=schedule)


# --------------------------------------------------------------------------- widgets
def _widgets() -> Any:  # noqa: ANN401
    try:
        import ipywidgets
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "Install ipywidgets (`pip install mpctrano[notebook]`) for the interactive tutorials."
        ) from error
    return ipywidgets


def plan_explorer(
    building: TutorialBuilding,
    prices: Mapping[str, Tariff],
    start: pd.Timestamp | str,
    on_solution: Callable[[MPCSolution, RCModel], Any] | None = None,
) -> Any:  # noqa: ANN401
    """Widgets re-solving one open-loop MPC plan live: prices, comfort band, horizon, weights.

    ``on_solution(solution, model)`` renders the plan (the plotly figure by default).
    """
    widgets = _widgets()
    from IPython.display import display

    from mpctrano.plot import plot_plan

    model, forecaster = building.control, building.forecaster
    start = pd.Timestamp(start)
    controls = {
        "price": widgets.Dropdown(options=list(prices), value=next(iter(prices)), description="Price"),
        "lower": widgets.FloatSlider(value=21.0, min=18.0, max=23.0, step=0.5, description="Occupied min [degC]"),
        "upper": widgets.FloatSlider(value=24.0, min=22.0, max=27.0, step=0.5, description="Occupied max [degC]"),
        "setback": widgets.FloatSlider(value=3.0, min=0.0, max=6.0, step=0.5, description="Night setback [K]"),
        "horizon": widgets.Dropdown(options=[12, 24, 36, 48], value=24, description="Horizon [h]"),
        "comfort_penalty": widgets.FloatLogSlider(
            value=5.0, base=10, min=-1, max=2, step=0.25, description="Comfort weight"
        ),
    }

    def update(price: str, lower: float, upper: float, setback: float, horizon: int, comfort_penalty: float) -> None:
        settings = MPCSettings(horizon=horizon, comfort_penalty=comfort_penalty)
        controller = ModelPredictiveController(
            model, settings, comfort_spec(lower, max(upper, lower + 0.5), setback), prices[price]
        )
        forecast = forecaster.forecast(start, horizon, settings.time_step)
        started = time.perf_counter()
        solution = controller.solve(model.initial_state, forecast)
        elapsed = time.perf_counter() - started
        print(
            f"{solution.status} in {solution.iterations} iterations, {elapsed:.2f} s; "
            f"cost {solution.total_cost:.2f} EUR"
        )
        if on_solution is None:
            display(plot_plan(solution, model))  # type: ignore[no-untyped-call]
        else:
            on_solution(solution, model)

    output = widgets.interactive_output(update, controls)
    layout = widgets.VBox(
        [widgets.HBox(list(controls.values())[:3]), widgets.HBox(list(controls.values())[3:]), output]
    )
    return layout


def closed_loop_explorer(
    cache: ResultCache,
    runs: Mapping[str, Mapping[str, Any]],
    compute: Callable[..., ClosedLoopResult],
    dimensions: Mapping[str, list[Any]],
    on_result: Callable[[ClosedLoopResult], Any] | None = None,
) -> Any:  # noqa: ANN401
    """Widgets selecting a cached closed-loop run along ``dimensions``; a button recomputes it.

    ``runs`` maps a run label to the keyword arguments of ``compute`` (which are also the cache
    parts); ``dimensions`` gives the widget options, one dropdown per dimension; the selected
    values must match one of ``runs``.
    """
    widgets = _widgets()
    from IPython.display import display

    from mpctrano.plot import plot_closed_loop

    controls = {
        name: widgets.Dropdown(options=options, value=options[0], description=name)
        for name, options in dimensions.items()
    }
    recompute = widgets.Button(description="Recompute", tooltip="Run the closed loop again (slow)")
    output = widgets.Output()

    def selected() -> tuple[str, Mapping[str, Any]] | None:
        values = {name: control.value for name, control in controls.items()}
        for label, parts in runs.items():
            if all(parts.get(name) == value for name, value in values.items()):
                return label, parts
        return None

    def show(force: bool = False) -> None:
        output.clear_output(wait=True)
        with output:
            choice = selected()
            if choice is None:
                print("No run for this combination; see the cache index.")
                return
            label, parts = choice
            started = time.perf_counter()
            result = cache.run(lambda: compute(**parts), force=force, **parts)
            print(f"{label}: {'computed' if force else 'from cache'} in {time.perf_counter() - started:.1f} s")
            if on_result is None:
                display(plot_closed_loop(result))  # type: ignore[no-untyped-call]
            else:
                on_result(result)

    for control in controls.values():
        control.observe(lambda _change: show(), names="value")
    recompute.on_click(lambda _button: show(force=True))
    show()
    return widgets.VBox([widgets.HBox([*controls.values(), recompute]), output])


def kpi_table(results: Mapping[str, ClosedLoopResult], columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Headline indicators of several runs, one row each."""
    columns = columns or (
        "total_cost", "energy_kwh", "export_kwh", "comfort_violation_kh", "peak_power_kw",
        "low_price_share", "failures",
    )  # fmt: skip
    frame = pd.DataFrame({name: result.summary() for name, result in results.items()}).T
    return frame[[column for column in columns if column in frame.columns]].astype(float).round(2)


def scenario(
    name: str,
    tariff: Tariff,
    comfort: ComfortSpec,
    start: pd.Timestamp | str,
    days_: float,
    **settings: Any,  # noqa: ANN401
) -> Scenario:
    """A scenario from the building blocks of the widgets."""
    return Scenario(
        name=name,
        description=name,
        tariff=tariff,
        comfort=comfort,
        settings=MPCSettings(**settings),
        start=str(pd.Timestamp(start).date()),
        days=days_,
    )


__all__ = [
    "ResultCache",
    "TutorialBuilding",
    "closed_loop_explorer",
    "comfort_spec",
    "days",
    "kpi_table",
    "plan_explorer",
    "price_options",
    "quick_mode",
    "scenario",
    "timed",
]
