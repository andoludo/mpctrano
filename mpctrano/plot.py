"""Plotly figures of the calibration fit, of MPC plans and of closed-loop runs (optional dependency ``plotly``).

The figures share one palette (categorical hues in a fixed order, recessive grid, 2 px lines,
hover on every trace) so that the tutorials read as one system.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import pandas as pd

from mpctrano.units import KELVIN

if TYPE_CHECKING:
    from mpctrano.flexibility import ActivationResult, FlexibilityEnvelope
    from mpctrano.model import BuildingModel
    from mpctrano.mpc import MPCSolution
    from mpctrano.simulation import ClosedLoopResult

OUTDOOR = "weather_inputs.TDryBul"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
"""Categorical hues, assigned in this fixed order (validated for colour-vision deficiencies)."""
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"
LAYOUT: dict[str, Any] = {
    "template": "plotly_white",
    "font": {"color": INK, "size": 13},
    "hovermode": "x unified",
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    "margin": {"l": 60, "r": 20, "t": 70, "b": 40},
}


def _plotly() -> tuple[Any, Any]:
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        from plotly.subplots import make_subplots
    except ImportError as error:  # pragma: no cover
        raise ImportError("Install plotly (`pip install mpctrano[plot]`) to draw the figures.") from error
    if pio.renderers.default in ("plotly_mimetype", "plotly_mimetype+notebook") and "IPython" in sys.modules:
        # an HTML rendering next to the plotly mimetype: the executed notebooks keep their figures when rendered
        # without a plotly-aware front end (the documentation site, nbviewer)
        pio.renderers.default = "plotly_mimetype+notebook_connected"
    return go, make_subplots


def _style(figure: Any, title: str | None, height: int) -> Any:  # noqa: ANN401
    figure.update_layout(title=title, height=height, **LAYOUT)
    figure.update_xaxes(showgrid=False, zeroline=False, showline=True, linecolor=GRID)
    figure.update_yaxes(gridcolor=GRID, zeroline=False)
    return figure


def _line(go: Any, x: Any, y: Any, name: str, color: str, **options: Any) -> Any:  # noqa: ANN401
    line = {"color": color, "width": 2} | options.pop("line", {})
    return go.Scatter(x=x, y=y, name=name, mode="lines", line=line, **options)


def _band(go: Any, figure: Any, index: Any, lower: Any, upper: Any, row: int) -> None:  # noqa: ANN401
    figure.add_trace(
        go.Scatter(
            x=index, y=upper - KELVIN, name="comfort band", line={"width": 0}, showlegend=False, hoverinfo="skip"
        ),
        row=row,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=index,
            y=lower - KELVIN,
            name="comfort band",
            fill="tonexty",
            fillcolor="rgba(82, 81, 78, 0.12)",
            line={"width": 0},
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )


def _four_rows(make_subplots: Any) -> Any:  # noqa: ANN401
    """Temperatures, controls, power and price, sharing the time axis."""
    figure = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.36, 0.22, 0.24, 0.18]
    )
    for row, title in enumerate(("Temperature [degC]", "Control [-]", "Power [kW]", "Price [EUR/kWh]"), start=1):
        figure.update_yaxes(title_text=title, row=row, col=1)
    return figure


def plot_fit(fit: pd.DataFrame, title: str = "Calibration fit") -> Any:  # noqa: ANN401
    """Measured (dotted) vs simulated zone air temperatures, and the meters when the frame has them (frame of
    :func:`mpctrano.calibration.simulate_fit`)."""
    go, make_subplots = _plotly()
    temperatures = [column for column in fit.columns if ".TAir_" in column]
    meters = [column for column in fit.columns if column not in temperatures]
    rows = 2 if meters else 1
    figure = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.08)
    for i, column in enumerate(temperatures):
        dash = "dot" if column.endswith("_measured") else "solid"
        color = SERIES[(i // 2) % len(SERIES)]
        figure.add_trace(_line(go, fit.index, fit[column] - KELVIN, column, color, line={"dash": dash}), row=1, col=1)
    for i, column in enumerate(meters):
        dash = "dot" if column.endswith("_measured") else "solid"
        color = SERIES[(i // 2 + 2) % len(SERIES)]
        figure.add_trace(_line(go, fit.index, fit[column], column, color, line={"dash": dash}), row=2, col=1)
    figure.update_yaxes(title_text="Temperature [degC]", row=1, col=1)
    if meters:
        figure.update_yaxes(title_text="Energy [kWh]", row=2, col=1)
    return _style(figure, title, 450 if not meters else 700)


def plot_plan(solution: MPCSolution, model: BuildingModel, title: str | None = None) -> Any:  # noqa: ANN401
    """An open-loop MPC plan: zone temperatures in the comfort band, controls, electricity bought, prices."""
    go, make_subplots = _plotly()
    figure = _four_rows(make_subplots)
    index = solution.comfort_temperatures.index
    _band(go, figure, index, solution.lower[0], solution.upper[0], 1)
    for i, (zone, state) in enumerate(zip(model.zone_names, model.comfort_states, strict=True)):
        temperatures = solution.comfort_temperatures[state] - KELVIN
        figure.add_trace(_line(go, index, temperatures, zone, SERIES[i % len(SERIES)]), row=1, col=1)
    for i, name in enumerate(model.control_names):
        color = SERIES[(i + 2) % len(SERIES)]
        figure.add_trace(
            _line(go, solution.index, solution.controls[name], name, color, line={"shape": "hv"}), row=2, col=1
        )
    if solution.meter is not None:
        figure.add_trace(
            _line(
                go, solution.index, solution.meter["P_meter"] / 1000, "electricity bought", INK, line={"shape": "hv"}
            ),
            row=3,
            col=1,
        )
        if solution.meter["P_export"].max() > 0:
            export = -solution.meter["P_export"] / 1000
            figure.add_trace(_line(go, solution.index, export, "sold", SERIES[3], line={"shape": "hv"}), row=3, col=1)
    figure.add_trace(_line(go, solution.index, solution.prices, "price", SERIES[7], line={"shape": "hv"}), row=4, col=1)
    return _style(figure, title or f"MPC plan, cost {solution.economic_cost:.2f} EUR", 820)


def plot_closed_loop(result: ClosedLoopResult, title: str | None = None) -> Any:  # noqa: ANN401
    """Air temperatures with the comfort band, controls, energy bought and sold, and the price of a run."""
    go, make_subplots = _plotly()
    frame = result.frame
    figure = _four_rows(make_subplots)
    first = result.zones[0]
    _band(go, figure, frame.index, frame[f"{first}_lower"], frame[f"{first}_upper"], 1)
    for i, (zone, state) in enumerate(zip(result.zones, result.comfort_states, strict=True)):
        figure.add_trace(
            _line(go, frame.index, frame[f"{state}_end"] - KELVIN, zone, SERIES[i % len(SERIES)]), row=1, col=1
        )
    if OUTDOOR in frame:
        outdoor = frame[OUTDOOR] - KELVIN
        figure.add_trace(_line(go, frame.index, outdoor, "outdoor", MUTED, line={"dash": "dash"}), row=1, col=1)
    controls = [column for column in frame.columns if column.endswith("_y.y")]
    for i, column in enumerate(controls):
        color = SERIES[(i + 2) % len(SERIES)]
        figure.add_trace(_line(go, frame.index, frame[column], column, color, line={"shape": "hv"}), row=2, col=1)
    hours = result.time_step / 3600
    for column, name, color, sign in (
        ("bought", "electricity bought", INK, 1.0),
        ("fuel", "fuel", SERIES[1], 1.0),
        ("sold", "electricity sold", SERIES[3], -1.0),
    ):
        if column in frame and frame[column].abs().sum() > 0:
            power = sign * frame[column] / hours
            figure.add_trace(_line(go, frame.index, power, name, color, line={"shape": "hv"}), row=3, col=1)
    figure.add_trace(_line(go, frame.index, frame["price"], "price", SERIES[7], line={"shape": "hv"}), row=4, col=1)
    return _style(figure, title or f"Closed loop: {result.name}", 820)


def plot_states(result: ClosedLoopResult, states: list[str], title: str | None = None) -> Any:  # noqa: ANN401
    """Estimated against true values of states of a closed-loop run (e.g. a wall layer or the buffer)."""
    go, _ = _plotly()
    figure = go.Figure()
    frame = result.frame
    for i, state in enumerate(states):
        color = SERIES[i % len(SERIES)]
        unit = -KELVIN if frame[f"{state}_true"].mean() > 100 else 0.0
        if f"{state}_estimate" in frame:
            figure.add_trace(_line(go, frame.index, frame[f"{state}_estimate"] + unit, f"{state} estimate", color))
        figure.add_trace(
            _line(go, frame.index, frame[f"{state}_true"] + unit, f"{state} true", color, line={"dash": "dot"})
        )
    figure.update_yaxes(title_text="Temperature [degC]")
    return _style(figure, title or "State estimation", 420)


def plot_comparison(
    comparison: pd.DataFrame,
    columns: tuple[str, ...] = ("total_cost", "energy_kwh", "comfort_violation_kh", "peak_power_kw"),
) -> Any:  # noqa: ANN401
    """Bar chart of the indicators of :func:`mpctrano.flexibility.compare`."""
    go, make_subplots = _plotly()
    figure = make_subplots(rows=1, cols=len(columns), subplot_titles=columns)
    for i, column in enumerate(columns, start=1):
        figure.add_trace(
            go.Bar(x=comparison.index, y=comparison[column], name=column, marker_color=SERIES[0]), row=1, col=i
        )
    figure.update_layout(showlegend=False)
    return _style(figure, "Scenario comparison", 400)


def plot_activation(result: ActivationResult, title: str | None = None) -> Any:  # noqa: ANN401
    """Electricity bought by the baseline and by the response around an activation, and their difference."""
    go, make_subplots = _plotly()
    frame = result.frame
    request = result.request
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.6, 0.4])
    for row in (1, 2):
        figure.add_vrect(
            x0=pd.Timestamp(request.start),
            x1=request.end,
            fillcolor="rgba(82, 81, 78, 0.12)",
            line_width=0,
            row=row,
            col=1,
        )
    figure.add_trace(
        _line(go, frame.index, frame["baseline"] / 1000, "baseline", MUTED, line={"shape": "hv"}), row=1, col=1
    )
    figure.add_trace(
        _line(go, frame.index, frame["response"] / 1000, "response", SERIES[0], line={"shape": "hv"}), row=1, col=1
    )
    figure.add_trace(
        go.Bar(x=frame.index, y=frame["delta"] / 1000, name=f"change ({request.direction})", marker_color=SERIES[1]),
        row=2,
        col=1,
    )
    figure.update_yaxes(title_text="Electricity bought [kW]", row=1, col=1)
    figure.update_yaxes(title_text="Change [kW]", row=2, col=1)
    metrics = result.metrics
    subtitle = (
        f"delivered {metrics['delivered_energy_kwh']:.1f} kWh, rebound {metrics['rebound_energy_kwh']:.1f} kWh "
        f"(peak {metrics['rebound_peak_kw']:.1f} kW)"
    )
    return _style(figure, f"{title or 'Activation'}: {subtitle}", 620)


def plot_envelope(envelope: FlexibilityEnvelope | pd.DataFrame, title: str | None = None) -> Any:  # noqa: ANN401
    """Power that can be shed (negative) or absorbed (positive) against the duration, with the cost of each plan
    (an envelope or its frame)."""
    go, make_subplots = _plotly()
    frame = envelope if isinstance(envelope, pd.DataFrame) else envelope.frame
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Power [kW]", "Extra cost of the plan [EUR]"))
    durations = [f"{duration:g} h" for duration in frame.index]
    figure.add_trace(go.Bar(x=durations, y=frame["up_kw"], name="absorb (up)", marker_color=SERIES[0]), row=1, col=1)
    figure.add_trace(go.Bar(x=durations, y=-frame["down_kw"], name="shed (down)", marker_color=SERIES[1]), row=1, col=1)
    figure.add_trace(go.Bar(x=durations, y=frame["up_cost"], name="absorb cost", marker_color=SERIES[0]), row=1, col=2)
    figure.add_trace(go.Bar(x=durations, y=frame["down_cost"], name="shed cost", marker_color=SERIES[1]), row=1, col=2)
    figure.update_layout(barmode="relative")
    figure.update_xaxes(title_text="Duration", row=1, col=1)
    figure.update_xaxes(title_text="Duration", row=1, col=2)
    return _style(figure, title or "Flexibility envelope", 420)


def plot_storage(source: MPCSolution | ClosedLoopResult, title: str | None = None) -> Any:  # noqa: ANN401
    """State of charge of the storages [kWh] and their charging (positive) and discharging power, from a plan
    (``MPCSolution.storage``) or a closed-loop run (the storage columns of its frame)."""
    go, make_subplots = _plotly()
    frame = source.storage if hasattr(source, "storage") else source.frame
    if frame is None:
        raise ValueError("No storage in this plan.")
    names = [column.removesuffix("_soc_kwh") for column in frame.columns if column.endswith("_soc_kwh")]
    if not names:
        raise ValueError("No storage in this run.")
    figure = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.5, 0.5])
    for i, name in enumerate(names):
        color = SERIES[i % len(SERIES)]
        figure.add_trace(_line(go, frame.index, frame[f"{name}_soc_kwh"], f"{name} [kWh]", color), row=1, col=1)
        power = (frame[f"{name}_charge_w"] - frame[f"{name}_discharge_w"]) / 1000
        figure.add_trace(
            go.Bar(x=frame.index, y=power, name=f"{name} charge (+) / discharge (-) [kW]", marker_color=color),
            row=2,
            col=1,
        )
    figure.update_layout(barmode="relative")
    figure.update_yaxes(title_text="Energy [kWh]", row=1, col=1)
    figure.update_yaxes(title_text="Power [kW]", row=2, col=1)
    return _style(figure, title or "Storages", 560)
