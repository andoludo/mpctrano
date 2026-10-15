"""Plotly figures of the calibration fit and of closed-loop runs (optional dependency ``plotly``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from mpctrano.units import KELVIN

if TYPE_CHECKING:
    from mpctrano.simulation import ClosedLoopResult

OUTDOOR = "weather_inputs.TDryBul"


def _plotly() -> tuple[Any, Any]:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as error:  # pragma: no cover
        raise ImportError("Install plotly (`pip install mpctrano[plot]`) to draw the figures.") from error
    return go, make_subplots


def plot_fit(fit: pd.DataFrame, title: str = "Calibration fit") -> Any:  # noqa: ANN401
    """Measured vs simulated zone air temperatures (frame of :func:`mpctrano.calibration.simulate_fit`)."""
    go, _ = _plotly()
    figure = go.Figure()
    for column in fit.columns:
        dash = "dot" if column.endswith("_measured") else "solid"
        figure.add_trace(go.Scatter(x=fit.index, y=fit[column] - KELVIN, name=column, line={"dash": dash}))
    figure.update_layout(title=title, yaxis_title="Temperature [degC]", xaxis_title="Time")
    return figure


def plot_closed_loop(result: ClosedLoopResult, title: str | None = None) -> Any:  # noqa: ANN401
    """Air temperatures with the comfort band, energy bought, controls and price of a closed-loop run."""
    go, make_subplots = _plotly()
    frame = result.frame
    figure = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.35, 0.25, 0.25, 0.15]
    )
    for state in result.comfort_states:
        figure.add_trace(go.Scatter(x=frame.index, y=frame[f"{state}_end"] - KELVIN, name=state), row=1, col=1)
    first = result.zones[0]
    for bound in ("lower", "upper"):
        figure.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[f"{first}_{bound}"] - KELVIN,
                name=f"comfort {bound}",
                line={"dash": "dot", "color": "grey"},
            ),
            row=1,
            col=1,
        )
    if OUTDOOR in frame:
        figure.add_trace(
            go.Scatter(x=frame.index, y=frame[OUTDOOR] - KELVIN, name="outdoor", line={"color": "black"}), row=1, col=1
        )
    hours = result.time_step / 3600
    for column, name in (("bought", "electricity bought"), ("fuel", "fuel"), ("sold", "electricity sold")):
        if frame[column].abs().sum() > 0:
            figure.add_trace(
                go.Scatter(x=frame.index, y=frame[column] / hours, name=name, line={"shape": "hv"}), row=2, col=1
            )
    controls = [column for column in frame.columns if column.endswith("_y.y")]
    for column in controls:
        figure.add_trace(go.Scatter(x=frame.index, y=frame[column], name=column, line={"shape": "hv"}), row=3, col=1)
    figure.add_trace(
        go.Scatter(x=frame.index, y=frame["price"], name="price", line={"shape": "hv", "color": "red"}), row=4, col=1
    )
    figure.update_yaxes(title_text="Temperature [degC]", row=1, col=1)
    figure.update_yaxes(title_text="Power [kW]", row=2, col=1)
    figure.update_yaxes(title_text="Control [-]", row=3, col=1)
    figure.update_yaxes(title_text="Price [EUR/kWh]", row=4, col=1)
    figure.update_layout(title=title or f"Closed loop: {result.name}", height=950)
    return figure


def plot_comparison(
    comparison: pd.DataFrame,
    columns: tuple[str, ...] = ("total_cost", "energy_kwh", "comfort_violation_kh", "peak_power_kw"),
) -> Any:  # noqa: ANN401
    """Bar chart of the indicators of :func:`mpctrano.flexibility.compare`."""
    go, make_subplots = _plotly()
    figure = make_subplots(rows=1, cols=len(columns), subplot_titles=columns)
    for i, column in enumerate(columns, start=1):
        figure.add_trace(go.Bar(x=comparison.index, y=comparison[column], name=column), row=1, col=i)
    figure.update_layout(title="Scenario comparison", showlegend=False)
    return figure
