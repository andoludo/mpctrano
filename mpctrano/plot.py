"""Plotly figures of the calibration fit, of MPC plans and of closed-loop runs (optional dependency ``plotly``).

The figures share one palette (categorical hues in a fixed order, recessive grid, 2 px lines,
hover on every trace) so that the tutorials read as one system.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from mpctrano.model import KELVIN

if TYPE_CHECKING:
    from trano.mpc import HeatPumpSpec

    from mpctrano.model import RCModel
    from mpctrano.mpc import MPCSolution
    from mpctrano.simulation import ClosedLoopResult

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
        from plotly.subplots import make_subplots
    except ImportError as error:  # pragma: no cover
        raise ImportError("Install plotly (`pip install mpctrano[plot]`) to draw the figures.") from error
    return go, make_subplots


def _style(figure: Any, title: str | None, height: int) -> Any:  # noqa: ANN401
    figure.update_layout(title=title, height=height, **LAYOUT)
    figure.update_xaxes(showgrid=False, zeroline=False, showline=True, linecolor=GRID)
    figure.update_yaxes(gridcolor=GRID, zeroline=False)
    return figure


def _line(go: Any, x: Any, y: Any, name: str, color: str, **options: Any) -> Any:  # noqa: ANN401
    line = {"color": color, "width": 2} | options.pop("line", {})
    return go.Scatter(x=x, y=y, name=name, mode="lines", line=line, **options)


def plot_fit(fit: pd.DataFrame, title: str = "Calibration fit") -> Any:  # noqa: ANN401
    """Measured vs simulated zone temperatures (frame of :func:`mpctrano.calibration.simulate_fit`)."""
    go, _ = _plotly()
    figure = go.Figure()
    for i, column in enumerate(fit.columns):
        dash = "dot" if column.endswith("_measured") else "solid"
        color = SERIES[(i // 2) % len(SERIES)]
        figure.add_trace(_line(go, fit.index, fit[column] - KELVIN, column, color, line={"dash": dash}))
    figure.update_yaxes(title_text="Temperature [degC]")
    return _style(figure, title, 450)


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


def plot_plan(solution: MPCSolution, model: RCModel, title: str | None = None) -> Any:  # noqa: ANN401
    """An open-loop MPC plan: zone temperatures in the comfort band, inputs and billed power, prices."""
    go, make_subplots = _plotly()
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.42, 0.38, 0.2])
    index = solution.comfort_temperatures.index
    _band(go, figure, index, solution.lower[0], solution.upper[0], 1)
    for i, zone in enumerate(model.zone_names):
        state = model.comfort_states[i]
        figure.add_trace(
            _line(go, index, solution.comfort_temperatures[state] - KELVIN, zone, SERIES[i % len(SERIES)]), row=1, col=1
        )
    for i, name in enumerate(model.control_names):
        figure.add_trace(
            _line(
                go,
                solution.index,
                solution.controls[name] / 1000,
                name,
                SERIES[(i + 2) % len(SERIES)],
                line={"shape": "hv"},
            ),
            row=2,
            col=1,
        )
    if solution.meter is not None:
        figure.add_trace(
            _line(
                go,
                solution.index,
                solution.meter["P_meter"] / 1000,
                "billed power",
                INK,
                line={"shape": "hv", "dash": "dot"},
            ),
            row=2,
            col=1,
        )
    figure.add_trace(_line(go, solution.index, solution.prices, "price", SERIES[7], line={"shape": "hv"}), row=3, col=1)
    figure.update_yaxes(title_text="Temperature [degC]", row=1, col=1)
    figure.update_yaxes(title_text="Power [kW]", row=2, col=1)
    figure.update_yaxes(title_text="Price [EUR/kWh]", row=3, col=1)
    return _style(figure, title or f"MPC plan, cost {solution.total_cost:.2f} EUR", 760)


def plot_closed_loop(result: ClosedLoopResult, title: str | None = None) -> Any:  # noqa: ANN401
    """Temperatures with the comfort band, inputs and billed power, and the price of a closed-loop run."""
    go, make_subplots = _plotly()
    frame = result.frame
    figure = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.42, 0.38, 0.2])
    first = result.zones[0]
    _band(go, figure, frame.index, frame[f"{first}_lower"], frame[f"{first}_upper"], 1)
    for i, zone in enumerate(result.zones):
        figure.add_trace(
            _line(go, frame.index, frame[f"{zone}_Ti_end"] - KELVIN, zone, SERIES[i % len(SERIES)]), row=1, col=1
        )
    figure.add_trace(
        _line(go, frame.index, frame["TOut"] - KELVIN, "outdoor", MUTED, line={"dash": "dash"}), row=1, col=1
    )
    for i, column in enumerate(result.heating_columns):
        figure.add_trace(
            _line(go, frame.index, frame[column] / 1000, column, SERIES[(i + 2) % len(SERIES)], line={"shape": "hv"}),
            row=2,
            col=1,
        )
    if "P_meter" in frame:
        figure.add_trace(
            _line(go, frame.index, frame["P_meter"] / 1000, "billed power", INK, line={"shape": "hv", "dash": "dot"}),
            row=2,
            col=1,
        )
    if "P_export" in frame and frame["P_export"].max() > 0:
        figure.add_trace(
            _line(go, frame.index, -frame["P_export"] / 1000, "export", SERIES[3], line={"shape": "hv"}), row=2, col=1
        )
    figure.add_trace(_line(go, frame.index, frame["price"], "price", SERIES[7], line={"shape": "hv"}), row=3, col=1)
    figure.update_yaxes(title_text="Temperature [degC]", row=1, col=1)
    figure.update_yaxes(title_text="Power [kW]", row=2, col=1)
    figure.update_yaxes(title_text="Price [EUR/kWh]", row=3, col=1)
    return _style(figure, title or f"Closed loop: {result.name}", 760)


def plot_states(result: ClosedLoopResult, states: list[str], title: str | None = None) -> Any:  # noqa: ANN401
    """Estimated against true values of states of a closed-loop run (e.g. the envelope temperature)."""
    go, _ = _plotly()
    figure = go.Figure()
    frame = result.frame
    for i, state in enumerate(states):
        color = SERIES[i % len(SERIES)]
        unit = -KELVIN if frame[f"{state}_estimate"].mean() > 100 else 0.0
        figure.add_trace(_line(go, frame.index, frame[f"{state}_estimate"] + unit, f"{state} estimate", color))
        if f"{state}_true" in frame:
            figure.add_trace(
                _line(go, frame.index, frame[f"{state}_true"] + unit, f"{state} true", color, line={"dash": "dot"})
            )
    figure.update_yaxes(title_text="State")
    return _style(figure, title or "State estimation", 420)


def plot_cop(heat_pump: HeatPumpSpec, supply_temperatures: tuple[float, ...] = (308.15, 318.15, 328.15)) -> Any:  # noqa: ANN401
    """COP of a heat pump against the outdoor temperature for a few supply temperatures."""
    go, _ = _plotly()
    outdoor = np.linspace(-10, 15, 26)
    figure = go.Figure()
    for i, supply in enumerate(supply_temperatures):
        cop = (
            heat_pump.cop_nominal
            + heat_pump.cop_outdoor_slope * (outdoor + KELVIN - 280.15)
            + heat_pump.cop_supply_slope * (supply - 308.15)
            + heat_pump.cop_cross_term * (outdoor + KELVIN - 280.15) * (supply - 308.15)
        )
        figure.add_trace(_line(go, outdoor, cop, f"supply {supply - KELVIN:.0f} degC", SERIES[i % len(SERIES)]))
    figure.update_xaxes(title_text="Outdoor temperature [degC]")
    figure.update_yaxes(title_text="COP [-]")
    return _style(figure, f"Heat pump {heat_pump.name}: COP polynomial", 400)


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
