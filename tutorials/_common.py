"""Helpers shared by the tutorial scripts (paths and a quick mode for the test suite)."""

import os
from pathlib import Path

HERE = Path(__file__).parent
MODEL = HERE.parent / "tests" / "models" / "single_zone_air_water_heat_pump.yaml"
"""A single-zone house heated by an air-to-water heat pump (Trano test suite), with its weather file next to it."""
OUTPUT = HERE / "output"
QUICK = os.environ.get("MPCTRANO_QUICK") == "1"
"""Set MPCTRANO_QUICK=1 to run the tutorials with shorter periods and horizons (used by the tests)."""


def days(value: float) -> float:
    """Length of a closed-loop run: 3 hours in quick mode."""
    return min(value, 3 / 24) if QUICK else value


def data_days(value: float) -> float:
    """Length of the monitoring data: 2 days in quick mode."""
    return min(value, 2.0) if QUICK else value


def horizon(value: int) -> int:
    """Prediction horizon [h]: 6 hours in quick mode."""
    return min(value, 6) if QUICK else value


def save_figure(figure: object, name: str) -> Path:
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / f"{name}.html"
    figure.write_html(path)  # type: ignore[attr-defined]
    print(f"figure written to {path}")
    return path
