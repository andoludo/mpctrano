"""Helpers shared by the tutorial scripts (paths and a quick mode for the test suite)."""

import os
from pathlib import Path

HERE = Path(__file__).parent
MODEL = HERE.parent / "tests" / "models" / "three_zones.yaml"
OUTPUT = HERE / "output"
QUICK = os.environ.get("MPCTRANO_QUICK") == "1"
"""Set MPCTRANO_QUICK=1 to run the tutorials with shorter periods (used by the tests)."""


def days(value: float) -> float:
    return min(value, 1.0) if QUICK else value


def save_figure(figure: object, name: str) -> Path:
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / f"{name}.html"
    figure.write_html(path)  # type: ignore[attr-defined]
    print(f"figure written to {path}")
    return path
