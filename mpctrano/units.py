"""Units and physical ranges shared by the modules."""

from __future__ import annotations

import numpy as np

KELVIN = 273.15
STATE_BOUNDS = (243.15, 373.15)
"""Physical range of the temperature states [K], used as bounds in the optimisation problems."""


def celsius(kelvin: np.ndarray | float) -> np.ndarray | float:
    return kelvin - KELVIN


def kelvin(celsius_value: np.ndarray | float) -> np.ndarray | float:
    return celsius_value + KELVIN
