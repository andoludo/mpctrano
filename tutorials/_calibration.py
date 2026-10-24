"""Study definition of the calibration tutorial (03): the annotated YAML, the true building, the recorded data and
the calibration, cached in ``tutorials/cache/calibration/``.

The building is the single-zone heat pump house with three parameters declared in its YAML
(``tests/models/single_zone_calibration.yaml``): the conductivity of the cavity wall insulation, its thickness and
the infiltration rate. The *true* building is the same YAML with other values of these parameters, operated by its
own controllers (the closed-loop BuildingModel); its monitoring data (noisy air temperature, electricity meter) is what
the calibration sees.
"""

from __future__ import annotations

import json
from functools import cache

import numpy as np
import pandas as pd

from mpctrano import (
    CalibrationResult,
    CalibrationSettings,
    CalibrationSpec,
    Calibrator,
    BuildingModel,
    YamlGroup,
    generate_closed_loop_data,
    spin_up,
)
from mpctrano.model import CLOSED_LOOP_LIBRARY
from tutorials._house import HERE

YAML = HERE.parent / "tests" / "models" / "single_zone_calibration.yaml"
CACHE = HERE / "cache" / "calibration"
START = pd.Timestamp("2023-01-09")
"""A Monday of the typical meteorological year: a cold week."""
TRAINING_DAYS = 3.0
VALIDATION_DAYS = 2.0
CONDUCTIVITY = "material[ROCKWOOL:CAL].thermal_conductivity"
THICKNESS = "constructions[CAVITYWALL:CAL].layers[1].thickness"
INFILTRATION = "spaces[SPACE:001].parameters.ach"
TRUTH = {
    CONDUCTIVITY: 0.06,  # the insulation is worse than its data sheet
    INFILTRATION: 0.6,  # the house is leakier than assumed
}
"""The values of the declared parameters in the true building (the YAML holds the assumed ones); the thickness
of the insulation is as drawn."""
CALIBRATED = (CONDUCTIVITY, INFILTRATION)
"""The declared fields the free fit adjusts: the thickness is left as drawn, it enters the model with the
conductivity only (``k / x``) and the two cannot be told apart."""
NOISE = {"measurement_std": 0.1, "meter_std": 0.02}
"""Sensor noise [K] and meter noise [kWh] of the recorded data."""


@cache
def spec() -> CalibrationSpec:
    return CalibrationSpec.from_yaml(YAML)


@cache
def model() -> BuildingModel:
    """The closed-loop model with the assumed values of the YAML: the model to calibrate."""
    return spec().model(CLOSED_LOOP_LIBRARY)


@cache
def truth() -> BuildingModel:
    """The true building: the YAML at :data:`TRUTH`, run by its own controllers."""
    return spec().model(CLOSED_LOOP_LIBRARY, TRUTH)


def declared_groups(force: bool = False) -> tuple[YamlGroup, ...]:
    """The calibration groups of every declared parameter (their sensitivities cached as JSON)."""
    path = CACHE / "groups.json"
    if path.exists() and not force:
        return tuple(YamlGroup.model_validate(item) for item in json.loads(path.read_text()))
    CACHE.mkdir(parents=True, exist_ok=True)
    result = spec().groups(model(), CLOSED_LOOP_LIBRARY)
    path.write_text(json.dumps([group.model_dump() for group in result], indent=1))
    return result


def groups(force: bool = False) -> tuple[YamlGroup, ...]:
    """The groups of the free fit (:data:`CALIBRATED`)."""
    return tuple(group for group in declared_groups(force) if group.name in CALIBRATED)


def recorded(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The monitoring data of the true building (cached as parquet): the data and the true states."""
    data_path, states_path = CACHE / "recorded.parquet", CACHE / "true_states.parquet"
    if data_path.exists() and states_path.exists() and not force:
        return pd.read_parquet(data_path), pd.read_parquet(states_path)
    CACHE.mkdir(parents=True, exist_ok=True)
    building = truth()
    x0 = spin_up(building, building, START, days=2.0)
    synthetic = generate_closed_loop_data(
        building,
        START,
        TRAINING_DAYS + VALIDATION_DAYS,
        x0=x0,
        seed=0,
        measurement_std=NOISE["measurement_std"],
        meter_std=NOISE["meter_std"],
    )
    synthetic.data.frame.to_parquet(data_path)
    synthetic.states.to_parquet(states_path)
    return synthetic.data.frame, synthetic.states


def split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training (the first days) and validation (the rest, from the last training row)."""
    steps = round(TRAINING_DAYS * 24)
    return data.iloc[: steps + 1], data.iloc[steps:]


@cache
def known_infiltration_model() -> BuildingModel:
    """The model with the infiltration rate measured (a blower-door test): only the envelope is left to calibrate."""
    return spec().model(CLOSED_LOOP_LIBRARY, {INFILTRATION: TRUTH[INFILTRATION]})


def known_infiltration_groups(force: bool = False) -> tuple[YamlGroup, ...]:
    """The conductivity alone, its sensitivities measured on :func:`known_infiltration_model` (cached as JSON)."""
    path = CACHE / "groups_known_infiltration.json"
    if path.exists() and not force:
        return tuple(YamlGroup.model_validate(item) for item in json.loads(path.read_text()))
    CACHE.mkdir(parents=True, exist_ok=True)
    result = spec().groups(
        known_infiltration_model(),
        CLOSED_LOOP_LIBRARY,
        values={INFILTRATION: TRUTH[INFILTRATION]},
        only=[CONDUCTIVITY],
    )
    path.write_text(json.dumps([group.model_dump() for group in result], indent=1))
    return result


def calibrate(
    force: bool = False, verbose: bool = False, known_infiltration: bool = False
) -> tuple[BuildingModel, CalibrationResult]:
    """The calibration of the model on the recorded data (the result cached as JSON; the calibrated model is
    rebuilt from it): the three declared parameters, or the conductivity alone on the model with the infiltration
    measured (``known_infiltration``)."""
    if known_infiltration:
        path, base, selected = (
            CACHE / "result_known_infiltration.json",
            known_infiltration_model(),
            known_infiltration_groups(),
        )
    else:
        path, base, selected = CACHE / "result.json", model(), groups()
    calibrator = Calibrator(base, CalibrationSettings(groups=selected, verbose=verbose, max_iterations=60))
    if path.exists() and not force:
        result = CalibrationResult.model_validate_json(path.read_text())
        return calibrator.calibrated(result.factors), result
    training, validation = split(recorded()[0])
    calibrated, result = calibrator.fit(training, validation)
    path.write_text(result.model_dump_json(indent=1))
    return calibrated, result


def initial_state(result: CalibrationResult) -> np.ndarray:
    return np.asarray(result.initial_state, dtype=float)
