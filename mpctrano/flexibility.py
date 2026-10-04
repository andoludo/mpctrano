"""Quantify the flexibility a building offers to a tariff: compare closed-loop runs."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from mpctrano.simulation import ClosedLoopResult


def compare(results: Mapping[str, ClosedLoopResult], baseline: str | None = None) -> pd.DataFrame:
    """One row per run: energy, cost, comfort, peak and savings relative to the baseline run."""
    frame = pd.DataFrame({name: result.summary() for name, result in results.items()}).T
    baseline = baseline or next(iter(results))
    reference = frame.loc[baseline]
    frame["cost_saving_pct"] = 100 * (reference["total_cost"] - frame["total_cost"]) / reference["total_cost"]
    frame["energy_change_pct"] = 100 * (frame["energy_kwh"] - reference["energy_kwh"]) / reference["energy_kwh"]
    frame["shifted_energy_kwh"] = [shifted_energy(results[name], results[baseline]) for name in frame.index]
    return frame


def shifted_energy(result: ClosedLoopResult, reference: ClosedLoopResult) -> float:
    """Energy [kWh] moved in time with respect to the reference run (half of the L1 distance)."""
    profile = _energy_profile(result)
    other = _energy_profile(reference)
    index = profile.index.intersection(other.index)
    return float(0.5 * np.abs(profile.loc[index] - other.loc[index]).sum())


def _energy_profile(result: ClosedLoopResult) -> pd.Series:
    powers = result.frame[[f"{zone}_QHea" for zone in result.zones]].sum(axis=1)
    return powers * result.time_step / 3.6e6


def price_response(result: ClosedLoopResult) -> pd.DataFrame:
    """Energy consumed per price level: shows whether the heating follows the tariff."""
    energy = _energy_profile(result)
    frame = pd.DataFrame({"price": result.frame["price"], "energy_kwh": energy})
    grouped = frame.groupby("price")["energy_kwh"].agg(["sum", "count"])
    grouped["share"] = grouped["sum"] / grouped["sum"].sum()
    return grouped
