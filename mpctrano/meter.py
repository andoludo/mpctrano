"""Electricity at the meter: consumption of the converters, battery flows, PV generation, import and export.

The interface of the model tells which control inputs draw electricity (heat pumps, chillers,
tanks, charging), which deliver it (battery discharging), which disturbances generate it (PV) and
which inputs draw gas (boilers, costed with their efficiency) or ideal heat (thermal inputs of a
zone without system, costed at the tariff price). :class:`Meter` turns those into the symbolic
balance used by the controller and into the columns of a closed-loop frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import casadi as ca
import numpy as np
import pandas as pd

from mpctrano.model import RCModel

METER_COLUMNS = ("P_consumption", "P_discharge", "P_generation", "P_net", "P_import", "P_export", "P_gas", "P_heat")
"""Columns added to a closed-loop frame [W]: see :meth:`Meter.evaluate`."""


@dataclass(frozen=True)
class Meter:
    """Electrical, gas and ideal-heat balances of a model, symbolic (CasADi) or numeric (frames)."""

    model: RCModel

    def _sum(self, matrix: Any, indices: list[int]) -> Any:  # noqa: ANN401
        if not indices:
            return ca.DM.zeros(1, matrix.shape[1]) if hasattr(matrix, "shape") else 0.0
        return ca.sum1(matrix[indices, :]) if len(indices) > 1 else matrix[indices[0], :]

    def consumption(self, U: Any) -> Any:  # noqa: ANN401, N803
        """Electricity drawn by the converters and the charging inputs [W] (``1 x N``)."""
        return self._sum(U, self.model.consumption_control_indices)

    def discharge(self, U: Any) -> Any:  # noqa: ANN401, N803
        return self._sum(U, self.model.discharge_control_indices)

    def generation(self, D: Any) -> Any:  # noqa: ANN401, N803
        return self._sum(D, self.model.generation_indices)

    def net(self, U: Any, D: Any) -> Any:  # noqa: ANN401, N803
        """Net power at the meter [W]: positive when importing."""
        return self.consumption(U) - self.discharge(U) - self.generation(D)

    def gas(self, U: Any) -> Any:  # noqa: ANN401, N803
        """Fuel power of the boilers [W] (heat divided by the efficiency)."""
        indices = self.model.gas_control_indices
        if not indices:
            return ca.DM.zeros(1, U.shape[1])
        scaled = ca.vertcat(
            *[U[index, :] / efficiency for index, efficiency in zip(indices, self.model.gas_efficiencies, strict=True)]
        )
        return ca.sum1(scaled) if len(indices) > 1 else scaled

    def heat(self, U: Any) -> Any:  # noqa: ANN401, N803
        """Ideal thermal control inputs [W] (costed at the tariff price, as electricity)."""
        return self._sum(U, self.model.heat_control_indices)

    # ----------------------------------------------------------------- numeric
    def evaluate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The meter columns of a frame holding the controls and disturbances of each step."""
        model = self.model
        U = frame[model.control_names].to_numpy(dtype=float).T
        D = frame[model.disturbance_names].to_numpy(dtype=float).T
        consumption = U[model.consumption_control_indices].sum(axis=0) if model.consumption_control_indices else 0.0
        discharge = U[model.discharge_control_indices].sum(axis=0) if model.discharge_control_indices else 0.0
        generation = D[model.generation_indices].sum(axis=0) if model.generation_indices else 0.0
        net = np.asarray(consumption - discharge - generation, dtype=float) * np.ones(len(frame))
        gas = (
            (U[model.gas_control_indices] / model.gas_efficiencies[:, None]).sum(axis=0)
            if model.gas_control_indices
            else np.zeros(len(frame))
        )
        heat = U[model.heat_control_indices].sum(axis=0) if model.heat_control_indices else np.zeros(len(frame))
        return pd.DataFrame(
            {
                "P_consumption": np.asarray(consumption, dtype=float) * np.ones(len(frame)),
                "P_discharge": np.asarray(discharge, dtype=float) * np.ones(len(frame)),
                "P_generation": np.asarray(generation, dtype=float) * np.ones(len(frame)),
                "P_net": net,
                "P_import": np.maximum(net, 0.0),
                "P_export": np.maximum(-net, 0.0),
                "P_gas": gas,
                "P_heat": heat,
            },
            index=frame.index,
        )


__all__ = ["METER_COLUMNS", "Meter"]
