"""Cooling with ideal emitters: negative bounds, the COP and EER of their electricity."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from mpctrano import ComfortSchedule, ComfortSpec, FastModel, FlatTariff, ModelPredictiveController, MPCSettings
from mpctrano.simulation import metered_energy

OFFICE = Path(__file__).parent.parent / "tutorials" / "buildings" / "office" / "base.yaml"


def test_the_metered_energy_costs_heating_and_cooling_through_the_cop_and_eer() -> None:
    model = SimpleNamespace(meters={}, state_names=[], control_names=["q1", "q2"], heat_controls=["q1", "q2"])
    before = after = np.zeros(0)
    heating = metered_energy(model, before, after, np.array([3500.0, 0.0]), 3600.0, heat_cop=3.5, cooling_eer=3.0)
    assert heating["bought"] == pytest.approx(1.0)
    cooling = metered_energy(model, before, after, np.array([0.0, -3000.0]), 3600.0, heat_cop=3.5, cooling_eer=3.0)
    assert cooling["bought"] == pytest.approx(1.0)
    both = metered_energy(model, before, after, np.array([3500.0, -3000.0]), 3600.0, heat_cop=3.5, cooling_eer=3.0)
    assert both["bought"] == pytest.approx(2.0)


@pytest.fixture(scope="module")
def office() -> FastModel:
    return FastModel.from_yaml(OFFICE)


def test_control_bounds_can_be_lowered_below_zero(office: FastModel) -> None:
    lower, upper = office.control_bounds
    np.testing.assert_array_equal(lower, 0.0)
    cooling = office.with_control_bounds({"emitter_001_Q.Q_flow": (-12000.0, None)})
    lower_c, upper_c = cooling.control_bounds
    assert lower_c[cooling.control_names.index("emitter_001_Q.Q_flow")] == -12000.0
    np.testing.assert_array_equal(upper_c, upper)
    assert office.control_bounds[0][0] == 0.0  # the original is untouched
    with pytest.raises(KeyError):
        office.with_control_bounds({"nope": (0.0, 1.0)})


@pytest.mark.slow
def test_the_controller_cools_the_office_in_summer(office: FastModel) -> None:
    from mpctrano import WeatherForecaster

    cooling = office.with_control_bounds(dict.fromkeys(office.control_names, (-10000.0, None)))
    band = ComfortSpec(
        default=ComfortSchedule(
            occupied_lower=21.0,
            occupied_upper=24.0,
            unoccupied_lower=15.0,
            unoccupied_upper=30.0,
            occupied_hours=(8, 18),
        )
    )
    controller = ModelPredictiveController(
        cooling, MPCSettings(horizon=12, heat_cop=3.5, cooling_eer=3.0), band, FlatTariff(price=0.25)
    )
    start = "2023-07-17 06:00"
    x0 = cooling.initial_state.copy()
    x0[cooling.comfort_indices] = 273.15 + 27.0  # a warm office at 6h
    forecast = WeatherForecaster.from_model(cooling).forecast(start, 12, 3600.0)
    plan = controller.solve(x0, forecast)
    assert plan.success, plan.status
    assert plan.controls.to_numpy().min() < -500.0  # it cools
    assert plan.meter is not None and plan.meter["P_meter"].sum() > 0  # and pays electricity for it
    occupied = plan.comfort_temperatures.iloc[2:]  # from 8h
    assert np.all(occupied.to_numpy() - 273.15 <= 24.0 + 0.5)
