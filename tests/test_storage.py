"""Electrical storages at the meter: the schedules of a vehicle, a battery and a vehicle in a plan."""

import pandas as pd
import pytest

from mpctrano import (
    Battery,
    ComfortSpec,
    ElectricVehicle,
    FastModel,
    ModelPredictiveController,
    MPCSettings,
    TimeOfUseTariff,
    WeatherForecaster,
)
from mpctrano.storage import storage_schedule
from tests.conftest import START


def test_the_vehicle_schedule() -> None:
    vehicle = ElectricVehicle(capacity_kwh=50, max_charge_w=7400, arrival_hour=18, departure_hour=7, driven_kwh=12)
    index = pd.date_range("2023-01-16", periods=48, freq="h")  # Monday and Tuesday
    plugged, departures = storage_schedule((vehicle,), index, 3600.0)
    assert plugged.shape == (1, 48)
    assert plugged[0, 6] == 1 and plugged[0, 7] == 0 and plugged[0, 17] == 0 and plugged[0, 18] == 1
    assert departures[0].sum() == 2 and departures[0, 7] == 1 and departures[0, 31] == 1
    weekdays = vehicle.model_copy(update={"weekdays_only": True})
    saturday = pd.date_range("2023-01-21", periods=24, freq="h")
    assert weekdays.plugged(saturday).min() == 1 and weekdays.departures(saturday, 3600.0).sum() == 0
    with pytest.raises(ValueError, match="departure energy"):
        ElectricVehicle(capacity_kwh=50, max_charge_w=7400, driven_kwh=40, departure_energy_kwh=30)
    battery = Battery(capacity_kwh=8, max_charge_w=3000, max_discharge_w=3000)
    assert battery.plugged(index).min() == 1 and battery.departures(index, 3600.0).sum() == 0
    assert battery.energy_bounds_kwh == (0.8, 8.0) and battery.initial_energy_kwh == 4.0


@pytest.mark.slow
def test_a_battery_buys_off_peak_and_a_vehicle_leaves_charged(heat_pump: FastModel) -> None:
    battery = Battery(capacity_kwh=8, max_charge_w=3000, max_discharge_w=3000, initial_soc=0.5)
    vehicle = ElectricVehicle(
        capacity_kwh=50, max_charge_w=7400, arrival_hour=18, departure_hour=7, driven_kwh=12, departure_energy_kwh=30,
        initial_soc=0.4,
    )  # fmt: skip
    settings = MPCSettings(horizon=24, batteries=(battery,), vehicles=(vehicle,))
    tariff = TimeOfUseTariff(peak_price=0.40, off_peak_price=0.10)
    controller = ModelPredictiveController(heat_pump, settings, ComfortSpec(), tariff)
    forecast = WeatherForecaster.from_model(heat_pump).forecast(START, 24, 3600.0)
    plan = controller.solve(heat_pump.initial_state, forecast)
    assert plan.success, plan.status
    assert plan.storage is not None
    storage = plan.storage
    peak = tariff.is_peak(plan.index)
    charge = storage["battery_charge_w"].to_numpy()[:-1]
    discharge = storage["battery_discharge_w"].to_numpy()[:-1]
    assert charge[~peak].sum() > charge[peak].sum()  # charges off peak
    assert discharge[peak].sum() > discharge[~peak].sum()  # discharges on peak
    soc = storage["battery_soc_kwh"].to_numpy()
    assert soc.min() >= 0.8 - 1e-6 and soc.max() <= 8.0 + 1e-6
    vehicle_soc = storage["vehicle_soc_kwh"].to_numpy()
    assert vehicle_soc[7] >= 30.0 - 0.5  # leaves at 7h with its departure energy
    assert vehicle_soc[8] == pytest.approx(vehicle_soc[7] - 12.0, abs=1e-6)  # and comes back 12 kWh lower
    assert storage["vehicle_charge_w"].to_numpy()[7:18].max() <= 1e-6  # no charging while away
    exchange = controller.advance(plan)
    assert controller.storage_state["battery"] == pytest.approx(soc[1])
    assert exchange == pytest.approx((charge[0] - discharge[0] + storage["vehicle_charge_w"].iloc[0]) / 1000)
