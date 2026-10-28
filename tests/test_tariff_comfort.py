from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from mpctrano import (
    CapacityTariff,
    ComfortSchedule,
    ComfortSpec,
    DemandResponseEvent,
    DynamicTariff,
    FlatTariff,
    TimeOfUseTariff,
)
from mpctrano.metrics import ControlMetrics, Prices
from mpctrano.scenarios import Scenario

INDEX = pd.date_range("2023-01-16", periods=48, freq="h")  # Monday and Tuesday
THREE_ZONES = SimpleNamespace(zone_names=["space_001", "space_002", "space_003"])


def test_flat_tariff() -> None:
    signal = FlatTariff(price=0.3, capacity_charge=5).signal(INDEX)
    assert np.all(signal.prices == 0.3)
    assert np.all(np.isinf(signal.power_limits))
    assert signal.capacity_charge == 5
    assert FlatTariff(price=0.3).cost(INDEX, np.full(48, 1000.0), 3600) == pytest.approx(0.3 * 48)


def test_time_of_use_tariff() -> None:
    tariff = TimeOfUseTariff(peak_price=0.4, off_peak_price=0.1, peak_hours=(7, 22))
    prices = tariff.energy_price(INDEX)
    assert prices[6] == 0.1 and prices[7] == 0.4 and prices[21] == 0.4 and prices[22] == 0.1
    weekend = pd.date_range("2023-01-14 12:00", periods=1, freq="h")
    assert tariff.energy_price(weekend)[0] == 0.1
    assert TimeOfUseTariff(weekend_peak=True).energy_price(weekend)[0] == 0.32
    with pytest.raises(ValidationError):
        TimeOfUseTariff(peak_hours=(22, 7))


def test_dynamic_tariff_holds_prices_and_checks_coverage() -> None:
    tariff = DynamicTariff.synthetic_day_ahead("2023-01-16", days=2, seed=1)
    prices = tariff.energy_price(INDEX)
    assert len(prices) == 48 and np.all(prices > 0)
    quarter = pd.date_range("2023-01-16 00:15", periods=2, freq="15min")
    assert tariff.energy_price(quarter)[0] == prices[0]  # held between hourly samples
    with pytest.raises(ValueError, match="does not cover"):
        tariff.energy_price(pd.date_range("2023-02-01", periods=2, freq="h"))


def test_dynamic_tariff_from_csv(tmp_path) -> None:
    series = pd.Series([0.1, 0.2], index=pd.date_range("2023-01-16", periods=2, freq="h"), name="price")
    series.to_csv(tmp_path / "prices.csv", index_label="time")
    tariff = DynamicTariff.from_csv(tmp_path / "prices.csv")
    assert tariff.energy_price(series.index).tolist() == [0.1, 0.2]


def test_demand_response_events() -> None:
    event = DemandResponseEvent(
        start=pd.Timestamp("2023-01-16 17:00"),
        end=pd.Timestamp("2023-01-16 20:00"),
        price_multiplier=3,
        power_limit=2000,
    )
    tariff = TimeOfUseTariff(peak_price=0.3, off_peak_price=0.1).with_events([event])
    signal = tariff.signal(INDEX)
    assert signal.prices[17] == pytest.approx(0.9) and signal.prices[20] == pytest.approx(0.3)
    assert signal.power_limits[17] == 2000 and np.isinf(signal.power_limits[16])
    assert tariff.kind == "demand_response"
    assert (
        Scenario.model_validate_json(Scenario(name="x", description="", tariff=tariff).model_dump_json()).tariff.kind
        == "demand_response"
    )


def test_power_floors_and_notice_of_the_events() -> None:
    known = FlatTariff().with_events([DemandResponseEvent(start=INDEX[17], end=INDEX[19], power_floor=1500)])
    assert known.signal(INDEX).floors[17] == 1500 and known.signal(INDEX).floors[19] == 0
    notified = FlatTariff().with_events(
        [DemandResponseEvent(start=INDEX[17], end=INDEX[19], power_floor=1500, notice_hours=2)]
    )
    assert notified.signal(INDEX[15:]).floors[2] == 1500  # notified two hours ahead
    assert notified.signal(INDEX[10:]).floors[7] == 0  # not yet notified at 10h
    assert FlatTariff().signal(INDEX).floors.sum() == 0


def test_capacity_tariff_monthly_charge_and_peaks() -> None:
    tariff = CapacityTariff(base=FlatTariff(price=0.25), capacity_price=48.0)
    assert tariff.capacity_charge == pytest.approx(4.0)
    index = pd.date_range("2023-01-30", periods=72, freq="h")
    power = np.zeros(72)
    power[10] = 6000.0  # January peak 6 kW
    power[60] = 1000.0  # February peak below the 2.5 kW floor
    assert list(tariff.monthly_peaks(index, power).round(2)) == [6.0, 2.5]
    assert tariff.capacity_cost(index, power) == pytest.approx((6.0 + 2.5) * 4.0)
    np.testing.assert_allclose(tariff.signal(index).prices, 0.25)
    with_event = tariff.with_events(
        [DemandResponseEvent(start="2023-01-30 10:00", end="2023-01-30 12:00", power_floor=2000.0)]
    )
    assert isinstance(with_event, CapacityTariff)
    assert with_event.power_floor(index)[10] == 2000.0 and with_event.power_floor(index)[13] == 0.0
    energy = pd.DataFrame({"fuel": 0.0, "bought": power / 1000, "sold": 0.0}, index=index)
    band = np.full((1, 72), 293.15)
    metrics = ControlMetrics.compute(energy, Prices.of(tariff, index), band, band, band, 3600.0)
    assert metrics.capacity_cost == pytest.approx((6.0 + 2.5) * 4.0)


def test_the_export_is_paid_at_the_export_price_of_the_tariff() -> None:
    series = pd.Series(0.30, index=INDEX)
    export = pd.Series(0.05, index=INDEX)
    with_export = DynamicTariff(series=series, export_series=export)
    assert with_export.has_export_price and with_export.with_events([]).has_export_price
    assert CapacityTariff(base=with_export).has_export_price
    np.testing.assert_allclose(Prices.of(with_export, INDEX).feed_in, 0.05)
    np.testing.assert_allclose(Prices.of(DynamicTariff(series=series), INDEX).feed_in, 0.30)  # net metering
    np.testing.assert_allclose(Prices.of(with_export, INDEX, feed_in_price=0.0).feed_in, 0.0)
    energy = pd.DataFrame({"fuel": 0.0, "bought": 0.0, "sold": 1.0, "generated": 2.0}, index=INDEX)
    band = np.full((1, 48), 293.15)
    metrics = ControlMetrics.compute(energy, Prices.of(with_export, INDEX), band, band, band, 3600.0)
    assert metrics.energy_cost == pytest.approx(-0.05 * 48)
    assert metrics.generation_kwh == pytest.approx(96.0)
    assert metrics.self_consumption == pytest.approx(0.5)


def test_comfort_schedule_bounds_and_transforms() -> None:
    schedule = ComfortSchedule(occupied_lower=21, occupied_upper=24, unoccupied_lower=17, unoccupied_upper=27)
    lower, upper = schedule.bounds(INDEX)
    assert lower[7] == pytest.approx(294.15) and lower[6] == pytest.approx(290.15)
    assert upper[22] == pytest.approx(300.15)
    wide = schedule.widen(1.0)
    assert wide.occupied_lower == 20 and wide.unoccupied_upper == 28
    upper = schedule.widen(2.0, "upper")
    assert upper.occupied_lower == 21 and upper.occupied_upper == 26
    with pytest.raises(ValueError, match="side"):
        schedule.widen(1.0, "sideways")
    constant = schedule.constant()
    assert constant.unoccupied_lower == 21
    spec = ComfortSpec(default=schedule, zones={"space_003": schedule.constant()})
    bounds = spec.bounds(THREE_ZONES, INDEX)
    assert bounds.lower.shape == (3, 48)
    assert bounds.lower[2, 3] == pytest.approx(294.15)  # constant band in zone 3
    assert bounds.lower[0, 3] == pytest.approx(290.15)
    widened = spec.map("widen", 0.5, "both")
    assert widened.zones["space_003"].occupied_lower == 20.5
    assert "space_001_lower" in bounds.frame().columns
    with pytest.raises(ValidationError):
        ComfortSchedule(occupied_lower=25, occupied_upper=24)
