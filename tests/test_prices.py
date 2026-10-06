"""Price signals: synthetic shapes, retail stack, transformations and the parsers of the data sources."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mpctrano.prices import PriceSignal, RetailComponents, daily_profile, parse_energy_charts, parse_entsoe_day_ahead

ENTSOE_DOCUMENT = """<?xml version="1.0" encoding="UTF-8"?>
<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3">
  <TimeSeries>
    <currency_Unit.name>EUR</currency_Unit.name>
    <price_Measure_Unit.name>MWH</price_Measure_Unit.name>
    <curveType>A03</curveType>
    <Period>
      <timeInterval><start>2024-01-15T23:00Z</start><end>2024-01-16T03:00Z</end></timeInterval>
      <resolution>PT60M</resolution>
      <Point><position>1</position><price.amount>80.5</price.amount></Point>
      <Point><position>2</position><price.amount>75.0</price.amount></Point>
      <Point><position>4</position><price.amount>-5.0</price.amount></Point>
    </Period>
  </TimeSeries>
</Publication_MarketDocument>
"""


def test_synthetic_shape_and_resolution() -> None:
    hourly = PriceSignal.synthetic("2024-01-15", days=2, noise=0.0)
    quarter = PriceSignal.synthetic("2024-01-15", days=2, resolution="15min", noise=0.0)
    assert len(hourly.series) == 48
    assert len(quarter.series) == 192
    assert hourly.resolution == pd.Timedelta(hours=1)
    assert quarter.resolution == pd.Timedelta(minutes=15)
    assert hourly.series.idxmax().hour in (8, 19)
    assert hourly.series.idxmin().hour in (2, 3, 4)


def test_retail_stack_and_tariff() -> None:
    components = RetailComponents(grid_fee=0.08, taxes=0.02, vat=0.06)
    signal = PriceSignal.synthetic("2024-01-15", days=1, noise=0.0, components=components)
    retail = signal.retail()
    expected = (signal.series / 1000 + 0.10) * 1.06
    np.testing.assert_allclose(retail.to_numpy(), expected.to_numpy())
    tariff = signal.tariff(capacity_charge=3.0)
    assert tariff.capacity_charge == 3.0
    index = pd.date_range("2024-01-15", periods=4, freq="h")
    np.testing.assert_allclose(tariff.energy_price(index), retail.iloc[:4].to_numpy())
    assert signal.export().iloc[0] == pytest.approx(signal.series.iloc[0] / 1000)


def test_resample_both_ways() -> None:
    hourly = PriceSignal.synthetic("2024-01-15", days=1, noise=0.0)
    quarter = hourly.resample("15min")
    assert len(quarter.series) == 96
    np.testing.assert_allclose(quarter.series.to_numpy()[::4], hourly.series.to_numpy())
    back = quarter.resample("1h")
    np.testing.assert_allclose(back.series.to_numpy(), hourly.series.to_numpy())
    assert hourly.resample("1h") is hourly


def test_spike_negative_slice_relocate() -> None:
    signal = PriceSignal.synthetic("2024-01-15", days=3, noise=0.0)
    spiked = signal.with_spike("2024-01-16 17:00", hours=2, price=800.0)
    assert spiked.series.loc["2024-01-16 17:00"] == 800.0
    assert spiked.series.loc["2024-01-16 19:00"] == signal.series.loc["2024-01-16 19:00"]
    negative = signal.with_negative_hours("2024-01-16 12:00", hours=3)
    assert negative.statistics()["negative_share"] == pytest.approx(3 / 72)
    day = signal.slice("2024-01-16", days=1)
    assert len(day.series) == 24
    relocated = signal.relocate("2023-01-16")  # a Monday, like 2024-01-15
    assert relocated.start == pd.Timestamp("2023-01-16")
    assert relocated.start.dayofweek == signal.start.dayofweek
    with pytest.raises(ValueError, match="outside"):
        signal.with_spike("2024-02-01", hours=1, price=1.0)


def test_csv_round_trip_and_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    signal = PriceSignal.synthetic("2024-01-15", days=1, seed=3)
    path = signal.to_csv(tmp_path / "sample.csv")
    loaded = PriceSignal.from_csv(path, components=RetailComponents(grid_fee=0.05))
    np.testing.assert_allclose(loaded.series.to_numpy(), signal.series.to_numpy())
    assert loaded.name == "sample"
    assert loaded.grid_fee == 0.05
    with pytest.raises(FileNotFoundError, match="fetch_prices"):
        PriceSignal.from_cached("does_not_exist")


def test_daily_profile() -> None:
    profile = daily_profile(PriceSignal.synthetic("2024-01-15", days=5, noise=0.0))
    assert list(profile.columns) == ["mean", "q10", "q90"]
    assert len(profile) == 24
    assert profile["mean"].idxmax() in (8.0, 19.0)


def test_parse_energy_charts() -> None:
    payload = {"unix_seconds": [1705359600, 1705363200], "price": [80.5, 75.0], "unit": "EUR/MWh"}
    series = parse_energy_charts(payload)
    assert list(series.index) == [pd.Timestamp("2024-01-16 00:00"), pd.Timestamp("2024-01-16 01:00")]
    assert series.iloc[1] == 75.0
    with pytest.raises(ValueError, match="unit"):
        parse_energy_charts({**payload, "unit": "ct/kWh"})


def test_parse_entsoe_fills_missing_positions() -> None:
    series = parse_entsoe_day_ahead(ENTSOE_DOCUMENT)
    assert list(series.index) == list(pd.date_range("2024-01-16 00:00", periods=4, freq="h"))
    np.testing.assert_allclose(series.to_numpy(), [80.5, 75.0, 75.0, -5.0])
    with pytest.raises(ValueError, match="No Period"):
        parse_entsoe_day_ahead('<Publication_MarketDocument xmlns="x"></Publication_MarketDocument>')
