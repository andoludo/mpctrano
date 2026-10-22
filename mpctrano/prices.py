"""Electricity price signals for the economic MPC: wholesale prices, retail stack and synthetic shapes.

A :class:`PriceSignal` holds a **wholesale** price series [EUR/MWh] (day-ahead market, hourly or
15-minute) together with the retail components added on top at the meter: grid fees, taxes and
levies [EUR/kWh] and VAT. :meth:`PriceSignal.retail` gives the price the building pays [EUR/kWh]
and :meth:`PriceSignal.tariff` turns it into the :class:`~mpctrano.tariff.DynamicTariff` of the
controller. Transformations (:meth:`resample`, :meth:`with_spike`, :meth:`with_negative_hours`,
:meth:`relocate`) build the scenarios of the tutorials from one cached sample of real prices.

Real prices are cached in ``mpctrano/data/prices/<name>.csv`` (``time,price_eur_mwh``) and loaded
with :meth:`PriceSignal.from_cached`; ``tutorials/scripts/fetch_prices.py`` downloads them from the
Energy-Charts API (Fraunhofer ISE, no key) or the ENTSO-E Transparency Platform (security token).
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from datetime import datetime, UTC
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from mpctrano.tariff import DynamicTariff, PriceSeries

MWH_TO_KWH = 1000.0
Resolution = Literal["15min", "1h"]
RESOLUTIONS: dict[Resolution, pd.Timedelta] = {"15min": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1)}
DATA_DIRECTORY = "data/prices"


class RetailComponents(BaseModel):
    """What the meter adds to the wholesale price."""

    model_config = ConfigDict(frozen=True)

    grid_fee: float = Field(0.0, ge=0, description="Distribution and transmission fees [EUR/kWh]")
    taxes: float = Field(0.0, ge=0, description="Taxes, levies and surcharges [EUR/kWh]")
    vat: float = Field(0.0, ge=0, lt=1, description="VAT rate applied on top of the whole stack")
    export_share: float = Field(
        1.0, ge=0, description="Export remuneration as a share of the wholesale price (1 = injection at market price)"
    )


class PriceSignal(RetailComponents):
    """Wholesale price series [EUR/MWh] and the retail components added at the meter."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    series: PriceSeries = Field(description="Wholesale price [EUR/MWh] indexed by the start of each interval")
    source: str = Field("", description="Provenance of the series (data set, licence, download date)")

    # ----------------------------------------------------------------- constructors
    @classmethod
    def from_cached(cls, name: str, components: RetailComponents | None = None) -> PriceSignal:
        """A sample shipped with the package (``mpctrano/data/prices/<name>.csv``)."""
        directory = resources.files("mpctrano") / DATA_DIRECTORY
        path = directory / f"{name}.csv"
        if not path.is_file():
            available = sorted(
                entry.name.removesuffix(".csv") for entry in directory.iterdir() if entry.name.endswith(".csv")
            )
            raise FileNotFoundError(
                f"No cached price sample '{name}'. Available: {available or 'none'}. "
                "Download samples with `python tutorials/scripts/fetch_prices.py` (see tutorials/README.md)."
            )
        with resources.as_file(path) as file:
            signal = cls.from_csv(file, name=name, components=components)
        provenance = directory / "provenance.md"
        return signal.model_copy(update={"source": _provenance(provenance, name)}) if provenance.is_file() else signal

    @classmethod
    def from_csv(
        cls,
        path: Path | str,
        name: str | None = None,
        time_column: str = "time",
        price_column: str = "price_eur_mwh",
        unit: Literal["EUR/MWh", "EUR/kWh"] = "EUR/MWh",
        components: RetailComponents | None = None,
    ) -> PriceSignal:
        frame = pd.read_csv(path)
        series = pd.Series(frame[price_column].to_numpy(dtype=float), index=pd.to_datetime(frame[time_column]))
        if unit == "EUR/kWh":
            series = series * MWH_TO_KWH
        return cls(name=name or Path(path).stem, series=series, **(components or RetailComponents()).model_dump())

    @classmethod
    def synthetic(
        cls,
        start: pd.Timestamp | str,
        days: int,
        resolution: Resolution = "1h",
        base: float = 90.0,
        amplitude: float = 60.0,
        noise: float = 10.0,
        seed: int | None = 0,
        name: str = "synthetic",
        components: RetailComponents | None = None,
    ) -> PriceSignal:
        """Day-ahead-like prices [EUR/MWh]: morning and evening peaks, a night trough, random variations."""
        step = RESOLUTIONS[resolution]
        index = pd.date_range(pd.Timestamp(start), periods=int(days * pd.Timedelta(days=1) / step), freq=step)
        hour = (index.hour + index.minute / 60).to_numpy()
        profile = (
            np.exp(-((hour - 8) ** 2) / 8) + np.exp(-((hour - 19) ** 2) / 8) - 0.6 * np.exp(-((hour - 3) ** 2) / 12)
        )
        generator = np.random.default_rng(seed)
        prices = base + amplitude * profile + noise * generator.standard_normal(len(index))
        return cls(
            name=name,
            series=pd.Series(prices, index=index),
            source="synthetic",
            **(components or RetailComponents()).model_dump(),
        )

    # ----------------------------------------------------------------- properties
    @property
    def resolution(self) -> pd.Timedelta:
        if len(self.series) < 2:
            raise ValueError("A price signal needs at least two samples.")
        return pd.Timedelta(pd.Series(self.series.index).diff().dropna().mode().iloc[0])

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.series.index[0])

    @property
    def end(self) -> pd.Timestamp:
        """End of the last interval."""
        return pd.Timestamp(self.series.index[-1]) + self.resolution

    def retail(self) -> pd.Series:
        """Price paid for the imported energy [EUR/kWh]: ``(wholesale + fees + taxes) * (1 + VAT)``."""
        return (self.series / MWH_TO_KWH + self.grid_fee + self.taxes) * (1 + self.vat)

    def export(self) -> pd.Series:
        """Price received for the exported energy [EUR/kWh] (no fees, no VAT)."""
        return self.series / MWH_TO_KWH * self.export_share

    def tariff(self, capacity_charge: float = 0.0) -> DynamicTariff:
        """The retail price (and the export remuneration) as the tariff of the controller."""
        return DynamicTariff(series=self.retail(), export_series=self.export(), capacity_charge=capacity_charge)

    def statistics(self) -> pd.Series:
        """Mean, spread and the share of negative intervals of the wholesale price [EUR/MWh]."""
        values = self.series.to_numpy(dtype=float)
        daily = self.series.groupby(pd.DatetimeIndex(self.series.index).normalize())
        return pd.Series(
            {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean_daily_spread": float((daily.max() - daily.min()).mean()),
                "negative_share": float((values < 0).mean()),
                "intervals": float(len(values)),
            },
            name=self.name,
        )

    # ----------------------------------------------------------------- transformations
    def slice(
        self, start: pd.Timestamp | str, end: pd.Timestamp | str | None = None, days: float | None = None
    ) -> PriceSignal:
        """The intervals from ``start`` to ``end`` (exclusive) or over ``days``."""
        start = pd.Timestamp(start)
        if end is None:
            if days is None:
                raise ValueError("Give either end or days.")
            end = start + pd.Timedelta(days=days)
        series = self.series.loc[(self.series.index >= start) & (self.series.index < pd.Timestamp(end))]
        if series.empty:
            raise ValueError(f"The price signal {self.name} does not cover {start} .. {end}.")
        return self.model_copy(update={"series": series})

    def relocate(self, start: pd.Timestamp | str, align_weekday: bool = True) -> PriceSignal:
        """Shift the series in time so that it starts on ``start`` (on the same weekday by default).

        Lets a cached sample of real prices be played over the weather period of a scenario.
        """
        start = pd.Timestamp(start)
        offset = start - self.start.normalize()
        if align_weekday:
            offset -= pd.Timedelta(days=(start.dayofweek - self.start.dayofweek) % 7)
        series = self.series.copy()
        series.index = series.index + offset
        return self.model_copy(update={"series": series, "name": f"{self.name}@{start.date()}"})

    def resample(self, resolution: Resolution) -> PriceSignal:
        """Hourly to 15-minute (held) or 15-minute to hourly (averaged)."""
        step = RESOLUTIONS[resolution]
        if step == self.resolution:
            return self
        if step < self.resolution:
            index = pd.date_range(self.start, self.end, freq=step, inclusive="left")
            series = self.series.reindex(index, method="ffill")
        else:
            series = self.series.resample(step).mean()
        return self.model_copy(update={"series": series, "name": f"{self.name}_{resolution}"})

    def with_spike(self, start: pd.Timestamp | str, hours: float, price: float) -> PriceSignal:
        """Set the wholesale price to ``price`` [EUR/MWh] during ``hours`` from ``start`` (scarcity event)."""
        return self._override(start, hours, price, suffix="spike")

    def with_negative_hours(self, start: pd.Timestamp | str, hours: float, price: float = -10.0) -> PriceSignal:
        """Negative wholesale prices during ``hours`` from ``start`` (renewable surplus)."""
        return self._override(start, hours, price, suffix="negative")

    def with_components(
        self, grid_fee: float, taxes: float, vat: float = 0.0, export_share: float = 1.0
    ) -> PriceSignal:
        return self.model_copy(update={"grid_fee": grid_fee, "taxes": taxes, "vat": vat, "export_share": export_share})

    def _override(self, start: pd.Timestamp | str, hours: float, price: float, suffix: str) -> PriceSignal:
        start = pd.Timestamp(start)
        end = start + pd.Timedelta(hours=hours)
        series = self.series.copy()
        mask = (series.index >= start) & (series.index < end)
        if not mask.any():
            raise ValueError(f"{start} .. {end} is outside the price signal {self.name}.")
        series[mask] = price
        return self.model_copy(update={"series": series, "name": f"{self.name}_{suffix}"})

    def to_csv(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        times = [stamp.strftime("%Y-%m-%dT%H:%M") for stamp in pd.DatetimeIndex(self.series.index)]
        frame = pd.DataFrame({"time": times, "price_eur_mwh": self.series.to_numpy()})
        frame.to_csv(path, index=False)
        return path


# --------------------------------------------------------------------------- parsers of the data sources
def parse_energy_charts(payload: str | bytes | dict[str, Any], timezone_name: str = "Europe/Brussels") -> pd.Series:
    """Prices [EUR/MWh] from the JSON of ``https://api.energy-charts.info/price`` (local, naive index)."""
    data = json.loads(payload) if isinstance(payload, str | bytes) else payload
    seconds = np.asarray(data["unix_seconds"], dtype="int64")
    index = pd.to_datetime(seconds, unit="s", utc=True).tz_convert(timezone_name).tz_localize(None)
    unit = str(data.get("unit", "EUR/MWh"))
    if unit not in ("EUR/MWh", "EUR / MWh"):
        raise ValueError(f"Unexpected unit {unit!r} in the Energy-Charts response.")
    return pd.Series(np.asarray(data["price"], dtype=float), index=index).dropna()


def parse_entsoe_day_ahead(payload: str | bytes, timezone_name: str = "Europe/Brussels") -> pd.Series:
    """Prices [EUR/MWh] from an ENTSO-E ``Publication_MarketDocument`` (document type A44).

    Missing positions (curve type A03, from 2025) are filled by holding the previous point.
    """
    root = ET.fromstring(payload)  # noqa: S314 - trusted response of the ENTSO-E API
    namespace = root.tag.split("}")[0].strip("{")
    ns = {"ns": namespace}
    if root.tag.endswith("Acknowledgement_MarketDocument"):
        reason = root.findtext(".//ns:Reason/ns:text", default="unknown reason", namespaces=ns)
        raise ValueError(f"ENTSO-E returned no data: {reason}")
    pieces: list[pd.Series] = []
    for period in root.iterfind(".//ns:TimeSeries/ns:Period", ns):
        start = _entsoe_time(period.findtext("ns:timeInterval/ns:start", default="", namespaces=ns))
        end = _entsoe_time(period.findtext("ns:timeInterval/ns:end", default="", namespaces=ns))
        resolution = pd.Timedelta(period.findtext("ns:resolution", default="PT60M", namespaces=ns))
        points = {
            int(point.findtext("ns:position", default="0", namespaces=ns)): float(
                point.findtext("ns:price.amount", default="nan", namespaces=ns)
            )
            for point in period.iterfind("ns:Point", ns)
        }
        index = pd.date_range(start, end, freq=resolution, inclusive="left")
        values = pd.Series([points.get(position + 1, np.nan) for position in range(len(index))], index=index)
        pieces.append(values.ffill())
    if not pieces:
        raise ValueError("No Period in the ENTSO-E document.")
    series = pd.concat(pieces).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series.index = pd.DatetimeIndex(series.index).tz_convert(timezone_name).tz_localize(None)
    return pd.Series(series, dtype=float)


def _entsoe_time(value: str) -> pd.Timestamp:
    return pd.Timestamp(datetime.strptime(value, "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC))


def _provenance(path: Any, name: str) -> str:  # noqa: ANN401 - importlib.resources traversable
    """The paragraph of ``provenance.md`` describing ``name`` (first line of its section)."""
    lines = str(path.read_text()).splitlines()
    for position, line in enumerate(lines):
        if line.startswith("## ") and name in line:
            body = [text for text in lines[position + 1 :] if text.strip()]
            return body[0].strip() if body else line
    return ""


def daily_profile(signal: PriceSignal) -> pd.DataFrame:
    """Mean, 10 % and 90 % quantiles of the wholesale price per time of day [EUR/MWh]."""
    frame = pd.DataFrame({"price": signal.series})
    frame["time_of_day"] = frame.index.hour + frame.index.minute / 60  # type: ignore[attr-defined]
    grouped = frame.groupby("time_of_day")["price"]
    return pd.DataFrame({"mean": grouped.mean(), "q10": grouped.quantile(0.1), "q90": grouped.quantile(0.9)})


__all__ = [
    "PriceSignal",
    "Resolution",
    "RetailComponents",
    "daily_profile",
    "parse_energy_charts",
    "parse_entsoe_day_ahead",
]
