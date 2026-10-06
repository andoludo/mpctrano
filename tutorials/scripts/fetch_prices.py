"""Download day-ahead electricity prices into the cached samples of ``mpctrano.prices``.

Sources (no key needed unless stated):

* ``energy-charts``: the Energy-Charts API of Fraunhofer ISE (``https://api.energy-charts.info``),
  day-ahead prices per bidding zone, hourly before and 15-minute after the SDAC 15-minute
  market time unit.
* ``entsoe``: the ENTSO-E Transparency Platform RESTful API (document type A44), needs a
  security token (``--token`` or the ``ENTSOE_TOKEN`` environment variable).

Examples::

    uv run python tutorials/scripts/fetch_prices.py energy-charts 2024-01-01 2025-01-01 --name be_day_ahead_2024
    uv run python tutorials/scripts/fetch_prices.py entsoe 2025-10-01 2025-10-15 --name be_day_ahead_2025_10_15min

Each download appends a section to ``mpctrano/data/prices/provenance.md``.
"""

from __future__ import annotations

import datetime as dt
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from mpctrano.prices import PriceSignal, parse_energy_charts, parse_entsoe_day_ahead

DATA = Path(__file__).resolve().parents[2] / "mpctrano" / "data" / "prices"
BIDDING_ZONES = {
    "BE": "10YBE----------2",
    "NL": "10YNL----------L",
    "DE-LU": "10Y1001A1001A82H",
    "FR": "10YFR-RTE------C",
}
app = typer.Typer(add_completion=False, help=__doc__)


def _get(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "mpctrano-tutorials (fetch_prices.py)"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return bytes(response.read())


def _store(series: pd.Series, name: str, source: str, licence: str, start: str, end: str) -> Path:
    signal = PriceSignal(name=name, series=series, source=source)
    path = signal.to_csv(DATA / f"{name}.csv")
    today = dt.date.today().isoformat()
    with (DATA / "provenance.md").open("a") as handle:
        handle.write(
            f"\n## {name}\n\n{source}, {start} to {end} (local time), {len(series)} intervals, "
            f"resolution {signal.resolution}. Licence: {licence}. Downloaded {today}.\n"
        )
    typer.echo(f"{path}: {len(series)} intervals, {signal.statistics().round(1).to_dict()}")
    return path


@app.command("energy-charts")
def energy_charts(
    start: str,
    end: str,
    name: Annotated[str, typer.Option(help="Name of the cached sample")] = "be_day_ahead",
    zone: Annotated[str, typer.Option(help="Bidding zone, e.g. BE, NL, DE-LU")] = "BE",
) -> None:
    """Day-ahead prices from api.energy-charts.info (Fraunhofer ISE)."""
    query = urllib.parse.urlencode({"bzn": zone, "start": start, "end": end})
    series = parse_energy_charts(_get(f"https://api.energy-charts.info/price?{query}"))
    _store(
        series,
        name,
        f"Energy-Charts API (Fraunhofer ISE), day-ahead price, bidding zone {zone}",
        "CC BY 4.0 (Bundesnetzagentur | SMARD.de via Energy-Charts); verify on energy-charts.info",
        start,
        end,
    )


@app.command()
def entsoe(
    start: str,
    end: str,
    name: Annotated[str, typer.Option(help="Name of the cached sample")] = "be_day_ahead",
    zone: Annotated[str, typer.Option(help="Bidding zone, e.g. BE, NL, DE-LU")] = "BE",
    token: Annotated[str | None, typer.Option(envvar="ENTSOE_TOKEN", help="Security token")] = None,
) -> None:
    """Day-ahead prices (A44) from the ENTSO-E Transparency Platform, one request per month."""
    if not token:
        raise typer.BadParameter("An ENTSO-E security token is needed (--token or ENTSOE_TOKEN).")
    domain = BIDDING_ZONES[zone]
    pieces = []
    for period_start in pd.date_range(start, end, freq="MS", inclusive="left").union([pd.Timestamp(start)]):
        period_end = min(pd.Timestamp(end), (period_start + pd.offsets.MonthBegin(1)).normalize())
        query = urllib.parse.urlencode(
            {
                "securityToken": token,
                "documentType": "A44",
                "in_Domain": domain,
                "out_Domain": domain,
                "periodStart": period_start.strftime("%Y%m%d%H%M"),
                "periodEnd": period_end.strftime("%Y%m%d%H%M"),
            }
        )
        pieces.append(parse_entsoe_day_ahead(_get(f"https://web-api.tp.entsoe.eu/api?{query}")))
    series = pd.concat(pieces).sort_index()
    series = series[~series.index.duplicated(keep="last")]
    _store(
        series,
        name,
        f"ENTSO-E Transparency Platform, day-ahead prices (A44), {zone} ({domain})",
        "ENTSO-E Transparency Platform terms of use (attribution; verify before redistribution)",
        start,
        end,
    )


if __name__ == "__main__":
    app()
