"""Tutorial 5 - Tariffs, prices and flexibility.

The same controller follows an energy tariff: flat, time-of-use, day-ahead prices, demand
response events (price spikes and power caps) or a capacity charge on the peak power. The
thermal mass and the comfort band are what make the building flexible: the MPC pre-heats when
the energy is cheap and lets the temperature float when it is expensive.

Costs are only comparable under the same tariff, so the value of the flexibility is measured
against a *price-agnostic* controller (same model, same comfort band, flat price) costed with
the same tariff: ``value_of_flexibility``. The controller can only exploit the storage it believes
in, so the model is calibrated first (tutorial 2).
"""

import pandas as pd

from mpctrano import (
    ComfortSchedule,
    ComfortSpec,
    DemandResponseEvent,
    DynamicTariff,
    FlatTariff,
    Project,
    Scenario,
    TimeOfUseTariff,
    compare,
    price_response,
    scenario_catalogue,
    value_of_flexibility,
)
from mpctrano.plot import plot_closed_loop, plot_comparison
from tutorials._common import MODEL, OUTPUT, days, save_figure

pd.set_option("display.width", 200)
columns = [
    "energy_kwh",
    "total_cost",
    "cost_saving_pct",
    "comfort_violation_kh",
    "peak_power_kw",
    "low_price_share",
    "shifted_energy_kwh",
]

project = Project.from_yaml(MODEL)
building = project.true_building()
project.calibrate(project.synthetic_data(days=days(7), building=building).data)
model = project.calibrated
catalogue = scenario_catalogue(start="2023-01-16", days=days(2))

# --- 1. What is the flexibility worth under each tariff? Price-aware vs price-agnostic MPC.
for name in ("time_of_use", "dynamic_price", "wide_comfort_band", "demand_response", "peak_shaving"):
    print(f"\n=== {name}: {catalogue[name].description}")
    print(value_of_flexibility(model, catalogue[name], plant=building)[columns].astype(float).round(2).to_string())

# --- 2. Why calibrate first: the uncalibrated Trano model mis-sizes the pre-heating and misses the comfort band.
print("\n=== time_of_use with the uncalibrated Trano model")
print(
    value_of_flexibility(project.model, catalogue["time_of_use"], plant=building)[columns]
    .astype(float)
    .round(2)
    .to_string()
)

# --- 3. Closed-loop trajectories and the energy bought at each price level (day-ahead prices).
dynamic = project.run(catalogue["dynamic_price"], plant=building)
save_figure(plot_closed_loop(dynamic), "05_dynamic_price")
print(price_response(dynamic).round(3).tail(6).to_string())

# --- 4. The same tariff, more or less room to pre-heat: the upper bound of the comfort band is the lever.
tariff = TimeOfUseTariff(peak_price=0.40, off_peak_price=0.10)
runs = {}
for delta in (0.0, 1.0, 2.0, 3.0):
    scenario = Scenario(
        name=f"upper+{delta:.0f}K",
        description="",
        tariff=tariff,
        comfort=ComfortSpec().map("widen", delta, "upper"),
        days=days(2),
    )
    runs[scenario.name] = project.run(scenario, plant=building)
comparison = compare(runs, baseline="upper+0K")
print(comparison[columns].round(2).to_string())
save_figure(plot_comparison(comparison), "05_comfort_band_vs_cost")

# --- 5. Peak shaving: a deep night setback creates a morning boost that a capacity charge spreads out.
setback = ComfortSpec(default=ComfortSchedule(unoccupied_lower=15.0))
peaks = {}
for charge in (0.0, 4.0, 8.0):
    scenario = Scenario(
        name=f"charge {charge:.0f} EUR/kW",
        description="",
        tariff=FlatTariff(price=0.25, capacity_charge=charge),
        comfort=setback,
        days=days(2),
    )
    peaks[scenario.name] = project.run(scenario, plant=building)
print(
    compare(peaks)[["energy_kwh", "energy_cost", "capacity_cost", "peak_power_kw", "comfort_violation_kh"]]
    .round(2)
    .to_string()
)

# --- 6. Your own tariff: day-ahead prices from a CSV (time, price), a capacity charge and a demand response event.
OUTPUT.mkdir(exist_ok=True)
prices = DynamicTariff.synthetic_day_ahead("2023-01-15", days=5, base=0.18, amplitude=0.18, seed=3).series
prices.rename("price").to_csv(OUTPUT / "day_ahead_prices.csv", index_label="time")
custom_tariff = DynamicTariff.from_csv(OUTPUT / "day_ahead_prices.csv", capacity_charge=2.0).with_events(
    [DemandResponseEvent(start="2023-01-16 18:00", end="2023-01-16 20:00", price_multiplier=2.0, power_limit=3000.0)]
)
custom = Scenario(
    name="my_tariff",
    description="day-ahead prices + capacity charge + DR event",
    tariff=custom_tariff,
    comfort=ComfortSpec().map("widen", 2.0, "upper"),
    days=days(2),
)
print(value_of_flexibility(model, custom, plant=building)[columns].astype(float).round(2).to_string())
