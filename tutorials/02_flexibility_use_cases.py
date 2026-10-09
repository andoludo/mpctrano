# %% [markdown]
# # 02 - Flexibility use cases: what the house can do for the grid, and what it costs
#
# **Learning objectives**
#
# 1. Quantify **implicit flexibility**: what the price-aware controller saves against a
#    price-agnostic one under day-ahead prices, hourly and at the 15-minute market time unit, and
#    how much energy it moves in time.
# 2. Shave the **peak** under a capacity tariff on the monthly quarter-hour peak (the Flemish
#    *capaciteitstarief*): how the controller trades energy cost against capacity cost.
# 3. Deliver **explicit flexibility**: respond to an up or down activation request against a
#    baseline, measure the delivered energy and the **rebound**.
# 4. Map the **flexibility envelope** (power versus duration, at what cost) and the response to a
#    price step (the flexibility function).
# 5. See what **negative prices** and PV self-consumption do to the dispatch, with and without a
#    battery and an EV.
#
# The definitions follow IEA EBC Annex 67 and the characterisation of Reynders, Diriken and
# Saelens (2017), the flexibility function of Junker et al. (2018) and the BOPTEST KPIs of Blum et
# al. (2021); every source is listed in `tutorials/references.md`. The closed-loop runs come from
# `tutorials/cache/flexibility/` (`make cache`); the live widgets solve one open-loop plan.
#
# **Runtimes** (this container): a 3-day hourly closed loop about 18 s, a 3-day 15-minute closed
# loop about 2 min, the flexibility envelope (two directions, four durations) about 20 s.

# %%
import pandas as pd
from IPython.display import display

from mpctrano import ComfortSpec, FlexibilityEnvelope, ModelPredictiveController, MPCSettings, compare, price_response
from mpctrano.flexibility import flexibility_function
from mpctrano.notebook import closed_loop_explorer, kpi_table, quick_mode, timed
from mpctrano.plot import plot_activation, plot_closed_loop, plot_envelope
from mpctrano.scenarios import build_plant
from tutorials import _flexibility as study
from tutorials._house import PERFECT, START

pd.set_option("display.width", 200)
print(f"quick mode: {quick_mode()}; cached runs: {study.DAYS:g} days from {START.date()}")

# %% [markdown]
# ## 1. Implicit flexibility: day-ahead prices, hourly and 15-minute
#
# Since 1 October 2025 the single day-ahead coupling trades 15-minute products. A controller that
# plans on the quarter-hour sees sharper price ramps and can follow them with a slow floor heating
# only through the compressor, so the gain against hourly planning is modest for a house and
# larger for fast systems. `value_of_flexibility` costs a *price-agnostic* run (flat price, same
# comfort band) with the same tariff: the difference is what the flexibility is worth.

# %%
implicit = {label: study.cached(**parts) for label, parts in study.implicit_runs().items()}
table = kpi_table(implicit, ("total_cost", "energy_kwh", "comfort_violation_kh", "peak_power_kw", "low_price_share"))
table["mean solve time [s]"] = [round(result.frame["solve_time"].mean(), 3) for result in implicit.values()]
display(table)
display(plot_closed_loop(implicit["base | 15min"], "base house, 15-minute day-ahead prices"))

# %% [markdown]
# The share of energy bought below the median price (`low_price_share`) is the simplest signature
# of load shifting; `price_response` gives the whole distribution: how many kWh were bought at each
# price level.

# %%
display(price_response(implicit["base | 1h"]).round(3).tail(8))

# %% [markdown]
# ## 2. Peak shaving under the capacity tariff
#
# The Flemish distribution tariff bills the average of the twelve monthly peaks (highest
# quarter-hour average power of the month, floored at 2.5 kW) at about 50 EUR/kW/year (2025
# levels: 48.9 to 56.9 depending on the area). One extra kilowatt of monthly peak therefore costs
# about 4 EUR; the controller carries the peak of the month so far and weighs the energy saving of
# a cheap-hour boost against the capacity cost of the higher peak. With hourly steps the peak is
# the hourly average, which underestimates the quarter-hour peak.

# %%
capacity = {label: study.cached(**parts) for label, parts in study.capacity_runs().items()}
rows = {}
for variant in ("base", "old", "radiators"):
    subset = {label: result for label, result in capacity.items() if label.startswith(variant)}
    comparison = compare(subset, baseline=f"{variant} | 0 EUR/kW/year")
    rows[variant] = comparison[
        ["total_cost", "capacity_cost", "peak_power_kw", "peak_reduction_pct", "comfort_violation_kh"]
    ]
display(pd.concat(rows.values()).round(2))

# %%
dimensions = {"variant": ["base", "old", "radiators"], "capacity_price": list(study.CAPACITY_PRICES)}
display(closed_loop_explorer(study.CACHE, study.capacity_runs(), study.compute, dimensions))

# %% [markdown]
# ## 3. Explicit flexibility: activation requests and the rebound
#
# An aggregator or the DSO asks for a change of the billed power during a window against a
# baseline (here the plan of the same controller without the request). The controller receives
# the request as a soft cap (down) or floor (up) on the billed power, delivers what the comfort
# band allows and **rebounds** afterwards: the heat not delivered during a shed has to be
# delivered later, the heat absorbed during an upward activation is saved later. The metrics
# (`ActivationResult.metrics`) give the delivered energy, the rebound energy and peak and their
# ratio, in the spirit of the available storage capacity and storage efficiency of Reynders et al.

# %%
results = {label: study.activation("base", label) for label in study.ACTIVATIONS}
metrics = pd.DataFrame({label: result.metrics for label, result in results.items()}).T
display(metrics.round(2))
display(plot_activation(results["shed 1 kW, 2 h at 17h"], "base house: shed 1 kW for 2 h at 17h, second day"))

# %% [markdown]
# Notice the **rebound peak** after a shed and the lower consumption after an absorption, and
# how the delivery ratio drops when the request exceeds what the comfort band leaves (long sheds
# in the evening). The `big_heat_pump` variant delivers more because its compressor limit is higher:

# %%
big = {label: study.activation("big_heat_pump", label) for label in study.ACTIVATIONS}
display(
    pd.DataFrame({label: result.metrics for label, result in big.items()})
    .T[["delivered_energy_kwh", "delivery_ratio", "rebound_energy_kwh", "rebound_ratio", "cost_change"]]
    .round(2)
)

# %% [markdown]
# ## 4. Flexibility envelope and flexibility function
#
# The **envelope** answers the aggregator's question before any request: from the current state,
# how much power can the house shed or absorb for one, two, four or eight hours without leaving the
# comfort band, and what does that plan cost against the unconstrained one? It is a sweep of
# constrained MPC solves (bisection on the cap or floor of the billed power).

# %%
house = study.building("base")
model = house.control
tariffs = study.price_options(START, study.DAYS, study.COMPONENTS)
controller = ModelPredictiveController(model, MPCSettings(horizon=24), ComfortSpec(), tariffs["day-ahead"])
at = START + pd.Timedelta(hours=6)
with timed("flexibility envelope"):
    envelope = FlexibilityEnvelope.compute(
        controller,
        model.initial_state,
        house.forecaster.forecast(at, 24, 3600.0),
        durations=(1.0, 2.0, 4.0, 8.0),
        iterations=6,
    )
display(envelope.frame.round(2))
display(plot_envelope(envelope, "base house at 6h, day-ahead prices"))

# %% [markdown]
# The **flexibility function** of Junker et al. characterises the demand response to a penalty
# (price) step rather than to a power request: here the price is multiplied by three for three
# hours and the change of the billed power is read like an activation.

# %%
run = study.scenario("day-ahead", tariffs["day-ahead"], study.comfort_spec(*study.BAND), START, study.LIVE_DAYS)
with timed("flexibility function"):
    response = flexibility_function(
        model, run, START + pd.Timedelta(hours=17), 3.0, price_factor=3.0,
        plant=lambda: build_plant(model, PERFECT, 3600.0),
    )  # fmt: skip
display(pd.Series(response.metrics).round(2).to_frame("price step x3, 3 h"))
display(plot_activation(response, "flexibility function: price x3 for 3 h at 17h on the first day"))

# %% [markdown]
# ## 5. Negative prices and PV self-consumption
#
# With PV on the roof the meter exports at midday; a negative wholesale price (a renewable
# surplus) turns the midday import into something the controller *wants* to do: it pre-heats the
# house and the tank, charges the battery and the vehicle when they are there. Self-consumption is
# the share of the generation used on site.

# %%
pv = {label: study.cached(**parts) for label, parts in study.pv_runs().items()}
display(
    kpi_table(
        pv,
        (
            "total_cost",
            "energy_kwh",
            "export_kwh",
            "export_revenue",
            "generation_kwh",
            "self_consumption",
            "comfort_violation_kh",
        ),
    )
)
display(
    plot_closed_loop(pv["pv_battery_ev | day-ahead + negative midday"], "PV, battery and EV: negative midday prices")
)

# %% [markdown]
# ## Key takeaways
#
# * Implicit flexibility pays a few percent on a day-ahead signal for a well-insulated house on
#   floor heating; the 15-minute resolution changes the plan more than the bill, because the
#   compressor, not the price, limits how fast the house can follow.
# * Under the capacity tariff the controller flattens the morning boost: the peak reduction is
#   worth about 4 EUR per kW and month, which competes with the energy saving of concentrating
#   the heating in the cheap hours.
# * Explicit activations are delivered as long as the comfort band and the compressor leave room;
#   the rebound returns a large share of the shed energy within a few hours, with a peak that the
#   DSO must expect, which is the argument for coordinating the activations (tutorial 06).
# * The envelope is asymmetric: the house absorbs more than it sheds in the morning (cold slab,
#   compressor headroom) and the reverse in the evening.
# * Negative prices reward absorption, and the storage order follows the round-trip efficiency:
#   tank and floor first, battery second, vehicle when plugged in.
#
# ## Further reading
#
# `tutorials/references.md`: Annex 67 and Annex 82 for the definitions, Reynders, Diriken and
# Saelens (2017) and Stinner et al. (2016) for the characterisation, Junker et al. (2018) for the
# flexibility function, the Vlaamse Nutsregulator sheets for the capacity tariff, and section B for
# the market context picked up in tutorial 07.
