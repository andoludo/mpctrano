# %% [markdown]
# # 02 - Flexibility use cases: what the house can do for the grid, and what it costs
#
# **Learning objectives**
#
# 1. Quantify **implicit flexibility**: what the price-aware controller saves against a price-agnostic one under
#    day-ahead prices, how much energy it moves in time, and what planning on the 15-minute market time unit
#    changes.
# 2. Shave the **peak** under a capacity tariff on the monthly peak (the Flemish *capaciteitstarief*): how the
#    controller trades energy cost against capacity cost, and why the floor of the tariff matters for a small
#    heat pump.
# 3. Deliver **explicit flexibility**: respond to an up or down activation request against a baseline, measure
#    the delivered energy and the **rebound**.
# 4. Map the **flexibility envelope** (power versus duration, at what cost) and the response to a price step (the
#    flexibility function).
# 5. See what **negative prices** do to a house with PV.
#
# The definitions follow IEA EBC Annex 67 and the characterisation of Reynders, Diriken and Saelens (2017), the
# flexibility function of Junker et al. (2018) and the BOPTEST KPIs of Blum et al. (2021); every source is listed
# in `tutorials/references.md`. The closed loops (against the true building of tutorial 01, with moving horizon
# estimation) come from `tutorials/cache/flexibility/` (`make cache`); the plans of section 1 are solved live.
#
# **Runtimes** (this container): a two-day hourly closed loop about ten minutes; a 24-hour plan 5 to 20 s at
# hourly steps, one to two minutes at 15-minute steps; the envelope about twenty plans.

# %%
import pandas as pd
from IPython.display import display

from mpctrano import ComfortSpec, ModelPredictiveController, MPCSettings, price_response
from mpctrano.flexibility import compare, shifted_energy
from mpctrano.notebook import closed_loop_explorer, kpi_table, quick_mode, timed
from mpctrano.plot import plot_activation, plot_closed_loop, plot_envelope, plot_plan
from tutorials import _flexibility as study
from tutorials._house import START, building

pd.set_option("display.width", 200)
print(f"quick mode: {quick_mode()}; cached runs: {study.DAYS:g} days from {START.date()}")
tariffs = study.tariffs()

# %% [markdown]
# ## 1. Implicit flexibility under day-ahead prices
#
# The *price-agnostic* controller has the same model, comfort band and constraints, but plans at the mean price of
# the period; costing its run with the day-ahead tariff gives what the house would pay without reacting to the
# price. The difference is what the flexibility is worth under that tariff (`value_of_flexibility`).

# %%
rows = {}
for variant in ("base", "old"):
    aware = study.cached(study="implicit", variant=variant, pricing="aware", days=study.DAYS)
    agnostic = study.cached(study="implicit", variant=variant, pricing="agnostic", days=study.DAYS)
    costed = agnostic.metrics_under(tariffs["day-ahead"])
    rows[variant] = {
        "aware cost": aware.metrics.total_cost,
        "agnostic cost": costed.total_cost,
        "saving %": 100 * (costed.total_cost - aware.metrics.total_cost) / costed.total_cost,
        "aware kWh": aware.metrics.energy_kwh,
        "agnostic kWh": costed.energy_kwh,
        "shifted kWh": shifted_energy(aware, agnostic),
        "low-price share aware": aware.metrics.low_price_share,
        "low-price share agnostic": costed.low_price_share,
        "comfort aware K.h": aware.metrics.comfort_violation_kh,
        "comfort agnostic K.h": costed.comfort_violation_kh,
    }
display(pd.DataFrame(rows).T.round(2))
aware = study.cached(study="implicit", variant="base", pricing="aware", days=study.DAYS)
display(plot_closed_loop(aware, "base house, price aware, day-ahead prices"))

# %% [markdown]
# The share of energy bought below the median price is the simplest signature of load shifting; `price_response`
# gives the whole distribution: how many kWh were bought at each price level.

# %%
display(price_response(aware).sort_index().round(3).tail(8))

# %% [markdown]
# ### Hourly or 15-minute planning
#
# Since 1 October 2025 the single day-ahead coupling trades 15-minute products. Below, the same plan at hourly
# and at 15-minute steps (the same 24-hour horizon: 24 or 96 steps, four times the variables). A heat pump on a
# buffer can follow a quarter-hour price only through its compressor: compare the energy bought and the cost.

# %%
house = building("base")
horizon_hours = 6 if quick_mode() else 24
plans = {}
for label, time_step in (("1 h", 3600.0), ("15 min", 900.0)):
    steps = round(horizon_hours * 3600 / time_step)
    controller = ModelPredictiveController(
        house.model, MPCSettings(horizon=steps, time_step=time_step), ComfortSpec(), tariffs["day-ahead"]
    )
    with timed(f"plan at {label}"):
        plans[label] = controller.solve(house.initial_state(), house.forecaster.forecast(START, steps, time_step))
summary = {}
for label, plan in plans.items():
    step = (plan.index[1] - plan.index[0]).total_seconds()
    summary[label] = {
        "status": plan.status,
        "cost EUR": plan.economic_cost,
        "electricity kWh": float(plan.meter["P_meter"].sum()) * step / 3.6e6,
        "peak kW": float(plan.meter["P_meter"].max()) / 1000,
        "solve s": plan.solve_time,
    }
display(pd.DataFrame(summary).round(3))
display(plot_plan(plans["15 min"], house.model, "base house, 15-minute plan"))

# %% [markdown]
# ## 2. Peak shaving under the capacity tariff
#
# The Flemish distribution tariff bills the average of the twelve monthly peaks (highest quarter-hour average
# power of the month, floored at 2.5 kW) at about 50 EUR/kW/year (2025 levels: 48.9 to 56.9 depending on the
# area). One extra kilowatt of monthly peak costs about 4 EUR; the controller carries the peak of the month so far
# and weighs the energy saving of a cheap-hour boost against the capacity cost of a higher peak. The runs use a
# time-of-use tariff and a deep night setback (5 K), so that the morning recovery makes the peak. With hourly
# steps the peak is the hourly average, which underestimates the quarter-hour peak.

# %%
capacity = {label: study.cached(**parts) for label, parts in study.capacity_runs().items()}
rows = []
for variant in ("base", "old"):
    subset = {label: result for label, result in capacity.items() if label.startswith(variant)}
    comparison = compare(subset, baseline=f"{variant} | none")
    rows.append(
        comparison[["total_cost", "capacity_cost", "peak_power_kw", "peak_reduction_pct", "comfort_violation_kh"]]
    )
display(pd.concat(rows).round(2))

# %%
dimensions = {"variant": ["base", "old"], "capacity": list(study.CAPACITY)}
display(closed_loop_explorer(study.CACHE, study.capacity_runs(), study.compute, dimensions))

# %% [markdown]
# ## 3. Explicit flexibility: activation requests and the rebound
#
# An aggregator or the DSO asks for a change of the electricity bought during a window against a baseline (here
# the run of the same controller without the request). The controller receives the request as a soft cap (down)
# or floor (up) one hour before the window (`ActivationRequest.notice_hours`), delivers what the comfort band
# allows and **rebounds** afterwards: the heat not delivered during a shed is delivered later, the heat stored
# during an upward activation is not bought later. The metrics give the delivered energy, the rebound energy and
# peak and their ratio, in the spirit of the available storage capacity and storage efficiency of Reynders et al.

# %%
results = {label: study.activation(label) for label in study.ACTIVATIONS}
display(pd.DataFrame({label: result.metrics for label, result in results.items()}).T.round(2))
display(plot_activation(results["shed 0.3 kW, 2 h at 17h"], "base house: shed 0.3 kW for 2 h at 17h"))
display(plot_activation(results["absorb 0.5 kW, 2 h at 13h"], "base house: absorb 0.5 kW for 2 h at 13h"))

# %% [markdown]
# ## 4. Flexibility envelope and flexibility function
#
# The **envelope** answers the aggregator's question before any request: from the current state, how much power
# can the house shed or absorb for one, two or four hours without leaving the comfort band, and what does that
# plan cost against the unconstrained one? It is a sweep of constrained MPC plans (bisection on the cap or floor
# of the electricity bought; absorbing is bounded by the full-load power of the heat pump).

# %%
envelope = study.envelope()
display(envelope.round(3))
display(plot_envelope(envelope, f"base house at {study.ENVELOPE_AT:%Hh}, day-ahead prices"))

# %% [markdown]
# The **flexibility function** of Junker et al. characterises the response to a penalty (price) step rather than
# to a power request: here the price is multiplied by three for three hours from 17h, and the change of the
# electricity bought reads like an activation.

# %%
response = study.activation("price step")
display(pd.Series(response.metrics).round(2).to_frame("price x3 for 3 h at 17h"))
display(plot_activation(response, "flexibility function: price x3 for 3 h at 17h"))

# %% [markdown]
# ## 5. Negative prices and PV
#
# With 20 m2 of PV the house exports at midday; a negative wholesale price (a renewable surplus) turns the midday
# import into something the controller *wants*: it charges the buffer and the walls, and the export is paid at the
# (negative) export price. Self-consumption is the share of the generation used on site.

# %%
pv = {label: study.cached(**parts) for label, parts in study.pv_runs().items()}
display(
    kpi_table(
        pv,
        ("total_cost", "energy_kwh", "sold_kwh", "generation_kwh", "self_consumption", "comfort_violation_kh"),
    )
)
display(plot_closed_loop(pv["pv | day-ahead + negative midday"], "PV house: negative midday prices"))

# %% [markdown]
# ## Key takeaways
#
# TAKEAWAYS
#
# ## Further reading
#
# `tutorials/references.md`: Annex 67 and Annex 82 for the definitions, Reynders, Diriken and Saelens (2017) and
# Stinner et al. (2016) for the characterisation, Junker et al. (2018) for the flexibility function, the Vlaamse
# Nutsregulator sheets for the capacity tariff.
