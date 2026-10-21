# %% [markdown]
# # 01 - A house with a heat pump: the reference implementation
#
# **Learning objectives**
#
# 1. Follow the pipeline from the Trano YAML to the CasADi model: what the FastModel of a house contains, which
#    inputs the controller decides and which ones it has to forecast, where the energy is metered.
# 2. Read an economic MPC plan: how the comfort band, the price signal and the horizon shape the heating of a heat
#    pump house with a buffer tank and a radiator.
# 3. Run the closed loop against a building that is *not* the control model (perturbed envelope, noisy sensors)
#    and compare it with the controllers written in the YAML.
# 4. See why state estimation matters: the controller needs the walls, the water and the buffer, it measures the
#    air temperature only.
#
# Everything here calls the library; the closed loops come from `tutorials/cache/house/`
# (`uv run python tutorials/scripts/build_cache.py house`). The plan widget solves live.
#
# **Runtimes** (hourly steps, this container): Trano and rumoca about 10 s per model; one 24-hour plan 5 to 20 s;
# one two-day closed loop with moving horizon estimation about ten minutes; the rule-based reference a second.

# %%
import pandas as pd
from IPython.display import display

from mpctrano.notebook import closed_loop_explorer, kpi_table, plan_explorer, quick_mode
from mpctrano.plot import plot_closed_loop, plot_states
from tutorials import _house

pd.set_option("display.width", 200)
print(f"quick mode: {quick_mode()}; cached runs: {_house.DAYS:g} days from {_house.START.date()}")

# %% [markdown]
# ## 1. From the YAML to the CasADi model
#
# `tutorials/buildings/house/base.yaml` describes a 100 m2 house: cavity walls, triple glazing, a concrete slab,
# a 3.5 kW air-to-water heat pump with a 150 l buffer tank, a mixing valve and a low-temperature radiator behind
# a thermostatic valve. Trano renders it with its `FastModelMPC` library (physics close to the Buildings library:
# layered walls, window optics, water nodes) where **the controls of the systems are inputs**, and with its
# `FastModel` library where the controllers of the YAML close the loop (the rule-based reference). rumoca
# translates each into an explicit CasADi ODE `xdot = f(t, x, u, d, p)`.

# %%
house = _house.building("base")
model = house.model
print(f"{model.n_states} states, {model.n_controls} controls, {model.n_disturbances} disturbances, "
      f"{model.n_parameters} parameters")  # fmt: skip
print("controls    ", model.control_names)
print("disturbances", model.disturbance_names)
print("comfort     ", model.comfort_states)
print("meters      ", model.meters)
print("states      ", model.state_names)

# %% [markdown]
# The roles come from the Modelica model itself: an input with a minimum and a maximum is a **control** (the
# compressor speed `boiler_001_y`, the share of supply water of the mixing valve, the opening of the radiator
# valve, all between 0 and 1), any other input a **disturbance** (seven weather channels and the internal gains
# of the occupancy), the air temperature `TAir` is the **comfort** variable and the state in kWh is the electricity
# **meter** of the heat pump. The controller never sees a COP: it optimises the valves and the compressor, and the
# electricity follows from the heat pump model and its meter.
#
# A FastModel starts with every temperature at its start value (walls at the set point, buffer at the supply
# temperature). Every run of the tutorials starts from the state the controllers of the YAML reach after three
# days (`spin_up`): the night setback has cooled the house at the start of the period.

# %%
x0 = pd.Series(house.initial_state(), index=model.state_names)
display((x0[[name for name in model.state_names if name.endswith(("TAir", ".T", "T[1]"))]] - 273.15).round(1))
forecast = house.forecaster.realisation(_house.START, 48, 3600.0)
display(forecast.describe().round(1).T[["mean", "min", "max"]])

# %% [markdown]
# ## 2. One economic MPC plan
#
# The controller minimises, over the horizon, the electricity bought (read from the meter of the heat pump) at the
# tariff price, plus penalties on the comfort band, subject to the ODE collocated on Radau points and the bounds
# of the controls; IPOPT solves the nonlinear programme. With a flat price it heats just enough and as late as the
# slow radiator allows; with a time-varying price it charges the buffer, the water and the walls when electricity
# is cheap. Try it: the widget re-solves the plan (5 to 20 s; the next solves warm start from the previous plan).

# %%
tariffs = _house.tariffs()
display(plan_explorer(house, tariffs))

# %% [markdown]
# What to look for:
#
# * **The morning recovery**: the setback ends at 7h; the plan starts the compressor before, so that the air is
#   in the band at 7h, and earlier when the night is cheap.
# * **The buffer** is the fast storage: the heat pump charges it at full speed in the cheap hours and the mixing
#   valve doses the supply of the radiator.
# * **The spike at 17h** empties the compressor in the evening: the house coasts on the buffer and the walls.
# * **The comfort weight** trades a few kelvin-hours for cost when the band is tight.

# %% [markdown]
# ## 3. Closed loop against the true building
#
# A plan is not a result: the controller applies the first step, the building responds, the sensor reports the
# air temperature, the state is estimated and the plan is rebuilt. The *true building* has its envelope
# conductances and capacities, air capacity and infiltration drawn at random (20 %) and a sensor noise of 0.1 K;
# the controller keeps the Trano model. The runs are cached; **Recompute** runs one again (about ten minutes).

# %%
runs = {label: parts for label, parts in _house.runs().items() if parts["estimator"] == "mhe"}
dimensions = {
    "variant": list(_house.VARIANTS),
    "price": [*_house.PRICES, "day-ahead + evening spike"],
    "band": list(_house.BANDS),
    "mismatch": ["mismatch"],
}
display(closed_loop_explorer(_house.CACHE, runs, _house.compute, dimensions))

# %% [markdown]
# ### MPC against the controllers of the YAML
#
# The same building, period, tariff and comfort band, controlled by the thermostat, heating curve and buffer
# controller written in the YAML (the closed-loop `FastModel`, simulated in a second): this is the reference an
# MPC has to beat.

# %%
comparison = {}
for variant in _house.VARIANTS:
    for price in _house.PRICES:
        parts = {"variant": variant, "price": price, "band": "standard", "mismatch": "mismatch"}
        comparison[f"{variant} | {price} | MPC"] = _house.cached(**parts, estimator="mhe", days=_house.DAYS)
        comparison[f"{variant} | {price} | rule based"] = _house.reference(**parts)
display(kpi_table(comparison, ("total_cost", "energy_kwh", "comfort_violation_kh", "peak_power_kw", "low_price_share")))
display(plot_closed_loop(comparison["base | day-ahead | MPC"], "base house, day-ahead prices, true building"))

# %% [markdown]
# ### The price of not knowing the building
#
# The same price and band with the plant equal to the control model (and its true state) and with the true
# building: the cost difference is the price of the model error, the comfort violations show where the
# controller was over-confident.

# %%
pairs = {}
for price in _house.PRICES:
    for mismatch, estimator in (("perfect", "perfect"), ("mismatch", "mhe")):
        pairs[f"{price} | {mismatch}"] = _house.cached(
            variant="base", price=price, band="standard", mismatch=mismatch, estimator=estimator, days=_house.DAYS
        )
display(kpi_table(pairs))

# %% [markdown]
# ## 4. Why state estimation matters
#
# The controller needs the full state: the wall layers, the water of the radiator and the buffer. It measures the
# air temperature only. The moving horizon estimator fits the trajectory of the model to the measurements of the
# last hours; below, its estimate of the buffer and of the inner layer of a wall against the true building, then
# three runs on the same true building: the true state handed to the controller, moving horizon estimation, and
# no estimation (the measured air temperature, the other states held at their expected start values).

# %%
estimation = {
    estimator: _house.cached(
        variant="base", price="day-ahead", band="standard", mismatch="mismatch", estimator=estimator, days=_house.DAYS
    )
    for estimator in ("perfect", "mhe", "none")
}
display(plot_states(estimation["mhe"], ["boiler_001.T", "externalwall_0.T[5]"], "Moving horizon estimation"))
display(kpi_table(estimation, ("total_cost", "energy_kwh", "comfort_violation_kh", "max_violation", "failures")))

# %% [markdown]
# ## Key takeaways
#
# TAKEAWAYS
#
# ## Further reading
#
# * `tutorials/references.md`, section A: Drgoňa et al. (2020) for MPC in buildings, Blum et al. (2021) for the
#   KPIs used here.
# * `docs/mpc.md` for the optimisation problem and `docs/estimation.md` for the estimator.
# * Next: `02_flexibility_use_cases` quantifies what this flexibility is worth under real price signals, capacity
#   tariffs and explicit activation requests.
