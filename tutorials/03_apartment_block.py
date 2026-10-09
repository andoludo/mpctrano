# %% [markdown]
# # 03 - Apartment block: one heat pump, six apartments and a hot water tank
#
# **Learning objectives**
#
# 1. Read a multi-zone model with **shared systems** from the trano interface: one air-to-water
#    heat pump serving six apartments through radiators and a 1 m3 domestic hot water (DHW) tank.
# 2. See how the controller **dispatches a shared compressor**: the compressor limit couples the
#    apartments and the tank, so a cheap hour is worth less when everybody wants it at once.
# 3. Quantify the **DHW tank as flexibility**: the same evening shed with the tank free to run
#    down between 45 and 55 degC and with the tank locked at its set temperature.
# 4. Compare **emitters** (floor heating, radiators at 55 degC, high-temperature radiators at
#    70 degC) through the COP and the compressor headroom they leave, and see what a **smaller
#    heat pump** does to the dispatch.
#
# The block is described in `tutorials/buildings/apartments.yaml` (the heat pump and the tank are
# ordinary trano `systems`); `tutorials/scripts/generate_models.py` writes the variants and the
# R3C2 control / R4C3 plant pairs under `tutorials/models/apartments/`. The closed-loop runs come
# from `tutorials/cache/apartments/` (`make cache`); the widgets solve one open-loop plan live.
#
# **Runtimes** (this container): one open-loop solve about 1.4 s (13 states, 7 controls, 24 steps),
# a 3-day hourly closed loop about 100 s, the envelope of section 4 about 40 s per tank setting.

# %%
import pandas as pd
from IPython.display import display

from mpctrano import FlexibilityEnvelope, ModelPredictiveController, MPCSettings, compressor_load
from mpctrano.notebook import closed_loop_explorer, kpi_table, plan_explorer, quick_mode, timed
from mpctrano.plot import plot_activation, plot_cop, plot_dispatch, plot_envelope, plot_plan
from tutorials import _apartments as study
from tutorials._house import START

pd.set_option("display.width", 200)
print(f"quick mode: {quick_mode()}; cached runs: {study.DAYS:g} days from {START.date()}")

# %% [markdown]
# ## 1. The block as the controller sees it
#
# Six apartments of 75 m2 on three floors (south and north facing, the ground floor on a slab,
# the top floor under the roof), radiators sized at 55/45 degC, one heat pump of 24 kW thermal
# (8 kW electrical) and a 1 m3 tank kept between 45 and 55 degC with a draw-off of 30 kWh/day.
# The interface lists the systems and the inputs they own: the heat pump owns the electrical
# input of every apartment and of the tank (`inputs` and `tank_inputs`), which is what the
# controller turns into one shared compressor limit (`control_groups`); the tank adds a storage
# state with a band (`storage_bands`).

# %%
house = study.building("base")
model = house.control
print(f"{model.n_states} states, {model.n_controls} controls, {len(model.disturbance_names)} disturbances")
for system in model.systems:
    owned = list(system.inputs) + list(getattr(system, "tank_inputs", []))
    print(f"{system.kind:10s} {system.name}: {', '.join(owned)}")
for group in model.control_groups:
    print(f"compressor {group.system}: {len(group.indices)} inputs share {group.max_power / 1000:.1f} kW")
for band in model.storage_bands:
    print(f"storage {band.system}: {band.lower - 273.15:.0f} to {band.upper - 273.15:.0f} degC")

# %% [markdown]
# The COP is the bilinear polynomial of the trano `mpc` library in the outdoor and the supply
# temperature; the supply temperature follows the emitter (`water_inlet_temperature_at_nominal_condition`
# of the radiators, plus a 5 K offset for the tank). Floor heating at 35 degC and radiators at
# 70 degC are the two other variants of this tutorial.

# %%
heat_pump = model.system("heatpump_001")
display(plot_cop(heat_pump, (308.15, 328.15, 343.15)))

# %% [markdown]
# A live open-loop plan: move the comfort band or the horizon and watch the compressor share
# itself between the apartments and the tank. The plan figure stacks the thermal inputs; the
# dispatch view below it is the one to read for the compressor.

# %%
tariffs = study.tariffs()
display(plan_explorer(house, tariffs, START, on_solution=lambda solution, model: display(plot_plan(solution, model))))

# %% [markdown]
# ## 2. Dispatching a shared compressor
#
# In the house of tutorial 01 the cheap hour belongs to one zone. Here seven consumers compete for
# 8 kW: the controller spreads the boost over several hours, serves the tank when the apartments
# are satisfied and pre-heats the apartments whose heat loss is highest (top floor, north) first.
# `plot_dispatch` stacks the electrical inputs against the compressor limit and shows the tank
# with its band under the zone temperatures.

# %%
block = {label: study.cached(**parts) for label, parts in study.block_runs().items()}
display(plot_dispatch(block["base | day-ahead"], model, "base block, day-ahead prices"))

# %% [markdown]
# `compressor_load` summarises the use of the compressor: the mean load factor, the share of the
# steps where the compressor is at its limit (binding, so that the price no longer decides) and
# the share of the electricity that went to each apartment and to the tank.

# %%
display(compressor_load(block["base | day-ahead"].frame, model).round(3).T)

# %% [markdown]
# The explorer below selects any cached run (variant and price); "Recompute" runs the closed loop
# again with the perturbed plant, which takes about 100 s.

# %%
dimensions = {"variant": list(study.VARIANTS), "price": list(study.PRICES)}
display(
    closed_loop_explorer(
        study.CACHE,
        study.block_runs(),
        study.compute,
        dimensions,
        on_result=lambda result: display(plot_dispatch(result, study.building(result.name.split(" ")[0]).control)),
    )
)

# %% [markdown]
# ## 3. Emitters and the size of the heat pump
#
# The supply temperature sets the COP: at 5 degC outdoor the polynomial gives 4.0 for floor
# heating at 35 degC, 2.6 for radiators at 55 degC and 1.5 for radiators at 70 degC. The same
# 8 kW of electricity therefore buys less heat at high temperature, so the compressor binds more
# often and the cheap hours are used less. The smaller heat pump (5.5 kW
# electrical) shows the same effect at 55 degC. Energy is the electricity bought at the meter.

# %%
columns = ("total_cost", "energy_kwh", "comfort_violation_kh", "peak_power_kw", "low_price_share")
table = kpi_table(block, columns)
usage = pd.DataFrame(
    {
        label: compressor_load(result.frame, study.building(label.split(" ")[0]).control).loc["heatpump_001"]
        for label, result in block.items()
    }
).T
table["mean load"] = usage["mean_load"].round(3)
table["binding share"] = usage["binding_share"].round(3)
table["tank share"] = usage["share_tank_001_PHea"].round(3)
display(table)

# %% [markdown]
# Read the table by variant: the high-temperature block costs the most and binds its compressor
# the most; the floor-heating block buys the least electricity and shifts the most into the cheap
# hours (`low_price_share`); the small heat pump keeps the comfort but pays for it with a flatter
# dispatch, less able to concentrate the heating in the cheap hours. The evening spike shows
# where the storage is: the tank and the slabs carry the apartments through the expensive hours.

# %%
display(plot_dispatch(block["small_heat_pump | day-ahead + evening spike"], study.building("small_heat_pump").control,
                      "small heat pump, evening spike"))  # fmt: skip

# %% [markdown]
# ## 4. The hot water tank as the cheapest flexibility
#
# The tank is the one storage that costs no comfort: between 45 and 55 degC the occupants see no
# difference, and 1 m3 of water over 10 K holds about 11.6 kWh of heat, roughly 3 kWh of
# electricity at the COP of the heat pump. The activation study asks the same question twice: shed
# 3 kW for two hours at 18h on the second day, with the tank free to run down to 45 degC and with
# the tank **locked** at its set temperature (`RCModel.with_storage_band`, a one-kelvin band). The
# difference is the flexibility the tank provides on its own.

# %%
results = {tank: study.activation(tank) for tank in ("free", "locked")}
metrics = pd.DataFrame({f"tank {tank}": result.metrics for tank, result in results.items()}).T
display(metrics.round(2))
display(plot_activation(results["free"], "tank free: shed 3 kW for 2 h at 18h, second day"))
display(plot_activation(results["locked"], "tank locked: the apartments alone carry the shed"))

# %% [markdown]
# With the tank free the controller heats the water before the window and lets it run down during
# the shed; the rebound is a tank reheat, which it can time to the cheap hours. Locked, the shed
# falls on the apartments: the delivered energy drops and the rebound lands right after the
# window, when the radiators have to catch up. The envelope makes the same point for every
# duration: how much power can the block shed or absorb from where it stands at 6h on the second
# day of the cached run (the state the controller estimated at that step)?

# %%
at = START + pd.Timedelta(days=1, hours=6)
state = block["base | day-ahead"].frame.loc[at, [f"{name}_estimate" for name in model.state_names]].to_numpy()
durations = (2.0,) if quick_mode() else (1.0, 2.0, 4.0)
envelopes = {}
for tank in ("free", "locked"):
    control = study.control_model("base", tank)
    controller = ModelPredictiveController(
        control, MPCSettings(horizon=24), study.comfort_spec(*study.BAND), tariffs["day-ahead"]
    )
    with timed(f"envelope, tank {tank}"):
        envelopes[tank] = FlexibilityEnvelope.compute(
            controller, state, house.forecaster.forecast(at, 24, 3600.0), durations=durations, iterations=5
        )
display(pd.concat({tank: envelope.frame for tank, envelope in envelopes.items()}).round(2))
display(plot_envelope(envelopes["free"], "base block at 6h on the second day, tank free"))

# %% [markdown]
# ## Key takeaways
#
# * A shared compressor turns the dispatch into a scheduling problem: the controller cannot give
#   every apartment the cheapest hour, so it ranks them by heat loss and fills the limit over
#   several hours. Reading `binding_share` tells whether the heat pump or the price is in charge.
# * The emitter temperature is the first design lever for flexibility: a low supply temperature
#   raises the COP and the compressor headroom at once, so the same heat pump both costs less and
#   can shift more.
# * The DHW tank delivers the shed with no comfort cost and the controller times its rebound;
#   locked at its set temperature the block delivers less and rebounds harder. Keeping the tank
#   band open is worth more than any comfort relaxation for a block of this size.
# * A smaller heat pump keeps the comfort on a January week but loses most of the implicit
#   flexibility: sizing for the design day and sizing for flexibility are different questions.
#
# ## Further reading
#
# `tutorials/references.md`: Annex 67 on the storage order (tank first) and the characterisation
# of Reynders, Diriken and Saelens (2018); the trano documentation of the `mpc` library for the
# COP polynomial and the tank model. Tutorial 04 adds cooling and an office schedule; tutorial 06
# puts several of these blocks on one feeder.
