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
# **Runtimes** (this container): one open-loop solve about 1.2 s (13 states, 7 controls, 24 steps),
# a 3-day hourly closed loop 75 to 95 s, the envelope of section 4 about 45 s per tank setting
# (15 s in quick mode).

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
# 8 kW: the controller spreads the boost over several hours, charges the tank in a few full-power
# hours and gives the largest shares to the ground-floor and top-floor apartments, whose losses
# are highest. `plot_dispatch` stacks the electrical inputs against the compressor limit and shows
# the tank with its band under the zone temperatures.

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
# Read the table by variant: the high-temperature block buys a third more electricity, costs a
# third more and doubles the comfort violations, because its compressor runs out of heat (binding
# in about a tenth of the hours); the floor-heating block buys the least electricity; the big tank
# shifts the most into the cheap hours (`low_price_share`) and charges at full power; the small
# heat pump binds the most often and pays about a tenth more for a flatter dispatch that cannot
# concentrate the heating in the cheap hours. The evening spike costs every variant two to three
# euros over three days: the tank and the envelope carry the apartments through the expensive
# hours.

# %%
display(plot_dispatch(block["small_heat_pump | day-ahead + evening spike"], study.building("small_heat_pump").control,
                      "small heat pump, evening spike"))  # fmt: skip

# %% [markdown]
# ## 4. The hot water tank as the cheapest flexibility
#
# The tank is the one storage that costs no comfort: between 45 and 55 degC the occupants see no
# difference, and 1 m3 of water over 10 K holds about 11.6 kWh of heat, roughly 3 kWh of
# electricity at the COP of the heat pump. The activation study asks the same question twice: shed
# 1.5 kW (half of the baseline) for two hours at 18h on the second day, notified two hours ahead
# as an intraday product would be (`ActivationRequest.notice_hours`), with the tank free to run
# down to 45 degC and with the tank **locked** at its set temperature
# (`RCModel.with_storage_band`, a one-kelvin band). The difference is the flexibility the tank
# provides on its own. A longer notice lets the controller pre-charge the envelope for hours, which
# is cheap in energy but takes the apartments to the top of the comfort band.

# %%
results = {tank: study.activation(tank) for tank in ("free", "locked")}
metrics = pd.DataFrame({f"tank {tank}": result.metrics for tank, result in results.items()}).T
display(metrics.round(2))
display(plot_activation(results["free"], "tank free: shed 1.5 kW for 2 h at 18h, second day"))
display(plot_activation(results["locked"], "tank locked: the apartments alone carry the shed"))

# %% [markdown]
# Both settings deliver the 1.5 kW, and the difference sits in the **baseline**: with the tank
# free the controller never heats the water in the evening anyway (the tank drains from 55 to
# 45 degC during the draw-off and is recharged in the cheap night hours), so the cap of the request
# lands on the apartments, whose temperature dips, and the rebound is small and spread over the
# evening. Locked, the baseline reheats the tank during the window; the request is settled against
# that higher baseline, part of the shed is the tank's share and it is paid back right after the
# window (a larger rebound peak, mostly the tank). The tank's flexibility is therefore already in
# the baseline, which costs about a tenth less than the locked one over the three days: a
# baseline-settled activation cannot pay for what the price signal has already taken. The
# envelope makes the point from the state the controller estimated at 6h on the second day of the
# cached run: how much power can the block shed or absorb for one, two or four hours?

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
# * The DHW tank is the cheapest flexibility of the block, and the price signal takes it first:
#   with the tank free the baseline already keeps the water heating out of the evening, so an
#   activation settled against that baseline falls on the apartments. Locking the tank raises
#   the baseline cost by about a tenth and turns the tank reheat into a rebound peak.
# * A smaller heat pump keeps the comfort on a January week but pays about a tenth more and
#   binds its compressor in an eighth of the hours: sizing for the design day and sizing for
#   flexibility are different questions.
#
# ## Further reading
#
# `tutorials/references.md`: Annex 67 on the storage order (tank first) and the characterisation
# of Reynders, Diriken and Saelens (2018); the trano documentation of the `mpc` library for the
# COP polynomial and the tank model. Tutorial 04 adds cooling and an office schedule; tutorial 06
# puts several of these blocks on one feeder.
