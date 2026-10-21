# %% [markdown]
# # 05 - PV, a battery and an electric vehicle next to the thermal storage
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andoludo/mpctrano/blob/main/tutorials/05_pv_battery_ev.ipynb)
#
# **Learning objectives**
#
# 1. Put electrical storages next to the thermal one: a stationary battery and a vehicle charged at home are
#    optimised in the balance of the meter, with the heat pump, the buffer and the walls of the house.
# 2. Read the dispatch order under day-ahead prices and under negative midday prices: which storage takes the
#    cheap or free energy first, what is exported, how much of the PV is consumed on site.
# 3. See the vehicle's deadline: it leaves at 7h with 30 kWh and comes back at 18h 12 kWh lower; the controller
#    charges it when the night is cheap, not when it arrives.
#
# The storages are linear state-of-charge models in the controller (`Battery`, `ElectricVehicle`,
# `MPCSettings.batteries` and `.vehicles`): Trano's FastModel describes the building and its thermal systems,
# mpctrano the electrical assets at the meter. The closed loops come from `tutorials/cache/pv/`
# (`uv run python tutorials/scripts/build_cache.py pv`); the plans are solved live.
#
# **Runtimes** (hourly steps, this container): one 24-hour plan of the PV house 10 to 50 s; one closed-loop day
# with the storages 10 to 30 minutes.

# %%
import sys

if "google.colab" in sys.modules:  # Colab: fetch the tutorials, install mpctrano and rumoca (a no-op elsewhere)
    import subprocess

    subprocess.run(  # a fixed command; the clone is skipped when it exists
        ["git", "clone", "-q", "--depth", "1", "https://github.com/andoludo/mpctrano", "/content/mpctrano"], check=False
    )
    sys.path.insert(0, "/content/mpctrano")
    from tutorials import _colab

    _colab.setup()

# %%
import pandas as pd
from IPython.display import display

from mpctrano import ComfortSpec, ModelPredictiveController, MPCSettings
from mpctrano.notebook import kpi_table, quick_mode, timed
from mpctrano.plot import plot_closed_loop, plot_plan, plot_storage
from tutorials import _pv

pd.set_option("display.width", 200)
print(f"quick mode: {quick_mode()}; cached runs: {_pv.DAYS:g} day from {_pv.START.date()}")

# %% [markdown]
# ## 1. The house with PV, and the storages
#
# `tutorials/buildings/house/pv.yaml` is the house of tutorial 01 with 20 m2 of PV on the south roof (about 4 kWp):
# Trano meters its production (`pv_001.EPro`), which the controller nets against the heat pump at the meter. The
# battery (8 kWh, 3 kW, 92 % round trip) and the vehicle (50 kWh, 7.4 kW charger, home from 18h to 7h, 12 kWh a
# day, 30 kWh required at departure) are declared in the controller settings.

# %%
house = _pv.building("pv")
model = house.model
print("meters", model.meters)
print("battery", _pv.BATTERY.model_dump())
print("vehicle", _pv.VEHICLE.model_dump())
x0 = house.initial_state()
tariffs = _pv.tariffs()
horizon = 6 if quick_mode() else 24

# %% [markdown]
# ## 2. One plan: the dispatch under day-ahead prices
#
# The plan below runs the house, the battery and the vehicle together over a January day. Look at where the
# battery charges (the cheap night, the midday PV) and discharges (the evening peak), and at the vehicle: charged
# in the cheap hours before 7h, idle while away, charged again after 18h only if the night is not cheaper.

# %%
plans = {}
for price in _pv.PRICES:
    settings = MPCSettings(horizon=horizon, **_pv.STORAGES["battery + vehicle"])
    controller = ModelPredictiveController(model, settings, ComfortSpec(), tariffs[price])
    with timed(f"plan under {price}"):
        plans[price] = controller.solve(x0, house.forecaster.forecast(_pv.START, horizon, 3600.0))
display(plot_plan(plans["day-ahead"], model, "PV house with a battery and a vehicle, day-ahead prices"))
display(plot_storage(plans["day-ahead"], "storages, day-ahead prices"))

# %% [markdown]
# ## 3. Negative prices at midday
#
# A negative wholesale price (a renewable surplus) makes importing worth money: the controller charges everything
# that can take it, the battery, the vehicle if it is home (it is not, at midday on a weekday), the buffer and the
# walls, and stops exporting the PV. Compare the two plans on the storages:

# %%
display(plot_storage(plans["day-ahead + negative midday"], "storages, negative midday prices"))
summary = {
    price: {
        "cost EUR": plan.economic_cost,
        "bought kWh": float(plan.meter["P_meter"].sum()) / 1000,
        "sold kWh": float(plan.meter["P_export"].sum()) / 1000,
        "battery charged kWh": float(plan.storage["battery_charge_w"].sum(skipna=True)) / 1000,
        "vehicle charged kWh": float(plan.storage["vehicle_charge_w"].sum(skipna=True)) / 1000,
    }
    for price, plan in plans.items()
}
display(pd.DataFrame(summary).round(2))

# %% [markdown]
# ## 4. The closed loop: with and without the storages
#
# One day against the true building (perturbed envelope, noisy sensors) with moving horizon estimation, for the
# house alone, with the battery, and with the battery and the vehicle, under the two price signals; the
# controllers of the YAML (no storage, the PV exports what the house does not use) as the reference. The vehicle's
# driving energy is bought in every run that has it: compare the cost per kWh rather than the total.

# %%
loops = {label: _pv.cached(**parts) for label, parts in _pv.runs().items()}
for price in _pv.PRICES:
    loops[f"pv | {price} | rule based"] = _pv.reference(price)
display(
    kpi_table(
        loops,
        (
            "total_cost",
            "energy_kwh",
            "sold_kwh",
            "generation_kwh",
            "self_consumption",
            "comfort_violation_kh",
            "peak_power_kw",
        ),
    )
)
display(
    plot_closed_loop(
        loops["pv | day-ahead + negative midday | battery + vehicle"], "negative midday prices, battery and vehicle"
    )
)

# %% [markdown]
# ## Key takeaways
#
# TAKEAWAYS
#
# ## Further reading
#
# * `tutorials/references.md`: Annex 67 for the storage order and the self-consumption indicators.
# * The storages are exact linear models here; a battery and a charger as FastModel components in Trano would
#   put them in the Modelica model with their losses (see `tutorials/PLAN.md`).
