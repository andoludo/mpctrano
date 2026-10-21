# %% [markdown]
# # 04 - An office in summer: cooling, occupancy and the price of comfort
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andoludo/mpctrano/blob/main/tutorials/04_office_cooling.ipynb)
#
# **Learning objectives**
#
# 1. Model cooling with Trano's FastModel: the ideal emitter of a zone is a heat flow the controller decides;
#    letting it go negative turns it into a chiller, costed at an EER.
# 2. Read a cooling plan: the sun and the occupants drive the load, the mass of the building and the price signal
#    decide when the chiller runs (pre-cooling at night under day-ahead prices).
# 3. Weigh comfort against cost: the office without cooling (the controllers of the YAML in summer) against the
#    MPC, and the peak of the chiller under a capacity charge.
#
# The closed loops come from `tutorials/cache/office/` (`uv run python tutorials/scripts/build_cache.py office`);
# the plans are solved live.
#
# **Runtimes** (hourly steps, this container): Trano and rumoca about 15 s; one 24-hour plan of this two-zone
# office one to two minutes (52 states, two heat flows); one closed-loop day about an hour.

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

from mpctrano import ModelPredictiveController, MPCSettings
from mpctrano.notebook import kpi_table, quick_mode, timed
from mpctrano.plot import plot_closed_loop, plot_plan
from tutorials import _office

pd.set_option("display.width", 200)
print(f"quick mode: {quick_mode()}; cached runs: {_office.DAYS:g} day from {_office.START.date()}")

# %% [markdown]
# ## 1. The office and its emitters
#
# `tutorials/buildings/office/base.yaml`: an open-plan office of 200 m2 with 36 m2 of south glazing and a meeting
# area of 80 m2, externally insulated walls, a flat roof, occupied from 8h to 18h (12 W/m2 of occupants and
# equipment in the open plan). Each zone has an ideal emitter (`radiator` of variant `ideal` in the YAML): in the
# `FastModelMPC` model it is an input `Q_flow` [W], which Trano bounds to heating (`min=0`). The study lets it go
# negative (`FastModel.with_control_bounds`): the same equations cool the zone. The electricity follows from a
# COP of 3.5 for heating and an EER of 3 for cooling (`MPCSettings.heat_cop`, `cooling_eer`).

# %%
office = _office.building()
model = office.model
print("controls", model.control_names, "bounds", model.control_bounds)
print("zones", model.zone_names, "| states", model.n_states, "| meters", model.meters or "none: the emitters are ideal")
x0 = office.initial_state()
print("spun-up air [degC]", {zone: round(x0[model.state_names.index(state)] - 273.15, 1)
                            for zone, state in zip(model.zone_names, model.comfort_states, strict=True)})  # fmt: skip

# %% [markdown]
# ## 2. A summer day: flat price against day-ahead prices
#
# The comfort band is 21 to 25 degC when occupied, free floating between 15 and 30 degC otherwise. With a flat
# price the controller cools as late as possible; with day-ahead prices it may pre-cool the mass at night, when
# electricity is cheap, if the saving beats the extra losses.

# %%
tariffs = _office.tariffs()
horizon = 6 if quick_mode() else 24
plans = {}
for price in _office.PRICES:
    controller = ModelPredictiveController(
        model, MPCSettings(horizon=horizon, **_office.EFFICIENCIES), _office.BAND, tariffs[price]
    )
    with timed(f"plan under {price} prices"):
        plans[price] = controller.solve(x0, office.forecaster.forecast(_office.START, horizon, 3600.0))
    display(plot_plan(plans[price], model, f"office, {price} prices"))
summary = {
    price: {
        "cost EUR": plan.economic_cost,
        "cooling kWh": -float(plan.controls.clip(upper=0).sum().sum()) / 1000,
        "heating kWh": float(plan.controls.clip(lower=0).sum().sum()) / 1000,
        "electricity kWh": float(plan.meter["P_meter"].sum()) / 1000,
        "peak kW": float(plan.meter["P_meter"].max()) / 1000,
    }
    for price, plan in plans.items()
}
display(pd.DataFrame(summary).round(2))

# %% [markdown]
# ## 3. The closed loop: the MPC against no cooling
#
# The true office (perturbed envelope, noisy sensors) over one working day with moving horizon estimation, against
# the office without cooling: the closed-loop model of the YAML has the thermostats of the heating only, so in
# summer it free floats. Its "cost" is the comfort it loses.

# %%
loops = {}
for price in _office.PRICES:
    loops[f"MPC | {price}"] = _office.cached(price=price, days=_office.DAYS)
loops["no cooling"] = _office.reference("flat")
display(kpi_table(loops, ("total_cost", "energy_kwh", "comfort_violation_kh", "max_violation", "peak_power_kw")))
display(plot_closed_loop(loops["MPC | day-ahead"], "office, day-ahead prices, true building"))

# %% [markdown]
# ## 4. The peak of the chiller under a capacity charge
#
# A capacity charge on the peak of the day (here 8 EUR/kW, the monthly charge of the Flemish capacity tariff at
# about 100 EUR/kW/year) makes the controller spread the cooling: lower peak, more kWh.

# %%
capped = {}
for charge in (0.0, 8.0):
    run = _office.study_scenario("flat", capacity_charge=charge)
    controller = ModelPredictiveController(
        model, MPCSettings(horizon=horizon, **_office.EFFICIENCIES), run.comfort, run.tariff
    )
    with timed(f"plan with a capacity charge of {charge:g} EUR/kW"):
        capped[f"{charge:g} EUR/kW"] = controller.solve(x0, office.forecaster.forecast(_office.START, horizon, 3600.0))
display(
    pd.DataFrame(
        {
            label: {
                "energy cost EUR": plan.costs["electricity"],
                "capacity cost EUR": plan.costs["capacity"],
                "electricity kWh": float(plan.meter["P_meter"].sum()) / 1000,
                "peak kW": float(plan.meter["P_meter"].max()) / 1000,
            }
            for label, plan in capped.items()
        }
    ).round(2)
)

# %% [markdown]
# ## Key takeaways
#
# TAKEAWAYS
#
# ## Further reading
#
# * `tutorials/references.md`: the characterisation of flexibility (Reynders et al.) applies to cooling as to
#   heating; Blum et al. (2021) for the KPIs.
# * The chiller is a constant EER here; a chiller and a cooling coil in the FastModel library of Trano would give
#   the part-load behaviour (see `tutorials/PLAN.md`).
