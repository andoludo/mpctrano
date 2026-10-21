# %% [markdown]
# # 06 - Your building: edit the YAML, plan, and compare with its own controllers
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andoludo/mpctrano/blob/main/tutorials/06_your_building.ipynb)
#
# **Learning objectives**
#
# 1. Change the building in the Trano YAML (insulation, windows, heat pump, buffer, set points, schedules) and
#    regenerate the CasADi model in seconds: the whole chain of the tutorials runs from one editable cell.
# 2. Plan the heating of your building with the economic MPC and read what the change did to the plan.
# 3. Run the closed loop against the controllers written in the YAML (the rule-based reference) and compare the
#    cost, the energy and the comfort of two versions of the building side by side.
#
# Everything is computed live (nothing is cached): Trano and rumoca take 10 to 30 s per building, a plan 5 to
# 20 s, one day of closed loop two to four minutes (six hours in quick mode).

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
import shutil
from pathlib import Path

import pandas as pd
from IPython.display import display

from mpctrano import ComfortSpec, ModelPredictiveController, MPCSettings, run_scenario
from mpctrano.notebook import (
    TutorialBuilding,
    comfort_spec,
    days,
    kpi_table,
    plan_explorer,
    price_options,
    quick_mode,
    scenario,
    timed,
)
from mpctrano.plot import plot_closed_loop, plot_plan
from mpctrano.simulation import ClosedLoopResult
from tutorials import _house

pd.set_option("display.width", 200)
START = _house.START
OUTPUT = (
    Path("tutorials/output/your_building").resolve()
    if Path("tutorials").is_dir()
    else _house.HERE / "output" / "your_building"
)
WEATHER = next(_house.BUILDINGS.glob("*.mos"))
LIVE_DAYS = days(1.0, quick=0.25)
print(f"quick mode: {quick_mode()}; buildings written to {OUTPUT}; closed loops of {LIVE_DAYS:g} day")

# %% [markdown]
# ## 1. The building, in one cell
#
# The YAML below is the house of tutorial 01 (`tutorials/buildings/house/base.yaml`): a 100 m2 single zone
# with cavity walls, triple glazing and a concrete slab, heated by a 3.5 kW air-to-water heat pump with a 150 l
# buffer and a low-temperature radiator. **Edit it and run the notebook again.** Some things to try:
#
# * the envelope: the wall and window surfaces, their azimuth (0 is south), the constructions (`CAVITYWALL:001`
#   is the insulated cavity wall of Trano's defaults; `tutorials/buildings/house/old.yaml` shows an uninsulated
#   one with its own `material` and `constructions` sections);
# * the systems: `nominal_heating_power` of the heat pump, `tank_volume` of the buffer, the supply set point
#   `temperature_supply_setpoint_heat_pump`, the radiator's `nominal_heating_power_positive_for_heating`;
# * the controllers of the YAML (the rule-based reference, not the MPC): the thermostat set point, setback and
#   schedule of `emission_control`, the occupancy schedule and gains.
#
# The weather file of the house (Uccle, typical meteorological year) is copied next to the YAML. The names of the
# elements (`SPACE:001`, `BOILER:001`...) are the names of the variables of the model; keep them, or read the new
# ones from `model.state_names`.

# %%
YAML = """
default: !include_default
spaces:
  - id: SPACE:001
    occupancy:
      parameters:
        occupancy: 3600*{7, 22}
        gain: "[40; 75; 40]"
        heat_gain_if_occupied: 1/7/3
    parameters:
      floor_area: 100.0
      average_room_height: 2.5
    external_boundaries:
      external_walls:
        - {surface: 20, azimuth: 0, tilt: wall, construction: CAVITYWALL:001}
        - {surface: 30, azimuth: 1.57, tilt: wall, construction: CAVITYWALL:001}
        - {surface: 50, azimuth: 3.14, tilt: wall, construction: CAVITYWALL:001}
      windows:
        - {surface: 5.0, azimuth: 0, tilt: wall, construction: INS3AR2020:001}
        - {surface: 2.0, azimuth: 3.14, tilt: wall, construction: INS3AR2020:001}
      floor_on_grounds:
        - {surface: 50.0, construction: CONCRETESLAB:001}
    emissions:
      - radiator:
          id: RADIATOR:001
          parameters:  # a low-temperature radiator: its design output at 45/35 degC
            nominal_heating_power_positive_for_heating: 3000
            water_inlet_temperature_at_nominal_condition: 318.15
            water_outlet_temperature_at_nominal_condition: 308.15
            air_temperature_at_nominal_condition: 293.15
      - valve:
          id: VALVE:001
          control:
            emission_control:
              id: EMISSION_CONTROL:001
              parameters:  # the thermostat of the rule-based reference: 21 degC from 7h to 22h, 17 degC at night
                schedule: 3600*{7, 22}
                temperature_heating_setpoint: 294.15
                temperature_heating_setback: 290.15
systems:
  - boiler:
      id: BOILER:001
      variant: air_water_heat_pump
      parameters:  # 3.5 kW at -8 degC outdoor and 45 degC supply, a 150 l buffer
        nominal_heating_power: 3500
        temperature_supply_setpoint_heat_pump: 318.15
        use_storage_tank: true
        tank_volume: 0.15
        height_of_tank_without_insulation: 1.0
      control:
        boiler_control:
          id: BOILER_CONTROL:001
          variant: simple
      inlets: [SPLIT_VALVE:002]
  - pump:
      id: PUMP:001
      parameters: {m_flow_nominal: 0.08}
      control:
        collector_control:
      inlets: [BOILER:001]
  - three_way_valve:
      id: THREE_WAY_VALVE:001
      parameters: {m_flow_nominal: 0.08}
      control:
        three_way_valve_control:
      inlets: [SPLIT_VALVE:002, PUMP:001]
      outlets: [TEMPERATURE_SENSOR:001]
  - temperature_sensor:
      id: TEMPERATURE_SENSOR:001
      outlets: [RADIATOR:001]
  - split_valve:
      id: SPLIT_VALVE:002
      inlets: [VALVE:001]
      outlets: [BOILER:001]
"""


def building(text: str, name: str) -> TutorialBuilding:
    """Write the YAML next to the weather file and wrap it as a tutorial building (the models are generated on
    first use)."""
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / f"{name}.yaml"
    path.write_text(text.lstrip())
    shutil.copy(WEATHER, OUTPUT / WEATHER.name)
    return TutorialBuilding(path, start=START, spin_up_days=1.0 if quick_mode() else 3.0)


yours = building(YAML, "your_building")
with timed("Trano + rumoca, the MPC model and the closed-loop model"):
    model, closed_loop = yours.model, yours.closed_loop
print(f"{model.n_states} states, {model.n_controls} controls, {model.n_disturbances} disturbances")
print("controls", model.control_names)
print("comfort ", model.comfort_states, "meters", model.meters)

# %% [markdown]
# ## 2. Plan its heating
#
# The widget re-solves one open-loop plan of your building from the state its own controllers reach after the
# spin-up (prices, comfort band, horizon, comfort weight); the static plan below it is the one the notebook
# renders without a kernel.

# %%
tariffs = price_options(START, 2.0)
display(plan_explorer(yours, tariffs))

# %%
horizon = 6 if quick_mode() else 24
controller = ModelPredictiveController(model, MPCSettings(horizon=horizon), ComfortSpec(), tariffs["day-ahead"])
plan = controller.solve(yours.initial_state(), yours.forecaster.forecast(START, horizon, 3600.0))
print(f"{plan.status} in {plan.iterations} iterations, {plan.solve_time:.1f} s; cost {plan.economic_cost:.2f} EUR")
display(plot_plan(plan, model, "your building, day-ahead prices"))

# %% [markdown]
# ## 3. Closed loop: the MPC against the controllers of the YAML
#
# One day of closed loop (the model is the building here: no mismatch, the true state is handed to the controller,
# so that the run takes minutes), and the same day with the thermostat, heating curve and buffer controller of
# the YAML, costed with the same tariff and assessed against the same band. The band is the one of tutorial 01
# (21 to 24 degC when occupied, a 3 K setback at night).


# %%
def compare(house: TutorialBuilding, label: str) -> dict[str, ClosedLoopResult]:
    """The MPC and the rule-based reference over the live period, as rows of the KPI table."""
    lower, upper, setback = _house.BAND_DEFAULT
    run = scenario(label, tariffs["day-ahead"], comfort_spec(lower, upper, setback), START, LIVE_DAYS)
    run = run.model_copy(update={"estimator": "perfect"})
    with timed(f"{label}: MPC closed loop"):
        mpc = run_scenario(house.model, run, plant=house.plant(), forecaster=house.forecaster)
    reference = house.rule_based(run)
    return {f"{label} | MPC": mpc, f"{label} | rule based": reference}


results = compare(yours, "your building")
display(kpi_table(results, ("total_cost", "energy_kwh", "comfort_violation_kh", "peak_power_kw", "low_price_share")))
display(plot_closed_loop(results["your building | MPC"], "your building, MPC, day-ahead prices"))
display(plot_closed_loop(results["your building | rule based"], "your building, the controllers of the YAML"))

# %% [markdown]
# ## 4. Two versions side by side
#
# A second version of the building, from the same text with one change (here a 300 l buffer instead of 150 l:
# twice the fast storage the MPC charges in the cheap hours), run through the same comparison. Replace the
# `replace` with your own change, or with a second YAML.

# %%
VARIANT = YAML.replace("tank_volume: 0.15", "tank_volume: 0.30")
variant = building(VARIANT, "your_variant")
results.update(compare(variant, "300 l buffer"))
display(kpi_table(results, ("total_cost", "energy_kwh", "comfort_violation_kh", "peak_power_kw", "low_price_share")))

# %% [markdown]
# ## Key takeaways
#
# * The YAML is the only input: one cell rewrites the building, Trano and rumoca regenerate the MPC model and the
#   rule-based reference, and the rest of the tutorials (plan widget, closed loop, KPIs) applies unchanged.
# * The comparison with the controllers of the YAML is the fair one: same building, same weather, same tariff,
#   same comfort band, same spun-up state; what the MPC gains comes from timing the heat pump on the price
#   and from using the buffer and the walls as storage.
# * A larger buffer helps the MPC and hurts the thermostat (as written: the MPC's day falls from 1.41 to 1.38 EUR
#   with two thirds of the energy bought in the cheap hours, the thermostat's rises from 1.53 to 1.64 EUR): the
#   reference charges the buffer to its set point whatever the price and loses more from a bigger tank, the MPC
#   charges it when electricity is cheap and only as much as the house will need.
#
# ## Further reading
#
# * Trano's documentation for the YAML schema (constructions, materials, systems and their parameters);
#   `tutorials/buildings/` for the other buildings of the tutorials (the old house, the PV house, the office).
# * `01_house_heat_pump` for what the plan and the closed loop show in detail; `03_calibration_from_yaml` to fit
#   the parameters of your YAML on monitoring data instead of guessing them.
