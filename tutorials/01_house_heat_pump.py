# %% [markdown]
# # 01 - Single-family house with a heat pump: the reference implementation
#
# **Learning objectives**
#
# 1. Follow the pipeline from the Trano YAML to the CasADi model: what the flat RC model contains,
#    where the heat pump sits in the equations and what the interface tells the controller.
# 2. Read an economic MPC plan: how the comfort band, the price signal and the horizon shape the
#    pre-heating of a high-mass house on floor heating.
# 3. Run the closed loop against a plant that is *not* the control model (another RC structure,
#    perturbed envelope, noisy sensors) and see why state estimation matters.
# 4. Weigh model complexity (R3C2 vs R4C3 controller) against solve time and tracking.
#
# Everything here calls the library; the heavy runs come from `tutorials/cache/house/`
# (`uv run python tutorials/scripts/build_cache.py`) so that the widgets answer in well under five
# seconds. In quick mode (`MPCTRANO_QUICK=1`, the CI) the runs computed live are shortened to one day.
#
# **Runtimes** (hourly steps, this container): one open-loop solve 0.1 to 0.3 s; one 3-day closed
# loop with the extended Kalman filter about 18 s; the full cache (72 runs) about 22 min.

# %%
import pandas as pd
from IPython.display import display
from trano.mpc import RCModelType

from mpctrano import ModelPredictiveController
from mpctrano.notebook import (
    TutorialBuilding,
    closed_loop_explorer,
    comfort_spec,
    kpi_table,
    plan_explorer,
    quick_mode,
    scenario,
    timed,
)
from mpctrano.plot import plot_closed_loop, plot_cop, plot_states
from mpctrano.scenarios import run_scenario
from mpctrano.simulation import run_closed_loop
from tutorials import _house

pd.set_option("display.width", 200)
print(f"quick mode: {quick_mode()}; cached runs: {_house.DAYS:g} days from {_house.START.date()}")
print(f"runs computed live in this notebook: {_house.LIVE_DAYS:g} days")

# %% [markdown]
# ## 1. From the YAML to the CasADi model
#
# `tutorials/buildings/house.yaml` describes two zones (day and night), cavity walls with 10 cm of
# insulation, floor heating at 35/30 degC and an air-water heat pump feeding the zones and a
# domestic hot water tank. `trano create-model house.yaml mpc` renders the flat RC model
# `building_mpc` and its interface; `rumoca` translates the Modelica into a CasADi ODE with
# **symbolic parameters**. The models of every variant are pre-generated in `tutorials/models/house/`
# (`generate_models.py` regenerates them from the YAML).

# %%
building = TutorialBuilding(_house.MODELS, "base")
model = building.control
print("states      ", model.state_names)
print("controls    ", model.control_names)
print("disturbances", model.disturbance_names)
print("systems     ", [(system.kind, system.name) for system in model.systems])
print(f"{model.n_parameters} parameters, e.g. {model.parameter_names[:4]} ...")

# %% [markdown]
# The heat pump does not appear as a block: Rumoca's CasADi target accepts only explicit smooth
# state equations, so Trano inlines it. The control input of each zone is the **electrical** power
# `<zone>_PHea` and the heat delivered is `P * COP(TOut, TSup)`, a bilinear polynomial around the
# EN 14825 rating point A7/W35. Below is the air balance of the day zone as rendered in the
# Modelica source, and the COP curve the controller believes in.

# %%
for line in model.source.splitlines():
    if line.strip().startswith("der(space_001_Ti)"):
        print(line.strip())
heat_pump = next(system for system in model.systems if system.kind == "heat_pump")
print(
    f"compressor limit {heat_pump.max_electrical_power:.0f} W shared by {heat_pump.inputs} and {heat_pump.tank_inputs}"
)
display(plot_cop(heat_pump))

# %% [markdown]
# The interface (`<variant>.mpc.json`) lists the states, inputs and parameters **in the order of the
# CasADi vectors**, the sources of the disturbances (weather variable, orientation of an
# irradiance, occupancy, draw-off profile) and the bounds the ODE cannot carry: compressor limit,
# tank temperature band. `WeatherForecaster` builds the disturbances from these sources and the
# TMY weather file of Uccle.

# %%
forecast = building.forecaster.realisation(_house.START, 48, 3600.0)
display(forecast.describe().round(1).T)

# %% [markdown]
# ## 2. One economic MPC plan
#
# The controller minimises, over the horizon, the electricity bought at the meter (plus comfort,
# tank and vehicle penalties), subject to the ODE, the compressor limit and the comfort band.
# With a flat price it simply tracks the lower bound at the lowest electrical energy; with a
# time-varying price it pre-heats the floor when electricity is cheap and lets the temperature
# float inside the band when it is expensive. Try it: the widget re-solves the plan live.

# %%
tariffs = _house.tariffs()
explorer = plan_explorer(building, tariffs, _house.START)
display(explorer)

# %% [markdown]
# What to look for:
#
# * **Floor heating is slow**: the emitter and the slab store heat, so the pre-heating starts hours
#   before the expensive period and the zone temperature overshoots the lower bound by a fraction
#   of a kelvin, not more. A wider band (lower minimum, higher maximum) lets the controller shift more.
# * **The spike at 17h** empties the compressor hours before; the tank is charged earlier too.
# * **The horizon** matters mostly through the end effect: with 12 h the controller does not see
#   the next morning and stores less.
# * **The comfort weight** trades a few kelvin-hours of violation for cost when the band is tight.

# %% [markdown]
# ## 3. Closed loop with plant mismatch
#
# A plan is not a result: the controller applies the first step, the real building responds,
# the sensors report, the state is estimated and the plan is rebuilt. The *plant* here is the
# R4C3 model of the same YAML (it has the emitter state the R3C2 controller lacks) with envelope
# parameters perturbed by 20 % and 0.1 K of sensor noise (`Mismatch`). The runs are cached; the
# **Recompute** button runs them again (about 18 s each).

# %%
runs = _house.runs()
dimensions = {
    "variant": list(_house.VARIANTS),
    "price": list(_house.PRICES),
    "band": list(_house.BANDS),
    "mismatch": ["mismatch", "perfect"],
}
display(closed_loop_explorer(_house.CACHE, runs, _house.compute, dimensions))

# %% [markdown]
# ### Perfect model vs mismatch, side by side
#
# The same price and band, with the plant equal to the control model and with the mismatch: the
# cost difference is the price of not knowing the building, the comfort violations show where the
# controller was over-confident.

# %%
pairs = {}
for mismatch in ("perfect", "mismatch"):
    for price in ("flat", "day-ahead"):
        pairs[f"{price} / {mismatch}"] = _house.cached(
            variant="base", price=price, band="standard", mismatch=mismatch, days=_house.DAYS
        )
display(kpi_table(pairs))
display(
    plot_closed_loop(pairs["day-ahead / mismatch"], "base house, day-ahead prices, R4C3 plant with perturbed envelope")
)

# %% [markdown]
# ## 4. Why state estimation matters
#
# The controller needs the full state, including the envelope temperature it cannot measure. The
# extended Kalman filter (exact for the linear envelope, linearised around the heat pump terms)
# and the moving horizon estimator both reconstruct it from the measured zone temperatures. Below,
# the estimated envelope temperature against the true one of the plant, then a run without
# estimation: the unmeasured states are held at their initial value.

# %%
reference = pairs["day-ahead / mismatch"]
display(plot_states(reference, ["space_001_Te"], "Envelope temperature of the day zone: estimate vs plant"))


class NoEstimator:
    """Keeps the measured states and leaves the others at their initial value (no estimation)."""

    def __init__(self, model) -> None:
        self.model = model
        self._x = model.initial_state.copy()

    @property
    def state(self):
        return self._x.copy()

    def correct(self, measurement):
        self._x[self.model.measured_indices] = measurement
        return self.state

    def predict(self, control, disturbance):
        return self.state


run = scenario(
    "day-ahead standard", tariffs["day-ahead"], comfort_spec(*_house.BANDS["standard"]), _house.START, _house.LIVE_DAYS
)
estimators = {}
with timed("three estimators, closed loop"):
    for label, estimator in (("Kalman", None), ("MHE", "mhe"), ("none", "none")):
        plant = building.plant(_house.MISMATCH)
        if estimator == "none":
            controller = ModelPredictiveController(model, run.settings, run.comfort, run.tariff)
            estimators[label] = run_closed_loop(
                controller, NoEstimator(model), plant, building.forecaster, run.start, run.steps, name=label
            )
        else:
            chosen = run.model_copy(update={"estimator": estimator or "kalman", "name": label})
            estimators[label] = run_scenario(model, chosen, plant=plant)
display(kpi_table(estimators))

# %% [markdown]
# ## 5. Model complexity: R3C2 vs R4C3 controller
#
# The R4C3 controller carries the emitter state and sees the lag of the floor heating; it is also
# bilinear in the emitter temperature (COP depends on it), so the solver works harder. The plant is
# the same perturbed R4C3 building in both cases.

# %%
complexity = {}
with timed("two controllers, closed loop"):
    for label, structure in (("R3C2 controller", RCModelType.r3c2), ("R4C3 controller", RCModelType.r4c3)):
        control_model = building.model(structure)
        plant = building.plant(_house.MISMATCH)
        complexity[label] = run_scenario(control_model, run.model_copy(update={"name": label}), plant=plant)
table = kpi_table(complexity, ("total_cost", "comfort_violation_kh", "peak_power_kw", "failures"))
table["mean solve time [s]"] = [result.frame["solve_time"].mean().round(3) for result in complexity.values()]
display(table)

# %% [markdown]
# ## Key takeaways
#
# * The YAML is the single source of truth: the same file gives the Buildings model and this
#   CasADi-ready RC model with its heat pump, tank and interface.
# * A heat pump on floor heating is a large, slow thermal battery: pre-heating pays when the price
#   spread is a few times, and the compressor limit, not the comfort band, often binds on cold days.
# * Against the R4C3 plant with a perturbed envelope and noisy sensors, the R3C2 controller of
#   the base house pays about 2 % more over three days and misses the band by some 11 K.h (up
#   to 1 K) where it was over-confident; whether the mismatch costs or saves depends on the
#   direction of the perturbation (the `old` and `radiators` variants show both), the violations
#   are the robust symptom. The state estimator removes most of the gap, the R4C3 controller a
#   little more at a solve-time cost.
#
# ## Further reading
#
# * `tutorials/references.md`, section A: Bacher & Madsen (2011) for the RC structures, Drgoňa et
#   al. (2020) for MPC in buildings, Blum et al. (2021) for the KPIs used here.
# * `docs/mpc.md` for the optimisation problem and `docs/estimation.md` for the estimators.
# * Next: `02_flexibility_use_cases` quantifies what this flexibility is worth under real
#   price signals, capacity tariffs and explicit activation requests.
