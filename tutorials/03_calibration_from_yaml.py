# %% [markdown]
# # 03 - Calibration: the YAML declares the parameters, IPOPT finds them
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andoludo/mpctrano/blob/main/tutorials/03_calibration_from_yaml.ipynb)
#
# **Learning objectives**
#
# 1. Declare the parameters to calibrate in the Trano YAML itself (`{value, parameter: true, min, max}`), keep it
#    the single source of truth, and see which parameters of the CasADi model each field moves.
# 2. Calibrate the **closed-loop** model (the controllers of the YAML inside, no control input) on monitoring data
#    a real building would give: a noisy air temperature and an electricity meter, no valve positions.
# 3. Read how IPOPT solves it: the whole period on Radau collocation, the states as decision variables, the
#    dynamics as constraints, a weighted least squares with priors.
# 4. Compare the model before and after: the temperatures and the energy on the training days and on two
#    validation days, and the recovered values against the true ones.
#
# The true building, its data and the calibration come from `tutorials/cache/calibration/` (`make cache`); the
# calibration is solved live when the notebook does not run in quick mode (a few minutes).
#
# **Runtimes** (this container): one regeneration of the model by Trano and rumoca about 15 s (one per declared
# parameter, for the sensitivities); the calibration of three days of hourly data, 38 states, two to five minutes.

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

from mpctrano import CalibrationSettings, Calibrator
from mpctrano.calibration import evaluate, simulate_fit
from mpctrano.notebook import quick_mode, timed
from mpctrano.plot import plot_fit
from tutorials import _calibration as study

pd.set_option("display.width", 200)
print(f"quick mode: {quick_mode()}")

# %% [markdown]
# ## 1. The parameters, declared in the YAML
#
# `tests/models/single_zone_calibration.yaml` is the heat pump house of tutorial 01 with its cavity wall spelled out
# and three fields annotated: the conductivity of the insulation, its thickness and the infiltration rate.
# `CalibrationSpec` reads them by path (the convention of `trano.runtime`) and writes the flat copy Trano reads.

# %%
spec = study.spec()
text = study.YAML.read_text()
print(text[: text.index("    id: SPACE:001")])
display(pd.DataFrame([parameter.model_dump(by_alias=True) for parameter in spec.parameters]).set_index("path"))

# %% [markdown]
# Which parameters of the CasADi model a field moves is **measured**: the model is regenerated with the field raised
# by 20 % and the exponent of every parameter that moved is read. The conductivity moves the conductance of the
# insulation layer of the three walls (exponent 1), the thickness moves it with -1 and the capacities of the
# neighbouring nodes a little, the infiltration rate is the `ACH` of the zone.

# %%
groups = study.groups()
for group in groups:
    moved = {
        name: round(exponent, 3)
        for name, exponent in group.exponents.items()
        if not name.startswith(("externalwall_1", "externalwall_2"))
    }
    print(f"{group.name}: value {group.value:g} in [{group.lower:g}, {group.upper:g}] -> {moved} (and the other walls)")

# %% [markdown]
# ## 2. The true building and its monitoring data
#
# The *true* building is the same YAML with other values of the three parameters: a worse insulation, thinner than
# drawn, and a leakier house. It is run by its own controllers (the closed-loop `FastModel`: thermostat, heating
# curve, buffer control) over five cold days of January, after two days of spin-up; the monitoring system records
# the weather, the air temperature with 0.1 K of noise and the electricity meter with 20 Wh of noise. It does
# **not** record the valves or the compressor: the model being calibrated has its controllers inside.

# %%
model = study.model()
print(
    "states",
    model.n_states,
    "| controls",
    model.control_names,
    "| measured",
    model.comfort_states,
    "| meters",
    list(model.meters),
)
display(pd.Series(study.TRUTH, name="true value").to_frame().join(pd.Series(spec.nominal, name="assumed (YAML)")))
data, true_states = study.recorded()
training, validation = study.split(data)
print(f"{len(training) - 1} training hours from {training.index[0]}, {len(validation) - 1} validation hours")
display(data[[*model.comfort_states, *model.meters]].describe().round(2).T)

# %% [markdown]
# ## 3. Before: the model with the values of the YAML
#
# An open-loop simulation of the model from the first measurement, with the assumed parameters: the house loses
# less heat than the true one, so it stays warmer and draws less electricity.

# %%
before = simulate_fit(model, training)
metrics_before, _ = evaluate(model, training)
display(plot_fit(before, "before calibration: training days (dotted: measured)"))
display(metrics_before.frame().round(3))

# %% [markdown]
# ## 4. How IPOPT calibrates it
#
# The calibration is one nonlinear programme over the whole training period:
#
# * **decision variables**: the states at every hour and on the Radau collocation points inside each hour (38
#   states, 2 points), the initial state (the walls and the water are not measured) and one variable per declared
#   parameter, `log(field / value)`, bounded by `log(min / value)` and `log(max / value)`;
# * **constraints**: the collocated dynamics `xdot = f(t, x, d, p(theta))` on every point, with the disturbances of
#   the data and the parameters as expressions of the fields (`p = p0 * (field / value) ** e`);
# * **objective**: the squared errors of the simulated air temperature against the measurements (weighted by the
#   sensor noise), of the energy metered over each hour against the meter (weighted by the meter noise), and the
#   squared deviations of the fields from their prior (the YAML).
#
# Keeping the states as variables (rather than simulating and differentiating) keeps the problem well conditioned
# over long periods and poor initial parameters: IPOPT moves the parameters and the trajectory together. The log
# below is IPOPT's own (one line per iteration: the objective, the infeasibility, the step).

# %%
if quick_mode():
    calibrated, result = study.calibrate()
    print("cached result:", result.status, "in", result.iterations, "iterations,", f"{result.solve_time:.0f} s")
else:
    with timed("calibration"):
        calibrated, result = Calibrator(model, CalibrationSettings(groups=groups, verbose=True)).fit(
            training, validation
        )
print(
    f"{result.variables} variables, {result.constraints} constraints, {result.iterations} iterations, "
    f"{result.solve_time:.0f} s, {result.status}"
)

# %% [markdown]
# ## 5. After: the recovered parameters, the fit on the training and validation days

# %%
recovered = pd.DataFrame({"assumed (YAML)": spec.nominal, "calibrated": result.factors, "true": study.TRUTH})
recovered["error %"] = 100 * (recovered["calibrated"] - recovered["true"]) / recovered["true"]
display(recovered.round(3))
display(result.metrics().round(3))

# %%
x0 = study.initial_state(result)
after = simulate_fit(calibrated, training, x0)
display(plot_fit(after, "after calibration: training days (dotted: measured)"))
after_validation = simulate_fit(calibrated, validation)
before_validation = simulate_fit(model, validation)
display(plot_fit(after_validation, "after calibration: validation days (dotted: measured)"))

# %% [markdown]
# ## Key takeaways
#
# TAKEAWAYS
#
# ## Further reading
#
# * `docs/calibration.md` for the groups and the problem; `mpctrano.yaml_parameters` for the annotations.
# * Trano's `runtime` package addresses the same YAML fields by path for the Buildings models compiled with
#   OpenModelica (`trano.runtime.specification`): the two conventions agree.
