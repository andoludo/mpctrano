# %% [markdown]
# # 00 - Setup check
#
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andoludo/mpctrano/blob/main/tutorials/00_setup_check.ipynb)
#
# **Learning objectives**
#
# - confirm that the toolchain of the tutorials is installed: `trano` (YAML to Modelica),
#   `rumoca` (Modelica to CasADi, the commit `uv sync` builds), `trano-optimization` (collocation),
#   `casadi` with IPOPT, `mpctrano`;
# - generate a first CasADi model from a Trano YAML with the `FastSmoothBuildingsMPC` library and plan its heating.
#
# Run this notebook first; every other tutorial assumes it passes. In continuous integration
# the notebooks run with `MPCTRANO_QUICK=1` (shorter periods), see `tutorials/README.md`.

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
from importlib.metadata import version

import casadi
import pandas as pd
import rumoca

import mpctrano
from tutorials._common import MODEL, QUICK, horizon

print(f"trano {version('trano')}")
print(f"trano-optimization {version('trano-optimization')}")
print(f"rumoca {rumoca.version()}")
print(f"casadi {casadi.__version__}")
print(f"mpctrano {mpctrano.__version__}")
print(f"quick mode: {QUICK}")
assert mpctrano.model.rumoca_exports_building_models(), (
    "rumoca cannot export the FastSmoothBuildings models: run `uv sync`"
)

# %% [markdown]
# ## From a Trano YAML to a CasADi model
#
# `BuildingModel.from_yaml` runs the whole chain: Trano renders the `FastSmoothBuildingsMPC` Modelica model of the
# building (`building_mpc`: the walls, the windows, the heat pump, its buffer, the mixing valve and the
# radiator, the controls of the systems as inputs), rumoca translates it into the CasADi ODE
# `xdot = f(t, x, u, p)`, and the roles of the variables are read from the Modelica model.

# %%
model = mpctrano.BuildingModel.from_yaml(MODEL)
print("states      ", model.n_states)
print("controls    ", model.control_names)
print("disturbances", model.disturbance_names)
print("comfort     ", model.comfort_states)
print("meters      ", model.meters)
print("parameters  ", model.n_parameters)

# %% [markdown]
# ## IPOPT is available
#
# A plan of the MPC on the model confirms that CasADi can call IPOPT (with SPRAL when CasADi ships it).

# %%
from mpctrano import ComfortSpec, FlatTariff, ModelPredictiveController, MPCSettings, WeatherForecaster

steps = horizon(12)
controller = ModelPredictiveController(model, MPCSettings(horizon=steps), ComfortSpec(), FlatTariff(price=0.30))
forecaster = WeatherForecaster.from_model(model)
solution = controller.solve(model.initial_state, forecaster.forecast(pd.Timestamp("2023-01-16"), steps, 3600.0))
assert solution.success, solution.status
print(f"IPOPT: {solution.status} in {solution.iterations} iterations, {solution.solve_time:.1f} s")
print(solution.controls.round(2))

# %% [markdown]
# **Key takeaways**: the toolchain works end to end; the next notebooks build on the same
# `BuildingModel` object. **Further reading**: `tutorials/README.md` (installation and troubleshooting),
# `docs/index.md` (the workflow of the package).
