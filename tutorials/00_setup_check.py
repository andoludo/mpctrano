# %% [markdown]
# # 00 - Setup check
#
# **Learning objectives**
#
# - confirm that the toolchain of the tutorials is installed: `trano` (YAML to Modelica),
#   `rumoca` (Modelica to CasADi), `casadi` with IPOPT, `mpctrano`;
# - generate a first CasADi model from a Trano YAML and simulate it.
#
# Run this notebook first; every other tutorial assumes it passes. In continuous integration
# the notebooks run with `MPCTRANO_QUICK=1` (shorter periods), see `tutorials/README.md`.

# %%
import casadi
import numpy as np
import rumoca
import trano

import mpctrano
from tutorials._common import MODEL, QUICK

print(f"trano {trano.__version__ if hasattr(trano, '__version__') else 'installed'}")
print(f"rumoca {rumoca.version()}")
print(f"casadi {casadi.__version__}")
print(f"mpctrano {mpctrano.__version__}")
print(f"quick mode: {QUICK}")

# %% [markdown]
# ## From a Trano YAML to a CasADi model
#
# `RCModel.from_yaml` runs the whole chain: Trano renders the flat RC Modelica model of the
# building (`building_mpc`) and its interface, Rumoca translates it into the CasADi ODE
# `xdot = f(x, u, d, p)` whose parameters stay symbolic.

# %%
model = mpctrano.RCModel.from_yaml(MODEL)
print("states      ", model.state_names)
print("controls    ", model.control_names)
print("disturbances", model.disturbance_names)
print("parameters  ", model.n_parameters)
print(model.continuous)

# %% [markdown]
# ## IPOPT is available
#
# A one-step MPC problem on the model confirms that CasADi can call IPOPT.

# %%
from mpctrano import ComfortSpec, FlatTariff, ModelPredictiveController, MPCSettings, WeatherForecaster

controller = ModelPredictiveController(model, MPCSettings(horizon=6), ComfortSpec(), FlatTariff(price=0.25))
forecaster = WeatherForecaster.from_model(model)
solution = controller.solve(model.initial_state, forecaster.forecast(np.datetime64("2023-01-16T00:00"), 6, 3600.0))
assert solution.success, solution.status
print(f"IPOPT: {solution.status} in {solution.iterations} iterations, {solution.solve_time * 1000:.0f} ms")
print(solution.controls.round(0))

# %% [markdown]
# **Key takeaways**: the toolchain works end to end; the next notebooks build on the same
# `RCModel` object. **Further reading**: `tutorials/README.md` (installation and troubleshooting),
# `docs/index.md` (the workflow of the package).
