"""Tutorial 2 - Monitoring data and calibration.

A real building is not the Trano model: the constructions, the air change rate or the furniture differ. mpctrano
calibrates groups of parameters of the BuildingModel (the conductances and capacities of the envelope, the heat capacity
of the air, the infiltration) on monitoring data: the zone air temperatures, the controls applied and, when read,
the energy meters. Here the "real" building is a copy of the Trano model with the groups at random values, so that
the calibrated values can be compared with the truth.
"""

import pandas as pd

from mpctrano import Calibrator, MonitoringData, Project, evaluate, random_factors, simulate_fit
from mpctrano.plot import plot_fit
from tutorials._common import MODEL, OUTPUT, data_days, save_figure

project = Project.from_yaml(MODEL)
truth = random_factors(relative_std=0.3, seed=4)
true_building = project.true_building(relative_std=0.3, seed=4)

# --- 1. Monitoring data. In practice: a CSV exported from the building management system with the time, the zone
#        air temperatures, the controls of the valves and of the heat pump, and the meter of the heat pump.
synthetic = project.synthetic_data(days=data_days(7), start="2023-01-02", building=true_building)
OUTPUT.mkdir(exist_ok=True)
csv = synthetic.data.to_csv(OUTPUT / "monitoring.csv", temperature_unit="C")
print(pd.read_csv(csv).head().round(2).to_string())

# --- 2. Load it. Columns follow the names of the model (space_001.TAir, valve_001_y.y, boiler_001.EEle...).
#        The weather and the internal gains are usually not measured: they are filled from the weather file and
#        the occupancy schedule of the YAML.
data = MonitoringData.from_csv(csv, temperature_unit="C")
model = project.model
partial = MonitoringData(data.frame[[*model.comfort_states, *model.control_names, *model.meters]])
print("missing disturbances:", partial.validate(model).missing_disturbances)
complete = partial.fill_disturbances(model, project.forecaster)
print("complete:", complete.validate(model).complete)

# --- 3. Calibrate with 25 % of the data kept for validation.
train, validation = complete.split(fraction=0.75)
calibrator = Calibrator(model)
print("groups:", {name: len(members) for name, members in calibrator.members.items()})
calibrated, result = calibrator.fit(train, validation)
print(f"IPOPT: {result.status} in {result.iterations} iterations, {result.solve_time:.1f} s")
print(result.metrics().round(3).to_string())
print(pd.DataFrame({"calibrated": result.factors, "true": {name: truth[name] for name in result.factors}}).round(3))

# --- 4. Plot measured vs simulated air temperatures on the validation period (simulated from where the training
#        ends).
_, states = evaluate(calibrated, train, result.initial_state)
x0 = states[:, -1]
fit = simulate_fit(calibrated, validation, x0)
save_figure(plot_fit(fit, "Validation period: measured (dotted) vs simulated"), "02_calibration_fit")

# --- 5. Keep the calibration: the parameters it changed and the fit, in JSON.
(OUTPUT / "house.calibration.json").write_text(result.model_dump_json(indent=2))
