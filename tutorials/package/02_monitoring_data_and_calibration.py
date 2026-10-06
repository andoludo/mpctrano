"""Tutorial 2 - Monitoring data and calibration.

A real building is not the Trano model: the constructions, the air change rate or the solar
apertures differ. mpctrano calibrates the RC parameters on monitoring data (zone temperatures,
heating powers and, when measured, the weather). Here the "real" building is a perturbed copy of
the Trano model so that the calibrated parameters can be compared with the truth.
"""

import pandas as pd

from mpctrano import CalibrationSettings, Calibrator, MonitoringData, Project, simulate_fit
from mpctrano.plot import plot_fit
from tutorials._common import MODEL, OUTPUT, days, save_figure

project = Project.from_yaml(MODEL)
true_building = project.true_building(relative_std=0.3, seed=0)

# --- 1. Monitoring data. In practice: a CSV exported from the building management system with
#        the time, the zone temperatures, the heating powers and the outdoor temperature.
synthetic = project.synthetic_data(days=days(14), start="2023-01-02", building=true_building)
OUTPUT.mkdir(exist_ok=True)
csv = synthetic.data.to_csv(OUTPUT / "monitoring.csv", temperature_unit="C")
print(pd.read_csv(csv).head().round(2).to_string())

# --- 2. Load it. Columns follow the interface names (<zone>_Ti, <zone>_QHea, TOut, ...).
#        The solar irradiance and the internal gains are usually not measured: they are filled
#        from the weather file and the occupancy schedule of the Trano model.
data = MonitoringData.from_csv(csv, temperature_unit="C")
partial = MonitoringData(
    data.frame.drop(columns=[c for c in data.columns if c.startswith("HSol") or c.endswith("QInt")])
)
report = partial.validate(project.model)
print("missing disturbances:", report.missing_disturbances)
complete = partial.fill_disturbances(project.model, project.forecaster)
print("complete:", complete.validate(project.model).complete)

# --- 3. Calibrate with 25 % of the data kept for validation.
train, validation = complete.split(fraction=0.75)
settings = CalibrationSettings(regularisation=1e-2, lower_factor=0.2, upper_factor=5.0)
calibrated, result = Calibrator(project.model, settings).fit(train, validation)
print(f"IPOPT: {result.status} in {result.iterations} iterations, {result.solve_time:.1f} s")
print(result.metrics().round(3).to_string())

# --- 4. Which parameters moved, and how far from the truth?
summary = result.summary(settings.lower_factor, settings.upper_factor)
summary["true_ratio"] = [true_building.parameter(name) / project.model.parameter(name) for name in summary.index]
print(summary.round(3).to_string())
print("Fast air-node parameters (Ci, Ria) are recovered; slow envelope ones stay near the prior when not identifiable.")

# --- 5. Plot measured vs simulated temperatures on the validation period.
fit = simulate_fit(calibrated, validation)
save_figure(plot_fit(fit, "Validation period: measured (dotted) vs simulated"), "02_calibration_fit")

# --- 6. Keep the calibrated model: the interface JSON now carries the calibrated values.
calibrated.write(OUTPUT, "house_calibrated")
