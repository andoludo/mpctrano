"""Tutorial 6 - The whole pipeline in a few lines, with the ``Project`` facade.

    Trano model -> monitoring data -> calibration -> state estimation + MPC -> scenario comparison

The command line offers the same steps:

    mpctrano generate house.yaml
    mpctrano synthetic-data house.yaml --output monitoring.csv --days 14
    mpctrano calibrate house.mo monitoring.csv
    mpctrano run house.yaml --calibrated house.calibrated.mpc.json --scenario time_of_use --days 2
"""

from mpctrano import CalibrationSettings, MonitoringData, Project, compare, scenario_catalogue
from tutorials._common import MODEL, OUTPUT, days

project = Project.from_yaml(MODEL)  # 1. Trano generates the RC model
building = project.true_building(seed=2)  # the "real" building (a perturbed copy)

monitoring = project.synthetic_data(days=days(10), building=building).data  # 2. monitoring data
monitoring.to_csv(OUTPUT / "pipeline_monitoring.csv")
data = MonitoringData.from_csv(OUTPUT / "pipeline_monitoring.csv")

result = project.calibrate(data, settings=CalibrationSettings(), validation_fraction=0.25)  # 3. calibration
print(result.metrics().round(3).to_string())

catalogue = scenario_catalogue(days=days(2))  # 4. closed-loop MPC with the calibrated model
runs = {name: project.run(catalogue[name], plant=building) for name in ("flat", "time_of_use", "dynamic_price")}
print(
    compare(runs, baseline="flat")[["energy_kwh", "total_cost", "cost_saving_pct", "comfort_violation_kh"]]
    .round(2)
    .to_string()
)

print(project.save(OUTPUT / "pipeline"))  # 5. everything on disk: models, calibrated interface, data, runs
