"""Tutorial 6 - The whole pipeline in a few lines, with the ``Project`` facade.

    Trano model -> monitoring data -> calibration -> state estimation + MPC -> comparison with the rule-based control

The command line offers the same steps:

    mpctrano generate house.yaml --closed-loop
    mpctrano synthetic-data house.yaml --output monitoring.csv --days 7
    mpctrano calibrate house.yaml monitoring.csv
    mpctrano run house.yaml --calibrated house.calibration.json --scenario time_of_use --days 2 --rule-based
"""

from mpctrano import MonitoringData, Project, compare, scenario_catalogue
from tutorials._common import MODEL, OUTPUT, data_days, days, horizon

project = Project.from_yaml(MODEL)  # 1. Trano generates the FastSmoothBuildingsMPC model
building = project.true_building(seed=2)  # the "real" building (a perturbed copy)

monitoring = project.synthetic_data(days=data_days(7), building=building).data  # 2. monitoring data
OUTPUT.mkdir(exist_ok=True)
monitoring.to_csv(OUTPUT / "pipeline_monitoring.csv")
data = MonitoringData.from_csv(OUTPUT / "pipeline_monitoring.csv")

result = project.calibrate(data, validation_fraction=0.25)  # 3. calibration of the parameter groups
print(result.metrics().round(3).to_string())

catalogue = scenario_catalogue(days=days(2), horizon=horizon(24))  # 4. closed-loop MPC with the calibrated model
runs = {name: project.run(catalogue[name], plant=building) for name in ("flat", "time_of_use")}
runs["rule based"] = project.run_rule_based(catalogue["time_of_use"], building=project.true_closed_loop(seed=2))
print(
    compare(runs, baseline="rule based")[["energy_kwh", "total_cost", "cost_saving_pct", "comfort_violation_kh"]]
    .round(2)
    .to_string()
)

print(project.save(OUTPUT / "pipeline"))  # 5. everything on disk: model, calibration, data, runs
