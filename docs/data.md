# Monitoring data format

`MonitoringData` wraps a regularly sampled `pandas.DataFrame` with a `DatetimeIndex`. Columns are
named after the signals of the Trano MPC interface (`house.mpc.json`), so the data can be bound to
the model without any mapping:

| Column               | Unit     | Required | Role                                                  |
|----------------------|----------|----------|-------------------------------------------------------|
| `<zone>_Ti`          | K (or °C)| ≥ 1 zone | measured zone air temperature                         |
| `<zone>_QHea`        | W        | yes      | heating power delivered to the zone                   |
| `TOut`               | K (or °C)| no       | outdoor temperature (else from the weather file)      |
| `HSol_<orientation>` | W/m²     | no       | irradiance on an orientation (else computed)          |
| `<zone>_QInt`        | W        | no       | internal gains (else from the occupancy schedule)     |

```python
data = MonitoringData.from_csv("bms_export.csv", time_column="timestamp", temperature_unit="C",
                               rename={"T_living": "living_Ti", "P_living": "living_QHea"})
report = data.validate(model)        # missing columns, gaps, sampling
data = data.resample(3600)           # average to hourly samples
data = data.fill_disturbances(model, WeatherForecaster.from_model(model))
train, validation = data.split(fraction=0.75)
```

Inputs are zero-order-hold: the value in row `k` is applied between sample `k` and `k+1`; the
last row of the inputs is ignored. Temperatures are stored in Kelvin internally.

`Project.synthetic_data()` (or `mpctrano synthetic-data`) produces such a file from a perturbed
copy of the Trano model driven by a randomised thermostat; `SyntheticData.states` keeps the true
states for checks.
