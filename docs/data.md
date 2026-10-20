# Monitoring data

`MonitoringData` wraps a regularly sampled, time-indexed `DataFrame` whose columns are named after
the variables of the `FastModel`:

| Column                          | Unit | Role                                                            |
|---------------------------------|------|-----------------------------------------------------------------|
| `time`                          |      | timestamp (first column of the CSV by default)                  |
| `<zone>.TAir`                   | K/°C | measured air temperature of a zone (at least one zone)          |
| `valve_001_y.y`, `boiler_001_y.y`, ... | - | the controls applied, held from each sample to the next   |
| `boiler_001.EEle`, `boiler_001.EFue`, ... | kWh | readings of the energy meters, cumulative (optional)  |
| `weather_inputs.TDryBul`, ...   | K, W/m², m/s, rad | the weather channels (optional)                     |
| `occupancy_1_gai.QCon`, `.QRad` | W/m² | internal gains per floor area (optional; `occupancy_1.gai.*` before Trano 0.57) |

```python
from mpctrano import MonitoringData

data = MonitoringData.from_csv("monitoring.csv", temperature_unit="C", rename={"Living room": "space_001.TAir"})
report = data.validate(model)              # measured zones, controls, meters, missing columns, gaps
data = data.fill_disturbances(model, forecaster)  # weather file and occupancy schedules for the rest
training, validation = data.split(fraction=0.75)  # the validation starts where the training ends
```

`resample(dt)` averages the data on another period, `window(start, end)` cuts a period,
`to_csv(path, temperature_unit="C")` writes it back.

Without a building at hand, `Project.synthetic_data()` (or `generate_monitoring_data`) records data on a
copy of the model with its parameter groups at random values (`perturb`), operated by a thermostat
with a random set point that excites the dynamics: the controls, the measured air temperatures with
sensor noise, the meter readings and the disturbances.
