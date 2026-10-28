# Model predictive control, tariffs and flexibility

## Optimisation problem

At every step, from the estimated state `x0`, the time `t0` and the forecasts over `N` steps of `dt`
seconds, `ModelPredictiveController` solves with IPOPT:

```
min   Σ_k (π_k b_k − π^in_k (b_k − e_k) + π^fuel_k f_k)    energy cost [EUR], prices in EUR/kWh
    + c_cap · P_peak / 1000                                 capacity charge [EUR/kW] on the peak electric power
    + Σ_k (w₁ (s⁻ + s⁺) + w₂ (s⁻² + s⁺²)) · dt / 3600       comfort violations [K·h]
    + w_lim · Σ_k s_lim,k · dt / 3.6e6                      electricity above a demand response power limit
    + w_move · Σ_k |v_k − v_{k−1}|² + w_effort · Σ_k |v_k|²   moves and effort of the controls (scaled to [0, 1])
s.t.  the BuildingModel collocated on Radau IIA points in each step
      the bounds of each control (0-1 for the valves, the heat pump speed and the boiler load)
      T_lower,k − s⁻_k ≤ T_zone,k+1 ≤ T_upper,k + s⁺_k        soft comfort band on every zone
      P_k ≤ limit_k + s_lim,k,     P_k ≤ P_peak                electric power bought from the grid
```

`e_k` is the net electricity of step `k` [kWh], read from the meters of the model (the electricity
of the heat pumps and fans, the heat of the ideal emitters, minus the PV production), `b_k` its
smooth positive part (bought, the rest is sold at the feed-in price), `f_k` the fuel the boilers drew
and `P_k = b_k / dt` the electric power. Since the energy comes from the meters, the efficiency of the
systems is part of the plan: the COP of a heat pump is better in the mild afternoon than in the cold
night, the efficiency of a boiler depends on its load and its water temperature.

`MPCSettings` holds the horizon, the time step, the penalties, `fuel_price` and `feed_in_price` (the
tariff price when not given) and the IPOPT options. The problem is built once (`casadi.Opti` with the
time, the forecasts, the prices and the bounds as parameters) and every `solve()` updates them and
warm starts IPOPT from a CVODES simulation of the previous plan shifted by one step, with a small
initial barrier parameter (`warm_mu_init`), which halves the iterations of the receding-horizon
solves. If IPOPT does not converge the last iterate is returned with `success=False` (the closed loop
keeps running and counts the failures), and the next solve starts cold.

Comfort bounds apply at the end of each step (the occupied band starting at 7h constrains the
temperature reached at 7h); prices apply to the energy of the step. `controller.energy` gives the
electricity bought and net, the fuel and the electric power of each step of the last plan.

On the heat pump house of the tests (27 states), a 24-hour plan with hourly steps solves in 5 to 20 s;
the six-zone heat pump house of the Trano test suite (111 states) takes over a minute.

## Tariffs

| Tariff                 | Description                                                           |
|------------------------|-----------------------------------------------------------------------|
| `FlatTariff`           | one price                                                             |
| `TimeOfUseTariff`      | peak / off-peak hours, weekends off-peak unless `weekend_peak`        |
| `DynamicTariff`        | a price series (`from_csv`, `synthetic_day_ahead`), held between samples |
| `DemandResponseTariff` | any tariff + `DemandResponseEvent`s: price multiplier and/or power cap |

The tariffs price the electricity; the fuel has its own price (`MPCSettings.fuel_price`). Every
tariff can carry a `capacity_charge` [EUR/kW] on the peak electric power of the horizon. A
`DemandResponseEvent` power cap applies to the electric power bought from the grid; it is a soft
constraint with a high penalty (`power_limit_penalty`): it is respected unless comfort would
collapse.

## Comfort

`ComfortSchedule` defines occupied/unoccupied bands in °C with the occupied hours;
`ComfortSpec` assigns a schedule per zone (`default` otherwise) and `map("widen", delta)` or
`map("constant")` transforms all of them.

## Flexibility

The flexibility of a building is what the thermal mass and the buffers allow between the bounds:
with a cheap night the controller pre-heats and lets the temperature float during the expensive
hours; with a capacity charge it spreads the morning boost; during a demand response event it sheds
load. The comfort band is the lever: `wide_comfort_band` in the scenario catalogue raises its upper
bound by 2 K.

Costs are only comparable under the same tariff. `value_of_flexibility(model, scenario)` runs the
scenario twice, with the price-aware MPC and with a price-agnostic one (flat price, same comfort
band and model), costs both runs with the tariff of the scenario and reports the saving, the
energy shifted and the share of energy bought at low prices.

The reference of a building is its own control: `run_rule_based` (or `Project.run_rule_based`)
simulates the controllers of the YAML on the same building. On the boiler house of the tests, over a
January day with the band of its thermostat, the MPC costs 12 % less than the controllers of the YAML
under a flat and under a time-of-use tariff, with almost no discomfort (0.05 K·h against 3.7 K·h, the
warm-up of the thermostat), and buys 30 % of its energy off-peak against 10 %.

The controller can only exploit the storage it believes in: with the uncalibrated Trano
parameters against a different real building, the pre-heating is mis-sized and the comfort band
is missed (tutorial 5 shows it). Calibrate before trusting the price response.

`compare(runs, baseline="flat")` puts several runs side by side (same tariff for the costs to be
comparable):

* `total_cost`, `cost_saving_pct` - energy plus capacity cost and the saving against the baseline,
* `energy_kwh`, `energy_change_pct` - pre-heating costs some extra energy,
* `shifted_energy_kwh` - energy moved in time against the baseline (half of the L1 distance of the
  hourly profiles),
* `low_price_share`, `mean_price` - share of the energy bought below the median price, average price paid,
* `comfort_violation_kh`, `max_violation`, `peak_power_kw`.

`price_response(run)` gives the energy bought at each price level.

## Cooling with ideal emitters

The ideal emitter of a zone (`radiator` of variant `ideal` in the YAML) is a heat flow input of the
`FastSmoothBuildingsMPC` model, which Trano bounds to heating (`min=0`). `BuildingModel.with_control_bounds` gives it
a negative lower bound: the same equations cool the zone. The controller costs its heat at
`MPCSettings.heat_cop` and its cooling at `MPCSettings.cooling_eer` (electricity per unit of heat or
cooling); the closed loop meters it alike.

```python
office = BuildingModel.from_yaml("office.yaml").with_control_bounds({"emitter_001_Q.Q_flow": (-12000.0, None)})
settings = MPCSettings(horizon=24, heat_cop=3.5, cooling_eer=3.0)
```

## Electrical storages at the meter

A stationary battery and an electric vehicle are optimised in the balance of the meter next to the
thermal storage of the building (`mpctrano.storage`): linear state-of-charge models with a round-trip
efficiency and power bounds; the vehicle charges while plugged in (`arrival_hour` to `departure_hour`),
leaves with at least `departure_energy_kwh` (soft, at `missing_energy_price`) and comes back
`driven_kwh` lower. The controller carries their state from one solve to the next
(`ModelPredictiveController.advance`, called by the closed loop) and reports the plan in
`MPCSolution.storage`.

```python
settings = MPCSettings(
    horizon=24,
    batteries=(Battery(capacity_kwh=8, max_charge_w=3000, max_discharge_w=3000),),
    vehicles=(ElectricVehicle(capacity_kwh=50, max_charge_w=7400, arrival_hour=18, departure_hour=7, driven_kwh=12),),
)
```

The energy stored in a battery at the end of the horizon is valued at the mean price of the horizon,
so that the controller does not empty it before the end.
