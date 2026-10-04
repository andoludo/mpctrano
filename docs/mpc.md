# Model predictive control, tariffs and flexibility

## Optimisation problem

At every step, from the estimated state `x0` and the forecasts over `N` steps of `dt` seconds:

```
min   Σ_k π_k · (Σ_z u_{z,k}) · dt / 3.6e6              energy cost [EUR], π in EUR/kWh
    + c_cap · P_peak / 1000                              capacity charge [EUR/kW] on the horizon peak
    + Σ_k (w₁ (s⁻ + s⁺) + w₂ (s⁻² + s⁺²)) · dt / 3600     comfort violations [K·h]
    + w_lim · Σ_k s_lim,k · dt / 3.6e6                   energy above a demand response power limit
    + w_ramp · Σ_k |u_k - u_{k-1}|²                       optional smoothing
s.t.  x_{k+1} = F(x_k, u_k, d_k, p)
      min_power ≤ u ≤ max_power (design power of each zone from the Trano interface by default)
      T_lower,k - s⁻_k ≤ T_zone,k+1 ≤ T_upper,k + s⁺_k     soft comfort band on every zone
      Σ_z u_{z,k} ≤ limit_k + s_lim,k,     Σ_z u_{z,k} ≤ P_peak
```

`MPCSettings` holds the horizon, the time step, the penalties and the power bounds. The problem
is built once (`casadi.Opti` with parameters) and every `solve()` only updates the initial state,
the disturbance forecast, the prices, the comfort bounds and the power limits, with a shifted warm
start. If IPOPT does not converge the last iterate is returned with `success=False` (the closed
loop keeps running and counts the failures).

Comfort bounds apply at the end of each step (the occupied band starting at 7h constrains the
temperature reached at 7h); prices apply to the energy of the step.

## Tariffs

| Tariff                 | Description                                                           |
|------------------------|-----------------------------------------------------------------------|
| `FlatTariff`           | one price                                                             |
| `TimeOfUseTariff`      | peak / off-peak hours, weekends off-peak unless `weekend_peak`        |
| `DynamicTariff`        | a price series (`from_csv`, `synthetic_day_ahead`), held between samples |
| `DemandResponseTariff` | any tariff + `DemandResponseEvent`s: price multiplier and/or power cap |

Every tariff can carry a `capacity_charge` [EUR/kW] on the peak power of the horizon. A
`DemandResponseEvent` power cap is a soft constraint with a high penalty (`power_limit_penalty`):
it is respected unless comfort would collapse.

## Comfort

`ComfortSchedule` defines occupied/unoccupied bands in °C with the occupied hours;
`ComfortSpec` assigns a schedule per zone (`default` otherwise) and `map("widen", delta)` or
`map("constant")` transforms all of them.

## Flexibility

The flexibility of a building is what the thermal mass allows between the bounds: with a cheap
night the controller pre-heats and lets the temperature float during the expensive hours; with a
capacity charge it spreads the morning boost; during a demand response event it sheds load. The
comfort band is the lever: `wide_comfort_band` in the scenario catalogue widens it by 1 K.

Costs are only comparable under the same tariff. `value_of_flexibility(model, scenario)` runs the
scenario twice, with the price-aware MPC and with a price-agnostic one (flat price, same comfort
band and model), costs both runs with the tariff of the scenario and reports the saving, the
energy shifted and the share of energy bought at low prices. For the three-zone test building the
envelope time constant is about 40 h: part of the heat stored at night pays back after the
expensive hours, so the controller starts pre-heating when the price spread exceeds roughly four
(e.g. 0.40 / 0.10 EUR/kWh), and more when the upper bound leaves room (`widen(2.0, "upper")`).

The controller can only exploit the storage it believes in: with the uncalibrated Trano
parameters against a different real building, the pre-heating is mis-sized and the comfort band
is missed (tutorial 5 shows both cases). Calibrate before trusting the price response.

`compare(runs, baseline="flat")` puts several runs side by side (same tariff for the costs to be
comparable):

* `total_cost`, `cost_saving_pct` - energy plus capacity cost and the saving against the baseline,
* `energy_kwh`, `energy_change_pct` - pre-heating costs some extra energy,
* `shifted_energy_kwh` - energy moved in time against the baseline (half of the L1 distance of the
  hourly profiles),
* `low_price_share`, `mean_price` - share of the energy bought below the median price, average price paid,
* `comfort_violation_kh`, `max_violation`, `peak_power_kw`.

`price_response(run)` gives the energy consumed at each price level.
