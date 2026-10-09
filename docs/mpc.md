# Model predictive control, tariffs and flexibility

## Energy systems and the meter

With trano >= 0.36 the YAML systems are part of the model (interface version 2): a heat pump
makes the heating input of its zones *electrical* (`<zone>_PHea`, heat `= COP(TOut, TSup) · P`),
a chiller adds `<zone>_PCoo`, a hot water tank, a battery and an electric vehicle add states and
inputs, PV adds a generation disturbance. `mpctrano.meter.Meter` builds the balance at the meter:

```
P_net    = Σ electrical inputs − Σ battery discharge − Σ PV generation
P_import − P_export = P_net,  P_import, P_export ≥ 0
P_meter  = P_heat + P_import          # billed at the tariff; P_heat = ideal thermal inputs
P_gas    = Σ Q_boiler / efficiency    # billed at MPCSettings.gas_price
```

The interface also gives what the ODE cannot carry: the compressor limit shared by the zones (and
tanks) of one heat pump, the state of charge bounds, the tank temperature band and the sessions of
the vehicles. `RCModel.control_groups`, `storage_bands`, `vehicles`, `control_bounds()` and
`state_bounds()` expose them.

## Optimisation problem

At every step, from the estimated state `x0` and the forecasts over `N` steps of `dt` seconds:

```
min   Σ_k π_k · P_meter,k · dt / 3.6e6                   energy cost [EUR], π in EUR/kWh
    − Σ_k π_exp,k · P_export,k · dt / 3.6e6               export revenue
    + π_gas · Σ_k P_gas,k · dt / 3.6e6                    fuel of the boilers
    + c_cap · P_peak / 1000                               capacity charge [EUR/kW] on the horizon peak
    + Σ_k (w₁ (s⁻ + s⁺) + w₂ (s⁻² + s⁺²)) · dt / 3600     comfort violations of the zones and tanks [K·h]
    + w_lim · Σ_k s_lim,k · dt / 3.6e6                    energy above a demand response power limit
    + w_ramp · Σ_k |u_k - u_{k-1}|²                        optional smoothing
    + w_ev · Σ_k v_k / 3.6e6                               energy missing in a vehicle at departure
s.t.  x_{k+1} = F(x_k, u_k, d_k, p)
      lower ≤ u ≤ upper (design power, compressor, charger and tank limits of the interface)
      Σ_{i ∈ heat pump} u_{i,k} ≤ P_el,max                 shared compressor limit
      state bounds (temperatures, state of charge)
      T_lower,k - s⁻_k ≤ T_zone,k+1 ≤ T_upper,k + s⁺_k      soft comfort band on every zone and tank
      P_meter,k ≤ limit_k + s_lim,k,     P_meter,k ≤ P_peak
      u_ev,k ≤ plugged_k · P_max,    E_ev,k+1 + v_k ≥ departure_k · E_target
```

`MPCSettings` holds the horizon, the time step, the penalties and the power bounds. The problem
is built once (`casadi.Opti` with parameters) and every `solve()` only updates the initial state,
the disturbance forecast, the prices, the comfort bounds and the power limits, with a shifted warm
start. If IPOPT does not converge the last iterate is returned with `success=False` (the closed
loop keeps running and counts the failures).

Comfort bounds apply at the end of each step (the occupied band starting at 7h constrains the
temperature reached at 7h); prices apply to the energy of the step.

## Plant mismatch

`Mismatch` describes how the plant differs from the control model: another RC structure generated
from the same YAML (`plant_model_type`, R4C3 by default so that the plant has the emitter state
the R3C2 controller lacks), log-normally perturbed envelope parameters (`parameter_std`), sensor
noise and forecast noise. `Project.plant(mismatch)` builds it and `Project.run(scenario)` uses it
unless a plant is given; `Scenario.mismatch` keeps it with the scenario. The signals of the plant
are matched by name, so the controller only needs the measured states (zone temperatures, tank,
batteries), the controls and the disturbances to exist in both.

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

## Capacity tariff

`CapacityTariff(base=..., capacity_price=50.0)` bills, like the Flemish distribution tariff, the
average of the monthly peaks (highest quarter-hour average power of the month, floored at
`minimum_peak_kw`) at `capacity_price` EUR/kW/year. One kilowatt of monthly peak costs
`capacity_price / 12`, the `capacity_charge` the controller applies to the larger of the peak of
the month so far (`peak_so_far`, carried by the closed loop) and the peak of its horizon. With
hourly steps the peak is an hourly average, which underestimates the quarter-hour peak.

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

### Explicit flexibility

`ActivationRequest(start, duration_hours, direction, power)` asks to shed (`down`) or absorb
(`up`) a power against the baseline; `respond(model, scenario, request, plant=factory)` runs the
baseline and the response from the same plant state (the request becomes a soft cap or floor on
the billed power relative to the mean baseline power of the window, as a baseline-settled
aggregator would do) and `ActivationResult.metrics` reports the delivered energy and power, the
delivery ratio, the rebound energy, peak, duration and ratio, the cost and comfort changes.
`flexibility_function` reads the response to a price step the same way (Junker et al., 2018).

### Flexibility envelope

`FlexibilityEnvelope.compute(controller, x0, forecast, durations)` finds, for each duration, the
largest constant change of the billed power the controller can hold from the current state while
its plan keeps the comfort violations below a tolerance (bisection on a per-solve cap or floor,
`solve(power_limits=..., power_floors=...)`), with the energy and the extra cost of that plan.
