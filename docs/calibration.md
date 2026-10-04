# Calibration

## Problem

Given monitoring data with `N` steps of length `dt`, measured zone temperatures `y_k` (a subset of
the states), heating powers `u_k` and disturbances `d_k`, the calibrator solves

```
min_{X, z}   1/(N+1) Σ_k Σ_m ((X[m,k] - y[m,k]) / σ)²  +  λ Σ_j z_j²
s.t.         X[:, k+1] = F(X[:, k], u_k, d_k, p(z))        k = 0..N-1     (multiple shooting)
             p_j = p0_j · exp(z_j),   log(lower_factor) ≤ z_j ≤ log(upper_factor)
             243 K ≤ X ≤ 373 K
```

* `σ` (`measurement_std`) normalises the residuals, `λ` (`regularisation`) pulls the parameters
  towards the Trano initial guess `p0` in log space;
* the multiple-shooting formulation keeps the problem well conditioned for long periods and poor
  initial parameters; IPOPT starts from the simulated trajectory of the initial model, which is
  feasible, so no restoration phase is needed;
* the unmeasured states (envelope, mass, emitter temperatures) are free (`free_initial_states`),
  their trajectories are identified with the parameters;
* fractions (unit `1`, ISO 13790 models) are kept below one; `TGro` is fixed unless requested.

`CalibrationSettings.parameters` selects the parameters with `fnmatch` patterns
(`["*_Ci", "*_Ria", "H_*"]`); by default every parameter but the ground temperature is calibrated.

## Result

`CalibrationResult` holds the parameter values before and after, the IPOPT status and the
`FitMetrics` (RMSE, MAE, maximum error, R² per zone) on the training period and, when given, on a
validation period. The validation fit is an open-loop simulation whose unmeasured initial states
are fitted by linear least squares with the parameters fixed (`evaluate(..., fit_initial_state=True)`).

```python
calibrated, result = Calibrator(model, settings).fit(train, validation)
result.metrics()                      # rmse / mae: training and validation, initial and calibrated
result.summary(0.2, 5.0)              # initial, calibrated, ratio, at_bound per parameter
simulate_fit(calibrated, validation)  # measured vs simulated frame for plots
```

## Identifiability

Zone air temperatures identify the parameters that shape their fast response: air capacities
(`Ci`), direct losses (`Ria`), solar apertures (`gA_*`), couplings (`H_*`). The envelope capacity
and the envelope-to-ground resistance (`Ce`, `Reg`) act on time scales of days and are correlated:
with a week of data they stay close to the prior, which is the purpose of the regularisation. Use
`summary()` to see which parameters moved; a parameter at a bound usually means a structural
mismatch (a missing gain, a wrong floor area) rather than a wrong bound.

Practical advice:

* at least one week of data with varying setpoints (night setback helps a lot) and some sunny days;
* hourly or 15-minute samples; `MonitoringData.resample()` averages finer data;
* keep 25 % of the data for validation (`MonitoringData.split`);
* with `hessian="limited-memory"` the problem needs more iterations but less memory.
