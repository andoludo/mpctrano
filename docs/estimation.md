# State estimation

The controller needs the full state `x` at the start of every horizon but only the zone air
temperatures `<zone>_Ti` are measured. The two estimators share the protocol

```python
estimate = estimator.correct(measurement)   # measured zone temperatures (NaN = sensor missing)
...                                          # the MPC uses `estimate`
estimator.predict(control, disturbance)      # inputs applied during the step
```

## Kalman filter

For fixed parameters the RK4 step of an RC model is exactly affine, `x⁺ = A x + Bu u + Bd d + c`
(`RCModel.affine`). `KalmanFilter` implements the standard predict/update with a process noise
`process_std` per state, a measurement noise `measurement_std` and an initial uncertainty
`initial_std`. Missing measurements are skipped.

## Moving horizon estimation

`MovingHorizonEstimator` solves, over the last `window` steps,

```
min  |x_0 - x_prior|² / prior_std²  +  Σ |w_k|² / process_std²  +  Σ |y_k - C x_k|² / measurement_std²
s.t. x_{k+1} = F(x_k, u_k, d_k) + w_k,   243 K ≤ x ≤ 373 K
```

with IPOPT (`casadi.Opti`, one problem per window length, cached). The prior of the next window is
the second state of the current solution (shifted arrival cost). It handles bounds and is the
natural counterpart of the MPC problem; for a linear RC model its estimates match the Kalman
filter's, at a higher computational cost.
