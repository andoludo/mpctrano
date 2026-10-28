# State estimation

The MPC needs the full state of the BuildingModel (the air, the layers of the walls, the panes, the
radiator and buffer water) at the start of each horizon, but only the zone air temperatures are
measured. Every estimator follows the cycle *correct with the measurement of the step → use the
state → predict with the inputs applied during the step* (`StateEstimator`).

`MovingHorizonEstimator(model, dt, start, settings=EstimatorSettings(window=12))` keeps the inputs
applied and the measurements of the last `window` steps and, at each `correct`, finds the trajectory
of the model closest to them with the moving horizon estimation of `trano_optimization` (Radau
collocation, IPOPT): the measured air temperatures (`measurement_noise`), the model errors over each
step (`process_noise`) and the deviation of the first state of the window from its prior
(`arrival_noise`) are weighted by their standard deviations. When the window slides, the estimate of
its second state becomes the prior of the next window; when an estimation fails, the prior is
propagated over the step instead. Each estimation takes a fraction of a second.

On the boiler house of the tests started 2 K colder than the estimator assumes, the mean error over
its 26 temperature states falls from 1.9 K to 0.05 K within a day, where the uncorrected model keeps
0.84 K; the closed-loop MPC with estimation costs the same as with the true state.

`PerfectEstimator(plant)` returns the true state of a simulated plant, to study the controller alone.
