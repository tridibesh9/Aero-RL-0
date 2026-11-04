# Training setup fixes — 4 November 2026

**Status:** source changes and non-training verification completed. No PPO learning, supervised learning, or model-weight updates were executed. Existing checkpoints remain unchanged. The new policy's convergence is unverified until a future training run.

## Changes and reasons

| Area | Previous behavior | New default |
| --- | --- | --- |
| Command meaning | Moving position target through position PID | World-frame net acceleration in m/s² |
| Inner control | Motor command recomputed at 20 Hz | Attitude control recomputed at 240 Hz, PPO decisions remain 20 Hz |
| Gravity | Position PID compensation | Explicitly added once when converting acceleration to force |
| Observation | Six raw velocity/error values | 19 scaled values including attitude, angular velocity, attitude-controller history and absolute position |
| Failed episodes | Crashes bootstrapped as truncations | Crashes/arena exits terminate the task; time limits remain truncations |
| Arrival | First entry into the 0.1 m sphere | Within 0.1 m, speed <=0.2 m/s, for five control samples |
| Initial goal sampling | 0.5–1 m | 0.2–0.4 m, then radii 1 m, 2.5 m, and full range |
| Training obstacles | Static obstacle in full stage | Distant obstacle for navigation learning |
| Reward | Paper's terminal/progress/boundary/time terms | Progress weight 5, plus distance and speed costs to encourage approach and braking |
| Exploration | Initial Gaussian standard deviation 1 | Initial standard deviation exp(-1.5), about 0.223 |
| PPO updates | Learning rate 3e-4, ten epochs, entropy coefficient 0.01 | Learning rate 1e-4, five epochs, entropy coefficient 0, target KL 0.015 |
| Network and rollouts | Two 256-unit layers, 2048 steps/environment | Two 64-unit layers, 1024 steps/environment, minibatches of 128 |
| Model selection | Four distant goals | Distant goals plus six deterministic goals appropriate to the current curriculum stage |
| Logging | Sparse validation records | Full episode records, reward-component totals, Monitor CSV and PPO progress CSV |
| Evaluator | Implicit old model path | Required model path and its saved configuration |
| Compatibility | Same action shape could hide changed semantics | Saved interface version and scales checked before evaluation/resume |
| PID baseline | Gains used with waypoint offsets | Acceleration gains Kp=1, Kd=2, Ki=0 on each axis |

The new observation order is velocity/2, goal error/5, current RPY/pi, world angular velocity/10, previous attitude-controller RPY/pi, and position/10. Each group contains three values, followed by one scalar giving the fraction of the goal hold interval completed. Scaling parameters are stored in the configuration and checkpoint metadata. No running normalization statistics are required.

The acceleration controller uses force m*(a+[0,0,g]), the desired thrust direction, and the simulator's attitude/motor mixer. It does not call the old position PID. Attitude integral gains are zero, so the observed previous RPY contains the controller history used by its derivative term. Hover, tilt transients and simulator drag remain part of the real PyBullet plant.

ACBF still predicts with an independent copy of that same plant and its controller state. PPO and PPO+ACBF receive exactly the same environment observation. No hand-written goal controller or teacher has been inserted into the PPO policy.

The new dense reward terms are -0.5*distance*dt and -0.1*speed²*dt. Terminal reward remains 100, boundary penalty 10, and time cost 0.1*dt. These are design choices requiring future learning experiments, not empirically tuned convergence guarantees.

## Code-only verification

The test suite passes 22 tests, including nine new learning-setup checks. The new tests patch PPO.learn and Adam.step to fail if accidentally invoked.

- Hover stays at the initial position.
- A 0.5 m/s² command produces the expected direction and scale of motion, allowing attitude lag and drag.
- Ground failure has TimeLimit.truncated=false through the SB3 vector wrapper. A genuine timeout has it true.
- Fast passage through the goal is not a success. Five stationary samples at the goal are.
- The acceleration forecast preserves the live simulator and matches subsequent held-command flight.
- An untrained PPO object receives identical observations and nominal actions with the filter inactive.
- Legacy or differently scaled checkpoints are rejected by the new interface.
- A failed current-stage validation blocks validated-model promotion.

A simple PD **calibration controller**, separate from PPO, reached goals in all six axis directions and at [5,0,2]. With ACBF it reached the common target in each moving-obstacle case.

| Calibration controller + ACBF | Arrival time | Final error | Minimum centre distance |
| --- | --- | --- | --- |
| Case 1 | 7.60 s | 0.0858 m | 1.8276 m |
| Case 2 | 7.85 s | 0.0872 m | 1.0026 m |
| Case 3 | 7.80 s | 0.0850 m | 0.8417 m |

These flights validate controllability and filter integration. They are **not RL results**. All exceeded the 0.5 m safety distance and none had a filter failure. Two accepted control steps across these flights had an inaccurate solver status and passed the existing numerical checks. Some ACBF interventions took over 50 ms; real-time operation remains unestablished.

## Paper alignment and limits

The acceleration command is closer to Equations (3) and (8) than the previous waypoint command. However, the expanded observation, dense reward additions, curriculum, PID baseline gains, stopping criterion and obstacle-free navigation training are explicit adaptations. The existing corrected barrier geometry, adaptive coefficient and relaxed higher-order formulation also remain adaptations.

The saved policy interface does not promise compatibility across arbitrary changes to simulator or dependency versions. Run manifests record those versions for reproducibility.

## What a future run must demonstrate

1. Deterministic success on the current curriculum suite, including targets below the drone.
2. Success on the separate distant-goal suite, with low final speed.
3. Success on additional held-out random goals and multiple training seeds before claiming robust convergence.
4. Arrival and clearance together when the identical policy is evaluated through ACBF in all three scenarios.

A run's final checkpoint is not automatically its best checkpoint. Use validation evidence. Preserve failed runs as evidence rather than replacing their numbers with calibration-controller results.

See README.md for commands to run later. No training command was executed during this change.
