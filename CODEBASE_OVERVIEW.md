# UAV navigation implementation and experiment guide

The project evaluates one Crazyflie CF2X travelling toward a goal in a PyBullet simulation with a moving spherical obstacle. The source method is described in [Gao and Sun, Advanced Mechatronics (2026)](https://doi.org/10.53941/am.2026.100003). This guide describes the local implementation. Paper Table 3 includes PPO policies trained for individual cases. The local comparison uses one saved PPO checkpoint in all three cases.

## Control loop

| Controller | Goal guidance | Obstacle handling |
|---|---|---|
| Pure RL | Deterministic PPO prediction | The policy observes drone velocity and goal error only |
| RL + ACBF | The same PPO checkpoint | IMM prediction followed by the ACBF filter |
| PID + ACBF | PID applied to position error | The same IMM and ACBF as RL + ACBF |

The environment passes a six-component observation to PPO, with velocity first and goal position minus drone position second. For all three controllers the applied command is bounded by the intersection of the configured actuator limits and the policy range. Each command component moves the PID target relative to the current position. The offset in metres is the command multiplied componentwise by \([0.5,0.5,0.15]\). The motor PID produces four rotor speeds. The CSVs record the applied command and measured position and velocity.

Each reset clears the motor controller's integral and attitude history. Random episodes use Gymnasium's seeded generator. Fixed scenarios pass the start, goal, initial obstacle position and obstacle motion into the environment. Mass and gravity come from the CF2X simulator. Control updates occur at 20 Hz and physics updates at 240 Hz with the supplied settings.

## Obstacle and drone forecasts

The IMM combines constant velocity (6 states), constant acceleration (9 states), and coordinated turn (7 states). It mixes the estimates in a common position, velocity and acceleration space. The turning state maps horizontal acceleration to \((-\omega v_y,\omega v_x)\). Covariance propagation uses the Jacobian of this map at the current estimate. The reverse map selects the position and velocity components and uses a turn rate estimated from heading history. The Kalman updates use the Joseph covariance form. Likelihood smoothing and model probability updates use log likelihoods.

For each of the next five control steps the IMM propagates each model, then averages its predicted position using the current model probabilities. The moving-obstacle benchmark supplies exact simulated position observations. Its filter noise matrices describe the estimator assumptions.

The flight evaluator supplies a separate DIRECT PyBullet world for predicting the drone. A candidate rollout starts at the live position, attitude and velocities with a copy of the motor PID state. It holds the candidate command for the prediction horizon using the same waypoint mapping, rotor controller and physics. The forecast world includes the ground and models free flight relative to the obstacle. The predicted obstacle enters through the IMM. Forecasts are evaluated independently of the live world's clock and state. Standalone point-mass studies can call the filter with the analytical acceleration rollout. Code controlling a UAVNavEnv should pass its predict_positions method to act_normalized, as the evaluator does.

## Adaptive barrier calculation

Let \(d_s=r_{\rm drone}+r_{\rm obs}+s_{\rm cushion}\). At prediction index \(i\),

\[
 h_i=\tfrac12(\|p_i-p_{o,i}\|^2-d_s^2),\qquad
 \psi_i=h_{i+1}-(1-\gamma_1)h_i.
\]

The boundary \(h=0\) lies at 0.50 m with the supplied radii and cushion. The geometric collision proxy uses the sum of radii, 0.40 m. Both distances refer to the two centres.

With \(p_r=p_o-p\) and \(v_r=v_o-v\), decreasing range gives \(p_r^\top v_r<0\). The local controller uses a fractional closing rate \(q=\max(0,-p_r^\top v_r/\|p_r\|^2)\) and a bounded reference \(\gamma^*=\operatorname{clip}(1-Tq,0,1)\), where \(T\) is the configured approach time reserve. For positive \(\psi_i\), a smaller \(\gamma\) raises the required next \(\psi\). This adaptation and the simulator rollout are implementation choices in this project.

For a candidate command and coefficient, the complete residual contains the predicted clearances \(h_1,\ldots,h_N\) and the second-order terms

\[
 F_i=\psi_{i+1}-(1-\gamma_2\gamma)\psi_i,\qquad i=0,\ldots,N-2.
\]

The objective penalizes deviation from the reference command, deviation from \(\gamma^*\), second-order relaxation \(\xi\), and Lyapunov relaxation \(\delta\). Future clearance is a hard constraint in the subproblem. The second-order condition allows \(F_i+\xi_i\ge0\). The Lyapunov condition uses \((\gamma-\gamma^*)^2\le\delta+(1-\alpha)V_{\rm prev}\).

A nominal command satisfying every residual at the target coefficient attains zero objective and can be used directly. Otherwise the filter linearizes the entire residual with respect to all three command components and the coefficient. Command derivatives use bounded finite differences of the plant rollout. Clarabel or SCS solves the resulting convex subproblem. Candidate solutions undergo finite-value, bound, Lyapunov, and linear constraint checks, followed by a fresh nonlinear trajectory check. If needed, the filter tightens the next subproblem and shrinks the search region. Exhausting the checks ends the episode with a filter failure before applying the proposed command.

These conditions apply to sampled, predicted positions subject to numerical tolerance. Tracking error, obstacle-estimation error, and the finite prediction horizon limit the conclusions that can be drawn about actual flight.

## Goal tracking and model selection

The reward includes goal arrival, reduction in distance to goal, a boundary penalty and a time penalty. Arrival requires distance strictly below 0.10 m. Timeouts occur at 500 control steps by default.

Future training uses a configurable curriculum. The first two stages sample goals in multiple directions up to 1 m and 2.5 m from the start. They park the obstacle outside the flight volume. The final stage uses the full random goal distribution and a static obstacle near the route. Advancement requires the configured arrival fraction across a full window of completed episodes from the current stage. The default is 80 percent across 20 episodes. The saved historical checkpoint predates this procedure.

The validation suite uses four specified goals with fixed seeds and a distant obstacle. Validation runs deterministic inference. Checkpoints rank first by arrivals, then by lower mean final distance and lower mean elapsed time. The best candidate can have failed missions. A separately labelled validated checkpoint requires success on the entire suite. Training and validation each allocate fresh output directories.

## Scenarios and metrics

| Case | Obstacle path | Initial velocity (m/s) |
|---|---|---|
| 1 | Constant velocity along the y axis | (0, 0.8, 0) |
| 2 | Constant y acceleration of 0.4 m/s² | (0, 0.5, 0) |
| 3 | Horizontal figure-eight at z = 1.5 m | (0.9, 0.9, 0) |

All three current cases start at (0, 0, 1) m and target (5, 0, 2) m. Read each case as a single deterministic flight per controller.

- Raw CSVs include the initial and terminal samples, measured state, obstacle position, reference and applied commands, instantaneous separation, surface clearance, geometric collision, and buffer entry. Command and diagnostic values in a post-step row describe the interval ending at that row. The initial row has a zero placeholder command.
- The summary distinguishes arrival, timeout, leaving the flight area, an evaluator step cap, and a filter failure. A failed solver attempt is recorded separately because no command from that attempt was applied.
- Filter records include solver attempts and status, accepted slack, the verified nonlinear residual, minimum predicted separation, predicted next position and computation time. Simulation elapsed time and controller computation time have different meanings.
- Each run saves a parameter snapshot, command settings, package versions, and SHA-256 fingerprints for project sources and the selected model. Old benchmark files stay in their original directory.

## Files and local evidence

| Path | Role |
|---|---|
| [Environment](envs/UAVNavEnv.py) | Episode state, goal sampling, waypoint actuation, free-flight prediction |
| [Training and goal validation](train_ppo.py) | PPO training, curriculum stages and checkpoint selection |
| [Controller composition](controllers/hierarchical_ctrl.py) | PPO and PID navigation references |
| [IMM](controllers/imm_predictor.py) | Obstacle motion estimation and short-horizon forecasts |
| [ACBF](controllers/acbf_filter.py) | Bounded optimization and nonlinear verification |
| [Evaluation](evaluate.py) and [logger](utils/logger.py) | Matched scenario replays, raw evidence and figures |
| [Configuration](configs/params.yaml) and [loader](utils/config.py) | Validated settings and run metadata |
| [Numerical checks](tests/test_navigation.py) | Regression coverage with no policy optimization |

The model at results/ppo_final.zip records 2,015,232 timesteps and eight training environments. The fixed-goal validation record and nine-flight benchmark below were generated using those existing weights. The revised training procedure has yet to produce a new trained model. Use the local run data for reported distances and mission outcomes.

[Goal validation](results/validation/btp_review_20261103/validation.jsonl) · [Benchmark table](results/evaluations/btp_review_20261103/summary_table.txt) · [Diagnostics by flight](results/evaluations/btp_review_20261103/summary.json) · [Original paper](2607004775.pdf)
