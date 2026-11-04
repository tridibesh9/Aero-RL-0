> 4 November update: the acceleration-based learning setup has been revised without training. Historical results below describe the older waypoint policies. See [TRAINING_FIXES.md](TRAINING_FIXES.md) for the new implementation and its limits.

# BTP professor meeting brief

Updated 3 November 2026 after the corrected benchmark replay.

**Deadline: 6 November 2026. Submission format: LaTeX. First meeting: 2 or 3 November. Proposed second review: 4 or 5 November, subject to the professor's availability.**

**Working title:** Simulation-Based Evaluation of UAV Navigation Using PPO, IMM Prediction and Adaptive Control Barrier Functions

**Purpose of this meeting:** Confirm the research question, expected contribution, and minimum experiments for the thesis.

## Opening statement

> My project studies how a drone can reach a target while avoiding moving obstacles. I am evaluating Gao and Sun's architecture in simulation. With the existing PPO checkpoint, adding the filter increased minimum separation in the linear case from 0.353 m to 0.620 m. Both PPO flights timed out. PID with the same filter reached the goal in all three cases. I would like to discuss how to present the implementation corrections, these results and their limitations in the thesis.

This statement describes the current repository and the proposed work. Adapt any statement about personal implementation or completed work to what you actually did.

## Six-slide outline for a 5-7 minute update

### Slide 1: Problem and objective

- A drone must reach a goal while avoiding a moving obstacle.
- A learned navigation policy alone may collide when obstacle motion changes.
- Objective: evaluate whether adding prediction and a safety filter improves collision avoidance while retaining useful goal-reaching performance.
- Current scope: one drone, one obstacle, simulation in PyBullet.

**What to say:** "I am evaluating navigation performance and obstacle avoidance together. A controller that avoids the obstacle but never reaches the goal is not a successful navigation solution."

### Slide 2: Base paper and system architecture

Base paper: Jian Gao and Jiankun Sun, *Hierarchical Safe Reinforcement Learning Control for UAVs via Adaptive Control Barrier Function and Interacting Multiple Model*, Advanced Mechatronics, 2026.

Official article: https://www.sciltp.com/journals/am/articles/2607004775

DOI: https://doi.org/10.53941/am.2026.100003

```text
Drone velocity + goal error -> PPO -> reference command -----+
                                                           |
Obstacle positions -> IMM -> predicted obstacle positions --+-> ACBF
                                                                |
                                                        adjusted command
                                                                |
                                                        PID -> simulation
```

- **PPO:** Proximal Policy Optimization; the learned navigation policy.
- **IMM:** Interacting Multiple Model estimator; combines straight, accelerating, and turning motion models.
- **ACBF:** Adaptive Control Barrier Function; adjusts commands using predicted separation and constraints.

**What to say:** "The navigation policy proposes a movement. The predictor estimates where the obstacle will be. The safety filter adjusts the proposed movement before the drone executes it."

### Slide 3: Existing implementation and checked status

- A custom Gymnasium environment simulates a Crazyflie drone in PyBullet.
- PPO receives drone velocity and goal error; it does not receive obstacle positions.
- Training uses static obstacles without the safety filter.
- Evaluation adds IMM and ACBF without retraining the policy.
- The configured controller period is 0.05 seconds; prediction covers five steps, or 0.25 seconds.
- The saved final PPO model records 2,015,232 timesteps.
- The corrected evaluator replayed nine flights and recorded start and terminal state, commands, obstacle distances, solver diagnostics and stop reasons.

The existing checkpoint also timed out on all four fixed validation goals with a distant obstacle. No new PPO training was performed during these corrections.

### Slide 4 · Measured comparisons

Controllers:

1. PPO alone.
2. PID with IMM and ACBF.
3. PPO with IMM and ACBF.

Obstacle cases: constant velocity, constant acceleration, and figure-eight motion.

Each cell gives the stop reason and the minimum sampled centre distance in metres. Each case is one deterministic flight per controller.

| Controller | Linear | Accelerating | Figure-eight |
|---|---|---|---|
| PPO + IMM + ACBF | Timeout, 0.620 | Timeout, 0.960 | Timeout, 2.361 |
| PID + IMM + ACBF | Goal, 1.561 | Goal, 0.842 | Goal, 0.722 |
| PPO alone | Timeout, 0.353 | Timeout, 0.909 | Timeout, 2.361 |

The collision threshold is 0.40 m and the safety-buffer boundary is 0.50 m. [Checked summary](results/evaluations/btp_review_20261103/summary_table.txt)

Use the raw CSVs to distinguish ground-truth measurements from predicted separation. The paper's separately trained Pure RL baselines answer a different experimental question.

### Slide 5: Validation work and proposed contribution

Completed implementation checks

- The filter forecasts the waypoint command using a copy of the simulated drone and its motor controller. It solves within the same bounds as the environment.
- Collision geometry, safety margin and the barrier zero level now agree with their respective configured radii.
- Every issued filtered command is checked against its predicted trajectory. The logs report relaxations, solver status, and time spent computing each command.
- The corrected run used one saved model and the same initial conditions for each comparison.

Proposed research question:

> How does the implemented PPO-IMM-ACBF system compare with PPO alone and PID-IMM-ACBF in goal completion, obstacle separation, and collisions across three obstacle-motion scenarios?

Proposed contribution: a documented implementation assessment and reproducible comparative simulation study, including any verified corrections and their effects. This is a candidate BTP scope for supervisor approval; it is not a claim of a new algorithm. Keep measurement-noise robustness as future work unless explicitly required and feasible before the deadline.

### Slide 6: Next milestone and decisions requested

Next milestone:

- Incorporate the verified comparison, plots and limitations in the LaTeX thesis.
- Discuss the old policy's goal-tracking failure and the revised checkpoint acceptance procedure.
- Agree with the supervisor on any further trials needed for submission.

Ask the professor:

1. Is the proposed implementation and comparative evaluation sufficient for this BTP, or is an additional contribution required before 6 November?
2. Is simulation-only evaluation acceptable?
3. Which corrections and experiments are essential for submission, and which should be documented as limitations or future work?
4. Which LaTeX template, front matter, and signatures are required, and what is the exact submission cutoff on 6 November?
5. Can we review the complete draft on 4 November, or on 5 November if necessary?

## Bring to the meeting

- This brief or six slides based on it.
- A compiling LaTeX draft containing the problem, objectives, architecture, and methodology; include results only where supported by actual recorded runs.
- The base paper.
- The checked nine-flight table and Case 3 trajectory and distance plots from the same run.
- A list of questions above and a place to record the decisions.

Use a simulation demonstration only after checking it in advance. Existing plots can be shown as preliminary artifacts; they are not a substitute for a documented experimental run.
