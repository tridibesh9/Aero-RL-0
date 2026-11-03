# BTP professor meeting brief

Prepared: 31 October 2026

**Deadline: 6 November 2026. Submission format: LaTeX. First meeting: 2 or 3 November. Proposed second review: 4 or 5 November, subject to the professor's availability.**

**Working title:** Simulation-Based Evaluation of UAV Navigation Using PPO, IMM Prediction and Adaptive Control Barrier Functions

**Purpose of this meeting:** Confirm the research question, expected contribution, and minimum experiments for the thesis.

## Opening statement

> My project studies how a drone can reach a target while avoiding moving obstacles. The current repository combines a PPO navigation policy, an IMM obstacle-motion predictor, and an adaptive control barrier function filter, following Gao and Sun's architecture. A trained model and preliminary outputs are available. With submission due on 6 November, I propose focusing the thesis on validating the implementation and comparing the three controllers across the three existing obstacle scenarios, with clear discussion of limitations. I am preparing the thesis in LaTeX and would like to confirm whether this scope meets the required contribution.

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
- A repository-review check loaded the model and ran the combined controller for ten simulation steps successfully.

**Limit:** A ten-step check establishes basic execution, not successful navigation, full reproduction, or a safety guarantee.

### Slide 4: Planned comparisons and preliminary evidence

Controllers:

1. PPO alone.
2. PID with IMM and ACBF.
3. PPO with IMM and ACBF.

Obstacle cases: constant velocity, constant acceleration, and figure-eight motion.

The existing saved summary contains the following **Case 1 preliminary results**:

| Controller | Minimum logged centre-to-centre distance | Logged collision |
|---|---:|---|
| PPO + IMM + ACBF | 0.952 m | No |
| PID + IMM + ACBF | 1.727 m | No |
| PPO alone | 0.353 m | Yes |

Source: `results/eval/summary_table.txt`.

These are existing saved outputs. They have not been reproduced in the current review, are not repeated-trial statistics, and do not establish goal-reaching success. The logger uses a 0.4 m distance threshold for its collision flag. Do not combine these numbers with paper results as if they were one local experiment.

**What to say:** "These outputs motivate further evaluation, but I will regenerate them with recorded settings and repeated trials before treating them as thesis results."

### Slide 5: Validation work and proposed contribution

Validation work:

- Assess and address the mismatch between the safety filter's assumed motion and limits and the command actually executed by the environment. Document unresolved limitations and avoid unsupported guarantees.
- Reconcile collision, clearance, and barrier-function thresholds.
- Record goal success, timeout, collisions, constraint violations, and solver failures explicitly.
- Regenerate all three cases using a reproducible experiment procedure.

Proposed research question:

> How does the implemented PPO-IMM-ACBF system compare with PPO alone and PID-IMM-ACBF in goal completion, obstacle separation, and collisions across three obstacle-motion scenarios?

Proposed contribution: a documented implementation assessment and reproducible comparative simulation study, including any verified corrections and their effects. This is a candidate BTP scope for supervisor approval; it is not a claim of a new algorithm. Keep measurement-noise robustness as future work unless explicitly required and feasible before the deadline.

### Slide 6: Next milestone and decisions requested

Next milestone:

- Reproducible baseline results for all three controllers and all three scenarios.
- One trajectory plot, one separation-over-time plot, and a table including goal-reaching success and collisions.
- A short list of implementation corrections and their effects.

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
- The preliminary Case 1 table, clearly labelled.
- A list of questions above and a place to record the decisions.

Use a simulation demonstration only after checking it in advance. Existing plots can be shown as preliminary artifacts; they are not a substitute for a documented experimental run.
