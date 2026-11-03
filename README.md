# Aero-RL · UAV navigation in simulation

This project compares a PPO policy, PPO with an adaptive control barrier function (ACBF), and PID with the same ACBF. Each drone must reach a goal while a spherical obstacle moves on a straight, accelerating, or figure-eight path. The controller architecture follows [Gao and Sun (2026)](https://doi.org/10.53941/am.2026.100003). The implementation and experiments used here are described in the [technical guide](CODEBASE_OVERVIEW.md).

## Evaluate an existing checkpoint

Run these commands from the project directory with the project's Python environment.

```powershell
python evaluate.py --model results/ppo_final.zip --case all
python train_ppo.py --validate-only results/ppo_final.zip
```

The first command runs the three moving-obstacle cases. The second checks four fixed navigation goals with the obstacle parked outside the flight volume. Both use deterministic inference. Each invocation creates a new run directory under `results/evaluations` or `results/validation`. Use `--output-dir` to supply a new destination.

Each benchmark run contains a model and source fingerprint, a configuration snapshot, a text and JSON summary, per-step CSVs, and plots. Read goal completion and stop reason alongside centre-to-centre distance. With the default settings, the geometric collision threshold is 0.40 m and the safety buffer begins below 0.50 m.

The reviewed final checkpoint records 2,015,232 prior training steps. On the four-goal validation suite it reached 0/4 goals. Use the [checked benchmark summary](results/evaluations/btp_review_20261103/summary_table.txt) and [goal validation record](results/validation/btp_review_20261103/validation.jsonl) for the local November 2026 experiments. Model archives and generated runs are stored outside version control.

## Training entry point

A future training run can be started with `python train_ppo.py --timesteps 1000000 --seed 42 --n_envs 4`. It creates a separate directory under `results/training`. Training starts with nearby goals and progresses to the full static-obstacle task. Checkpoint selection uses validation arrivals, final error, and elapsed time. A `validated_model.zip` requires arrivals at every validation goal. Its scope is goal tracking on that suite.

## Verification

```powershell
python -B -m unittest discover -s tests -v
```

The regression suite covers the motion forecasts, barrier equations, failure handling, episode resets, metrics, and checkpoint selection. Runtime dependencies include NumPy, PyYAML, PyTorch, Stable-Baselines3, Gymnasium, PyBullet, gym-pybullet-drones, CVXPY with Clarabel and SCS, and Matplotlib. Exact package versions are recorded in each run manifest.
