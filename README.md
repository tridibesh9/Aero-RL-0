# UAV navigation with PPO, IMM and ACBF

The current configuration uses a new acceleration interface intended to address the failed goal tracking of the historical waypoint policies. **No new policy has been trained or demonstrated to converge with this interface.** The physics and learning contracts have been checked without learning. Read [the changes and verification record](TRAINING_FIXES.md).

PPO supplies navigation commands. The same checkpoint can be evaluated alone or with the IMM/ACBF safety filter. A separate PID reference is a control baseline. The project follows the architecture of [Gao and Sun (2026)](https://doi.org/10.53941/am.2026.100003), with the documented adaptations in [the technical guide](CODEBASE_OVERVIEW.md).

## Training later

Run from this directory using the project Python environment. The following starts training only when you choose to run it.

```powershell
python -B train_ppo.py --timesteps 200000 --n_envs 1 --seed 42
```

This is a budget, not a convergence guarantee. New runs use configs/params.yaml, fresh weights, and fresh dated directories under results/training. Do not resume an old six-input waypoint model with the new 19-input acceleration environment.

Training begins with goals 0.2–0.4 m away. The curriculum advances through radii 1 m, 2.5 m, and the full goal distribution. Navigation training keeps obstacles distant. The same PPO is later tested with the safety filter around moving obstacles. Goal success requires distance below 0.1 m and speed at most 0.2 m/s for five consecutive control steps.

Read validation.jsonl, episodes.jsonl, env_0.monitor.csv and progress.csv in the run directory. A higher return alone is insufficient. best_candidate.zip can still fail. validated_model.zip is created only when the fixed goal suite and the current-stage suite both pass. This is a validation gate, not proof of general convergence.

## Evaluation

Specify the exact checkpoint. The examples below are paths to replace with your actual new run.

```powershell
python -B train_ppo.py --validate-only results/training/YOUR_RUN/validated_model.zip
python -B evaluate.py --model results/training/YOUR_RUN/validated_model.zip --case all
```

Evaluation uses the params.yaml saved with that checkpoint and checks its interface metadata. Results go into a new directory under results/evaluations. The manifests record exact model hashes, source hashes and effective parameters.

To inspect the old waypoint policy, select its explicit legacy configuration.

```powershell
python -B evaluate.py --model results/ppo_final.zip --config configs/params_legacy.yaml --case all
```

Historical results are preserved. The previous 2,015,232-step policy and the later 51,200-step waypoint policy failed goal validation. They do not measure the new acceleration implementation.

## Code-only verification

```powershell
python -B -m unittest discover -s tests -v
```

Tests cover physics calibration, crash versus timeout handling, stopping and holding at a goal, observation consistency, checkpoint compatibility, forecasts, barrier constraints and model selection. The new learning-setup tests forbid PPO learning and Adam optimizer updates.

Dependencies include NumPy, SciPy, PyYAML, PyTorch, Stable-Baselines3, Gymnasium, PyBullet, gym-pybullet-drones, CVXPY with Clarabel/SCS, and Matplotlib. Each run records installed versions. The numerical safety filter can exceed a 50 ms computation budget during intervention, so these checks do not establish real-time deployment performance.
