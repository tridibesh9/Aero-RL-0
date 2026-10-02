"""
train_ppo.py
─────────────
Train a PPO agent for UAV goal-reaching using Stable-Baselines3.

Key design choices (matching the paper):
  • Agent is trained on STATIC obstacles only — the ACBF handles dynamic
    scenarios at inference time (zero-shot generalisation).
  • Obs: s(k) = [v(k)ᵀ, e(k)ᵀ] ∈ ℝ⁶
  • Act: normalised acceleration a ∈ [-1,1]³
  • Reward: composite r_tr + r_ds + r_bp + r_tp  (Eqs. 9-13)

Usage
─────
    python train_ppo.py [--timesteps 1000000] [--seed 42] [--gui]

Output
──────
    results/ppo_model/          SB3 model checkpoints
    results/ppo_final.zip       Final model
    results/training_log.csv    Reward vs timesteps
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
import torch

# ── Path setup ─────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from envs.UAVNavEnv import UAVNavEnv


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train PPO for UAV navigation")
    p.add_argument("--config",     default="configs/params.yaml")
    p.add_argument("--timesteps",  type=int,   default=None,
                   help="Override total timesteps from config")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--n_envs",     type=int,   default=4,
                   help="Parallel training environments")
    p.add_argument("--gui",        action="store_true",
                   help="Open PyBullet GUI (single env, slow)")
    p.add_argument("--resume",     default=None,
                   help="Path to existing .zip to resume training")
    return p.parse_args()


# ── Callbacks ──────────────────────────────────────────────────────────────────

class TimingCallback(BaseCallback):
    """Prints estimated time remaining every N steps."""
    def __init__(self, total_steps: int, log_interval: int = 10_000):
        super().__init__()
        self.total   = total_steps
        self.log_int = log_interval
        self.t0      = None

    def _on_training_start(self):
        self.t0 = time.time()

    def _on_step(self) -> bool:
        n = self.num_timesteps
        if n % self.log_int == 0:
            elapsed = time.time() - self.t0
            frac    = n / self.total
            eta     = elapsed / (frac + 1e-9) * (1 - frac)
            print(f"  [Train] step {n:>8}/{self.total}  "
                  f"({100*frac:.1f}%)  ETA {eta/60:.1f} min")
        return True


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    total_steps = args.timesteps or cfg['ppo_total_timesteps']
    os.makedirs("results/ppo_model", exist_ok=True)

    print("=" * 60)
    print(" UAV PPO Training")
    print(f"  Config  : {args.config}")
    print(f"  Steps   : {total_steps:,}")
    print(f"  N envs  : {args.n_envs}")
    print(f"  Seed    : {args.seed}")
    print(f"  GPU     : {torch.cuda.is_available()}")
    print("=" * 60)

    # ── Environment factory ────────────────────────────────────────────────────
    # Static obstacle (policy=None) during training.
    # Use DummyVecEnv with explicit factory closures — avoids SB3's make_vec_env
    # misinterpreting our two-level factory.
    from stable_baselines3.common.vec_env import DummyVecEnv

    def make_env_fn(rank: int):
        """Returns a zero-argument callable that creates one env instance."""
        def _init():
            env = UAVNavEnv(
                config=cfg,
                obstacle_policy=None,   # STATIC for training
                gui=(args.gui and rank == 0),
            )
            return Monitor(env)
        return _init

    n_envs = 1 if args.gui else args.n_envs
    train_env = DummyVecEnv([make_env_fn(i) for i in range(n_envs)])

    # Seed the vec env
    train_env.seed(args.seed)

    # Separate eval env (no GUI, deterministic)
    eval_env = Monitor(UAVNavEnv(config=cfg, obstacle_policy=None, gui=False))

    # ── Model ──────────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )

    if args.resume:
        print(f"[Train] Resuming from {args.resume}")
        model = PPO.load(args.resume, env=train_env, verbose=1)
        model.set_env(train_env)
    else:
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=cfg['ppo_learning_rate'],
            n_steps=cfg['ppo_n_steps'],
            batch_size=cfg['ppo_batch_size'],
            n_epochs=cfg['ppo_n_epochs'],
            gamma=cfg['ppo_gamma'],
            clip_range=cfg['ppo_clip_range'],
            ent_coef=cfg['ppo_ent_coef'],
            policy_kwargs=policy_kwargs,
            verbose=1,
            seed=args.seed,
            device="cpu",   # MLP policies train faster on CPU (SB3 recommendation)
        )

    # ── Callbacks ──────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path="results/ppo_model/",
        name_prefix="ppo_uav",
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="results/ppo_model/best/",
        log_path="results/ppo_model/eval_logs/",
        eval_freq=20_000,
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )
    timing_cb = TimingCallback(total_steps)

    # ── Train ──────────────────────────────────────────────────────────────────
    t_start = time.time()
    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_cb, eval_cb, timing_cb],
        progress_bar=False,
        reset_num_timesteps=not args.resume,
    )
    elapsed = time.time() - t_start

    # ── Save ───────────────────────────────────────────────────────────────────
    model.save("results/ppo_final")
    print(f"\n[Train] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("[Train] Model saved → results/ppo_final.zip")
    print("[Train] Best model  → results/ppo_model/best/best_model.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
