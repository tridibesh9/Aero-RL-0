"""
evaluate.py
────────────
Evaluate all three controllers (RL+ACBF, PID+ACBF, Pure RL) across all three
scenario cases and reproduce Table 3 + Figures 5-13 from the paper.

Usage
─────
    python evaluate.py --model results/ppo_final.zip [--case 1|2|3|all] [--gui]

Output
──────
    results/eval/case{N}/3d_traj.png
    results/eval/case{N}/min_dist.png
    results/eval/case{N}/{method}_states.png
    results/eval/summary_table.txt
"""

import argparse
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(__file__))

from envs.UAVNavEnv          import UAVNavEnv
from controllers.hierarchical_ctrl import HierarchicalController, PIDWithACBF
from utils.logger            import (EpisodeLogger, plot_3d_trajectories,
                                     plot_min_distance, plot_states,
                                     print_summary_table)
from utils.math_utils        import safety_distance

import scenarios.case1_linear  as case1
import scenarios.case2_accel   as case2
import scenarios.case3_figure8 as case3

SCENARIOS = {1: case1, 2: case2, 3: case3}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate hierarchical safe RL controller")
    p.add_argument("--model",  default="results/ppo_final.zip",
                   help="Path to trained PPO model .zip")
    p.add_argument("--config", default="configs/params.yaml")
    p.add_argument("--case",   default="all",
                   help="Which case(s) to run: 1, 2, 3, or all")
    p.add_argument("--gui",    action="store_true",
                   help="Open PyBullet GUI during evaluation")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Override max episode steps")
    return p.parse_args()


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(env: UAVNavEnv,
                controller,        # HierarchicalController | PIDWithACBF | None
                scenario_mod,      # case1/2/3 module
                label: str,
                ds: float,
                max_steps: int) -> EpisodeLogger:
    """
    Run one evaluation episode and collect metrics.

    For 'Pure RL', controller=None and we use the raw PPO inside a separately
    loaded model passed as `ppo_model` attribute on the env.
    """
    scfg = scenario_mod.SCENARIO_CONFIG
    goal = scfg['goal_pos']

    obs, info = env.reset()
    p_uav, v_uav = env.get_uav_state()

    if hasattr(controller, 'reset'):
        controller.reset(scfg['obs_init_pos'])

    log = EpisodeLogger(label)
    t   = 0.0
    dt  = env.CTRL_TIMESTEP

    for step in range(max_steps):
        p_uav, v_uav = env.get_uav_state()
        z_obs        = env.get_obs_pos()

        if controller is None:
            # Pure RL: use PPO directly (no safety filter)
            action_norm, _ = env._pure_ppo.predict(obs, deterministic=True)
            u_logged = env.cfg['k_ma'] * action_norm.flatten()
            obs_traj_log = np.tile(z_obs, (env.cfg['N_pred']+1, 1))
        else:
            action_norm, obs_traj_log = controller.act_normalized(
                p_uav, v_uav, goal, z_obs)
            u_logged = env.cfg['k_ma'] * action_norm.flatten()

        log.log(t, p_uav, v_uav, u_logged, z_obs)

        obs, reward, terminated, truncated, info = env.step(action_norm)
        t += dt

        if terminated or truncated:
            break

    return log


# ── Per-case evaluation ────────────────────────────────────────────────────────

def evaluate_case(case_num: int, cfg: dict, args, summary: dict):
    scn = SCENARIOS[case_num]
    scfg = scn.SCENARIO_CONFIG
    print(f"\n{'='*60}")
    print(f"  {scfg['name']}")
    print('='*60)

    ds       = safety_distance(cfg['drone_radius'], cfg['obs_radius'], cfg['safety_cushion'])
    max_steps= args.max_steps or cfg['max_episode_steps']
    out_dir  = f"results/eval/case{case_num}"
    os.makedirs(out_dir, exist_ok=True)

    def make_env(gui=False):
        return UAVNavEnv(
            config=cfg,
            obstacle_policy=scn.obstacle_policy,
            target_pos=scfg['goal_pos'],
            obstacle_init_pos=scfg['obs_init_pos'],
            gui=gui,
        )

    logs = []

    # ── 1. RL + ACBF ──────────────────────────────────────────────────────────
    print("\n[Case %d] Running RL+ACBF ..." % case_num)
    env = make_env(gui=args.gui)
    ctrl = HierarchicalController(args.model, cfg)
    t0 = time.time()
    log_rl_acbf = run_episode(env, ctrl, scn, "RL+ACBF", ds, max_steps)
    train_s = 0  # training time logged separately by train_ppo.py
    env.close()
    logs.append(log_rl_acbf)
    print(f"  min_dist={log_rl_acbf.min_distance():.3f}m  "
          f"collision={'YES' if log_rl_acbf.collision else 'no'}  "
          f"t={log_rl_acbf.mission_time():.1f}s")

    # ── 2. PID + ACBF ─────────────────────────────────────────────────────────
    print("[Case %d] Running PID+ACBF ..." % case_num)
    env = make_env()
    ctrl_pid = PIDWithACBF(cfg)
    log_pid_acbf = run_episode(env, ctrl_pid, scn, "PID+ACBF", ds, max_steps)
    env.close()
    logs.append(log_pid_acbf)
    print(f"  min_dist={log_pid_acbf.min_distance():.3f}m  "
          f"collision={'YES' if log_pid_acbf.collision else 'no'}  "
          f"t={log_pid_acbf.mission_time():.1f}s")

    # ── 3. Pure RL (no safety filter) ─────────────────────────────────────────
    print("[Case %d] Running Pure RL ..." % case_num)
    from stable_baselines3 import PPO as _PPO
    ppo_model = _PPO.load(args.model, device="cpu")
    k_ma = cfg['k_ma']

    env_pure = make_env()
    obs, _ = env_pure.reset()
    # Attach PPO so run_episode can reach it
    env_pure._pure_ppo = ppo_model
    log_pure_rl = run_episode(env_pure, None, scn, "Pure RL", ds, max_steps)
    env_pure.close()
    logs.append(log_pure_rl)
    print(f"  min_dist={log_pure_rl.min_distance():.3f}m  "
          f"collision={'YES' if log_pure_rl.collision else 'no'}  "
          f"t={log_pure_rl.mission_time():.1f}s")

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_3d_trajectories(logs, scfg['goal_pos'], scfg['start_pos'],
                         ds, f"{out_dir}/3d_traj.png")
    plot_min_distance(logs, ds, f"{out_dir}/min_dist.png")
    for lg in logs:
        plot_states(lg, f"{out_dir}/{lg.label.replace('+','_').replace(' ','_')}_states.png")

    # ── Summary dict ──────────────────────────────────────────────────────────
    for lg in logs:
        key = f"Case{case_num} {lg.label}"
        summary[key] = {
            'min_dist':  lg.min_distance(),
            'collision': lg.collision,
            'mission_t': lg.mission_time(),
        }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs("results/eval", exist_ok=True)

    cases = [1, 2, 3] if args.case == "all" else [int(args.case)]
    summary = {}

    for c in cases:
        evaluate_case(c, cfg, args, summary)

    print_summary_table(summary)

    # Save summary to file
    with open("results/eval/summary_table.txt", "w") as f:
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_summary_table(summary)
        f.write(buf.getvalue())

    print("\n[Eval] All results saved to results/eval/")


if __name__ == "__main__":
    main()
