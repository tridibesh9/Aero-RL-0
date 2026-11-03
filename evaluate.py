"""Evaluate the three navigation controllers on the three moving-obstacle cases."""
import argparse
import contextlib
import io
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from stable_baselines3 import PPO

from controllers.acbf_filter import FilterFailure
from controllers.hierarchical_ctrl import HierarchicalController, PIDWithACBF
from envs.UAVNavEnv import UAVNavEnv
from scenarios import case1_linear, case2_accel, case3_figure8
from utils.config import load_config, new_run_dir, write_run_manifest
from utils.logger import (EpisodeLogger, plot_3d_trajectories, plot_min_distance,
                          plot_states, print_summary_table)

SCENARIOS = {1: case1_linear, 2: case2_accel, 3: case3_figure8}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='results/ppo_final.zip')
    parser.add_argument('--config', default='configs/params.yaml')
    parser.add_argument('--case', choices=('1', '2', '3', 'all'), default='all')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--max_steps', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', help='New directory. Defaults to a dated run in results/evaluations.')
    return parser.parse_args()


def run_episode(env, controller, scenario_mod, label, ds, max_steps, *, ppo=None, seed=42):
    if max_steps < 1:
        raise ValueError('max_steps must be positive')
    obs, info = env.reset(seed=seed)
    if controller is not None:
        controller.reset(info['obs_pos'])
    elif ppo is None:
        ppo = env._pure_ppo  # compatibility with previous run_episode callers

    goal = scenario_mod.SCENARIO_CONFIG['goal_pos']
    log = EpisodeLogger(label, env.cfg)
    if not np.isclose(ds, log.safety_distance):
        raise ValueError('Logger and controller must use the same safety distance')
    log.log(0.0, info['pos'], info['vel'], info['command'], info['obs_pos'])
    for _ in range(max_steps):
        start = perf_counter()
        diagnostics = None
        if controller is None:
            action, _ = ppo.predict(obs, deterministic=True)
        else:
            try:
                action, _ = controller.act_normalized(info['pos'], info['vel'], goal, info['obs_pos'],
                                                      predict_positions=env.predict_positions)
            except FilterFailure as err:
                log.stop_reason = 'filter_failure'
                log.failed_attempt = {k: v for k, v in controller.acbf.diagnostics.items() if k != 'reference'}
                log.failed_attempt.update(reason=str(err), time_s=info['t'],
                                          control_ms=1000 * (perf_counter() - start))
                break
            diagnostics = controller.acbf.diagnostics
        compute_ms = 1000 * (perf_counter() - start)
        obs, _, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float32).reshape(1, 3))
        log.log(info['t'], info['pos'], info['vel'], info['command'], info['obs_pos'], diagnostics, compute_ms)
        if terminated or truncated:
            log.reached_goal = info['is_success']
            log.stop_reason = info['termination_reason']
            break
    return log


def evaluate_case(case_num, cfg, args, summary, model=None):
    scenario = SCENARIOS[case_num]
    scfg = scenario.SCENARIO_CONFIG
    ds = cfg['drone_radius'] + cfg['obs_radius'] + cfg['safety_cushion']
    steps = args.max_steps if args.max_steps is not None else cfg['max_episode_steps']
    if model is None:
        model = PPO.load(args.model, device='cpu')
    output = Path(args.output_dir) / f'case{case_num}'
    output.mkdir()
    logs = []
    for label, controller in [('RL+ACBF', HierarchicalController(model, cfg)),
                              ('PID+ACBF', PIDWithACBF(cfg)), ('Pure RL', None)]:
        env = UAVNavEnv(cfg, obstacle_policy=scenario.obstacle_policy, target_pos=scfg['goal_pos'],
                        obstacle_init_pos=scfg['obs_init_pos'], start_pos=scfg['start_pos'],
                        gui=args.gui and label == 'RL+ACBF')
        try:
            log = run_episode(env, controller, scenario, label, ds, steps, ppo=model, seed=args.seed)
        finally:
            env.close()
        logs.append(log)
        stem = label.replace('+', '_').replace(' ', '_')
        log.write_csv(output / f'{stem}.csv', scfg['goal_pos'])
        summary[f'Case{case_num} {label}'] = log.summary(scfg['goal_pos'])
        print(f"{scfg['name']} / {label}: {log.stop_reason}, {log.mission_time():.2f} s", flush=True)
        plot_states(log, output / f'{stem}_states.png')
    plot_3d_trajectories(logs, scfg['goal_pos'], scfg['start_pos'], ds, output / '3d_traj.png')
    plot_min_distance(logs, ds, output / 'min_dist.png')


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError('--max_steps must be positive')
    model = PPO.load(args.model, device='cpu')
    args.output_dir = new_run_dir(args.output_dir, 'results/evaluations')
    cases = list(SCENARIOS) if args.case == 'all' else [int(args.case)]
    write_run_manifest(args.output_dir, cfg, args.model, cases=cases, seed=args.seed,
                       max_steps=args.max_steps or cfg['max_episode_steps'], model_timesteps=model.num_timesteps,
                       uav_forecast='PyBullet CF2X waypoint/PID free flight')
    summary = {}
    for case_num in cases:
        evaluate_case(case_num, cfg, args, summary, model)
        (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    table = io.StringIO()
    with contextlib.redirect_stdout(table):
        print_summary_table(summary)
    print(table.getvalue())
    (args.output_dir / 'summary_table.txt').write_text(f"Model: {args.model}\n" + table.getvalue(), encoding='utf-8')
    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
