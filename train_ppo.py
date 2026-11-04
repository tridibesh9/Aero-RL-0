"""Train PPO for navigation or validate a saved checkpoint using only inference."""
import argparse
from collections import deque
from contextlib import ExitStack, closing
import json

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.UAVNavEnv import UAVNavEnv
from utils.config import load_config, new_run_dir, write_run_manifest
from utils.policy import check_model_interface, config_for_model, interface_spec


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='New runs use configs/params.yaml; saved models use their own snapshot')
    parser.add_argument('--timesteps', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_envs', type=int, default=4)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--torch-threads', type=int, default=1)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resume', help='Continue a saved PPO in a new output directory')
    mode.add_argument('--validate-only', metavar='MODEL', help='Evaluate the fixed goal suite without learning')
    parser.add_argument('--output-dir', help='New directory for this run')
    return parser.parse_args()


DEFAULT_GOALS = [[-3., 2., 1.8], [1., -4., 1.2], [2., 2., 2.5], [-2., -2., 0.7]]


def validate_goals(model, env, goals, seed):
    """Measure deterministic goal arrivals on a fixed obstacle-distant suite."""
    episodes = []
    for index, goal in enumerate(goals):
        obs, info = env.reset(seed=seed + index, options={'target_pos': goal})
        for _ in range(env.cfg['max_episode_steps']):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        episodes.append(dict(goal=list(goal), seed=seed + index, reached=info['is_success'],
                             final_error_m=info['goal_error'], elapsed_s=info['t'],
                             reason=info['termination_reason'], final_speed_m_s=info['goal_speed']))
    return dict(timesteps=int(model.num_timesteps), goals_reached=sum(ep['reached'] for ep in episodes),
                goals_tested=len(episodes), timeouts=sum(ep['reason'] == 'timeout' for ep in episodes),
                mean_final_error_m=float(np.mean([ep['final_error_m'] for ep in episodes])),
                mean_elapsed_s=float(np.mean([ep['elapsed_s'] for ep in episodes])), episodes=episodes)


def record_validation(directory, report):
    with (directory / 'validation.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(report, allow_nan=False) + '\n')
    print(f"Validation at {report['timesteps']} steps: {report['goals_reached']}/{report['goals_tested']} goals, "
          f"mean final error {report['mean_final_error_m']:.3f} m", flush=True)


class NavigationCallback(BaseCallback):
    """Advance the curriculum on arrivals and select checkpoints by goal tracking."""
    def __init__(self, validation_env, cfg, directory, seed, goals, stages, stage):
        super().__init__()
        self.validation_env = validation_env
        self.directory = directory
        self.seed = seed
        self.goals = goals
        self.stages = stages
        self.stage = stage
        self.window = deque(maxlen=cfg.get('curriculum_window', 20))
        self.pass_fraction = cfg.get('curriculum_success_fraction', 0.8)
        self.interval = cfg.get('validation_interval', 20000)
        if self.interval < 1 or self.window.maxlen < 1 or not 0 < self.pass_fraction <= 1:
            raise ValueError('Invalid curriculum or validation settings')
        self.best_score = None
        self.best_validated_score = None
        self.cfg = cfg
        self._continue_training = True
        self.success_streak = 0
        self.last_validation_update = None

    def _on_training_start(self):
        self.next_eval = self.model.num_timesteps + self.interval
        self.model.navigation_stage = self.stage

    def _validate(self):
        report = validate_goals(self.model, self.validation_env, self.goals, self.seed)
        report['curriculum_stage'] = self.stage
        if self.cfg.get('stage_validation', False):
            radius = self.stages[self.stage] or 4.0
            centre = self.validation_env.start_pos
            stage_goals = []
            for axis in np.eye(3):
                for sign in (-1, 1):
                    goal = centre + sign * 0.75 * radius * axis
                    goal = np.clip(goal, -self.cfg['flight_bounds'] + 0.1,
                                   self.cfg['flight_bounds'] - 0.1)
                    goal[2] = max(0.3, goal[2])
                    stage_goals.append(goal)
            stage_report = validate_goals(self.model, self.validation_env, stage_goals, self.seed + 100)
            report['stage_validation'] = stage_report
        record_validation(self.directory, report)
        passed = report['goals_reached'] == report['goals_tested']
        if 'stage_validation' in report:
            stage_report = report['stage_validation']
            passed &= stage_report['goals_reached'] == stage_report['goals_tested']
        score = (report['goals_reached'], -report['mean_final_error_m'], -report['mean_elapsed_s'])
        if self.best_score is None or score > self.best_score:
            self.model.save(self.directory / 'best_candidate.zip')
            self.best_score = score
        if passed and (self.best_validated_score is None or score > self.best_validated_score):
            self.model.save(self.directory / 'validated_model.zip')
            self.best_validated_score = score
        update = getattr(self.model, '_n_updates', None)
        if update != self.last_validation_update:
            self.success_streak = self.success_streak + 1 if passed else 0
            patience = self.cfg.get('validation_success_patience', 0)
            if patience and self.success_streak >= patience:
                self._continue_training = False
                print('Stopping after repeated successful validation.', flush=True)
            self.last_validation_update = update

    def _on_step(self):
        for done, info in zip(self.locals['dones'], self.locals['infos']):
            if done:
                record = {key: info.get(key) for key in ('is_success', 'termination_reason',
                          'goal_error', 'goal_speed', 'episode_radius', 'reward_components', 'reward_totals', 'episode')}
                record['timesteps'] = self.num_timesteps
                with (self.directory / 'episodes.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, allow_nan=False) + '\n')
        if self.stage + 1 < len(self.stages):
            for done, info in zip(self.locals['dones'], self.locals['infos']):
                if done and info['episode_radius'] == self.stages[self.stage]:
                    self.window.append(info['is_success'])
            if len(self.window) == self.window.maxlen and np.mean(self.window) >= self.pass_fraction:
                self.stage += 1
                self.model.navigation_stage = self.stage
                self.training_env.env_method('set_training_radius', self.stages[self.stage])
                self.window.clear()
                print(f"Goal curriculum advanced to stage {self.stage + 1}")
        if self.num_timesteps >= self.next_eval:
            self._validate()
            self.next_eval += self.interval
        return self._continue_training

    def _on_training_end(self):
        # Include the weights after the last PPO update and short runs.
        self._validate()


def main():
    args = parse_args()
    saved_model = args.resume or args.validate_only
    config_path = config_for_model(saved_model, args.config) if saved_model else (args.config or 'configs/params.yaml')
    cfg = load_config(config_path)
    if args.torch_threads < 1:
        raise ValueError('torch-threads must be positive')
    torch.set_num_threads(args.torch_threads)
    goals = np.asarray(cfg.get('validation_goals', DEFAULT_GOALS), dtype=float)
    if goals.ndim != 2 or goals.shape[1] != 3 or not len(goals) or not np.isfinite(goals).all():
        raise ValueError('validation_goals must contain at least one finite 3D goal')
    if np.any(np.abs(goals) > cfg['flight_bounds']) or np.any(goals[:, 2] < 0.05):
        raise ValueError('Validation goals must lie inside the flight area')
    total_steps = cfg['ppo_total_timesteps'] if args.timesteps is None else args.timesteps
    n_envs = 1 if args.gui else args.n_envs
    if total_steps < 1 or n_envs < 1:
        raise ValueError('Timesteps and environment count must be positive')
    stages = cfg.get('training_goal_radii', [None])
    if not stages or stages[-1] is not None or any(
            r is None or not np.isfinite(r) or r <= cfg['acceptance_radius'] for r in stages[:-1]):
        raise ValueError('Use positive curriculum radii followed by null for the full navigation task')
    parent = 'results/validation' if args.validate_only else 'results/training'
    directory = new_run_dir(args.output_dir, parent)
    distant_obstacle = np.full(3, cfg['flight_bounds'] + 2.0)
    with ExitStack() as stack:
        validation_env = stack.enter_context(closing(UAVNavEnv(cfg, obstacle_init_pos=distant_obstacle)))
        if args.validate_only:
            model = PPO.load(args.validate_only, device='cpu')
            check_model_interface(model, cfg)
            write_run_manifest(directory, cfg, args.validate_only, mode='validation', seed=args.seed)
            report = validate_goals(model, validation_env, goals, args.seed)
            record_validation(directory, report)
            print(f"Validation results saved to {directory}")
            return

        def make_env(rank):
            def create():
                return Monitor(UAVNavEnv(cfg, gui=args.gui and rank == 0),
                               filename=str(directory / f'env_{rank}.monitor.csv'),
                               info_keywords=('is_success', 'termination_reason', 'goal_error', 'goal_speed'))
            return create

        train_env = stack.enter_context(closing(DummyVecEnv([make_env(i) for i in range(n_envs)])))
        train_env.seed(args.seed)
        if args.resume:
            model = PPO.load(args.resume, env=train_env, device='cpu')
            check_model_interface(model, cfg)
            stage = min(getattr(model, 'navigation_stage', len(stages) - 1), len(stages) - 1)
        else:
            model = PPO('MlpPolicy', train_env, learning_rate=cfg['ppo_learning_rate'],
                        n_steps=cfg['ppo_n_steps'], batch_size=cfg['ppo_batch_size'],
                        n_epochs=cfg['ppo_n_epochs'], gamma=cfg['ppo_gamma'],
                        clip_range=cfg['ppo_clip_range'], ent_coef=cfg['ppo_ent_coef'],
                        target_kl=cfg.get('ppo_target_kl'),
                        policy_kwargs=dict(net_arch=dict(pi=cfg.get('ppo_net_arch', [256,256]),
                                                         vf=cfg.get('ppo_net_arch', [256,256])),
                                           log_std_init=cfg.get('ppo_log_std_init', 0.),
                                           activation_fn=torch.nn.Tanh),
                        verbose=1, seed=args.seed, device='cpu')
            stage = 0
            model.navigation_interface = interface_spec(cfg)
            model.training_method = 'PPO from scratch'
        model.set_logger(configure(str(directory), ['stdout', 'csv']))
        train_env.env_method('set_training_radius', stages[stage])
        write_run_manifest(directory, cfg, args.resume, mode='training', seed=args.seed,
                           environments=n_envs, requested_timesteps=total_steps, start_stage=stage,
                           training_method=model.training_method if hasattr(model, 'training_method') else 'legacy PPO',
                           interface=interface_spec(cfg))
        selection = NavigationCallback(validation_env, cfg, directory, args.seed, goals, stages, stage)
        checkpoints = CheckpointCallback(save_freq=max(1, 50000 // n_envs),
                                         save_path=str(directory / 'checkpoints'), name_prefix='ppo_uav')
        model.learn(total_timesteps=total_steps, callback=[selection, checkpoints],
                    reset_num_timesteps=args.resume is None)
        model.save(directory / 'ppo_final.zip')
    print(f"Training outputs saved to {directory}")


if __name__ == '__main__':
    main()
