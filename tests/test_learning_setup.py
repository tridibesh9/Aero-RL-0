"""Physics and learning-contract regression checks. No learning is performed."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pybullet as p
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from controllers.hierarchical_ctrl import HierarchicalController
from envs.UAVNavEnv import UAVNavEnv
from train_ppo import NavigationCallback
from utils.config import load_config, validate_config
from utils.policy import check_model_interface, config_for_model, interface_spec

CFG = load_config(Path(__file__).resolve().parents[1] / 'configs/params.yaml')


class LearningSetupTests(unittest.TestCase):
    def setUp(self):
        self.no_learning = patch.object(PPO, 'learn', side_effect=AssertionError('Training is forbidden in tests'))
        self.no_learning.start()
        self.addCleanup(self.no_learning.stop)
        self.no_updates = patch.object(torch.optim.Adam, 'step', side_effect=AssertionError('No optimizer updates'))
        self.no_updates.start()
        self.addCleanup(self.no_updates.stop)

    def env(self, **kwargs):
        cfg = kwargs.pop('config', CFG)
        with contextlib.redirect_stdout(io.StringIO()):
            env = UAVNavEnv(cfg, obstacle_init_pos=[12,12,12], **kwargs)
        self.addCleanup(env.close)
        return env

    def test_hover_and_acceleration_units(self):
        env = self.env(target_pos=[5,0,2])
        env.reset(seed=1)
        for _ in range(40):
            info = env.step(np.zeros((1,3)))[-1]
        np.testing.assert_allclose(info['pos'], [0,0,1], atol=1e-7)
        for axis in range(3):
            env.reset(seed=1)
            action = np.zeros((1,3)); action[0,axis] = .5
            for _ in range(40):
                info = env.step(action)[-1]
            # Two seconds at .5 m/s², allowing attitude transients and drag.
            self.assertGreater(info['vel'][axis], .8)
            self.assertLess(info['vel'][axis], 1.05)
            other = [i for i in range(3) if i != axis]
            self.assertLess(np.abs(info['vel'][other]).max(), .01)

    def test_crash_is_terminal_and_timeout_bootstraps(self):
        for crash in (True, False):
            cfg = dict(CFG, max_episode_steps=1 if not crash else 500)
            with contextlib.redirect_stdout(io.StringIO()):
                vec = DummyVecEnv([lambda: Monitor(UAVNavEnv(cfg))])
            try:
                vec.reset(); env = vec.envs[0].unwrapped
                if crash:
                    p.resetBasePositionAndOrientation(env.DRONE_IDS[0], [0,0,.03], [0,0,0,1], physicsClientId=env.CLIENT)
                    env._updateAndStoreKinematicInformation()
                _, _, done, info = vec.step(np.zeros((1,1,3), np.float32))
                self.assertTrue(done[0])
                self.assertEqual(info[0]['TimeLimit.truncated'], not crash)
                self.assertEqual(info[0]['termination_reason'], 'out_of_bounds' if crash else 'timeout')
                self.assertFalse(info[0]['is_success'])
            finally:
                vec.close()

    def test_arrival_requires_braking_and_hold(self):
        env = self.env(target_pos=[0,0,1])
        env.reset(seed=1)
        for i in range(CFG['goal_hold_steps']):
            _, reward, term, trunc, info = env.step(np.zeros((1,3)))
            self.assertEqual(term, i == CFG['goal_hold_steps']-1)
            self.assertFalse(trunc)
        self.assertTrue(info['is_success'])
        self.assertGreater(reward, 90)
        env.reset(seed=1)
        p.resetBaseVelocity(env.DRONE_IDS[0], [1,0,0], physicsClientId=env.CLIENT)
        env._updateAndStoreKinematicInformation()
        _, _, term, _, info = env.step(np.zeros((1,3)))
        self.assertLess(info['goal_error'], .1)
        self.assertFalse(term)
        self.assertEqual(info['reward_components']['terminal'], 0)

    def test_observation_and_forecast_include_fast_controller(self):
        env = self.env(target_pos=[5,0,2])
        obs, info = env.reset(seed=3)
        self.assertEqual(obs.shape, (19,))
        np.testing.assert_allclose(obs[3:6], (env.target_pos-info['pos'])/CFG['obs_error_scale'])
        for _ in range(15):
            env.step(np.array([[.6,-.3,.2]], np.float32))
        state = env._getDroneStateVector(0).copy()
        memory = env.ctrl[0].last_rpy.copy()
        command = np.array([-.2,.3,-.1])
        forecast = env.predict_positions(command, 5)
        np.testing.assert_array_equal(state, env._getDroneStateVector(0))
        np.testing.assert_array_equal(memory, env.ctrl[0].last_rpy)
        actual = np.array([env.step(command[None,:])[-1]['pos'] for _ in range(5)])
        np.testing.assert_allclose(forecast[1:], actual, atol=1e-4)

    def test_pd_calibration_can_brake_in_all_directions(self):
        # This verifies the plant, not a claim that PPO has learned this policy.
        env = self.env()
        goals = [[.3,0,1],[-.3,0,1],[0,.3,1],[0,-.3,1],[0,0,.6],[0,0,1.4],[5,0,2]]
        for goal in goals:
            _, info = env.reset(seed=4, options={'target_pos': goal})
            for _ in range(500):
                command = np.clip(np.asarray(goal)-info['pos']-2*info['vel'], -1, 1)
                _, _, term, trunc, info = env.step(command[None,:])
                if term or trunc:
                    break
            self.assertTrue(info['is_success'], (goal, info['termination_reason']))
            self.assertLess(info['goal_speed'], CFG['goal_speed_tolerance'])

    def test_policy_compatibility_and_same_observation_in_filter(self):
        env = self.env(target_pos=[5,0,2])
        obs, info = env.reset(seed=4)
        model = PPO('MlpPolicy', env, n_steps=8, batch_size=8, seed=4,
                    policy_kwargs=dict(net_arch=[16,16]), device='cpu')
        with self.assertRaises(ValueError):
            check_model_interface(model, CFG)
        model.navigation_interface = interface_spec(CFG)
        check_model_interface(model, CFG)
        with self.assertRaises(ValueError):
            check_model_interface(model, dict(CFG, obs_error_scale=6.))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'initial_untrained.zip'
            model.save(path)
            check_model_interface(PPO.load(path, device='cpu'), CFG)
        ctrl = HierarchicalController(model, CFG)
        expected, _ = model.predict(obs, deterministic=True)
        action, _ = ctrl.act_normalized(info['pos'],info['vel'],env.target_pos,info['obs_pos'],
                                        predict_positions=env.predict_positions, policy_observation=obs)
        np.testing.assert_allclose(action, expected, atol=1e-7)
        with self.assertRaisesRegex(ValueError, 'policy_observation'):
            ctrl.act(info['pos'], info['vel'], env.target_pos, info['obs_pos'],
                     predict_positions=env.predict_positions)

    def test_stage_failures_prevent_validation_promotion(self):
        with tempfile.TemporaryDirectory() as temp:
            cb = NavigationCallback(SimpleNamespace(start_pos=np.array([0,0,1.])), CFG,
                                    Path(temp), 42, [[3.,0,1]], [.4,None], 0)
            cb.model = Mock(num_timesteps=100, _n_updates=10)
            good = dict(timesteps=100,goals_reached=1,goals_tested=1,mean_final_error_m=.05,mean_elapsed_s=3.)
            bad = dict(good,goals_reached=0)
            with patch('train_ppo.validate_goals', side_effect=[good,bad]):
                cb._validate()
            self.assertEqual([c.args[0].name for c in cb.model.save.call_args_list], ['best_candidate.zip'])
            self.assertEqual(cb.success_streak, 0)
            self.assertIn('stage_validation', json.loads((Path(temp)/'validation.jsonl').read_text()))

    def test_invalid_acceleration_configuration(self):
        for override in [dict(action_mode='invalid'), dict(k_ma=10), dict(obs_error_scale=0),
                         dict(goal_hold_steps=0), dict(distance_cost=-1)]:
            with self.assertRaises(ValueError):
                validate_config(dict(CFG, **override))

    def test_model_config_resolution_and_episode_logging(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            with self.assertRaises(ValueError):
                config_for_model(directory/'old.zip')
            (directory/'params.yaml').write_text('snapshot')
            self.assertEqual(Path(config_for_model(directory/'checkpoints'/'model.zip')),
                             directory/'params.yaml')
            self.assertEqual(config_for_model(directory/'model.zip', 'explicit.yaml'), 'explicit.yaml')
            env = self.env(target_pos=[0,0,1])
            env.reset(seed=1)
            total = 0.
            for _ in range(CFG['goal_hold_steps']):
                _, reward, _, _, info = env.step(np.zeros((1,3)))
                total += reward
            cb = NavigationCallback(None, CFG, directory, 42, [[1,0,1]], [.4,None], 0)
            cb.model = Mock(num_timesteps=5)
            cb._on_training_start()
            cb.locals = dict(dones=[True], infos=[info])
            cb._on_step()
            saved = json.loads((directory/'episodes.jsonl').read_text())
            self.assertTrue(saved['is_success'])
            self.assertAlmostEqual(sum(saved['reward_totals'].values()), total)


if __name__ == '__main__':
    unittest.main()
