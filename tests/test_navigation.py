"""Regression checks for the safety geometry, plant model, selection and evidence."""
import contextlib
import csv
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pybullet as bullet
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from controllers.acbf_filter import ACBFFilter, FilterFailure
from controllers.hierarchical_ctrl import PIDWithACBF
from controllers.imm_predictor import CT, IMMPredictor
from envs.UAVNavEnv import UAVNavEnv
from evaluate import run_episode
from scenarios import case1_linear, case2_accel, case3_figure8
from train_ppo import NavigationCallback, validate_goals
from utils.config import command_bounds, load_config, new_run_dir, validate_config
from utils.logger import EpisodeLogger
from utils.math_utils import adaptive_gamma_star, predict_uav_trajectory, safety_fn


CFG = load_config(Path(__file__).resolve().parents[1] / "configs/params_legacy.yaml")


class SafetyAndPredictionTests(unittest.TestCase):
    def test_geometry_and_closing_rate(self):
        origin = np.zeros(3)
        axis = np.array([1., 0., 0.])
        self.assertEqual(safety_fn(origin, 0.5 * axis, 0.5), 0)
        self.assertLess(safety_fn(origin, 0.49 * axis, 0.5), 0)
        approach = adaptive_gamma_star(origin, axis, axis, origin, 2.)
        recede = adaptive_gamma_star(origin, -axis, axis, origin, 2.)
        # The lower bound on the next positive psi must be higher when closing.
        self.assertGreater(1 - CFG["gamma2"] * approach, 1 - CFG["gamma2"] * recede)
        self.assertEqual(recede, 1.)

    def test_full_barrier_linearization(self):
        f = ACBFFilter(CFG)
        p, v, u = np.array([0., 0., 1.]), np.array([.8, .1, 0]), np.array([.4, -.3, .6])
        obstacles = np.array([[.8 + .03*i, .3, 1.] for i in range(6)])
        rollout = lambda cmd: predict_uav_trajectory(p, v, cmd, .05, 5)
        gamma = .63
        x = np.r_[u, gamma]
        def exact(z):
            relative = rollout(z[:3]) - obstacles
            h = 0.5 * (np.sum(relative**2, axis=1) - f.ds**2)
            psi = np.diff(h) + f.gamma1 * h[:-1]
            return np.r_[h[1:], np.diff(psi) + f.gamma2*z[3]*psi[:-1]]
        r, psi = f._residuals(rollout(u), obstacles, gamma)
        rows, rhs = f._linearise(rollout, obstacles, u, gamma, r, psi)
        np.testing.assert_allclose(rows @ x - rhs, exact(x), atol=1e-10)
        for i, perturbation in enumerate(np.eye(4)*1e-5):
            derivative = (exact(x+perturbation) - exact(x-perturbation)) / 2e-5
            np.testing.assert_allclose(rows[:, i], derivative, atol=1e-8)

    def test_turn_model_uncertainty_and_predictions(self):
        m = IMMPredictor(CFG)
        x = np.array([.2, 1., -.3, .5, 1., -.2, .7])
        m.x_hat[CT] = x
        J = m._projection_matrix(CT)
        for i, delta in enumerate(np.eye(7)*1e-6):
            numerical = (m._to_common(CT, x+delta) - m._to_common(CT, x-delta)) / 2e-6
            np.testing.assert_allclose(J[:, i], numerical, atol=1e-9)
        P = np.diag(np.arange(1., 10.))
        np.testing.assert_equal(np.diag(m._cov_from_common(CT, P)), [1, 4, 2, 5, 3, 6, 0])
        for case in (case1_linear, case2_accel, case3_figure8):
            m.reset(case.OBS_INIT)
            errors = []
            cfg_dt = CFG['dt']
            for i in range(96):
                t = i * cfg_dt
                m.update(case.obstacle_policy(t, case.OBS_INIT))
                forecast = m.predict()
                truth = case.obstacle_policy(t + CFG['N_pred']*cfg_dt, case.OBS_INIT)
                errors.append(np.linalg.norm(forecast[-1] - truth))
                self.assertAlmostEqual(m.mu.sum(), 1.)
                self.assertTrue(all(np.linalg.eigvalsh(cov).min() >= -1e-10 for cov in m.P_hat))
            self.assertLess(np.sqrt(np.mean(np.square(errors))), 0.25)
        m.reset(np.zeros(3))
        m._update_probs(np.array([-2000., -2001., -1999.]))
        self.assertAlmostEqual(m.mu.sum(), 1.)
        dt = 1e-5
        velocity = (case3_figure8.obstacle_policy(dt, None) - case3_figure8.obstacle_policy(-dt, None)) / (2*dt)
        np.testing.assert_allclose(case3_figure8.SCENARIO_CONFIG['obs_velocity'], velocity)

    def test_config_and_existing_run_protection(self):
        self.assertIsInstance(CFG['lambda_delta'], float)
        self.assertEqual(command_bounds(dict(CFG, k_ma=.8, u_min=-.4)), (-.4, .8))
        for bad in (dict(CFG, dt=.2), dict(CFG, N_pred=1), dict(CFG, u_max=np.nan)):
            with self.assertRaises(ValueError):
                validate_config(bad)
        with tempfile.TemporaryDirectory() as temp:
            target = new_run_dir(Path(temp)/'run', temp)
            (target/'keep.txt').write_text('old results')
            with self.assertRaises(FileExistsError):
                new_run_dir(target, temp)
            self.assertEqual((target/'keep.txt').read_text(), 'old results')


class FlightTests(unittest.TestCase):
    def flight(self, config=CFG, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            env = UAVNavEnv(config, **kwargs)
        self.addCleanup(env.close)
        return env

    def test_seed_controller_reset_start_and_curriculum(self):
        e = self.flight(start_pos=[1., -1., 1.])
        def sequence():
            goals = []
            for seed in (15, None, None):
                obs, info = e.reset(seed=seed)
                np.testing.assert_allclose(info['pos'], [1., -1., 1.])
                np.testing.assert_allclose(obs[3:], e.target_pos-info['pos'], atol=1e-6)
                self.assertEqual(e.ctrl[0].control_counter, 0)
                e.step(np.ones((1,3)))
                goals.append(e.target_pos.copy())
            return goals
        e.set_training_radius(1.)
        goals = sequence()
        np.testing.assert_array_equal(goals, sequence())
        self.assertTrue(all(.5 <= np.linalg.norm(g-e.start_pos) <= 1. for g in goals))
        self.assertFalse(np.array_equal(goals[0], goals[1]))
        e.set_training_radius(None)
        e.reset(seed=1)
        self.assertGreater(np.linalg.norm(e.target_pos - e.start_pos), 2.)

    def test_waypoint_rollout_preserves_live_world_and_matches_flight(self):
        e = self.flight(target_pos=[5,0,2], obstacle_init_pos=[9,9,9])
        e.reset(seed=7)
        for _ in range(12):
            e.step(np.array([[.5,-.2,.3]], np.float32))
        u = np.array([.6,.35,-.2])
        state = e._getDroneStateVector(0).copy()
        counter, step = e.ctrl[0].control_counter, e._step_count
        predicted = e.predict_positions(u, 5)
        e.predict_positions(-u, 5)
        np.testing.assert_allclose(predicted, e.predict_positions(u, 5), atol=1e-9)
        np.testing.assert_array_equal(e._getDroneStateVector(0), state)
        self.assertEqual((counter, step), (e.ctrl[0].control_counter, e._step_count))
        actual = np.array([e.step(u[None,:].astype(np.float32))[-1]['pos'] for _ in range(5)])
        self.assertLess(np.linalg.norm(predicted[1:] - actual, axis=1).max(), .001)

        f = ACBFFilter(CFG)
        p, v = e.get_uav_state()
        obs = np.tile(p + [.65,0,0], (6,1))
        cmd = f.filter([1,0,0], p, v, obs, predict_positions=e.predict_positions)
        residual, _ = f._residuals(e.predict_positions(cmd, 5), obs, f.gamma)
        self.assertGreater(f.diagnostics['solve_passes'], 0)
        self.assertGreaterEqual(residual[:5].min(), -f.tol)
        self.assertGreaterEqual(residual.min() + f.diagnostics['slack_max'], -f.tol)
        self.assertTrue(np.all(np.abs(cmd) <= CFG['k_ma']))
        info = e.step((cmd/CFG['k_ma'])[None,:].astype(np.float32))[-1]
        np.testing.assert_allclose(info['command'], cmd, atol=1e-7)

    def test_failure_never_applies_reference(self):
        e = self.flight(target_pos=case1_linear.GOAL_POS, obstacle_init_pos=case1_linear.OBS_INIT)
        with patch('controllers.acbf_filter.cp', None):
            log = run_episode(e, PIDWithACBF(CFG), case1_linear, 'PID+ACBF', .5, 3)
        self.assertEqual(log.stop_reason, 'filter_failure')
        self.assertEqual(e._step_count, 0)
        self.assertEqual(log.summary(case1_linear.GOAL_POS)['filter_failures'], 1)
        self.assertEqual(len(log.times), 1)

    def test_flight_controller_requires_a_waypoint_forecast(self):
        ctrl = PIDWithACBF(CFG)
        ctrl.reset(case1_linear.OBS_INIT)
        with self.assertRaisesRegex(ValueError, 'predict_positions'):
            ctrl.act(case1_linear.START_POS, np.zeros(3), case1_linear.GOAL_POS, case1_linear.OBS_INIT)

    def test_out_of_bounds_and_vectorized_episode_info(self):
        e = self.flight(start_pos=[CFG['flight_bounds'], 0, 1], target_pos=[0,0,1],
                        obstacle_init_pos=[9,9,9])
        e.reset()
        bullet.resetBaseVelocity(e.DRONE_IDS[0], [1,0,0], physicsClientId=e.CLIENT)
        e._updateAndStoreKinematicInformation()
        _, _, terminated, truncated, info = e.step(np.zeros((1,3)))
        self.assertFalse(truncated)
        self.assertTrue(terminated)
        self.assertEqual(info['termination_reason'], 'out_of_bounds')
        with contextlib.redirect_stdout(io.StringIO()):
            vec = DummyVecEnv([lambda: Monitor(UAVNavEnv(dict(CFG, max_episode_steps=1)))])
        self.addCleanup(vec.close)
        vec.seed(42)
        vec.env_method('set_training_radius', 1.0)
        vec.reset()
        _, _, dones, infos = vec.step(np.zeros((1,1,3), dtype=np.float32))
        self.assertTrue(dones[0])
        self.assertIn('terminal_observation', infos[0])
        self.assertEqual(infos[0]['episode_radius'], 1.)
        self.assertEqual(infos[0]['termination_reason'], 'timeout')

    def test_raw_data_goal_timeout_limit_and_geometry(self):
        cfg = dict(CFG, max_episode_steps=2, drone_radius=.1, obs_radius=.15)
        e = self.flight(cfg, target_pos=case1_linear.GOAL_POS, obstacle_init_pos=[9,9,9])
        hover = Mock()
        hover.predict.return_value = (np.zeros((1,3)), None)
        for cap, reason in [(1, 'step_limit'), (3, 'timeout')]:
            log = run_episode(e, None, case1_linear, 'Pure RL', .35, cap, ppo=hover)
            self.assertEqual(log.stop_reason, reason)
            self.assertEqual(len(log.times), min(cap,2)+1)
            np.testing.assert_allclose(log.positions[-1], e.get_uav_state()[0])
        report = validate_goals(SimpleNamespace(predict=hover.predict, num_timesteps=100), e,
                                [[0,0,1.01], [2,0,1]], 100)
        self.assertEqual((report['goals_reached'], report['timeouts']), (1,1))
        geom = EpisodeLogger('test', cfg)
        for i, distance in enumerate([.3,.25]):
            geom.log(i*.05, np.zeros(3), np.zeros(3), np.zeros(3), np.array([distance,0,0]))
            self.assertTrue(geom.summary(np.zeros(3))['buffer_violation'])
            self.assertEqual(geom.collision, i == 1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'samples.csv'
            geom.write_csv(path, np.zeros(3))
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([r['collision'] for r in rows], ['False', 'True'])
            self.assertEqual(rows[-1]['termination_reason'], 'step_limit')

    def test_solver_exhaustion_is_reported(self):
        f = ACBFFilter(CFG)
        zero = np.zeros(3)
        import cvxpy as cp
        with patch.object(cp.Problem, 'solve', side_effect=cp.error.SolverError('injected failure')):
            with self.assertRaises(FilterFailure):
                f.filter(zero,zero,zero,np.tile([.2,0,0], (6,1)))
        self.assertEqual(f.diagnostics['solver_status'], 'solver_failure')
        self.assertTrue(any('SCS' in item for item in f.diagnostics['solver_attempts']))

    def test_unavoidable_contact_is_infeasible_even_with_soft_barrier_terms(self):
        f = ACBFFilter(CFG)
        # In this point-mass example p(1) is fixed by p(0) and v(0).
        # No bounded command can put p(1) outside the specified sphere.
        with self.assertRaises(FilterFailure):
            f.filter(np.zeros(3), np.zeros(3), np.zeros(3), np.tile([.2,0,0], (6,1)))
        self.assertEqual(f.diagnostics['solver_status'], 'solver_failure')

    def test_checkpoint_selection_and_curriculum_without_learning(self):
        with tempfile.TemporaryDirectory() as temp:
            cb = NavigationCallback(None, dict(CFG,curriculum_window=2), Path(temp), 42,
                                    [[0,0,1],[1,0,1]], [1.,None], 0)
            cb.model = Mock(num_timesteps=0)
            cb._on_training_start()
            cb.locals = dict(dones=[True,True], infos=[dict(episode_radius=1.,is_success=True)]*2)
            cb._on_step()
            cb.model.get_env().env_method.assert_called_once_with('set_training_radius', None)
            for arrivals, error in [(0,.2), (1,2.), (2,.08), (0,.01)]:
                result = dict(timesteps=0,goals_reached=arrivals,goals_tested=2,
                              mean_final_error_m=error,mean_elapsed_s=1.)
                with patch('train_ppo.validate_goals',return_value=result):
                    cb._validate()
            self.assertEqual([c.args[0].name for c in cb.model.save.call_args_list],
                             ['best_candidate.zip']*3 + ['validated_model.zip'])
            cb.model.learn.assert_not_called()


if __name__ == '__main__':
    unittest.main()
