"""Adaptive finite-horizon barrier filter for one UAV and one obstacle."""
import numpy as np

try:
    import cvxpy as cp
except ImportError:
    cp = None

from utils.config import command_bounds
from utils.math_utils import adaptive_gamma_star, predict_uav_trajectory, safety_distance


class FilterFailure(RuntimeError):
    """The evaluation must stop before sending an unchecked command."""


class ACBFFilter:
    def __init__(self, config):
        self.dt = config['dt']
        self.N = config['N_pred']
        self.gamma1 = config['gamma1']
        self.gamma2 = config['gamma2']
        self.alpha = config['alpha_clf']
        self.closing_time_scale = config.get('closing_time_scale', config.get('gamma_max', 2.0))
        self.lam_u = float(config['lambda_u'])
        self.lam_g = float(config['lambda_gamma'])
        self.lam_d = float(config['lambda_delta'])
        self.u_min, self.u_max = command_bounds(config)
        self.ds = safety_distance(config['drone_radius'], config['obs_radius'], config['safety_cushion'])
        self.tol = 1e-6
        self._problem = None
        self.reset()

    def reset(self):
        self._gamma_prev = self._gamma_star_prev = 0.5
        self.diagnostics = {}

    def _residuals(self, positions, obstacles, gamma):
        # Include clearance at each future point as well as all order-2 conditions.
        relative = positions - obstacles
        h = 0.5 * (np.sum(relative ** 2, axis=1) - self.ds ** 2)
        psi = h[1:] - (1 - self.gamma1) * h[:-1]
        return np.r_[h[1:], psi[1:] - (1 - self.gamma2 * gamma) * psi[:-1]], psi

    def _linearise(self, rollout, obstacles, u0, gamma0, residual, psi):
        """Differentiate the entire residual around (u0, gamma0), within the box."""
        rows = np.zeros((len(residual), 4))
        eps = 0.001 * (self.u_max - self.u_min)
        for axis in range(3):
            upper, lower = u0.copy(), u0.copy()
            upper[axis] = min(u0[axis] + eps, self.u_max)
            lower[axis] = max(u0[axis] - eps, self.u_min)
            plus = residual if upper[axis] == u0[axis] else self._residuals(rollout(upper), obstacles, gamma0)[0]
            minus = residual if lower[axis] == u0[axis] else self._residuals(rollout(lower), obstacles, gamma0)[0]
            rows[:, axis] = (plus - minus) / (upper[axis] - lower[axis])
        rows[len(obstacles) - 1:, 3] = self.gamma2 * psi[:-1]
        return rows, rows @ np.r_[u0, gamma0] - residual

    def _build_problem(self, count):
        horizon = (count + 1) // 2
        self._u = cp.Variable(3)
        self._gamma = cp.Variable()
        self._delta = cp.Variable(nonneg=True)
        self._xi = cp.Variable(horizon - 1, nonneg=True)
        self._ref = cp.Parameter(3)
        self._target = cp.Parameter()
        self._allowance = cp.Parameter(nonneg=True)
        self._rows = cp.Parameter((count, 4))
        self._rhs = cp.Parameter(count)
        self._lower = cp.Parameter(4)
        self._upper = cp.Parameter(4)
        x = cp.hstack([self._u, self._gamma])
        cost = (self.lam_u * cp.sum_squares(self._u - self._ref)
                + self.lam_g * cp.square(self._gamma - self._target)
                + self.lam_d * (cp.square(self._delta) + cp.sum_squares(self._xi)))
        self._problem = cp.Problem(cp.Minimize(cost), [
            x >= self._lower, x <= self._upper,
            cp.square(self._gamma - self._target) <= self._delta + self._allowance,
            self._rows[:horizon] @ x >= self._rhs[:horizon],
            self._rows[horizon:] @ x + self._xi >= self._rhs[horizon:],
        ])

    def _solve(self, rows, rhs, ref, target, allowance, lower, upper):
        if self._problem is None or self._rhs.size != len(rhs):
            self._build_problem(len(rhs))
        self._rows.value, self._rhs.value = rows, rhs
        self._ref.value, self._target.value, self._allowance.value = ref, target, allowance
        self._lower.value, self._upper.value = lower, upper
        attempts = []
        for solver in ('CLARABEL', 'SCS'):
            try:
                self._problem.solve(solver=solver, warm_start=True, verbose=False)
            except cp.error.SolverError as err:
                attempts.append(f"{solver}: {err}")
                continue
            status = self._problem.status
            attempts.append(f"{solver}: {status}")
            self.diagnostics['inaccurate_seen'] |= status == cp.OPTIMAL_INACCURATE
            if status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                continue
            u, gamma = self._u.value, self._gamma.value
            slack = np.r_[np.zeros(len(rhs) - self._xi.size), self._xi.value]
            delta = self._delta.value
            if not np.isfinite(np.r_[u, gamma, delta, slack]).all():
                continue
            x = np.r_[u, gamma]
            primal_error = max(np.max(lower - x), np.max(x - upper),
                               (gamma - target)**2 - delta - allowance, -delta, -slack.min(),
                               np.max(rhs - rows @ x - slack))
            if primal_error <= self.tol:
                self.diagnostics.update(solver=solver, solver_status=status,
                                        used_backup_solver=(self.diagnostics['used_backup_solver'] or len(attempts) > 1))
                self.diagnostics['solver_attempts'].extend(attempts)
                return np.clip(u, self.u_min, self.u_max), float(np.clip(gamma, 0, 1)), slack, float(delta)
        self.diagnostics.update(solver_status='solver_failure', used_backup_solver=True)
        self.diagnostics['solver_attempts'].extend(attempts)
        raise FilterFailure("No solver returned a finite solution within tolerance: " + ' | '.join(attempts))

    def filter(self, u_ref, p_uav, v_uav, obs_traj, v_obs=None, *, predict_positions=None):
        """Filter a held waypoint command using the supplied plant rollout.

        Standalone acceleration-model studies may omit predict_positions. The
        flight evaluator always supplies the environment's waypoint/PID rollout.
        """
        ref = np.clip(np.asarray(u_ref, dtype=float).reshape(3), self.u_min, self.u_max)
        obstacles = np.asarray(obs_traj, dtype=float)[:self.N + 1]
        steps = len(obstacles) - 1
        self.diagnostics = dict(solver='', solver_status='unavailable', used_backup_solver=False,
                                solver_attempts=[], inaccurate_seen=False, reference=ref.copy(), slack_max=None,
                                clf_slack=None, predicted_violation=None, gamma=None,
                                linearization_margin=0.0, solve_passes=0)
        if cp is None:
            raise FilterFailure("CVXPY is unavailable")
        if steps < 2 or obstacles.shape != (steps + 1, 3) or not np.isfinite(obstacles).all():
            raise FilterFailure("Invalid obstacle prediction horizon")
        v_obs = np.zeros(3) if v_obs is None else v_obs
        if not all(np.isfinite(value).all() for value in (u_ref, p_uav, v_uav, v_obs)):
            raise FilterFailure("Non-finite input to the safety filter")

        def rollout(u):
            if predict_positions is not None:
                positions = predict_positions(u, steps)
            else:
                positions = predict_uav_trajectory(p_uav, v_uav, u, self.dt, steps)
            if positions.shape != (steps + 1, 3) or not np.isfinite(positions).all():
                raise FilterFailure("Invalid UAV forecast")
            return positions

        target = adaptive_gamma_star(p_uav, v_uav, obstacles[0], v_obs, self.closing_time_scale)
        allowance = (1 - self.alpha) * (self._gamma_prev - self._gamma_star_prev)**2
        u, gamma = ref.copy(), target
        positions = rollout(u)
        residual, psi = self._residuals(positions, obstacles, gamma)
        if not np.isfinite(residual).all():
            raise FilterFailure("Non-finite nominal barrier residual")
        slack = np.zeros_like(residual)
        delta = 0.0
        if np.min(residual) >= 0:
            # The reference attains the zero lower bound of the objective.
            self.diagnostics['solver_status'] = 'reference_feasible'
        else:
            margin = np.zeros_like(residual)
            box_lo = np.r_[np.full(3, self.u_min), 0.]
            box_hi = np.r_[np.full(3, self.u_max), 1.]
            for iteration in range(6):
                rows, rhs = self._linearise(rollout, obstacles, u, gamma, residual, psi)
                self.diagnostics['solve_passes'] += 1
                self.diagnostics['linearization_margin'] = float(margin.max())
                centre = np.r_[u, gamma]
                radius = (box_hi - box_lo) * 0.25**iteration
                lower, upper = np.maximum(box_lo, centre-radius), np.minimum(box_hi, centre+radius)
                u, gamma, slack, delta = self._solve(rows, rhs + margin, ref, target, allowance, lower, upper)
                positions = rollout(u)
                residual, psi = self._residuals(positions, obstacles, gamma)
                if not np.isfinite(residual).all():
                    raise FilterFailure("Non-finite verification rollout")
                self.diagnostics.update(slack_max=float(slack.max()), clf_slack=delta, gamma=gamma,
                                        predicted_violation=float(max(0, -residual.min())))
                deficit = -(residual + slack)
                if deficit.max() <= self.tol:
                    break
                # Carry the observed remainder into the next subproblem so
                # successive tangents cannot repeatedly accept the same error.
                margin += np.where(deficit > self.tol, deficit + self.tol, 0)
            else:
                self.diagnostics['solver_status'] = 'nonlinear_check_failed'
                raise FilterFailure("Rollout exceeded the allowed barrier slack after relinearization")
        self.diagnostics.update(slack_max=float(slack.max()), clf_slack=delta, gamma=gamma,
                                predicted_violation=float(max(0, -residual.min())),
                                next_position=positions[1].tolist(),
                                predicted_min_dist=float(np.linalg.norm(positions[1:] - obstacles[1:], axis=1).min()))
        self._gamma_prev, self._gamma_star_prev = gamma, target
        return u

    @property
    def gamma(self):
        return self._gamma_prev
