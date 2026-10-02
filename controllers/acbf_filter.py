"""
controllers/acbf_filter.py
──────────────────────────
Adaptive Control Barrier Function (ACBF) safety filter.

Implements Sec. 3.3 of the paper:
  • Second-order discrete-time CBF with N-step prediction horizon (Eq. 22-24)
  • Adaptive coefficient γ(k) driven by a CLF (Eqs. 25-28)
  • QP optimization problem (Eq. 29) solved via CVXPY + CLARABEL/OSQP

Decision variables:
    u     ∈ ℝ³          safe control acceleration
    γ     ∈ [0, 1]      adaptive safety margin coefficient
    δ     ∈ ℝ₊          CLF slack (relaxation for feasibility)

QP objective (Eq. 29):
    min  λ_u‖u - u_f‖² + λ_γ(γ - γ*)² + λ_δ·δ²

Constraints:
    [CBF-i]  g_iᵀu + γ₂·ψ₁_i·γ  ≥  rhs_i        ∀i ∈ {2,...,N}
    [CLF]    (γ - γ*)²           ≤  δ + (1-α)·V_prev
    [Box]    u_min ≤ u ≤ u_max
    [Bounds] 0 ≤ γ ≤ 1,  δ ≥ 0

The CBF constraints are LINEARISED around u_ref (PPO output) to keep
the problem a convex QCQP solvable by standard solvers.
"""

from __future__ import annotations
import numpy as np

try:
    import cvxpy as cp
    _CVXPY_AVAILABLE = True
except ImportError:
    _CVXPY_AVAILABLE = False
    print("[ACBF] WARNING: cvxpy not installed — ACBF will pass through u_ref unchanged.")

from utils.math_utils import (
    predict_uav_position_i,
    safety_fn,
    adaptive_gamma_star,
    G_VEC,
)


class ACBFFilter:
    """
    ACBF safety filter wrapping the QP formulation from Eq. (29).

    Parameters
    ----------
    config : dict   loaded from configs/params.yaml
    """

    def __init__(self, config: dict):
        self.cfg      = config
        self.dt       = config['dt']
        self.N        = config['N_pred']
        self.gamma1   = config['gamma1']
        self.gamma2   = config['gamma2']
        self.gamma_max= config['gamma_max']
        self.alpha    = config['alpha_clf']
        self.lam_u    = config['lambda_u']
        self.lam_g    = config['lambda_gamma']
        self.lam_d    = config['lambda_delta']
        self.u_max    = config['u_max']
        self.u_min    = config['u_min']
        self.ds       = (config['drone_radius']
                         + config['obs_radius']
                         + config['safety_cushion'])

        # Persistent state: γ from previous step
        self._gamma_prev      = 0.5
        self._gamma_star_prev = 0.5

    def reset(self):
        """Reset filter state (call at start of each episode)."""
        self._gamma_prev      = 0.5
        self._gamma_star_prev = 0.5

    # ── Main interface ────────────────────────────────────────────────────────────

    def filter(self,
               u_ref: np.ndarray,
               p_uav: np.ndarray, v_uav: np.ndarray,
               obs_traj: np.ndarray,
               v_obs: np.ndarray | None = None) -> np.ndarray:
        """
        Compute a safety-certified control input u(k).

        Parameters
        ----------
        u_ref    : (3,)    PPO (or PID) reference acceleration
        p_uav    : (3,)    current UAV position
        v_uav    : (3,)    current UAV velocity
        obs_traj : (N+1,3) predicted obstacle trajectory [p̂_o(k|k),...,p̂_o(k+N|k)]
        v_obs    : (3,) | None   estimated obstacle velocity (for γ* computation)

        Returns
        -------
        u_safe : (3,)  minimally-modified safe control
        """
        if not _CVXPY_AVAILABLE:
            return u_ref.copy()

        # Normalise all inputs to (3,) — guard against (1,3) shapes from SB3
        u_ref   = np.array(u_ref).flatten()[:3]
        p_uav   = np.array(p_uav).flatten()[:3]
        v_uav   = np.array(v_uav).flatten()[:3]

        # Obstacle velocity estimate (fallback to zero if unknown)
        v_obs_est = v_obs if v_obs is not None else np.zeros(3)
        p_obs_now = obs_traj[0]

        # ── Adaptive γ* from relative kinematics (Eq. 28) ──────────────────────
        gamma_star = adaptive_gamma_star(
            p_uav, v_uav, p_obs_now, v_obs_est, self.gamma_max)
        gamma_star = float(np.clip(gamma_star, 0.0, 1.0))

        # ── Precompute h values at reference input (u_ref) ─────────────────────
        h = self._compute_h_sequence(p_uav, v_uav, u_ref, obs_traj)

        # ── ψ₁ evaluated at current state (independent of u) ─────────────────
        # ψ₁(k) = h₁ - h₀ + γ₁·h₀ = h₁ - (1-γ₁)·h₀
        # (h₀ and h₁ don't involve u — see Sec 3.3 discussion)
        psi1 = h[1] - (1.0 - self.gamma1) * h[0]

        # ── Linearise CBF constraints for i = 2,...,N ─────────────────────────
        A_cbf, b_cbf = self._linearise_cbf(p_uav, v_uav, u_ref, obs_traj,
                                            h, psi1)

        # ── CLF V_prev ─────────────────────────────────────────────────────────
        V_prev = (self._gamma_prev - self._gamma_star_prev) ** 2

        # ── Solve QP ───────────────────────────────────────────────────────────
        u_safe, gamma_new = self._solve_qp(u_ref, gamma_star, V_prev,
                                           A_cbf, b_cbf, psi1)

        # Store for next step
        self._gamma_prev      = gamma_new
        self._gamma_star_prev = gamma_star

        return u_safe

    # ── QP solver ────────────────────────────────────────────────────────────────

    def _solve_qp(self, u_ref, gamma_star, V_prev, A_cbf, b_cbf, psi1):
        """
        Solve Eq. (29) with CVXPY.

        Returns (u_safe, gamma_new).
        """
        u     = cp.Variable(3)
        gamma = cp.Variable()
        delta = cp.Variable()

        # Objective
        obj = (self.lam_u * cp.sum_squares(u - u_ref)
               + self.lam_g * cp.square(gamma - gamma_star)
               + self.lam_d * cp.square(delta))

        constraints = [
            u >= self.u_min,
            u <= self.u_max,
            gamma >= 0.0,
            gamma <= 1.0,
            delta >= 0.0,
            # CLF constraint (Eq. 26): (γ-γ*)² ≤ δ + (1-α)·V_prev
            cp.square(gamma - gamma_star) <= delta + (1 - self.alpha) * V_prev,
        ]

        # CBF constraints: A_cbf[i] @ [u; gamma] ≥ b_cbf[i]
        xu = cp.hstack([u, gamma])  # shape (4,)
        for row, rhs in zip(A_cbf, b_cbf):
            constraints.append(row @ xu >= rhs)

        prob = cp.Problem(cp.Minimize(obj), constraints)

        # Try multiple solvers for robustness
        for solver in [cp.CLARABEL, cp.OSQP, cp.SCS]:
            try:
                prob.solve(solver=solver, warm_start=True, verbose=False)
                if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    u_val     = np.array(u.value, dtype=float).flatten()
                    gamma_val = float(gamma.value)
                    return np.clip(u_val, self.u_min, self.u_max), gamma_val
            except Exception:
                continue

        # Fallback: return reference (QP infeasible)
        print("[ACBF] QP infeasible — returning u_ref")
        return u_ref.copy(), self._gamma_prev

    # ── Precompute helpers ────────────────────────────────────────────────────────

    def _compute_h_sequence(self, p0, v0, u_ref, obs_traj) -> np.ndarray:
        """
        h[i] = ½‖p̂_uav(k+i|k) - p̂_obs(k+i|k)‖² - d_s²   (Eq. 22)
        evaluated at control input u_ref.
        """
        N = min(self.N, len(obs_traj) - 1)
        h = np.zeros(N + 1)
        for i in range(N + 1):
            p_i = predict_uav_position_i(p0, v0, u_ref, self.dt, i)
            h[i] = safety_fn(p_i, obs_traj[i], self.ds)
        return h

    def _linearise_cbf(self, p0, v0, u_ref, obs_traj, h, psi1) -> tuple:
        """
        Build linearised CBF constraint rows for i ∈ {2, ..., N}.

        Linearisation of h_i(u) around u_ref:
            h_i(u) ≈ h_i(u_ref) + g_iᵀ(u - u_ref)

        where g_i = ∇_u h_i = c_i·dt²·(p̂_i - p̂_o_i)  with c_i = i(i-1)/2

        Constraint (from Eq. 24, rearranged):
            h_i(u) + γ₂·ψ₁·γ ≥ h_{i-1}·(1-γ₁) + ε_h
            → g_iᵀu + γ₂·ψ₁·γ ≥ h_{i-1}·(1-γ₁) - h_i(u_ref) + g_iᵀu_ref

        Decision-variable vector: [u(3), γ(1)]  → each row has shape (4,)
        """
        N   = min(self.N, len(obs_traj) - 1)
        dt  = self.dt
        rows = []
        rhs  = []

        for i in range(2, N + 1):
            c_i  = i * (i - 1) / 2          # coefficient for acceleration term
            p_i  = predict_uav_position_i(p0, v0, u_ref, dt, i)
            diff = p_i - obs_traj[i]          # (3,) relative position vector

            # Gradient of h_i w.r.t. u
            g_i = c_i * (dt ** 2) * diff     # (3,)

            # Build row [g_iᵀ, γ₂·ψ₁] for decision vector [u; γ]
            row = np.zeros(4)
            row[:3] = g_i
            row[3]  = self.gamma2 * psi1

            # RHS: h_{i-1}·(1-γ₁) - h_i(u_ref) + g_iᵀu_ref
            rhs_i = (h[i - 1] * (1.0 - self.gamma1)
                     - h[i]
                     + float(g_i @ u_ref))

            rows.append(row)
            rhs.append(rhs_i)

        if not rows:
            # Horizon too short — add a no-op constraint
            rows.append(np.zeros(4))
            rhs.append(-1e6)

        return np.array(rows), np.array(rhs)

    # ── Diagnostics ──────────────────────────────────────────────────────────────

    @property
    def gamma(self) -> float:
        """Most recent adaptive coefficient γ(k)."""
        return self._gamma_prev
