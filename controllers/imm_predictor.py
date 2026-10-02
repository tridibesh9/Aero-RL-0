"""
controllers/imm_predictor.py
─────────────────────────────
Interacting Multiple Model (IMM) filter for obstacle state estimation and
finite-horizon trajectory prediction.

Implements Sec. 3.2 of the paper:
  • Three kinematic models: CV (Eq.15), CA (Eq.16), CT (Eq.17)
  • IMM cycle: Interaction → Model-matched KF → Probability update & Fusion
  • N-step ahead prediction via Eq. (20)

All state spaces are unified to 9-D (CA space) for mixing:
    common state = [px, py, pz, vx, vy, vz, ax, ay, az]

Model native spaces:
    CV : 6-D  [px,py,pz, vx,vy,vz]
    CA : 9-D  (= common)
    CT : 7-D  [px,vx, py,vy, pz,vz, ω]
"""

from __future__ import annotations
import numpy as np
from collections import deque
from typing import List


# ── Model indices ───────────────────────────────────────────────────────────────
CV, CA, CT = 0, 1, 2
N_MODELS = 3

# Common space dimension
DIM_COMMON = 9      # CA native = common
DIM_CV     = 6
DIM_CA     = 9
DIM_CT     = 7

# Position indices inside each model's native state
POS_IDX_CV = [0, 1, 2]
POS_IDX_CA = [0, 1, 2]
POS_IDX_CT = [0, 2, 4]   # [px, py, pz] in [px,vx,py,vy,pz,vz,ω]


class IMMPredictor:
    """
    IMM obstacle state estimator and trajectory predictor.

    Usage
    -----
        imm = IMMPredictor(config)
        imm.reset(initial_obs_pos)
        while running:
            p_hat = imm.update(z)          # fused position estimate
            traj  = imm.predict(N)         # shape (N+1, 3)
    """

    def __init__(self, config: dict):
        self.dt  = config['dt']
        self.N   = config['N_pred']
        self.ct_window        = config.get('ct_window', 20)
        self.likelihood_smooth = config.get('likelihood_smooth', 2)

        # Noise covariances
        q = config['Q_noise']
        r = config['R_noise']
        self.Q_cv = q * np.eye(DIM_CV)
        self.Q_ca = q * np.eye(DIM_CA)
        self.Q_ct = q * np.eye(DIM_CT)
        self.R    = r * np.eye(3)

        # Markov transition matrix Π [3×3]
        d, od = config['markov_diag'], config['markov_offdiag']
        self.Pi = np.array([
            [d,  od, od],
            [od, d,  od],
            [od, od, d ],
        ])

        # Mode probabilities μ[j]
        self.mu = np.ones(N_MODELS) / N_MODELS

        # Per-model state estimates (native spaces) and covariances
        self.x_hat: list[np.ndarray] = [
            np.zeros(DIM_CV),
            np.zeros(DIM_CA),
            np.zeros(DIM_CT),
        ]
        self.P_hat: list[np.ndarray] = [
            np.eye(DIM_CV) * 1.0,
            np.eye(DIM_CA) * 1.0,
            np.eye(DIM_CT) * 1.0,
        ]

        # CT turn-rate estimation
        self.pos_history: deque = deque(maxlen=self.ct_window)
        self.omega: float = 0.0

        # Likelihood smoothing
        self.like_history: list[deque] = [
            deque(maxlen=self.likelihood_smooth) for _ in range(N_MODELS)
        ]

        # Fused position and velocity estimates
        self.fused_pos: np.ndarray = np.zeros(3)
        self.fused_vel: np.ndarray = np.zeros(3)

        self.initialized = False

    # ── Public API ──────────────────────────────────────────────────────────────

    def reset(self, init_pos: np.ndarray):
        """Reset filter state to a known initial position."""
        self.mu = np.ones(N_MODELS) / N_MODELS
        # Initialize CV: position known, velocity = 0
        self.x_hat[CV] = np.zeros(DIM_CV)
        self.x_hat[CV][:3] = init_pos.copy()
        # CA: extend CV to 9D
        self.x_hat[CA] = np.zeros(DIM_CA)
        self.x_hat[CA][:3] = init_pos.copy()
        # CT: map position to [px,vx,py,vy,pz,vz,ω]
        self.x_hat[CT] = np.zeros(DIM_CT)
        self.x_hat[CT][0] = init_pos[0]
        self.x_hat[CT][2] = init_pos[1]
        self.x_hat[CT][4] = init_pos[2]

        self.P_hat = [np.eye(DIM_CV)*1.0, np.eye(DIM_CA)*1.0, np.eye(DIM_CT)*1.0]
        for h in self.like_history:
            h.clear()
        self.pos_history.clear()
        self.pos_history.append(init_pos.copy())
        self.omega = 0.0
        self.fused_pos = init_pos.copy()
        self.fused_vel = np.zeros(3)
        self.initialized = True

    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Perform one IMM cycle given position measurement z ∈ ℝ³.

        Returns fused position estimate p̂_o(k).
        """
        if not self.initialized:
            self.reset(z)

        self.pos_history.append(z.copy())
        self._estimate_turn_rate()

        # IMM cycle
        x_mix, P_mix = self._interact()
        likelihoods   = self._filter(z, x_mix, P_mix)
        self._update_probs(likelihoods)
        self.fused_pos, self.fused_vel = self._fuse()

        return self.fused_pos.copy()

    def predict(self, N: int | None = None) -> np.ndarray:
        """
        Predict obstacle positions over N steps using Eq. (20).

        Returns
        -------
        traj : (N+1, 3)   [p̂(k|k), p̂(k+1|k), ..., p̂(k+N|k)]
        """
        if N is None:
            N = self.N

        traj = np.zeros((N + 1, 3))
        traj[0] = self.fused_pos.copy()

        # Propagate each model forward independently then fuse
        for i in range(1, N + 1):
            pos_i = np.zeros(3)
            for j in range(N_MODELS):
                x_prop = self._propagate_model(j, i)
                pos_i += self.mu[j] * self._extract_pos(j, x_prop)
            traj[i] = pos_i

        return traj

    # ── IMM internals ────────────────────────────────────────────────────────────

    def _interact(self):
        """
        Step 1 — Interaction / Mixing  (Eq. 18).

        Computes mixed initial state for each filter in the COMMON 9-D space.

        Returns
        -------
        x_mix : list[(9,)]  mixed states for each model
        P_mix : list[(9,9)] mixed covariances
        """
        # Mixing probabilities  μ_{i|j} = Π_{ij} μ_i / c_j
        c_bar = self.Pi.T @ self.mu          # normalising constants (N_MODELS,)
        mu_mix = np.zeros((N_MODELS, N_MODELS))
        for i in range(N_MODELS):
            for j in range(N_MODELS):
                mu_mix[i, j] = self.Pi[i, j] * self.mu[i] / (c_bar[j] + 1e-12)

        x_common = [self._to_common(j, self.x_hat[j]) for j in range(N_MODELS)]
        P_common = [self._cov_to_common(j, self.P_hat[j]) for j in range(N_MODELS)]

        x_mix = []
        P_mix = []
        for j in range(N_MODELS):
            x_j = sum(mu_mix[i, j] * x_common[i] for i in range(N_MODELS))
            P_j = sum(
                mu_mix[i, j] * (
                    P_common[i]
                    + np.outer(x_common[i] - x_j, x_common[i] - x_j)
                )
                for i in range(N_MODELS)
            )
            x_mix.append(x_j)
            P_mix.append(P_j)

        return x_mix, P_mix

    def _filter(self, z: np.ndarray,
                x_mix: list[np.ndarray],
                P_mix: list[np.ndarray]) -> np.ndarray:
        """
        Step 2 — Model-matched Kalman filtering.

        Each model runs its own KF starting from the mixed initial condition
        (projected back to the model's native space).

        Returns
        -------
        likelihoods : (N_MODELS,)  Gaussian likelihood Λ_j(k)
        """
        likelihoods = np.zeros(N_MODELS)

        for j in range(N_MODELS):
            # Project mixed state back to model's native space
            x0 = self._from_common(j, x_mix[j])
            P0 = self._cov_from_common(j, P_mix[j])

            F_j, C_j, Q_j = self._model_matrices(j)

            # Prediction step
            x_pred = F_j @ x0
            P_pred = F_j @ P0 @ F_j.T + Q_j

            # Innovation
            y      = z - C_j @ x_pred
            S      = C_j @ P_pred @ C_j.T + self.R
            S_inv  = np.linalg.inv(S)

            # Kalman gain
            K = P_pred @ C_j.T @ S_inv

            # Update
            self.x_hat[j] = x_pred + K @ y
            self.P_hat[j] = (np.eye(len(x0)) - K @ C_j) @ P_pred

            # Likelihood (multivariate Gaussian)
            sign, log_det = np.linalg.slogdet(2 * np.pi * S)
            exponent = -0.5 * y @ S_inv @ y
            likelihoods[j] = np.exp(exponent - 0.5 * log_det)

        return likelihoods

    def _update_probs(self, likelihoods: np.ndarray):
        """
        Step 3 — Probability update via Bayes' rule + likelihood smoothing.
        """
        # Apply moving-average smoothing to likelihoods
        for j in range(N_MODELS):
            self.like_history[j].append(likelihoods[j])
        smooth_likes = np.array([
            np.mean(self.like_history[j]) for j in range(N_MODELS)
        ])

        c_bar = self.Pi.T @ self.mu
        new_mu = c_bar * smooth_likes
        total  = new_mu.sum()
        self.mu = new_mu / (total + 1e-12)

    def _fuse(self):
        """Fused position and velocity estimates  (Eq. 19)."""
        pos = np.zeros(3)
        vel = np.zeros(3)
        for j in range(N_MODELS):
            x_c = self._to_common(j, self.x_hat[j])
            pos += self.mu[j] * x_c[:3]
            vel += self.mu[j] * x_c[3:6]
        return pos, vel

    # ── Model matrices ────────────────────────────────────────────────────────────

    def _model_matrices(self, j: int):
        """Return (F, C, Q) for model j in its NATIVE space."""
        dt = self.dt
        if j == CV:
            F = np.block([
                [np.eye(3), dt * np.eye(3)],
                [np.zeros((3, 3)), np.eye(3)]
            ])                                         # (6,6)
            C = np.hstack([np.eye(3), np.zeros((3, 3))])  # (3,6)
            Q = self.Q_cv
        elif j == CA:
            F = np.block([
                [np.eye(3), dt*np.eye(3), 0.5*dt**2*np.eye(3)],
                [np.zeros((3,3)), np.eye(3), dt*np.eye(3)],
                [np.zeros((3,3)), np.zeros((3,3)), np.eye(3)],
            ])                                         # (9,9)
            C = np.hstack([np.eye(3), np.zeros((3,6))])   # (3,9)
            Q = self.Q_ca
        else:  # CT
            F = self._ct_transition(self.omega)        # (7,7)
            C = np.zeros((3, DIM_CT))
            C[0, 0] = 1; C[1, 2] = 1; C[2, 4] = 1   # extract [px,py,pz]
            Q = self.Q_ct
        return F, C, Q

    def _ct_transition(self, omega: float) -> np.ndarray:
        """CT transition matrix F3(ω) — Eq. (17).  State = [px,vx,py,vy,pz,vz,ω]"""
        dt = self.dt
        if abs(omega) < 1e-4:
            # Degenerate to CV in x-y + linear in z
            omega = 1e-4
        s, c = np.sin(omega * dt), np.cos(omega * dt)
        ow   = omega
        F = np.zeros((DIM_CT, DIM_CT))
        # x-axis block
        F[0, 0] = 1;  F[0, 1] = s/ow;    F[0, 2] = 0;  F[0, 3] = -(1-c)/ow
        F[1, 0] = 0;  F[1, 1] = c;       F[1, 2] = 0;  F[1, 3] = -s
        # y-axis block
        F[2, 0] = 0;  F[2, 1] = (1-c)/ow; F[2, 2] = 1; F[2, 3] = s/ow
        F[3, 0] = 0;  F[3, 1] = s;        F[3, 2] = 0; F[3, 3] = c
        # z-axis block (linear)
        F[4, 4] = 1;  F[4, 5] = dt
        F[5, 5] = 1
        # Turn rate (constant)
        F[6, 6] = 1
        return F

    def _estimate_turn_rate(self):
        """Estimate ω from position history using heading differences."""
        pos_hist = list(self.pos_history)
        if len(pos_hist) < 3:
            self.omega = 0.0
            return
        headings = []
        for i in range(1, len(pos_hist)):
            dp = pos_hist[i][:2] - pos_hist[i-1][:2]
            if np.linalg.norm(dp) > 1e-4:
                headings.append(np.arctan2(dp[1], dp[0]))
        if len(headings) < 2:
            self.omega = 0.0
            return
        dh = np.diff(headings)
        dh = (dh + np.pi) % (2 * np.pi) - np.pi   # wrap to [-π, π]
        self.omega = float(np.mean(dh) / self.dt)

    # ── State-space projections (native ↔ common 9-D) ───────────────────────────

    def _to_common(self, j: int, x: np.ndarray) -> np.ndarray:
        """Map model j native state to 9-D common space."""
        xc = np.zeros(DIM_COMMON)
        if j == CV:
            xc[:3] = x[:3]; xc[3:6] = x[3:6]
        elif j == CA:
            xc = x.copy()
        else:  # CT: [px,vx,py,vy,pz,vz,ω] → [px,py,pz,vx,vy,vz,ax,ay,az]
            xc[0] = x[0]; xc[1] = x[2]; xc[2] = x[4]   # pos
            xc[3] = x[1]; xc[4] = x[3]; xc[5] = x[5]   # vel
            # Centripetal acceleration: a = ω × v (approximate)
            omega = x[6]
            xc[6] = -omega * x[3]    # ax ≈ -ω·vy
            xc[7] =  omega * x[1]    # ay ≈  ω·vx
            xc[8] = 0.0
        return xc

    def _from_common(self, j: int, xc: np.ndarray) -> np.ndarray:
        """Project 9-D common state back to model j native space."""
        if j == CV:
            x = np.zeros(DIM_CV)
            x[:3] = xc[:3]; x[3:6] = xc[3:6]
        elif j == CA:
            x = xc.copy()
        else:  # CT
            x = np.zeros(DIM_CT)
            x[0] = xc[0]; x[2] = xc[1]; x[4] = xc[2]   # pos
            x[1] = xc[3]; x[3] = xc[4]; x[5] = xc[5]   # vel
            x[6] = self.omega
        return x

    def _cov_to_common(self, j: int, P: np.ndarray) -> np.ndarray:
        """Embed model j covariance into 9×9 common covariance."""
        Pc = np.zeros((DIM_COMMON, DIM_COMMON))
        T  = self._projection_matrix(j)
        Pc = T @ P @ T.T
        return Pc

    def _cov_from_common(self, j: int, Pc: np.ndarray) -> np.ndarray:
        """Project 9×9 common covariance to model j native space."""
        T = self._projection_matrix(j)
        return T.T @ Pc @ T + 1e-6 * np.eye(self._dim(j))

    def _projection_matrix(self, j: int) -> np.ndarray:
        """Linear projection T_j: native_dim → 9."""
        d = self._dim(j)
        T = np.zeros((DIM_COMMON, d))
        if j == CV:
            T[0, 0] = 1; T[1, 1] = 1; T[2, 2] = 1   # pos
            T[3, 3] = 1; T[4, 4] = 1; T[5, 5] = 1   # vel
        elif j == CA:
            T = np.eye(DIM_COMMON)
        else:  # CT
            T[0, 0] = 1; T[1, 2] = 1; T[2, 4] = 1   # pos (px,py,pz)
            T[3, 1] = 1; T[4, 3] = 1; T[5, 5] = 1   # vel (vx,vy,vz)
            T[6, 1] = -1  # crude acceleration from vx column (sign of centripetal)
            T[7, 3] =  1
        return T

    def _dim(self, j: int) -> int:
        return {CV: DIM_CV, CA: DIM_CA, CT: DIM_CT}[j]

    def _extract_pos(self, j: int, x: np.ndarray) -> np.ndarray:
        """Extract 3-D position from model j native state."""
        idx = {CV: [0,1,2], CA: [0,1,2], CT: [0,2,4]}[j]
        return x[idx]

    def _propagate_model(self, j: int, steps: int) -> np.ndarray:
        """Propagate model j state forward `steps` steps."""
        F, _, _ = self._model_matrices(j)
        x = self.x_hat[j].copy()
        for _ in range(steps):
            x = F @ x
        return x

    # ── Diagnostics ─────────────────────────────────────────────────────────────

    @property
    def mode_probabilities(self) -> np.ndarray:
        """Current mode probabilities [μ_CV, μ_CA, μ_CT]."""
        return self.mu.copy()

    @property
    def estimated_velocity(self) -> np.ndarray:
        """Fused velocity estimate (3,)."""
        return self.fused_vel.copy()
