"""
controllers/hierarchical_ctrl.py
─────────────────────────────────
Top-level controller combining:
    Upper layer  →  trained PPO agent  (Sec. 3.1)
    Lower layer  →  IMM predictor + ACBF filter  (Sec. 3.2-3.3)

Control loop (one step):
    1. Observe UAV state x(k) = [p(k), v(k)]
    2. Observe raw obstacle position z_obs(k)
    3. IMM.update(z_obs)         → p̂_o(k)  [fused estimate]
    4. IMM.predict(N)            → {p̂_o(k+i|k)}  [trajectory]
    5. PPO.predict(obs)          → u_f(k)   [reference acceleration]
    6. ACBF.filter(u_f, state, traj) → u(k)  [safe acceleration]
    7. Env receives u(k) as the action
"""

from __future__ import annotations
import numpy as np
from stable_baselines3 import PPO

from controllers.imm_predictor import IMMPredictor
from controllers.acbf_filter   import ACBFFilter


class HierarchicalController:
    """
    Hierarchical safe RL controller (inference only).

    Parameters
    ----------
    ppo_model_path : str   path to a saved SB3 PPO .zip
    config         : dict  loaded from configs/params.yaml
    """

    def __init__(self, ppo_model_path: str, config: dict):
        self.cfg  = config
        self.k_ma = config['k_ma']
        self.dt   = config['dt']

        print(f"[HierCtrl] Loading PPO from {ppo_model_path}")
        self.ppo  = PPO.load(ppo_model_path, device="cpu")

        self.imm  = IMMPredictor(config)
        self.acbf = ACBFFilter(config)

        self._initialized = False

    def reset(self, init_obs_pos: np.ndarray):
        """Call at the start of each evaluation episode."""
        self.imm.reset(init_obs_pos)
        self.acbf.reset()
        self._initialized = True

    def act(self,
            p_uav: np.ndarray,
            v_uav: np.ndarray,
            target_pos: np.ndarray,
            z_obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the safe control acceleration for one step.

        Parameters
        ----------
        p_uav      : (3,)  current UAV position
        v_uav      : (3,)  current UAV velocity
        target_pos : (3,)  goal position
        z_obs      : (3,)  raw obstacle position measurement

        Returns
        -------
        u_safe  : (3,)  safe acceleration (ready for env._preprocessAction)
        obs_traj: (N+1,3) predicted obstacle trajectory (for logging)
        """
        if not self._initialized:
            self.reset(z_obs)

        # ── IMM update & predict ─────────────────────────────────────────────
        self.imm.update(z_obs)
        obs_traj = self.imm.predict()            # (N+1, 3)
        v_obs    = self.imm.estimated_velocity   # (3,)

        # ── PPO reference ────────────────────────────────────────────────────
        err     = target_pos - p_uav
        ppo_obs = np.array([*v_uav, *err], dtype=np.float32)
        a_norm, _  = self.ppo.predict(ppo_obs, deterministic=True)
        a_norm     = np.array(a_norm).flatten()[:3]  # ensure (3,) regardless of SB3 output shape
        u_ref      = self.k_ma * np.clip(a_norm, -1.0, 1.0)

        # ── ACBF safety filter ───────────────────────────────────────────────
        u_safe = self.acbf.filter(
            u_ref=u_ref,
            p_uav=p_uav,
            v_uav=v_uav,
            obs_traj=obs_traj,
            v_obs=v_obs,
        )

        return u_safe, obs_traj

    def act_normalized(self,
                       p_uav, v_uav, target_pos, z_obs
                       ) -> tuple[np.ndarray, np.ndarray]:
        """
        Like act() but returns the action NORMALIZED to [-1,1] so it can be
        passed directly to UAVNavEnv.step().
        """
        u_safe, obs_traj = self.act(p_uav, v_uav, target_pos, z_obs)
        a_norm = np.clip(u_safe / self.k_ma, -1.0, 1.0)
        return a_norm.reshape(1, 3).astype(np.float32), obs_traj


# ── Simple PID baseline controller (for PID+ACBF comparison) ─────────────────

class PIDController:
    """
    Proportional-Integral-Derivative position controller.
    Used as the 'PID+ACBF' baseline.

    Outputs reference acceleration in the inertial frame.
    """

    def __init__(self, config: dict):
        self.dt  = config['dt']
        self.kp  = np.array(config['pid_pos_kp'])
        self.ki  = np.array(config['pid_pos_ki'])
        self.kd  = np.array(config['pid_pos_kd'])
        self.k_ma= config['k_ma']
        self._int_e  = np.zeros(3)
        self._prev_e = None

    def reset(self):
        self._int_e  = np.zeros(3)
        self._prev_e = None

    def act(self, p_uav: np.ndarray, v_uav: np.ndarray,
            target_pos: np.ndarray) -> np.ndarray:
        """Returns reference acceleration u_ref (m/s²)."""
        e = target_pos - p_uav
        self._int_e  += e * self.dt
        d_e = (e - self._prev_e) / self.dt if self._prev_e is not None else np.zeros(3)
        self._prev_e = e.copy()
        u_ref = self.kp * e + self.ki * self._int_e + self.kd * d_e
        return np.clip(u_ref, -self.k_ma, self.k_ma)


class PIDWithACBF:
    """PID + ACBF safety filter baseline controller."""

    def __init__(self, config: dict):
        self.pid  = PIDController(config)
        self.imm  = IMMPredictor(config)
        self.acbf = ACBFFilter(config)
        self.k_ma = config['k_ma']

    def reset(self, init_obs_pos: np.ndarray):
        self.pid.reset()
        self.imm.reset(init_obs_pos)
        self.acbf.reset()

    def act(self, p_uav, v_uav, target_pos, z_obs) -> tuple[np.ndarray, np.ndarray]:
        self.imm.update(z_obs)
        obs_traj = self.imm.predict()
        v_obs    = self.imm.estimated_velocity
        u_ref    = self.pid.act(p_uav, v_uav, target_pos)
        u_safe   = self.acbf.filter(u_ref, p_uav, v_uav, obs_traj, v_obs)
        return u_safe, obs_traj

    def act_normalized(self, p_uav, v_uav, target_pos, z_obs):
        u_safe, obs_traj = self.act(p_uav, v_uav, target_pos, z_obs)
        a_norm = np.clip(u_safe / self.k_ma, -1.0, 1.0)
        return a_norm.reshape(1, 3).astype(np.float32), obs_traj
