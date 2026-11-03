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
    5. PPO.predict(obs)          → u_f(k)   [reference waypoint command]
    6. ACBF.filter(u_f, state, traj) → u(k)  [bounded waypoint command]
    7. Env receives u(k) as the action
"""

from __future__ import annotations
import numpy as np
from stable_baselines3 import PPO

from controllers.imm_predictor import IMMPredictor
from controllers.acbf_filter   import ACBFFilter


def _waypoint_rollout(override, configured):
    rollout = configured if override is None else override
    if rollout is None:
        raise ValueError("Supply predict_positions=env.predict_positions for waypoint-controlled flight")
    return rollout


class HierarchicalController:
    """
    Hierarchical safe RL controller (inference only).

    Parameters
    ----------
    ppo_model_path : str   path to a saved SB3 PPO .zip
    config         : dict  loaded from configs/params.yaml
    """

    def __init__(self, ppo_model_path: str | PPO, config: dict, *, predict_positions=None):
        self.cfg  = config
        self.k_ma = config['k_ma']
        self.dt   = config['dt']
        self.predict_positions = predict_positions

        self.ppo = (ppo_model_path if isinstance(ppo_model_path, PPO)
                    else PPO.load(ppo_model_path, device="cpu"))

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
            z_obs: np.ndarray, *, predict_positions=None) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the filtered waypoint command for one step.

        Parameters
        ----------
        p_uav      : (3,)  current UAV position
        v_uav      : (3,)  current UAV velocity
        target_pos : (3,)  goal position
        z_obs      : (3,)  raw obstacle position measurement

        Returns
        -------
        u_safe  : (3,)  bounded command in the environment's k_ma scale
        obs_traj: (N+1,3) predicted obstacle trajectory (for logging)
        """
        predict_positions = _waypoint_rollout(predict_positions, self.predict_positions)
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
            predict_positions=predict_positions,
        )

        return u_safe, obs_traj

    def act_normalized(self,
                       p_uav, v_uav, target_pos, z_obs, *, predict_positions=None
                       ) -> tuple[np.ndarray, np.ndarray]:
        """
        Like act() but returns the action NORMALIZED to [-1,1] so it can be
        passed directly to UAVNavEnv.step().
        """
        u_safe, obs_traj = self.act(p_uav, v_uav, target_pos, z_obs, predict_positions=predict_positions)
        a_norm = u_safe / self.k_ma
        return a_norm.reshape(1, 3).astype(np.float32), obs_traj


# ── Simple PID baseline controller (for PID+ACBF comparison) ─────────────────

class PIDController:
    """
    Proportional-Integral-Derivative position controller.
    Used as the 'PID+ACBF' baseline.

    Outputs a reference waypoint command in the inertial frame.
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
        """Return a bounded reference waypoint command."""
        e = target_pos - p_uav
        self._int_e  += e * self.dt
        d_e = (e - self._prev_e) / self.dt if self._prev_e is not None else np.zeros(3)
        self._prev_e = e.copy()
        u_ref = self.kp * e + self.ki * self._int_e + self.kd * d_e
        return np.clip(u_ref, -self.k_ma, self.k_ma)


class PIDWithACBF:
    """PID + ACBF safety filter baseline controller."""

    def __init__(self, config: dict, *, predict_positions=None):
        self.predict_positions = predict_positions
        self.pid  = PIDController(config)
        self.imm  = IMMPredictor(config)
        self.acbf = ACBFFilter(config)
        self.k_ma = config['k_ma']

    def reset(self, init_obs_pos: np.ndarray):
        self.pid.reset()
        self.imm.reset(init_obs_pos)
        self.acbf.reset()

    def act(self, p_uav, v_uav, target_pos, z_obs, *, predict_positions=None) -> tuple[np.ndarray, np.ndarray]:
        predict_positions = _waypoint_rollout(predict_positions, self.predict_positions)
        self.imm.update(z_obs)
        obs_traj = self.imm.predict()
        v_obs    = self.imm.estimated_velocity
        u_ref    = self.pid.act(p_uav, v_uav, target_pos)
        u_safe   = self.acbf.filter(u_ref, p_uav, v_uav, obs_traj, v_obs, predict_positions=predict_positions)
        return u_safe, obs_traj

    def act_normalized(self, p_uav, v_uav, target_pos, z_obs, *, predict_positions=None):
        u_safe, obs_traj = self.act(p_uav, v_uav, target_pos, z_obs, predict_positions=predict_positions)
        a_norm = u_safe / self.k_ma
        return a_norm.reshape(1, 3).astype(np.float32), obs_traj
