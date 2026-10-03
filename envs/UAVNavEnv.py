"""
envs/UAVNavEnv.py
─────────────────
Custom Gymnasium environment for UAV navigation with dynamic obstacle support.

Architecture
────────────
• Inherits from gym_pybullet_drones BaseRLAviary (ActionType.PID, ObservationType.KIN)
• Overrides obs space → 6-D  [v(k)ᵀ, e(k)ᵀ]  (Sec. 3.1)
• Overrides action space → 3-D normalized acceleration  a ∈ [-1,1]³
• Overrides _preprocessAction → converts acceleration to PID target position
• Overrides reward → composite r(k) = r_tr + r_ds + r_bp + r_tp  (Eqs. 9-13)
• Manages a single spherical obstacle (dynamic or static)

Notes
─────
The safety filter (ACBF) is NOT active during training — the PPO learns
unconstrained goal-reaching.  At evaluation time, the HierarchicalController
intercepts actions before they reach the env.
"""

import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    DroneModel, Physics, ActionType, ObservationType
)


class UAVNavEnv(BaseRLAviary):
    """
    Single-UAV goal-reaching environment with one spherical obstacle.

    Parameters
    ----------
    config : dict
        Loaded from configs/params.yaml.
    obstacle_policy : callable | None
        f(t, p_obs) → new_p_obs.  None = static obstacle.
    target_pos : array-like (3,) | None
        Fixed goal; if None a random goal is sampled each reset.
    obstacle_init_pos : array-like (3,) | None
        Fixed obstacle start; if None random each reset.
    gui : bool
        Open PyBullet GUI (slow; use only for visual inspection).
    """

    # ── Defaults ────────────────────────────────────────────────────────────────
    INIT_XYZ   = np.array([[0.0, 0.0, 1.0]])   # drone start (1 drone × 3)
    INIT_RPY   = np.array([[0.0, 0.0, 0.0]])

    def __init__(self,
                 config: dict,
                 obstacle_policy=None,
                 target_pos=None,
                 obstacle_init_pos=None,
                 gui: bool = False):
        self.cfg = config
        self.obstacle_policy    = obstacle_policy
        self._target_pos_fixed  = (np.array(target_pos, dtype=np.float64)
                                   if target_pos is not None else None)
        self._obs_pos_fixed     = (np.array(obstacle_init_pos, dtype=np.float64)
                                   if obstacle_init_pos is not None else None)

        # These will be set in reset()
        self.target_pos   : np.ndarray = np.zeros(3)
        self.obs_pos      : np.ndarray = np.zeros(3)
        self.obstacle_id  : int        = -1
        self._prev_dist   : float      = np.inf
        self._step_count  : int        = 0
        self._t           : float      = 0.0
        self._prev_u      : np.ndarray = np.zeros(3)

        ctrl_freq = config['ctrl_freq']

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=self.INIT_XYZ,
            initial_rpys=self.INIT_RPY,
            physics=Physics.PYB,
            pyb_freq=config['pyb_freq'],
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=False,
            obs=ObservationType.KIN,
            act=ActionType.PID,
        )

    # ── Spaces ───────────────────────────────────────────────────────────────────

    def _observationSpace(self):
        """
        6-D observation:  s(k) = [v(k)ᵀ, e(k)ᵀ]  ∈ ℝ⁶

        v(k) = velocity (3D)
        e(k) = p*(k) - p(k) = position error (3D)
        """
        lo = np.full(6, -np.inf, dtype=np.float32)
        hi = np.full(6,  np.inf, dtype=np.float32)
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def _actionSpace(self):
        """3-D normalized acceleration  a ∈ [-1, 1]³."""
        lo = np.full((1, 3), -1.0, dtype=np.float32)
        hi = np.full((1, 3),  1.0, dtype=np.float32)
        # Initialise action buffer with zeros (required by BaseRLAviary)
        from collections import deque
        self.ACTION_BUFFER_SIZE = int(self.CTRL_FREQ // 2)
        self.action_buffer = deque(maxlen=self.ACTION_BUFFER_SIZE)
        for _ in range(self.ACTION_BUFFER_SIZE):
            self.action_buffer.append(np.zeros((1, 3)))
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    # ── Observation ──────────────────────────────────────────────────────────────

    def _computeObs(self):
        """Return 6-D state [vx,vy,vz, ex,ey,ez]."""
        state = self._getDroneStateVector(0)
        vel   = state[10:13]                        # vx, vy, vz
        pos   = state[0:3]
        err   = self.target_pos - pos               # e(k) = p* - p
        return np.array([*vel, *err], dtype=np.float32)

    # ── Action preprocessing ─────────────────────────────────────────────────────

    def _preprocessAction(self, action):
        """
        Convert normalized PPO action → PID target position.

        a ∈ [-1,1]³  →  u_f = k_ma · a  (reference acceleration, Eq. 8)
        target_pos = p(k) + v(k)·Δt + u_f·Δt²   (1-step kinematic look-ahead)

        The PID controller then computes RPMs to reach target_pos.
        """
        self.action_buffer.append(action)

        k_ma = self.cfg['k_ma']
        dt   = self.CTRL_TIMESTEP

        state   = self._getDroneStateVector(0)
        cur_pos = state[0:3]
        cur_vel = state[10:13]
        cur_quat = state[3:7]
        cur_ang_vel = state[13:16]

        a = action[0]                               # (3,) normalized
        u_f = k_ma * np.clip(a, -1.0, 1.0)         # reference acceleration

        # DSLPIDControl in gym-pybullet-drones already adds [0, 0, mg] internally
        # for full gravity compensation to maintain hover at target_pos.
        # u_f is the commanded maneuvering acceleration (hover at u_f=0).
        net_acc = u_f

        # Kinematic look-ahead to compute PID waypoint
        target_pos = cur_pos + cur_vel * dt + 0.5 * net_acc * (dt ** 2)

        # Clip to flight bounds
        bound = self.cfg['flight_bounds']
        target_pos = np.clip(target_pos, -bound, bound)

        # Store last control for logging
        self._prev_u = u_f.copy()

        # Compute RPMs via DSL PID controller
        rpm, _, _ = self.ctrl[0].computeControl(
            control_timestep=self.CTRL_TIMESTEP,
            cur_pos=cur_pos,
            cur_quat=cur_quat,
            cur_vel=cur_vel,
            cur_ang_vel=cur_ang_vel,
            target_pos=target_pos,
        )
        return rpm.reshape(1, 4)

    # ── Reward ───────────────────────────────────────────────────────────────────

    def _computeReward(self):
        """
        Composite reward  r(k) = r_tr + r_ds + r_bp + r_tp   (Eq. 9)
        """
        cfg = self.cfg
        state = self._getDroneStateVector(0)
        pos   = state[0:3]
        err   = np.linalg.norm(self.target_pos - pos)

        # r_tr: terminal reward (Eq. 10)
        r_tr = cfg['M_tr'] if err < cfg['acceptance_radius'] else 0.0

        # r_ds: distance-shaping reward (Eq. 11)
        prev_dist = self._prev_dist if np.isfinite(self._prev_dist) else err
        r_ds = cfg['M_ds'] * (prev_dist - err)
        self._prev_dist = err

        # r_bp: boundary penalty (Eq. 12)
        bound = cfg['flight_bounds']
        out_of_bounds = np.any(np.abs(pos) > bound) or pos[2] < 0.05
        r_bp = -cfg['M_bp'] if out_of_bounds else 0.0

        # r_tp: time penalty (Eq. 13)
        r_tp = -cfg['M_tp'] * self.CTRL_TIMESTEP

        return float(r_tr + r_ds + r_bp + r_tp)

    # ── Termination ──────────────────────────────────────────────────────────────

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        pos   = state[0:3]
        err   = np.linalg.norm(self.target_pos - pos)
        return bool(err < self.cfg['acceptance_radius'])

    def _computeTruncated(self):
        state = self._getDroneStateVector(0)
        pos   = state[0:3]
        bound = self.cfg['flight_bounds']
        oob   = np.any(np.abs(pos) > bound) or pos[2] < 0.05
        timeout = self._step_count >= self.cfg['max_episode_steps']
        return bool(oob or timeout)

    def _computeInfo(self):
        state   = self._getDroneStateVector(0)
        pos     = state[0:3]
        vel     = state[10:13]
        dist_to_obs = float(np.linalg.norm(pos - self.obs_pos))
        return {
            'pos':          pos.copy(),
            'vel':          vel.copy(),
            'obs_pos':      self.obs_pos.copy(),
            'dist_to_obs':  dist_to_obs,
            'step':         self._step_count,
            't':            self._t,
        }

    # ── Reset & obstacle management ──────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        # Randomize goal and obstacle if not fixed
        rng = np.random.default_rng(seed)

        if self._target_pos_fixed is not None:
            self.target_pos = self._target_pos_fixed.copy()
        else:
            # Random goal in flight envelope, at least 2 m from start
            while True:
                self.target_pos = rng.uniform(-6, 6, size=3)
                self.target_pos[2] = rng.uniform(0.5, 5.0)
                if np.linalg.norm(self.target_pos) > 2.0:
                    break

        if self._obs_pos_fixed is not None:
            self.obs_pos = self._obs_pos_fixed.copy()
        else:
            # Place static obstacle roughly between start and goal
            self.obs_pos = self.target_pos * rng.uniform(0.3, 0.7) + rng.uniform(-0.5, 0.5, 3)
            self.obs_pos[2] = max(0.3, self.obs_pos[2])

        self._prev_dist  = np.linalg.norm(self.target_pos)
        self._step_count = 0
        self._t          = 0.0
        self._prev_u     = np.zeros(3)

        obs, info = super().reset(seed=seed, options=options)

        # Spawn obstacle sphere in PyBullet
        self._spawn_obstacle()

        return obs, info

    def step(self, action):
        self._step_count += 1
        self._t += self.CTRL_TIMESTEP

        # Move obstacle if a policy is provided
        if self.obstacle_policy is not None:
            self.obs_pos = self.obstacle_policy(self._t, self.obs_pos)
            if self.obstacle_id >= 0:
                p.resetBasePositionAndOrientation(
                    self.obstacle_id,
                    self.obs_pos.tolist(),
                    [0, 0, 0, 1],
                    physicsClientId=self.CLIENT
                )

        return super().step(action)

    # ── Accessors for hierarchical controller ────────────────────────────────────

    def get_uav_state(self):
        """Return (pos, vel) as two (3,) arrays."""
        state = self._getDroneStateVector(0)
        return state[0:3].copy(), state[10:13].copy()

    def get_obs_pos(self) -> np.ndarray:
        return self.obs_pos.copy()

    # ── Helpers ──────────────────────────────────────────────────────────────────

    def _spawn_obstacle(self):
        """Create / replace the spherical obstacle in the PyBullet world."""
        r = self.cfg['obs_radius']
        col_id = p.createCollisionShape(
            p.GEOM_SPHERE, radius=r, physicsClientId=self.CLIENT)
        vis_id = p.createVisualShape(
            p.GEOM_SPHERE, radius=r,
            rgbaColor=[1.0, 0.3, 0.3, 0.8],
            physicsClientId=self.CLIENT)
        self.obstacle_id = p.createMultiBody(
            baseMass=0,                         # kinematic body (no physics)
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=self.obs_pos.tolist(),
            physicsClientId=self.CLIENT
        )

    # ── Unused base methods we must stub ─────────────────────────────────────────

    def _addObstacles(self):
        """Obstacle is added in reset() → suppress base-class version."""
        pass
