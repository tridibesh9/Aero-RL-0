"""
envs/UAVNavEnv.py
─────────────────
Custom Gymnasium environment for UAV navigation with dynamic obstacle support.

Architecture
────────────
• Inherits from gym_pybullet_drones BaseRLAviary (ActionType.PID, ObservationType.KIN)
• acceleration_v2: net acceleration commands with a 240 Hz attitude loop
• Scaled 19-D state includes velocity, goal error, attitude and controller memory
• waypoint_v1: explicit compatibility mode for historical six-input policies
• Paper-inspired reward with optional distance and speed costs
• Manages a single spherical obstacle (dynamic or static)

Notes
─────
The safety filter (ACBF) is NOT active during training — the PPO learns
unconstrained goal-reaching.  At evaluation time, the HierarchicalController
intercepts actions before they reach the env.
"""

from copy import deepcopy

import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary
from gym_pybullet_drones.utils.enums import (
    DroneModel, Physics, ActionType, ObservationType
)
from utils.config import command_bounds, validate_config
from controllers.acceleration_ctrl import acceleration_rpm


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
                 gui: bool = False,
                 start_pos=None):
        validate_config(config)
        self.cfg = config
        self.action_mode = config.get('action_mode', 'waypoint_v1')
        self.command_min, self.command_max = command_bounds(config)
        self.start_pos = np.asarray(self.INIT_XYZ[0] if start_pos is None else start_pos,
                                    dtype=float).reshape(3).copy()
        if not np.isfinite(self.start_pos).all() or np.any(np.abs(self.start_pos) > config['flight_bounds']) or self.start_pos[2] < 0.05:
            raise ValueError("Drone start must lie inside the flight volume")
        self.training_radius = None
        self._episode_radius = None
        self._forecast_env = None
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
        self._goal_steps = 0
        self.reward_components = {}
        self.reward_totals = {}

        ctrl_freq = config['ctrl_freq']

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=1,
            initial_xyzs=self.start_pos.reshape(1, 3),
            initial_rpys=self.INIT_RPY,
            physics=Physics.PYB,
            pyb_freq=config['pyb_freq'],
            ctrl_freq=ctrl_freq,
            gui=gui,
            record=False,
            obs=ObservationType.KIN,
            act=ActionType.PID,
        )
        if self.action_mode == 'acceleration_v2':
            self.ctrl[0].I_COEFF_TOR[:] = 0.
        for key, value in (("mass", self.M), ("g", self.G)):
            if key in config and not np.isclose(float(config[key]), value):
                self.close()
                raise ValueError(f"{key} is set by the CF2X simulator ({value})")

    # ── Spaces ───────────────────────────────────────────────────────────────────

    def _observationSpace(self):
        """
        Legacy: [velocity, goal error]. Acceleration v2 adds current RPY,
        angular velocity, previous controller RPY and absolute position.
        """
        size = 19 if self.action_mode == 'acceleration_v2' else 6
        lo = np.full(size, -np.inf, dtype=np.float32)
        hi = np.full(size,  np.inf, dtype=np.float32)
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    def _actionSpace(self):
        """Normalized net acceleration (v2), or legacy waypoint command."""
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
        """Build the versioned policy observation with fixed physical scales."""
        state = self._getDroneStateVector(0)
        vel   = state[10:13]                        # vx, vy, vz
        pos   = state[0:3]
        err   = self.target_pos - pos               # e(k) = p* - p
        if self.action_mode == 'acceleration_v2':
            # Fixed scaling and observed attitude/controller memory avoid a
            # changing policy interface or a hidden moving-position controller.
            return np.r_[vel / self.cfg['obs_velocity_scale'],
                         err / self.cfg['obs_error_scale'], state[7:10] / np.pi,
                         state[13:16] / self.cfg['obs_angular_scale'],
                         self.ctrl[0].last_rpy / np.pi,
                         pos / self.cfg['flight_bounds'],
                         min(1., self._goal_steps / self.cfg.get('goal_hold_steps', 1))].astype(np.float32)
        return np.array([*vel, *err], dtype=np.float32)

    # ── Action preprocessing ─────────────────────────────────────────────────────

    def _preprocessAction(self, action):
        """
        Legacy interface only: normalized PPO action → PID target position.

        a ∈ [-1,1]³ gives a bounded waypoint command u = k_ma * a.
        The waypoint offset in metres is u * [0.5, 0.5, 0.15].

        The PID controller then computes RPMs to reach target_pos.
        """
        self.action_buffer.append(action)

        k_ma = self.cfg['k_ma']

        state   = self._getDroneStateVector(0)
        cur_pos = state[0:3]
        cur_vel = state[10:13]
        cur_quat = state[3:7]
        cur_ang_vel = state[13:16]

        a = action[0]                               # (3,) normalized
        u_f = np.clip(k_ma * a, self.command_min, self.command_max)

        # Keep the waypoint interface used by the existing trained policies.
        lookahead = np.array([0.5, 0.5, 0.15])
        target_pos = cur_pos + u_f * lookahead

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
        speed = np.linalg.norm(state[10:13])
        inside = err < cfg['acceptance_radius'] and speed <= cfg.get('goal_speed_tolerance', np.inf)
        self._goal_steps = self._goal_steps + 1 if inside else 0
        r_tr = cfg['M_tr'] if self._is_success(pos) else 0.0

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

        r_distance = -cfg.get('distance_cost', 0.) * err * self.CTRL_TIMESTEP
        r_speed = -cfg.get('speed_cost', 0.) * speed**2 * self.CTRL_TIMESTEP
        self.reward_components = dict(terminal=r_tr, progress=r_ds, boundary=r_bp,
                                      time=r_tp, distance=r_distance, speed=r_speed)
        for name, value in self.reward_components.items():
            self.reward_totals[name] = self.reward_totals.get(name, 0.) + float(value)
        return float(sum(self.reward_components.values()))

    # ── Termination ──────────────────────────────────────────────────────────────

    def _computeTerminated(self):
        state = self._getDroneStateVector(0)
        pos   = state[0:3]
        return bool(self._is_success(pos) or self._is_failure(pos))

    def _is_failure(self, pos):
        return bool(np.any(np.abs(pos) > self.cfg['flight_bounds']) or pos[2] < 0.05)

    def _is_success(self, pos):
        return bool(not self._is_failure(pos)
                    and self._goal_steps >= self.cfg.get('goal_hold_steps', 1))

    def _computeTruncated(self):
        timeout = self._step_count >= self.cfg['max_episode_steps']
        return bool(timeout and not self._computeTerminated())

    def _computeInfo(self):
        state   = self._getDroneStateVector(0)
        pos     = state[0:3]
        vel     = state[10:13]
        dist_to_obs = float(np.linalg.norm(pos - self.obs_pos))
        goal_error = float(np.linalg.norm(self.target_pos - pos))
        reached = self._is_success(pos)
        if reached:
            reason = "goal"
        elif np.any(np.abs(pos) > self.cfg['flight_bounds']) or pos[2] < 0.05:
            reason = "out_of_bounds"
        elif self._step_count >= self.cfg['max_episode_steps']:
            reason = "timeout"
        else:
            reason = None
        return {
            'pos':          pos.copy(),
            'vel':          vel.copy(),
            'obs_pos':      self.obs_pos.copy(),
            'dist_to_obs':  dist_to_obs,
            'step':         self._step_count,
            't':            self._t,
            'goal_error':   goal_error,
            'goal_speed':   float(np.linalg.norm(vel)),
            'reward_components': self.reward_components.copy(),
            'reward_totals': self.reward_totals.copy(),
            'is_success':   reached,
            'termination_reason': reason,
            'command':      self._prev_u.copy(),
            'episode_radius': self._episode_radius,
        }

    # ── Reset & obstacle management ──────────────────────────────────────────────

    def set_training_radius(self, radius):
        """Apply a new goal sampling stage at the next episode reset."""
        if radius is not None and (not np.isfinite(radius) or radius <= self.cfg['acceptance_radius']):
            raise ValueError("Training radius must exceed the goal tolerance")
        self.training_radius = radius

    def reset(self, seed=None, options=None):
        # Reset both the physics world and its controller. Otherwise the next
        # episode starts with motor integral and attitude errors from the last.
        super().reset(seed=seed, options=options)
        for motor_controller in self.ctrl:
            motor_controller.reset()
        # Randomize goal and obstacle if not fixed
        rng = self.np_random

        options = options or {}
        start = self.start_pos
        self._episode_radius = self.training_radius
        if 'target_pos' in options:
            self.target_pos = np.asarray(options['target_pos'], dtype=float).reshape(3).copy()
        elif self._target_pos_fixed is not None:
            self.target_pos = self._target_pos_fixed.copy()
        elif self.training_radius is not None:
            # Uniform directions include goals below the initial altitude.
            for _ in range(256):
                direction = rng.normal(size=3)
                length = rng.uniform(max(self.training_radius * 0.5, self.cfg['acceptance_radius']),
                                     self.training_radius)
                self.target_pos = start + length * direction / np.linalg.norm(direction)
                if 0.3 < self.target_pos[2] < self.cfg['flight_bounds'] and np.all(np.abs(self.target_pos) < self.cfg['flight_bounds']):
                    break
            else:
                raise ValueError("Cannot sample a curriculum goal within the flight volume")
        else:
            # Random goal in flight envelope, at least 2 m from start
            width = min(5.0, self.cfg['flight_bounds'])
            for _ in range(256):
                self.target_pos = rng.uniform(-width, width, size=3)
                self.target_pos[2] = rng.uniform(min(0.5, width), min(3.0, width))
                if np.linalg.norm(self.target_pos - start) > 2.0:
                    break
            else:
                raise ValueError("Cannot sample a goal two metres from the start within the flight volume")

        if (not np.isfinite(self.target_pos).all() or self.target_pos[2] < 0.05
                or np.any(np.abs(self.target_pos) > self.cfg['flight_bounds'])):
            raise ValueError("Goal must lie inside the flight volume")

        if self._obs_pos_fixed is not None:
            self.obs_pos = self._obs_pos_fixed.copy()
        elif self.training_radius is not None or self.cfg.get('training_obstacle', 'static') == 'distant':
            self.obs_pos = np.full(3, self.cfg['flight_bounds'] + 2.0)
        else:
            # Place static obstacle roughly between start and goal
            self.obs_pos = start + (self.target_pos - start) * rng.uniform(0.3, 0.7) + rng.uniform(-0.3, 0.3, 3)
            self.obs_pos[2] = max(0.3, self.obs_pos[2])

        self._prev_dist  = float(np.linalg.norm(self.target_pos - start))
        self._step_count = 0
        self._t          = 0.0
        self._prev_u     = np.zeros(3)
        self._goal_steps = 0
        self.reward_components = {}
        self.reward_totals = {}

        # Spawn obstacle sphere in PyBullet
        self._spawn_obstacle()

        return self._computeObs(), self._computeInfo()

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

        if self.action_mode == 'waypoint_v1':
            return super().step(action)
        action = np.asarray(action, dtype=float)
        if action.shape != (1, 3) or not np.isfinite(action).all():
            raise ValueError('Action must be a finite (1, 3) array')
        self._prev_u = np.clip(self.cfg['k_ma'] * action[0], self.command_min, self.command_max)
        self.action_buffer.append(action.copy())
        for _ in range(self.PYB_STEPS_PER_CTRL):
            self._updateAndStoreKinematicInformation()
            state = self._getDroneStateVector(0)
            rpm = acceleration_rpm(self.ctrl[0], self._prev_u, state[3:7],
                                   self.M, self.G, self.PYB_TIMESTEP)
            self._physics(rpm, 0)
            p.stepSimulation(physicsClientId=self.CLIENT)
            self.last_clipped_action = rpm.reshape(1, 4)
        self.step_counter += self.PYB_STEPS_PER_CTRL
        self._updateAndStoreKinematicInformation()
        reward = self._computeReward()
        return self._computeObs(), reward, self._computeTerminated(), self._computeTruncated(), self._computeInfo()

    # ── Accessors for hierarchical controller ────────────────────────────────────

    def get_uav_state(self):
        """Return (pos, vel) as two (3,) arrays."""
        state = self._getDroneStateVector(0)
        return state[0:3].copy(), state[10:13].copy()

    def get_obs_pos(self) -> np.ndarray:
        return self.obs_pos.copy()

    def predict_positions(self, command, steps):
        """Free-flight response to a held command in an independent DIRECT world.

        The forecast uses the live pose, velocities and attitude controller memory.
        Obstacle motion is supplied independently by the IMM to the barrier.
        """
        if self._forecast_env is None:
            self._forecast_env = UAVNavEnv(self.cfg, start_pos=self.start_pos)
        shadow = self._forecast_env
        state = self._getDroneStateVector(0)
        body = shadow.DRONE_IDS[0]
        p.resetBasePositionAndOrientation(body, state[:3], state[3:7], physicsClientId=shadow.CLIENT)
        p.resetBaseVelocity(body, state[10:13], state[13:16], physicsClientId=shadow.CLIENT)
        shadow._updateAndStoreKinematicInformation()
        shadow.ctrl[0] = deepcopy(self.ctrl[0])
        shadow.last_clipped_action = self.last_clipped_action.copy()
        shadow.step_counter = self.step_counter
        shadow._step_count, shadow._t = self._step_count, self._t
        normalized = (np.asarray(command) / self.cfg['k_ma']).reshape(1, 3).astype(np.float32)
        positions = np.empty((steps + 1, 3))
        positions[0] = state[:3]
        for i in range(1, steps + 1):
            _, _, _, _, info = shadow.step(normalized)
            positions[i] = info['pos']
        return positions

    def close(self):
        if self._forecast_env is not None:
            self._forecast_env.close()
            self._forecast_env = None
        super().close()

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
