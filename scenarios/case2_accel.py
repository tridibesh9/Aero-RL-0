"""
scenarios/case2_accel.py
─────────────────────────
Case 2: Obstacle with constant acceleration (linearly increasing speed).
"""

import numpy as np

START_POS = np.array([0.0,  0.0,  1.0])
GOAL_POS  = np.array([5.0,  0.0,  2.0])
OBS_INIT  = np.array([2.5, -1.5,  1.5])   # starts off-path, accelerates across
OBS_INIT_VEL = np.array([0.0,  0.5,  0.0])
OBS_ACCEL    = np.array([0.0,  0.4,  0.0])   # constant acceleration (m/s²)

SCENARIO_CONFIG = {
    "name":            "Case 2 — Accelerating Obstacle",
    "start_pos":       START_POS,
    "goal_pos":        GOAL_POS,
    "obs_init_pos":    OBS_INIT,
    "obs_velocity":    OBS_INIT_VEL,
    "description":     "Obstacle with constant acceleration along a linear path.",
}


def obstacle_policy(t: float, p_obs: np.ndarray) -> np.ndarray:
    """
    p(t) = p0 + v0·t + ½·a·t²   (kinematic motion with constant acceleration)
    """
    return OBS_INIT + OBS_INIT_VEL * t + 0.5 * OBS_ACCEL * t ** 2
