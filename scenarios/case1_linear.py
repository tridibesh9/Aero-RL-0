"""
scenarios/case1_linear.py
──────────────────────────
Case 1: Obstacle with constant linear velocity.

The obstacle moves at a fixed velocity and is initially placed on (or near)
the straight-line path between the UAV start and goal positions.
"""

import numpy as np

# ── Scenario geometry ─────────────────────────────────────────────────────────
START_POS = np.array([0.0,  0.0,  1.0])   # UAV spawn  (also env INIT_XYZ)
GOAL_POS  = np.array([5.0,  0.0,  2.0])   # navigation target
OBS_INIT  = np.array([2.5,  0.0,  1.5])   # obstacle start (on the direct path)
OBS_VEL   = np.array([0.0,  0.8,  0.0])   # constant velocity (m/s)

SCENARIO_CONFIG = {
    "name":            "Case 1 — Linear Obstacle",
    "start_pos":       START_POS,
    "goal_pos":        GOAL_POS,
    "obs_init_pos":    OBS_INIT,
    "obs_velocity":    OBS_VEL,
    "description":     "Constant-velocity spherical obstacle on the optimal path.",
}


def obstacle_policy(t: float, p_obs: np.ndarray) -> np.ndarray:
    """
    Move the obstacle at constant velocity.

    Parameters
    ----------
    t     : current simulation time (s)
    p_obs : current obstacle position (unused; we integrate from init)

    Returns
    -------
    new_pos : (3,) updated obstacle position
    """
    return OBS_INIT + OBS_VEL * t
