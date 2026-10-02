"""
scenarios/case3_figure8.py
───────────────────────────
Case 3: Obstacle following an "8"-shaped (lemniscate) trajectory.

Parametric figure-8 in the x-y plane centred at a midpoint:
    x(t) = cx + A · sin(ω·t)
    y(t) = cy + A · sin(ω·t) · cos(ω·t)     [= A/2 · sin(2ω·t)]
    z(t) = cz  (constant height)

This creates continuous, simultaneous changes in speed AND heading,
exercising the CT model in the IMM and the ACBF's real-time adaptation.
"""

import numpy as np

START_POS = np.array([0.0,  0.0,  1.0])
GOAL_POS  = np.array([5.0,  0.0,  2.0])

# Figure-8 centre (roughly in the middle of the path)
_CX, _CY, _CZ = 2.5, 0.0, 1.5
_A   = 1.5     # amplitude (m)
_OMG = 0.6     # angular frequency (rad/s)  → period ≈ 10 s

# Initial obstacle position (at t=0)
OBS_INIT = np.array([_CX, _CY, _CZ])

SCENARIO_CONFIG = {
    "name":            "Case 3 — Figure-8 Obstacle",
    "start_pos":       START_POS,
    "goal_pos":        GOAL_POS,
    "obs_init_pos":    OBS_INIT,
    "obs_velocity":    np.zeros(3),          # instantaneous vel at t=0
    "description":     "Obstacle on a figure-8 (lemniscate) trajectory.",
}


def obstacle_policy(t: float, p_obs: np.ndarray) -> np.ndarray:
    """
    Returns obstacle position on the figure-8 at time t.

    x(t) = cx + A·sin(ω·t)
    y(t) = cy + A·sin(ω·t)·cos(ω·t)
    z(t) = cz
    """
    x = _CX + _A * np.sin(_OMG * t)
    y = _CY + _A * np.sin(_OMG * t) * np.cos(_OMG * t)
    z = _CZ
    return np.array([x, y, z])


def obstacle_velocity(t: float) -> np.ndarray:
    """Analytical velocity  (used for γ* computation in ACBF)."""
    vx = _A * _OMG * np.cos(_OMG * t)
    vy = _A * _OMG * (np.cos(_OMG * t)**2 - np.sin(_OMG * t)**2)
    vz = 0.0
    return np.array([vx, vy, vz])
