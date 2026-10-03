"""
utils/math_utils.py
───────────────────
Shared math helpers: UAV kinematic prediction, rotation, geometry.
"""
import numpy as np


# ── Gravity ────────────────────────────────────────────────────────────────────
G_VEC = np.array([0.0, 0.0, 9.8])   # [0,0,g]  (gravity in inertial frame)


# ── Forward-Euler UAV position prediction ──────────────────────────────────────
def predict_uav_trajectory(p0: np.ndarray, v0: np.ndarray,
                            u: np.ndarray, dt: float, N: int) -> np.ndarray:
    """
    Predict UAV position over N steps assuming constant maneuvering acceleration u.

    Implements Eq. (21) from the paper where u is maneuvering acceleration (hover at u=0):
        v(k+1|k) = v(k) + u(k)·Δt
        p(k+1|k) = p(k) + v(k)·Δt

    Parameters
    ----------
    p0 : (3,)  current position
    v0 : (3,)  current velocity
    u  : (3,)  maneuvering acceleration (inertial frame, hover=0)
    dt : float  timestep
    N  : int    horizon length

    Returns
    -------
    pos_seq : (N+1, 3) positions  [p(k|k), p(k+1|k), ..., p(k+N|k)]
    """
    net_acc = u                  # maneuvering acceleration (u=0 is hover)
    pos_seq = np.zeros((N + 1, 3))
    vel_seq = np.zeros((N + 1, 3))
    pos_seq[0] = p0
    vel_seq[0] = v0
    for i in range(N):
        vel_seq[i + 1] = vel_seq[i] + net_acc * dt
        pos_seq[i + 1] = pos_seq[i] + vel_seq[i] * dt
    return pos_seq


def predict_uav_position_i(p0: np.ndarray, v0: np.ndarray,
                            u: np.ndarray, dt: float, i: int) -> np.ndarray:
    """
    Analytical i-step position prediction (closed form of Eq. 21).

        p(k+i|k) = p0 + i·v0·dt + i·(i-1)/2·u·dt²

    Parameters
    ----------
    p0, v0, u : (3,) arrays (u is maneuvering acceleration, hover=0)
    dt : float
    i  : int   prediction step index

    Returns
    -------
    (3,) position at step i
    """
    net_acc = u
    return p0 + i * v0 * dt + (i * (i - 1) / 2) * net_acc * (dt ** 2)


# ── Safety distance ────────────────────────────────────────────────────────────
def safety_distance(drone_r: float, obs_r: float, cushion: float) -> float:
    """d_s = r_drone + r_obs + s_cushion  (Eq. 22 notation)."""
    return drone_r + obs_r + cushion


def safety_fn(p_uav: np.ndarray, p_obs: np.ndarray, ds: float) -> float:
    """
    Scalar CBF safety function h(p_uav, p_obs).

    h = ½‖p_uav - p_obs‖² - d_s²   (Eq. 22)

    Safe iff h ≥ 0.
    """
    diff = np.array(p_uav).flatten() - np.array(p_obs).flatten()
    return 0.5 * float(np.dot(diff, diff)) - ds ** 2


# ── Rotation helpers ───────────────────────────────────────────────────────────
def rpy_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert roll-pitch-yaw (rad) to 3×3 rotation matrix (ZYX convention)."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr               ]
    ])
    return R


# ── Collision angle (adaptive γ* computation) ──────────────────────────────────
def relative_collision_angle(p_uav: np.ndarray, v_uav: np.ndarray,
                              p_obs: np.ndarray, v_obs: np.ndarray):
    """
    Compute cos(θ_r) = (p_r · v_r) / (‖p_r‖ · ‖v_r‖)   (Eq. 27)

    p_r = p_obs - p_uav,   v_r = v_obs - v_uav

    Returns
    -------
    cos_theta_r : float   (positive → converging)
    v_rel       : (3,)    relative velocity vector
    p_rel       : (3,)    relative position vector
    """
    p_rel = p_obs - p_uav
    v_rel = v_obs - v_uav
    norm_p = np.linalg.norm(p_rel)
    norm_v = np.linalg.norm(v_rel)
    if norm_p < 1e-6 or norm_v < 1e-6:
        return 0.0, v_rel, p_rel
    cos_theta_r = np.dot(p_rel, v_rel) / (norm_p * norm_v)
    return float(cos_theta_r), v_rel, p_rel


def adaptive_gamma_star(p_uav, v_uav, p_obs, v_obs, gamma_max: float) -> float:
    """
    Compute γ*(k) from Eq. (28):

        γ*(k) = γ_max · ‖v_r‖ · max(0, cos θ_r) / ‖p_r‖

    Tightens safety margins as relative speed increases or distance decreases.
    """
    cos_theta_r, v_rel, p_rel = relative_collision_angle(p_uav, v_uav, p_obs, v_obs)
    norm_v = np.linalg.norm(v_rel)
    norm_p = np.linalg.norm(p_rel)
    if norm_p < 1e-6:
        return gamma_max
    gamma_star = gamma_max * norm_v * max(0.0, cos_theta_r) / norm_p
    return float(np.clip(gamma_star, 0.0, gamma_max))
