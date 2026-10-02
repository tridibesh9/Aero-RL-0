"""
utils/logger.py
───────────────
Collects per-step metrics during evaluation and generates paper-quality plots
matching Figures 5-13 from the paper.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401


class EpisodeLogger:
    """Records trajectory, distance, and state data for one evaluation episode."""

    def __init__(self, label: str):
        self.label = label
        self.positions  : list[np.ndarray] = []   # UAV (3,)
        self.velocities : list[np.ndarray] = []
        self.accels     : list[np.ndarray] = []   # control inputs u(k)
        self.obs_positions: list[np.ndarray] = []  # obstacle (3,)
        self.min_dists  : list[float] = []
        self.times      : list[float] = []
        self.collision  : bool = False

    def log(self, t: float, p_uav: np.ndarray, v_uav: np.ndarray,
            u: np.ndarray, p_obs: np.ndarray):
        self.times.append(t)
        self.positions.append(p_uav.copy())
        self.velocities.append(v_uav.copy())
        self.accels.append(u.copy())
        self.obs_positions.append(p_obs.copy())
        dist = float(np.linalg.norm(p_uav - p_obs))
        self.min_dists.append(dist)
        if dist < 1e-3:
            self.collision = True

    def min_distance(self) -> float:
        return float(np.min(self.min_dists)) if self.min_dists else np.inf

    def mission_time(self) -> float:
        return self.times[-1] if self.times else np.inf

    def np_pos(self)  -> np.ndarray: return np.array(self.positions)
    def np_vel(self)  -> np.ndarray: return np.array(self.velocities)
    def np_acc(self)  -> np.ndarray: return np.array(self.accels)
    def np_obs(self)  -> np.ndarray: return np.array(self.obs_positions)
    def np_dist(self) -> np.ndarray: return np.array(self.min_dists)
    def np_time(self) -> np.ndarray: return np.array(self.times)


# ── Plotting helpers ────────────────────────────────────────────────────────────

COLORS = {"RL+ACBF": "blue", "PID+ACBF": "green", "Pure RL": "red"}
STYLES = {"RL+ACBF": "-",    "PID+ACBF": "--",     "Pure RL": ":"}


def plot_3d_trajectories(logs: list[EpisodeLogger],
                         goal: np.ndarray, start: np.ndarray,
                         ds: float, save_path: str):
    """Reproduce Figures 5 / 9 / 11 — 3-D trajectory comparison."""
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    for lg in logs:
        pos = lg.np_pos()
        col = COLORS.get(lg.label, "gray")
        sty = STYLES.get(lg.label, "-")
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                linestyle=sty, color=col, linewidth=1.5, label=lg.label)
        # Obstacle path
        obs = lg.np_obs()
        ax.plot(obs[:, 0], obs[:, 1], obs[:, 2],
                linestyle='-.', color='orange', linewidth=1.0,
                label="Obstacle" if lg == logs[0] else None)

    ax.scatter(*start, c='black', marker='o', s=80, label='Start')
    ax.scatter(*goal,  c='purple', marker='*', s=120, label='Goal')

    # Draw safety sphere around obstacle (first position)
    if logs:
        _draw_safety_sphere(ax, logs[0].np_obs()[0], ds)

    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectories')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Logger] Saved 3D trajectory → {save_path}")


def plot_min_distance(logs: list[EpisodeLogger], ds: float, save_path: str):
    """Reproduce Figures 6 / 10 / 12 — minimum distance over time."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for lg in logs:
        col = COLORS.get(lg.label, "gray")
        sty = STYLES.get(lg.label, "-")
        ax.plot(lg.np_time(), lg.np_dist(),
                linestyle=sty, color=col, linewidth=1.5, label=lg.label)
    ax.axhline(ds, color='black', linestyle='--', linewidth=1.0,
               label=f'Safety boundary d_s={ds:.2f} m')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Min distance (m)')
    ax.set_title('Minimum Distance between UAV and Obstacle')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Logger] Saved distance plot → {save_path}")


def plot_states(log: EpisodeLogger, save_path: str):
    """Reproduce Figures 7 / 8 / 13 — position, velocity, acceleration profiles."""
    t   = log.np_time()
    pos = log.np_pos()
    vel = log.np_vel()
    acc = log.np_acc()
    axes_labels = ['X', 'Y', 'Z']

    fig, axes = plt.subplots(3, 3, figsize=(13, 8))
    for col, (data, ylabel) in enumerate(
            [(pos, 'Position (m)'), (vel, 'Velocity (m/s)'), (acc, 'Accel (m/s²)')]):
        for row, lbl in enumerate(axes_labels):
            axes[row, col].plot(t, data[:, row], linewidth=1.2)
            axes[row, col].set_ylabel(f'{lbl} {ylabel}')
            axes[row, col].grid(True, alpha=0.3)
            if row == 2:
                axes[row, col].set_xlabel('Time (s)')

    fig.suptitle(f'State Profiles — {log.label}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Logger] Saved state plot → {save_path}")


def print_summary_table(results: dict):
    """Print Table-3 equivalent to console."""
    header = f"{'Method':<20} {'Training (s)':>14} {'Min Dist (m)':>14} {'Collision':>10}"
    print("\n" + "="*62)
    print(header)
    print("-"*62)
    for method, info in results.items():
        print(f"{method:<20} {str(info.get('train_s','N/A')):>14} "
              f"{info.get('min_dist', 0):.3f}{'':>10} "
              f"{'YES' if info.get('collision') else 'no':>10}")
    print("="*62 + "\n")


# ── Private helpers ─────────────────────────────────────────────────────────────

def _draw_safety_sphere(ax, center: np.ndarray, radius: float, n: int = 20):
    """Draw a wireframe sphere on a 3D axis."""
    u_a = np.linspace(0, 2 * np.pi, n)
    v_a = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u_a), np.sin(v_a))
    y = center[1] + radius * np.outer(np.sin(u_a), np.sin(v_a))
    z = center[2] + radius * np.outer(np.ones(n), np.cos(v_a))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.15, linewidth=0.4)
