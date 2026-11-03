"""
utils/logger.py
───────────────
Collects per-step metrics during evaluation and generates paper-quality plots
matching Figures 5-13 from the paper.
"""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from utils.config import load_config


class EpisodeLogger:
    """Records trajectory, distance, and state data for one evaluation episode."""

    def __init__(self, label: str, config=None):
        self.label = label
        if config is None:
            config = load_config(Path(__file__).resolve().parents[1] / 'configs/params.yaml')
        self.collision_distance = config['drone_radius'] + config['obs_radius']
        self.safety_distance = self.collision_distance + config['safety_cushion']
        self.positions  : list[np.ndarray] = []   # UAV (3,)
        self.velocities : list[np.ndarray] = []
        self.accels     : list[np.ndarray] = []   # control inputs u(k)
        self.obs_positions: list[np.ndarray] = []  # obstacle (3,)
        self.min_dists  : list[float] = []
        self.times      : list[float] = []
        self.collision  : bool = False
        self.reached_goal: bool = False
        self.stop_reason = 'step_limit'
        self.step_diagnostics = []
        self.compute_times_ms = []
        self.failed_attempt = None

    def log(self, t: float, p_uav: np.ndarray, v_uav: np.ndarray,
            u: np.ndarray, p_obs: np.ndarray, diagnostics=None, compute_ms=0.0):
        self.times.append(t)
        self.positions.append(p_uav.copy())
        self.velocities.append(v_uav.copy())
        self.accels.append(u.copy())
        self.obs_positions.append(p_obs.copy())
        dist = float(np.linalg.norm(p_uav - p_obs))
        self.min_dists.append(dist)
        self.step_diagnostics.append(dict(diagnostics or {}))
        self.compute_times_ms.append(float(compute_ms))
        if dist <= self.collision_distance:
            self.collision = True

    def summary(self, goal):
        solved = [d for d in self.step_diagnostics if d]
        times = self.compute_times_ms[1:]
        if self.failed_attempt is not None:
            times = times + [self.failed_attempt['control_ms']]
        errors = [np.linalg.norm(pos - d['next_position']) for pos, d in
                  zip(self.positions, self.step_diagnostics) if d]
        return dict(reached_goal=self.reached_goal, stop_reason=self.stop_reason,
                    goal_error=float(np.linalg.norm(self.positions[-1] - goal)),
                    mission_t=self.mission_time(), min_dist=self.min_distance(),
                    min_surface_clearance=self.min_distance() - self.collision_distance,
                    collision=self.collision, buffer_violation=self.min_distance() < self.safety_distance,
                    path_length=float(np.linalg.norm(np.diff(self.np_pos(), axis=0), axis=1).sum()),
                    max_cbf_slack=max((d['slack_max'] for d in solved), default=None),
                    max_clf_slack=max((d['clf_slack'] for d in solved), default=None),
                    max_predicted_violation=max((d['predicted_violation'] for d in solved), default=None),
                    inaccurate_steps=sum(d['inaccurate_seen'] for d in solved),
                    backup_solver_steps=sum(d['used_backup_solver'] for d in solved),
                    filter_failures=int(self.failed_attempt is not None),
                    failed_attempt=self.failed_attempt,
                    one_step_prediction_rmse_m=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                    p95_control_ms=float(np.percentile(times, 95)) if times else None)

    def write_csv(self, path, goal):
        vector_names = ('uav', 'velocity', 'obstacle', 'reference', 'command')
        diagnostic_keys = ('solver', 'solver_status', 'used_backup_solver', 'inaccurate_seen',
                           'gamma', 'slack_max', 'clf_slack', 'predicted_violation',
                           'linearization_margin', 'solve_passes', 'predicted_min_dist')
        with open(path, 'w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['time_s', *(f'{name}_{axis}' for name in vector_names for axis in 'xyz'),
                             'goal_error_m', 'centre_distance_m', 'surface_clearance_m', 'collision',
                             'buffer_violation', 'control_ms', *diagnostic_keys, 'solver_attempts',
                             'predicted_uav_x', 'predicted_uav_y', 'predicted_uav_z', 'termination_reason'])
            for i, t in enumerate(self.times):
                d = self.step_diagnostics[i]
                dist = self.min_dists[i]
                ref = d.get('reference', self.accels[i])
                writer.writerow([t, *self.positions[i], *self.velocities[i], *self.obs_positions[i],
                                 *ref, *self.accels[i], np.linalg.norm(self.positions[i] - goal),
                                 dist, dist - self.collision_distance, dist <= self.collision_distance,
                                 dist < self.safety_distance, self.compute_times_ms[i],
                                 *(d.get(key) for key in diagnostic_keys), json.dumps(d.get('solver_attempts', [])),
                                 *d.get('next_position', [None]*3),
                                 self.stop_reason if i == len(self.times) - 1 else ''])

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
    print(f"[Logger] Saved 3D trajectory -> {save_path}")


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
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Centre-to-centre distance (m)')
    ax.set_title('Separation between UAV and obstacle')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[Logger] Saved distance plot -> {save_path}")


def plot_states(log: EpisodeLogger, save_path: str):
    """Plot measured position and velocity alongside the applied waypoint command."""
    t   = log.np_time()
    pos = log.np_pos()
    vel = log.np_vel()
    acc = log.np_acc()
    axes_labels = ['X', 'Y', 'Z']

    fig, axes = plt.subplots(3, 3, figsize=(13, 8))
    for col, (data, ylabel) in enumerate(
            [(pos, 'Position (m)'), (vel, 'Velocity (m/s)'), (acc, 'Waypoint command')]):
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
    print(f"[Logger] Saved state plot -> {save_path}")


def print_summary_table(results: dict):
    """Print Table-3 equivalent to console."""
    header = f"{'Method':<20} {'End reason':>15} {'Goal error (m)':>14} {'Elapsed (s)':>12} {'Min dist (m)':>13} {'Collision':>10} {'Buffer entered':>14}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for method, info in results.items():
        print(f"{method:<20} {info['stop_reason']:>15} "
              f"{info['goal_error']:14.3f} {info['mission_t']:12.2f} "
              f"{info['min_dist']:13.3f} "
              f"{'YES' if info.get('collision') else 'no':>10} "
              f"{'yes' if info['buffer_violation'] else 'no':>14}")
    print("=" * len(header) + "\n")


# ── Private helpers ─────────────────────────────────────────────────────────────

def _draw_safety_sphere(ax, center: np.ndarray, radius: float, n: int = 20):
    """Draw a wireframe sphere on a 3D axis."""
    u_a = np.linspace(0, 2 * np.pi, n)
    v_a = np.linspace(0, np.pi, n)
    x = center[0] + radius * np.outer(np.cos(u_a), np.sin(v_a))
    y = center[1] + radius * np.outer(np.sin(u_a), np.sin(v_a))
    z = center[2] + radius * np.outer(np.ones(n), np.cos(v_a))
    ax.plot_wireframe(x, y, z, color='gray', alpha=0.15, linewidth=0.4)
