"""Shared simulation settings and reproducible output locations."""
from datetime import datetime, timezone
import hashlib
import json
import platform
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml


def command_bounds(cfg):
    """Intersection of actuator limits and the normalized policy interface."""
    scale = float(cfg["k_ma"])
    lower, upper = float(cfg["u_min"]), float(cfg["u_max"])
    lo, hi = max(-scale, lower), min(scale, upper)
    if not np.all(np.isfinite([scale, lower, upper])) or not lo < 0 < hi:
        raise ValueError("Command limits must be finite and allow braking in both directions")
    return lo, hi


def validate_config(cfg):
    hz, physics_hz = cfg["ctrl_freq"], cfg["pyb_freq"]
    if not isinstance(hz, int) or not isinstance(physics_hz, int) or hz < 1 or physics_hz < hz or physics_hz % hz:
        raise ValueError("pyb_freq must be a positive integer multiple of ctrl_freq")
    if not np.isclose(cfg["dt"], 1 / hz, rtol=1e-9, atol=0):
        raise ValueError("dt must equal 1 / ctrl_freq")
    if not isinstance(cfg["N_pred"], int) or cfg["N_pred"] < 2:
        raise ValueError("A second-order barrier needs N_pred >= 2")
    if not 0 < cfg["gamma1"] <= 1 or not 0 < cfg["gamma2"] <= 1 or not 0 < cfg["alpha_clf"] < 1:
        raise ValueError("Invalid barrier or Lyapunov decay coefficient")
    for name in ("lambda_delta", "lambda_u", "lambda_gamma", "Q_noise", "R_noise",
                 "drone_radius", "obs_radius", "acceptance_radius", "flight_bounds"):
        if not np.isfinite(float(cfg[name])) or float(cfg[name]) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if (not np.isfinite(cfg["safety_cushion"]) or cfg["safety_cushion"] < 0
            or not isinstance(cfg["max_episode_steps"], int) or cfg["max_episode_steps"] < 1):
        raise ValueError("Invalid safety cushion or episode length")
    gain = cfg.get("closing_time_scale", cfg.get("gamma_max", 2.0))
    if not np.isfinite(gain) or gain < 0:
        raise ValueError("closing_time_scale must be finite and nonnegative")
    diag, other = cfg["markov_diag"], cfg["markov_offdiag"]
    if min(diag, other) < 0 or not np.isclose(diag + 2 * other, 1):
        raise ValueError("IMM transition probabilities must sum to one per row")
    if cfg.get("ct_window", 20) < 3 or cfg.get("likelihood_smooth", 2) < 1:
        raise ValueError("IMM windows must contain enough samples")
    command_bounds(cfg)


def load_config(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    # Older parameter files wrote the scientific exponent without a sign.
    cfg["lambda_delta"] = float(cfg["lambda_delta"])
    validate_config(cfg)
    return cfg


def new_run_dir(requested, parent):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    path = Path(requested) if requested else Path(parent) / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_run_manifest(directory, cfg, model_path=None, **details):
    """Capture the effective config, executable sources and checkpoint identity."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    files = list(root.glob("*.py"))
    for package in ("controllers", "envs", "utils", "scenarios"):
        files.extend((root / package).glob("*.py"))
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b'\r\n', b'\n'))
    details["source_sha256"] = digest.hexdigest()
    details["source_hash_format"] = "Ordered relative paths and source bytes with LF line endings"
    if model_path:
        details["model"] = str(Path(model_path).resolve())
        details["model_sha256"] = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
    details['versions'] = {'python': platform.python_version()}
    for package in ('numpy', 'torch', 'gymnasium', 'stable-baselines3', 'pybullet',
                    'gym-pybullet-drones', 'cvxpy', 'clarabel', 'scs', 'matplotlib', 'pyyaml'):
        try:
            details['versions'][package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            details['versions'][package] = 'unknown'
    (directory / 'params.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    (directory / 'manifest.json').write_text(json.dumps(details, indent=2), encoding='utf-8')
