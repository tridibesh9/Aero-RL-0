"""Prevent using policy weights with incompatible observation/action semantics."""
from pathlib import Path


def config_for_model(model_path, explicit=None):
    """Use an explicit config or the checkpoint's own snapshot, never a guess."""
    if explicit:
        return explicit
    directory = Path(model_path).resolve().parent
    if directory.name == 'checkpoints':
        directory = directory.parent
    candidate = directory / 'params.yaml'
    if not candidate.is_file():
        raise ValueError('No params.yaml beside this model. Supply --config explicitly; '
                         'historical waypoint policies use configs/params_legacy.yaml.')
    return str(candidate)


def interface_spec(cfg):
    mode = cfg.get('action_mode', 'waypoint_v1')
    fields = ['k_ma', 'u_min', 'u_max', 'ctrl_freq', 'pyb_freq', 'flight_bounds']
    if mode == 'acceleration_v2':
        fields += ['obs_velocity_scale', 'obs_error_scale', 'obs_angular_scale']
    return dict(version=mode, **{key: cfg[key] for key in fields})


def check_model_interface(model, cfg):
    expected = interface_spec(cfg)
    actual = getattr(model, 'navigation_interface', None)
    if actual is None and expected['version'] == 'waypoint_v1':
        return  # Explicit legacy configuration for historical checkpoints.
    if actual != expected:
        raise ValueError('Checkpoint action/observation interface differs from this configuration. '
                         'Use its saved params.yaml, or train a new compatible policy.')
