"""World-frame net acceleration to motor thrust and attitude."""
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation


def acceleration_rpm(controller, acceleration, quaternion, mass, gravity, timestep):
    """Fast attitude loop with gravity added once and no position PID."""
    force = mass * (np.asarray(acceleration) + [0., 0., gravity])
    rotation = np.asarray(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
    thrust = max(0., float(force @ rotation[:, 2]))
    pwm = (np.sqrt(thrust / (4 * controller.KF)) - controller.PWM2RPM_CONST) / controller.PWM2RPM_SCALE
    z_axis = force / np.linalg.norm(force)
    y_axis = np.cross(z_axis, [1., 0., 0.])
    y_axis /= np.linalg.norm(y_axis)
    target_rotation = np.column_stack((np.cross(y_axis, z_axis), y_axis, z_axis))
    target_rpy = Rotation.from_matrix(target_rotation).as_euler('XYZ')
    return controller._dslPIDAttitudeControl(timestep, pwm, quaternion, target_rpy, np.zeros(3))
