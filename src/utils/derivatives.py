import numpy as np

def compute_kinematic_derivatives(q: np.ndarray, fps: float = 60.0):
    """
    Computes position (q), velocity (q_dot), acceleration (q_ddot),
    and jerk (q_ddot) using central finite differences.

    q shape: (T, K, 3) where T is frames, K is joints
    """
    dt = 1.0 / fps
    T, K, C = q.shape

    # 1st derivative: Velocity (q_dot)
    q_dot = np.zeros_like(q)
    q_dot[1:-1] = (q[2:] - q[:-2]) / (2 * dt)
    q_dot[0] = (q[1] - q[0]) / dt
    q_dot[-1] = (q[-1] - q[-2]) / dt

    # 2nd derivative: Acceleration (q_ddot)
    q_ddot = np.zeros_like(q)
    q_ddot[1:-1] = (q[2:] - 2 * q[1:-1] + q[:-2]) / (dt ** 2)
    q_ddot[0] = (q[2] - 2 * q[1] + q[0]) / (dt ** 2)
    q_ddot[-1] = (q[-1] - 2 * q[-2] + q[-3]) / (dt ** 2)

    # 3rd derivative: Jerk (q_dddot)
    q_dddot = np.zeros_like(q)
    q_dddot[2:-2] = (q[4:] - 2 * q[3:-1] + 2 * q[1:-3] - q[:-4]) / (2 * (dt ** 3))

    return q, q_dot, q_ddot, q_dddot