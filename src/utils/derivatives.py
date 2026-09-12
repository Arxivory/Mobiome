import numpy as np

def compute_kinematic_derivatives(q: np.ndarray, fps: float = 60.0):
    """
    Computes generalized position (q), velocity (q_dot), acceleration (q_ddot),
    and jerk (q_dddot) using central finite differences.

    Supports:
      - Cartesian Poses: (T, K, 3)
      - Generalized Joint Angles: (T, D)
    """
    if q.ndim not in [2, 3] or q.shape[0] < 3:
        raise ValueError("q must have shape (T, D) or (T, K, 3) with T >= 3")

    dt = 1.0 / fps
    
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