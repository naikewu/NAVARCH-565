import numpy as np

def ACC_Controller(t, x, param):
    """Return the CLF-CBF QP matrices for decision variables [Fw, delta]."""
    vd = param["vd"]
    v0 = param["v0"]
    m = param["m"]
    Cag = param["Cag"]
    Cdg = param["Cdg"]

    # The objective is (Fw / m)^2 + w * delta^2. Multiplying it by
    # m^2 gives the notebook's Fw^2 + (m^2 * w) * delta^2 objective.
    lam = 1.0
    alpha = 0.2
    w = 100.0

    D, v = x
    error = v - vd
    relative_speed = v - v0
    h = 0.5 * error**2
    barrier_times_Cdg = Cdg * (D - 1.8 * v) - 0.5 * relative_speed**2

    P = np.diag([2.0 / m**2, 2.0 * w])
    q = np.zeros((2, 1))

    # First form the inequalities without divisions, then normalize rows
    # to acceleration/CLF units so large forces do not dominate QP tolerances.
    # Rows: relaxed tracking, hard safety, traction, braking, delta >= 0.
    A = np.array([
        [error, -m],
        [1.8 * Cdg + relative_speed, 0.0],
        [1.0, 0.0],
        [-1.0, 0.0],
        [0.0, -1.0],
    ])
    b = np.array([
        -m * lam * h,
        m * (-relative_speed * Cdg + alpha * barrier_times_Cdg),
        m * Cag,
        m * Cdg,
        0.0,
    ])
    row_scale = np.array([m, m * Cdg, m, m, 1.0])
    A = A / row_scale[:, None]
    b = b / row_scale

    return A, b, P, q
