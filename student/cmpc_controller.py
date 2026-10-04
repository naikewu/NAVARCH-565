from functools import lru_cache

import numpy as np
import cvxpy as cp


def calc_Jacobian(x, u, param):
    """Jacobians of the simulator's Euler-discretized bicycle model."""
    L_f = param["L_f"]
    L_r = param["L_r"]
    dt   = param["h"]

    psi = x[2]
    v   = x[3]
    delta = u[1]
    B = np.zeros((4, 2))

    # Match the supplied simulator, including its nested arctan formula.
    ratio = L_r / (L_f + L_r)
    beta = np.arctan(ratio * np.arctan(delta))
    dbeta = ratio / ((1.0 + delta**2) *
                     (1.0 + (ratio * np.arctan(delta))**2))
    angle = psi + beta

    A = np.eye(4)
    A[0, 2] = -dt * v * np.sin(angle)
    A[0, 3] = dt * np.cos(angle)
    A[1, 2] = dt * v * np.cos(angle)
    A[1, 3] = dt * np.sin(angle)
    A[2, 3] = dt * np.sin(beta) / L_r

    B[0, 1] = -dt * v * np.sin(angle) * dbeta
    B[1, 1] = dt * v * np.cos(angle) * dbeta
    B[2, 1] = dt * v * np.cos(beta) * dbeta / L_r
    B[3, 0] = dt

    return [A, B]


def LQR_Controller(x_bar, u_bar, x0, param):
    """Unconstrained time-varying LQR solved as a finite-horizon QP."""
    return _tracking_controller(x_bar, u_bar, x0, param, constrained=False)


def CMPC_Controller(x_bar, u_bar, x0, param):
    """Linear MPC with bounds on actual acceleration and steering."""
    return _tracking_controller(x_bar, u_bar, x0, param, constrained=True)


@lru_cache(maxsize=8)
def _tracking_problem(horizon, constrained, lower, upper, recovering=False):
    """Build a reusable finite-horizon LQR or constrained MPC problem."""
    # With large MPC errors, prioritize aligning heading and speed before
    # aggressively correcting position with a model linearized at the reference.
    Q = np.diag([10.0, 10.0, 20.0, 10.0] if recovering else
                [20.0, 20.0, 10.0, 5.0])
    # Unconstrained LQR needs a larger steering penalty to stay near the
    # linearization when the initial heading/position errors are large.
    R = np.diag([1.0, 5.0] if recovering else
                ([0.3, 2.0] if constrained else [1.0, 20.0]))
    Pt = 2.0 * Q

    state_error = cp.Variable((horizon + 1, 4))
    input_error = cp.Variable((horizon, 2))
    initial_error = cp.Parameter(4)
    reference_input = cp.Parameter((horizon, 2))
    A = [cp.Parameter((4, 4)) for _ in range(horizon)]
    B = [cp.Parameter((4, 2)) for _ in range(horizon)]

    cost = cp.quad_form(state_error[horizon], Pt)
    constraints = [state_error[0] == initial_error]
    for k in range(horizon):
        cost += cp.quad_form(state_error[k], Q)
        cost += cp.quad_form(input_error[k], R)
        constraints.append(state_error[k + 1] ==
                           A[k] @ state_error[k] + B[k] @ input_error[k])
        if constrained:
            constraints += [input_error[k] >= np.array(lower) - reference_input[k],
                            input_error[k] <= np.array(upper) - reference_input[k]]

    problem = cp.Problem(cp.Minimize(cost), constraints)
    return problem, input_error, initial_error, reference_input, A, B


def _tracking_controller(x_bar, u_bar, x0, param, constrained):
    horizon = len(u_bar)
    lower = (param["a_lim"][0], param["delta_lim"][0]) if constrained else ()
    upper = (param["a_lim"][1], param["delta_lim"][1]) if constrained else ()
    initial_error = np.asarray(x0, dtype=float).reshape(4) - x_bar[0]
    recovering = constrained and (np.linalg.norm(initial_error[:2]) > 3.0 or
                                   abs(initial_error[2]) > 0.5)
    problem, du, initial, reference, A, B = _tracking_problem(
        horizon, constrained, lower, upper, recovering)

    initial.value = initial_error
    reference.value = u_bar
    for k in range(horizon):
        A[k].value, B[k].value = calc_Jacobian(x_bar[k], u_bar[k], param)

    try:
        problem.solve(solver=cp.OSQP, warm_start=True, verbose=False,
                      eps_abs=1e-6, eps_rel=1e-6, max_iter=10000)
    except cp.error.SolverError:
        problem.solve(solver=cp.CLARABEL, verbose=False)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError("Tracking QP failed: " + problem.status)

    # Apply only the first action, then solve again using the next state.
    u_act = np.asarray(du.value[0] + u_bar[0], dtype=float)
    if constrained:
        # Remove any tiny bound violation due to the solver's tolerances.
        u_act = np.clip(u_act, lower, upper)
    return u_act
