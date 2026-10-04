"""Run the supplied notebook examples and additional controller checks.

Usage (from the project directory): python tests/run_tests.py
Install requirements.txt in a Python environment outside this submission folder.
Notebook autoreload magics are omitted because this is a fresh Python process.
"""

import json
from contextlib import redirect_stdout
import io
import itertools
import os
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
STUDENT = ROOT / "student"
OUTPUT = ROOT / "test_results"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / "matplotlib"))
sys.path.insert(0, str(STUDENT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp

from acc_controller import ACC_Controller
from acc_utils import sim_vehicle
from cmpc_controller import calc_Jacobian, LQR_Controller, CMPC_Controller
from cmpc_utils import BicycleSim


def run_notebook(name):
    notebook = json.loads((STUDENT / name).read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    figures = []
    original_show = plt.show

    def save_figures(*args, **kwargs):
        for number in plt.get_fignums():
            figure = plt.figure(number)
            path = OUTPUT / f"{Path(name).stem}_{len(figures) + 1:02d}.png"
            figure.savefig(path, dpi=130, bbox_inches="tight")
            figures.append(str(path.relative_to(ROOT)))
            plt.close(figure)

    plt.show = save_figures
    try:
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            source = "\n".join(line for line in source.splitlines()
                               if not line.lstrip().startswith("%"))
            exec(compile(source, f"{name}:cell{i}", "exec"), namespace)
    finally:
        plt.show = original_show
    return namespace, figures


def check_acc(name, param, initial, simulation=None):
    started = time.perf_counter()
    if simulation is None:
        # Some OSQP versions print a redundant polishing message even with
        # verbose=False. Keep solver chatter out of the test progress output.
        with redirect_stdout(io.StringIO()):
            simulation = sim_vehicle(ACC_Controller, param.copy(), initial)
    t, barrier, states, force = simulation
    assert np.all(np.isfinite(states)) and np.all(np.isfinite(force)), name
    acceleration = force.ravel() / param["m"]
    before = t < param["switch_time"]
    full_barrier = (states[:, 0] - 1.8 * states[:, 1] -
                    0.5 * (states[:, 1] - np.where(before, param["v01"],
                                                 param["v02"]))**2 / param["Cdg"])
    assert np.min(states[:, 0]) > 0, f"{name}: collision"
    assert np.min(barrier[before]) >= -1e-3, f"{name}: unsafe before switch"
    assert np.min(full_barrier[before]) >= -1e-3, f"{name}: notebook barrier"
    assert acceleration.min() >= -param["Cdg"] - 1e-4, f"{name}: braking bound"
    assert acceleration.max() <= param["Cag"] + 1e-4, f"{name}: traction bound"
    expected_speed = min(param["vd"], param["v02"])
    assert abs(states[-1, 1] - expected_speed) < 0.3, f"{name}: speed tracking"
    result = {
        "name": name, "initial_state": initial.tolist(),
        "min_distance_m": float(states[:, 0].min()),
        "min_barrier_before_switch": float(barrier[before].min()),
        "min_notebook_barrier_before_switch": float(full_barrier[before].min()),
        "min_barrier_all_time": float(barrier.min()),
        "final_speed_m_s": float(states[-1, 1]),
        "expected_final_speed_m_s": float(expected_speed),
        "acceleration_range": [float(acceleration.min()), float(acceleration.max())],
        "seconds": time.perf_counter() - started,
    }
    print("PASS ACC", name, "min D", round(result["min_distance_m"], 3),
          "final v", round(result["final_speed_m_s"], 4), flush=True)
    return result


def check_tracking(name, function, param, initial, simulation=None):
    started = time.perf_counter()
    simulator = BicycleSim(param)
    reference_input, reference_state = simulator.GenRef(5, 5)
    states, inputs = (simulation if simulation is not None else
                      simulator.SimVehicle(reference_state, reference_input, 20,
                                           initial, lambda x, u, x0:
                                           function(x, u, x0, param)))
    assert np.all(np.isfinite(states)) and np.all(np.isfinite(inputs)), name
    error = states - reference_state
    tail_error = np.linalg.norm(error[-100:], "fro")
    # Use physical tracking tolerances for the last five seconds: position
    # RMSE < 5 cm, heading RMSE < 0.05 rad, speed RMSE < 0.05 m/s.
    tail_rmse = np.sqrt(np.mean(error[-100:]**2, axis=0))
    position_rmse = np.sqrt(np.mean(np.sum(error[-100:, :2]**2, axis=1)))
    assert position_rmse < 0.05, f"{name}: position RMSE {position_rmse}"
    assert tail_rmse[2] < 0.05, f"{name}: heading RMSE {tail_rmse[2]}"
    assert tail_rmse[3] < 0.05, f"{name}: speed RMSE {tail_rmse[3]}"
    assert np.linalg.norm(error[-1]) < 0.01, f"{name}: final tracking error"
    if function is CMPC_Controller:
        lower = np.array([param["a_lim"][0], param["delta_lim"][0]])
        upper = np.array([param["a_lim"][1], param["delta_lim"][1]])
        assert np.all(inputs >= lower - 1e-9), f"{name}: lower input bound"
        assert np.all(inputs <= upper + 1e-9), f"{name}: upper input bound"
    result = {
        "name": name, "initial_state": initial.tolist(),
        "last_100_steps_error": float(tail_error),
        "last_100_steps_position_rmse_m": float(position_rmse),
        "last_100_steps_component_rmse": tail_rmse.tolist(),
        "final_state_error": error[-1].tolist(),
        "input_min": inputs.min(axis=0).tolist(),
        "input_max": inputs.max(axis=0).tolist(),
        "seconds": time.perf_counter() - started,
    }
    print("PASS", name, "tail error", f"{tail_error:.6g}", flush=True)
    return result


def check_jacobian(rng):
    worst = 0.0
    for _ in range(40):
        param = {"L_f": rng.uniform(0.7, 2.5), "L_r": rng.uniform(0.7, 2.5),
                 "h": rng.uniform(0.02, 0.2), "T": 1.0,
                 "a_lim": [-10.0, 4.0], "delta_lim": [-0.8, 0.8]}
        sim = BicycleSim(param)
        state = rng.uniform([-5, -5, -3, 0], [5, 5, 3, 30])
        control = rng.uniform([-10, -0.8], [4, 0.8])
        A, B = calc_Jacobian(state, control, param)
        eps = 1e-6
        numerical_A = np.column_stack([
            (sim.Fun_dynamics_dt(state + eps * e, control) -
             sim.Fun_dynamics_dt(state - eps * e, control)) / (2 * eps)
            for e in np.eye(4)])
        numerical_B = np.column_stack([
            (sim.Fun_dynamics_dt(state, control + eps * e) -
             sim.Fun_dynamics_dt(state, control - eps * e)) / (2 * eps)
            for e in np.eye(2)])
        worst = max(worst, np.max(np.abs(A - numerical_A)),
                    np.max(np.abs(B - numerical_B)))
    assert worst < 1e-7, f"Jacobian error: {worst}"
    print("PASS Jacobian: 40 finite-difference comparisons", flush=True)
    return {"cases": 40, "max_absolute_error": float(worst)}


def check_acc_qps(rng):
    worst = 0.0
    for _ in range(200):
        v = rng.uniform(0, 50)
        v0 = rng.uniform(0, 60)
        param = {"vd": rng.uniform(0, 30), "v0": v0,
                 "m": rng.uniform(1000, 3000), "Cag": 0.3 * 9.81,
                 "Cdg": 0.8 * 9.81}
        minimum_distance = 1.8 * v + 0.5 * (v - v0)**2 / param["Cdg"]
        distance = rng.uniform(minimum_distance, 500)
        A, b, P, q = ACC_Controller(0, np.array([distance, v]), param)
        z = cp.Variable(2)
        problem = cp.Problem(cp.Minimize(0.5 * cp.quad_form(z, P) + q.T @ z),
                             [A @ z <= b])
        problem.solve()
        assert problem.status == cp.OPTIMAL, f"ACC QP: {problem.status}"
        residual = np.max(A @ z.value - b)
        worst = max(worst, residual)
        assert residual < 5e-4, f"ACC QP residual: {residual}"
    print("PASS ACC: 200 feasible-state QPs", flush=True)
    return {"cases": 200, "max_constraint_residual": float(worst)}


def main():
    OUTPUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(535)
    summary = {"acc": [], "trajectory": [], "figures": []}
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter("always")
        summary["jacobian"] = check_jacobian(rng)
        summary["acc_qps"] = check_acc_qps(rng)

        print("Running AdaptiveCruiseControl.ipynb ...", flush=True)
        ns, figures = run_notebook("AdaptiveCruiseControl.ipynb")
        summary["figures"] += figures
        summary["acc"].append(check_acc("notebook_example", ns["param"], ns["y0"],
                                        (ns["t"], ns["B"], ns["y"], ns["u"])))

        print("Running TrajectoryTracking.ipynb ...", flush=True)
        ns, figures = run_notebook("TrajectoryTracking.ipynb")
        summary["figures"] += figures
        # The notebook's final log belongs to Task 2 (MPC).
        tracking_param = ns["param"]
        summary["trajectory"].append(check_tracking("MPC_notebook_example",
            CMPC_Controller, tracking_param, ns["x0"], (ns["x_log"], ns["u_log"])))

        base = {"m": 2000, "Cag": 0.3 * 9.81, "Cdg": 0.8 * 9.81,
                "switch_time": 30, "terminal_time": 50}
        cases = [
            ("stationary_lead", [80, 10], {"vd": 15, "v01": 0, "v02": 0}),
            ("emergency_stop", [150, 20], {"vd": 20, "v01": 20, "v02": 0,
                                          "switch_time": 20, "terminal_time": 60}),
            ("fast_lead", [80, 10], {"vd": 15, "v01": 30, "v02": 30}),
            ("high_desired_speed", [200, 25], {"vd": 30, "v01": 10, "v02": 40}),
            ("small_safety_margin", [53, 20], {"vd": 30, "v01": 5, "v02": 5}),
            ("zero_desired_speed", [100, 20], {"vd": 0, "v01": 0, "v02": 0}),
        ]
        for name, initial, values in cases:
            summary["acc"].append(check_acc(name, dict(base, **values),
                                            np.array(initial, dtype=float)))

        initial_states = [np.array([2, 2, 3.14 * 0.3, 2])]
        initial_states += [np.array(values, dtype=float) for values in
                           itertools.product([0, 5], [0, 5],
                                             [0, 0.6 * np.pi], [0, 5])]
        initial_states += [rng.uniform([0, 0, 0, 0], [5, 5, 0.6 * np.pi, 5])
                           for _ in range(3)]
        for i, initial in enumerate(initial_states):
            for name, function in [("LQR", LQR_Controller), ("MPC", CMPC_Controller)]:
                summary["trajectory"].append(check_tracking(
                    f"{name}_initial_{i}", function, tracking_param, initial))

    summary["warnings"] = sorted(set(str(w.message) for w in notices))
    summary["seconds"] = time.perf_counter() - started
    summary["status"] = "passed"
    summary["versions"] = {"python": sys.version.split()[0], "numpy": np.__version__,
                           "cvxpy": cp.__version__, "matplotlib": matplotlib.__version__}
    (OUTPUT / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"ALL CHECKS PASSED in {summary['seconds']:.1f}s", flush=True)
    print("Results: test_results/results.json", flush=True)
    for message in summary["warnings"]:
        print("Warning:", message, flush=True)


if __name__ == "__main__":
    main()
