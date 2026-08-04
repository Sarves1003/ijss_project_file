#!/usr/bin/env python3
"""
Paper_Project/python/evaluate_progressive_controllers.py
===========================================================
Head-to-head evaluation of the three-stage progressive controller framework
on held-out randomized test episodes (full 2.0 s / 100-step protocol,
matching benchmark_runner.py so results are directly comparable to the
PID/CTC/SMC/LQR/H-infinity benchmark):

    Stage 1: PID + gravity compensation (conventional baseline)
    Stage 2: Computed Torque Control (model-based feedback linearization)
    Stage 3: RL-scheduled CTC (ES-trained adaptive gain scheduling on top of Stage 2)

Uses the ES policy checkpoint produced by rl_gain_scheduler.py. All numbers
are computed by actually running the simulation -- no placeholder values.
"""
import os
import sys
import json
import time
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from kinematics_engine import MyCobot280Kinematics, TrajectoryPlanner
from dynamics_engine import MyCobot280Dynamics
from controllers import PIDGravityController, ComputedTorqueController
from rl_gain_scheduler import policy_action, CKPT_PATH, KP_CTC, KD_CTC

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

# Best metaheuristic-tuned CTC gains (winner = lowest cost among PSO/GWO/BO,
# see run_optimization_suite.py / results/optimization_results.json). Used
# below to give the "conventional PID vs RL vs optimization-tuned-parameters"
# comparison a genuine fourth, optimizer-derived operating point rather than
# re-using the hand-tuned CTC design gains.
_OPT_RESULTS_PATH = os.path.join(RESULTS_DIR, "optimization_results.json")
with open(_OPT_RESULTS_PATH) as _f:
    _opt_results = json.load(_f)
_best_name = min(_opt_results, key=lambda k: _opt_results[k]["best_cost"])
OPT_KP = float(_opt_results[_best_name]["best_Kp"])
OPT_KD = float(_opt_results[_best_name]["best_Kd"])
OPT_ALGO_NAME = _best_name


def run_test_episode(controller_name, theta, seed, n_steps=100, dt=0.02):
    rng = np.random.RandomState(seed)
    kin = MyCobot280Kinematics()

    payload = rng.uniform(0.0, 0.15)
    dyn = MyCobot280Dynamics(payload_mass=payload)

    q_start = np.radians([0.0, -82.5, 0.0, 0.0, 0.0, 90.0])
    q_goal = q_start + np.radians([20.0, -25.0, 30.0, -15.0, 20.0, -40.0])
    t_vec, q_d, dq_d, ddq_d = TrajectoryPlanner.minimum_jerk_multi_joint(q_start, q_goal, dt * n_steps, n_steps=n_steps)

    q = q_start + rng.normal(0, 0.002, 6)
    dq = np.zeros(6)

    if controller_name == "PID":
        ctrl = PIDGravityController()
    elif controller_name == "CTC":
        ctrl = ComputedTorqueController()
    elif controller_name == "CTC-Opt":
        ctrl = ComputedTorqueController(Kp=np.diag([OPT_KP] * 6), Kd=np.diag([OPT_KD] * 6))
    # RL uses CTC's structure directly with scheduled gains below.

    pos_err_mm, torque_energy = [], []

    for i in range(n_steps):
        q_meas = q + rng.normal(0, 0.0005, 6)
        dq_meas = dq + rng.normal(0, 0.002, 6)

        e = q_d[i] - q_meas
        de = dq_d[i] - dq_meas

        if controller_name == "RL":
            payload_proxy = payload / 0.15
            state = np.concatenate([e, de, [payload_proxy]])
            alpha_p, alpha_d = policy_action(theta, state)
            Kp_eff = alpha_p * KP_CTC
            Kd_eff = alpha_d * KD_CTC
            M = dyn.compute_mass_matrix(q_meas)
            C = dyn.compute_coriolis_matrix(q_meas, dq_meas)
            G = dyn.compute_gravity_vector(q_meas)
            F = dyn.compute_friction_torque(dq_meas)
            v = ddq_d[i] + Kd_eff @ de + Kp_eff @ e
            tau = M @ v + C @ dq_meas + G + F
        else:
            tau = ctrl.compute_control(q_meas, dq_meas, q_d[i], dq_d[i], ddq_d[i], dt)

        tau_ext = np.zeros(6)
        if 1.0 <= t_vec[i] <= 1.1:
            tau_ext = np.array([0.3, -0.4, 0.2, -0.1, 0.05, 0.0])

        q, dq = dyn.rk4_step(q, dq, tau, dt, tau_ext=tau_ext)

        T_curr = kin.forward_kinematics(q)
        T_des = kin.forward_kinematics(q_d[i])
        err_mm = float(np.linalg.norm(T_des[:3, 3] - T_curr[:3, 3])) * 1000.0
        pos_err_mm.append(err_mm)
        torque_energy.append(float(np.sum(tau ** 2)))

    rmse = float(np.sqrt(np.mean(np.square(pos_err_mm))))
    mae = float(np.mean(pos_err_mm))
    max_err = float(np.max(pos_err_mm))
    energy = float(np.sum(torque_energy) * dt)
    return rmse, mae, max_err, energy, payload


OUT_PATH = os.path.join(RESULTS_DIR, "progressive_controller_comparison.csv")


def main(n_trials=20, max_seconds=38.0):
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError("RL checkpoint not found -- run rl_gain_scheduler.py training first.")
    ckpt = np.load(CKPT_PATH, allow_pickle=True)
    theta = ckpt["theta"]
    n_gen_trained = int(ckpt["generation"])

    if os.path.exists(OUT_PATH):
        df_existing = pd.read_csv(OUT_PATH)
        done = set(zip(df_existing["Controller"], df_existing["Trial"]))
        records = df_existing.to_dict("records")
    else:
        done = set()
        records = []

    t0 = time.time()
    remaining = 0
    for controller_name in ["PID", "CTC", "CTC-Opt", "RL"]:
        for trial in range(n_trials):
            if (controller_name, trial + 1) in done:
                continue
            if time.time() - t0 > max_seconds:
                remaining += 1
                continue
            seed = 50000 + trial  # identical seed sequence -> identical randomized
            # conditions (payload, noise, disturbance) across the three controllers,
            # so differences in outcome are attributable to the controller alone.
            rmse, mae, max_err, energy, payload = run_test_episode(controller_name, theta, seed)
            records.append({
                "Controller": controller_name, "Trial": trial + 1,
                "RMSE_mm": rmse, "MAE_mm": mae, "MaxError_mm": max_err,
                "ControlEnergy_J": energy, "Payload_kg": payload,
            })
            done.add((controller_name, trial + 1))

    df = pd.DataFrame(records)
    df.to_csv(OUT_PATH, index=False)
    total_needed = 4 * n_trials
    print(f"Progress: {len(df)}/{total_needed} trials complete (RL policy trained for {n_gen_trained} ES generations)")
    if len(df) < total_needed:
        print("PARTIAL -- rerun to continue")
    else:
        print("COMPLETE")
        summary = df.groupby("Controller")[["RMSE_mm", "MAE_mm", "ControlEnergy_J"]].agg(["mean", "std"])
        print(summary)
    return df


if __name__ == "__main__":
    main()
