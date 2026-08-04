#!/usr/bin/env python3
"""
Paper_Project/python/rl_gain_scheduler.py
==========================================
Reinforcement-Learning Gain-Scheduling Controller ("Stage 3") for the myCobot 280.

Design rationale
-----------------
This sandbox has no PyTorch/Gymnasium/Stable-Baselines3 and no reliable network
access to install them (see rl_suite.py, whose DRL agents fall back to literal
random actions without torch -- that fallback must NOT be used as a result, as
it would misrepresent an untrained policy as a trained RL controller).

Instead we implement Natural Evolution Strategies (ES) -- a legitimate,
well-established black-box policy-search reinforcement-learning algorithm
(Salimans et al., 2017, "Evolution Strategies as a Scalable Alternative to
Reinforcement Learning") -- entirely in numpy. ES is a good fit here: the
policy is low-dimensional (adaptive gain scheduling on top of the CTC
baseline), gradients through the nonlinear rigid-body dynamics are not
required, and the algorithm is simple enough to audit end-to-end.

Policy
------
The policy observes a feature vector built from the tracking error, its
derivative, and the (measured) payload proxy, and outputs a per-joint
multiplicative gain-scheduling factor for the CTC baseline gains:
    [alpha_p, alpha_d] = sigmoid(W @ phi(state) + b),  scaled to [g_min, g_max]
    Kp_eff = alpha_p * Kp_CTC,   Kd_eff = alpha_d * Kd_CTC
    tau = M(q)[ddq_d + Kd_eff*de + Kp_eff*e] + C(q,dq)dq + G(q) + F(dq)

Training
--------
Mirrored-sampling Evolution Strategy: at each generation, sample perturbation
vectors, evaluate mean +/- perturbation on randomized rollouts (random
payload, sensor noise, external disturbance impulse -- the same distribution
used for the controller benchmark), and update the policy mean along the
reward-weighted average perturbation direction.
"""

import json
import os
import sys
import time
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from kinematics_engine import MyCobot280Kinematics, TrajectoryPlanner
from dynamics_engine import MyCobot280Dynamics

CKPT_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "rl_checkpoint.npz")
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "rl_training_log.csv")

STATE_DIM = 13   # e(6), de(6), payload_proxy(1)
ACTION_DIM = 2   # alpha_p, alpha_d (shared scalar gain-scaling per joint group for a tractable search space)
GAIN_MIN, GAIN_MAX = 0.4, 2.2

KP_CTC = np.diag([400.0] * 6)
KD_CTC = np.diag([40.0] * 6)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def policy_action(theta: np.ndarray, state: np.ndarray) -> np.ndarray:
    """theta is a flattened (ACTION_DIM x (STATE_DIM+1)) weight+bias matrix."""
    W = theta[:ACTION_DIM * STATE_DIM].reshape(ACTION_DIM, STATE_DIM)
    b = theta[ACTION_DIM * STATE_DIM:]
    raw = W @ state + b
    return GAIN_MIN + (GAIN_MAX - GAIN_MIN) * sigmoid(raw)


def run_episode(theta: np.ndarray, seed: int, n_steps: int = 50, dt: float = 0.02,
                 record: bool = False):
    """
    One randomized rollout under the RL-scheduled CTC controller.
    Returns a scalar cost (lower is better) and, if record=True, full traces.
    """
    rng = np.random.RandomState(seed)
    kin = MyCobot280Kinematics()

    payload = rng.uniform(0.0, 0.15)
    dyn = MyCobot280Dynamics(payload_mass=payload)

    q_start = np.radians([0.0, -82.5, 0.0, 0.0, 0.0, 90.0])
    q_goal = q_start + np.radians([20.0, -25.0, 30.0, -15.0, 20.0, -40.0])
    t_vec, q_d, dq_d, ddq_d = TrajectoryPlanner.minimum_jerk_multi_joint(q_start, q_goal, dt * n_steps, n_steps=n_steps)

    q = q_start + rng.normal(0, 0.002, 6)
    dq = np.zeros(6)

    pos_err_mm, torque_energy = [], []
    gains_log = []

    for i in range(n_steps):
        q_meas = q + rng.normal(0, 0.0005, 6)
        dq_meas = dq + rng.normal(0, 0.002, 6)

        e = q_d[i] - q_meas
        de = dq_d[i] - dq_meas
        payload_proxy = payload / 0.15  # normalized [0,1] proxy available to the policy
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

        # External impulse disturbance at the episode's 40-50% mark (matches the
        # relative timing of the 1.0-1.1s pulse used in the 2.0s benchmark episodes).
        t_total = n_steps * dt
        tau_ext = np.zeros(6)
        if 0.4 * t_total <= t_vec[i] <= 0.5 * t_total:
            tau_ext = np.array([0.3, -0.4, 0.2, -0.1, 0.05, 0.0])

        q, dq = dyn.rk4_step(q, dq, tau, dt, tau_ext=tau_ext)

        T_curr = kin.forward_kinematics(q)
        T_des = kin.forward_kinematics(q_d[i])
        err_mm = float(np.linalg.norm(T_des[:3, 3] - T_curr[:3, 3])) * 1000.0
        pos_err_mm.append(err_mm)
        torque_energy.append(float(np.sum(tau ** 2)))
        gains_log.append((alpha_p, alpha_d))

    rmse = float(np.sqrt(np.mean(np.square(pos_err_mm))))
    energy = float(np.sum(torque_energy) * dt)
    cost = rmse + 0.002 * energy

    if record:
        return cost, {
            "t": t_vec, "pos_err_mm": np.array(pos_err_mm),
            "energy_series": np.array(torque_energy), "gains": np.array(gains_log),
            "rmse": rmse, "energy": energy,
        }
    return cost


def train_es(n_generations: int = 20, pop_size: int = 8, sigma: float = 0.3, lr: float = 0.05,
             max_seconds: float = 28.0):
    """
    Resumable ES training loop. Saves a checkpoint (theta, generation, sigma,
    reward history) after every generation so it can be re-invoked across
    multiple bounded-time process calls until n_generations is reached.
    """
    dim = ACTION_DIM * (STATE_DIM + 1)
    os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

    if os.path.exists(CKPT_PATH):
        ckpt = np.load(CKPT_PATH, allow_pickle=True)
        theta = ckpt["theta"]
        start_gen = int(ckpt["generation"])
        sigma = float(ckpt["sigma"])
        history = list(ckpt["history"])
        rng_state = int(ckpt["seed_counter"])
    else:
        rng0 = np.random.RandomState(0)
        theta = rng0.normal(0, 0.1, dim)
        start_gen = 0
        history = []
        rng_state = 10000

    t0 = time.time()
    gen = start_gen
    seed_counter = rng_state

    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w") as f:
            f.write("generation,mean_cost,best_cost,sigma,wallclock_s\n")

    while gen < n_generations and (time.time() - t0) < max_seconds:
        rng = np.random.RandomState(1000 + gen)
        eps = rng.normal(0, 1, (pop_size, dim))
        costs = np.zeros(2 * pop_size)

        episode_seed = seed_counter
        seed_counter += 1

        for i in range(pop_size):
            theta_plus = theta + sigma * eps[i]
            theta_minus = theta - sigma * eps[i]
            costs[2 * i] = run_episode(theta_plus, episode_seed)
            costs[2 * i + 1] = run_episode(theta_minus, episode_seed)

        # Fitness shaping: lower cost = higher reward. Rank-normalize for robustness.
        rewards = -costs
        ranks = np.argsort(np.argsort(rewards))
        shaped = (ranks / (len(ranks) - 1)) - 0.5

        grad = np.zeros(dim)
        for i in range(pop_size):
            grad += shaped[2 * i] * eps[i] - shaped[2 * i + 1] * eps[i]
        grad /= (pop_size * sigma)

        theta = theta + lr * grad
        sigma = max(sigma * 0.98, 0.05)

        mean_cost = float(np.mean(costs))
        best_cost = float(np.min(costs))
        history.append((gen, mean_cost, best_cost, sigma))

        with open(LOG_PATH, "a") as f:
            f.write(f"{gen},{mean_cost:.6f},{best_cost:.6f},{sigma:.6f},{time.time()-t0:.2f}\n")

        gen += 1

        np.savez(CKPT_PATH, theta=theta, generation=gen, sigma=sigma,
                 history=np.array(history), seed_counter=seed_counter)

    elapsed = time.time() - t0
    print(f"Trained generations {start_gen} -> {gen} (target {n_generations}) in {elapsed:.1f}s")
    return gen >= n_generations


if __name__ == "__main__":
    done = train_es()
    print("COMPLETE" if done else "PARTIAL -- rerun to continue training")
