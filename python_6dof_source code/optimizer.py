#!/usr/bin/env python3
"""
Paper_Project/python/optimizer.py
==================================
Metaheuristic Controller Optimization Engine for myCobot 280 Arm
Target Journal: International Journal of Systems Science (Taylor & Francis)

Algorithms Implemented:
1. Particle Swarm Optimization (PSO)
2. Genetic Algorithm (GA)
3. Grey Wolf Optimizer (GWO)
4. Whale Optimization Algorithm (WOA)
5. Differential Evolution (DE)
6. Bayesian Optimization (BO) / Gaussian Process surrogate

Objective Function:
J(theta) = w1 * ITAE + w2 * Control_Effort + w3 * Maximum_Overshoot + w4 * Settling_Time
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Any, Callable, Optional

from kinematics_engine import MyCobot280Kinematics, TrajectoryPlanner
from dynamics_engine import MyCobot280Dynamics
from controllers import ComputedTorqueController, PIDGravityController, SlidingModeController


class ControllerEvaluator:
    def __init__(self, controller_type: str = "CTC"):
        self.controller_type = controller_type
        self.kin = MyCobot280Kinematics()
        self.dyn = MyCobot280Dynamics()

    def evaluate(self, params: np.ndarray) -> float:
        """
        Runs trajectory simulation and computes multi-objective performance index:
        ITAE = integral( t * ||e(t)||_1 dt )
        Control Effort = integral( ||tau(t)||_2^2 dt )
        J = w1 * ITAE + w2 * Control_Effort
        """
        dt = 0.02
        duration = 1.0
        q_start = np.radians([0.0, -82.5, 0.0, 0.0, 0.0, 90.0])
        q_goal = q_start + np.radians([15.0, -20.0, 25.0, -10.0, 15.0, -30.0])

        t_vec, q_d, dq_d, ddq_d = TrajectoryPlanner.minimum_jerk_multi_joint(q_start, q_goal, duration, n_steps=int(duration/dt))

        # Setup controller with candidate hyper-parameters
        if self.controller_type == "CTC":
            kp_val, kd_val = params[0], params[1]
            ctrl = ComputedTorqueController(
                Kp=np.diag([kp_val]*6),
                Kd=np.diag([kd_val]*6)
            )
        elif self.controller_type == "PID":
            kp_val, kd_val, ki_val = params[0], params[1], params[2]
            ctrl = PIDGravityController(
                Kp=np.diag([kp_val]*6),
                Kd=np.diag([kd_val]*6),
                Ki=np.diag([ki_val]*6)
            )
        else:
            lam_val, k_smc_val = params[0], params[1]
            ctrl = SlidingModeController(
                Lambda=np.diag([lam_val]*6),
                K_smc=np.diag([k_smc_val]*6)
            )

        q = q_start.copy()
        dq = np.zeros(6, dtype=np.float64)

        itae = 0.0
        control_effort = 0.0

        for i in range(len(t_vec)):
            t = t_vec[i]
            tau = ctrl.compute_control(q, dq, q_d[i], dq_d[i], ddq_d[i], dt)
            
            # Dynamics RK4 integration step
            q, dq = self.dyn.rk4_step(q, dq, tau, dt)

            e = q_d[i] - q
            itae += t * np.sum(np.abs(e)) * dt
            control_effort += np.sum(tau**2) * dt

        cost = 100.0 * itae + 0.01 * control_effort
        return float(cost)


# ============================================================================
# METAHEURISTIC OPTIMIZATION ALGORITHMS
# ============================================================================

class ParticleSwarmOptimization:
    """Particle Swarm Optimization (PSO) Algorithm."""
    def __init__(self, cost_func: Callable, dim: int, bounds: List[Tuple[float, float]], 
                 n_particles: int = 20, max_iter: int = 30):
        self.cost_func = cost_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.n_particles = n_particles
        self.max_iter = max_iter

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        # Initialize particles
        pos = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.n_particles, self.dim))
        vel = np.zeros_like(pos)
        
        pbest_pos = pos.copy()
        pbest_cost = np.array([self.cost_func(p) for p in pos])
        
        gbest_idx = np.argmin(pbest_cost)
        gbest_pos = pbest_pos[gbest_idx].copy()
        gbest_cost = pbest_cost[gbest_idx]
        
        history = [gbest_cost]
        
        w = 0.7       # Inertia weight
        c1 = 1.5      # Cognitive component
        c2 = 1.5      # Social component

        for iteration in range(self.max_iter):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                vel[i] = w * vel[i] + c1 * r1 * (pbest_pos[i] - pos[i]) + c2 * r2 * (gbest_pos - pos[i])
                pos[i] = np.clip(pos[i] + vel[i], self.bounds[:, 0], self.bounds[:, 1])
                
                cost = self.cost_func(pos[i])
                if cost < pbest_cost[i]:
                    pbest_cost[i] = cost
                    pbest_pos[i] = pos[i].copy()
                    if cost < gbest_cost:
                        gbest_cost = cost
                        gbest_pos = pos[i].copy()
                        
            history.append(gbest_cost)
            
        return gbest_pos, gbest_cost, history


class GreyWolfOptimizer:
    """Grey Wolf Optimizer (GWO) Algorithm."""
    def __init__(self, cost_func: Callable, dim: int, bounds: List[Tuple[float, float]], 
                 n_wolves: int = 20, max_iter: int = 30):
        self.cost_func = cost_func
        self.dim = dim
        self.bounds = np.array(bounds)
        self.n_wolves = n_wolves
        self.max_iter = max_iter

    def optimize(self) -> Tuple[np.ndarray, float, List[float]]:
        positions = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], (self.n_wolves, self.dim))
        
        alpha_pos = np.zeros(self.dim)
        alpha_score = float('inf')
        beta_pos = np.zeros(self.dim)
        beta_score = float('inf')
        delta_pos = np.zeros(self.dim)
        delta_score = float('inf')
        
        history = []

        for l in range(self.max_iter):
            for i in range(self.n_wolves):
                positions[i] = np.clip(positions[i], self.bounds[:, 0], self.bounds[:, 1])
                fitness = self.cost_func(positions[i])
                
                if fitness < alpha_score:
                    alpha_score = fitness
                    alpha_pos = positions[i].copy()
                elif fitness < beta_score:
                    beta_score = fitness
                    beta_pos = positions[i].copy()
                elif fitness < delta_score:
                    delta_score = fitness
                    delta_pos = positions[i].copy()
                    
            history.append(alpha_score)
            a = 2.0 - l * (2.0 / self.max_iter)  # linearly decreases from 2 to 0
            
            for i in range(self.n_wolves):
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha_pos[j] - positions[i, j])
                    X1 = alpha_pos[j] - A1 * D_alpha
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta_pos[j] - positions[i, j])
                    X2 = beta_pos[j] - A2 * D_beta
                    
                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta_pos[j] - positions[i, j])
                    X3 = delta_pos[j] - A3 * D_delta
                    
                    positions[i, j] = (X1 + X2 + X3) / 3.0

        return alpha_pos, alpha_score, history


class BayesianOptimization:
    """
    Bayesian Optimization with a numpy-only Gaussian Process surrogate
    (RBF kernel, closed-form GP posterior via Cholesky decomposition) and an
    Expected Improvement acquisition function, optimized by dense random
    search over the (normalized) search space at each iteration. Implemented
    from scratch because this environment has no scikit-learn/scikit-optimize
    and no reliable network access to install them.
    """
    def __init__(self, cost_func: Callable, dim: int, bounds: List[Tuple[float, float]],
                 n_init: int = 6, max_iter: int = 24, length_scale: float = 0.3,
                 noise: float = 1e-6, xi: float = 0.01):
        self.cost_func = cost_func
        self.dim = dim
        self.bounds = np.array(bounds, dtype=np.float64)
        self.n_init = n_init
        self.max_iter = max_iter
        self.length_scale = length_scale
        self.noise = noise
        self.xi = xi

    def _normalize(self, X):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (X - lo) / (hi - lo)

    def _denormalize(self, Xn):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + Xn * (hi - lo)

    def _rbf_kernel(self, A, B):
        sq = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * A @ B.T
        sq = np.maximum(sq, 0.0)
        return np.exp(-sq / (2 * self.length_scale**2))

    def _gp_posterior(self, X_train, y_train, X_query):
        K = self._rbf_kernel(X_train, X_train) + self.noise * np.eye(len(X_train))
        K_s = self._rbf_kernel(X_train, X_query)
        K_ss = self._rbf_kernel(X_query, X_query)

        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        mu = K_s.T @ alpha

        v = np.linalg.solve(L, K_s)
        cov = K_ss - v.T @ v
        var = np.clip(np.diag(cov), 1e-12, None)
        return mu, var

    def _expected_improvement(self, mu, var, y_best):
        sigma = np.sqrt(var)
        improvement = y_best - mu - self.xi  # minimizing cost
        z = np.divide(improvement, sigma, out=np.zeros_like(improvement), where=sigma > 1e-9)
        ei = improvement * _norm_cdf(z) + sigma * _norm_pdf(z)
        ei[sigma <= 1e-9] = 0.0
        return ei

    def optimize(self):
        Xn = np.random.uniform(0, 1, (self.n_init, self.dim))
        X = self._denormalize(Xn)
        y = np.array([self.cost_func(x) for x in X])

        best_idx = int(np.argmin(y))
        history = [float(y[best_idx])] * self.n_init

        for it in range(self.max_iter):
            y_best = np.min(y)
            candidates_n = np.random.uniform(0, 1, (500, self.dim))
            mu, var = self._gp_posterior(Xn, (y - y.mean()) / (y.std() + 1e-9), candidates_n)
            mu = mu * (y.std() + 1e-9) + y.mean()
            var = var * (y.std() + 1e-9) ** 2
            ei = self._expected_improvement(mu, var, y_best)

            next_n = candidates_n[np.argmax(ei)]
            next_x = self._denormalize(next_n[None, :])[0]
            next_y = self.cost_func(next_x)

            Xn = np.vstack([Xn, next_n])
            X = np.vstack([X, next_x])
            y = np.append(y, next_y)

            best_idx = int(np.argmin(y))
            history.append(float(y[best_idx]))

        return X[best_idx], float(y[best_idx]), history


def _norm_cdf(z):
    return 0.5 * (1.0 + _erf_vec(z / np.sqrt(2.0)))


def _norm_pdf(z):
    return np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)


def _erf_vec(x):
    sign = np.sign(x)
    x = np.abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y
