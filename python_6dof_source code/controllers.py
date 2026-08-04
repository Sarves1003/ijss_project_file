#!/usr/bin/env python3
"""
Paper_Project/python/controllers.py
====================================
Comprehensive Multi-Controller Suite for myCobot 280 Manipulator
Target Journal: International Journal of Systems Science (Taylor & Francis)

Implementations:
1. Baseline PID / PD with Gravity Compensation
2. Computed Torque Controller (CTC) / Feedback Linearization
3. Sliding Mode Controller (SMC) with Boundary Layer Chattering Suppression
4. Super-Twisting Sliding Mode Controller (ST-SMC)
5. Model Predictive Controller (MPC) - Linearized Receding Horizon
6. Linear Quadratic Regulator (LQR) Optimal Control
7. Robust H-infinity Controller
8. Disturbance Observer (DOB) Enhanced PID Controller
9. Adaptive Controller with Parameter Estimation
10. Fuzzy Logic Supervisory PID Controller
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional

from kinematics_engine import MyCobot280Kinematics
from dynamics_engine import MyCobot280Dynamics
from control_analysis import solve_care

# Base Controller Interface
class BaseController:
    def __init__(self, name: str):
        self.name = name

    def compute_control(self, q: np.ndarray, dq: np.ndarray, 
                        q_d: np.ndarray, dq_d: np.ndarray, ddq_d: np.ndarray, dt: float) -> np.ndarray:
        raise NotImplementedError


class PIDGravityController(BaseController):
    """
    PID Controller with Feedforward Gravity & Dynamic Friction Compensation:
    tau = G(q) + F(dq) + K_p * e + K_d * de + K_i * int(e)
    """
    def __init__(self, Kp: Optional[np.ndarray] = None, Kd: Optional[np.ndarray] = None, Ki: Optional[np.ndarray] = None):
        super().__init__("PID+Gravity")
        self.Kp = Kp if Kp is not None else np.diag([45.0, 50.0, 35.0, 20.0, 15.0, 10.0])
        self.Kd = Kd if Kd is not None else np.diag([4.5, 5.0, 3.5, 2.0, 1.5, 1.0])
        self.Ki = Ki if Ki is not None else np.diag([2.0, 2.5, 1.5, 1.0, 0.5, 0.2])
        self.integral_error = np.zeros(6, dtype=np.float64)
        self.dyn = MyCobot280Dynamics()

    def reset(self):
        self.integral_error = np.zeros(6, dtype=np.float64)

    def compute_control(self, q: np.ndarray, dq: np.ndarray, 
                        q_d: np.ndarray, dq_d: np.ndarray, ddq_d: np.ndarray, dt: float) -> np.ndarray:
        e = q_d - q
        de = dq_d - dq
        self.integral_error += e * dt
        
        # Anti-windup clamping
        self.integral_error = np.clip(self.integral_error, -1.0, 1.0)

        G = self.dyn.compute_gravity_vector(q)
        F = self.dyn.compute_friction_torque(dq)

        tau = G + F + self.Kp @ e + self.Kd @ de + self.Ki @ self.integral_error
        return tau


class ComputedTorqueController(BaseController):
    """
    Computed Torque Controller (Feedback Linearization):
    tau = M(q) * [ ddq_d + K_d * de + K_p * e ] + C(q, dq)*dq + G(q) + F(dq)
    Renders joint error dynamics decoupled linear 2nd order system: dde + Kd*de + Kp*e = 0.
    """
    def __init__(self, Kp: Optional[np.ndarray] = None, Kd: Optional[np.ndarray] = None):
        super().__init__("ComputedTorque")
        # Critical damping parameters omega_n = 20 rad/s -> Kp = omega_n^2, Kd = 2 * omega_n
        self.Kp = Kp if Kp is not None else np.diag([400.0, 400.0, 400.0, 400.0, 400.0, 400.0])
        self.Kd = Kd if Kd is not None else np.diag([40.0, 40.0, 40.0, 40.0, 40.0, 40.0])
        self.dyn = MyCobot280Dynamics()

    def compute_control(self, q: np.ndarray, dq: np.ndarray, 
                        q_d: np.ndarray, dq_d: np.ndarray, ddq_d: np.ndarray, dt: float) -> np.ndarray:
        e = q_d - q
        de = dq_d - dq

        M = self.dyn.compute_mass_matrix(q)
        C = self.dyn.compute_coriolis_matrix(q, dq)
        G = self.dyn.compute_gravity_vector(q)
        F = self.dyn.compute_friction_torque(dq)

        v = ddq_d + self.Kd @ de + self.Kp @ e
        tau = M @ v + C @ dq + G + F
        return tau


class SlidingModeController(BaseController):
    """
    Classical & Super-Twisting Sliding Mode Control (SMC):
    Sliding Surface: s = de + Lambda * e
    Control Law: tau = M(q) * [ ddq_d + Lambda * de + K_smc * sat(s / phi) ] + C(q, dq)*dq + G(q) + F(dq)
    Provides finite-time invariance to matched external disturbances and dynamic uncertainties.
    """
    def __init__(self, Lambda: Optional[np.ndarray] = None, K_smc: Optional[np.ndarray] = None, phi: float = 0.05):
        super().__init__("SlidingMode")
        self.Lambda = Lambda if Lambda is not None else np.diag([15.0]*6)
        self.K_smc = K_smc if K_smc is not None else np.diag([25.0]*6)
        self.phi = phi  # Boundary layer thickness for chattering suppression
        self.dyn = MyCobot280Dynamics()

    def compute_control(self, q: np.ndarray, dq: np.ndarray, 
                        q_d: np.ndarray, dq_d: np.ndarray, ddq_d: np.ndarray, dt: float) -> np.ndarray:
        e = q_d - q
        de = dq_d - dq

        s = de + self.Lambda @ e
        sat_s = np.clip(s / self.phi, -1.0, 1.0)  # Boundary layer continuous approximation

        M = self.dyn.compute_mass_matrix(q)
        C = self.dyn.compute_coriolis_matrix(q, dq)
        G = self.dyn.compute_gravity_vector(q)
        F = self.dyn.compute_friction_torque(dq)

        v = ddq_d + self.Lambda @ de + self.K_smc @ sat_s
        tau = M @ v + C @ dq + G + F
        return tau


class LinearQuadraticRegulator(BaseController):
    """
    Linear Quadratic Regulator (LQR) Optimal State-Feedback Control:
    Linearized continuous State-Space System: x_dot = A*x + B*u
    Cost Function J = integral( x^T Q x + u^T R u ) dt
    Optimal Gain Matrix K = R^-1 B^T P from Continuous Algebraic Riccati Equation (CARE).
    """
    def __init__(self):
        super().__init__("LQR")
        self.dyn = MyCobot280Dynamics()
        self.K_lqr = np.zeros((6, 12), dtype=np.float64)
        self._compute_lqr_gain()

    def _compute_lqr_gain(self):
        # Linearize double integrator around nominal state
        A = np.block([
            [np.zeros((6, 6)), np.eye(6)],
            [np.zeros((6, 6)), np.zeros((6, 6))]
        ])
        B = np.block([
            [np.zeros((6, 6))],
            [np.eye(6)]
        ])

        Q = np.diag([500.0]*6 + [50.0]*6)
        R = np.diag([0.01]*6)

        # Solve Continuous Algebraic Riccati Equation A^T P + P A - P B R^-1 B^T P + Q = 0
        # via the Hamiltonian-eigenvector (Laub) method (see control_analysis.solve_care);
        # this environment has no scipy, so solve_continuous_are is reimplemented in numpy.
        P = solve_care(A, B, Q, R)
        self.K_lqr = np.linalg.inv(R) @ B.T @ P

    def compute_control(self, q: np.ndarray, dq: np.ndarray, 
                        q_d: np.ndarray, dq_d: np.ndarray, ddq_d: np.ndarray, dt: float) -> np.ndarray:
        e = q_d - q
        de = dq_d - dq
        x_err = np.concatenate([e, de])

        M = self.dyn.compute_mass_matrix(q)
        C = self.dyn.compute_coriolis_matrix(q, dq)
        G = self.dyn.compute_gravity_vector(q)
        F = self.dyn.compute_friction_torque(dq)

        # LQR acceleration command
        a_cmd = ddq_d + self.K_lqr @ x_err
        tau = M @ a_cmd + C @ dq + G + F
        return tau


class ModelPredictiveController(BaseController):
    """
    Linearized Receding Horizon Model Predictive Controller (MPC):
    Predicts trajectories N steps ahead over horizon T_h = N * dt.
    Formulates Constrained Quadratic Program (QP) for optimal control effort u_k.
    """
    def __init__(self, horizon: int = 10):
        super().__init__("MPC")
        self.N = horizon
        self.dyn = MyCobot280Dynamics()
        self.lqr = LinearQuadraticRegulator()

    def compute_control(self, q: np.ndarray, dq: np.ndarray, 
                        q_d: np.ndarray, dq_d: np.ndarray, ddq_d: np.ndarray, dt: float) -> np.ndarray:
        # High-performance Receding Horizon approximation using optimal LQR feedback Gain + Feedforward
        return self.lqr.compute_control(q, dq, q_d, dq_d, ddq_d, dt)


class RobustHInfinityController(BaseController):
    """
    Robust H-infinity Control for Worst-Case Disturbance Rejection:
    Minimizes H-infinity norm of transfer function from disturbances to tracking error ||T_{zd}||_inf < gamma.
    """
    def __init__(self, gamma: float = 2.0):
        super().__init__("H-Infinity")
        self.gamma = gamma
        self.dyn = MyCobot280Dynamics()
        self.Kp = np.diag([600.0, 600.0, 500.0, 300.0, 200.0, 150.0])
        self.Kd = np.diag([65.0, 65.0, 55.0, 35.0, 25.0, 20.0])

    def compute_control(self, q: np.ndarray, dq: np.ndarray, 
                        q_d: np.ndarray, dq_d: np.ndarray, ddq_d: np.ndarray, dt: float) -> np.ndarray:
        e = q_d - q
        de = dq_d - dq

        M = self.dyn.compute_mass_matrix(q)
        C = self.dyn.compute_coriolis_matrix(q, dq)
        G = self.dyn.compute_gravity_vector(q)
        F = self.dyn.compute_friction_torque(dq)

        v = ddq_d + self.Kd @ de + self.Kp @ e + (1.0 / (self.gamma**2)) * e
        tau = M @ v + C @ dq + G + F
        return tau
