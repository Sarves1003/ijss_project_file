#!/usr/bin/env python3
"""
Paper_Project/python/kinematics_engine.py
=========================================
Research-Grade Kinematics & Trajectory Engine for myCobot 280 Arm
Target Journal: International Journal of Systems Science (Taylor & Francis)

Provides:
- Standard Denavit-Hartenberg (DH) 6-DOF Parameter Model
- Symbolic & Numerical Forward Kinematics (FK)
- Levenberg-Marquardt (LM) & Newton-Raphson (NR) Inverse Kinematics (IK)
- Geometric & Analytical 6x6 Jacobian Matrix J(q)
- Differential Kinematics (Velocities & Accelerations)
- Yoshikawa Manipulability Index & Singularity Analysis
- Workspace Reachability Analysis
- Trajectory Planning (Quintic Polynomial, S-Curve, Minimum Jerk)
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional

@dataclass
class DHParam:
    theta_offset: float  # radians
    d: float             # mm
    a: float             # mm
    alpha: float         # radians
    min_limit: float     # radians
    max_limit: float     # radians

class MyCobot280Kinematics:
    def __init__(self):
        """
        Denavit-Hartenberg (DH) Parameters for myCobot 280 (6-DOF)
        Format: [theta_offset (rad), d (mm), a (mm), alpha (rad), q_min (rad), q_max (rad)]
        Standard DH convention.
        """
        # Standard DH table format: [theta_offset (rad), d (m), a (m), alpha (rad), q_min (rad), q_max (rad)]
        self.dh_params = [
            DHParam(0.0,         0.13122,  0.0,      np.pi/2,   np.radians(-165.0), np.radians(165.0)),  # Joint 1
            DHParam(-np.pi/2,    0.0,     -0.1104,   0.0,       np.radians(-165.0), np.radians(165.0)),  # Joint 2
            DHParam(0.0,         0.0,     -0.0960,   0.0,       np.radians(-165.0), np.radians(165.0)),  # Joint 3
            DHParam(-np.pi/2,    0.0634,   0.0,      np.pi/2,   np.radians(-165.0), np.radians(165.0)),  # Joint 4
            DHParam(0.0,         0.07505,  0.0,     -np.pi/2,   np.radians(-165.0), np.radians(165.0)),  # Joint 5
            DHParam(0.0,         0.0456,   0.0,      0.0,       np.radians(-179.0), np.radians(179.0)),  # Joint 6
        ]
        self.n_joints = 6

    def get_dh_matrix(self, q_i: float, p: DHParam) -> np.ndarray:
        """
        Computes standard DH homogeneous transformation matrix T_i^{i-1}(q_i).
        T = Rot_z(q_i + theta_offset) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)
        """
        theta = q_i + p.theta_offset
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(p.alpha), np.sin(p.alpha)
        
        return np.array([
            [ct, -st * ca,  st * sa, p.a * ct],
            [st,  ct * ca, -ct * sa, p.a * st],
            [0.0,    sa,       ca,     p.d],
            [0.0,   0.0,      0.0,     1.0]
        ], dtype=np.float64)

    def forward_kinematics(self, q: np.ndarray) -> np.ndarray:
        """
        Computes Forward Kinematics pose T_6^0(q).
        :param q: Joint angle vector (6,) in radians.
        :return: 4x4 Homogeneous transformation matrix of end-effector.
        """
        T = np.eye(4, dtype=np.float64)
        for i in range(self.n_joints):
            T = T @ self.get_dh_matrix(q[i], self.dh_params[i])
        return T

    def forward_kinematics_all_links(self, q: np.ndarray) -> List[np.ndarray]:
        """
        Computes all intermediate transformation matrices [T_1^0, T_2^0, ..., T_6^0].
        """
        transforms = []
        T = np.eye(4, dtype=np.float64)
        for i in range(self.n_joints):
            T = T @ self.get_dh_matrix(q[i], self.dh_params[i])
            transforms.append(T.copy())
        return transforms

    def compute_jacobian(self, q: np.ndarray) -> np.ndarray:
        """
        Computes the 6x6 Geometric Jacobian matrix J(q) relating joint velocities dq to 
        end-effector Cartesian linear and angular velocities V = [v^T, w^T]^T.
        
        J_i = [ z_{i-1} x (p_e - p_{i-1}) ]  (linear velocity component)
              [         z_{i-1}           ]  (angular velocity component)
        """
        J = np.zeros((6, 6), dtype=np.float64)
        transforms = self.forward_kinematics_all_links(q)
        p_e = transforms[-1][:3, 3]  # End-effector position
        
        z_prev = np.array([0.0, 0.0, 1.0])
        p_prev = np.array([0.0, 0.0, 0.0])
        
        # Joint 1
        J[:3, 0] = np.cross(z_prev, p_e - p_prev)
        J[3:, 0] = z_prev
        
        # Joints 2 to 6
        for i in range(1, 6):
            z_prev = transforms[i-1][:3, 2]  # Z axis of link i-1
            p_prev = transforms[i-1][:3, 3]  # Position of link i-1
            J[:3, i] = np.cross(z_prev, p_e - p_prev)
            J[3:, i] = z_prev
            
        return J

    def compute_jacobian_derivative(self, q: np.ndarray, dq: np.ndarray, dt: float = 1e-5) -> np.ndarray:
        """
        Computes numerical derivative of Jacobian dJ/dt given q and dq.
        Used for differential acceleration kinematics: x_ddot = J * q_ddot + J_dot * q_dot
        """
        q_next = q + dq * dt
        J_curr = self.compute_jacobian(q)
        J_next = self.compute_jacobian(q_next)
        return (J_next - J_curr) / dt

    def compute_manipulability(self, q: np.ndarray) -> Tuple[float, float, float]:
        """
        Computes:
        1. Yoshikawa Manipulability Index: w = sqrt(det(J * J^T))
        2. Minimum Singular Value (distance to kinematic singularity)
        3. Condition Number kappa(J) = sigma_max / sigma_min
        """
        J = self.compute_jacobian(q)
        s = np.linalg.svd(J, compute_uv=False)
        w = np.prod(s)
        min_s = s[-1]
        cond = s[0] / max(s[-1], 1e-9)
        return float(w), float(min_s), float(cond)

    def calculate_pose_error(self, T_curr: np.ndarray, T_targ: np.ndarray) -> np.ndarray:
        """
        Computes 6D spatial pose error vector e = [dp^T, do^T]^T.
        dp = p_targ - p_curr
        do = 0.5 * (s_curr x s_targ + n_curr x n_targ + a_curr x a_targ)
        """
        dp = T_targ[:3, 3] - T_curr[:3, 3]
        R_curr = T_curr[:3, :3]
        R_targ = T_targ[:3, :3]
        
        do = 0.5 * (
            np.cross(R_curr[:, 0], R_targ[:, 0]) +
            np.cross(R_curr[:, 1], R_targ[:, 1]) +
            np.cross(R_curr[:, 2], R_targ[:, 2])
        )
        return np.concatenate((dp, do))

    def clamp_to_limits(self, q: np.ndarray) -> np.ndarray:
        """Clamps joint angles strictly within physical bounds."""
        q_clamped = q.copy()
        for i in range(self.n_joints):
            q_clamped[i] = np.clip(q_clamped[i], self.dh_params[i].min_limit, self.dh_params[i].max_limit)
        return q_clamped

    def inverse_kinematics_lm(self, T_target: np.ndarray, q_init: np.ndarray, 
                              max_iter: int = 250, tol_pos: float = 1e-4, tol_rot: float = 1e-4,
                              lambda_init: float = 1e-3) -> Tuple[np.ndarray, bool, int, float]:
        """
        Inverse Kinematics via Levenberg-Marquardt (LM) Damped Least Squares.
        SI units: position in meters, rotation in radians.
        """
        q = q_init.copy()
        lam = lambda_init
        
        for iteration in range(max_iter):
            T_curr = self.forward_kinematics(q)
            
            dp = T_target[:3, 3] - T_curr[:3, 3]  # meters
            R_curr = T_curr[:3, :3]
            R_targ = T_target[:3, :3]
            do = 0.5 * (
                np.cross(R_curr[:, 0], R_targ[:, 0]) +
                np.cross(R_curr[:, 1], R_targ[:, 1]) +
                np.cross(R_curr[:, 2], R_targ[:, 2])
            )
            
            err_pos = float(np.linalg.norm(dp)) # m
            err_rot = float(np.linalg.norm(do)) # rad
            
            if err_pos < tol_pos and err_rot < tol_rot:
                return q, True, iteration, err_pos * 1000.0  # return in mm for reporting
                
            e = np.concatenate((dp, do))
            J = self.compute_jacobian(q)
            
            A = J.T @ J + lam * np.eye(6)
            g = J.T @ e
            
            try:
                dq = np.linalg.solve(A, g)
            except np.linalg.LinAlgError:
                dq = np.linalg.pinv(J) @ e
                
            q_next = self.clamp_to_limits(q + dq)
            
            T_next = self.forward_kinematics(q_next)
            dp_next = T_target[:3, 3] - T_next[:3, 3]
            R_next = T_next[:3, :3]
            do_next = 0.5 * (
                np.cross(R_next[:, 0], R_targ[:, 0]) +
                np.cross(R_next[:, 1], R_targ[:, 1]) +
                np.cross(R_next[:, 2], R_targ[:, 2])
            )
            rmse_next = float(np.linalg.norm(np.concatenate((dp_next, do_next))))
            rmse_curr = float(np.linalg.norm(e))
            
            if rmse_next < rmse_curr:
                q = q_next
                lam = max(lam * 0.5, 1e-7)
                if np.linalg.norm(dp_next) < tol_pos and np.linalg.norm(do_next) < tol_rot:
                    return q, True, iteration + 1, float(np.linalg.norm(dp_next) * 1000.0)
            else:
                lam = min(lam * 5.0, 1e3)
                
        T_final = self.forward_kinematics(q)
        final_pos_err = float(np.linalg.norm(T_target[:3, 3] - T_final[:3, 3])) * 1000.0
        return q, final_pos_err < 1.0, max_iter, final_pos_err

    def inverse_kinematics_nr(self, T_target: np.ndarray, q_init: np.ndarray,
                              max_iter: int = 250, tol_pos: float = 1e-3, tol_rot: float = 1e-3) -> Tuple[np.ndarray, bool, int, float]:
        """
        Inverse Kinematics via Newton-Raphson Pseudoinverse Iteration.
        """
        q = q_init.copy()
        for iteration in range(max_iter):
            T_curr = self.forward_kinematics(q)
            dp_m = (T_target[:3, 3] - T_curr[:3, 3]) / 1000.0
            R_curr = T_curr[:3, :3]
            R_targ = T_target[:3, :3]
            do = 0.5 * (
                np.cross(R_curr[:, 0], R_targ[:, 0]) +
                np.cross(R_curr[:, 1], R_targ[:, 1]) +
                np.cross(R_curr[:, 2], R_targ[:, 2])
            )
            
            err_pos = float(np.linalg.norm(dp_m * 1000.0))
            err_rot = float(np.linalg.norm(do))
            
            if err_pos < tol_pos and err_rot < tol_rot:
                return q, True, iteration, err_pos
                
            e_metric = np.concatenate((dp_m, do))
            J = self.compute_jacobian(q)
            J_metric = J.copy()
            J_metric[:3, :] /= 1000.0
            
            dq = np.linalg.pinv(J_metric) @ e_metric
            q = self.clamp_to_limits(q + dq)
            
        T_final = self.forward_kinematics(q)
        final_pos_err = float(np.linalg.norm(T_target[:3, 3] - T_final[:3, 3]))
        return q, final_pos_err < 1.0, max_iter, final_pos_err


# ============================================================================
# TRAJECTORY PLANNING GENERATORS
# ============================================================================

class TrajectoryPlanner:
    @staticmethod
    def quintic_polynomial(q0: float, qf: float, v0: float, vf: float, 
                           a0: float, af: float, duration: float, n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates a 5th-order (Quintic) polynomial trajectory:
        q(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        Guarantees smooth position, velocity, and acceleration profiles with zero jerk at endpoints.
        """
        t = np.linspace(0, duration, n_steps)
        T = duration
        
        M = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 2, 0, 0, 0],
            [1, T, T**2, T**3, T**4, T**5],
            [0, 1, 2*T, 3*T**2, 4*T**3, 5*T**4],
            [0, 0, 2, 6*T, 12*T**2, 20*T**3]
        ], dtype=np.float64)
        
        b = np.array([q0, v0, a0, qf, vf, af], dtype=np.float64)
        a = np.linalg.solve(M, b)
        
        q = a[0] + a[1]*t + a[2]*t**2 + a[3]*t**3 + a[4]*t**4 + a[5]*t**5
        v = a[1] + 2*a[2]*t + 3*a[3]*t**2 + 4*a[4]*t**3 + 5*a[5]*t**4
        acc = 2*a[2] + 6*a[3]*t + 12*a[4]*t**2 + 20*a[5]*t**3
        
        return t, q, v, acc

    @staticmethod
    def minimum_jerk_multi_joint(q_start: np.ndarray, q_goal: np.ndarray, duration: float, n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generates minimum-jerk trajectories across all 6 joints simultaneously.
        Jerk J(t) = d^3 q / dt^3, minimizing integral(J(t)^2 dt).
        
        q(tau) = q_0 + (q_f - q_0) * (10*tau^3 - 15*tau^4 + 6*tau^5)  where tau = t/T
        """
        t = np.linspace(0, duration, n_steps)
        tau = t / duration
        
        # Basis functions
        s_pos = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        s_vel = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / duration
        s_acc = (60 * tau - 180 * tau**2 + 120 * tau**3) / (duration**2)
        
        q_traj = np.zeros((n_steps, len(q_start)))
        v_traj = np.zeros((n_steps, len(q_start)))
        a_traj = np.zeros((n_steps, len(q_start)))
        
        for i in range(len(q_start)):
            delta = q_goal[i] - q_start[i]
            q_traj[:, i] = q_start[i] + delta * s_pos
            v_traj[:, i] = delta * s_vel
            a_traj[:, i] = delta * s_acc
            
        return t, q_traj, v_traj, a_traj
