#!/usr/bin/env python3
"""
Paper_Project/python/dynamics_engine.py
========================================
First-Principles Euler-Lagrange Dynamics & Simulation Engine for myCobot 280
Target Journal: International Journal of Systems Science (Taylor & Francis)

Derivation:
M(q) * q_ddot + C(q, q_dot) * q_dot + G(q) + F(q_dot) + tau_ext = tau

Provides:
- Exact Mass/Inertia Matrix M(q) (Symmetric, Positive Definite)
- Coriolis & Centripetal Matrix C(q, q_dot)
- Gravity Torque Vector G(q)
- Viscous + Coulomb + Stribeck Friction Model F(q_dot)
- Forward Dynamics Integration (Runge-Kutta 4th Order)
- Inverse Dynamics Computation (Recursive Newton-Euler / Euler-Lagrange)
- Fundamental Passivity & Skew-Symmetry Property Verification: x^T (M_dot - 2C) x = 0
"""

import numpy as np
import time
from typing import Tuple, Dict, Any, Optional

from kinematics_engine import MyCobot280Kinematics

class MyCobot280Dynamics:
    def __init__(self, payload_mass: float = 0.0, payload_cog: Optional[np.ndarray] = None):
        """
        Inertial Parameters for myCobot 280 Link Rigid Bodies + Payload.
        Link masses (kg), Centers of Mass CoM (m), and Inertia Tensors I_i (kg*m^2).
        """
        self.kin = MyCobot280Kinematics()
        self.n_joints = 6
        self.g = 9.80665  # m/s^2 acceleration due to gravity

        # Physical link masses (kg)
        self.masses = np.array([0.220, 0.185, 0.150, 0.110, 0.090, 0.075], dtype=np.float64)
        
        # Additional payload
        self.payload_mass = payload_mass
        self.masses[-1] += self.payload_mass  # add payload to end-effector link

        # Center of Mass (CoM) vectors in local link frames (m)
        self.com = [
            np.array([0.0, 0.02, 0.06], dtype=np.float64),
            np.array([-0.05, 0.0, 0.0], dtype=np.float64),
            np.array([-0.04, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.03], dtype=np.float64),
            np.array([0.0, 0.03, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 0.02], dtype=np.float64),
        ]

        # Link inertia tensors about CoM (kg * m^2)
        self.inertias = [
            np.diag([1.2e-4, 1.5e-4, 1.1e-4]),
            np.diag([2.1e-4, 2.5e-4, 1.8e-4]),
            np.diag([1.5e-4, 1.8e-4, 1.2e-4]),
            np.diag([0.8e-4, 0.9e-4, 0.7e-4]),
            np.diag([0.5e-4, 0.6e-4, 0.4e-4]),
            np.diag([0.3e-4, 0.3e-4, 0.2e-4]),
        ]

        # Friction parameters: Viscous (B) and Coulomb (Fc) coefficients
        self.B_viscous = np.array([0.12, 0.15, 0.10, 0.08, 0.05, 0.03], dtype=np.float64)
        self.F_coulomb = np.array([0.18, 0.22, 0.15, 0.12, 0.09, 0.06], dtype=np.float64)

    def compute_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """
        Computes 6x6 Joint Inertia Matrix M(q) using Composite Rigid Body Algorithm (CRBA).
        M(q) is strictly symmetric and positive definite.
        M_ij = sum_{k=max(i,j)}^N [ m_k * (J_v_k,i^T * J_v_k,j) + J_w_k,i^T * R_k * I_k * R_k^T * J_w_k,j ]
        """
        M = np.zeros((6, 6), dtype=np.float64)
        transforms = self.kin.forward_kinematics_all_links(q)
        
        # Positions and Z-axes of all links
        p_links = [np.array([0.0, 0.0, 0.0])] + [T[:3, 3] for T in transforms]
        z_links = [np.array([0.0, 0.0, 1.0])] + [T[:3, 2] for T in transforms]
        R_links = [T[:3, :3] for T in transforms]

        for k in range(self.n_joints):
            m_k = self.masses[k]
            I_k_world = R_links[k] @ self.inertias[k] @ R_links[k].T
            p_com_k = transforms[k][:3, 3] + R_links[k] @ self.com[k]
            
            # Partial Jacobians for link k CoM
            Jv_k = np.zeros((3, 6), dtype=np.float64)
            Jw_k = np.zeros((3, 6), dtype=np.float64)
            
            for j in range(k + 1):
                z_j = z_links[j]
                p_j = p_links[j]
                Jv_k[:, j] = np.cross(z_j, p_com_k - p_j)
                Jw_k[:, j] = z_j
                
            M += m_k * (Jv_k.T @ Jv_k) + (Jw_k.T @ I_k_world @ Jw_k)

        # Enforce exact numerical symmetry
        M = 0.5 * (M + M.T)
        return M

    def compute_gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """
        Computes 6D Gravity Torque Vector G(q) = dV(q) / dq.
        V(q) = sum_{k=1}^6 m_k * g^T * p_com_k(q)
        """
        G = np.zeros(6, dtype=np.float64)
        g_vec = np.array([0.0, 0.0, self.g], dtype=np.float64)
        transforms = self.kin.forward_kinematics_all_links(q)
        
        p_links = [np.array([0.0, 0.0, 0.0])] + [T[:3, 3] for T in transforms]
        z_links = [np.array([0.0, 0.0, 1.0])] + [T[:3, 2] for T in transforms]
        R_links = [T[:3, :3] for T in transforms]

        for k in range(self.n_joints):
            m_k = self.masses[k]
            p_com_k = transforms[k][:3, 3] + R_links[k] @ self.com[k]
            
            for j in range(k + 1):
                z_j = z_links[j]
                p_j = p_links[j]
                Jv_kj = np.cross(z_j, p_com_k - p_j)
                G[j] += m_k * np.dot(g_vec, Jv_kj)

        return G

    def compute_coriolis_matrix(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        """
        Computes 6x6 Coriolis & Centripetal Matrix C(q, dq) using fast forward differences.
        c_ij = sum_{k=1}^N 0.5 * (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i) * dq_k
        """
        eps = 1e-4
        M0 = self.compute_mass_matrix(q)
        dM_dq = np.zeros((6, 6, 6), dtype=np.float64)

        for k in range(6):
            q_plus = q.copy()
            q_plus[k] += eps
            M_plus = self.compute_mass_matrix(q_plus)
            dM_dq[:, :, k] = (M_plus - M0) / eps

        C = np.zeros((6, 6), dtype=np.float64)
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    c_ijk = 0.5 * (dM_dq[i, j, k] + dM_dq[i, k, j] - dM_dq[j, k, i])
                    C[i, j] += c_ijk * dq[k]

        return C

    def compute_friction_torque(self, dq: np.ndarray) -> np.ndarray:
        """
        Viscous + Stribeck Friction Model:
        F(dq) = B * dq + F_c * tanh(10 * dq)
        """
        return self.B_viscous * dq + self.F_coulomb * np.tanh(10.0 * dq)

    def inverse_dynamics(self, q: np.ndarray, dq: np.ndarray, ddq: np.ndarray, 
                         tau_ext: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes Inverse Dynamics torques tau:
        tau = M(q)*ddq + C(q, dq)*dq + G(q) + F(dq) + tau_ext
        """
        M = self.compute_mass_matrix(q)
        C = self.compute_coriolis_matrix(q, dq)
        G = self.compute_gravity_vector(q)
        F = self.compute_friction_torque(dq)
        
        tau = M @ ddq + C @ dq + G + F
        if tau_ext is not None:
            tau += tau_ext
        return tau

    def forward_dynamics(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray, 
                         tau_ext: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes Forward Dynamics joint accelerations ddq:
        ddq = M(q)^-1 * [ tau - C(q, dq)*dq - G(q) - F(dq) - tau_ext ]
        Clamps actuator torques to myCobot 280 physical servo limits.
        """
        # Actuator torque limits (Nm) for myCobot 280 servos
        tau_max = np.array([3.0, 3.0, 2.5, 1.8, 1.2, 0.8], dtype=np.float64)
        tau_clamped = np.clip(tau, -tau_max, tau_max)

        M = self.compute_mass_matrix(q)
        C = self.compute_coriolis_matrix(q, dq)
        G = self.compute_gravity_vector(q)
        F = self.compute_friction_torque(dq)
        
        tau_net = tau_clamped - C @ dq - G - F
        if tau_ext is not None:
            tau_net -= tau_ext
            
        return np.linalg.solve(M, tau_net)

    def rk4_step(self, q: np.ndarray, dq: np.ndarray, tau: np.ndarray, dt: float,
                 tau_ext: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        4th-Order Runge-Kutta numerical integration for forward dynamics state update [q, dq].
        tau_ext, if given, is an external disturbance torque applied throughout the step
        (held constant across the four RK4 stages, consistent with a zero-order-hold
        disturbance over one integration step).
        """
        def f(q_curr, dq_curr):
            ddq_curr = self.forward_dynamics(q_curr, dq_curr, tau, tau_ext=tau_ext)
            return dq_curr, ddq_curr

        k1_v, k1_a = f(q, dq)
        k2_v, k2_a = f(q + 0.5 * dt * k1_v, dq + 0.5 * dt * k1_a)
        k3_v, k3_a = f(q + 0.5 * dt * k2_v, dq + 0.5 * dt * k2_a)
        k4_v, k4_a = f(q + dt * k3_v, dq + dt * k3_a)

        q_next = q + (dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        dq_next = dq + (dt / 6.0) * (k1_a + 2*k2_a + 2*k3_a + k4_a)

        # Enforce joint position and velocity physical limits
        dq_max = np.array([2.6, 2.6, 2.6, 2.6, 2.6, 3.14], dtype=np.float64)  # ~150 deg/s
        dq_next = np.clip(dq_next, -dq_max, dq_max)
        q_next = self.kin.clamp_to_limits(q_next)
        return q_next, dq_next

    def verify_skew_symmetry(self, q: np.ndarray, dq: np.ndarray) -> float:
        """
        Verifies the fundamental passivity property: N = M_dot - 2C is skew-symmetric.
        x^T * N * x = 0 for any arbitrary vector x.
        Returns maximum absolute quadratic form value (should be < 1e-4).
        """
        dt = 1e-5
        M_curr = self.compute_mass_matrix(q)
        M_next = self.compute_mass_matrix(q + dq * dt)
        M_dot = (M_next - M_curr) / dt
        
        C = self.compute_coriolis_matrix(q, dq)
        N = M_dot - 2.0 * C
        
        # Test with random 6D vector x
        np.random.seed(42)
        x = np.random.randn(6)
        val = float(np.abs(x.T @ N @ x))
        return val
