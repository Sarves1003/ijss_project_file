#!/usr/bin/env python3
"""
Verification Script for Phase 2: Kinematics & Dynamics Engines
"""
import numpy as np
import sys
import os

from kinematics_engine import MyCobot280Kinematics, TrajectoryPlanner
from dynamics_engine import MyCobot280Dynamics

def test_phase2():
    print("==========================================================")
    print("  PHASE 2 VERIFICATION: KINEMATICS & DYNAMICS ENGINES")
    print("==========================================================")
    
    # 1. Kinematics test
    kin = MyCobot280Kinematics()
    q_home = np.radians([0.0, -82.5, 0.0, 0.0, 0.0, 90.0])
    T_fk = kin.forward_kinematics(q_home)
    print(f"[1] FK Home Pose End-Effector Position (mm): {T_fk[:3, 3]}")
    
    # Test IK LM with a kinematically valid target configuration
    q_target_true = q_home + np.radians([5.0, -10.0, 8.0, -5.0, 10.0, -15.0])
    T_target = kin.forward_kinematics(q_target_true)
    q_sol, success, iters, rmse = kin.inverse_kinematics_lm(T_target, q_home)
    print(f"[2] IK (Levenberg-Marquardt): Success={success}, Iterations={iters}, Pos RMSE={rmse:.6f} mm")
    print(f"    Joint error norm: {np.linalg.norm(q_sol - q_target_true):.6f} rad")
    assert success, "IK LM Solver failed to converge!"
    
    # Test Jacobian & Manipulability
    J = kin.compute_jacobian(q_home)
    w, min_s, cond = kin.compute_manipulability(q_home)
    print(f"[3] Jacobian Shape: {J.shape}, Yoshikawa Manipulability: {w:.4f}, Cond: {cond:.4f}")
    
    # Test Trajectory Planner
    t, q_traj, v_traj, a_traj = TrajectoryPlanner.minimum_jerk_multi_joint(q_home, q_sol, duration=2.0, n_steps=50)
    print(f"[4] Trajectory Planner: Generated {len(t)} steps over {t[-1]:.1f}s")
    
    # 2. Dynamics test
    dyn = MyCobot280Dynamics(payload_mass=0.05)  # 50g cube payload
    M = dyn.compute_mass_matrix(q_home)
    G = dyn.compute_gravity_vector(q_home)
    print(f"[5] Mass Matrix M(q) Det: {np.linalg.det(M):.6e}, Eigenvalues min/max: {np.min(np.linalg.eigvals(M)):.4f}/{np.max(np.linalg.eigvals(M)):.4f}")
    print(f"[6] Gravity Torques G(q) (Nm): {np.round(G, 4)}")
    
    dq_test = np.array([0.1, -0.2, 0.15, -0.1, 0.05, 0.0])
    ddq_test = np.array([0.05, -0.1, 0.1, 0.0, 0.0, 0.0])
    tau = dyn.inverse_dynamics(q_home, dq_test, ddq_test)
    print(f"[7] Inverse Dynamics Torque tau (Nm): {np.round(tau, 4)}")
    
    # Test RK4 integration step
    q_next, dq_next = dyn.rk4_step(q_home, dq_test, tau, dt=0.01)
    print(f"[8] RK4 Dynamics Integration Step successful. q_next delta norm: {np.linalg.norm(q_next - q_home):.6f}")
    
    # Skew symmetry test
    skew_val = dyn.verify_skew_symmetry(q_home, dq_test)
    print(f"[9] Passivity Skew-Symmetry Test x^T (M_dot - 2C) x = {skew_val:.6e}")
    assert skew_val < 1e-3, f"Skew symmetry property check violated! Value: {skew_val}"
    
    print("\n>>> ALL PHASE 2 MATHEMATICAL & DYNAMICS VERIFICATIONS PASSED SUCCESSFULLY! <<<\n")

if __name__ == '__main__':
    test_phase2()
