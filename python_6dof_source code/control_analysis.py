#!/usr/bin/env python3
"""
Paper_Project/python/control_analysis.py
=========================================
Numpy-only Linear Systems Analysis Toolkit for the myCobot 280 Manipulator.

Rationale: this build environment has no scipy/control/torch and no reliable
network access to install them (pip downloads consistently time out on this
sandbox). Every routine below is implemented from first principles with numpy
only, so all outputs are genuine computed results, not placeholders:

- Numerical linearization of the open-loop Euler-Lagrange dynamics about an
  operating point (finite-difference Jacobians of the nonlinear state
  equation), giving a continuous-time state-space model (A, B).
- Closed-loop error-dynamics transfer function for the feedback-linearizing
  (Computed Torque) controller, which is exact and analytical (not a small
  signal approximation): each joint's error dynamics is the LTI system
  ddot(e) + Kd*dot(e) + Kp*e = 0.
- Continuous Algebraic Riccati Equation (CARE) solver via the Hamiltonian
  eigenvector method (Laub-style), used for the LQR baseline gain -- replaces
  scipy.linalg.solve_continuous_are.
- Controllability/observability matrix rank tests (Kalman criterion).
- Frequency response (Bode/Nyquist) via direct evaluation of transfer
  functions at s = j*omega.
- Root locus via companion-matrix eigenvalues of the closed-loop
  characteristic polynomial as a function of a swept gain.
"""

import math
import numpy as np
from typing import Tuple, Optional

from kinematics_engine import MyCobot280Kinematics
from dynamics_engine import MyCobot280Dynamics


# ---------------------------------------------------------------------------
# 1. Numerical linearization of the open-loop nonlinear plant
# ---------------------------------------------------------------------------

def nonlinear_state_derivative(x: np.ndarray, u: np.ndarray, dyn: MyCobot280Dynamics) -> np.ndarray:
    """
    Continuous-time nonlinear state equation for the 12-state manipulator model
    x = [q (6); dq (6)], u = tau (6):
        dq/dt      = dq
        ddq/dt     = M(q)^-1 [ u - C(q,dq) dq - G(q) - F(dq) ]
    """
    q, dq = x[:6], x[6:]
    ddq = dyn.forward_dynamics(q, dq, u)
    return np.concatenate([dq, ddq])


def linearize_plant(q0: np.ndarray, dq0: np.ndarray, u0: np.ndarray,
                     dyn: MyCobot280Dynamics, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Central-difference linearization of the nonlinear plant about (q0, dq0, u0):
        A = d(f)/d(x) |_(x0,u0)   (12x12)
        B = d(f)/d(u) |_(x0,u0)   (12x6)
    """
    x0 = np.concatenate([q0, dq0])
    n = x0.shape[0]
    m = u0.shape[0]

    A = np.zeros((n, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        f_plus = nonlinear_state_derivative(x0 + dx, u0, dyn)
        f_minus = nonlinear_state_derivative(x0 - dx, u0, dyn)
        A[:, i] = (f_plus - f_minus) / (2 * eps)

    B = np.zeros((n, m))
    for j in range(m):
        du = np.zeros(m)
        du[j] = eps
        f_plus = nonlinear_state_derivative(x0, u0 + du, dyn)
        f_minus = nonlinear_state_derivative(x0, u0 - du, dyn)
        B[:, j] = (f_plus - f_minus) / (2 * eps)

    return A, B


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    cols = [B]
    Ak = np.eye(n)
    for _ in range(1, n):
        Ak = A @ Ak
        cols.append(Ak @ B)
    return np.hstack(cols)


def observability_matrix(A: np.ndarray, C: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    rows = [C]
    Ak = np.eye(n)
    for _ in range(1, n):
        Ak = Ak @ A
        rows.append(C @ Ak)
    return np.vstack(rows)


# ---------------------------------------------------------------------------
# 2. Continuous Algebraic Riccati Equation via Hamiltonian eigenvectors
# ---------------------------------------------------------------------------

def solve_care(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Solves A^T P + P A - P B R^-1 B^T P + Q = 0 for the stabilizing P >= 0
    using the Hamiltonian-matrix eigenvector (Laub) method:
        H = [[A, -B R^-1 B^T], [-Q, -A^T]]
    The stable invariant subspace (eigenvalues with negative real part) of H,
    partitioned as columns [X1; X2], gives P = X2 X1^-1.
    """
    n = A.shape[0]
    Rinv = np.linalg.inv(R)
    H = np.block([
        [A, -B @ Rinv @ B.T],
        [-Q, -A.T]
    ])
    eigvals, eigvecs = np.linalg.eig(H)
    stable_idx = np.where(eigvals.real < 0)[0]
    if len(stable_idx) != n:
        raise ValueError(f"Expected {n} stable eigenvalues, found {len(stable_idx)}; system may not be stabilizable.")
    V = eigvecs[:, stable_idx]
    X1, X2 = V[:n, :], V[n:, :]
    P = X2 @ np.linalg.inv(X1)
    P = np.real((P + P.conj().T) / 2)
    return P


def lqr_gain(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> np.ndarray:
    P = solve_care(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


# ---------------------------------------------------------------------------
# 3. Closed-loop CTC error-dynamics LTI benchmark system (exact, analytical)
# ---------------------------------------------------------------------------

def ctc_error_poles(Kp: float, Kd: float) -> np.ndarray:
    """
    Roots of the exact per-joint CTC closed-loop characteristic polynomial
        s^2 + Kd s + Kp = 0
    """
    return np.roots([1.0, Kd, Kp])


def ctc_transfer_function_response(Kp: float, Kd: float, omega: np.ndarray) -> np.ndarray:
    """
    Frequency response E(jw)/Ed(jw) of the closed-loop reference-tracking
    transfer function T(s) = Kp / (s^2 + Kd s + Kp) evaluated at s = j*omega.
    (Disturbance-to-error transfer function is S(s) = s^2/(s^2+Kd s+Kp); both
    are derived from the same closed-loop characteristic polynomial.)
    """
    s = 1j * omega
    return Kp / (s**2 + Kd * s + Kp)


def ctc_root_locus(Kp_range: np.ndarray, Kd: float) -> np.ndarray:
    """
    Root locus of the CTC closed-loop poles as the position gain Kp is swept
    with Kd held fixed at its design (critically-damped) value.
    """
    poles = np.array([ctc_error_poles(Kp, Kd) for Kp in Kp_range])
    return poles


# ---------------------------------------------------------------------------
# 4. Statistics (ANOVA, Wilcoxon signed-rank, CI, effect size) -- numpy only
# ---------------------------------------------------------------------------

def one_way_anova(*groups: np.ndarray) -> Tuple[float, float]:
    """
    One-way ANOVA F-statistic and p-value (via the regularized incomplete beta
    function implemented through a numerically stable continued-fraction, to
    avoid a scipy dependency for the F-distribution CDF).
    """
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    grand_mean = np.concatenate(groups).mean()

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)

    df_between = k - 1
    df_within = n_total - k

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within

    p_value = _f_dist_sf(f_stat, df_between, df_within)
    return float(f_stat), float(p_value)


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3.0e-10) -> float:
    """Continued-fraction evaluation for the incomplete beta function (NR 6.4)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta function I_x(a,b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_beta = (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b))
    front = np.exp(np.log(x) * a + np.log(1.0 - x) * b - ln_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    else:
        return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _f_dist_sf(f: float, d1: int, d2: int) -> float:
    """Survival function (1 - CDF) of the F-distribution with (d1, d2) dof."""
    x = d2 / (d2 + d1 * f)
    return _betainc(d2 / 2.0, d1 / 2.0, x)


def wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test (paired, two-sided) using the normal
    approximation with continuity correction (standard practice for n > ~20,
    matches scipy's default asymptotic method for ties-free data).
    """
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    n = len(d)
    ranks = _rankdata(np.abs(d))
    w_pos = np.sum(ranks[d > 0])
    w_neg = np.sum(ranks[d < 0])
    w_stat = min(w_pos, w_neg)

    mean_w = n * (n + 1) / 4.0
    std_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w_stat - mean_w + 0.5) / std_w
    p_value = 2.0 * _norm_sf(abs(z))
    return float(w_stat), float(min(p_value, 1.0))


def _rankdata(a: np.ndarray) -> np.ndarray:
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = obs.cumsum()[inv]
    count = np.r_[np.nonzero(obs)[0], len(obs)]
    ranks_avg = 0.5 * (count[dense] + count[dense - 1] + 1)
    return ranks_avg


def _norm_sf(z: float) -> float:
    """Standard-normal survival function via the erfc identity (numpy has erf)."""
    return 0.5 * _erfc(z / np.sqrt(2.0))


def _erfc(x: float) -> float:
    return 1.0 - _erf(x)


def _erf(x: float) -> float:
    # Abramowitz-Stegun 7.1.26 approximation, |error| < 1.5e-7
    sign = 1 if x >= 0 else -1
    x = abs(x)
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return sign * y


def confidence_interval_95(data: np.ndarray) -> float:
    """Half-width of the 95% CI on the mean, using the normal approximation (n=30 per group)."""
    return 1.96 * np.std(data, ddof=1) / np.sqrt(len(data))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d effect size for two independent samples."""
    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2))
    return float((np.mean(x) - np.mean(y)) / pooled_std)


if __name__ == "__main__":
    # Self-test: linearize about the home configuration and verify controllability.
    kin = MyCobot280Kinematics()
    dyn = MyCobot280Dynamics()
    q0 = np.radians([0.0, -82.5, 0.0, 0.0, 0.0, 90.0])
    dq0 = np.zeros(6)
    u0 = dyn.compute_gravity_vector(q0)  # gravity-compensating torque at equilibrium

    A, B = linearize_plant(q0, dq0, u0, dyn)
    Wc = controllability_matrix(A, B)
    rank_c = np.linalg.matrix_rank(Wc, tol=1e-6)
    print(f"Open-loop linearization about home configuration: A is {A.shape}, B is {B.shape}")
    print(f"Controllability matrix rank: {rank_c} / {A.shape[0]} -> {'CONTROLLABLE' if rank_c == A.shape[0] else 'NOT CONTROLLABLE'}")

    C_full = np.eye(12)
    Wo = observability_matrix(A, C_full)
    rank_o = np.linalg.matrix_rank(Wo, tol=1e-6)
    print(f"Observability (full state) matrix rank: {rank_o} / 12")

    poles = ctc_error_poles(400.0, 40.0)
    print(f"CTC closed-loop poles (Kp=400, Kd=40): {poles}")
