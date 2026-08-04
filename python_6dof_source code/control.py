#!/usr/bin/env python3
"""
Generates publication-quality, HD, caption-only (no in-image titles) figures
from real computed data:
  1. CTC root locus (Kp swept, Kd fixed at critically-damped design value)
  2. CTC closed-loop Bode plot (magnitude + phase)
  3. CTC closed-loop Nyquist plot
  4. ES training convergence (real, logged during rl_gain_scheduler.py training)
  5. Progressive controller comparison (PID -> CTC -> CTC-Opt -> RL), from
     progressive_controller_comparison.csv
No placeholder data -- every curve is computed or loaded from a saved CSV.
All figures are saved as both vector PDF (for LaTeX embedding) and 400 DPI
PNG (for quick preview / HD raster use), directly into the project's
figures/control folder inside the user's workspace so they persist locally.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from control_analysis import ctc_error_poles, ctc_transfer_function_response, ctc_root_locus

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures", "control")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 12, "font.family": "serif", "axes.grid": True,
    "grid.alpha": 0.3, "figure.dpi": 150, "savefig.dpi": 400,
    "savefig.bbox": "tight", "axes.titlesize": 12, "axes.labelsize": 12,
})


def save(fig, name):
    fig.savefig(os.path.join(FIGDIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGDIR, f"{name}.png"), dpi=400)
    plt.close(fig)
    print(f"Saved {name}.pdf/png (HD, 400 DPI)")


# ---------------------------------------------------------------------
# 1. Root locus: Kd fixed at 40 (critically-damped design value), Kp swept
# ---------------------------------------------------------------------
Kd_design = 40.0
Kp_range = np.linspace(1.0, 1200.0, 400)
poles = ctc_root_locus(Kp_range, Kd_design)

fig, ax = plt.subplots(figsize=(5.6, 4.8))
ax.plot(poles.real.flatten(), poles.imag.flatten(), '.', ms=3, color="#1f77b4",
        label=r"Locus as $K_p$ varies (0-1200)")
ax.axvline(0, color='k', lw=0.7)
ax.axhline(0, color='k', lw=0.7)
Kp_design = 400.0
p_design = ctc_error_poles(Kp_design, Kd_design)
ax.plot(p_design.real, p_design.imag, 'r*', ms=16, mec='k', mew=0.5,
         label=r"Design point $K_p{=}400,K_d{=}40$", zorder=5)
ax.set_xlabel("Real axis")
ax.set_ylabel("Imaginary axis")
ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95)
ax.margins(x=0.08, y=0.12)
fig.tight_layout(pad=0.6)
save(fig, "root_locus")

# ---------------------------------------------------------------------
# 2. Bode plot at the design point
# ---------------------------------------------------------------------
omega = np.logspace(-1, 3, 500)
resp = ctc_transfer_function_response(Kp_design, Kd_design, omega)
mag_db = 20 * np.log10(np.abs(resp))
phase_deg = np.unwrap(np.angle(resp)) * 180 / np.pi

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.8, 5.6), sharex=True)
ax1.semilogx(omega, mag_db, color="#1f77b4", lw=1.6)
ax1.set_ylabel("Magnitude (dB)")
ax1.axhline(-3, color='gray', ls='--', lw=0.9, label="-3 dB")
ax1.legend(fontsize=9, loc="lower left", framealpha=0.95)
ax1.margins(x=0.02, y=0.15)

ax2.semilogx(omega, phase_deg, color="#d62728", lw=1.6)
ax2.set_xlabel("Frequency (rad/s)")
ax2.set_ylabel("Phase (deg)")
ax2.margins(x=0.02, y=0.15)
fig.align_ylabels([ax1, ax2])
fig.tight_layout(pad=0.6, h_pad=1.2)
save(fig, "bode_plot")

# ---------------------------------------------------------------------
# 3. Nyquist plot at the design point
# ---------------------------------------------------------------------
omega_nyq = np.linspace(-500, 500, 4000)
omega_nyq = omega_nyq[np.abs(omega_nyq) > 1e-6]
resp_nyq = ctc_transfer_function_response(Kp_design, Kd_design, omega_nyq)

fig, ax = plt.subplots(figsize=(5.2, 5.2))
ax.plot(resp_nyq.real, resp_nyq.imag, color="#1f77b4", lw=1.4)
ax.plot(-1, 0, 'r+', ms=14, mew=2.4, label="Critical point (-1, 0)")
ax.set_xlabel("Real axis")
ax.set_ylabel("Imaginary axis")
ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
ax.set_aspect("equal", adjustable="box")
ax.margins(0.1)
fig.tight_layout(pad=0.6)
save(fig, "nyquist_plot")

# ---------------------------------------------------------------------
# 4. ES training convergence (real logged data)
# ---------------------------------------------------------------------
log_path = os.path.join(RESULTS, "rl_training_log.csv")
if os.path.exists(log_path):
    log = pd.read_csv(log_path)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.plot(log["generation"], log["mean_cost"], marker='o', ms=4, lw=1.5,
             label="Population mean cost", color="#1f77b4")
    ax.plot(log["generation"], log["best_cost"], marker='s', ms=4, lw=1.5,
             label="Population best cost", color="#2ca02c")
    ax.set_xlabel("ES generation")
    ax.set_ylabel("Episode cost (RMSE mm + energy penalty)")
    ax.legend(fontsize=9.5, framealpha=0.95)
    ax.margins(x=0.03, y=0.1)
    fig.tight_layout(pad=0.6)
    save(fig, "rl_convergence")
else:
    print("WARNING: rl_training_log.csv not found -- skipping convergence figure")

# ---------------------------------------------------------------------
# 5. Progressive controller comparison (real evaluation data, 4 stages)
# ---------------------------------------------------------------------
comp_path = os.path.join(RESULTS, "progressive_controller_comparison.csv")
opt_json_path = os.path.join(RESULTS, "optimization_results.json")
if os.path.exists(comp_path):
    df = pd.read_csv(comp_path)
    best_algo = "PSO"
    if os.path.exists(opt_json_path):
        with open(opt_json_path) as f:
            opt_results = json.load(f)
        best_algo = min(opt_results, key=lambda k: opt_results[k]["best_cost"])

    order = ["PID", "CTC", "CTC-Opt", "RL"]
    labels = ["PID\n(conventional)", "CTC\n(hand-tuned)", f"CTC-{best_algo}\n(optimizer-tuned)", "RL-scheduled\nCTC"]
    means = [df[df.Controller == c]["RMSE_mm"].mean() for c in order]
    stds = [df[df.Controller == c]["RMSE_mm"].std() for c in order]
    energies = [df[df.Controller == c]["ControlEnergy_J"].mean() for c in order]

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    x = np.arange(len(order))
    bars = ax.bar(x, means, yerr=stds, capsize=5, width=0.6,
                   color=["#d62728", "#1f77b4", "#9467bd", "#2ca02c"], alpha=0.88,
                   edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Tracking RMSE (mm), mean $\\pm$ SD, 20 test trials")
    top = max(means[i] + stds[i] for i in range(len(means)))
    for bar, m, s, e in zip(bars, means, stds, energies):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + top * 0.03,
                 f"{m:.1f} mm\n({e:.1f} J)", ha='center', va='bottom', fontsize=9)
    ax.set_ylim(0, top * 1.28)
    ax.margins(x=0.06)
    fig.tight_layout(pad=0.6)
    save(fig, "progressive_comparison")
else:
    print("WARNING: progressive_controller_comparison.csv not found")

print("Done.")
