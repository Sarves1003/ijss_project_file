#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures", "vision")
TABDIR = os.path.join(os.path.dirname(__file__), "..", "latex", "tables_v2")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(TABDIR, exist_ok=True)

df = pd.read_csv(os.path.join(RESULTS, "vision_robustness.csv"))

# ---- Figure: 3-panel robustness sweep ----
plt.rcParams.update({
    "font.size": 11.5, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
})
fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))

panel_labels = ["(a) Brightness", "(b) Contrast", "(c) Pixel noise"]
lines_for_legend = None
for ax, factor, xlabel, panel_label in zip(
    axes,
    ["brightness", "contrast", "pixel_noise_std"],
    ["Brightness multiplier", "Contrast multiplier", "Pixel noise std (0-255 scale)"],
    panel_labels,
):
    sub = df[df.factor == factor].sort_values("level")
    ax2 = ax.twinx()
    l1, = ax.plot(sub.level, sub.detection_rate * 100, "o-", color="#1f77b4", ms=5, lw=1.6, label="Detection rate (%)")
    l2, = ax2.plot(sub.level, sub.mean_centroid_error_px, "s--", color="#d62728", ms=5, lw=1.6, label="Centroid error (px)")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Detection rate (%)", color="#1f77b4")
    ax2.set_ylabel("Centroid error (px)", color="#d62728")
    ax.set_ylim(-5, 105)
    ax.tick_params(axis='y', labelcolor="#1f77b4")
    ax2.tick_params(axis='y', labelcolor="#d62728")
    ax.set_xlabel(f"{xlabel}\n{panel_label}")
    if lines_for_legend is None:
        lines_for_legend = [l1, l2]

fig.legend(lines_for_legend, ["Detection rate (%)", "Centroid error (px)"],
           loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.04), framealpha=0.95)
fig.tight_layout(pad=0.7, w_pad=1.8, rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIGDIR, "vision_robustness.pdf"))
fig.savefig(os.path.join(FIGDIR, "vision_robustness.png"), dpi=400)
plt.close(fig)
print("Saved vision_robustness.pdf/png (HD, 400 DPI)")

# ---- Table: compact summary (subset of levels to keep it to ~12 rows) ----
keep = df[df.level.isin([0.3, 0.5, 1.0, 2.0, 0.4, 0.6, 1.2, 0.0, 10.0, 20.0, 35.0])].copy()
keep = keep.sort_values(["factor", "level"])

lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Measured vision-pipeline robustness under systematically varied imaging conditions (40 trials/condition; synthetic-scene test, Section~\ref{sec:simulation}).}",
    r"\label{tab:vision_robustness}",
    r"\small",
    r"\begin{tabular}{llccc}",
    r"\toprule",
    r"\textbf{Factor} & \textbf{Level} & \textbf{Detection rate} & \textbf{Centroid err.\ (px)} & \textbf{Orientation err.\ (deg)} \\",
    r"\midrule",
]
for _, row in keep.iterrows():
    ce = f"{row.mean_centroid_error_px:.2f}" if pd.notna(row.mean_centroid_error_px) else "--"
    oe = f"{row.mean_orientation_error_deg:.2f}" if pd.notna(row.mean_orientation_error_deg) else "--"
    lines.append(f"{row.factor.replace('_',' ')} & {row.level:g} & {row.detection_rate*100:.1f}\\% & {ce} & {oe} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

with open(os.path.join(TABDIR, "vision_robustness_table.tex"), "w") as f:
    f.write("\n".join(lines))
print("Saved vision_robustness_table.tex")
