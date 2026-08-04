#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from control_analysis import one_way_anova, wilcoxon_signed_rank, confidence_interval_95, cohens_d

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
TABDIR = os.path.join(os.path.dirname(__file__), "..", "latex", "tables_v2")
os.makedirs(TABDIR, exist_ok=True)

import json
OPT_RESULTS_PATH = os.path.join(RESULTS, "optimization_results.json")
with open(OPT_RESULTS_PATH) as _f:
    _opt_results = json.load(_f)
_best_name = min(_opt_results, key=lambda k: _opt_results[k]["best_cost"])
OPT_KP = _opt_results[_best_name]["best_Kp"]
OPT_KD = _opt_results[_best_name]["best_Kd"]

df = pd.read_csv(os.path.join(RESULTS, "progressive_controller_comparison.csv"))
pid = df[df.Controller == "PID"].sort_values("Trial")["RMSE_mm"].values
ctc = df[df.Controller == "CTC"].sort_values("Trial")["RMSE_mm"].values
opt = df[df.Controller == "CTC-Opt"].sort_values("Trial")["RMSE_mm"].values
rl = df[df.Controller == "RL"].sort_values("Trial")["RMSE_mm"].values

f, p = one_way_anova(pid, ctc, opt, rl)

lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Progressive controller comparison: tracking RMSE over 20 matched-seed held-out trials, with statistical tests (Section~\ref{sec:simulation}). CTC-" + _best_name + r" uses gains $K_p{=}" + f"{OPT_KP:.1f}" + r"$, $K_d{=}" + f"{OPT_KD:.1f}" + r"$ from Table~\ref{tab:opt_results}.}",
    r"\label{tab:progressive_comparison}",
    r"\small",
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"\textbf{Stage} & \textbf{RMSE (mm)} & \textbf{95\% CI} & \textbf{Mean energy (J)} & \textbf{Mean payload (kg)} \\",
    r"\midrule",
]
groups = [
    ("PID", pid, "Stage 1: PID"),
    ("CTC", ctc, "Stage 2: CTC (hand-tuned)"),
    ("CTC-Opt", opt, f"Stage 2b: CTC-{_best_name} (optimizer-tuned)"),
    ("RL", rl, "Stage 3: RL-scheduled CTC"),
]
for name, arr, label in groups:
    sub = df[df.Controller == name]
    ci = confidence_interval_95(arr)
    lines.append(f"{label} & {np.mean(arr):.1f} $\\pm$ {np.std(arr,ddof=1):.1f} & $\\pm${ci:.2f} & {sub.ControlEnergy_J.mean():.1f} & {sub.Payload_kg.mean():.3f} \\\\")
lines += [
    r"\midrule",
    f"\\multicolumn{{5}}{{l}}{{One-way ANOVA (4 groups): $F={f:.2f}$, $p={p:.2e}$}} \\\\",
]

for label, a, b in [("CTC vs.\\ PID", ctc, pid), (f"CTC-{_best_name} vs.\\ CTC", opt, ctc),
                     ("RL vs.\\ CTC", rl, ctc), ("RL vs.\\ PID", rl, pid), (f"RL vs.\\ CTC-{_best_name}", rl, opt)]:
    w, pw = wilcoxon_signed_rank(a, b)
    d = cohens_d(a, b)
    lines.append(f"\\multicolumn{{5}}{{l}}{{Wilcoxon {label}: $W={w:.1f}$, $p={pw:.2e}$, Cohen's $d={d:.2f}$}} \\\\")

lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

with open(os.path.join(TABDIR, "progressive_comparison_table.tex"), "w") as f_out:
    f_out.write("\n".join(lines))
print("Saved progressive_comparison_table.tex")
print(f"ANOVA F={f:.4f} p={p:.6e}")
