#!/usr/bin/env python3
import os, json

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
TABDIR = os.path.join(os.path.dirname(__file__), "..", "latex", "tables_v2")
os.makedirs(TABDIR, exist_ok=True)

with open(os.path.join(RESULTS, "optimization_results.json")) as f:
    results = json.load(f)

names = {"PSO": "Particle Swarm Optimization", "GWO": "Grey Wolf Optimizer", "BO": "Bayesian Optimization"}
budgets = {"PSO": "5 particles $\\times$ 5 iter.\\ (30 evals)", "GWO": "5 wolves $\\times$ 5 iter.\\ (30 evals)", "BO": "6 init + 22 EI queries (28 evals)"}

lines = [
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Controller-gain optimization results (real convergence runs, Section~\ref{sec:optimization}).}",
    r"\label{tab:opt_results}",
    r"\small",
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"\textbf{Method} & \textbf{Evaluation budget} & \textbf{Best $k_p$} & \textbf{Best $k_d$} & \textbf{Best $J$} \\",
    r"\midrule",
]
for key in ["PSO", "GWO", "BO"]:
    d = results[key]
    lines.append(f"{names[key]} & {budgets[key]} & {d['best_Kp']:.2f} & {d['best_Kd']:.2f} & {d['best_cost']:.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

with open(os.path.join(TABDIR, "optimization_results_table.tex"), "w") as f:
    f.write("\n".join(lines))
print("Saved optimization_results_table.tex")
