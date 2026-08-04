#!/usr/bin/env python3
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures", "control")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 12, "font.family": "serif", "axes.grid": True,
    "grid.alpha": 0.3, "figure.dpi": 150, "savefig.dpi": 400,
    "savefig.bbox": "tight",
})

with open(os.path.join(RESULTS, "optimization_results.json")) as f:
    results = json.load(f)

fig, ax = plt.subplots(figsize=(6.4, 4.6))
colors = {"PSO": "#1f77b4", "GWO": "#2ca02c", "BO": "#d62728"}
markers = {"PSO": "o", "GWO": "s", "BO": "^"}
for name, data in results.items():
    hist = data["history"]
    ax.plot(range(len(hist)), hist, marker=markers.get(name, "o"), ms=4, lw=1.5,
             label=f"{name} (best $J$={data['best_cost']:.1f})", color=colors.get(name))

ax.set_xlabel("Function evaluation / iteration")
ax.set_ylabel(r"Best-so-far cost $J = 100\cdot\mathrm{ITAE} + 0.01\cdot\mathrm{Effort}$")
ax.legend(fontsize=9.5, framealpha=0.95)
ax.margins(x=0.03, y=0.08)
fig.tight_layout(pad=0.6)
fig.savefig(os.path.join(FIGDIR, "optimization_convergence.pdf"))
fig.savefig(os.path.join(FIGDIR, "optimization_convergence.png"), dpi=400)
plt.close(fig)
print("Saved optimization_convergence.pdf/png (HD, 400 DPI)")

for name, data in results.items():
    print(f"{name}: Kp={data['best_Kp']:.2f}, Kd={data['best_Kd']:.2f}, J={data['best_cost']:.3f}, n_evals={len(data['history'])}")
