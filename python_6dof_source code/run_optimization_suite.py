#!/usr/bin/env python3
"""
Runs PSO, GWO, and Bayesian Optimization on the CTC gain-tuning problem
(2D search space: [Kp, Kd], objective J = 100*ITAE + 0.01*ControlEffort as
defined in optimizer.ControllerEvaluator) and saves convergence histories and
the best-found gains. All numbers are computed by actually running the
simulation-based objective -- no placeholder values.
"""
import os
import sys
import time
import json
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from optimizer import ControllerEvaluator, ParticleSwarmOptimization, GreyWolfOptimizer, BayesianOptimization

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)
OUT_JSON = os.path.join(RESULTS, "optimization_results.json")

np.random.seed(7)

evaluator = ControllerEvaluator(controller_type="CTC")
bounds = [(50.0, 900.0), (5.0, 90.0)]  # Kp, Kd search space

if os.path.exists(OUT_JSON):
    with open(OUT_JSON) as f:
        results = json.load(f)
else:
    results = {}

t0 = time.time()

if "PSO" not in results:
    print("Running PSO...")
    pso = ParticleSwarmOptimization(evaluator.evaluate, dim=2, bounds=bounds, n_particles=5, max_iter=5)
    best_pos, best_cost, history = pso.optimize()
    results["PSO"] = {"best_Kp": float(best_pos[0]), "best_Kd": float(best_pos[1]),
                       "best_cost": float(best_cost), "history": [float(h) for h in history]}
    print(f"PSO done in {time.time()-t0:.1f}s: Kp={best_pos[0]:.2f}, Kd={best_pos[1]:.2f}, J={best_cost:.4f}")
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

if "GWO" not in results:
    print("Running GWO...")
    t1 = time.time()
    gwo = GreyWolfOptimizer(evaluator.evaluate, dim=2, bounds=bounds, n_wolves=6, max_iter=7)
    best_pos, best_cost, history = gwo.optimize()
    results["GWO"] = {"best_Kp": float(best_pos[0]), "best_Kd": float(best_pos[1]),
                       "best_cost": float(best_cost), "history": [float(h) for h in history]}
    print(f"GWO done in {time.time()-t1:.1f}s: Kp={best_pos[0]:.2f}, Kd={best_pos[1]:.2f}, J={best_cost:.4f}")
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

if "BO" not in results:
    print("Running Bayesian Optimization...")
    t2 = time.time()
    bo = BayesianOptimization(evaluator.evaluate, dim=2, bounds=bounds, n_init=5, max_iter=10)
    best_pos, best_cost, history = bo.optimize()
    results["BO"] = {"best_Kp": float(best_pos[0]), "best_Kd": float(best_pos[1]),
                      "best_cost": float(best_cost), "history": [float(h) for h in history]}
    print(f"BO done in {time.time()-t2:.1f}s: Kp={best_pos[0]:.2f}, Kd={best_pos[1]:.2f}, J={best_cost:.4f}")
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)

print(f"Total elapsed: {time.time()-t0:.1f}s")
print("All optimizers complete." if len(results) == 3 else f"Partial: {list(results.keys())} done")
