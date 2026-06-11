"""
Executes the comprehensive 1D Poisson equation benchmarks.

This module orchestrates the sequential execution of the simulation sweeps 
defined in Sections IV A-D of the primary reference literature. It handles 
system instantiation, algorithm execution pacing, metric aggregation, and 
automated data exportation to standard output and CSV formats.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from core.config import SimConfig1D
from problems.poisson_1d import PoissonProblem1D
from solvers.classical.thomas import thomas_solve
from solvers.quantum.hhl_1d import hhl_solve
from benchmark.metrics import BenchmarkResult, compute_errors
from benchmark.reporting import print_result_table, print_hhl_summary
from benchmark.plotting import plot_sweep_pairs

# ── Global Execution Directives ───────────────────────────────────────────────

OUTPUT_CSV  = True          # Dictates CSV exportation to the results directory
VERBOSE_HHL = True          # Enables high-granularity node-by-node HHL reporting
RESULTS_DIR = Path("results")


# ── Execution Subroutines ─────────────────────────────────────────────────────

def run_pair(cfg: SimConfig1D) -> tuple[BenchmarkResult, BenchmarkResult]:
    """
    Instantiates the discretised problem and sequentially executes both the 
    classical Thomas and quantum HHL algorithms.

    The classical baseline is explicitly executed prior to the quantum solver 
    to ensure the Thomas solution vector is available for absolute error 
    computation during non-homogeneous boundary condition evaluations.
    """
    problem = PoissonProblem1D(cfg)
    print(f"\n  → {problem.summary()}")

    # Classical Reference Execution (O(N) temporal complexity)
    t0 = time.perf_counter()
    thomas_sr = thomas_solve(problem)
    t_thomas  = time.perf_counter() - t0

    # Quantum Solver Execution (Simulated statevector extraction)
    t0 = time.perf_counter()
    hhl_sr    = hhl_solve(problem)
    t_hhl     = time.perf_counter() - t0

    print(f"     Thomas: {t_thomas:.2f}s  |  HHL: {t_hhl:.1f}s")

    # Metric Computation Phase
    thomas_br = compute_errors(problem, thomas_sr, u_thomas=None)
    hhl_br    = compute_errors(problem, hhl_sr,    u_thomas=thomas_sr.u)

    if VERBOSE_HHL and hhl_br.u_exact is not None:
        print_hhl_summary(hhl_br)

    return thomas_br, hhl_br


# ── Benchmark Sweeps ──────────────────────────────────────────────────────────

def sweep_a() -> list[BenchmarkResult]:
    """
    Executes Sweep A: Homogeneous Boundary Conditions (Reference Section IV A).

    Evaluates all source functions (fS, fL, fH) across varying mesh resolutions 
    (N = 8, 16) at a constant algorithmic precision (ε = 0.01).
    """
    print("\n" + "=" * 70)
    print("SWEEP A — Homogeneous BCs, ε = 0.01")
    print("=" * 70)

    configs = [
        SimConfig1D(N=N, epsilon=0.01, source_fn=fn)
        for fn in ("fS", "fL", "fH")
        for N  in (8, 16)
    ]

    all_results: list[BenchmarkResult] = []
    for cfg in configs:
        thomas_br, hhl_br = run_pair(cfg)
        all_results.extend([thomas_br, hhl_br])

    return all_results


def sweep_b() -> list[BenchmarkResult]:
    """
    Executes Sweep B: Precision Sensitivity Analysis (Reference Section IV D).

    Investigates the correlation between the Trotterisation precision parameter 
    (ε) and resultant error magnitudes for homogeneous systems.
    """
    print("\n" + "=" * 70)
    print("SWEEP B — ε sensitivity, homogeneous BCs, N = 16")
    print("=" * 70)

    configs = [
        SimConfig1D(N=16, epsilon=eps, source_fn=fn)
        for fn  in ("fL", "fH")
        for eps in (0.1, 0.01, 0.001)
    ]

    all_results: list[BenchmarkResult] = []
    for cfg in configs:
        thomas_br, hhl_br = run_pair(cfg)
        all_results.extend([thomas_br, hhl_br])

    return all_results


def sweep_c() -> list[BenchmarkResult]:
    """
    Executes Sweep C: Non-Homogeneous Boundary Conditions (Reference Section IV B).

    Evaluates the algorithm's stability when subjected to asymmetrical Dirichlet 
    boundary constraints, specifically targeting configurations where |u(0)| ≠ |u(1)|.
    """
    print("\n" + "=" * 70)
    print("SWEEP C — Non-homogeneous BCs")
    print("=" * 70)

    configs = [
        SimConfig1D(N=16, epsilon=0.005,  source_fn="fH", alpha=0.0,  beta=0.5),
        SimConfig1D(N=16, epsilon=0.005,  source_fn="fH", alpha=-0.5, beta=0.5),
        SimConfig1D(N=32, epsilon=0.0038, source_fn="fH", alpha=0.0,  beta=0.5),
        SimConfig1D(N=32, epsilon=0.001,  source_fn="fH", alpha=-0.5, beta=0.5),
    ]

    all_results: list[BenchmarkResult] = []
    for cfg in configs:
        thomas_br, hhl_br = run_pair(cfg)
        all_results.extend([thomas_br, hhl_br])

    return all_results


def sweep_d() -> None:
    """
    Executes Sweep D: Condition Number Scaling Verification (Reference Appendix B.1).

    Validates the theoretical O(N²) scaling profile of the TST matrix condition 
    number. This routine bypasses full quantum execution for computational efficiency.
    """
    print("\n" + "=" * 70)
    print("SWEEP D — Condition number scaling κ(A) ~ (4/π²)(N+1)²")
    print("=" * 70)
    print(f"\n  {'N':>4}  {'κ(A) computed':>16}  {'κ(A) theoretical':>18}  {'ratio':>8}")
    print("  " + "-" * 52)

    for N in (4, 8, 16, 32):
        cfg  = SimConfig1D(N=N, epsilon=0.01, source_fn="fS")
        prob = PoissonProblem1D(cfg)
        kappa_computed    = prob.kappa
        kappa_theoretical = (4.0 / np.pi**2) * (N + 1)**2
        ratio = kappa_computed / kappa_theoretical
        
        print(
            f"  {N:>4}  {kappa_computed:>16.4f}  "
            f"{kappa_theoretical:>18.4f}  {ratio:>8.4f}"
        )


# ── Data Exportation ──────────────────────────────────────────────────────────

def save_to_csv(results: list[BenchmarkResult], filename: str) -> None:
    """
    Serialises the aggregated scalar metrics to a Comma-Separated Values format.

    Note: High-dimensional spatial solution arrays are deliberately excluded 
    from this tabular format to preserve readability.
    """
    RESULTS_DIR.mkdir(exist_ok=True)
    filepath = RESULTS_DIR / filename

    fieldnames = [
        "solver", "N", "source_fn", "alpha", "beta", "epsilon",
        "max_rel_error_pct", "avg_rel_error_pct",
        "max_abs_error", "avg_abs_error",
        "euclidean_residual", "prop_const", "kappa",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for r in results:
            cfg = r.config
            prob = PoissonProblem1D(cfg)
            writer.writerow({
                "solver":             r.solver,
                "N":                  cfg.N,
                "source_fn":          cfg.source_fn,
                "alpha":              cfg.alpha,
                "beta":               cfg.beta,
                "epsilon":            cfg.epsilon,
                "max_rel_error_pct":  r.max_rel_error,
                "avg_rel_error_pct":  r.avg_rel_error,
                "max_abs_error":      r.max_abs_error,
                "avg_abs_error":      r.avg_abs_error,
                "euclidean_residual": r.euclidean_residual,
                "prop_const":         r.prop_const,
                "kappa":              prob.kappa,
            })

    print(f"\n  Results exported to {filepath}")


# ── Primary Execution Orchestrator ────────────────────────────────────────────

def main() -> None:
    """
    Primary driver function for the benchmark suite.

    Sequentially invokes the predefined sweeps (D → A → B → C) to incrementally 
    evaluate spatial resolution, precision sensitivity, and boundary conditions. 
    Metrics are systematically printed, exported, and visualised.
    """
    print("\n" + "=" * 70)
    print("1D Poisson HHL Benchmark")
    print("Replicating Ghafourpour & Laizet, Phys. Rev. Applied 24, 024032 (2025)")
    print("=" * 70)

    sweep_d()

    results_a = sweep_a()
    print("\n--- Sweep A Summary ---")
    print_result_table(results_a)
    if OUTPUT_CSV:
        save_to_csv(results_a, "sweep_a_homogeneous.csv")
    plot_sweep_pairs(results_a, save_fig=OUTPUT_CSV)

    results_b = sweep_b()
    print("\n--- Sweep B Summary ---")
    print_result_table(results_b)
    if OUTPUT_CSV:
        save_to_csv(results_b, "sweep_b_epsilon_sensitivity.csv")

    results_c = sweep_c()
    print("\n--- Sweep C Summary ---")
    print_result_table(results_c)
    if OUTPUT_CSV:
        save_to_csv(results_c, "sweep_c_nonhomogeneous.csv")
    plot_sweep_pairs(results_c, save_fig=OUTPUT_CSV)

    print("\n" + "=" * 70)
    print("Benchmark evaluations concluded.")
    print("=" * 70)