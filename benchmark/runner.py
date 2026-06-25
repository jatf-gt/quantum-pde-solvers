"""
Orchestrates the sequential execution sweeps for both the 1D and 2D Poisson benchmarks.

This module acts as the primary driver for replicating the numerical experiments 
detailed in the primary reference literature.

1D Sweeps (Sections IV A–D)
---------------------------
sweep_a : Homogeneous BCs, all source functions, N ∈ {8, 16}, ε = 0.01.
sweep_b : Trotterisation precision (ε) sensitivity, homogeneous BCs, N = 16.
sweep_c : Non-homogeneous BCs, fH, N ∈ {16, 32}.
sweep_d : Condition number scaling verification (bypasses quantum solver).

2D Sweeps (Sections IV E–F)
---------------------------
sweep_e : Homogeneous BCs, all source functions, N ∈ {8, 16}, baseline ε = 0.01.
sweep_f : Non-homogeneous BCs, N = 8, evaluates specific asymmetric convergence criteria.
sweep_g : Condition number scaling verification for the 2D line-Jacobi row matrix.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from core.config import SimConfig1D, SimConfig2D
from problems.poisson_1d import PoissonProblem1D
from problems.poisson_2d import PoissonProblem2D
from solvers.classical.thomas import thomas_solve
from solvers.classical.thomas_2d import thomas_solve_2d
from solvers.classical.numpy_ref import numpy_solve
from solvers.quantum.hhl_1d import hhl_solve
from solvers.quantum.hhl_2d import hhl_solve_2d
from solvers.quantum.vqls_1d import VQLSConfig
from benchmark.metrics import (
    BenchmarkResult,
    BenchmarkResult2D,
    compute_errors,
    compute_errors_2d,
)
from benchmark.reporting import (
    print_result_table,
    print_result_table_2d,
    print_convergence_summary,
)
from benchmark.plotting import (
    plot_solution_comparison_1d,
    plot_solution_contours_2d,
    plot_convergence_history,
)

# ── Global Execution Directives ───────────────────────────────────────────────

OUTPUT_CSV  = True
SAVE_FIGS   = True
RESULTS_DIR = Path("results")


# ── 1D Execution Orchestration ────────────────────────────────────────────────

def run_pair_1d(cfg: SimConfig1D) -> tuple[BenchmarkResult, BenchmarkResult]:
    """
    Instantiates the 1D problem, executes both classical Thomas and quantum HHL 
    resolutions, and aggregates the resulting metrics.

    The classical baseline is explicitly executed prior to the quantum solver 
    to ensure the Thomas solution vector is available for absolute error 
    computations (required for non-homogeneous configurations lacking analytical derivations).
    """
    problem = PoissonProblem1D(cfg)
    print(f"\n  → {problem.summary()}")

    t0        = time.perf_counter()
    thomas_sr = thomas_solve(problem)
    t_thomas  = time.perf_counter() - t0

    t0     = time.perf_counter()
    hhl_sr = hhl_solve(problem)
    t_hhl  = time.perf_counter() - t0

    print(f"     Thomas: {t_thomas:.3f}s  |  HHL: {t_hhl:.1f}s")

    thomas_br = compute_errors(problem, thomas_sr, u_thomas=None)
    hhl_br    = compute_errors(problem, hhl_sr,    u_thomas=thomas_sr.u)

    return thomas_br, hhl_br


# ── 2D Execution Orchestration ────────────────────────────────────────────────

def run_pair_2d(
    cfg: SimConfig2D,
) -> tuple[BenchmarkResult2D, BenchmarkResult2D]:
    """
    Instantiates the 2D problem and evaluates both Thomas-2D and HHL-2D 
    line-Jacobi iterative solvers.

    The high-fidelity classical reference solution (direct NumPy resolution 
    on the full N²×N² system) is computed precisely once and shared between 
    both error evaluation pipelines. This replicates the refined-mesh Thomas 
    solution utilised as ground truth in Section IV E of the primary literature.

    While the reference computation is computationally intensive for N=16 
    (a 256×256 coupled system), it remains highly efficient relative to the 
    subsequent HHL line-Jacobi iterative cycle.

    The fine solve uses the tridiagonal residual computation so no large matrix 
    is ever allocated.
    """
    problem = PoissonProblem2D(cfg)
    print(f"\n  → {problem.summary()}")

    # Classical direct reference — computed once, shared by both solver evaluations.
    t0      = time.perf_counter()
    u_ref   = problem.classical_reference_solve()   # fine Thomas, no full matrix
    t_ref   = time.perf_counter() - t0
    print(f"     Reference solve: {t_ref:.2f}s")

    # Classical Thomas line-Jacobi execution.
    t0        = time.perf_counter()
    thomas_sr = thomas_solve_2d(problem)
    t_thomas  = time.perf_counter() - t0
    print(
        f"     Thomas-2D: {t_thomas:.2f}s, "
        f"{thomas_sr.iterations} iters, "
        f"converged={thomas_sr.converged}"
    )

    # Quantum HHL line-Jacobi execution.
    t0     = time.perf_counter()
    hhl_sr = hhl_solve_2d(problem)
    t_hhl  = time.perf_counter() - t0
    print(
        f"     HHL-2D:    {t_hhl:.1f}s, "
        f"{hhl_sr.iterations} iters, "
        f"converged={hhl_sr.converged}"
    )

    thomas_br = compute_errors_2d(problem, thomas_sr, u_reference=u_ref)
    hhl_br    = compute_errors_2d(problem, hhl_sr,    u_reference=u_ref)

    return thomas_br, hhl_br


# ── 1D Benchmark Sweeps ───────────────────────────────────────────────────────

def sweep_a() -> list[BenchmarkResult]:
    """
    Executes Sweep A: Homogeneous boundary conditions (Reference Section IV A).
    Evaluates all source functions across varied spatial resolutions at ε = 0.01.
    """
    print("\n" + "=" * 70)
    print("SWEEP A — 1D Homogeneous BCs, ε=0.01")
    print("=" * 70)

    configs = [
        SimConfig1D(N=N, epsilon=0.01, source_fn=fn)
        for fn in ("fS", "fL", "fH")
        for N  in (8, 16)
    ]
    return _run_1d_sweep(configs)


def sweep_b() -> list[BenchmarkResult]:
    """
    Executes Sweep B: Precision sensitivity analysis (Reference Section IV D).
    Examines the impact of Trotterisation variance on system error vectors.
    """
    print("\n" + "=" * 70)
    print("SWEEP B — 1D ε sensitivity, N=16")
    print("=" * 70)

    configs = [
        SimConfig1D(N=16, epsilon=eps, source_fn=fn)
        for fn  in ("fL", "fH")
        for eps in (0.1, 0.01, 0.001)
    ]
    return _run_1d_sweep(configs)


def sweep_c() -> list[BenchmarkResult]:
    """
    Executes Sweep C: Non-homogeneous boundary conditions (Reference Section IV B).
    Assesses algorithmic stability under asymmetric Dirichlet constraints.
    """
    print("\n" + "=" * 70)
    print("SWEEP C — 1D Non-homogeneous BCs")
    print("=" * 70)

    configs = [
        SimConfig1D(N=16, epsilon=0.005,  source_fn="fH", alpha=0.0,  beta=0.5),
        SimConfig1D(N=16, epsilon=0.005,  source_fn="fH", alpha=-0.5, beta=0.5),
        SimConfig1D(N=32, epsilon=0.0038, source_fn="fH", alpha=0.0,  beta=0.5),
        SimConfig1D(N=32, epsilon=0.001,  source_fn="fH", alpha=-0.5, beta=0.5),
    ]
    return _run_1d_sweep(configs)


def sweep_d() -> None:
    """
    Executes Sweep D: Condition number scaling verification (Reference Appendix B.1).
    Validates theoretical κ(A) ~ O(N²) scaling for the 1D TST matrix.
    """
    print("\n" + "=" * 70)
    print("SWEEP D — 1D Condition number scaling")
    print("=" * 70)
    print(f"\n  {'N':>4}  {'κ computed':>14}  {'κ theoretical':>16}  {'ratio':>8}")
    print("  " + "-" * 48)
    for N in (4, 8, 16, 32):
        cfg   = SimConfig1D(N=N, epsilon=0.01, source_fn="fS")
        prob  = PoissonProblem1D(cfg)
        kappa_c = prob.kappa
        kappa_t = (4.0 / np.pi**2) * (N + 1)**2
        print(
            f"  {N:>4}  {kappa_c:>14.4f}  "
            f"{kappa_t:>16.4f}  {kappa_c/kappa_t:>8.4f}"
        )


# ── 2D Benchmark Sweeps ───────────────────────────────────────────────────────

def sweep_e() -> list[BenchmarkResult2D]:
    """
    Executes Sweep E: 2D Homogeneous boundary conditions (Reference Section IV E).

    Evaluates baseline parameters (ε = 0.01) across all source functions. 
    Maintains strict methodological parity with the literature by simulating 
    a relaxed precision (ε = 0.5) specifically for fS at N=16 to ensure 
    convergence consistency (Reference Figure 12).
    """
    print("\n" + "=" * 70)
    print("SWEEP E — 2D Homogeneous BCs")
    print("=" * 70)

    configs = [
        # N=8, all source functions, ε=0.01 (Reference Figure 10)
        SimConfig2D(N=8, epsilon=0.01, source_fn="fS"),
        SimConfig2D(N=8, epsilon=0.01, source_fn="fL"),
        SimConfig2D(N=8, epsilon=0.01, source_fn="fH"),
        # N=16, fL, ε=0.01 (Converges per literature specification)
        SimConfig2D(N=16, epsilon=0.01, source_fn="fL", tol=1e-4),              # TODO: Run tol=1e-8 version in remote machine
        # N=16, fS, ε=0.5 (Reference Figure 12 — relaxed precision required)
        SimConfig2D(N=16, epsilon=0.5, source_fn="fS", tol=1e-4),
        # N=16, fH, ε=0.03 (Reference Figure 13)
        SimConfig2D(N=16, epsilon=0.03, source_fn="fH", tol=1e-4),
    ]
    return _run_2d_sweep(configs)


def sweep_f() -> list[BenchmarkResult2D]:
    """
    Executes Sweep F: 2D Non-homogeneous boundary conditions (Reference Section IV F).

    Investigates algorithmic stability and iterative divergence under specific 
    asymmetric boundary constraints at N=8. The second configuration explicitly 
    imposes an iteration ceiling (max_iter=200) to replicate the non-converging 
    residuals visualised in Figure 15.
    """
    print("\n" + "=" * 70)
    print("SWEEP F — 2D Non-homogeneous BCs, N=8")
    print("=" * 70)

    configs = [
        # Reference Figure 14: fH, complex asymmetric boundaries
        SimConfig2D(
            N=8, epsilon=0.01, source_fn="fH",
            bc_x0=0.5, bc_x1=0.0,
            bc_y0=0.0, bc_y1=0.0,
        ),
        # Reference Figure 15: fS, asymmetric boundaries (divergence anticipated)
        SimConfig2D(
            N=8, epsilon=0.01, source_fn="fS",
            bc_x0=0.0,  bc_x1=0.08,
            bc_y0=0.08, bc_y1=0.0,
            max_iter=200,
        ),
        # Reference Figure 16: fS, symmetric boundaries (convergence anticipated)
        SimConfig2D(
            N=8, epsilon=0.01, source_fn="fS",
            bc_x0=0.0,  bc_x1=0.08,
            bc_y0=0.0,  bc_y1=0.08,
        ),
    ]
    return _run_2d_sweep(configs)


def sweep_g() -> None:
    """
    Executes Sweep G: Condition number scaling verification for the 2D operator.

    Validates the analytical derivation (Appendix B.1) demonstrating that the 
    condition number of the 2D line-Jacobi sub-matrix (a=-4, b=1) approaches 
    a limit of 3 as N → ∞, contrasting the O(N²) divergence of the 1D system.
    """
    print("\n" + "=" * 70)
    print("SWEEP G — 2D row matrix condition number (κ → 3 as N → ∞)")
    print("=" * 70)
    print(
        f"\n  {'N':>4}  {'κ_1D':>10}  {'κ_2D_row':>12}  "
        f"{'κ_2D theoretical':>18}"
    )
    print("  " + "-" * 50)
    for N in (4, 8, 16, 32):
        from problems.poisson_2d import condition_number_2d
        cfg_1d  = SimConfig1D(N=N, epsilon=0.01, source_fn="fS")
        prob_1d = PoissonProblem1D(cfg_1d)
        kappa_1d  = prob_1d.kappa
        kappa_2d  = condition_number_2d(N)
        
        # Theoretical limit evaluation: (6 - (π/(N+1))²) / (2 + (π/(N+1))²)
        theta     = np.pi / (N + 1)
        kappa_2d_t = (6 - theta**2) / (2 + theta**2)
        print(
            f"  {N:>4}  {kappa_1d:>10.4f}  {kappa_2d:>12.6f}  "
            f"{kappa_2d_t:>18.6f}"
        )


# ── Data Exportation ──────────────────────────────────────────────────────────

def save_to_csv_1d(results: list[BenchmarkResult], filename: str) -> None:
    """Serialises aggregated 1D benchmark scalar metrics to a CSV format."""
    RESULTS_DIR.mkdir(exist_ok=True)
    filepath = RESULTS_DIR / filename
    fieldnames = [
        "solver", "N", "source_fn", "alpha", "beta", "epsilon",
        "max_rel_error_pct", "avg_rel_error_pct",
        "max_abs_error", "avg_abs_error",
        "euclidean_residual", "prop_const",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            cfg = r.config
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
            })
    print(f"\n  Saved to {filepath}")


def save_to_csv_2d(results: list[BenchmarkResult2D], filename: str) -> None:
    """Serialises aggregated 2D benchmark scalar metrics to a CSV format."""
    RESULTS_DIR.mkdir(exist_ok=True)
    filepath = RESULTS_DIR / filename
    fieldnames = [
        "solver", "N", "source_fn", "epsilon",
        "bc_x0", "bc_x1", "bc_y0", "bc_y1",
        "max_rel_error_pct", "avg_rel_error_pct",
        "max_abs_error", "avg_abs_error",
        "iterations", "converged", "euclidean_residual",
    ]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            cfg = r.config
            writer.writerow({
                "solver":             r.solver,
                "N":                  cfg.N,
                "source_fn":          cfg.source_fn,
                "epsilon":            cfg.epsilon,
                "bc_x0":              cfg.bc_x0,
                "bc_x1":              cfg.bc_x1,
                "bc_y0":              cfg.bc_y0,
                "bc_y1":              cfg.bc_y1,
                "max_rel_error_pct":  r.max_rel_error,
                "avg_rel_error_pct":  r.avg_rel_error,
                "max_abs_error":      r.max_abs_error,
                "avg_abs_error":      r.avg_abs_error,
                "iterations":         r.iterations,
                "converged":          r.converged,
                "euclidean_residual": r.euclidean_residual,
            })
    print(f"\n  Saved to {filepath}")


# ── Private Utility Methods ───────────────────────────────────────────────────

def _run_1d_sweep(configs: list[SimConfig1D]) -> list[BenchmarkResult]:
    """Evaluates sequential configurations via 1D Thomas and HHL algorithms."""
    all_results: list[BenchmarkResult] = []
    for cfg in configs:
        thomas_br, hhl_br = run_pair_1d(cfg)
        all_results.extend([thomas_br, hhl_br])
    return all_results


def _run_2d_sweep(configs: list[SimConfig2D]) -> list[BenchmarkResult2D]:
    """Evaluates sequential configurations via 2D Thomas and HHL line-Jacobi loops."""
    all_results: list[BenchmarkResult2D] = []
    for cfg in configs:
        thomas_br, hhl_br = run_pair_2d(cfg)
        all_results.extend([thomas_br, hhl_br])
    return all_results


def _plot_1d_pairs(
    results: list[BenchmarkResult],
    save_fig: bool = False,
) -> None:
    """Iterates through alternating (Thomas, HHL) pairs to trigger 1D rendering."""
    for i in range(0, len(results), 2):
        plot_solution_comparison_1d(results[i], results[i + 1], save_fig)


def _plot_2d_pairs(
    results:            list[BenchmarkResult2D],
    use_relative_error: bool = True,
    save_fig:           bool = False,
) -> None:
    """
    Iterates through alternating (Thomas, HHL) pairs to trigger 2D contour 
    mapping and iterative convergence rendering.
    """
    for i in range(0, len(results), 2):
        thomas_br = results[i]
        hhl_br    = results[i + 1]
        
        plot_solution_contours_2d(
            thomas_br, hhl_br,
            use_relative_error=use_relative_error,
            save_fig=save_fig,
        )
        
        plot_convergence_history(
            [thomas_br, hhl_br],
            save_fig=save_fig,
        )


# ── 1D Execution Orchestration ────────────────────────────────────────────────

def run_pair_1d(
    cfg:         SimConfig1D,
    run_vqls:    bool       = True,
    vqls_config: "VQLSConfig" = None,
) -> tuple:
    """
    Instantiates the 1D problem and sequentially executes classical Thomas, 
    quantum HHL, and optionally VQLS resolutions, aggregating the resultant metrics.

    Returns a dynamically sized tuple containing the benchmark data structures 
    contingent upon the `run_vqls` execution flag.

    Parameters
    ----------
    cfg : SimConfig1D
        Configuration parameters governing the 1D simulation instance.
    run_vqls : bool, default=True
        Boolean flag dictating the execution of the Variational Quantum Linear Solver.
    vqls_config : VQLSConfig, optional
        Hyperparameter structure governing the variational optimisation. 
        Defaults to DEFAULT_VQLS_CONFIG if omitted.
    """
    from solvers.quantum.vqls_1d import vqls_solve, VQLSConfig, DEFAULT_VQLS_CONFIG

    problem = PoissonProblem1D(cfg)
    print(f"\n  → {problem.summary()}")

    # Classical Reference Execution
    t0        = time.perf_counter()
    thomas_sr = thomas_solve(problem)
    t_thomas  = time.perf_counter() - t0

    # Quantum HHL Execution
    t0     = time.perf_counter()
    hhl_sr = hhl_solve(problem)
    t_hhl  = time.perf_counter() - t0

    print(
        f"     Thomas: {t_thomas:.3f}s  |  "
        f"HHL: {t_hhl:.1f}s",
        end="",
    )

    thomas_br = compute_errors(problem, thomas_sr, u_thomas=None)
    hhl_br    = compute_errors(problem, hhl_sr,    u_thomas=thomas_sr.u)

    if not run_vqls:
        print()
        return thomas_br, hhl_br

    # Variational Quantum Linear Solver (VQLS) Execution
    vc = vqls_config if vqls_config is not None else DEFAULT_VQLS_CONFIG
    t0      = time.perf_counter()
    vqls_sr = vqls_solve(problem, config=vc)
    t_vqls  = time.perf_counter() - t0

    print(
        f"  |  VQLS: {t_vqls:.1f}s "
        f"(cost={vqls_sr.final_cost:.4f}, "
        f"evals={vqls_sr.n_circuit_evals})"
    )

    vqls_br = compute_errors(problem, vqls_sr, u_thomas=thomas_sr.u)

    return thomas_br, hhl_br, vqls_br