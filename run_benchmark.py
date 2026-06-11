"""
run_benchmark.py
----------------
Main entry point for the 1D Poisson HHL benchmark.

Reproduces the simulation configurations from Sections IV A and IV B of
Ghafourpour & Laizet (2025), comparing the HHL algorithm against the
Thomas algorithm and the analytical solution.

Usage
-----
    python run_benchmark.py

The script runs four sweeps in order:

  Sweep A — Homogeneous BCs, all three source functions, N ∈ {8, 16},
             ε = 0.01.  Reproduces Figs. 3 and 4 of the paper.

  Sweep B — Homogeneous BCs, fL and fH, N = 16, ε ∈ {0.01, 0.001}.
             Reproduces the ε-sensitivity discussion in Section IV D.

  Sweep C — Non-homogeneous BCs, fH, N ∈ {16, 32}, selected ε values.
             Reproduces Fig. 5 of the paper.

  Sweep D — Condition number study: prints κ(A) for N ∈ {4, 8, 16, 32}
             alongside the theoretical O(N²) scaling.

Results are printed to stdout as formatted tables and optionally saved
as CSV files if OUTPUT_CSV = True.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from config import SimConfig
from problem_setup import PoissonProblem1D
from solvers import thomas_solve, hhl_solve, numpy_solve
from benchmark import (
    BenchmarkResult,
    compute_errors,
    print_result_table,
    print_hhl_summary,
)

# ── Global options ────────────────────────────────────────────────────────────
OUTPUT_CSV   = True          # write results to CSV files in ./results/
VERBOSE_HHL  = True          # print node-by-node HHL breakdown
RESULTS_DIR  = Path("results")


# ── Helper: run one Thomas + HHL pair and return both BenchmarkResults ────────

def run_pair(cfg: SimConfig) -> tuple[BenchmarkResult, BenchmarkResult]:
    """
    Build the problem, solve with both Thomas and HHL, compute all error
    metrics, and return (thomas_result, hhl_result).

    Running Thomas first is deliberate: its solution is passed into
    compute_errors for the HHL result so that absolute errors against
    Thomas are available for non-homogeneous cases.
    """
    problem = PoissonProblem1D(cfg)
    print(f"\n  → {problem.summary()}")

    # Classical reference — should be near machine precision.
    t0 = time.perf_counter()
    thomas_sr = thomas_solve(problem)
    t_thomas  = time.perf_counter() - t0

    # Quantum solver — the slow step.
    t0 = time.perf_counter()
    hhl_sr    = hhl_solve(problem)
    t_hhl     = time.perf_counter() - t0

    print(f"     Thomas: {t_thomas:.2f}s  |  HHL: {t_hhl:.1f}s")

    # Compute error metrics.
    thomas_br = compute_errors(problem, thomas_sr, u_thomas=None)
    hhl_br    = compute_errors(problem, hhl_sr,    u_thomas=thomas_sr.u)

    if VERBOSE_HHL and hhl_br.u_exact is not None:
        print_hhl_summary(hhl_br)

    return thomas_br, hhl_br


# ── Sweep A: homogeneous BCs, ε = 0.01 ───────────────────────────────────────

def sweep_a() -> list[BenchmarkResult]:
    """
    Section IV A of the paper.

    Source functions: fS, fL, fH
    Mesh sizes:       N = 8, 16
    Epsilon:          0.01
    BCs:              u(0) = u(1) = 0
    """
    print("\n" + "=" * 70)
    print("SWEEP A — Homogeneous BCs, ε = 0.01")
    print("=" * 70)

    configs = [
        SimConfig(N=N, epsilon=0.01, source_fn=fn)
        for fn in ("fS", "fL", "fH")
        for N  in (8, 16)
    ]

    all_results: list[BenchmarkResult] = []
    for cfg in configs:
        thomas_br, hhl_br = run_pair(cfg)
        all_results.extend([thomas_br, hhl_br])

    return all_results


# ── Sweep B: ε sensitivity, homogeneous BCs ───────────────────────────────────

def sweep_b() -> list[BenchmarkResult]:
    """
    Section IV D of the paper.

    Investigates how ε affects accuracy for homogeneous BCs.
    The paper shows that reducing ε consistently lowers errors for
    homogeneous problems (Fig. 8a), so we sweep ε ∈ {0.1, 0.01, 0.001}.

    Source functions: fL, fH  (fS behaves similarly to fH)
    Mesh size:        N = 16
    BCs:              u(0) = u(1) = 0
    """
    print("\n" + "=" * 70)
    print("SWEEP B — ε sensitivity, homogeneous BCs, N = 16")
    print("=" * 70)

    configs = [
        SimConfig(N=16, epsilon=eps, source_fn=fn)
        for fn  in ("fL", "fH")
        for eps in (0.1, 0.01, 0.001)
    ]

    all_results: list[BenchmarkResult] = []
    for cfg in configs:
        thomas_br, hhl_br = run_pair(cfg)
        all_results.extend([thomas_br, hhl_br])

    return all_results


# ── Sweep C: non-homogeneous BCs ──────────────────────────────────────────────

def sweep_c() -> list[BenchmarkResult]:
    """
    Section IV B of the paper.

    Tests non-homogeneous Dirichlet BCs.  The paper identifies two
    sub-cases:
      (i)  |u(0)| = |u(1)| — HHL performs comparably to Thomas
      (ii) |u(0)| ≠ |u(1)| — HHL shows increased sensitivity to ε

    We replicate the specific cases shown in Fig. 5 of the paper.
    Note: for non-homogeneous BCs there is no closed-form analytical
    solution in our EXACT_SOLUTIONS dict, so errors are reported
    against the Thomas solution.
    """
    print("\n" + "=" * 70)
    print("SWEEP C — Non-homogeneous BCs")
    print("=" * 70)

    # These configurations match Fig. 5 of the paper exactly.
    # (a) N=16, fH, u(0)=0, u(1)=0.5, ε=0.005
    # (b) N=16, fH, u(0)=-0.5, u(1)=0.5, ε=0.005
    # (c) N=32, fH, u(0)=0, u(1)=0.5, ε=0.0038
    # (d) N=32, fH, u(0)=-0.5, u(1)=0.5, ε=0.001
    configs = [
        SimConfig(N=16, epsilon=0.005,  source_fn="fH", alpha=0.0,  beta=0.5),
        SimConfig(N=16, epsilon=0.005,  source_fn="fH", alpha=-0.5, beta=0.5),
        SimConfig(N=32, epsilon=0.0038, source_fn="fH", alpha=0.0,  beta=0.5),
        SimConfig(N=32, epsilon=0.001,  source_fn="fH", alpha=-0.5, beta=0.5),
    ]

    all_results: list[BenchmarkResult] = []
    for cfg in configs:
        thomas_br, hhl_br = run_pair(cfg)
        all_results.extend([thomas_br, hhl_br])

    return all_results


# ── Sweep D: condition number scaling ─────────────────────────────────────────

def sweep_d() -> None:
    """
    Verify the O(N²) condition number scaling from Appendix B.1 of the paper.

    The theoretical formula is κ(A) ≈ (4/π²)(N+1)².
    This is printed as a table — no HHL run needed.
    """
    print("\n" + "=" * 70)
    print("SWEEP D — Condition number scaling κ(A) ~ (4/π²)(N+1)²")
    print("=" * 70)
    print(f"\n  {'N':>4}  {'κ(A) computed':>16}  {'κ(A) theoretical':>18}  {'ratio':>8}")
    print("  " + "-" * 52)

    for N in (4, 8, 16, 32):
        # We only need the matrix, not a full problem, so use epsilon=0.01
        # as a placeholder (it does not affect A or κ).
        cfg  = SimConfig(N=N, epsilon=0.01, source_fn="fS")
        prob = PoissonProblem1D(cfg)
        kappa_computed    = prob.kappa
        kappa_theoretical = (4.0 / np.pi**2) * (N + 1)**2
        ratio = kappa_computed / kappa_theoretical
        print(
            f"  {N:>4}  {kappa_computed:>16.4f}  "
            f"{kappa_theoretical:>18.4f}  {ratio:>8.4f}"
        )


# ── CSV export ────────────────────────────────────────────────────────────────

def save_to_csv(results: list[BenchmarkResult], filename: str) -> None:
    """
    Write a list of BenchmarkResults to a CSV file in RESULTS_DIR.

    Each row corresponds to one solver/config combination.  The CSV
    includes all scalar metrics; the solution vectors themselves are
    not saved here (use numpy.save separately if needed).
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
            # Rebuild the problem just to get κ — it's fast.
            prob = PoissonProblem1D(cfg)
            writer.writerow({
                "solver":            r.solver,
                "N":                 cfg.N,
                "source_fn":         cfg.source_fn,
                "alpha":             cfg.alpha,
                "beta":              cfg.beta,
                "epsilon":           cfg.epsilon,
                "max_rel_error_pct": r.max_rel_error,
                "avg_rel_error_pct": r.avg_rel_error,
                "max_abs_error":     r.max_abs_error,
                "avg_abs_error":     r.avg_abs_error,
                "euclidean_residual": r.euclidean_residual,
                "prop_const":        r.prop_const,
                "kappa":             prob.kappa,
            })

    print(f"\n  Results saved to {filepath}")


# ── Plotting (optional, requires matplotlib) ──────────────────────────────────

def plot_solution_comparison(
    thomas_br: BenchmarkResult,
    hhl_br:    BenchmarkResult,
    save_fig:  bool = False,
) -> None:
    """
    Reproduce the two-panel layout used in the paper's Figs. 3–5.

    Left panel:  solution curves (Analytical, Thomas, HHL) vs x
    Right panel: HHL absolute relative error (%) vs x

    Parameters
    ----------
    thomas_br : BenchmarkResult for the Thomas solver
    hhl_br    : BenchmarkResult for the HHL solver
    save_fig  : if True, save to results/<source_fn>_N<N>_eps<eps>.png
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot.")
        return

    cfg = hhl_br.config
    x   = hhl_br.x

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # ── Left: solution ────────────────────────────────────────────────────────
    if hhl_br.u_exact is not None:
        ax1.plot(x, hhl_br.u_exact,   "k-",  lw=2,   label="Analytical")
    ax1.plot(x, thomas_br.u_solver,    "g-o", lw=1.5, ms=5, label="Thomas")
    ax1.plot(x, hhl_br.u_solver,       "r--*",lw=1.5, ms=5, label="HHL")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    ax1.set_title(
        f"f = {cfg.source_fn},  N = {cfg.N},  ε = {cfg.epsilon},  "
        f"α = {cfg.alpha},  β = {cfg.beta}"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Right: HHL relative error ─────────────────────────────────────────────
    if hhl_br.rel_error is not None:
        # Replace NaN (near-zero nodes) with 0 for plotting clarity.
        rel_err_plot = np.where(np.isnan(hhl_br.rel_error), 0.0, hhl_br.rel_error)
        ax2_color = "r"
        ax2.plot(x, rel_err_plot, color=ax2_color, lw=1.5)
        ax2.set_ylabel("HHL absolute relative error (%)", color=ax2_color)
        ax2.tick_params(axis="y", labelcolor=ax2_color)
    else:
        # Non-homogeneous case: plot absolute error against Thomas instead.
        ax2.plot(x, hhl_br.abs_error, "r-", lw=1.5)
        ax2.set_ylabel("|u_HHL − u_Thomas|")

    ax2.set_xlabel("x")
    ax2.set_title("HHL error profile")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_fig:
        RESULTS_DIR.mkdir(exist_ok=True)
        fname = (
            RESULTS_DIR /
            f"{cfg.source_fn}_N{cfg.N}_eps{cfg.epsilon:.4f}"
            f"_a{cfg.alpha}_b{cfg.beta}.png"
        )
        plt.savefig(fname, dpi=150)
        print(f"  Figure saved to {fname}")

    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Run all four sweeps, print summary tables, and optionally save CSVs
    and figures.

    The sweeps are ordered from cheapest to most expensive:
      D (no HHL) → A (small N, moderate ε) → B (ε sensitivity) → C (large N)

    For a quick first run, comment out sweeps B and C and just run A and D.
    """
    print("\n" + "=" * 70)
    print("1D Poisson HHL Benchmark")
    print("Replicating Ghafourpour & Laizet, Phys. Rev. Applied 24, 024032 (2025)")
    print("=" * 70)

    # ── Sweep D: condition numbers (fast, no HHL) ─────────────────────────────
    sweep_d()

    # ── Sweep A: homogeneous BCs ──────────────────────────────────────────────
    results_a = sweep_a()
    print("\n--- Sweep A Summary ---")
    print_result_table(results_a)
    if OUTPUT_CSV:
        save_to_csv(results_a, "sweep_a_homogeneous.csv")

    # Optionally plot each Thomas/HHL pair from Sweep A.
    # We iterate in pairs (Thomas, HHL) since run_pair returns them together.
    # Here we re-pair them from the flat list for plotting.
    _plot_sweep_pairs(results_a, save_fig=OUTPUT_CSV)

    # ── Sweep B: ε sensitivity ────────────────────────────────────────────────
    results_b = sweep_b()
    print("\n--- Sweep B Summary ---")
    print_result_table(results_b)
    if OUTPUT_CSV:
        save_to_csv(results_b, "sweep_b_epsilon_sensitivity.csv")

    # ── Sweep C: non-homogeneous BCs ──────────────────────────────────────────
    results_c = sweep_c()
    print("\n--- Sweep C Summary ---")
    print_result_table(results_c)
    if OUTPUT_CSV:
        save_to_csv(results_c, "sweep_c_nonhomogeneous.csv")

    _plot_sweep_pairs(results_c, save_fig=OUTPUT_CSV)

    print("\n" + "=" * 70)
    print("Benchmark complete.")
    print("=" * 70)


def _plot_sweep_pairs(
    results: list[BenchmarkResult],
    save_fig: bool = False,
) -> None:
    """
    Iterate through a flat list of BenchmarkResults in (Thomas, HHL) pairs
    and call plot_solution_comparison for each pair.

    The list is assumed to be ordered as [thomas_0, hhl_0, thomas_1, hhl_1, …]
    which is exactly what sweep_a, sweep_b, and sweep_c produce.
    """
    for i in range(0, len(results), 2):
        thomas_br = results[i]
        hhl_br    = results[i + 1]
        # Only plot if we have an analytical solution or Thomas reference.
        if hhl_br.u_exact is not None or hhl_br.u_thomas is not None:
            plot_solution_comparison(thomas_br, hhl_br, save_fig=save_fig)


if __name__ == "__main__":
    main()