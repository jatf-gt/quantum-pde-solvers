"""
Executes the HHL and VQLS quantum solvers against the 1D Hall Effect 
Thruster (HET) plasma Poisson benchmark.

Sweep Structure
---------------
Sweep H1 : Gaussian profile, homogeneous boundary conditions (V_d=0), 
           N ∈ {4, 8, 16}. Verifies fundamental solver fidelity against 
           refined classical baselines.
Sweep H2 : Physical boundary conditions (V_d=300V), Gaussian profile, 
           N ∈ {4, 8}. Evaluates algorithmic stability under non-homogeneous constraints.
Sweep H3 : Comparative accuracy evaluation across all three charge density 
           profiles at N=8 under physical boundary conditions.
Sweep H4 : Pure diagnostic analysis of condition number and scaling parameter 
           (α) behaviour, entirely bypassing quantum solver execution.
"""
from __future__ import annotations

import time
import csv
from pathlib import Path

import numpy as np

from core.het_config import HETConfig
from problems.het_plasma_1d import HETPoissonProblem1D
from solvers.classical.thomas import thomas_solve_system
from solvers.quantum.hhl_1d import hhl_solve_system
from solvers.quantum.vqls_1d import vqls_solve_system, VQLSConfig
from benchmark.reporting import print_result_table

RESULTS_DIR = Path("results/het")


# ── Core Execution Subroutine ─────────────────────────────────────────────────

def run_het_trio(
    cfg:         HETConfig,
    vqls_config: VQLSConfig = None,
    verbose:     bool       = False,
) -> dict:
    """
    Evaluates a singular HET configuration sequentially utilising the classical 
    Thomas, quantum HHL, and Variational Quantum Linear Solver (VQLS) algorithms.

    Returns a structured dictionary comprising the aggregated statistical 
    metrics and spatial solution fields for subsequent serialisation and plotting.
    """
    from solvers.quantum.vqls_1d import DEFAULT_VQLS_CONFIG
    vc = vqls_config or DEFAULT_VQLS_CONFIG

    problem = HETPoissonProblem1D(cfg)
    if verbose:
        print(f"\n  → {problem.summary()}")

    A = problem.A
    b = problem.b

    # Classical Reference Execution (O(N) temporal complexity)
    t0      = time.perf_counter()
    u_thomas = thomas_solve_system(A, b)
    t_thomas = time.perf_counter() - t0

    # Quantum HHL Execution
    t0 = time.perf_counter()
    try:
        u_hhl, _, c_hhl = hhl_solve_system(A, b, cfg.epsilon)
        hhl_ok = True
    except Exception as e:
        u_hhl  = np.zeros_like(b)
        c_hhl  = 0.0
        hhl_ok = False
        if verbose:
            print(f"    HHL execution failed: {e}")
    t_hhl = time.perf_counter() - t0

    # Variational Quantum Linear Solver (VQLS) Execution
    t0 = time.perf_counter()
    try:
        vqls_result = vqls_solve_system(A, b, vc)
        u_vqls      = vqls_result.u
        vqls_cost   = vqls_result.final_cost
        vqls_evals  = vqls_result.n_circuit_evals
        vqls_ok     = vqls_result.optimiser_success
    except Exception as e:
        u_vqls     = np.zeros_like(b)
        vqls_cost  = float("nan")
        vqls_evals = 0
        vqls_ok    = False
        if verbose:
            print(f"    VQLS execution failed: {e}")
    t_vqls = time.perf_counter() - t0

    # Analytical Reference (Contingent upon specific profile viability)
    u_exact = problem.analytical_solution()

    # Statistical Deviation Computation
    def _errors(u_solver, u_ref):
        if u_ref is None:
            return None, None
        abs_err = np.abs(u_solver - u_ref)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_err = np.where(
                np.abs(u_ref) > 1e-10,
                abs_err / np.abs(u_ref) * 100.0,
                np.nan,
            )
        valid = rel_err[~np.isnan(rel_err)]
        max_rel = float(np.max(valid))  if valid.size > 0 else None
        return float(np.max(abs_err)), max_rel

    ref = u_exact if u_exact is not None else u_thomas

    thomas_max_abs, thomas_max_rel = _errors(u_thomas, u_exact)
    hhl_max_abs,    hhl_max_rel    = _errors(u_hhl,    ref)
    vqls_max_abs,   vqls_max_rel   = _errors(u_vqls,   ref)

    if verbose:
        print(
            f"     Thomas: {t_thomas*1e3:.1f}ms  |  "
            f"HHL: {t_hhl:.1f}s  |  "
            f"VQLS: {t_vqls:.1f}s (cost={vqls_cost:.4f})"
        )
        print(
            f"     Max|HHL-ref|={hhl_max_abs:.3e}  "
            f"Max|VQLS-ref|={vqls_max_abs:.3e}  "
            f"Max|VQLS-Thomas|="
            f"{np.max(np.abs(u_vqls - u_thomas)):.3e}"
        )

    return {
        "N":              cfg.N,
        "profile":        cfg.rho_profile,
        "alpha":          cfg.alpha,
        "alpha_bc":       cfg.alpha_bc,
        "kappa":          problem.kappa,
        "epsilon":        cfg.epsilon,
        "thomas_max_abs": thomas_max_abs,
        "thomas_max_rel": thomas_max_rel,
        "hhl_ok":         hhl_ok,
        "hhl_max_abs":    hhl_max_abs,
        "hhl_max_rel":    hhl_max_rel,
        "hhl_time":       t_hhl,
        "vqls_ok":        vqls_ok,
        "vqls_max_abs":   vqls_max_abs,
        "vqls_max_rel":   vqls_max_rel,
        "vqls_cost":      vqls_cost,
        "vqls_evals":     vqls_evals,
        "vqls_time":      t_vqls,
        "u_thomas":       u_thomas,
        "u_hhl":          u_hhl,
        "u_vqls":         u_vqls,
        "u_exact":        u_exact,
        "x":              problem.x,
    }


# ── Benchmark Sweeps ──────────────────────────────────────────────────────────

def sweep_h1(verbose: bool = True) -> list[dict]:
    """
    Executes Sweep H1: Gaussian profile under homogeneous boundaries (V_d=0).

    Evaluates fundamental algorithmic fidelity independent of non-homogeneous 
    boundary complexities. Spatial resolution is constrained to N ∈ {4, 8} 
    to preserve computational tractability.
    """
    print("\n" + "=" * 70)
    print("SWEEP H1 — HET Gaussian profile, homogeneous BCs")
    print("=" * 70)

    vc = VQLSConfig(n_layers=5, optimiser="COBYLA",
                    max_iter=500, tol=1e-5, verbose=verbose)

    results = []
    for N in (4, 8):
        cfg = HETConfig(
            N=N, epsilon=0.01,
            rho_profile="gaussian",
            V_discharge=0.0,   # Homogeneous constraint
        )
        r = run_het_trio(cfg, vqls_config=vc, verbose=verbose)
        results.append(r)

    return results


def sweep_h2(verbose: bool = True) -> list[dict]:
    """
    Executes Sweep H2: Gaussian profile under physical boundaries (V_d=300V).

    Constitutes the primary stability assessment for the HHL algorithm under 
    physical, non-homogeneous constraints, directly addressing supervisory 
    diagnostic directives.
    """
    print("\n" + "=" * 70)
    print("SWEEP H2 — HET Physical BCs (V_d=300V), Gaussian profile")
    print("=" * 70)

    vc = VQLSConfig(n_layers=5, optimiser="COBYLA",
                    max_iter=500, tol=1e-5, verbose=verbose)

    results = []
    for N in (4, 8):
        cfg = HETConfig(
            N=N, epsilon=0.01,
            rho_profile="gaussian",
            V_discharge=300.0,
        )
        r = run_het_trio(cfg, vqls_config=vc, verbose=verbose)
        results.append(r)

    return results


def sweep_h3(verbose: bool = True) -> list[dict]:
    """
    Executes Sweep H3: Physical boundaries across all charge density profiles.

    Facilitates a comprehensive comparative accuracy evaluation across disparate 
    source term topologies at a constant resolution (N=8).
    """
    print("\n" + "=" * 70)
    print("SWEEP H3 — HET All profiles, N=8, physical BCs")
    print("=" * 70)

    vc = VQLSConfig(n_layers=5, optimiser="COBYLA",
                    max_iter=500, tol=1e-5, verbose=verbose)

    results = []
    for profile in ("gaussian", "linear", "step"):
        cfg = HETConfig(
            N=8, epsilon=0.01,
            rho_profile=profile,
            V_discharge=300.0,
        )
        r = run_het_trio(cfg, vqls_config=vc, verbose=verbose)
        results.append(r)

    return results


def sweep_h4() -> None:
    """
    Executes Sweep H4: Pure condition number and α scaling analytics.

    Characterises the asymptotic trajectory of κ(A) and α relative to N and 
    associated physical parameters. This diagnostic data directly informs 
    the requisite preconditioning analysis specified for Phase 6.
    """
    print("\n" + "=" * 70)
    print("SWEEP H4 — HET Condition number and α scaling")
    print("=" * 70)
    print(
        f"\n  {'N':>4}  {'κ(A)':>10}  {'α':>10}  "
        f"{'λ_D [μm]':>10}  {'α_bc':>8}"
    )
    print("  " + "-" * 50)

    for N in (4, 8, 16, 32):
        cfg = HETConfig(N=N, epsilon=0.01, rho_profile="gaussian")
        prob = HETPoissonProblem1D(cfg)
        print(
            f"  {N:>4}  {prob.kappa:>10.4f}  "
            f"{cfg.alpha:>10.1f}  "
            f"{cfg.lambda_D*1e6:>10.3f}  "
            f"{cfg.alpha_bc:>8.2f}"
        )


# ── Data Exportation ──────────────────────────────────────────────────────────

def save_het_results(results: list[dict], filename: str) -> None:
    """
    Serialises aggregated HET benchmark scalar metrics to a CSV format, 
    systematically excluding high-dimensional spatial arrays to preserve readability.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RESULTS_DIR / filename

    scalar_keys = [
        k for k in results[0].keys()
        if not isinstance(results[0][k], np.ndarray)
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in scalar_keys})

    print(f"\n  Saved to {filepath}")


# ── Graphical Visualisation ───────────────────────────────────────────────────

def plot_het_solutions(results: list[dict], save_fig: bool = False) -> None:
    """
    Generates a comparative spatial visualisation of the potential profile φ̃(x̃) 
    utilising the Thomas, HHL, and VQLS algorithms.

    Constructs individual subplots per execution configuration, dynamically 
    overlaying the analytical resolution exclusively when a mathematically 
    derived closed-form solution exists.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plot.")
        return

    n_plots = len(results)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    for ax, r in zip(axes, results):
        x = r["x"]
        ax.plot(x, r["u_thomas"], "g-o",  ms=4, lw=1.5, label="Thomas")
        ax.plot(x, r["u_hhl"],    "b--s", ms=4, lw=1.5, label="HHL")
        ax.plot(x, r["u_vqls"],   "r--^", ms=4, lw=1.5, label="VQLS")
        
        if r["u_exact"] is not None:
            ax.plot(x, r["u_exact"], "k-", lw=2, label="Analytical")

        ax.set_xlabel(r"$\tilde{x}$")
        ax.set_ylabel(r"$\tilde{\phi}$")
        ax.set_title(
            f"N={r['N']}, {r['profile']}, "
            f"α={r['alpha']:.0f}, α_bc={r['alpha_bc']:.1f}"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_fig:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / "het_solutions.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Figure saved to {path}")

    plt.show()


# ── Primary Execution Orchestrator ────────────────────────────────────────────

def main() -> None:
    """
    Primary driver function for the HET plasma benchmark suite.
    """
    print("\n" + "=" * 70)
    print("HET Plasma Poisson Benchmark")
    print("HHL and VQLS on the 1D axial discharge channel")
    print("=" * 70)

    # Condition number diagnostics — fast, bypasses quantum execution.
    sweep_h4()

    # Homogeneous BCs — pure algorithmic fidelity evaluation.
    results_h1 = sweep_h1(verbose=True)
    save_het_results(results_h1, "sweep_h1_homogeneous.csv")
    plot_het_solutions(results_h1, save_fig=True)

    # Physical BCs — primary HHL/VQLS stability test.
    results_h2 = sweep_h2(verbose=True)
    save_het_results(results_h2, "sweep_h2_physical_bcs.csv")
    plot_het_solutions(results_h2, save_fig=True)

    # Comprehensive profile topology analysis.
    results_h3 = sweep_h3(verbose=True)
    save_het_results(results_h3, "sweep_h3_all_profiles.csv")
    plot_het_solutions(results_h3, save_fig=True)


if __name__ == "__main__":
    main()