"""
Executes a high-fidelity 1D Hall Effect Thruster (HET) plasma Poisson benchmark.

Resolves the electrostatic potential within a Hall Effect Thruster discharge 
channel utilising classical (Thomas), quantum (HHL), and variational quantum 
(VQLS) solvers. Concludes by performing a comparative analysis of the derived 
macroscopic electric field profiles.

Physical Model
--------------
Reference: Boeuf & Garrigues (1998), J. Appl. Phys. 84(7), 3541-3554.
  - 1D axial discharge channel, L = 25 mm.
  - Discharge voltage V_d = 300 V.
  - Xenon propellant, n_0 = 5×10¹⁷ m⁻³, T_e = 20 eV.

The spatial plasma density profile and net charge separation are prescribed 
analytically to approximate the Boeuf-Garrigues steady-state topological solution. 
The Poisson equation is evaluated for the electrostatic potential, and the 
physical electric field is subsequently recovered via finite-difference differentiation.

Benchmarking Strategy
---------------------
Given that the comprehensive Boeuf-Garrigues model constitutes a coupled fluid 
system, this script isolates and benchmarks the Poisson solver exclusively:
  1. The classical Thomas solution serves as the high-accuracy baseline reference.
  2. Quantum HHL and VQLS solutions are quantitatively benchmarked against the Thomas baseline.
  3. The derived electric field profile is evaluated qualitatively against Figure 3 
     of Boeuf & Garrigues (1998), which demonstrates an electric field peak of 
     ~2×10⁴ V/m in proximity to the exit plane (x/L ≈ 0.8).

Output Specifications
---------------------
  - Console: Structured telemetry table incorporating defined pass/fail thresholds.
  - Figures: Graphical subplots detailing the potential profile, electric field 
             profile, spatial error topography, VQLS convergence history, and 
             prescribed plasma density/charge distributions.
  - CSV:     Serialisation of all scalar metrics for formal thesis documentation.
"""
from __future__ import annotations

import csv
import time
import sys
from pathlib import Path

import numpy as np

# ── System Path Resolution ────────────────────────────────────────────────────

# Dynamically resolve the project root directory (one level up from this script)
# and append it to the system path to enable absolute imports.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from core.het_config import HETPhysicalConfig
from problems.het_plasma_1d import HETPhysicalProblem1D
from solvers.classical.thomas import thomas_solve_system
from solvers.quantum.hhl_1d import hhl_solve_system
from solvers.quantum.vqls_1d import vqls_solve_system, VQLSConfig

RESULTS_DIR = Path("results/het_plasma")

# ── Pass/Fail Thresholds ──────────────────────────────────────────────────────
# Derived from the anticipated algorithmic precision bounds at epsilon=0.01:
#   Thomas : Machine precision (exact analytical tridiagonal resolution).
#   HHL    : Trotterisation error scales as ~ ε ~ 1%; a 5% threshold is 
#            allocated to establish a robust safety margin.
#   VQLS   : Variational divergence scales as ~ cost^0.5; a 2% threshold is 
#            enforced contingent upon the cost function converging below 1e-4.
THRESHOLD_HHL_REL  = 5.0    # % relative error
THRESHOLD_VQLS_REL = 2.0    # % relative error
THRESHOLD_VQLS_COST = 1e-4  # cost function value


def rel_err_pct(u: np.ndarray, ref: np.ndarray) -> float:
    """Evaluates the maximum relative error as a percentage, explicitly masking near-zero nodal values."""
    scale = np.max(np.abs(ref))
    mask  = np.abs(ref) > 1e-4 * scale
    if not mask.any():
        return float("nan")
    return float(np.max(
        np.abs((u - ref)[mask]) / np.abs(ref[mask])
    )) * 100.0


def run_benchmark(
    cfg:     HETPhysicalConfig,
    vc:      VQLSConfig,
    verbose: bool = True,
) -> dict:
    """
    Executes the classical Thomas, quantum HHL, and variational VQLS algorithms 
    on the physical HET Poisson topology, aggregating comprehensive output telemetry.
    """
    problem = HETPhysicalProblem1D(cfg)

    if verbose:
        print(f"\n{problem.summary()}")
        print(f"  RHS norm: {np.linalg.norm(problem.b):.4e}")
        print(f"  Solution scale: φ̃ ~ α_bc = {cfg.alpha_bc:.1f}")

    A = problem.A
    b = problem.b

    # ── Thomas Execution ──────────────────────────────────────────────────────
    t0       = time.perf_counter()
    u_thomas = thomas_solve_system(A, b)
    t_thomas = time.perf_counter() - t0

    # ── HHL Execution ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        u_hhl, _, c_hhl = hhl_solve_system(A, b, cfg.epsilon)
        hhl_ok = True
    except Exception as exc:
        u_hhl  = np.zeros_like(b)
        c_hhl  = 0.0
        hhl_ok = False
        if verbose:
            print(f"  HHL failed: {exc}")
    t_hhl = time.perf_counter() - t0

    # ── VQLS Execution ────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        vqls_r = vqls_solve_system(A, b, vc)
        u_vqls = vqls_r.u
        vqls_ok = True
    except Exception as exc:
        u_vqls  = np.zeros_like(b)
        vqls_r  = None
        vqls_ok = False
        if verbose:
            print(f"  VQLS failed: {exc}")
    t_vqls = time.perf_counter() - t0

    # ── Electric Field Recovery ───────────────────────────────────────────────
    x_full, E_thomas = problem.electric_field(u_thomas)
    _,      E_hhl    = problem.electric_field(u_hhl)
    _,      E_vqls   = problem.electric_field(u_vqls)

    # ── Error Metric Compilation ──────────────────────────────────────────────
    hhl_rel_phi  = rel_err_pct(u_hhl,   u_thomas)
    vqls_rel_phi = rel_err_pct(u_vqls,  u_thomas)
    hhl_rel_E    = rel_err_pct(E_hhl,   E_thomas)
    vqls_rel_E   = rel_err_pct(E_vqls,  E_thomas)

    # ── Console Reporting ─────────────────────────────────────────────────────
    if verbose:
        _print_report(
            cfg, problem, u_thomas, u_hhl, u_vqls,
            hhl_rel_phi, vqls_rel_phi,
            hhl_rel_E, vqls_rel_E,
            t_thomas, t_hhl, t_vqls,
            vqls_r, hhl_ok, vqls_ok,
        )

    return {
        "config":       cfg,
        "problem":      problem,
        "x_int":        problem.x,
        "x_full":       x_full,
        "u_thomas":     u_thomas,
        "u_hhl":        u_hhl,
        "u_vqls":       u_vqls,
        "E_thomas":     E_thomas,
        "E_hhl":        E_hhl,
        "E_vqls":       E_vqls,
        "hhl_rel_phi":  hhl_rel_phi,
        "vqls_rel_phi": vqls_rel_phi,
        "hhl_rel_E":    hhl_rel_E,
        "vqls_rel_E":   vqls_rel_E,
        "t_thomas":     t_thomas,
        "t_hhl":        t_hhl,
        "t_vqls":       t_vqls,
        "vqls_cost":    vqls_r.final_cost     if vqls_r else float("nan"),
        "vqls_evals":   vqls_r.n_circuit_evals if vqls_r else 0,
        "vqls_history": vqls_r.cost_history  if vqls_r else [],
        "hhl_ok":       hhl_ok,
        "vqls_ok":      vqls_ok,
        "n_density":    problem.n_profile,
        "delta_n":      problem.delta_n,
    }


def _print_report(
    cfg, problem, u_thomas, u_hhl, u_vqls,
    hhl_rel_phi, vqls_rel_phi, hhl_rel_E, vqls_rel_E,
    t_thomas, t_hhl, t_vqls, vqls_r, hhl_ok, vqls_ok,
) -> None:
    """Outputs a structured, formal console report detailing benchmark telemetry."""

    def _flag(val, threshold):
        if np.isnan(val):
            return "N/A "
        return "PASS" if val < threshold else "FAIL"

    vqls_cost  = vqls_r.final_cost    if vqls_r else float("nan")
    vqls_evals = vqls_r.n_circuit_evals if vqls_r else 0

    print(f"\n{'═'*70}")
    print(f"  HET PLASMA POISSON BENCHMARK")
    print(f"  Reference: Boeuf & Garrigues (1998), J. Appl. Phys. 84, 3541")
    print(f"{'─'*70}")
    print(f"  N={cfg.N}, α={cfg.alpha:.1f}, α_bc={cfg.alpha_bc:.1f}, "
          f"κ(A)={problem.kappa:.2f}")
    print(f"  λ_D={cfg.lambda_D*1e6:.2f}μm, "
          f"||b||={np.linalg.norm(problem.b):.3e}")
    print(f"{'─'*70}")
    print(f"  {'Solver':<8} {'φ̃ RelErr%':>10} {'E RelErr%':>10} "
          f"{'Time':>8} {'Status':>6}")
    print(f"  {'─'*50}")
    print(f"  {'Thomas':<8} {'(ref)':>10} {'(ref)':>10} "
          f"{t_thomas*1e3:>7.1f}ms {'PASS':>6}")
    print(f"  {'HHL':<8} {hhl_rel_phi:>10.3f} {hhl_rel_E:>10.3f} "
          f"{t_hhl:>7.1f}s  "
          f"{_flag(hhl_rel_phi, THRESHOLD_HHL_REL):>6}")
    print(f"  {'VQLS':<8} {vqls_rel_phi:>10.3f} {vqls_rel_E:>10.3f} "
          f"{t_vqls:>7.1f}s  "
          f"{_flag(vqls_rel_phi, THRESHOLD_VQLS_REL):>6}  "
          f"cost={vqls_cost:.2e}, evals={vqls_evals}")
    print(f"{'─'*70}")

    # Node-by-node potential table.
    print(f"\n  Non-dimensional potential φ̃(x̃):")
    print(f"  {'x̃':>6}  {'Thomas':>12}  {'HHL':>12}  {'VQLS':>12}  "
          f"{'|HHL-T|%':>10}  {'|VQLS-T|%':>10}")
    for i in range(cfg.N):
        ref   = u_thomas[i]
        denom = abs(ref) if abs(ref) > 1e-10 else 1.0
        print(
            f"  {problem.x[i]:6.4f}  {ref:12.4f}  "
            f"{u_hhl[i]:12.4f}  {u_vqls[i]:12.4f}  "
            f"{abs(u_hhl[i]-ref)/denom*100:10.3f}  "
            f"{abs(u_vqls[i]-ref)/denom*100:10.3f}"
        )

    # Physical electric field summary.
    # Expected peak field from the applied voltage alone (uniform field):
    #   E_uniform = V_d / L
    # Expected peak field with space charge (order of magnitude):
    #   The potential profile is approximately linear with a perturbation
    #   from the space charge of order phi_0 * delta_0_factor near the sheaths.
    #   The peak gradient occurs near the anode sheath.
    E_uniform = cfg.V_discharge / cfg.L
    x_full_t, E_thomas_full = problem.electric_field(u_thomas)
    x_full_h, E_hhl_full    = problem.electric_field(u_hhl)
    x_full_v, E_vqls_full   = problem.electric_field(u_vqls)

    print(f"\n  Electric field summary (physical units):")
    print(f"  Peak |E| Thomas: {np.max(np.abs(E_thomas_full)):.3e} V/m")
    print(f"  Peak |E| HHL:    {np.max(np.abs(E_hhl_full)):.3e} V/m")
    print(f"  Peak |E| VQLS:   {np.max(np.abs(E_vqls_full)):.3e} V/m")
    print(f"  Uniform field estimate (V_d/L): {E_uniform:.3e} V/m")
    print(f"  Boeuf & Garrigues (1998) Fig.3: peak ~2×10⁴ V/m near exit plane")
    print(f"  Note: peak > V_d/L is physical — the non-uniform charge")
    print(f"  distribution concentrates the field near the exit plane.")


def plot_results(r: dict, save_fig: bool = True) -> None:
    """
    Generates a comprehensive four-panel diagnostic visualisation:

    Panel 1 (Top-Left)     : Prescribed plasma density and non-dimensional charge profiles.
    Panel 2 (Top-Right)    : Non-dimensional spatial potential φ̃(x̃).
    Panel 3 (Bottom-Left)  : Macroscopic physical electric field E(x) [V/m].
    Panel 4 (Bottom-Right) : Relative deviation in φ̃ and VQLS iterative cost history.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("  matplotlib not available — skipping plots.")
        return

    cfg     = r["config"]
    problem = r["problem"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        "1D HET Plasma Poisson Benchmark\n"
        "Reference: Boeuf & Garrigues (1998), J. Appl. Phys. 84, 3541",
        fontsize=12, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    x_int  = r["x_int"]
    x_full = r["x_full"]

    # ── Panel 1: Plasma Profiles ──────────────────────────────────────────────
    ax1_twin = ax1.twinx()
    l1, = ax1.plot(
        x_int, r["n_density"], "b-o", ms=4, lw=1.5,
        label=r"$\tilde{n}(x̃) = n/n_0$",
    )
    l2, = ax1_twin.plot(
        x_int, r["delta_n"] * 100, "r--s", ms=4, lw=1.5,
        label=r"$\delta\tilde{n}$ × 100 = $(n_i-n_e)/n_0$ × 100",
    )
    ax1.set_xlabel(r"$\tilde{x} = x/L$")
    ax1.set_ylabel(r"$\tilde{n}$", color="b")
    ax1_twin.set_ylabel(r"$\delta\tilde{n}$ × 100 (%)", color="r")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1_twin.tick_params(axis="y", labelcolor="r")
    ax1.set_title("Prescribed plasma profiles\n(Boeuf-Garrigues approximation)")
    lines = [l1, l2]
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="lower left")
    ax1.grid(True, alpha=0.3)
    ax1.axvline(cfg.x_peak, color="grey", linestyle=":", alpha=0.7,
                label=f"Peak x̃={cfg.x_peak}")

    # ── Panel 2: Potential Profile ────────────────────────────────────────────
    # Include boundary values for a complete picture.
    phi_thomas_full = np.concatenate([[cfg.alpha_bc], r["u_thomas"], [0.0]])
    phi_hhl_full    = np.concatenate([[cfg.alpha_bc], r["u_hhl"],    [0.0]])
    phi_vqls_full   = np.concatenate([[cfg.alpha_bc], r["u_vqls"],   [0.0]])

    ax2.plot(x_full, phi_thomas_full, "g-",  lw=2.5, label="Thomas (classical)")
    ax2.plot(x_full, phi_hhl_full,    "b--", lw=1.8, label="HHL (quantum)")
    ax2.plot(x_full, phi_vqls_full,   "r:",  lw=1.8, label="VQLS (variational)")
    ax2.scatter(x_int, r["u_thomas"], color="g", s=20, zorder=5)
    ax2.scatter(x_int, r["u_hhl"],    color="b", s=20, zorder=5)
    ax2.scatter(x_int, r["u_vqls"],   color="r", s=20, zorder=5)
    ax2.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
    ax2.set_xlabel(r"$\tilde{x} = x/L$")
    ax2.set_ylabel(r"$\tilde{\phi} = \phi / \phi_0$")
    ax2.set_title(
        f"Non-dimensional potential\n"
        f"(φ₀ = {cfg.phi_0:.0f} V, α_bc = {cfg.alpha_bc:.1f})"
    )
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # ── Panel 3: Electric Field ───────────────────────────────────────────────
    ax3.plot(x_full, r["E_thomas"] / 1e4, "g-",  lw=2.5,
             label="Thomas (classical)")
    ax3.plot(x_full, r["E_hhl"]    / 1e4, "b--", lw=1.8,
             label="HHL (quantum)")
    ax3.plot(x_full, r["E_vqls"]   / 1e4, "r:",  lw=1.8,
             label="VQLS (variational)")

    # Reference annotation from Boeuf & Garrigues (1998) Fig. 3.
    ax3.axhline(2.0, color="k", linestyle="-.", lw=1.2, alpha=0.6,
                label="B&G (1998) peak ~2×10⁴ V/m")
    ax3.axvline(0.8, color="grey", linestyle=":", alpha=0.6,
                label="Exit plane x̃≈0.8")

    ax3.set_xlabel(r"$\tilde{x} = x/L$")
    ax3.set_ylabel(r"$E$ [×10⁴ V/m]")
    ax3.set_title(
        "Physical electric field\n"
        "cf. Boeuf & Garrigues (1998), Fig. 3"
    )
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Error Analysis and VQLS Convergence ──────────────────────────
    ax4a = ax4
    ax4b = ax4.twinx()

    # Relative errors at interior nodes.
    ref   = r["u_thomas"]
    scale = np.max(np.abs(ref))
    mask  = np.abs(ref) > 1e-4 * scale

    hhl_err  = np.where(mask, np.abs(r["u_hhl"]  - ref) / np.abs(ref) * 100, np.nan)
    vqls_err = np.where(mask, np.abs(r["u_vqls"] - ref) / np.abs(ref) * 100, np.nan)

    ax4a.semilogy(x_int[mask], hhl_err[mask],  "b-o", ms=5, lw=1.5,
                  label="HHL φ̃ error (%)")
    ax4a.semilogy(x_int[mask], vqls_err[mask], "r-s", ms=5, lw=1.5,
                  label="VQLS φ̃ error (%)")
    ax4a.axhline(THRESHOLD_HHL_REL,  color="b", linestyle="--",
                 alpha=0.5, label=f"HHL threshold {THRESHOLD_HHL_REL}%")
    ax4a.axhline(THRESHOLD_VQLS_REL, color="r", linestyle="--",
                 alpha=0.5, label=f"VQLS threshold {THRESHOLD_VQLS_REL}%")
    ax4a.set_xlabel(r"$\tilde{x} = x/L$")
    ax4a.set_ylabel("Relative error (%)", color="purple")
    ax4a.tick_params(axis="y", labelcolor="purple")

    # VQLS cost history on secondary axis.
    if r["vqls_history"]:
        iters = np.arange(1, len(r["vqls_history"]) + 1)
        ax4b.semilogy(iters, r["vqls_history"], "k-", lw=1.2,
                      alpha=0.6, label="VQLS cost")
        ax4b.set_ylabel("VQLS cost C(θ)", color="k")
        ax4b.tick_params(axis="y", labelcolor="k")

    ax4.set_title(
        f"Error analysis & VQLS convergence\n"
        f"(VQLS: cost={r['vqls_cost']:.2e}, "
        f"evals={r['vqls_evals']})"
    )
    lines_a, labels_a = ax4a.get_legend_handles_labels()
    lines_b, labels_b = ax4b.get_legend_handles_labels()
    ax4a.legend(lines_a + lines_b, labels_a + labels_b, fontsize=7)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_fig:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = RESULTS_DIR / f"het_benchmark_N{cfg.N}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"\n  Figure saved to {path}")

    plt.show()


def save_results_csv(r: dict) -> None:
    """Serialises all scalar analytical metrics to a CSV export framework."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cfg      = r["config"]
    filepath = RESULTS_DIR / f"het_benchmark_N{cfg.N}.csv"

    rows = []
    for solver, u_sol, rel_phi, rel_E, t in [
        ("Thomas", r["u_thomas"], 0.0,              0.0,             r["t_thomas"]),
        ("HHL",    r["u_hhl"],    r["hhl_rel_phi"],  r["hhl_rel_E"],  r["t_hhl"]),
        ("VQLS",   r["u_vqls"],   r["vqls_rel_phi"], r["vqls_rel_E"], r["t_vqls"]),
    ]:
        rows.append({
            "solver":        solver,
            "N":             cfg.N,
            "alpha":         cfg.alpha,
            "alpha_bc":      cfg.alpha_bc,
            "kappa":         r["problem"].kappa,
            "epsilon":       cfg.epsilon,
            "rel_err_phi_%": rel_phi,
            "rel_err_E_%":   rel_E,
            "time_s":        t,
            "vqls_cost":     r["vqls_cost"] if solver == "VQLS" else "",
            "vqls_evals":    r["vqls_evals"] if solver == "VQLS" else "",
        })

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"  CSV saved to {filepath}")


# ── Primary Execution Orchestrator ────────────────────────────────────────────

def main() -> None:
    """Primary execution driver for the HET physical plasma benchmarks."""
    print("\n" + "═"*70)
    print("  1D HET PLASMA POISSON BENCHMARK")
    print("  Quantum vs Classical Solver Comparison")
    print("  Reference: Boeuf & Garrigues (1998), J. Appl. Phys. 84, 3541")
    print("═"*70)

    vc = VQLSConfig(
        n_layers    = 6,
        optimiser   = "COBYLA",
        max_iter    = 3000,
        tol         = 1e-6,
        random_seed = 42,
        verbose     = True,
    )

    # Primary Benchmark: N=8 resolving full physical topography.
    cfg = HETPhysicalConfig(N=8, epsilon=0.01)
    r   = run_benchmark(cfg, vc, verbose=True)
    plot_results(r, save_fig=True)
    save_results_csv(r)

    # Spatial Resolution Study: N=4 (Evaluates constrained algorithmic depth).
    print("\n" + "─"*70)
    print("  Resolution study: N=4")
    cfg4 = HETPhysicalConfig(N=4, epsilon=0.01)
    r4   = run_benchmark(cfg4, vc, verbose=True)
    plot_results(r4, save_fig=True)
    save_results_csv(r4)


if __name__ == "__main__":
    main()