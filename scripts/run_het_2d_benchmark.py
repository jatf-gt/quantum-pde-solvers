"""
2-D Hall Effect Thruster Plasma Poisson Benchmark.

Physical context
----------------
This script benchmarks the HHL and VQLS quantum linear solvers on the
2-D electrostatic Poisson equation arising in Hall Effect Thruster (HET)
plasma modelling. Two test cases are considered:

Case A — Sinusoidal charge density, homogeneous BCs (analytical solution).
    Source: f(x̃,ỹ) = -2π² sin(πx̃) sin(πỹ)
    Solution: φ̃(x̃,ỹ) = sin(πx̃) sin(πỹ)
    This manufactured solution enables rigorous quantitative error
    assessment independent of any classical reference solver.

Case B — Boeuf-Garrigues charge density, physical BCs (V_d = 300 V).
    Source: prescribed 2-D Gaussian profile approximating the
    steady-state plasma density of Boeuf & Garrigues (1998).
    Reference: Thomas line-Jacobi on a refined mesh.
    This case tests the solver on a physically realistic configuration.

Both cases use N=4 (2 qubits per row sub-problem) for computational
tractability. The framework is designed for straightforward scaling to
N=8 or N=16 on HPC resources.

Output artefacts
----------------
    figure_a_2d_het_sinusoidal.pdf  : Case A solution contours and errors
    figure_b_2d_het_physical.pdf    : Case B solution contours
    figure_c_2d_het_efield.pdf      : Electric field vector plots
    het_2d_metrics.csv              : All scalar metrics

References
----------
Boeuf & Garrigues, J. Appl. Phys. 84(7), 3541-3554 (1998).
Ghafourpour & Laizet, Phys. Rev. Applied 24, 024032 (2025).
Bravo-Prieto et al., Quantum 7, 1188 (2023).
Harrow, Hassidim & Lloyd, Phys. Rev. Lett. 103, 150502 (2009).
"""
from __future__ import annotations

import csv
import time
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── System Path Resolution ────────────────────────────────────────────────────

# Dynamically resolve the project root directory (one level up from this script)
# and append it to the system path to enable absolute imports.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from problems.het_plasma_2d import (
    HETConfig2D,
    HETPoissonProblem2D,
    HETSinusoidalProblem2D,
)
from solvers.classical.thomas_2d import thomas_solve_2d
from solvers.quantum.hhl_2d import hhl_solve_2d
from solvers.quantum.vqls_1d import VQLSConfig1D
from solvers.quantum.vqls_2d import VQLSConfig2D, vqls_solve_2d

RESULTS_DIR = Path("results/het_2d")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family":   "serif",
    "font.size":     11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi":    130,
    "lines.linewidth": 1.8,
})

COLOURS = {
    "thomas":     "#2ca02c",
    "hhl":        "#1f77b4",
    "vqls":       "#d62728",
    "analytical": "#000000",
}

# -- Solver configurations ----------------------------------------------------

_INNER = VQLSConfig1D(
    n_layers    = 3,
    optimiser   = "COBYLA",
    max_iter    = 1000,
    tol         = 1e-2,
    random_seed = 0,
    verbose     = False,
)
VQLS_CFG = VQLSConfig2D(
    inner_config = _INNER,
    warm_start   = True,
    verbose      = True,
)


# -- Utility ------------------------------------------------------------------

def _rel_err_2d(u: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Pointwise absolute relative error in percent.

    Nodes where |ref| < 1e-4·max|ref| are masked to NaN.

    Parameters
    ----------
    u : np.ndarray, shape (N, N)
    ref : np.ndarray, shape (N, N)

    Returns
    -------
    err : np.ndarray, shape (N, N)
    """
    scale = np.max(np.abs(ref))
    mask  = np.abs(ref) > 1e-4 * scale
    return np.where(mask, np.abs(u - ref) / np.abs(ref) * 100.0, np.nan)


def _max_rel_err(u: np.ndarray, ref: np.ndarray) -> float:
    """Maximum relative error in percent, excluding masked nodes."""
    err   = _rel_err_2d(u, ref)
    valid = err[~np.isnan(err)]
    return float(np.max(valid)) if valid.size > 0 else float("nan")


def _print_header(title: str) -> None:
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


def _print_row(
    label    : str,
    iters    : int,
    converged: bool,
    rel_err  : float,
    abs_err  : float,
    residual : float,
    elapsed  : float,
) -> None:
    conv_str = "Yes" if converged else "No "
    print(
        f"  {label:<12} {iters:>5}  {conv_str}  "
        f"{rel_err:>10.3f}%  {abs_err:>12.4e}  "
        f"{residual:>12.4e}  {elapsed:>8.2f}s"
    )


# -- Case A: sinusoidal source, analytical solution ---------------------------

def run_case_a(cfg: HETConfig2D) -> dict:
    """
    Run Case A: 2-D HET Poisson with sinusoidal source and homogeneous
    BCs. Compares Thomas, HHL, and VQLS against the analytical solution
    φ̃(x̃,ỹ) = sin(πx̃) sin(πỹ).

    Parameters
    ----------
    cfg : HETConfig2D
        Physical and numerical configuration.

    Returns
    -------
    dict
        Solution fields, error arrays, electric field components,
        and scalar metrics for all three solvers.
    """
    _print_header("Case A — 2-D HET Sinusoidal Source (Analytical Solution)")

    problem   = HETSinusoidalProblem2D(cfg)
    u_exact   = problem.analytical_solution()
    Ex_exact, Ey_exact = problem.analytical_electric_field()

    print(f"  {problem.summary()}")
    print(f"  max|φ̃_exact| = {np.max(np.abs(u_exact)):.4f}  "
          f"(expected 1.0 at domain centre)")
    print(f"\n  {'Solver':<12} {'Iters':>5}  {'Conv':>4}  "
          f"{'MaxRelErr':>10}  {'MaxAbsErr':>12}  "
          f"{'Residual':>12}  {'Time':>8}")
    print(f"  {'─'*72}")

    results = {"cfg": cfg, "problem": problem, "u_exact": u_exact,
               "Ex_exact": Ex_exact, "Ey_exact": Ey_exact}

    for label, solver_fn, solver_kwargs in [
        ("Thomas-2D", thomas_solve_2d, {}),
        ("HHL-2D",    hhl_solve_2d,    {}),
        ("VQLS-2D",   vqls_solve_2d,   {"config": VQLS_CFG}),
    ]:
        t0 = time.perf_counter()
        r  = solver_fn(problem, **solver_kwargs)
        t  = time.perf_counter() - t0

        _print_row(
            label, r.iterations, r.converged,
            _max_rel_err(r.u, u_exact),
            float(np.max(np.abs(r.u - u_exact))),
            r.euclidean_residual,
            t,
        )
        results[label] = {"result": r, "time": t}

    return results


# -- Case B: Boeuf-Garrigues profile, physical BCs ----------------------------

def run_case_b(cfg: HETConfig2D) -> dict:
    """
    Run Case B: 2-D HET Poisson with the Boeuf-Garrigues Gaussian charge
    density profile and physical BCs (V_d = 300 V).

    The Thomas line-Jacobi solution on a refined mesh serves as the
    reference. The electric field magnitude is compared qualitatively
    against the 1-D result of Boeuf & Garrigues (1998), Fig. 3.

    Parameters
    ----------
    cfg : HETConfig2D
        Physical and numerical configuration.

    Returns
    -------
    dict
        Solution fields, error metrics, and electric field arrays.
    """
    _print_header("Case B — 2-D HET Boeuf-Garrigues Profile, V_d = 300 V")

    problem = HETPoissonProblem2D(cfg)
    print(f"  {problem.summary()}")

    # Refined Thomas reference.
    print("  Computing refined reference (refine_factor=9)...")
    t0    = time.perf_counter()
    u_ref = problem.classical_reference_solve(refine_factor=9)
    t_ref = time.perf_counter() - t0
    print(f"  Reference completed in {t_ref:.1f}s.")

    print(f"\n  {'Solver':<12} {'Iters':>5}  {'Conv':>4}  "
          f"{'MaxRelErr':>10}  {'MaxAbsErr':>12}  "
          f"{'Residual':>12}  {'Time':>8}")
    print(f"  {'─'*72}")

    results = {"cfg": cfg, "problem": problem, "u_ref": u_ref}

    for label, solver_fn, solver_kwargs in [
        ("Thomas-2D", thomas_solve_2d, {}),
        ("HHL-2D",    hhl_solve_2d,    {}),
        ("VQLS-2D",   vqls_solve_2d,   {"config": VQLS_CFG}),
    ]:
        t0 = time.perf_counter()
        r  = solver_fn(problem, **solver_kwargs)
        t  = time.perf_counter() - t0

        _print_row(
            label, r.iterations, r.converged,
            _max_rel_err(r.u, u_ref),
            float(np.max(np.abs(r.u - u_ref))),
            r.euclidean_residual,
            t,
        )
        results[label] = {"result": r, "time": t}

    return results


# -- Figure A: Case A solution contours and errors ----------------------------

def plot_case_a(data: dict, save: bool = True) -> None:
    """
    Generate Figure A: six-panel comparison of the 2-D HET sinusoidal
    solution for Thomas, HHL, and VQLS against the analytical solution.

    Layout (2 rows × 3 columns):
        Row 1: solution contours — Thomas | HHL | VQLS
        Row 2: absolute error vs analytical — Thomas | HHL | VQLS

    The colour scale for the solution panels is shared across all three
    solvers to enable direct visual comparison. The error panels use
    individual scales to reveal the structure of each solver's error.

    Parameters
    ----------
    data : dict
        Output of run_case_a().
    save : bool
        If True, save to RESULTS_DIR/figure_a_2d_het_sinusoidal.pdf.
    """
    cfg     = data["cfg"]
    problem = data["problem"]
    X, Y    = problem.X, problem.Y
    u_exact = data["u_exact"]

    solver_keys   = ["Thomas-2D", "HHL-2D", "VQLS-2D"]
    solver_labels = ["Thomas (classical)", "HHL (quantum)", "VQLS (variational)"]
    solver_cols   = [COLOURS["thomas"], COLOURS["hhl"], COLOURS["vqls"]]

    u_sols = [data[k]["result"].u for k in solver_keys]

    # Shared colour scale for solution panels.
    u_all    = np.stack([u_exact] + u_sols)
    u_min, u_max = u_all.min(), u_all.max()
    levels_u = np.linspace(u_min, u_max, 25)

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        r"Case A — 2-D HET Poisson: Sinusoidal Source, $\tilde{\phi} = \sin(\pi\tilde{x})\sin(\pi\tilde{y})$"
        "\nAnalytical solution available — quantitative error assessment",
        fontsize=12,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)

    for col, (u_sol, label, colour) in enumerate(
        zip(u_sols, solver_labels, solver_cols)
    ):
        # -- Row 0: solution contour ------------------------------------------
        ax = fig.add_subplot(gs[0, col])
        cf = ax.contourf(X, Y, u_sol, levels=levels_u, cmap="viridis")
        ax.contour(X, Y, u_sol, levels=levels_u,
                   colors="white", linewidths=0.3, alpha=0.4)
        fig.colorbar(cf, ax=ax, shrink=0.85)

        # Overlay analytical contour lines for direct comparison.
        ax.contour(X, Y, u_exact, levels=8,
                   colors=colour, linewidths=0.8,
                   linestyles="--", alpha=0.7)

        ax.set_xlabel(r"$\tilde{x}$")
        ax.set_ylabel(r"$\tilde{y}$")
        ax.set_title(
            f"{label}\n"
            f"Max rel. err. = {_max_rel_err(u_sol, u_exact):.2f}%"
        )
        ax.set_aspect("equal")

        # -- Row 1: absolute error --------------------------------------------
        ax = fig.add_subplot(gs[1, col])
        err = np.abs(u_sol - u_exact)
        cf  = ax.contourf(X, Y, err, levels=20, cmap="hot_r")
        fig.colorbar(cf, ax=ax, shrink=0.85,
                     label=r"$|\tilde{\phi}_{solver} - \tilde{\phi}_{exact}|$")
        ax.set_xlabel(r"$\tilde{x}$")
        ax.set_ylabel(r"$\tilde{y}$")
        ax.set_title(f"{label} — absolute error")
        ax.set_aspect("equal")

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / "figure_a_2d_het_sinusoidal.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"\n  Figure A saved to {path}")
    plt.show()


# -- Figure B: Case B solution contours ---------------------------------------

def plot_case_b(data: dict, save: bool = True) -> None:
    """
    Generate Figure B: four-panel comparison of the 2-D HET
    Boeuf-Garrigues solution for Thomas and VQLS.

    Layout (2 rows × 2 columns):
        (a) Thomas solution contour
        (b) VQLS solution contour
        (c) Thomas absolute error vs refined reference
        (d) VQLS absolute error vs refined reference

    Parameters
    ----------
    data : dict
        Output of run_case_b().
    save : bool
        If True, save to RESULTS_DIR/figure_b_2d_het_physical.pdf.
    """
    cfg     = data["cfg"]
    problem = data["problem"]
    X, Y    = problem.X, problem.Y
    u_ref   = data["u_ref"]

    r_thomas = data["Thomas-2D"]["result"]
    r_vqls   = data["VQLS-2D"]["result"]

    u_all    = np.stack([r_thomas.u, r_vqls.u])
    u_min, u_max = u_all.min(), u_all.max()
    levels_u = np.linspace(u_min, u_max, 25)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(
        f"Case B — 2-D HET Poisson: Boeuf-Garrigues Profile, $V_d = {cfg.V_discharge:.0f}$ V\n"
        "Reference: Thomas line-Jacobi on refined mesh (refine_factor=9)\n"
        "Physical parameters: Boeuf & Garrigues (1998), J. Appl. Phys. 84, 3541",
        fontsize=11,
    )

    panels = [
        (axes[0, 0], r_thomas.u, levels_u, "viridis",
         f"Thomas-2D solution ({r_thomas.iterations} iters)"),
        (axes[0, 1], r_vqls.u,   levels_u, "viridis",
         f"VQLS-2D solution ({r_vqls.iterations} iters)"),
        (axes[1, 0], np.abs(r_thomas.u - u_ref), None, "hot_r",
         "Thomas-2D absolute error vs reference"),
        (axes[1, 1], np.abs(r_vqls.u   - u_ref), None, "hot_r",
         "VQLS-2D absolute error vs reference"),
    ]

    for ax, Z, lvls, cmap, title in panels:
        levels = lvls if lvls is not None else np.linspace(0, Z.max() or 1, 20)
        cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap)
        fig.colorbar(cf, ax=ax, shrink=0.85)
        ax.set_xlabel(r"$\tilde{x} = x/L_x$")
        ax.set_ylabel(r"$\tilde{y} = y/L_y$")
        ax.set_title(title)
        ax.set_aspect("equal")

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / "figure_b_2d_het_physical.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Figure B saved to {path}")
    plt.show()


# -- Figure C: electric field vector plots ------------------------------------

def plot_electric_field(data_a: dict, save: bool = True) -> None:
    """
    Generate Figure C: electric field vector plots for Case A, comparing
    the analytical field against the Thomas and VQLS numerical fields.

    The electric field is recovered from the non-dimensional potential
    via second-order centred finite differences and converted to
    physical units [V/m] using φ_0/L_x and φ_0/L_y.

    Layout (1 row × 3 columns):
        Left:   Analytical electric field magnitude and vectors
        Centre: Thomas electric field magnitude and vectors
        Right:  VQLS electric field magnitude and vectors

    Parameters
    ----------
    data_a : dict
        Output of run_case_a().
    save : bool
        If True, save to RESULTS_DIR/figure_c_2d_het_efield.pdf.
    """
    cfg     = data_a["cfg"]
    problem = data_a["problem"]
    X, Y    = problem.X, problem.Y
    N       = cfg.N
    h       = problem.h

    Ex_exact, Ey_exact = data_a["Ex_exact"], data_a["Ey_exact"]
    E_mag_exact = np.sqrt(Ex_exact**2 + Ey_exact**2)

    def _numerical_efield(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Recover E_x and E_y from the non-dimensional potential u via
        second-order centred finite differences at interior nodes.
        """
        # Augment with zero boundary values.
        phi = np.zeros((N + 2, N + 2))
        phi[1:N+1, 1:N+1] = u

        # Centred differences at interior nodes.
        dphidx = -(phi[2:N+2, 1:N+1] - phi[0:N,   1:N+1]) / (2.0 * h)
        dphidy = -(phi[1:N+1, 2:N+2] - phi[1:N+1, 0:N  ]) / (2.0 * h)

        return dphidx * cfg.phi_0 / cfg.L_x, dphidy * cfg.phi_0 / cfg.L_y

    solver_keys   = ["Thomas-2D", "VQLS-2D"]
    solver_labels = ["Thomas (classical)", "VQLS (variational)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        r"Case A — 2-D HET Electric Field: $\mathbf{E} = -\nabla\phi$"
        "\nColour: field magnitude [V/m]; arrows: field direction",
        fontsize=12,
    )

    # Shared colour scale.
    E_max = float(E_mag_exact.max()) * 1.2

    for ax, (key, label) in zip(
        axes[1:],
        zip(solver_keys, solver_labels),
    ):
        u_sol      = data_a[key]["result"].u
        Ex, Ey     = _numerical_efield(u_sol)
        E_mag      = np.sqrt(Ex**2 + Ey**2)

        cf = ax.contourf(X, Y, E_mag, levels=20, cmap="plasma",
                         vmin=0, vmax=E_max)
        ax.quiver(X, Y, Ex / E_mag, Ey / E_mag,
                  alpha=0.6, scale=25, color="white", width=0.004)
        fig.colorbar(cf, ax=ax, label=r"$|\mathbf{E}|$ [V/m]")
        ax.set_xlabel(r"$\tilde{x}$")
        ax.set_ylabel(r"$\tilde{y}$")
        ax.set_title(
            f"{label}\n"
            f"Max |E| = {E_mag.max():.2e} V/m"
        )
        ax.set_aspect("equal")

    # Analytical field.
    ax = axes[0]
    cf = ax.contourf(X, Y, E_mag_exact, levels=20, cmap="plasma",
                     vmin=0, vmax=E_max)
    ax.quiver(X, Y, Ex_exact / E_mag_exact, Ey_exact / E_mag_exact,
              alpha=0.6, scale=25, color="white", width=0.004)
    fig.colorbar(cf, ax=ax, label=r"$|\mathbf{E}|$ [V/m]")
    ax.set_xlabel(r"$\tilde{x}$")
    ax.set_ylabel(r"$\tilde{y}$")
    ax.set_title(
        f"Analytical\nMax |E| = {E_mag_exact.max():.2e} V/m"
    )
    ax.set_aspect("equal")

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / "figure_c_2d_het_efield.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Figure C saved to {path}")
    plt.show()


# -- CSV export ---------------------------------------------------------------

def export_csv(data_a: dict, data_b: dict) -> None:
    """
    Export all scalar benchmark metrics to CSV.

    Parameters
    ----------
    data_a : dict
        Output of run_case_a().
    data_b : dict
        Output of run_case_b().
    """
    filepath = RESULTS_DIR / "het_2d_metrics.csv"
    fieldnames = [
        "case", "N", "solver", "iterations", "converged",
        "max_rel_err_pct", "max_abs_err", "residual", "time_s",
    ]

    rows = []

    for key, label in [
        ("Thomas-2D", "Thomas-2D"),
        ("HHL-2D",    "HHL-2D"),
        ("VQLS-2D",   "VQLS-2D"),
    ]:
        r   = data_a[key]["result"]
        ref = data_a["u_exact"]
        rows.append({
            "case":            "A_sinusoidal_hom",
            "N":               data_a["cfg"].N,
            "solver":          label,
            "iterations":      r.iterations,
            "converged":       r.converged,
            "max_rel_err_pct": _max_rel_err(r.u, ref),
            "max_abs_err":     float(np.max(np.abs(r.u - ref))),
            "residual":        r.euclidean_residual,
            "time_s":          data_a[key]["time"],
        })

    for key, label in [
        ("Thomas-2D", "Thomas-2D"),
        ("HHL-2D",    "HHL-2D"),
        ("VQLS-2D",   "VQLS-2D"),
    ]:
        r   = data_b[key]["result"]
        ref = data_b["u_ref"]
        rows.append({
            "case":            "B_gaussian_phys",
            "N":               data_b["cfg"].N,
            "solver":          label,
            "iterations":      r.iterations,
            "converged":       r.converged,
            "max_rel_err_pct": _max_rel_err(r.u, ref),
            "max_abs_err":     float(np.max(np.abs(r.u - ref))),
            "residual":        r.euclidean_residual,
            "time_s":          data_b[key]["time"],
        })

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Metrics exported to {filepath}")


# -- Main entry point ---------------------------------------------------------

def main() -> None:
    """
    Execute the full 2-D HET benchmark and generate all output artefacts.

    Execution sequence:
        1. Case A: sinusoidal source, analytical solution (N=4)
        2. Case B: Boeuf-Garrigues profile, physical BCs (N=4)
        3. Figure generation (3 figures)
        4. CSV export

    Estimated total runtime: 20–45 minutes on a standard workstation.
    Reduce max_iter in VQLS_CFG.inner_config or cfg.max_iter to
    accelerate at the cost of solution accuracy.
    """
    print("\n" + "═"*65)
    print("  2-D HET PLASMA POISSON BENCHMARK")
    print("  Quantum vs Classical Solver Comparison")
    print("  Reference: Boeuf & Garrigues (1998), J. Appl. Phys. 84, 3541")
    print("═"*65)

    cfg = HETConfig2D(N=4, epsilon=0.01, max_iter=300, V_discharge=300.0)

    t_total = time.perf_counter()
    data_a  = run_case_a(cfg)
    data_b  = run_case_b(cfg)
    elapsed = time.perf_counter() - t_total

    print(f"\n  All cases completed in {elapsed:.1f}s.")
    print("  Generating figures...")

    plot_case_a(data_a, save=True)
    plot_case_b(data_b, save=True)
    plot_electric_field(data_a, save=True)
    export_csv(data_a, data_b)

    print(f"\n  All outputs saved to {RESULTS_DIR.resolve()}")
    print("═"*65)


if __name__ == "__main__":
    main()