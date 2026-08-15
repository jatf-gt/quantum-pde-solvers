"""
2-D Hall Effect Thruster Plasma Poisson Benchmark.

Physical context
----------------
This script benchmarks the HHL, VQLS and QSVT quantum linear solvers on the 2-D
electrostatic Poisson equation arising in Hall Effect Thruster (HET) plasma
modelling. Two test cases are considered.

Case A — Sinusoidal charge density, homogeneous BCs (analytical solution).
    Source:   f(x̃,ỹ) = −2π² sin(πx̃) sin(πỹ)
    Solution: φ̃(x̃,ỹ) = sin(πx̃) sin(πỹ)
    This manufactured solution enables rigorous quantitative error assessment
    independent of any classical reference solver, and is the only 2-D case in
    the suite for which that is possible.

Case B — Boeuf-Garrigues charge density, physical BCs (V_d = 300 V).
    Source:    the analytical 2-D sheath profile of `HETConfig2D`, approximating
               the steady-state plasma of Boeuf & Garrigues (1998).
    Reference: fine-mesh classical solve (`benchmark.reference_2d`).
    This case tests the solvers on a physically realistic configuration.

Architecture
------------
Both cases are driven through `solvers.outer.solve`: the problem is a
`PoissonLine2D` decomposed into 1-D strips, and each solver differs only in the
inner strip solve. The comparison is therefore strictly like-for-like — same
outer iteration, same stopping test, same right-hand side assembly — with the
strip solver as the sole independent variable.

Boundary conditions
-------------------
The radial walls are grounded, φ̃(x̃,0) = φ̃(x̃,1) = 0, per the physical model:
the anode potential is applied at x̃ = 0 only. This corrects an inconsistency in
the retired implementation, which solved with the inner wall held at α_bc whilst
scoring its residual against a grounded wall — the reported residual there
belonged to a different system than the one solved. Case B numbers are
consequently not directly comparable with output predating this correction.

Output artefacts
----------------
    figure_a_2d_het_sinusoidal.pdf  : Case A solution contours and errors
    figure_b_2d_het_physical.pdf    : Case B solution contours
    figure_c_2d_het_efield.pdf      : Electric field vector plots
    het_2d_metrics.csv              : All scalar metrics

Execution time
--------------
Dominated by the quantum strip solves. At the default N = 4 the whole script
completes in a few minutes; N = 8 raises this to roughly an hour, the HHL case
dominating. Adjust N, MAX_ITER and TOL below rather than editing the solver
configurations.

References
----------
Boeuf & Garrigues, J. Appl. Phys. 84(7), 3541-3554 (1998).
Ghafourpour & Laizet, Phys. Rev. Applied 24, 024032 (2025).
Bravo-Prieto et al., Quantum 7, 1188 (2023).
Harrow, Hassidim & Lloyd, Phys. Rev. Lett. 103, 150502 (2009).
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# -- System Path Resolution ----------------------------------------------------

# Dynamically resolve the project root directory (one level up from this script)
# and append it to the system path to enable absolute imports.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from benchmark.reference_2d import fine_mesh_reference
from core.het_config import HETConfig2D
from problems.het_plasma_2d import (
    build_het_problem,
    build_het_sinusoidal,
    sinusoidal_electric_field,
    sinusoidal_solution,
)
from solvers.outer import solve as outer_solve

RESULTS_DIR = Path("results/het_2d")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -- Execution Parameters ------------------------------------------------------

# Interior nodes per direction. Must be a power of two: a strip of length N is
# amplitude-encoded on log₂(N) qubits.
N = 4

# Precision parameter of the HHL strip solves. Floored as in the generic 2-D
# sweeps: the strip operator is so well conditioned (κ(A_row) ≈ 2.36 at N = 4,
# bounded above by 3) that ten Trotter steps saturate the accuracy the outer
# iteration can exploit, and a smaller ε merely multiplies circuit depth.
EPSILON = 0.1

# Outer line-Jacobi controls, matching Section IV E of the reference literature.
TOL      = 1e-8
MAX_ITER = 300

# Mesh refinement multiplier for the Case B reference solve.
REFINE_FACTOR = 9

# Outer scheme. "jacobi" reproduces the original validated line-Jacobi loop.
SCHEME = "jacobi"

plt.rcParams.update({
    "font.family":     "serif",
    "font.size":       11,
    "axes.labelsize":  12,
    "axes.titlesize":  12,
    "legend.fontsize":  9,
    "figure.dpi":      130,
    "lines.linewidth": 1.8,
})

COLOURS = {
    "thomas":     "#2ca02c",
    "hhl":        "#1f77b4",
    "vqls":       "#d62728",
    "analytical": "#000000",
}

# Execution plan: (display label, inner strip solver, inner solver options).
# The classical entry is first so its timing anchors the console table.
SOLVERS = [
    ("Thomas-2D", "thomas", {}),
    ("HHL-2D",    "hhl",    {"epsilon": EPSILON}),
    ("VQLS-2D",   "vqls",   {"n_layers": 3, "optimiser": "COBYLA",
                             "max_iter": 1000, "tol": 1e-2, "random_seed": 0}),
]


# -- Utility -------------------------------------------------------------------

def _rel_err_2d(u: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Computes the pointwise absolute relative error as a percentage.

    Nodes satisfying |ref| < 1e-4·max|ref| are masked to NaN. The mask is
    relative to the field's own scale rather than absolute, because the HET
    potential spans several orders of magnitude across the sheath and a fixed
    threshold would either mask the entire cathode region or none of it.

    Parameters
    ----------
    u : np.ndarray
        (N, N) solver solution field.
    ref : np.ndarray
        (N, N) reference field.

    Returns
    -------
    err : np.ndarray
        (N, N) relative error [%], NaN at masked nodes.
    """
    scale = np.max(np.abs(ref))
    mask  = np.abs(ref) > 1e-4 * scale
    return np.where(mask, np.abs(u - ref) / np.abs(ref) * 100.0, np.nan)


def _max_rel_err(u: np.ndarray, ref: np.ndarray) -> float:
    """Returns the supremum of the relative error [%], excluding masked nodes."""
    err   = _rel_err_2d(u, ref)
    valid = err[~np.isnan(err)]
    return float(np.max(valid)) if valid.size > 0 else float("nan")


def _print_header(title: str) -> None:
    """Emits a delimited section header to standard output."""
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


def _print_table_header() -> None:
    """Emits the aligned column header shared by both case tables."""
    print(f"\n  {'Solver':<12} {'Iters':>5}  {'Conv':>4}  "
          f"{'MaxRelErr':>10}  {'MaxAbsErr':>12}  "
          f"{'Residual':>12}  {'Time':>8}")
    print(f"  {'─'*72}")


def _print_row(
    label:     str,
    iters:     int,
    converged: bool,
    rel_err:   float,
    abs_err:   float,
    residual:  float,
    elapsed:   float,
) -> None:
    """Emits one aligned metric row of a case table."""
    conv_str = "Yes" if converged else "No "
    print(
        f"  {label:<12} {iters:>5}  {conv_str}  "
        f"{rel_err:>10.3f}%  {abs_err:>12.4e}  "
        f"{residual:>12.4e}  {elapsed:>8.2f}s"
    )


def _run_solvers(problem, reference: np.ndarray) -> dict:
    """
    Executes every solver of `SOLVERS` against a single problem instance.

    Stagnation detection is disabled by setting `patience` beyond the iteration
    ceiling. The detector exists to abort an outer loop that has reached its
    inner solver's error floor; here it would truncate the convergence histories
    the benchmark sets out to record.

    Parameters
    ----------
    problem : PoissonLine2D
        Line-decomposed problem instance.
    reference : np.ndarray
        (N, N) reference field against which each solver is scored.

    Returns
    -------
    results : dict
        Mapping from display label to {'result': OuterResult, 'time': float}.
    """
    _print_table_header()

    results: dict = {}
    for label, inner, options in SOLVERS:
        t0 = time.perf_counter()
        r  = outer_solve(
            problem, inner=inner, scheme=SCHEME, inner_options=options,
            tol=TOL, max_iter=MAX_ITER, patience=MAX_ITER + 1,
        )
        elapsed = time.perf_counter() - t0

        _print_row(
            label, r.n_outer, r.converged,
            _max_rel_err(r.u, reference),
            float(np.max(np.abs(r.u - reference))),
            r.residual,
            elapsed,
        )
        results[label] = {"result": r, "time": elapsed}

    return results


# -- Case A: sinusoidal source, analytical solution ----------------------------

def run_case_a(cfg: HETConfig2D) -> dict:
    """
    Executes Case A: 2-D HET Poisson with a sinusoidal source and homogeneous BCs.

    Compares every solver of `SOLVERS` against the analytical solution
    φ̃(x̃,ỹ) = sin(πx̃)·sin(πỹ).

    Parameters
    ----------
    cfg : HETConfig2D
        Physical parameterisation of the discharge channel.

    Returns
    -------
    data : dict
        Solution fields, analytical electric field components and scalar metrics
        for every solver, plus the mesh required by the plotting routines.
    """
    _print_header("Case A — 2-D HET Sinusoidal Source (Analytical Solution)")

    problem = build_het_sinusoidal(cfg, N)
    u_exact = sinusoidal_solution(cfg, N)
    Ex_exact, Ey_exact = sinusoidal_electric_field(cfg, N)
    X, Y = cfg.grid(N)

    print(f"  {cfg.summary()}")
    print(f"  N={N}, κ(A_row)={problem.kappa_row():.4f}, "
          f"scheme={SCHEME}, tol={TOL:.1e}")
    print(f"  max|φ̃_exact| = {np.max(np.abs(u_exact)):.4f}  "
          f"(expected 1.0 at domain centre)")

    data = {"cfg": cfg, "problem": problem, "X": X, "Y": Y,
            "u_exact": u_exact, "Ex_exact": Ex_exact, "Ey_exact": Ey_exact}
    data.update(_run_solvers(problem, u_exact))
    return data


# -- Case B: Boeuf-Garrigues profile, physical BCs -----------------------------

def run_case_b(cfg: HETConfig2D) -> dict:
    """
    Executes Case B: 2-D HET Poisson with the Boeuf-Garrigues charge density.

    No closed-form solution exists, so the solvers are scored against a
    classical solve on a mesh refined by `REFINE_FACTOR`, computed by
    `benchmark.reference_2d.fine_mesh_reference`. The analytical source profile
    is re-evaluated on the refined mesh rather than interpolated, so the
    reference carries only its own O(h_fine²) truncation error.

    Parameters
    ----------
    cfg : HETConfig2D
        Physical parameterisation of the discharge channel.

    Returns
    -------
    data : dict
        Solution fields, the reference field and scalar metrics for every
        solver, plus the mesh required by the plotting routines.
    """
    _print_header("Case B — 2-D HET Boeuf-Garrigues Profile, V_d = 300 V")

    problem = build_het_problem(cfg, N)
    X, Y = cfg.grid(N)

    print(f"  {cfg.summary()}")
    print(f"  N={N}, κ(A_row)={problem.kappa_row():.4f}, "
          f"scheme={SCHEME}, tol={TOL:.1e}")

    print(f"  Computing refined reference (refine_factor={REFINE_FACTOR})...")
    t0    = time.perf_counter()
    u_ref = fine_mesh_reference(
        cfg.poisson_source_at, N,
        bc_x0=cfg.alpha_bc,   # Anode
        bc_x1=0.0,            # Cathode
        bc_y0=0.0,            # Inner wall, grounded
        bc_y1=0.0,            # Outer wall, grounded
        refine_factor=REFINE_FACTOR,
    )
    print(f"  Reference completed in {time.perf_counter() - t0:.1f}s.")

    data = {"cfg": cfg, "problem": problem, "X": X, "Y": Y, "u_ref": u_ref}
    data.update(_run_solvers(problem, u_ref))
    return data


# -- Figure A: Case A solution contours and errors -----------------------------

def plot_case_a(data: dict, save: bool = True) -> None:
    """
    Generates Figure A: contours and errors for the analytical Case A.

    Layout (2 rows × 3 columns):
        Row 1: solution contours — Thomas | HHL | VQLS
        Row 2: absolute error against the analytical solution

    The solution panels share a colour scale to permit direct visual comparison;
    the error panels are scaled individually to reveal the structure of each
    solver's error, which differs by orders of magnitude between them.

    Parameters
    ----------
    data : dict
        Output of `run_case_a`.
    save : bool, default=True
        If True, exports to RESULTS_DIR/figure_a_2d_het_sinusoidal.pdf.
    """
    X, Y    = data["X"], data["Y"]
    u_exact = data["u_exact"]

    solver_keys   = [label for label, _, _ in SOLVERS]
    solver_labels = ["Thomas (classical)", "HHL (quantum)", "VQLS (variational)"]
    solver_cols   = [COLOURS["thomas"], COLOURS["hhl"], COLOURS["vqls"]]

    u_sols = [data[k]["result"].u for k in solver_keys]

    u_all = np.stack([u_exact] + u_sols)
    levels_u = np.linspace(u_all.min(), u_all.max(), 25)

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(
        r"Case A — 2-D HET Poisson: Sinusoidal Source, "
        r"$\tilde{\phi} = \sin(\pi\tilde{x})\sin(\pi\tilde{y})$"
        "\nAnalytical solution available — quantitative error assessment",
        fontsize=12,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.30)

    for col, (u_sol, label, colour) in enumerate(
        zip(u_sols, solver_labels, solver_cols)
    ):
        # -- Row 0: solution contour -------------------------------------------
        ax = fig.add_subplot(gs[0, col])
        cf = ax.contourf(X, Y, u_sol, levels=levels_u, cmap="viridis")
        ax.contour(X, Y, u_sol, levels=levels_u,
                   colors="white", linewidths=0.3, alpha=0.4)
        fig.colorbar(cf, ax=ax, shrink=0.85)

        # Overlay analytical contour lines for direct comparison.
        ax.contour(X, Y, u_exact, levels=8, colors=colour,
                   linewidths=0.8, linestyles="--", alpha=0.7)

        ax.set_xlabel(r"$\tilde{x}$")
        ax.set_ylabel(r"$\tilde{y}$")
        ax.set_title(f"{label}\n"
                     f"Max rel. err. = {_max_rel_err(u_sol, u_exact):.2f}%")
        ax.set_aspect("equal")

        # -- Row 1: absolute error ---------------------------------------------
        ax = fig.add_subplot(gs[1, col])
        cf = ax.contourf(X, Y, np.abs(u_sol - u_exact), levels=20, cmap="hot_r")
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


# -- Figure B: Case B solution contours ----------------------------------------

def plot_case_b(data: dict, save: bool = True) -> None:
    """
    Generates Figure B: contours and errors for the physical Case B.

    Layout (2 rows × 2 columns):
        (a) Thomas solution contour        (b) VQLS solution contour
        (c) Thomas error vs reference      (d) VQLS error vs reference

    Parameters
    ----------
    data : dict
        Output of `run_case_b`.
    save : bool, default=True
        If True, exports to RESULTS_DIR/figure_b_2d_het_physical.pdf.
    """
    cfg   = data["cfg"]
    X, Y  = data["X"], data["Y"]
    u_ref = data["u_ref"]

    r_thomas = data["Thomas-2D"]["result"]
    r_vqls   = data["VQLS-2D"]["result"]

    u_all = np.stack([r_thomas.u, r_vqls.u])
    levels_u = np.linspace(u_all.min(), u_all.max(), 25)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(
        f"Case B — 2-D HET Poisson: Boeuf-Garrigues Profile, "
        f"$V_d = {cfg.V_discharge:.0f}$ V\n"
        f"Reference: classical solve on a mesh refined "
        f"{REFINE_FACTOR}× per direction\n"
        "Physical parameters: Boeuf & Garrigues (1998), J. Appl. Phys. 84, 3541",
        fontsize=11,
    )

    panels = [
        (axes[0, 0], r_thomas.u, levels_u, "viridis",
         f"Thomas-2D solution ({r_thomas.n_outer} iters)"),
        (axes[0, 1], r_vqls.u, levels_u, "viridis",
         f"VQLS-2D solution ({r_vqls.n_outer} iters)"),
        (axes[1, 0], np.abs(r_thomas.u - u_ref), None, "hot_r",
         "Thomas-2D absolute error vs reference"),
        (axes[1, 1], np.abs(r_vqls.u - u_ref), None, "hot_r",
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


# -- Figure C: electric field vector plots -------------------------------------

def plot_electric_field(data_a: dict, save: bool = True) -> None:
    """
    Generates Figure C: electric field vector plots for Case A.

    The field is recovered from the non-dimensional potential by second-order
    centred finite differences and converted to physical units [V/m] via the
    chain rule, ∂/∂x = (1/L_x)·∂/∂x̃, so that E_x carries φ_0/L_x and E_y
    carries φ_0/L_y.

    Layout (1 row × 3 columns): analytical | Thomas | VQLS, each showing field
    magnitude as colour and direction as unit arrows.

    Parameters
    ----------
    data_a : dict
        Output of `run_case_a`.
    save : bool, default=True
        If True, exports to RESULTS_DIR/figure_c_2d_het_efield.pdf.
    """
    cfg  = data_a["cfg"]
    X, Y = data_a["X"], data_a["Y"]
    h    = 1.0 / (N + 1)          # Non-dimensional spacing, identical on both axes

    Ex_exact, Ey_exact = data_a["Ex_exact"], data_a["Ey_exact"]
    E_mag_exact = np.sqrt(Ex_exact**2 + Ey_exact**2)

    def _numerical_efield(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Recovers E_x and E_y from the potential by centred differences.

        The field is padded with the homogeneous Dirichlet data of Case A before
        differencing, so the one-sided nodes adjacent to the boundary remain
        second-order accurate.
        """
        phi = np.zeros((N + 2, N + 2))
        phi[1:N+1, 1:N+1] = u

        dphidx = -(phi[2:N+2, 1:N+1] - phi[0:N,   1:N+1]) / (2.0 * h)
        dphidy = -(phi[1:N+1, 2:N+2] - phi[1:N+1, 0:N  ]) / (2.0 * h)

        return dphidx * cfg.phi_0 / cfg.L_x, dphidy * cfg.phi_0 / cfg.L_y

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        r"Case A — 2-D HET Electric Field: $\mathbf{E} = -\nabla\phi$"
        "\nColour: field magnitude [V/m]; arrows: field direction",
        fontsize=12,
    )

    E_max = float(E_mag_exact.max()) * 1.2

    # Analytical field.
    ax = axes[0]
    cf = ax.contourf(X, Y, E_mag_exact, levels=20, cmap="plasma",
                     vmin=0, vmax=E_max)
    ax.quiver(X, Y, Ex_exact / E_mag_exact, Ey_exact / E_mag_exact,
              alpha=0.6, scale=25, color="white", width=0.004)
    fig.colorbar(cf, ax=ax, label=r"$|\mathbf{E}|$ [V/m]")
    ax.set_xlabel(r"$\tilde{x}$")
    ax.set_ylabel(r"$\tilde{y}$")
    ax.set_title(f"Analytical\nMax |E| = {E_mag_exact.max():.2e} V/m")
    ax.set_aspect("equal")

    # Numerical fields.
    for ax, (key, label) in zip(
        axes[1:],
        zip(["Thomas-2D", "VQLS-2D"], ["Thomas (classical)", "VQLS (variational)"]),
    ):
        Ex, Ey = _numerical_efield(data_a[key]["result"].u)
        E_mag  = np.sqrt(Ex**2 + Ey**2)

        cf = ax.contourf(X, Y, E_mag, levels=20, cmap="plasma",
                         vmin=0, vmax=E_max)
        ax.quiver(X, Y, Ex / E_mag, Ey / E_mag,
                  alpha=0.6, scale=25, color="white", width=0.004)
        fig.colorbar(cf, ax=ax, label=r"$|\mathbf{E}|$ [V/m]")
        ax.set_xlabel(r"$\tilde{x}$")
        ax.set_ylabel(r"$\tilde{y}$")
        ax.set_title(f"{label}\nMax |E| = {E_mag.max():.2e} V/m")
        ax.set_aspect("equal")

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / "figure_c_2d_het_efield.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Figure C saved to {path}")
    plt.show()


# -- CSV export ----------------------------------------------------------------

def export_csv(data_a: dict, data_b: dict) -> None:
    """
    Exports all scalar benchmark metrics to CSV.

    Parameters
    ----------
    data_a : dict
        Output of `run_case_a`.
    data_b : dict
        Output of `run_case_b`.
    """
    filepath = RESULTS_DIR / "het_2d_metrics.csv"
    fieldnames = [
        "case", "N", "solver", "iterations", "converged",
        "max_rel_err_pct", "max_abs_err", "residual", "time_s",
    ]

    rows = []
    for case_name, data, ref_key in (
        ("A_sinusoidal_hom", data_a, "u_exact"),
        ("B_gaussian_phys",  data_b, "u_ref"),
    ):
        ref = data[ref_key]
        for label, _, _ in SOLVERS:
            r = data[label]["result"]
            rows.append({
                "case":            case_name,
                "N":               N,
                "solver":          label,
                "iterations":      r.n_outer,
                "converged":       r.converged,
                "max_rel_err_pct": _max_rel_err(r.u, ref),
                "max_abs_err":     float(np.max(np.abs(r.u - ref))),
                "residual":        r.residual,
                "time_s":          data[label]["time"],
            })

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Metrics exported to {filepath}")


# -- Main entry point ----------------------------------------------------------

def main() -> None:
    """
    Executes the full 2-D HET benchmark and generates all output artefacts.

    Execution sequence:
        1. Case A: sinusoidal source, analytical solution.
        2. Case B: Boeuf-Garrigues profile, physical BCs.
        3. Figure generation (three figures).
        4. CSV export.
    """
    print("\n" + "═"*65)
    print("  2-D HET PLASMA POISSON BENCHMARK")
    print("  Quantum vs Classical Solver Comparison")
    print("  Reference: Boeuf & Garrigues (1998), J. Appl. Phys. 84, 3541")
    print("═"*65)

    cfg = HETConfig2D(V_discharge=300.0)

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
