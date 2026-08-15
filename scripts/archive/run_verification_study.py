"""
Quantum Poisson Solver — Verification and Validation Study.

Purpose
-------
This script provides a structured verification and validation (V&V)
study of the HHL and VQLS quantum linear solvers applied to the 1-D
and 2-D Poisson equation in the context of Hall Effect Thruster (HET)
plasma modelling. It is designed to produce visually interpretable
results suitable for a progress report or supervisor meeting, whilst
remaining computationally tractable on a standard laptop.

The study is structured in three cases of increasing complexity:

    Case 1 — 1-D HET Poisson, linear charge density, homogeneous BCs.
        An analytical solution exists, enabling rigorous quantitative
        error assessment independent of any classical reference solver.
        This case constitutes the primary accuracy validation.

    Case 2 — 1-D HET Poisson, Gaussian charge density, physical BCs.
        No closed-form solution; the Thomas algorithm serves as the
        high-accuracy classical reference. The electric field profile
        is compared qualitatively against Boeuf & Garrigues (1998),
        Fig. 3, which reports a peak field of ~2×10⁴ V/m near the
        exit plane (x̃ ≈ 0.8).

    Case 3 — 2-D Poisson, sinusoidal source, homogeneous BCs.
        The Thomas line-Jacobi solution on a refined mesh serves as
        the reference. Contour plots of the potential and error fields
        are produced for all three solvers.

Computational parameters are deliberately set to low resolution
(N ∈ {4, 8}) to ensure completion within 10–20 minutes on a standard
workstation. The framework is designed for straightforward scaling to
higher resolution on HPC resources.

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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# -- System Path Resolution ----------------------------------------------------

# Dynamically resolve the project root directory (one level up from this script)
# and append it to the system path to enable absolute imports.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from core.config import SimConfig1D, SimConfig2D
from core.het_config import HETConfig
from core.exact_solutions import HET_EXACT_SOLUTIONS
from problems.het_plasma_1d import HETPoissonProblem1D
from problems.poisson_1d import PoissonProblem1D
from problems.poisson_2d import PoissonProblem2D
from solvers.classical.thomas import thomas_solve_system
from solvers.classical.thomas_2d import thomas_solve_2d
from solvers.quantum.hhl_1d import hhl_solve_system
from solvers.quantum.hhl_2d import hhl_solve_2d
from solvers.quantum.vqls_1d import VQLSConfig1D, vqls_solve_system
from solvers.quantum.vqls_2d import VQLSConfig2D, vqls_solve_2d

RESULTS_DIR = Path("results/verification")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -- Matplotlib style ---------------------------------------------------------
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    12,
    "legend.fontsize":   10,
    "figure.dpi":        130,
    "lines.linewidth":   1.8,
    "lines.markersize":  6,
})

COLOURS = {
    "thomas":     "#2ca02c",   # green
    "hhl":        "#1f77b4",   # blue
    "vqls":       "#d62728",   # red
    "analytical": "#000000",   # black
}


# -- Shared solver configurations ---------------------------------------------

# 1-D VQLS: 6 layers, 3 restarts × 300 iterations, suitable for N ∈ {4, 8}.
VQLS1D_CFG = VQLSConfig1D(
    n_layers    = 6,
    optimiser   = "COBYLA",
    max_iter    = 300,
    tol         = 1e-6,
    random_seed = 42,
    verbose     = False,
)

# 2-D VQLS inner solver: 3 layers, 100 iterations per restart.
# Outer loop limited to 20 iterations for tractability at N=4.
_INNER_2D = VQLSConfig1D(
    n_layers    = 3,
    optimiser   = "COBYLA",
    max_iter    = 100,
    tol         = 1e-2,
    random_seed = 0,
    verbose     = False,
)
VQLS2D_CFG = VQLSConfig2D(
    inner_config = _INNER_2D,
    warm_start   = True,
    verbose      = True,
)


# -- Utility functions --------------------------------------------------------

def _rel_err_pct(u: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Compute the pointwise absolute relative error in percent.

    Nodes where |ref| < 1e-4 · max|ref| are masked to NaN to prevent
    division by near-zero values from inflating the error metric.

    Parameters
    ----------
    u : np.ndarray, shape (N,)
        Solver solution vector.
    ref : np.ndarray, shape (N,)
        Reference solution vector.

    Returns
    -------
    err : np.ndarray, shape (N,)
        Pointwise relative error in percent; NaN at masked nodes.
    """
    scale = np.max(np.abs(ref))
    mask  = np.abs(ref) > 1e-4 * scale
    err   = np.where(mask, np.abs(u - ref) / np.abs(ref) * 100.0, np.nan)
    return err


def _max_rel_err_pct(u: np.ndarray, ref: np.ndarray) -> float:
    """Maximum relative error in percent, excluding masked nodes."""
    err   = _rel_err_pct(u, ref)
    valid = err[~np.isnan(err)]
    return float(np.max(valid)) if valid.size > 0 else float("nan")


def _electric_field(
    phi_int : np.ndarray,
    alpha_bc: float,
    phi_0   : float,
    L       : float,
    N       : int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Recover the physical electric field E(x) [V/m] from the
    non-dimensional interior potential via second-order centred
    finite differences, with one-sided differences at the boundaries.

    Parameters
    ----------
    phi_int : np.ndarray, shape (N,)
        Non-dimensional potential at interior nodes.
    alpha_bc : float
        Non-dimensional anode potential φ̃(0) = V_d/φ_0.
    phi_0 : float
        Thermal voltage [V]: φ_0 = T_e [eV].
    L : float
        Channel length [m].
    N : int
        Number of interior nodes.

    Returns
    -------
    x_full : np.ndarray, shape (N+2,)
        Non-dimensional coordinates including boundary nodes.
    E_phys : np.ndarray, shape (N+2,)
        Physical electric field [V/m].
    """
    dx           = 1.0 / (N + 1)
    phi_full     = np.zeros(N + 2)
    phi_full[0]  = alpha_bc
    phi_full[1:N+1] = phi_int
    phi_full[N+1]   = 0.0

    x_full   = np.linspace(0.0, 1.0, N + 2)
    E_nd     = np.zeros(N + 2)
    E_nd[1:-1] = -(phi_full[2:] - phi_full[:-2]) / (2.0 * dx)
    E_nd[0]    = -(phi_full[1]  - phi_full[0])   / dx
    E_nd[-1]   = -(phi_full[-1] - phi_full[-2])  / dx

    return x_full, E_nd * phi_0 / L


def _print_case_header(title: str) -> None:
    print(f"\n{'═'*65}")
    print(f"  {title}")
    print(f"{'═'*65}")


def _print_solver_row(
    label    : str,
    rel_err  : float,
    abs_err  : float,
    residual : float,
    elapsed  : float,
    extra    : str = "",
) -> None:
    print(
        f"  {label:<10} {rel_err:>10.3f}%  {abs_err:>12.4e}  "
        f"{residual:>12.4e}  {elapsed:>8.2f}s  {extra}"
    )


# -- Case 1: 1-D HET, linear profile, homogeneous BCs ------------------------

def run_case_1() -> dict:
    """
    Case 1: 1-D HET Poisson with linear charge density and homogeneous
    Dirichlet boundary conditions.

    The analytical solution φ̃(x̃) = α·ρ_0·x̃·(1 − x̃²)/6 provides an
    exact reference for quantitative error assessment. Both HHL and VQLS
    are evaluated at N=4 (2 qubits) and N=8 (3 qubits).

    Returns
    -------
    dict
        Scalar metrics and solution arrays for all solvers and both
        mesh sizes, used for figure generation and CSV export.
    """
    _print_case_header(
        "Case 1 — 1-D HET Poisson: linear profile, homogeneous BCs"
    )
    print(f"  {'Solver':<10} {'MaxRelErr':>10}  {'MaxAbsErr':>12}  "
          f"{'Residual':>12}  {'Time':>8}")
    print(f"  {'─'*62}")

    results = {}

    for N in (4, 8):
        cfg  = HETConfig(N=N, epsilon=0.01, rho_profile="linear",
                         V_discharge=0.0)
        prob = HETPoissonProblem1D(cfg)
        A, b = prob.A, prob.b

        u_exact = HET_EXACT_SOLUTIONS["linear"](prob.x, cfg.rho_0, cfg.alpha)

        # Thomas.
        t0       = time.perf_counter()
        u_thomas = thomas_solve_system(A, b)
        t_thomas = time.perf_counter() - t0

        # HHL.
        t0 = time.perf_counter()
        u_hhl, _, _ = hhl_solve_system(A, b, cfg.epsilon)
        t_hhl = time.perf_counter() - t0

        # VQLS.
        t0     = time.perf_counter()
        vr     = vqls_solve_system(A, b, VQLS1D_CFG)
        u_vqls = vr.u
        t_vqls = time.perf_counter() - t0

        print(f"\n  N={N}:")
        _print_solver_row(
            "Thomas",
            _max_rel_err_pct(u_thomas, u_exact),
            float(np.max(np.abs(u_thomas - u_exact))),
            float(np.linalg.norm(A @ u_thomas - b) / np.linalg.norm(b)),
            t_thomas,
        )
        _print_solver_row(
            "HHL",
            _max_rel_err_pct(u_hhl, u_exact),
            float(np.max(np.abs(u_hhl - u_exact))),
            float(np.linalg.norm(A @ u_hhl - b) / np.linalg.norm(b)),
            t_hhl,
        )
        _print_solver_row(
            "VQLS",
            _max_rel_err_pct(u_vqls, u_exact),
            float(np.max(np.abs(u_vqls - u_exact))),
            float(np.linalg.norm(A @ u_vqls - b) / np.linalg.norm(b)),
            t_vqls,
            f"cost={vr.final_cost:.2e}",
        )

        results[N] = {
            "x":        prob.x,
            "u_exact":  u_exact,
            "u_thomas": u_thomas,
            "u_hhl":    u_hhl,
            "u_vqls":   u_vqls,
            "cfg":      cfg,
            "vqls_cost": vr.final_cost,
            "t_thomas": t_thomas,
            "t_hhl":    t_hhl,
            "t_vqls":   t_vqls,
        }

    return results


# -- Case 2: 1-D HET, Gaussian profile, physical BCs -------------------------

def run_case_2() -> dict:
    """
    Case 2: 1-D HET Poisson with Gaussian charge density and physical
    Dirichlet boundary conditions (V_d = 300 V).

    No analytical solution is available; the Thomas algorithm serves as
    the exact discrete reference. The electric field profile is compared
    qualitatively against Boeuf & Garrigues (1998), Fig. 3.

    Returns
    -------
    dict
        Scalar metrics, solution arrays, and electric field profiles.
    """
    _print_case_header(
        "Case 2 — 1-D HET Poisson: Gaussian profile, V_d = 300 V"
    )
    print(f"  {'Solver':<10} {'MaxRelErr':>10}  {'MaxAbsErr':>12}  "
          f"{'Residual':>12}  {'Time':>8}")
    print(f"  {'─'*62}")

    cfg  = HETConfig(N=8, epsilon=0.01, rho_profile="gaussian",
                     V_discharge=300.0)
    prob = HETPoissonProblem1D(cfg)
    A, b = prob.A, prob.b

    t0       = time.perf_counter()
    u_thomas = thomas_solve_system(A, b)
    t_thomas = time.perf_counter() - t0

    t0 = time.perf_counter()
    u_hhl, _, _ = hhl_solve_system(A, b, cfg.epsilon)
    t_hhl = time.perf_counter() - t0

    t0     = time.perf_counter()
    vr     = vqls_solve_system(A, b, VQLS1D_CFG)
    u_vqls = vr.u
    t_vqls = time.perf_counter() - t0

    ref = u_thomas
    _print_solver_row(
        "Thomas", 0.0,
        0.0,
        float(np.linalg.norm(A @ u_thomas - b) / np.linalg.norm(b)),
        t_thomas, "(reference)",
    )
    _print_solver_row(
        "HHL",
        _max_rel_err_pct(u_hhl, ref),
        float(np.max(np.abs(u_hhl - ref))),
        float(np.linalg.norm(A @ u_hhl - b) / np.linalg.norm(b)),
        t_hhl,
    )
    _print_solver_row(
        "VQLS",
        _max_rel_err_pct(u_vqls, ref),
        float(np.max(np.abs(u_vqls - ref))),
        float(np.linalg.norm(A @ u_vqls - b) / np.linalg.norm(b)),
        t_vqls,
        f"cost={vr.final_cost:.2e}",
    )

    # Electric field recovery.
    x_full, E_thomas = _electric_field(
        u_thomas, cfg.alpha_bc, cfg.phi_0, cfg.L, cfg.N
    )
    _, E_hhl  = _electric_field(u_hhl,  cfg.alpha_bc, cfg.phi_0, cfg.L, cfg.N)
    _, E_vqls = _electric_field(u_vqls, cfg.alpha_bc, cfg.phi_0, cfg.L, cfg.N)

    E_peak = np.max(np.abs(E_thomas))
    print(f"\n  Peak |E| Thomas: {E_peak:.3e} V/m")
    print(f"  Peak |E| HHL:    {np.max(np.abs(E_hhl)):.3e} V/m")
    print(f"  Peak |E| VQLS:   {np.max(np.abs(E_vqls)):.3e} V/m")
    print(f"  B&G (1998) Fig.3 reference: ~2×10⁴ V/m near x̃ ≈ 0.8")

    return {
        "x_int":    prob.x,
        "x_full":   x_full,
        "u_thomas": u_thomas,
        "u_hhl":    u_hhl,
        "u_vqls":   u_vqls,
        "E_thomas": E_thomas,
        "E_hhl":    E_hhl,
        "E_vqls":   E_vqls,
        "cfg":      cfg,
        "vqls_cost": vr.final_cost,
        "t_thomas": t_thomas,
        "t_hhl":    t_hhl,
        "t_vqls":   t_vqls,
    }


# -- Case 3: 2-D Poisson, sinusoidal source, homogeneous BCs -----------------

def run_case_3() -> dict:
    """
    Case 3: 2-D Poisson equation with sinusoidal source function fS and
    homogeneous Dirichlet boundary conditions.

    The Thomas line-Jacobi solution on a refined mesh (refine_factor=17)
    serves as the reference, consistent with the methodology of
    Ghafourpour & Laizet (2025), Section IV E. Both Thomas and VQLS
    line-Jacobi solvers are evaluated at N=4 with max_iter=30.

    Returns
    -------
    dict
        Solution fields, reference solution, and error metrics for
        both solvers.
    """
    _print_case_header(
        "Case 3 — 2-D Poisson: sinusoidal source, homogeneous BCs, N=4"
    )

    cfg     = SimConfig2D(N=4, epsilon=0.01, source_fn="fS", max_iter=30)
    problem = PoissonProblem2D(cfg)

    print("  Computing refined reference solution (refine_factor=17)...")
    t0    = time.perf_counter()
    u_ref = problem.classical_reference_solve(refine_factor=17)
    t_ref = time.perf_counter() - t0
    print(f"  Reference solve completed in {t_ref:.1f}s.")

    t0       = time.perf_counter()
    r_thomas = thomas_solve_2d(problem)
    t_thomas = time.perf_counter() - t0

    t0      = time.perf_counter()
    r_vqls  = vqls_solve_2d(problem, config=VQLS2D_CFG)
    t_vqls  = time.perf_counter() - t0

    def _2d_max_rel(u):
        scale = np.max(np.abs(u_ref))
        mask  = np.abs(u_ref) > 1e-4 * scale
        if not mask.any():
            return float("nan")
        return float(np.max(
            np.abs((u - u_ref)[mask]) / np.abs(u_ref[mask])
        )) * 100.0

    print(f"\n  {'Solver':<12} {'Iters':>6}  {'Conv':>5}  "
          f"{'MaxRelErr':>10}  {'MaxAbsErr':>12}  {'Time':>8}")
    print(f"  {'─'*62}")
    print(
        f"  {'Thomas-2D':<12} {r_thomas.iterations:>6}  "
        f"{'Yes' if r_thomas.converged else 'No':>5}  "
        f"{_2d_max_rel(r_thomas.u):>10.3f}%  "
        f"{np.max(np.abs(r_thomas.u - u_ref)):>12.4e}  "
        f"{t_thomas:>8.2f}s"
    )
    print(
        f"  {'VQLS-2D':<12} {r_vqls.iterations:>6}  "
        f"{'Yes' if r_vqls.converged else 'No':>5}  "
        f"{_2d_max_rel(r_vqls.u):>10.3f}%  "
        f"{np.max(np.abs(r_vqls.u - u_ref)):>12.4e}  "
        f"{t_vqls:>8.2f}s"
    )

    return {
        "problem":  problem,
        "u_ref":    u_ref,
        "r_thomas": r_thomas,
        "r_vqls":   r_vqls,
        "t_thomas": t_thomas,
        "t_vqls":   t_vqls,
    }


# -- Figure generation --------------------------------------------------------

def plot_case_1(data: dict, save: bool = True) -> None:
    """
    Generate Figure 1: 1-D HET potential and electric field profiles
    with quantitative error analysis for the linear charge density case.

    Layout (2 rows × 3 columns):
        Row 1 (N=4): potential profile | electric field | relative error
        Row 2 (N=8): potential profile | electric field | relative error

    Parameters
    ----------
    data : dict
        Output of run_case_1(), keyed by N ∈ {4, 8}.
    save : bool
        If True, save to RESULTS_DIR/figure_1_1d_het_linear.pdf.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(
        "Case 1 — 1-D HET Poisson: Linear Charge Density, Homogeneous BCs\n"
        r"$d^2\tilde{\phi}/d\tilde{x}^2 = -\alpha\rho_0\tilde{x}$, "
        r"$\tilde{\phi}(0)=\tilde{\phi}(1)=0$",
        fontsize=12,
    )

    for row_idx, N in enumerate((4, 8)):
        d   = data[N]
        cfg = d["cfg"]
        x   = d["x"]

        # Augment with boundary nodes for the potential plot.
        x_full     = np.concatenate([[0.0], x, [1.0]])
        phi_exact  = np.concatenate([[0.0], d["u_exact"],  [0.0]])
        phi_thomas = np.concatenate([[0.0], d["u_thomas"], [0.0]])
        phi_hhl    = np.concatenate([[0.0], d["u_hhl"],    [0.0]])
        phi_vqls   = np.concatenate([[0.0], d["u_vqls"],   [0.0]])

        # Electric field (homogeneous BCs: alpha_bc = 0).
        xE, E_thomas = _electric_field(
            d["u_thomas"], 0.0, cfg.phi_0, cfg.L, N
        )
        _, E_hhl  = _electric_field(d["u_hhl"],  0.0, cfg.phi_0, cfg.L, N)
        _, E_vqls = _electric_field(d["u_vqls"], 0.0, cfg.phi_0, cfg.L, N)
        _, E_exact = _electric_field(d["u_exact"], 0.0, cfg.phi_0, cfg.L, N)

        # -- Panel 1: potential -----------------------------------------------
        ax = axes[row_idx, 0]
        ax.plot(x_full, phi_exact,  color=COLOURS["analytical"],
                lw=2.2, label="Analytical", zorder=5)
        ax.plot(x_full, phi_thomas, color=COLOURS["thomas"],
                ls="--", marker="o", ms=5, label="Thomas")
        ax.plot(x_full, phi_hhl,    color=COLOURS["hhl"],
                ls="-.", marker="s", ms=5, label="HHL")
        ax.plot(x_full, phi_vqls,   color=COLOURS["vqls"],
                ls=":",  marker="^", ms=5, label="VQLS")
        ax.set_xlabel(r"$\tilde{x}$")
        ax.set_ylabel(r"$\tilde{\phi}$")
        ax.set_title(f"Potential profile (N={N})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # -- Panel 2: electric field ------------------------------------------
        ax = axes[row_idx, 1]
        ax.plot(xE, E_exact  / 1e3, color=COLOURS["analytical"],
                lw=2.2, label="Analytical")
        ax.plot(xE, E_thomas / 1e3, color=COLOURS["thomas"],
                ls="--", marker="o", ms=5, label="Thomas")
        ax.plot(xE, E_hhl    / 1e3, color=COLOURS["hhl"],
                ls="-.", marker="s", ms=5, label="HHL")
        ax.plot(xE, E_vqls   / 1e3, color=COLOURS["vqls"],
                ls=":",  marker="^", ms=5, label="VQLS")
        ax.set_xlabel(r"$\tilde{x}$")
        ax.set_ylabel(r"$E$ [kV/m]")
        ax.set_title(f"Electric field (N={N})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # -- Panel 3: relative error ------------------------------------------
        ax = axes[row_idx, 2]
        err_hhl  = _rel_err_pct(d["u_hhl"],  d["u_exact"])
        err_vqls = _rel_err_pct(d["u_vqls"], d["u_exact"])
        err_thomas = _rel_err_pct(d["u_thomas"], d["u_exact"])

        ax.semilogy(x, err_thomas, color=COLOURS["thomas"],
                    marker="o", ms=5, ls="--", label="Thomas")
        ax.semilogy(x, err_hhl,   color=COLOURS["hhl"],
                    marker="s", ms=5, ls="-.", label="HHL")
        ax.semilogy(x, err_vqls,  color=COLOURS["vqls"],
                    marker="^", ms=5, ls=":",  label="VQLS")
        ax.set_xlabel(r"$\tilde{x}$")
        ax.set_ylabel("Relative error (%)")
        ax.set_title(f"Error vs analytical (N={N})")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / "figure_1_1d_het_linear.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"\n  Figure 1 saved to {path}")
    plt.show()


def plot_case_2(data: dict, save: bool = True) -> None:
    """
    Generate Figure 2: 1-D HET potential and electric field profiles
    for the physical operating condition (V_d = 300 V, Gaussian source).

    Layout (1 row × 3 columns):
        Left:   non-dimensional potential φ̃(x̃)
        Centre: physical electric field E(x) [V/m]
        Right:  relative error vs Thomas reference

    The electric field panel includes a horizontal reference line at
    2×10⁴ V/m corresponding to the peak field reported in
    Boeuf & Garrigues (1998), Fig. 3.

    Parameters
    ----------
    data : dict
        Output of run_case_2().
    save : bool
        If True, save to RESULTS_DIR/figure_2_1d_het_physical.pdf.
    """
    cfg = data["cfg"]
    N   = cfg.N

    x_int  = data["x_int"]
    x_full = data["x_full"]

    phi_thomas_full = np.concatenate([[cfg.alpha_bc], data["u_thomas"], [0.0]])
    phi_hhl_full    = np.concatenate([[cfg.alpha_bc], data["u_hhl"],    [0.0]])
    phi_vqls_full   = np.concatenate([[cfg.alpha_bc], data["u_vqls"],   [0.0]])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Case 2 — 1-D HET Poisson: Gaussian Charge Density, $V_d = 300$ V\n"
        "Reference: Boeuf & Garrigues (1998), J. Appl. Phys. 84, 3541",
        fontsize=12,
    )

    # -- Panel 1: potential ---------------------------------------------------
    ax = axes[0]
    ax.plot(x_full, phi_thomas_full, color=COLOURS["thomas"],
            lw=2.2, label="Thomas (reference)")
    ax.plot(x_full, phi_hhl_full,    color=COLOURS["hhl"],
            ls="--", marker="s", ms=5, label="HHL")
    ax.plot(x_full, phi_vqls_full,   color=COLOURS["vqls"],
            ls=":",  marker="^", ms=5, label="VQLS")
    ax.scatter(x_int, data["u_thomas"], color=COLOURS["thomas"], s=25, zorder=5)
    ax.scatter(x_int, data["u_hhl"],    color=COLOURS["hhl"],    s=25, zorder=5)
    ax.scatter(x_int, data["u_vqls"],   color=COLOURS["vqls"],   s=25, zorder=5)
    ax.set_xlabel(r"$\tilde{x} = x/L$")
    ax.set_ylabel(r"$\tilde{\phi} = \phi/\phi_0$")
    ax.set_title(
        f"Non-dimensional potential\n"
        r"($\phi_0 = $" + f"{cfg.phi_0:.0f} V, "
        r"$\alpha_{bc} = $" + f"{cfg.alpha_bc:.1f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -- Panel 2: electric field ----------------------------------------------
    ax = axes[1]
    ax.plot(x_full, data["E_thomas"] / 1e4, color=COLOURS["thomas"],
            lw=2.2, label="Thomas (reference)")
    ax.plot(x_full, data["E_hhl"]    / 1e4, color=COLOURS["hhl"],
            ls="--", marker="s", ms=5, label="HHL")
    ax.plot(x_full, data["E_vqls"]   / 1e4, color=COLOURS["vqls"],
            ls=":",  marker="^", ms=5, label="VQLS")
    ax.axhline(2.0, color="grey", ls="-.", lw=1.2, alpha=0.7,
               label=r"B&G (1998): $\sim 2\times10^4$ V/m")
    ax.axvline(0.8, color="grey", ls=":",  lw=1.0, alpha=0.5,
               label=r"Exit plane $\tilde{x} \approx 0.8$")
    ax.set_xlabel(r"$\tilde{x} = x/L$")
    ax.set_ylabel(r"$E$ [$\times 10^4$ V/m]")
    ax.set_title(
        "Physical electric field\n"
        "cf. Boeuf & Garrigues (1998), Fig. 3"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # -- Panel 3: relative error vs Thomas ------------------------------------
    ax = axes[2]
    err_hhl  = _rel_err_pct(data["u_hhl"],  data["u_thomas"])
    err_vqls = _rel_err_pct(data["u_vqls"], data["u_thomas"])
    ax.semilogy(x_int, err_hhl,  color=COLOURS["hhl"],
                marker="s", ms=5, ls="-.", label="HHL")
    ax.semilogy(x_int, err_vqls, color=COLOURS["vqls"],
                marker="^", ms=5, ls=":",  label="VQLS")
    ax.set_xlabel(r"$\tilde{x} = x/L$")
    ax.set_ylabel("Relative error vs Thomas (%)")
    ax.set_title(f"Error profile (N={N})")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / "figure_2_1d_het_physical.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Figure 2 saved to {path}")
    plt.show()


def plot_case_3(data: dict, save: bool = True) -> None:
    """
    Generate Figure 3: 2-D Poisson solution contour plots.

    Layout (2 rows × 2 columns), matching the paper's Fig. 10 layout:
        (a) Thomas-2D solution contour
        (b) VQLS-2D solution contour
        (c) Thomas-2D error vs refined reference
        (d) VQLS-2D error vs refined reference

    Parameters
    ----------
    data : dict
        Output of run_case_3().
    save : bool
        If True, save to RESULTS_DIR/figure_3_2d_poisson.pdf.
    """
    problem  = data["problem"]
    u_ref    = data["u_ref"]
    r_thomas = data["r_thomas"]
    r_vqls   = data["r_vqls"]
    X, Y     = problem.X, problem.Y

    u_all    = np.concatenate([r_thomas.u.ravel(), r_vqls.u.ravel()])
    u_min, u_max = u_all.min(), u_all.max()
    levels_u = np.linspace(u_min, u_max, 20)

    err_thomas = np.abs(r_thomas.u - u_ref)
    err_vqls   = np.abs(r_vqls.u   - u_ref)
    err_max    = max(err_thomas.max(), err_vqls.max())
    levels_e   = np.linspace(0.0, err_max if err_max > 0 else 1.0, 20)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(
        "Case 3 — 2-D Poisson: Sinusoidal Source, Homogeneous BCs, N=4\n"
        "Reference: Thomas line-Jacobi on refined mesh (refine_factor=17)",
        fontsize=12,
    )

    panels = [
        (axes[0, 0], r_thomas.u, levels_u, "viridis",
         f"Thomas-2D solution ({r_thomas.iterations} iters)"),
        (axes[0, 1], r_vqls.u,   levels_u, "viridis",
         f"VQLS-2D solution ({r_vqls.iterations} iters)"),
        (axes[1, 0], err_thomas, levels_e, "hot_r",
         "Thomas-2D absolute error"),
        (axes[1, 1], err_vqls,   levels_e, "hot_r",
         "VQLS-2D absolute error"),
    ]

    for ax, Z, lvls, cmap, title in panels:
        cf = ax.contourf(X, Y, Z, levels=lvls, cmap=cmap)
        ax.contour(X, Y, Z, levels=lvls, colors="white",
                   linewidths=0.3, alpha=0.4)
        fig.colorbar(cf, ax=ax, shrink=0.85)
        ax.set_xlabel(r"$\tilde{x}$")
        ax.set_ylabel(r"$\tilde{y}$")
        ax.set_title(title)
        ax.set_aspect("equal")

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / "figure_3_2d_poisson.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Figure 3 saved to {path}")
    plt.show()


def plot_summary_table(
    c1_data : dict,
    c2_data : dict,
    c3_data : dict,
    save    : bool = True,
) -> None:
    """
    Generate Figure 4: algorithm comparison summary table rendered as
    a matplotlib figure, suitable for inclusion in a progress report.

    The table reports, for each case and solver: maximum relative error,
    computation time, and VQLS cost (where applicable).

    Parameters
    ----------
    c1_data : dict
        Output of run_case_1() (keyed by N).
    c2_data : dict
        Output of run_case_2().
    c3_data : dict
        Output of run_case_3().
    save : bool
        If True, save to RESULTS_DIR/figure_4_summary_table.pdf.
    """
    rows = []

    # Case 1, N=4.
    for N in (4, 8):
        d   = c1_data[N]
        ref = d["u_exact"]
        rows += [
            ["1 (linear, hom.)", f"N={N}", "Thomas",
             f"{_max_rel_err_pct(d['u_thomas'], ref):.3f}",
             f"{d['t_thomas']*1e3:.1f} ms", "—"],
            ["", "", "HHL",
             f"{_max_rel_err_pct(d['u_hhl'], ref):.3f}",
             f"{d['t_hhl']:.1f} s", "—"],
            ["", "", "VQLS",
             f"{_max_rel_err_pct(d['u_vqls'], ref):.3f}",
             f"{d['t_vqls']:.1f} s",
             f"{d['vqls_cost']:.2e}"],
        ]

    # Case 2, N=8.
    ref2 = c2_data["u_thomas"]
    rows += [
        ["2 (Gaussian, phys.)", "N=8", "Thomas",
         "—", f"{c2_data['t_thomas']*1e3:.1f} ms", "—"],
        ["", "", "HHL",
         f"{_max_rel_err_pct(c2_data['u_hhl'], ref2):.3f}",
         f"{c2_data['t_hhl']:.1f} s", "—"],
        ["", "", "VQLS",
         f"{_max_rel_err_pct(c2_data['u_vqls'], ref2):.3f}",
         f"{c2_data['t_vqls']:.1f} s",
         f"{c2_data['vqls_cost']:.2e}"],
    ]

    # Case 3, N=4.
    u_ref3 = c3_data["u_ref"]
    def _2d_rel(u):
        return f"{_max_rel_err_pct(u.ravel(), u_ref3.ravel()):.3f}"
    rows += [
        ["3 (2-D sinusoidal)", "N=4", "Thomas-2D",
         _2d_rel(c3_data["r_thomas"].u),
         f"{c3_data['t_thomas']:.1f} s", "—"],
        ["", "", "VQLS-2D",
         _2d_rel(c3_data["r_vqls"].u),
         f"{c3_data['t_vqls']:.1f} s", "—"],
    ]

    col_labels = ["Case", "Resolution", "Solver",
                  "Max Rel. Err. (%)", "Wall Time", "VQLS Cost"]

    fig, ax = plt.subplots(figsize=(13, 0.45 * len(rows) + 1.5))
    ax.axis("off")
    tbl = ax.table(
        cellText   = rows,
        colLabels  = col_labels,
        cellLoc    = "centre",
        loc        = "centre",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.55)

    # Style the header row.
    for col in range(len(col_labels)):
        tbl[0, col].set_facecolor("#2c3e50")
        tbl[0, col].set_text_props(color="white", fontweight="bold")

    # Alternate row shading.
    for row_idx in range(1, len(rows) + 1):
        colour = "#f0f4f8" if row_idx % 2 == 0 else "#ffffff"
        for col in range(len(col_labels)):
            tbl[row_idx, col].set_facecolor(colour)

    ax.set_title(
        "Quantum Poisson Solver — Algorithm Comparison Summary\n"
        "HHL vs VQLS vs Thomas (classical reference)",
        fontsize=12, fontweight="bold", pad=12,
    )

    plt.tight_layout()
    if save:
        path = RESULTS_DIR / "figure_4_summary_table.pdf"
        plt.savefig(path, bbox_inches="tight")
        print(f"  Figure 4 saved to {path}")
    plt.show()


# -- CSV export ---------------------------------------------------------------

def export_csv(
    c1_data : dict,
    c2_data : dict,
    c3_data : dict,
) -> None:
    """
    Export all scalar benchmark metrics to a single CSV file for
    subsequent analysis or thesis table generation.

    Parameters
    ----------
    c1_data, c2_data, c3_data : dict
        Outputs of run_case_1(), run_case_2(), run_case_3().
    """
    filepath = RESULTS_DIR / "verification_metrics.csv"
    fieldnames = [
        "case", "N", "solver", "max_rel_err_pct",
        "max_abs_err", "residual", "time_s", "vqls_cost",
    ]

    rows = []

    for N in (4, 8):
        d   = c1_data[N]
        ref = d["u_exact"]
        A   = HETPoissonProblem1D(
            HETConfig(N=N, epsilon=0.01, rho_profile="linear",
                      V_discharge=0.0)
        ).A
        b   = HETPoissonProblem1D(
            HETConfig(N=N, epsilon=0.01, rho_profile="linear",
                      V_discharge=0.0)
        ).b
        for label, u_sol, t_sol, cost in [
            ("Thomas", d["u_thomas"], d["t_thomas"], ""),
            ("HHL",    d["u_hhl"],    d["t_hhl"],    ""),
            ("VQLS",   d["u_vqls"],   d["t_vqls"],   d["vqls_cost"]),
        ]:
            rows.append({
                "case":            "1_linear_hom",
                "N":               N,
                "solver":          label,
                "max_rel_err_pct": _max_rel_err_pct(u_sol, ref),
                "max_abs_err":     float(np.max(np.abs(u_sol - ref))),
                "residual":        float(
                    np.linalg.norm(A @ u_sol - b) / np.linalg.norm(b)
                ),
                "time_s":          t_sol,
                "vqls_cost":       cost,
            })

    ref2 = c2_data["u_thomas"]
    cfg2 = c2_data["cfg"]
    prob2 = HETPoissonProblem1D(cfg2)
    for label, u_sol, t_sol, cost in [
        ("Thomas", c2_data["u_thomas"], c2_data["t_thomas"], ""),
        ("HHL",    c2_data["u_hhl"],    c2_data["t_hhl"],    ""),
        ("VQLS",   c2_data["u_vqls"],   c2_data["t_vqls"],   c2_data["vqls_cost"]),
    ]:
        rows.append({
            "case":            "2_gaussian_phys",
            "N":               cfg2.N,
            "solver":          label,
            "max_rel_err_pct": _max_rel_err_pct(u_sol, ref2) if label != "Thomas" else 0.0,
            "max_abs_err":     float(np.max(np.abs(u_sol - ref2))),
            "residual":        float(
                np.linalg.norm(prob2.A @ u_sol - prob2.b) / np.linalg.norm(prob2.b)
            ),
            "time_s":          t_sol,
            "vqls_cost":       cost,
        })

    u_ref3 = c3_data["u_ref"]
    for label, r_sol, t_sol in [
        ("Thomas-2D", c3_data["r_thomas"], c3_data["t_thomas"]),
        ("VQLS-2D",   c3_data["r_vqls"],   c3_data["t_vqls"]),
    ]:
        rows.append({
            "case":            "3_2d_sinusoidal",
            "N":               4,
            "solver":          label,
            "max_rel_err_pct": _max_rel_err_pct(
                r_sol.u.ravel(), u_ref3.ravel()
            ),
            "max_abs_err":     float(np.max(np.abs(r_sol.u - u_ref3))),
            "residual":        r_sol.euclidean_residual,
            "time_s":          t_sol,
            "vqls_cost":       "",
        })

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n  Metrics exported to {filepath}")


# -- Main entry point ---------------------------------------------------------

def main() -> None:
    """
    Execute the full verification study and generate all output artefacts.

    Execution sequence:
        1. Case 1: 1-D HET linear profile (N=4 and N=8)
        2. Case 2: 1-D HET Gaussian profile, physical BCs (N=8)
        3. Case 3: 2-D Poisson sinusoidal source (N=4)
        4. Figure generation (4 figures)
        5. CSV export

    Estimated total runtime: 15–30 minutes on a standard workstation.
    """
    print("\n" + "═"*65)
    print("  QUANTUM POISSON SOLVER — VERIFICATION AND VALIDATION STUDY")
    print("  HHL and VQLS applied to HET plasma modelling")
    print("  Reference: Boeuf & Garrigues (1998); Ghafourpour & Laizet (2025)")
    print("═"*65)

    t_total = time.perf_counter()

    c1_data = run_case_1()
    c2_data = run_case_2()
    c3_data = run_case_3()

    print(f"\n{'─'*65}")
    print(f"  All cases completed in "
          f"{time.perf_counter() - t_total:.1f}s. Generating figures...")

    plot_case_1(c1_data, save=True)
    plot_case_2(c2_data, save=True)
    plot_case_3(c3_data, save=True)
    plot_summary_table(c1_data, c2_data, c3_data, save=True)
    export_csv(c1_data, c2_data, c3_data)

    print(f"\n  All outputs saved to {RESULTS_DIR.resolve()}")
    print("═"*65)


if __name__ == "__main__":
    main()