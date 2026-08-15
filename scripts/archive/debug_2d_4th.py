#!/usr/bin/env python3
"""
scripts/debug_2d_4th.py
-----------------------
Debug and validation script for the mixed-order 2D Poisson solver using
fourth-order spatial discretisation in the implicit (strip) direction.

The 2D Poisson equation  ∂²u/∂x² + ∂²u/∂y² = f(x,y)  is solved via
line-Jacobi iteration.  Each x-strip is discretised with the fourth-order
pentadiagonal stencil [-1, 16, -30, 16, -1]/(12h²), while the explicit
y-coupling uses the standard second-order stencil [1, -2, 1]/h².

This is a mixed-order scheme: the strip solves are fourth-order accurate
in x, but the global 2D accuracy is limited by the second-order y-coupling.
The approach is the natural extension of the existing ``solvers/outer``
architecture and requires no changes to the outer iteration machinery.

Strip matrix convention
-----------------------
The strip matrix absorbs the y-coupling diagonal shift, following the
same convention as ``PoissonLine2D._build_row_matrix()`` in the
second-order case.  In the integer-coefficient (12h²-scaled) form::

    diag(A_strip)  =  -30 - 24  =  -54   (interior rows)
                   =  -29 - 24  =  -53   (boundary rows, ghost-point +1)
    off-diag ±1    =  +16
    off-diag ±2    =  -1

The y-coupling on the RHS is scaled by 12 (from 12h² × 1/h²)::

    b[:,j]  =  12h² f[:,j]  -  12 u[:,j-1]^{old}  -  12 u[:,j+1]^{old}

At convergence this recovers the correct mixed-order 2D equation::

    A_pent @ u/(12h²)  +  (u_{j-1} - 2u_j + u_{j+1})/h²  =  f

Usage
-----
    python scripts/debug_2d_4th.py --N 4
    python scripts/debug_2d_4th.py --N 4 --inner vqls
    python scripts/debug_2d_4th.py --N 4 --inner qsvt --max-degree 200
    python scripts/debug_2d_4th.py --N 4 --inner all --plot
    python scripts/debug_2d_4th.py --compare-orders --N-values 4 8
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# -- Colour codes --------------------------------------------------------------
G = "\033[92m"
Y = "\033[93m"
R = "\033[91m"
C = "\033[96m"
B = "\033[1m"
X = "\033[0m"


# -- Error metrics -------------------------------------------------------------

def _max_rel_err(u: np.ndarray, ref: np.ndarray) -> float:
    mask = np.abs(ref) > 1e-10
    if not np.any(mask):
        return float(np.max(np.abs(u - ref))) * 100.0
    return float(np.max(np.abs((u[mask] - ref[mask]) / ref[mask]))) * 100.0


def _residual_2d(A_strip: np.ndarray, phi: np.ndarray,
                 f_vals: np.ndarray, dx: float) -> float:
    """Global residual: max over all strips of ||A_strip @ phi[:,j] - b_j||."""
    N = phi.shape[0]
    max_res = 0.0
    for j in range(N):
        b_j = _build_rhs_strip(j, phi, f_vals, dx)
        res = np.linalg.norm(A_strip @ phi[:, j] - b_j)
        max_res = max(max_res, res)
    return max_res


def _colour(err: float) -> str:
    return G if err < 5.0 else (Y if err < 20.0 else R)


# -- Problem setup -------------------------------------------------------------

def build_grid_2d(N: int):
    """Interior grid for the unit square."""
    dx = 1.0 / (N + 1)
    pts = np.arange(1, N + 1) * dx
    x, y = np.meshgrid(pts, pts, indexing="ij")
    return x, y, dx


def f_sin_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Source: -2π²sin(πx)sin(πy) — manufactured solution u=sin(πx)sin(πy)."""
    return -2.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def u_exact_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Exact solution for f_sin_2d with homogeneous BCs."""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def build_strip_matrix_4th(N: int, dx: float) -> np.ndarray:
    """
    Build the N×N fourth-order pure x-direction pentadiagonal matrix.

    Returns the 1D pentadiagonal matrix from ``PoissonProblem1D4th`` with
    integer coefficients [-1, 16, -30, 16, -1] and ghost-point boundary
    corrections on A[0,0] and A[-1,-1].  The ``dx`` parameter is accepted
    for interface consistency but unused (``PoissonProblem1D4th`` computes
    its own spacing internally).

    In the 2D/3D context, the caller must add the transverse diagonal
    shift to obtain the full strip operator — see ``jacobi_2d_4th``.
    """
    from problems.poisson_1d_4th import PoissonProblem1D4th
    # Evaluate with fS as a dummy source — only the matrix is required, not the RHS.
    prob = PoissonProblem1D4th(N=N, source_fn="fS")
    return prob.A.copy()


def build_strip_matrix_2nd(N: int, dx: float) -> np.ndarray:
    """
    Build the N×N second-order 2D strip matrix (with y-diagonal absorbed).

    In the h²-scaled convention for the unit square (dx = dy = h), the
    strip matrix diagonal is -4 (not -2), because it includes the
    y-coupling diagonal shift -2/dy² × h² = -2, matching the convention
    used by ``PoissonLine2D._build_row_matrix()``.
    """
    A = -4.0 * np.eye(N)
    if N > 1:
        np.fill_diagonal(A[1:, :], 1.0)
        np.fill_diagonal(A[:, 1:], 1.0)
    return A


def _build_rhs_strip(
    j: int,
    phi: np.ndarray,
    f_vals: np.ndarray,
    dx: float,
    dy: float | None = None,
    bc_x0: float | np.ndarray = 0.0,
    bc_x1: float | np.ndarray = 0.0,
    bc_y0: float | np.ndarray = 0.0,
    bc_y1: float | np.ndarray = 0.0,
) -> np.ndarray:
    if dy is None:
        dy = dx
    kappa_aniso = (dx / dy)**2
    N = phi.shape[0]
    b = 12.0 * dx**2 * f_vals[:, j].copy()

    # X-boundary corrections (4th order implicit direction)
    ax0 = bc_x0[j] if isinstance(bc_x0, np.ndarray) else bc_x0
    ax1 = bc_x1[j] if isinstance(bc_x1, np.ndarray) else bc_x1
    
    b[0] -= 18.0 * ax0
    if N > 1:
        b[1] += ax0
    b[-1] -= 18.0 * ax1
    if N > 1:
        b[-2] += ax1

    # Y-boundary corrections (2nd order explicit direction, scaled by 12)
    if j > 0:
        b -= 12.0 * kappa_aniso * phi[:, j - 1]
    else:
        ay0 = bc_y0 if isinstance(bc_y0, np.ndarray) else np.full(N, bc_y0)
        b -= 12.0 * kappa_aniso * ay0

    if j < N - 1:
        b -= 12.0 * kappa_aniso * phi[:, j + 1]
    else:
        ay1 = bc_y1 if isinstance(bc_y1, np.ndarray) else np.full(N, bc_y1)
        b -= 12.0 * kappa_aniso * ay1

    return b


# -- Classical Thomas solver for strips ---------------------------------------

def thomas_strip(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve the strip system via NumPy direct solver."""
    return np.linalg.solve(A, b)


# -- Outer iteration -----------------------------------------------------------

def jacobi_2d_4th(
    N: int,
    f_vals: np.ndarray,
    dx: float,
    dy: float | None = None,
    bc_x0: float | np.ndarray = 0.0,
    bc_x1: float | np.ndarray = 0.0,
    bc_y0: float | np.ndarray = 0.0,
    bc_y1: float | np.ndarray = 0.0,
    inner: str = "thomas",
    inner_kwargs: dict | None = None,
    u_exact: np.ndarray | None = None,
    u_thomas: np.ndarray | None = None,
    tol: float = 1e-6,
    max_iter: int = 200,
    print_every: int = 1,
) -> tuple[np.ndarray, int, bool, list[float]]:
    """
    2D mixed-order line-Jacobi outer iteration.

    Uses the fourth-order pentadiagonal strip matrix (with y-diagonal
    absorbed) for the implicit direction and 12-scaled second-order
    coupling in the explicit y-direction.
    """
    if inner_kwargs is None:
        inner_kwargs = {}
    if dy is None:
        dy = dx
    kappa_aniso = (dx / dy)**2

    A_strip = build_strip_matrix_4th(N, dx)
    A_strip -= 24.0 * kappa_aniso * np.eye(N)
    phi = np.zeros((N, N))
    errors = []

    # Pre-build solver function
    solve_strip = _make_strip_solver(inner, A_strip, inner_kwargs)

    print(f"\n  {C}{'─'*60}{X}")
    print(f"  {B}{inner.upper()}-2D-4th{X}  N={N}  tol={tol:.0e}  "
          f"max_iter={max_iter}")
    kappa = float(np.abs(np.linalg.eigvalsh(A_strip)).max() /
                  np.abs(np.linalg.eigvalsh(A_strip)).min())
    print(f"  Strip matrix: pentadiagonal  kappa={kappa:.3f}")
    print(f"  {'─'*60}")

    for it in range(max_iter):
        phi_new = phi.copy()
        for j in range(N):
            b_j = _build_rhs_strip(j, phi, f_vals, dx, dy, bc_x0, bc_x1, bc_y0, bc_y1)
            phi_new[:, j] = solve_strip(b_j)

        delta = float(np.max(np.abs(phi_new - phi)))
        errors.append(delta)
        phi = phi_new

        if (it + 1) % print_every == 0:
            err_e = _max_rel_err(phi, u_exact)
            err_t = _max_rel_err(phi, u_thomas)
            col = _colour(err_e)
            print(f"  iter {it+1:4d}  delta={delta:.3e}  "
                  f"vs_exact={col}{err_e:7.3f}%{X}  "
                  f"vs_thomas={err_t:7.3f}%")

        if delta < tol:
            print(f"  {G}Converged at iter {it+1}  (delta={delta:.3e}){X}")
            return phi, it + 1, True, errors

    print(f"  {R}Max iterations ({max_iter}) reached. "
          f"Final delta={errors[-1]:.3e}{X}")
    return phi, max_iter, False, errors


def _make_strip_solver(inner: str, A_strip: np.ndarray, kwargs: dict):
    """Return a callable (b) -> u for the given inner solver."""
    if inner == "thomas":
        return lambda b: thomas_strip(A_strip, b)

    elif inner == "vqls":
        from solvers.quantum.vqls_1d import vqls_solve_system, VQLSConfig1D
        n_layers = kwargs.get("n_layers", 3)
        cfg = VQLSConfig1D(n_layers=n_layers)
        def vqls_strip(b):
            result = vqls_solve_system(A_strip, b, config=cfg)
            return np.array(result.u)
        return vqls_strip

    elif inner == "qsvt":
        from solvers.quantum.qsvt_1d import qsvt_solve_system, QSVTConfig1D
        cfg = QSVTConfig1D(
            epsilon=kwargs.get("epsilon", 0.05),
            max_degree=kwargs.get("max_degree", None),
            angle_method=kwargs.get("angle_method", "auto"),
        )
        def qsvt_strip(b):
            result = qsvt_solve_system(A_strip, b, config=cfg)
            return np.array(result.u)
        return qsvt_strip

    elif inner == "hhl":
        from solvers.quantum.hhl_1d_4th import hhl_solve_4th
        class _ProbWrapper:
            pass
        prob = _ProbWrapper()
        prob.A = A_strip
        prob.N = A_strip.shape[0]
        eps = kwargs.get("epsilon", 0.05)
        t_steps = kwargs.get("trotter_steps", None)

        def hhl_strip(b):
            prob.b = b
            result = hhl_solve_4th(prob, epsilon=eps, trotter_steps=t_steps)
            return np.array(result.u)
        return hhl_strip

    else:
        raise ValueError(f"Unknown inner solver: '{inner}'")


# -- Comparison with second-order 2D ------------------------------------------

def run_thomas_2d_2nd(
    N: int,
    f_vals: np.ndarray,
    dx: float,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> np.ndarray:
    """Run the second-order 2D Thomas solver for comparison."""
    A_2nd = build_strip_matrix_2nd(N, dx)
    phi = np.zeros((N, N))

    for _ in range(max_iter):
        phi_new = phi.copy()
        for j in range(N):
            b_j = dx**2 * f_vals[:, j].copy()
            if j > 0:
                b_j -= phi[:, j - 1]
            if j < N - 1:
                b_j -= phi[:, j + 1]
            phi_new[:, j] = np.linalg.solve(A_2nd, b_j)
        delta = float(np.max(np.abs(phi_new - phi)))
        phi = phi_new
        if delta < tol:
            break

    return phi


# -- Plotting ------------------------------------------------------------------

def plot_solutions(
    x: np.ndarray,
    y: np.ndarray,
    u_exact: np.ndarray,
    solutions: dict[str, np.ndarray],
    N: int,
    out_dir: Path,
) -> None:
    """Plot solution fields and error maps."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print(f"  {Y}matplotlib not available — skipping plot.{X}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    palette = {"Thomas-4th": "black", "Thomas-2nd": "grey",
               "VQLS-4th": "royalblue", "QSVT-4th": "crimson"}

    n_cols = 1 + len(solutions)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7),
                             squeeze=False)

    vmin, vmax = float(u_exact.min()), float(u_exact.max())

    # Exact
    im = axes[0, 0].pcolormesh(x, y, u_exact, cmap="RdBu_r",
                                vmin=vmin, vmax=vmax, shading="auto")
    axes[0, 0].set_title("Exact", fontweight="bold")
    axes[0, 0].set_aspect("equal")
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
    axes[1, 0].axis("off")

    for ci, (label, phi) in enumerate(solutions.items(), start=1):
        col = palette.get(label, "purple")
        im = axes[0, ci].pcolormesh(x, y, phi, cmap="RdBu_r",
                                     vmin=vmin, vmax=vmax, shading="auto")
        axes[0, ci].set_title(label, fontweight="bold")
        axes[0, ci].set_aspect("equal")
        plt.colorbar(im, ax=axes[0, ci], shrink=0.8)

        err = phi - u_exact
        abs_max = max(float(np.abs(err).max()), 1e-12)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        im2 = axes[1, ci].pcolormesh(x, y, err, cmap="seismic",
                                      norm=norm, shading="auto")
        rel = _max_rel_err(phi, u_exact)
        axes[1, ci].set_title(f"Error ({rel:.2f}%)", fontsize=9)
        axes[1, ci].set_aspect("equal")
        plt.colorbar(im2, ax=axes[1, ci], shrink=0.8)

    fig.suptitle(f"Fourth-Order 2D Poisson — N={N}", fontweight="bold")
    plt.tight_layout()
    out_path = out_dir / f"4th_order_2d_N{N}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  {G}Plot saved to: {out_path}{X}")


# -- Header and table helpers --------------------------------------------------

def _header(title: str) -> None:
    print(f"\n{B}{C}{'═'*64}{X}")
    print(f"{B}{C}  {title}{X}")
    print(f"{B}{C}{'═'*64}{X}")


def _table_header() -> None:
    print(f"\n  {'Solver':<14}  {'vs_exact':>10}  {'vs_thomas4':>11}  "
          f"{'iters':>6}  {'conv':>5}  {'time':>8}")
    print(f"  {'─'*60}")


def _table_row(
    label: str,
    phi: np.ndarray,
    u_exact: np.ndarray,
    u_thomas4: np.ndarray,
    n_iters: int,
    converged: bool,
    wall: float,
) -> None:
    err_e = _max_rel_err(phi, u_exact)
    err_t = _max_rel_err(phi, u_thomas4)
    col = _colour(err_e)
    conv_str = f"{G}Yes{X}" if converged else f"{R}No{X}"
    print(f"  {label:<14}  "
          f"vs_exact={col}{err_e:7.3f}%{X}  "
          f"vs_thomas4={err_t:7.3f}%  "
          f"{n_iters:>6}  {conv_str}  "
          f"time={wall:.2f}s")


# -- Compare orders ------------------------------------------------------------

def compare_orders(N_values: list[int]) -> None:
    """Compare second- and fourth-order 2D accuracy at multiple N."""
    _header("2D: Second-Order vs Fourth-Order Accuracy Comparison")
    print(f"\n  Source: -2π²sin(πx)sin(πy)  |  Homogeneous BCs\n")
    print(f"  {'N':>4}  {'err_2nd%':>10}  {'err_4th%':>10}  "
          f"{'improvement':>12}")
    print(f"  {'─'*45}")

    for N in N_values:
        x, y, dx = build_grid_2d(N)
        f_vals = f_sin_2d(x, y)
        u_ex = u_exact_2d(x, y)

        # Second-order
        t0 = time.perf_counter()
        phi_2nd = run_thomas_2d_2nd(N, f_vals, dx)
        _ = time.perf_counter() - t0

        # Fourth-order Thomas
        t0 = time.perf_counter()
        phi_4th, _, _, _ = jacobi_2d_4th(
            N, f_vals, dx, "thomas", {},
            u_ex, phi_2nd,
            tol=1e-8, max_iter=500, print_every=9999,
        )
        _ = time.perf_counter() - t0

        err_2nd = _max_rel_err(phi_2nd, u_ex)
        err_4th = _max_rel_err(phi_4th, u_ex)
        improvement = err_2nd / max(err_4th, 1e-12)

        c2 = _colour(err_2nd)
        c4 = _colour(err_4th)
        print(f"  {N:>4}  "
              f"{c2}{err_2nd:>9.3f}%{X}  "
              f"{c4}{err_4th:>9.3f}%{X}  "
              f"{'×'+f'{improvement:.1f}':>12}")
    print()


# -- Main run ------------------------------------------------------------------

def run_single(
    N: int,
    inner: str,
    n_layers: int,
    epsilon: float,
    max_degree: Optional[int],
    angle_method: str,
    trotter_steps: Optional[int],
    tol: float,
    max_iter: int,
    do_plot: bool,
) -> None:
    x, y, dx = build_grid_2d(N)
    f_vals = f_sin_2d(x, y)
    u_ex = u_exact_2d(x, y)

    _header(f"Fourth-Order 2D Poisson  —  N={N}")
    print(f"\n  Domain: unit square [0,1]²  |  h={dx:.4f}")
    print(f"  Source: -2π²sin(πx)sin(πy)  |  Exact: sin(πx)sin(πy)")
    print(f"  Strip matrix: pentadiagonal (4th-order in x, 2nd-order coupling in y)")

    solutions: dict[str, np.ndarray] = {}

    # -- Thomas 4th-order reference --------------------------------------------
    _header_inner = f"Thomas-4th (reference)"
    print(f"\n  Running {_header_inner}...")
    t0 = time.perf_counter()
    phi_thomas4, n_it, conv, _ = jacobi_2d_4th(
        N, f_vals, dx, "thomas", {},
        u_ex, np.zeros((N, N)),
        tol=tol, max_iter=max_iter, print_every=max_iter + 1,
    )
    t_thomas4 = time.perf_counter() - t0
    solutions["Thomas-4th"] = phi_thomas4

    # -- Thomas 2nd-order for comparison --------------------------------------
    t0 = time.perf_counter()
    phi_thomas2 = run_thomas_2d_2nd(N, f_vals, dx)
    t_thomas2 = time.perf_counter() - t0
    solutions["Thomas-2nd"] = phi_thomas2

    _table_header()
    _table_row("Thomas-4th", phi_thomas4, u_ex, phi_thomas4,
               n_it, conv, t_thomas4)
    _table_row("Thomas-2nd", phi_thomas2, u_ex, phi_thomas4,
               0, True, t_thomas2)

    # -- Quantum solvers -------------------------------------------------------
    solvers_to_run = (
        ["vqls", "qsvt", "hhl"] if inner == "all"
        else ([inner] if inner not in ("thomas",) else [])
    )

    inner_kwargs = {
        "n_layers": n_layers,
        "epsilon": epsilon,
        "max_degree": max_degree,
        "angle_method": angle_method,
        "trotter_steps": trotter_steps,
    }

    for sname in solvers_to_run:
        label = f"{sname.upper()}-4th"
        print(f"\n  Running {label}...")
        try:
            t0 = time.perf_counter()
            phi_q, n_it_q, conv_q, _ = jacobi_2d_4th(
                N, f_vals, dx, sname, inner_kwargs,
                u_ex, phi_thomas4,
                tol=tol, max_iter=max_iter, print_every=10,
            )
            wall_q = time.perf_counter() - t0
            solutions[label] = phi_q
            _table_row(label, phi_q, u_ex, phi_thomas4,
                       n_it_q, conv_q, wall_q)
        except Exception as exc:
            print(f"  {label:<14}  {R}FAILED: {exc}{X}")

    # -- Plot ------------------------------------------------------------------
    if do_plot:
        out_dir = REPO_ROOT / "results" / "debugging"
        plot_solutions(x, y, u_ex, solutions, N, out_dir)


# -- Entry point ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug tool for the fourth-order 2D Poisson solver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--N", type=int, default=4,
                        help="Grid size N×N (power of 2, >=4). Default: 4.")
    parser.add_argument("--inner", type=str, default="thomas",
                        choices=["thomas", "vqls", "qsvt", "hhl", "all"],
                        help="Inner solver(s). Default: thomas.")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="VQLS ansatz depth. Default: 3.")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="QSVT/HHL epsilon. Default: 0.05.")
    parser.add_argument("--max-degree", type=int, default=None,
                        help="QSVT max polynomial degree. Default: None.")
    parser.add_argument("--angle-method", type=str, default="auto",
                        help="QSP angle method. Default: auto.")
    parser.add_argument("--trotter-steps", type=int, default=None,
                        help="HHL trotter steps (None = auto-compute). Default: None.")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Outer iteration tolerance. Default: 1e-6.")
    parser.add_argument("--max-iter", type=int, default=200,
                        help="Max outer iterations. Default: 200.")
    parser.add_argument("--plot", action="store_true",
                        help="Save solution field plots.")
    parser.add_argument("--compare-orders", action="store_true",
                        help="Print 2nd vs 4th order accuracy table.")
    parser.add_argument("--N-values", type=int, nargs="+", default=[4, 8],
                        help="N values for --compare-orders. Default: 4 8.")
    args = parser.parse_args()

    if args.compare_orders:
        compare_orders(args.N_values)
    else:
        run_single(
            N=args.N,
            inner=args.inner,
            n_layers=args.n_layers,
            epsilon=args.epsilon,
            max_degree=args.max_degree,
            angle_method=args.angle_method,
            trotter_steps=args.trotter_steps,
            tol=args.tol,
            max_iter=args.max_iter,
            do_plot=args.plot,
        )


if __name__ == "__main__":
    main()