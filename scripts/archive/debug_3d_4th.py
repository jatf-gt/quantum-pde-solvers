#!/usr/bin/env python3
"""
scripts/debug_3d_4th.py
-----------------------
Debug and validation script for the mixed-order 3D Poisson solver using
fourth-order spatial discretisation in the implicit (strip) direction.

Extends the 2D mixed-order approach (``debug_2d_4th.py``) to three
dimensions.  The implicit x-direction uses the fourth-order pentadiagonal
stencil [-1, 16, -30, 16, -1]/(12h²); the two explicit directions (y, z)
use the standard second-order stencil [1, -2, 1]/h².

Strip matrix convention
-----------------------
The strip matrix absorbs the transverse diagonal shifts from both y and
z coupling.  Each contributes -24 to the diagonal (12h² × (-2/h²)),
for a total shift of -48::

    diag(A_strip)  =  -30 - 48  =  -78   (interior rows)
                   =  -29 - 48  =  -77   (boundary rows, ghost-point +1)

The transverse coupling on the RHS is scaled by 12::

    b[:,j,k]  =  12h² f[:,j,k]
              -  12 u[:,j-1,k]  -  12 u[:,j+1,k]   (y coupling)
              -  12 u[:,j,k-1]  -  12 u[:,j,k+1]   (z coupling)

Manufactured solution:
    u(x,y,z) = sin(πx)sin(πy)sin(πz)
    f(x,y,z) = -3π²sin(πx)sin(πy)sin(πz)

Usage
-----
    python scripts/debug_3d_4th.py --N 4
    python scripts/debug_3d_4th.py --N 4 --inner vqls
    python scripts/debug_3d_4th.py --N 4 --inner qsvt --max-degree 200
    python scripts/debug_3d_4th.py --N 4 --inner all --plot
    python scripts/debug_3d_4th.py --compare-orders --N-values 4 8
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

G = "\033[92m"
Y = "\033[93m"
R = "\033[91m"
C = "\033[96m"
B = "\033[1m"
X = "\033[0m"


# ── Error metrics ─────────────────────────────────────────────────────────────

def _max_rel_err(u: np.ndarray, ref: np.ndarray) -> float:
    mask = np.abs(ref) > 1e-10
    if not np.any(mask):
        return float(np.max(np.abs(u - ref))) * 100.0
    return float(np.max(np.abs((u[mask] - ref[mask]) / ref[mask]))) * 100.0


def _colour(err: float) -> str:
    return G if err < 5.0 else (Y if err < 20.0 else R)


# ── Problem setup ─────────────────────────────────────────────────────────────

def build_grid_3d(N: int):
    """Interior grid for the unit cube."""
    dx = 1.0 / (N + 1)
    pts = np.arange(1, N + 1) * dx
    x, y, z = np.meshgrid(pts, pts, pts, indexing="ij")
    return x, y, z, dx


def f_sin_3d(x, y, z) -> np.ndarray:
    """Source: -3π²sin(πx)sin(πy)sin(πz)."""
    return -3.0 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z)


def u_exact_3d(x, y, z) -> np.ndarray:
    """Exact solution: sin(πx)sin(πy)sin(πz)."""
    return np.sin(np.pi*x) * np.sin(np.pi*y) * np.sin(np.pi*z)


def build_strip_matrix_4th(N: int, dx: float) -> np.ndarray:
    """
    Fourth-order pure x-direction pentadiagonal matrix.

    Returns the 1D pentadiagonal matrix from ``PoissonProblem1D4th``.
    In 3D, the caller must add the transverse diagonal shift (-48)
    to obtain the full strip operator — see ``jacobi_3d_4th``.
    """
    from problems.poisson_1d_4th import PoissonProblem1D4th
    prob = PoissonProblem1D4th(N=N, source_fn="fS")
    return prob.A.copy()


def build_strip_matrix_2nd(N: int, dx: float) -> np.ndarray:
    """
    Second-order 3D strip matrix (with y- and z-diagonals absorbed).

    In the h²-scaled convention for the unit cube (dx = dy = dz = h),
    the strip matrix diagonal is -6 (not -2): the three Laplacian
    directions each contribute -2 to the diagonal.
    """
    A = -6.0 * np.eye(N)
    if N > 1:
        np.fill_diagonal(A[1:, :], 1.0)
        np.fill_diagonal(A[:, 1:], 1.0)
    return A


def _build_rhs_strip_3d(
    j: int,
    k: int,
    phi: np.ndarray,
    f_vals: np.ndarray,
    dx: float,
) -> np.ndarray:
    """
    RHS for strip (j,k) in the 3D mixed-order line-Jacobi scheme.

    Implicit direction: x (fourth-order, handled by strip matrix with
    transverse diagonals absorbed).  Explicit directions: y and z
    (second-order coupling, scaled by 12)::

        b_i  =  12h² f(x_i, y_j, z_k)
             -  12 φ[i, j-1, k]  -  12 φ[i, j+1, k]   (y)
             -  12 φ[i, j, k-1]  -  12 φ[i, j, k+1]   (z)
    """
def _build_rhs_strip(
    j: int,
    k: int,
    phi: np.ndarray,
    f_vals: np.ndarray,
    dx: float,
    dy: float | None = None,
    dz: float | None = None,
    bc_lo: tuple = (0.0, 0.0, 0.0),
    bc_hi: tuple = (0.0, 0.0, 0.0),
    periodic: tuple = (False, False, False),
) -> np.ndarray:
    if dy is None:
        dy = dx
    if dz is None:
        dz = dx
    kappa_y = (dx / dy)**2
    kappa_z = (dx / dz)**2
    N = phi.shape[0]
    b = 12.0 * dx**2 * f_vals[:, j, k].copy()

    # X-boundary corrections (4th order implicit direction, non-periodic)
    ax0 = bc_lo[0]
    ax1 = bc_hi[0]
    def _extract_bc(bc_array, idx_j, idx_k):
        if isinstance(bc_array, np.ndarray):
            return bc_array[idx_j, idx_k]
        return bc_array
    
    val0 = _extract_bc(ax0, j, k)
    val1 = _extract_bc(ax1, j, k)
    b[0] -= 18.0 * val0
    if N > 1:
        b[1] += val0
    b[-1] -= 18.0 * val1
    if N > 1:
        b[-2] += val1

    # Y-boundary corrections (2nd order explicit direction)
    if j > 0:
        b -= 12.0 * kappa_y * phi[:, j-1, k]
    elif periodic[1]:
        b -= 12.0 * kappa_y * phi[:, -1, k]
    else:
        ay0 = bc_lo[1]
        v_ay0 = ay0[:, k] if isinstance(ay0, np.ndarray) else np.full(N, ay0)
        b -= 12.0 * kappa_y * v_ay0

    if j < N - 1:
        b -= 12.0 * kappa_y * phi[:, j+1, k]
    elif periodic[1]:
        b -= 12.0 * kappa_y * phi[:, 0, k]
    else:
        ay1 = bc_hi[1]
        v_ay1 = ay1[:, k] if isinstance(ay1, np.ndarray) else np.full(N, ay1)
        b -= 12.0 * kappa_y * v_ay1

    # Z-boundary corrections (2nd order explicit direction)
    if k > 0:
        b -= 12.0 * kappa_z * phi[:, j, k-1]
    elif periodic[2]:
        b -= 12.0 * kappa_z * phi[:, j, -1]
    else:
        az0 = bc_lo[2]
        v_az0 = az0[:, j] if isinstance(az0, np.ndarray) else np.full(N, az0)
        b -= 12.0 * kappa_z * v_az0

    if k < N - 1:
        b -= 12.0 * kappa_z * phi[:, j, k+1]
    elif periodic[2]:
        b -= 12.0 * kappa_z * phi[:, j, 0]
    else:
        az1 = bc_hi[2]
        v_az1 = az1[:, j] if isinstance(az1, np.ndarray) else np.full(N, az1)
        b -= 12.0 * kappa_z * v_az1

    return b


# ── Outer iteration ───────────────────────────────────────────────────────────

def jacobi_3d_4th(
    N: int,
    f_vals: np.ndarray,
    dx: float,
    dy: float | None = None,
    dz: float | None = None,
    bc_lo: tuple = (0.0, 0.0, 0.0),
    bc_hi: tuple = (0.0, 0.0, 0.0),
    periodic: tuple = (False, False, False),
    inner: str = "thomas",
    inner_kwargs: dict | None = None,
    u_exact: np.ndarray | None = None,
    u_thomas: np.ndarray | None = None,
    tol: float = 1e-6,
    max_iter: int = 300,
    print_every: int = 20,
) -> tuple[np.ndarray, int, bool, list[float]]:
    """
    3D mixed-order line-Jacobi outer iteration.

    Uses the fourth-order pentadiagonal strip matrix (with y- and
    z-diagonals absorbed, total shift -48) for the implicit direction
    and 12-scaled second-order coupling in the explicit y,z-directions.
    """
    if inner_kwargs is None:
        inner_kwargs = {}
    if dy is None:
        dy = dx
    if dz is None:
        dz = dx
    kappa_y = (dx / dy)**2
    kappa_z = (dx / dz)**2

    A_strip = build_strip_matrix_4th(N, dx)
    A_strip -= 24.0 * (kappa_y + kappa_z) * np.eye(N)
    phi = np.zeros((N, N, N))
    errors = []

    solve_strip = _make_strip_solver_3d(inner, A_strip, inner_kwargs)

    kappa = float(np.abs(np.linalg.eigvalsh(A_strip)).max() /
                  np.abs(np.linalg.eigvalsh(A_strip)).min())

    print(f"\n  {C}{'─'*60}{X}")
    print(f"  {B}{inner.upper()}-3D-4th{X}  N={N}  tol={tol:.0e}  "
          f"max_iter={max_iter}")
    print(f"  Strip matrix: pentadiagonal  kappa={kappa:.3f}")
    print(f"  {'─'*60}")

    for it in range(max_iter):
        phi_new = phi.copy()
        for j in range(N):
            for k in range(N):
                b_jk = _build_rhs_strip(
                    j, k, phi, f_vals, dx, dy, dz, bc_lo, bc_hi, periodic
                )
                phi_new[:, j, k] = solve_strip(b_jk)

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


def _make_strip_solver_3d(inner: str, A_strip: np.ndarray, kwargs: dict):
    """Return a callable (b) -> u for the given inner solver."""
    if inner == "thomas":
        return lambda b: np.linalg.solve(A_strip, b)

    elif inner == "vqls":
        from solvers.quantum.vqls_1d import vqls_solve_system, VQLSConfig1D
        cfg = VQLSConfig1D(n_layers=kwargs.get("n_layers", 3))
        def vqls_strip(b):
            return np.array(vqls_solve_system(A_strip, b, config=cfg).u)
        return vqls_strip

    elif inner == "qsvt":
        from solvers.quantum.qsvt_1d import qsvt_solve_system, QSVTConfig1D
        cfg = QSVTConfig1D(
            epsilon=kwargs.get("epsilon", 0.05),
            max_degree=kwargs.get("max_degree", None),
            angle_method=kwargs.get("angle_method", "auto"),
        )
        def qsvt_strip(b):
            return np.array(qsvt_solve_system(A_strip, b, config=cfg).u)
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


def run_thomas_3d_2nd(
    N: int,
    f_vals: np.ndarray,
    dx: float,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> np.ndarray:
    """Second-order 3D Thomas reference solver."""
    A_2nd = build_strip_matrix_2nd(N, dx)
    phi = np.zeros((N, N, N))

    for _ in range(max_iter):
        phi_new = phi.copy()
        for j in range(N):
            for k in range(N):
                b = dx**2 * f_vals[:, j, k].copy()
                if j > 0:
                    b -= phi[:, j-1, k]
                if j < N-1:
                    b -= phi[:, j+1, k]
                if k > 0:
                    b -= phi[:, j, k-1]
                if k < N-1:
                    b -= phi[:, j, k+1]
                phi_new[:, j, k] = np.linalg.solve(A_2nd, b)
        delta = float(np.max(np.abs(phi_new - phi)))
        phi = phi_new
        if delta < tol:
            break

    return phi


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_slices(
    x: np.ndarray,
    y: np.ndarray,
    u_exact: np.ndarray,
    solutions: dict[str, np.ndarray],
    N: int,
    out_dir: Path,
) -> None:
    """Plot mid-plane slices (z=0.5) for all solvers."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print(f"  {Y}matplotlib not available — skipping plot.{X}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    slices_idx = [N // 4, N // 2, 3 * N // 4]
    slice_names = ["0.25", "0.50", "0.75"]

    for mid, sname in zip(slices_idx, slice_names):
        x2d = x[:, :, mid]
        y2d = y[:, :, mid]
        u_ex_slice = u_exact[:, :, mid]

        palette = {"Thomas-4th": "black", "Thomas-2nd": "grey",
                   "VQLS-4th": "royalblue", "QSVT-4th": "crimson",
                   "HHL-4th": "darkorange"}

        n_cols = 1 + len(solutions)
        fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 7), squeeze=False)
        vmin, vmax = float(u_ex_slice.min()), float(u_ex_slice.max())

        im = axes[0, 0].pcolormesh(x2d, y2d, u_ex_slice, cmap="RdBu_r",
                                    vmin=vmin, vmax=vmax, shading="auto")
        axes[0, 0].set_title(f"Exact (z={sname})", fontweight="bold")
        axes[0, 0].set_aspect("equal")
        plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
        axes[1, 0].axis("off")

        for ci, (label, phi) in enumerate(solutions.items(), start=1):
            phi_slice = phi[:, :, mid]
            im = axes[0, ci].pcolormesh(x2d, y2d, phi_slice, cmap="RdBu_r",
                                         vmin=vmin, vmax=vmax, shading="auto")
            axes[0, ci].set_title(label, fontweight="bold")
            axes[0, ci].set_aspect("equal")
            plt.colorbar(im, ax=axes[0, ci], shrink=0.8)

            err = phi_slice - u_ex_slice
            abs_max = max(float(np.abs(err).max()), 1e-12)
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            im2 = axes[1, ci].pcolormesh(x2d, y2d, err, cmap="seismic",
                                          norm=norm, shading="auto")
            rel = _max_rel_err(phi, u_exact)
            axes[1, ci].set_title(f"Error ({rel:.2f}%)", fontsize=9)
            axes[1, ci].set_aspect("equal")
            plt.colorbar(im2, ax=axes[1, ci], shrink=0.8)

        fig.suptitle(f"Fourth-Order 3D Poisson — N={N} (z={sname} slice)",
                     fontweight="bold")
        plt.tight_layout()
        out_path = out_dir / f"4th_order_3d_N{N}_z{sname.replace('.', '')}.png"
        plt.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  {G}Plot saved to: {out_path}{X}")


# ── Header and table helpers ──────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{B}{C}{'═'*64}{X}")
    print(f"{B}{C}  {title}{X}")
    print(f"{B}{C}{'═'*64}{X}")


def _table_header() -> None:
    print(f"\n  {'Solver':<14}  {'vs_exact':>10}  {'vs_thomas4':>11}  "
          f"{'iters':>6}  {'conv':>5}  {'time':>8}")
    print(f"  {'─'*62}")


def _table_row(label, phi, u_exact, u_thomas4, n_iters, converged, wall):
    err_e = _max_rel_err(phi, u_exact)
    err_t = _max_rel_err(phi, u_thomas4)
    col = _colour(err_e)
    conv_str = f"{G}Yes{X}" if converged else f"{R}No{X}"
    print(f"  {label:<14}  "
          f"vs_exact={col}{err_e:7.3f}%{X}  "
          f"vs_thomas4={err_t:7.3f}%  "
          f"{n_iters:>6}  {conv_str}  "
          f"time={wall:.2f}s")


# ── Compare orders ────────────────────────────────────────────────────────────

def compare_orders(N_values: list[int]) -> None:
    _header("3D: Second-Order vs Fourth-Order Accuracy Comparison")
    print(f"\n  Source: -3π²sin(πx)sin(πy)sin(πz)  |  Homogeneous BCs\n")
    print(f"  {'N':>4}  {'err_2nd%':>10}  {'err_4th%':>10}  "
          f"{'improvement':>12}")
    print(f"  {'─'*45}")

    for N in N_values:
        x, y, z, dx = build_grid_3d(N)
        f_vals = f_sin_3d(x, y, z)
        u_ex = u_exact_3d(x, y, z)

        phi_2nd = run_thomas_3d_2nd(N, f_vals, dx)
        phi_4th, _, _, _ = jacobi_3d_4th(
            N, f_vals, dx, "thomas", {},
            u_ex, phi_2nd,
            tol=1e-8, max_iter=500, print_every=9999,
        )

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


# ── Main run ──────────────────────────────────────────────────────────────────

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
    x, y, z, dx = build_grid_3d(N)
    f_vals = f_sin_3d(x, y, z)
    u_ex = u_exact_3d(x, y, z)

    _header(f"Fourth-Order 3D Poisson  —  N={N}")
    print(f"\n  Domain: unit cube [0,1]³  |  h={dx:.4f}")
    print(f"  Source: -3π²sin(πx)sin(πy)sin(πz)")
    print(f"  Strip matrix: pentadiagonal (4th-order in x, 2nd-order in y,z)")

    solutions: dict[str, np.ndarray] = {}

    # ── Thomas 4th-order reference ────────────────────────────────────────────
    print(f"\n  Running Thomas-4th (reference)...")
    t0 = time.perf_counter()
    phi_thomas4, n_it, conv, _ = jacobi_3d_4th(
        N, f_vals, dx, "thomas", {},
        u_ex, np.zeros((N, N, N)),
        tol=tol, max_iter=max_iter, print_every=max_iter + 1,
    )
    t_thomas4 = time.perf_counter() - t0
    solutions["Thomas-4th"] = phi_thomas4

    # ── Thomas 2nd-order ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    phi_thomas2 = run_thomas_3d_2nd(N, f_vals, dx)
    t_thomas2 = time.perf_counter() - t0
    solutions["Thomas-2nd"] = phi_thomas2

    _table_header()
    _table_row("Thomas-4th", phi_thomas4, u_ex, phi_thomas4,
               n_it, conv, t_thomas4)
    _table_row("Thomas-2nd", phi_thomas2, u_ex, phi_thomas4,
               0, True, t_thomas2)

    # ── Quantum solvers ───────────────────────────────────────────────────────
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
            phi_q, n_it_q, conv_q, _ = jacobi_3d_4th(
                N, f_vals, dx, sname, inner_kwargs,
                u_ex, phi_thomas4,
                tol=tol, max_iter=max_iter, print_every=20,
            )
            wall_q = time.perf_counter() - t0
            solutions[label] = phi_q
            _table_row(label, phi_q, u_ex, phi_thomas4,
                       n_it_q, conv_q, wall_q)
        except Exception as exc:
            print(f"  {label:<14}  {R}FAILED: {exc}{X}")

    if do_plot:
        out_dir = REPO_ROOT / "results" / "debugging"
        plot_slices(x, y, u_ex, solutions, N, out_dir)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug tool for the fourth-order 3D Poisson solver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--N", type=int, default=4,
                        help="Grid size N×N×N (power of 2, >=4). Default: 4.")
    parser.add_argument("--inner", type=str, default="thomas",
                        choices=["thomas", "vqls", "qsvt", "hhl", "all"],
                        help="Inner solver(s). Default: thomas.")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="VQLS ansatz depth. Default: 3.")
    parser.add_argument("--epsilon", type=float, default=0.05,
                        help="QSVT epsilon. Default: 0.05.")
    parser.add_argument("--max-degree", type=int, default=None,
                        help="QSVT max polynomial degree. Default: None.")
    parser.add_argument("--angle-method", type=str, default="auto",
                        help="QSP angle method. Default: auto.")
    parser.add_argument("--trotter-steps", type=int, default=None,
                        help="HHL trotter steps (None = auto-compute). Default: None.")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Outer iteration tolerance. Default: 1e-6.")
    parser.add_argument("--max-iter", type=int, default=300,
                        help="Max outer iterations. Default: 300.")
    parser.add_argument("--plot", action="store_true",
                        help="Save mid-plane slice plots.")
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