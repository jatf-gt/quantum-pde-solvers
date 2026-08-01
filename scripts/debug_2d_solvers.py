#!/usr/bin/env python3
"""
=================================================================
Debug tool for 2D Poisson Line-Jacobi with HHL, VQLS, QSVT inner solvers.

Confirmed interfaces from --introspect output:

  HHL:  hhl_solve_system(A, b, epsilon) -> tuple[np.ndarray, np.ndarray, float]
        result[0] = solution vector u

  VQLS: vqls_solve_system(A, b, config=VQLSConfig1D(...)) -> VQLSSolverResult
        result.u              = solution vector
        result.final_cost     = final cost function value
        result.euclidean_residual = residual

  QSVT: qsvt_solve_system(A, b, config=QSVTConfig1D(...)) -> QSVTSolverResult
        result.u              = solution vector
        result.polynomial_degree
        result.circuit_depth
        result.euclidean_residual

Plots are saved to results/debugging/

Usage:
    python scripts/debug_2d_solvers.py [--N 4] [--plot] [--solver all|hhl|vqls|qsvt]
    python scripts/debug_2d_solvers.py --introspect
"""

from __future__ import annotations

import argparse
import inspect
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# ── Repo root on path ─────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = REPO_ROOT / "results" / "debugging"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Terminal colours ──────────────────────────────────────────────────────────
_G = "\033[92m"   # green
_Y = "\033[93m"   # yellow
_R = "\033[91m"   # red
_C = "\033[96m"   # cyan
_B = "\033[1m"    # bold
_X = "\033[0m"    # reset

# ── HHL epsilon (matches your 1D benchmark runs) ──────────────────────────────
HHL_EPSILON = 0.01


# ============================================================================
#  Problem definition
# ============================================================================

def build_grid_2d(N: int):
    """Interior grid for [0,1]². Returns (x, y, dx)."""
    dx = 1.0 / (N + 1)
    pts = np.arange(1, N + 1) * dx
    x, y = np.meshgrid(pts, pts, indexing="ij")   # shape (N, N)
    return x, y, dx


def f_sin2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Source: f(x,y) = sin(πx)·sin(πy)"""
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def u_exact_sin2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Analytical solution: φ = -sin(πx)·sin(πy) / (2π²)"""
    return -np.sin(np.pi * x) * np.sin(np.pi * y) / (2.0 * np.pi**2)


def build_tst_row(N: int, dx: float) -> tuple[np.ndarray, float]:
    """
    N×N TST matrix for one row of the 2D Line-Jacobi update.
    Main diagonal = -4, off-diagonals = +1.
    κ(A_row) ≈ 3 for all N (bounded, O(1)) — much better than 1D.
    """
    A = (-4.0 * np.eye(N)
         + np.diag(np.ones(N - 1), 1)
         + np.diag(np.ones(N - 1), -1))
    eigs = np.abs(np.linalg.eigvalsh(A))
    kappa = float(eigs.max() / eigs.min())
    return A, kappa


def build_rhs_row(j: int, phi: np.ndarray,
                  f_vals: np.ndarray, dx: float) -> np.ndarray:
    """
    RHS for the j-th interior row update (0-indexed).
    phi shape: (N, N) — current iterate
    f_vals shape: (N, N) — source at interior nodes
    """
    N = phi.shape[0]
    b = dx**2 * f_vals[:, j].copy()
    if j > 0:
        b -= phi[:, j - 1]
    if j < N - 1:
        b -= phi[:, j + 1]
    return b


# ============================================================================
#  Classical Thomas solver (reference)
# ============================================================================

def _thomas_1d(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Thomas algorithm for the tridiagonal row system (a=-4, b=1)."""
    N = len(b)
    diag = -4.0 * np.ones(N)
    off  =  1.0 * np.ones(N)
    d    = b.copy()
    for i in range(1, N):
        m = off[i - 1] / diag[i - 1]
        diag[i] -= m * off[i - 1]
        d[i]    -= m * d[i - 1]
    u = np.zeros(N)
    u[-1] = d[-1] / diag[-1]
    for i in range(N - 2, -1, -1):
        u[i] = (d[i] - off[i] * u[i + 1]) / diag[i]
    return u


def jacobi_2d_thomas(N: int, f_vals: np.ndarray, dx: float,
                     tol: float = 1e-8, max_iter: int = 300,
                     verbose: bool = False
                     ) -> tuple[np.ndarray, int, bool, list[float]]:
    """2D Line-Jacobi with Thomas inner solver. Returns (phi, iters, converged, deltas)."""
    phi = np.zeros((N, N))
    A, _ = build_tst_row(N, dx)
    deltas = []
    for it in range(max_iter):
        phi_new = phi.copy()
        for j in range(N):
            b = build_rhs_row(j, phi, f_vals, dx)
            phi_new[:, j] = _thomas_1d(A, b)
        delta = float(np.max(np.abs(phi_new - phi)))
        deltas.append(delta)
        phi = phi_new
        if verbose and (it + 1) % 10 == 0:
            print(f"    Thomas iter {it+1:4d}  Δ={delta:.3e}")
        if delta < tol:
            return phi, it + 1, True, deltas
    return phi, max_iter, False, deltas


# ============================================================================
#  Solver call adapters — built from introspection output
# ============================================================================

def _call_hhl(A: np.ndarray, b: np.ndarray) -> tuple[Optional[np.ndarray], float, float, dict]:
    """
    HHL adapter.
    Signature: hhl_solve_system(A, b, epsilon) -> tuple[u, raw_state, prop_const]
    """
    from solvers.quantum.hhl_1d import hhl_solve_system
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = hhl_solve_system(A, b, HHL_EPSILON)
        wall = time.perf_counter() - t0
        # result is tuple[np.ndarray, np.ndarray, float]
        u = np.asarray(result[0], dtype=float)
        prop_const = float(result[2]) if len(result) > 2 else float("nan")
        res = float(np.linalg.norm(A @ u - b) / (np.linalg.norm(b) + 1e-300))
        return u, res, wall, {"prop_const": prop_const}
    except Exception as e:
        wall = time.perf_counter() - t0
        return None, float("nan"), wall, {"error": str(e)}


def _call_vqls(A: np.ndarray, b: np.ndarray) -> tuple[Optional[np.ndarray], float, float, dict]:
    """
    VQLS adapter.
    Signature: vqls_solve_system(A, b, config=VQLSConfig1D(...)) -> VQLSSolverResult
    Relevant attributes: .u, .final_cost, .euclidean_residual, .optimiser_success
    """
    from solvers.quantum.vqls_1d import vqls_solve_system
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = vqls_solve_system(A, b)
        wall = time.perf_counter() - t0
        u = np.asarray(result.u, dtype=float)
        res = float(np.linalg.norm(A @ u - b) / (np.linalg.norm(b) + 1e-300))
        extra = {
            "final_cost":        getattr(result, "final_cost", float("nan")),
            "optimiser_success": getattr(result, "optimiser_success", None),
            "n_circuit_evals":   getattr(result, "n_circuit_evals", None),
        }
        return u, res, wall, extra
    except Exception as e:
        wall = time.perf_counter() - t0
        return None, float("nan"), wall, {"error": str(e)}


def _call_qsvt(A: np.ndarray, b: np.ndarray) -> tuple[Optional[np.ndarray], float, float, dict]:
    """
    QSVT adapter.
    Signature: qsvt_solve_system(A, b, config=QSVTConfig1D(...)) -> QSVTSolverResult
    Relevant attributes: .u, .polynomial_degree, .circuit_depth, .euclidean_residual
    """
    from solvers.quantum.qsvt_1d import qsvt_solve_system
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = qsvt_solve_system(A, b)
        wall = time.perf_counter() - t0
        u = np.asarray(result.u, dtype=float)
        res = float(np.linalg.norm(A @ u - b) / (np.linalg.norm(b) + 1e-300))
        extra = {
            "polynomial_degree": getattr(result, "polynomial_degree", None),
            "circuit_depth":     getattr(result, "circuit_depth", None),
            "n_qubits":          getattr(result, "n_qubits", None),
            "alpha":             getattr(result, "alpha", None),
            "kappa_effective":   getattr(result, "kappa_effective", None),
        }
        return u, res, wall, extra
    except Exception as e:
        wall = time.perf_counter() - t0
        return None, float("nan"), wall, {"error": str(e)}


ADAPTERS = {
    "hhl":  _call_hhl,
    "vqls": _call_vqls,
    "qsvt": _call_qsvt,
}


# ============================================================================
#  2D Line-Jacobi with quantum inner solver
# ============================================================================

def jacobi_2d_quantum(
    N: int,
    f_vals: np.ndarray,
    dx: float,
    solver_name: str,
    u_exact: np.ndarray,
    u_thomas: np.ndarray,
    tol: float = 1e-6,
    max_iter: int = 100,
    print_every: int = 5,
) -> tuple[Optional[np.ndarray], int, bool, list[float], dict]:
    """
    2D Line-Jacobi with a quantum inner solver.
    Returns (phi, n_iters, converged, delta_history, diagnostics).
    """
    phi = np.zeros((N, N))
    A, kappa = build_tst_row(N, dx)
    adapter  = ADAPTERS[solver_name]
    label    = solver_name.upper()

    diag = {
        "kappa_row":     kappa,
        "inner_times":   [],
        "extra_per_row": [],
        "vs_exact_err":  [],
        "vs_thomas_err": [],
        "row_residuals": [],
    }
    deltas = []

    print(f"\n  {_C}{'─'*60}{_X}")
    print(f"  {_B}{label}-2D{_X}  N={N}  κ(A_row)={kappa:.4f}  "
          f"tol={tol:.0e}  max_iter={max_iter}")
    print(f"  {'─'*60}")

    for it in range(max_iter):
        phi_new = phi.copy()

        for j in range(N):
            b = build_rhs_row(j, phi, f_vals, dx)
            u_row, res_row, t_row, extra = adapter(A, b)

            if u_row is None:
                err_msg = extra.get("error", "unknown")
                print(f"  {_R}[FAIL] iter {it+1} row {j}: {err_msg}{_X}")
                return phi, it + 1, False, deltas, diag

            # Shape guard
            if u_row.shape != (N,):
                print(f"  {_R}[SHAPE] iter {it+1} row {j}: "
                      f"got {u_row.shape}, expected ({N},){_X}")
                return phi, it + 1, False, deltas, diag

            phi_new[:, j] = u_row
            diag["inner_times"].append(t_row)
            diag["extra_per_row"].append(extra)
            if (it + 1) % print_every == 0:
                diag["row_residuals"].append((it + 1, j, res_row))

        delta = float(np.max(np.abs(phi_new - phi)))
        deltas.append(delta)
        phi = phi_new

        if (it + 1) % print_every == 0:
            err_e = _max_rel(phi, u_exact)
            err_t = _max_rel(phi, u_thomas)
            diag["vs_exact_err"].append((it + 1, err_e))
            diag["vs_thomas_err"].append((it + 1, err_t))
            col = _G if err_e < 5.0 else (_Y if err_e < 20.0 else _R)
            print(f"  iter {it+1:4d}  Δ={delta:.3e}  "
                  f"vs_exact={col}{err_e:7.3f}%{_X}  "
                  f"vs_thomas={err_t:7.3f}%")

        if delta < tol:
            print(f"  {_G}Converged at iter {it+1}  (Δ={delta:.3e}){_X}")
            return phi, it + 1, True, deltas, diag

    print(f"  {_R}Max iterations reached. Final Δ={deltas[-1]:.3e}{_X}")
    _check_stagnation(deltas, phi, u_exact, u_thomas, label)
    return phi, max_iter, False, deltas, diag


def _max_rel(u: np.ndarray, ref: np.ndarray) -> float:
    """Max relative error in percent."""
    return float(np.max(np.abs(u - ref))) / (float(np.max(np.abs(ref))) + 1e-300) * 100.0


def _check_stagnation(deltas: list[float], phi: np.ndarray,
                      u_exact: np.ndarray, u_thomas: np.ndarray,
                      label: str) -> None:
    """Detect and diagnose stagnation."""
    if len(deltas) < 20:
        return
    last = np.array(deltas[-20:])
    if np.std(last) / (np.mean(last) + 1e-300) > 0.01:
        return   # not stagnated

    err_e = _max_rel(phi, u_exact)
    err_t = _max_rel(phi, u_thomas)
    err_thomas_e = _max_rel(u_thomas, u_exact)

    print(f"\n  {_R}⚠  STAGNATION DETECTED  (Δ≈{np.mean(last):.3e}){_X}")
    print(f"  {'─'*55}")
    print(f"  Error decomposition:")
    print(f"    Thomas vs exact  (Jacobi discretisation): "
          f"{_Y}{err_thomas_e:.3f}%{_X}")
    print(f"    {label} vs Thomas (quantum algorithmic):  "
          f"{_R}{err_t:.3f}%{_X}")
    print(f"    {label} vs exact  (total):                "
          f"{_R}{err_e:.3f}%{_X}")

    if err_t > 50.0:
        print(f"\n  {_R}Dominant error is QUANTUM ALGORITHMIC (not Jacobi).{_X}")
        print(f"  Likely cause for {label}:")
        if label == "QSVT":
            print(f"    • Proportionality recovery c = <b|A|x>/||Ax||²")
            print(f"      may be using wrong matrix (a=-2 not a=-4).")
            print(f"    • Block encoding normalisation α may be wrong")
            print(f"      for the 2D row matrix (a=-4, b=1, κ≈3).")
            print(f"    • The QSVT solution direction is correct but")
            print(f"      scale is wrong — check result.prop_const.")
        elif label == "HHL":
            print(f"    • HHL proportionality recovery may assume a=-2.")
            print(f"    • Check that TridiagonalToeplitz gets a=-4/||A||₂.")
        elif label == "VQLS":
            print(f"    • VQLS cost function may not be converging for")
            print(f"      the 2D row matrix (different condition number).")
            print(f"    • Check result.final_cost — if >1e-3, not converged.")
    print(f"  {'─'*55}")


# ============================================================================
#  Summary printer
# ============================================================================

def print_summary(label: str, phi: Optional[np.ndarray],
                  u_exact: np.ndarray, u_thomas: np.ndarray,
                  n_iters: int, converged: bool,
                  diag: dict, wall_total: float) -> None:
    sep = "═" * 60
    print(f"\n  {_B}{sep}{_X}")
    print(f"  {_B}SUMMARY: {label}-2D{_X}")
    print(f"  {sep}")

    status = f"{_G}CONVERGED{_X}" if converged else f"{_R}NOT CONVERGED{_X}"
    print(f"  Status      : {status}  ({n_iters} iters)  {wall_total:.2f}s")

    if phi is not None and not np.allclose(phi, 0.0):
        err_e = _max_rel(phi, u_exact)
        err_t = _max_rel(phi, u_thomas)
        rms_e = float(np.sqrt(np.mean((phi - u_exact)**2)))
        col_e = _G if err_e < 5.0 else (_Y if err_e < 20.0 else _R)
        col_t = _G if err_t < 5.0 else (_Y if err_t < 20.0 else _R)
        print(f"  MaxRelErr vs exact  : {col_e}{err_e:8.3f}%{_X}")
        print(f"  MaxRelErr vs Thomas : {col_t}{err_t:8.3f}%{_X}")
        print(f"  RMS error vs exact  : {rms_e:.3e}")
    else:
        print(f"  {_R}Solution is zero or None — solver returned no output.{_X}")

    # Inner solver timing
    if diag.get("inner_times"):
        t = np.array(diag["inner_times"])
        print(f"  Inner timing: mean={t.mean():.4f}s  "
              f"max={t.max():.4f}s  total={t.sum():.2f}s")

    # VQLS cost
    costs = [e.get("final_cost") for e in diag.get("extra_per_row", [])
             if e.get("final_cost") is not None and not np.isnan(e["final_cost"])]
    if costs:
        c = np.array(costs, dtype=float)
        col = _G if c.max() < 1e-4 else (_Y if c.max() < 1e-2 else _R)
        print(f"  VQLS cost   : mean={c.mean():.2e}  "
              f"max={col}{c.max():.2e}{_X}  min={c.min():.2e}")
        if c.max() > 1e-3:
            print(f"  {_Y}  ↳ High cost — VQLS not fully converging on some rows.{_X}")

    # QSVT degree/depth
    degrees = [e.get("polynomial_degree") for e in diag.get("extra_per_row", [])
               if e.get("polynomial_degree") is not None]
    depths  = [e.get("circuit_depth") for e in diag.get("extra_per_row", [])
               if e.get("circuit_depth") is not None]
    if degrees:
        d = np.array(degrees)
        print(f"  QSVT degree : mean={d.mean():.0f}  max={d.max():.0f}  min={d.min():.0f}")
    if depths:
        dep = np.array(depths)
        print(f"  QSVT depth  : mean={dep.mean():.0f}  max={dep.max():.0f}")

    # HHL prop_const
    props = [e.get("prop_const") for e in diag.get("extra_per_row", [])
             if e.get("prop_const") is not None and not np.isnan(e["prop_const"])]
    if props:
        p = np.array(props)
        print(f"  HHL c (prop): mean={p.mean():.4f}  "
              f"std={p.std():.4f}  min={p.min():.4f}  max={p.max():.4f}")
        if p.std() / (abs(p.mean()) + 1e-300) > 0.1:
            print(f"  {_Y}  ↳ High variance in prop_const — "
                  f"recovery may be unstable.{_X}")

    print(f"  {sep}")


# ============================================================================
#  Plotting
# ============================================================================

def plot_solutions(N: int, x: np.ndarray, y: np.ndarray,
                   u_exact: np.ndarray, results: dict) -> None:
    """2-row figure: solution fields (top) and error maps (bottom)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print(f"{_Y}matplotlib not available — skipping plot.{_X}")
        return

    valid = {k: v for k, v in results.items()
             if v is not None and not np.allclose(v, 0.0)}
    n_cols = 1 + len(valid)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    vmin, vmax = u_exact.min(), u_exact.max()

    # Exact solution
    im = axes[0, 0].pcolormesh(x, y, u_exact, cmap="RdBu_r",
                                vmin=vmin, vmax=vmax, shading="auto")
    axes[0, 0].set_title("Exact", fontweight="bold", fontsize=10)
    axes[0, 0].set_aspect("equal")
    axes[0, 0].set_xlabel("x"); axes[0, 0].set_ylabel("y")
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
    axes[1, 0].axis("off")

    for ci, (name, phi) in enumerate(valid.items(), 1):
        # Solution
        im = axes[0, ci].pcolormesh(x, y, phi, cmap="RdBu_r",
                                     vmin=vmin, vmax=vmax, shading="auto")
        axes[0, ci].set_title(name, fontweight="bold", fontsize=10)
        axes[0, ci].set_aspect("equal")
        axes[0, ci].set_xlabel("x"); axes[0, ci].set_ylabel("y")
        plt.colorbar(im, ax=axes[0, ci], shrink=0.8)

        # Error map
        err = phi - u_exact
        abs_max = max(np.abs(err).max(), 1e-12)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        im2 = axes[1, ci].pcolormesh(x, y, err, cmap="seismic",
                                      norm=norm, shading="auto")
        rel = _max_rel(phi, u_exact)
        axes[1, ci].set_title(f"Error  ({rel:.2f}%)", fontsize=9)
        axes[1, ci].set_aspect("equal")
        axes[1, ci].set_xlabel("x"); axes[1, ci].set_ylabel("y")
        plt.colorbar(im2, ax=axes[1, ci], shrink=0.8)

    fig.suptitle(f"2D Poisson Debug — N={N}  [f = sin(πx)sin(πy)]",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / f"debug_2d_solutions_N{N}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  {_G}Solutions figure saved: {out}{_X}")
    plt.close(fig)


def plot_convergence(errors_by_solver: dict, N: int) -> None:
    """Convergence curves (Jacobi delta vs iteration)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    colours = {"Thomas": "black", "HHL": "royalblue",
               "VQLS": "darkorange", "QSVT": "crimson"}
    fig, ax = plt.subplots(figsize=(8, 4))

    for name, errs in errors_by_solver.items():
        if errs:
            ax.semilogy(range(1, len(errs) + 1), errs,
                        label=name, color=colours.get(name, "grey"),
                        linewidth=1.8, alpha=0.9)

    ax.set_xlabel("Jacobi Iteration", fontsize=11)
    ax.set_ylabel("Max Δ  (convergence delta)", fontsize=11)
    ax.set_title(f"2D Line-Jacobi Convergence — N={N}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / f"debug_2d_convergence_N{N}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  {_G}Convergence figure saved: {out}{_X}")
    plt.close(fig)


def plot_row_residuals(diag_by_solver: dict, N: int) -> None:
    """
    Per-row residual heatmap for each solver.
    Rows on x-axis, iterations on y-axis, residual magnitude as colour.
    Useful for spotting which rows are problematic.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    solvers_with_data = {k: v for k, v in diag_by_solver.items()
                         if v.get("row_residuals")}
    if not solvers_with_data:
        return

    n_cols = len(solvers_with_data)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]

    for ax, (name, diag) in zip(axes, solvers_with_data.items()):
        rr = diag["row_residuals"]   # list of (iter, row, residual)
        if not rr:
            ax.set_title(f"{name} — no data")
            continue
        iters = sorted(set(r[0] for r in rr))
        rows  = sorted(set(r[1] for r in rr))
        mat   = np.full((len(iters), len(rows)), np.nan)
        iter_idx = {v: i for i, v in enumerate(iters)}
        row_idx  = {v: i for i, v in enumerate(rows)}
        for it, row, res in rr:
            if not np.isnan(res):
                mat[iter_idx[it], row_idx[row]] = np.log10(res + 1e-16)
        im = ax.imshow(mat, aspect="auto", cmap="plasma",
                       origin="lower", interpolation="nearest")
        ax.set_xlabel("Row index j")
        ax.set_ylabel("Iteration")
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(rows)
        ax.set_yticks(range(len(iters)))
        ax.set_yticklabels(iters)
        ax.set_title(f"{name} — log₁₀(residual per row)", fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8, label="log₁₀(res)")

    fig.suptitle(f"Per-row inner solver residuals — N={N}", fontsize=11)
    plt.tight_layout()
    out = OUT_DIR / f"debug_2d_row_residuals_N{N}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  {_G}Row residuals figure saved: {out}{_X}")
    plt.close(fig)


# ============================================================================
#  Introspection (unchanged from v2)
# ============================================================================

def introspect_solvers() -> None:
    print(f"\n{_B}SOLVER INTROSPECTION{_X}")
    print("=" * 60)
    specs = {
        "HHL":  ("solvers.quantum.hhl_1d",  "hhl_solve_system"),
        "VQLS": ("solvers.quantum.vqls_1d", "vqls_solve_system"),
        "QSVT": ("solvers.quantum.qsvt_1d", "qsvt_solve_system"),
    }
    N = 4
    dx = 1.0 / (N + 1)
    A_test = (-4.0 * np.eye(N) + np.diag(np.ones(N-1), 1)
              + np.diag(np.ones(N-1), -1))
    b_test = np.array([0.04, 0.04, 0.04, 0.04])

    for label, (mod_path, fn_name) in specs.items():
        print(f"\n{_C}--- {label} ---{_X}")
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            fn  = getattr(mod, fn_name)
            print(f"  Signature: {fn_name}{inspect.signature(fn)}")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = fn(A_test, b_test) if label != "HHL" else fn(A_test, b_test, 0.01)
                print(f"  Return type: {type(result)}")
                if hasattr(result, "__dict__"):
                    print(f"  Attributes:  {list(result.__dict__.keys())}")
                elif isinstance(result, (tuple, list)):
                    print(f"  Tuple len:   {len(result)}")
                    for i, v in enumerate(result):
                        print(f"    [{i}] type={type(v).__name__}  "
                              f"shape={getattr(v, 'shape', 'N/A')}")
            except Exception as e:
                print(f"  {_R}Call failed: {e}{_X}")
        except (ImportError, AttributeError) as e:
            print(f"  {_R}Import failed: {e}{_X}")
    print("\n" + "=" * 60 + "\n")


# ============================================================================
#  Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug tool for 2D quantum Poisson solvers.")
    parser.add_argument("--N", type=int, default=4,
                        help="Grid size N×N. Default: 4.")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="Max Jacobi iterations. Default: 100.")
    parser.add_argument("--tol", type=float, default=1e-6,
                        help="Convergence tolerance. Default: 1e-6.")
    parser.add_argument("--solver", default="all",
                        choices=["all", "hhl", "vqls", "qsvt", "thomas"],
                        help="Which solver(s) to run.")
    parser.add_argument("--plot", action="store_true",
                        help="Save figures to results/debugging/.")
    parser.add_argument("--print-every", type=int, default=5,
                        help="Print diagnostics every N iters. Default: 5.")
    parser.add_argument("--qsvt-max-iter", type=int, default=50,
                        help="Max iters for QSVT (slow). Default: 50.")
    parser.add_argument("--introspect", action="store_true",
                        help="Print solver signatures and exit.")
    args = parser.parse_args()

    if args.introspect:
        introspect_solvers()
        return

    N = args.N
    print(f"\n{_B}{'═'*64}{_X}")
    print(f"{_B}  2D QUANTUM POISSON SOLVER DEBUG TOOL  (v3){_X}")
    print(f"{_B}  N={N}  tol={args.tol:.0e}  HHL_ε={HHL_EPSILON}{_X}")
    print(f"{_B}  Output: {OUT_DIR}{_X}")
    print(f"{_B}{'═'*64}{_X}")

    # Problem setup
    x, y, dx = build_grid_2d(N)
    f_vals   = f_sin2d(x, y)
    u_exact  = u_exact_sin2d(x, y)
    _, kappa = build_tst_row(N, dx)

    print(f"\n  Problem : ∇²φ = sin(πx)sin(πy)  on [0,1]²  φ=0 on ∂Ω")
    print(f"  Grid    : {N}×{N} interior nodes  h={dx:.4f}")
    print(f"  κ(A_row): {kappa:.4f}  (row matrix a=-4, b=1)")
    print(f"  max|u_exact| = {np.max(np.abs(u_exact)):.6f}")

    # Thomas reference
    print(f"\n{_B}  Thomas-2D (reference){_X}")
    t0 = time.perf_counter()
    u_thomas, n_th, conv_th, errs_th = jacobi_2d_thomas(
        N, f_vals, dx, tol=args.tol, max_iter=args.max_iter, verbose=True)
    t_th = time.perf_counter() - t0
    err_th = _max_rel(u_thomas, u_exact)
    col = _G if err_th < 5.0 else _R
    print(f"  Thomas: {n_th} iters  "
          f"MaxRelErr={col}{err_th:.3f}%{_X}  "
          f"Time={t_th:.3f}s  Converged={conv_th}")

    results_dict    = {"Thomas": u_thomas}
    errors_by_solver = {"Thomas": errs_th}
    diag_by_solver   = {}

    solvers_to_run = (
        ["hhl", "vqls", "qsvt"] if args.solver == "all"
        else ([] if args.solver == "thomas" else [args.solver])
    )

    for sname in solvers_to_run:
        label  = sname.upper()
        max_it = args.qsvt_max_iter if sname == "qsvt" else args.max_iter

        t0 = time.perf_counter()
        phi, n_iters, converged, errs, diag = jacobi_2d_quantum(
            N=N, f_vals=f_vals, dx=dx,
            solver_name=sname,
            u_exact=u_exact, u_thomas=u_thomas,
            tol=args.tol, max_iter=max_it,
            print_every=args.print_every,
        )
        wall = time.perf_counter() - t0

        print_summary(label, phi, u_exact, u_thomas,
                      n_iters, converged, diag, wall)

        results_dict[label]     = phi
        errors_by_solver[label] = errs
        diag_by_solver[label]   = diag

    # Final comparison table
    print(f"\n{_B}{'═'*64}{_X}")
    print(f"{_B}  FINAL COMPARISON TABLE{_X}")
    print(f"{'─'*64}")
    print(f"  {'Solver':<8} {'Iters':>6} {'MaxRelErr%':>12} "
          f"{'vs Thomas%':>12} {'Conv':>8}")
    print(f"{'─'*64}")
    for name, phi in results_dict.items():
        if phi is None or np.allclose(phi, 0.0):
            print(f"  {name:<8} {'—':>6} {'FAILED':>12} {'—':>12} {'—':>8}")
            continue
        err_e = _max_rel(phi, u_exact)
        err_t = _max_rel(phi, u_thomas)
        n_it  = n_th if name == "Thomas" else "—"
        cv    = conv_th if name == "Thomas" else "—"
        col_e = _G if err_e < 5.0 else (_Y if err_e < 20.0 else _R)
        print(f"  {name:<8} {str(n_it):>6} "
              f"{col_e}{err_e:>11.3f}%{_X} "
              f"{err_t:>11.3f}% {str(cv):>8}")
    print(f"{'═'*64}\n")

    # Plots
    if args.plot:
        print(f"{_B}  Saving figures to {OUT_DIR} ...{_X}")
        plot_solutions(N, x, y, u_exact, results_dict)
        plot_convergence(errors_by_solver, N)
        plot_row_residuals(diag_by_solver, N)


if __name__ == "__main__":
    main()