#!/usr/bin/env python3
"""
debug_2d_solvers.py  (SOR outer iteration)
================================================
Key changes vs v4:
  1. Line-Jacobi replaced by Line-SOR throughout (Thomas and quantum).
     SOR gives O(N) convergence vs O(N^2) for Jacobi.
     omega_opt computed analytically from the spectral radius.
  2. Relative convergence tolerance: delta/max|phi| < tol.
     This is N-independent and correctly tracks the discretisation error.
  3. All helper functions defined locally — no missing references.
  4. HET case uses SOR consistently.
  5. run_het_2d_case called only once in main().
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "results" / "debugging"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_G = "\033[92m"; _Y = "\033[93m"; _R = "\033[91m"
_C = "\033[96m"; _B = "\033[1m";  _X = "\033[0m"

HHL_EPSILON = 0.01

EARLY_STOP_MIN_IMPROVEMENT = 0.01
EARLY_STOP_PATIENCE        = 4
VQLS_THOMAS_TOL            = 0.005   # 0.5%

# Relative convergence tolerance: delta / max|phi| < SOR_TOL
# N-independent: automatically adapts to solution magnitude.
SOR_TOL      = 1e-6
SOR_MAX_ITER = 2000

# HET physical parameters (SPT-100)
HET_Lz   = 0.025
HET_Lr   = 0.020
HET_phi0 = 300.0


# ============================================================================
#  SOR parameter
# ============================================================================

def _sor_omega(N: int) -> float:
    """
    Optimal SOR relaxation parameter for the 2D Poisson 5-point stencil.

    Spectral radius of Line-Jacobi: rho_J = cos(pi/(N+1))
    Optimal omega: 2 / (1 + sqrt(1 - rho_J^2))

    Gives O(N) convergence vs O(N^2) for Jacobi.
    Clamped to (1.0, 1.99) — omega=1 is Gauss-Seidel, omega>=2 diverges.
    """
    rho_J = np.cos(np.pi / (N + 1))
    omega = 2.0 / (1.0 + np.sqrt(1.0 - rho_J**2))
    return float(np.clip(omega, 1.0, 1.99))


# ============================================================================
#  Generic Poisson problem (unit square)
# ============================================================================

def build_grid_2d(N: int):
    """Interior grid for [0,1]^2. Returns (x, y, dx) with dx=dy."""
    dx = 1.0 / (N + 1)
    pts = np.arange(1, N + 1) * dx
    x, y = np.meshgrid(pts, pts, indexing="ij")
    return x, y, dx

def f_sin2d(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def u_exact_sin2d(x, y):
    return -np.sin(np.pi * x) * np.sin(np.pi * y) / (2.0 * np.pi**2)

def _build_row_matrix_square(N: int, dx: float):
    """
    Row matrix for unit square (dx=dy).
    Main diag: -4/dx^2 * (dx^2) = -4 (after scaling by dx^2).
    We work in the scaled system A*phi = dx^2*f so A has entries -4, +1.
    """
    A = (-4.0 * np.eye(N)
         + np.diag(np.ones(N-1), 1)
         + np.diag(np.ones(N-1), -1))
    eigs = np.abs(np.linalg.eigvalsh(A))
    return A, float(eigs.max() / eigs.min())

def _build_rhs_row_square(j: int, phi: np.ndarray,
                           f_vals: np.ndarray, dx: float) -> np.ndarray:
    """
    RHS for j-th strip of the unit square problem.
    System: A*phi[:,j] = dx^2*f[:,j] - phi[:,j-1] - phi[:,j+1]
    Uses the CURRENT phi (partially updated in SOR — Gauss-Seidel ordering).
    """
    N = phi.shape[0]
    b = dx**2 * f_vals[:, j].copy()
    if j > 0:
        b -= phi[:, j-1]
    if j < N - 1:
        b -= phi[:, j+1]
    return b

def _thomas_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Thomas algorithm for a general tridiagonal system Ax=b.
    Extracts diagonals from A — works for any tridiagonal, not just -4/+1.
    """
    N = len(b)
    main  = A.diagonal(0).copy()
    upper = A.diagonal(1).copy()
    lower = A.diagonal(-1).copy()
    d = b.copy()
    for i in range(1, N):
        m = lower[i-1] / main[i-1]
        main[i] -= m * upper[i-1]
        d[i]    -= m * d[i-1]
    u = np.zeros(N)
    u[-1] = d[-1] / main[-1]
    for i in range(N-2, -1, -1):
        u[i] = (d[i] - upper[i] * u[i+1]) / main[i]
    return u

def _max_rel(u: np.ndarray, ref: np.ndarray) -> float:
    return (float(np.max(np.abs(u - ref)))
            / (float(np.max(np.abs(ref))) + 1e-300) * 100.0)


# ============================================================================
#  Solver adapters
# ============================================================================

def _call_hhl(A: np.ndarray, b: np.ndarray):
    from solvers.quantum.hhl_1d import hhl_solve_system
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = hhl_solve_system(A, b, HHL_EPSILON)
        wall = time.perf_counter() - t0
        u = np.asarray(result[0], dtype=float)
        c = float(result[2]) if len(result) > 2 else float("nan")
        return u, wall, {"prop_const": c}
    except Exception as e:
        return None, time.perf_counter()-t0, {"error": str(e)}

def _call_vqls(A: np.ndarray, b: np.ndarray):
    from solvers.quantum.vqls_1d import vqls_solve_system
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = vqls_solve_system(A, b)
        wall = time.perf_counter() - t0
        u = np.asarray(result.u, dtype=float)
        cost = float(getattr(result, "final_cost", float("nan")))
        return u, wall, {"final_cost": cost}
    except Exception as e:
        return None, time.perf_counter()-t0, {"error": str(e)}

def _call_qsvt(A: np.ndarray, b: np.ndarray):
    from solvers.quantum.qsvt_1d import qsvt_solve_system, QSVTConfig1D
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = QSVTConfig1D(angle_method="auto", verbose=False,
                               max_degree=500)
            result = qsvt_solve_system(A, b, config=cfg)
        wall = time.perf_counter() - t0
        u = np.asarray(result.u, dtype=float)
        return u, wall, {
            "polynomial_degree": getattr(result, "polynomial_degree", None),
            "circuit_depth":     getattr(result, "circuit_depth", None),
        }
    except Exception as e:
        return None, time.perf_counter()-t0, {"error": str(e)}

ADAPTERS = {"hhl": _call_hhl, "vqls": _call_vqls, "qsvt": _call_qsvt}


# ============================================================================
#  SOR solvers — generic Poisson (unit square)
# ============================================================================

def sor_2d_thomas(N: int, f_vals: np.ndarray, dx: float,
                  tol: float = SOR_TOL, max_iter: int = SOR_MAX_ITER,
                  verbose: bool = False
                  ) -> tuple[np.ndarray, int, bool, list[float]]:
    """
    2D Line-SOR with Thomas inner solver on the unit square.

    SOR update per strip:
        phi[:,j] = omega * phi_thomas[:,j] + (1-omega) * phi[:,j]

    Gauss-Seidel ordering: strips 0..j-1 already updated this sweep,
    so their new values are used immediately in the RHS of strip j.

    Convergence: relative delta = max|phi_new - phi_old| / max|phi| < tol.
    This is N-independent and tracks the discretisation error correctly.
    """
    omega = _sor_omega(N)
    A, kappa = _build_row_matrix_square(N, dx)
    phi = np.zeros((N, N))
    deltas = []

    if verbose:
        print(f"    Thomas-SOR: N={N}  omega={omega:.4f}  kappa={kappa:.4f}")

    for it in range(max_iter):
        phi_old = phi.copy()

        for j in range(N):
            b = _build_rhs_row_square(j, phi, f_vals, dx)
            phi_j = _thomas_solve(A, b)
            phi[:, j] = omega * phi_j + (1.0 - omega) * phi[:, j]

        delta = float(np.max(np.abs(phi - phi_old)))
        phi_scale = float(np.max(np.abs(phi)))
        rel_delta = delta / phi_scale if phi_scale > 0.0 else delta
        deltas.append(rel_delta)

        if verbose and (it + 1) % 10 == 0:
            print(f"    Thomas-SOR iter {it+1:4d}  rel_Δ={rel_delta:.3e}")

        if rel_delta < tol:
            return phi, it + 1, True, deltas

    return phi, max_iter, False, deltas


def sor_2d_quantum(N: int, f_vals: np.ndarray, dx: float,
                   solver_name: str,
                   phi_thomas: np.ndarray,
                   u_exact: np.ndarray,
                   tol: float = SOR_TOL,
                   max_iter: int = SOR_MAX_ITER,
                   print_every: int = 5
                   ) -> tuple[Optional[np.ndarray], int, bool, list[float], dict, str]:
    """
    2D Line-SOR with a quantum inner solver on the unit square.

    Same SOR update as sor_2d_thomas. The quantum solver provides
    u_row = A^{-1}b (after proportionality recovery). SOR relaxation
    is applied to the physical solution.

    Returns (phi, n_iters, converged, rel_delta_history, diag, stop_reason).
    """
    omega  = _sor_omega(N)
    A, kappa = _build_row_matrix_square(N, dx)
    adapter  = ADAPTERS[solver_name]
    label    = solver_name.upper()
    is_vqls  = (solver_name == "vqls")

    phi = np.zeros((N, N))
    diag = {"kappa": kappa, "omega": omega,
            "costs": [], "degrees": [], "depths": [], "prop_consts": []}
    deltas: list[float] = []
    best_delta = float("inf")
    no_improve = 0
    stop_reason = "max_iter"

    print(f"\n  {_C}{'─'*60}{_X}")
    print(f"  {_B}{label}-2D{_X}  N={N}  κ={kappa:.4f}  "
          f"ω={omega:.4f}  tol={tol:.0e}  max_iter={max_iter}")
    print(f"  {'─'*60}")

    for it in range(max_iter):
        phi_old = phi.copy()

        for j in range(N):
            b = _build_rhs_row_square(j, phi, f_vals, dx)
            result = adapter(A, b)
            u_row, wall_row, extra_row = result

            if u_row is None:
                err = extra_row.get("error", "?")
                print(f"  {_R}[FAIL] iter {it+1} row {j}: {err}{_X}")
                return phi, it+1, False, deltas, diag, "solver_failure"

            u_row = np.asarray(u_row, dtype=float)
            if u_row.shape != (N,):
                print(f"  {_R}[SHAPE] iter {it+1} row {j}: {u_row.shape}{_X}")
                return phi, it+1, False, deltas, diag, "shape_error"

            # SOR relaxation
            phi[:, j] = omega * u_row + (1.0 - omega) * phi[:, j]

            # Collect metadata
            if solver_name == "hhl":
                diag["prop_consts"].append(extra_row.get("prop_const", float("nan")))
            elif solver_name == "vqls":
                diag["costs"].append(extra_row.get("final_cost", float("nan")))
            elif solver_name == "qsvt":
                diag["degrees"].append(extra_row.get("polynomial_degree"))
                diag["depths"].append(extra_row.get("circuit_depth"))

        delta = float(np.max(np.abs(phi - phi_old)))
        phi_scale = float(np.max(np.abs(phi)))
        rel_delta = delta / phi_scale if phi_scale > 0.0 else delta
        deltas.append(rel_delta)

        if (it + 1) % print_every == 0:
            err_e = _max_rel(phi, u_exact)
            err_t = _max_rel(phi, phi_thomas)
            col = _G if err_e < 5.0 else (_Y if err_e < 20.0 else _R)
            print(f"  iter {it+1:4d}  rel_Δ={rel_delta:.3e}  "
                  f"vs_exact={col}{err_e:7.3f}%{_X}  "
                  f"vs_thomas={err_t:7.3f}%")

            # VQLS noise-floor stop
            if is_vqls and err_t < VQLS_THOMAS_TOL * 100.0:
                print(f"  {_G}[VQLS] vs_thomas={err_t:.3f}% < threshold{_X}")
                return phi, it+1, True, deltas, diag, "vqls_noise_floor"

            # Early stopping on relative delta
            if rel_delta < best_delta * (1.0 - EARLY_STOP_MIN_IMPROVEMENT):
                best_delta = rel_delta
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    print(f"  {_Y}[EARLY STOP] rel_Δ stagnated{_X}")
                    converged = _max_rel(phi, phi_thomas) < 2.0
                    return phi, it+1, converged, deltas, diag, "early_stop_stagnation"

        if rel_delta < tol:
            print(f"  {_G}Converged at iter {it+1}  (rel_Δ={rel_delta:.3e}){_X}")
            return phi, it+1, True, deltas, diag, "tol_met"

    print(f"  {_R}Max iterations reached. Final rel_Δ={deltas[-1]:.3e}{_X}")
    return phi, max_iter, False, deltas, diag, stop_reason


# ============================================================================
#  HET MMS problem (non-square domain)
# ============================================================================

def build_grid_2d_het(Nz: int, Nr: int):
    dz = HET_Lz / (Nz + 1)
    dr = HET_Lr / (Nr + 1)
    z_pts = np.arange(1, Nz+1) * dz
    r_pts = np.arange(1, Nr+1) * dr
    z, r = np.meshgrid(z_pts, r_pts, indexing="ij")
    return z, r, dz, dr

def phi_het_mms(z, r):
    return HET_phi0 * np.sin(np.pi*z/HET_Lz) * np.cos(np.pi*r/(2*HET_Lr))

def f_het_mms(z, r):
    """f = nabla^2(phi_MMS). Negative coefficient. Verified O(h^2)."""
    coeff = -HET_phi0 * np.pi**2 * (1.0/HET_Lz**2 + 1.0/(4.0*HET_Lr**2))
    return coeff * np.sin(np.pi*z/HET_Lz) * np.cos(np.pi*r/(2*HET_Lr))

def _build_row_matrix_het(Nz: int, dz: float, dr: float):
    """
    Row matrix for HET domain (dz != dr).
    Physical system: A*phi[:,j] = b where A encodes 1/dz^2 coefficients.
    Main diag: -2*(1/dz^2 + 1/dr^2), off-diag: 1/dz^2.
    """
    a = -2.0 * (1.0/dz**2 + 1.0/dr**2)
    b = 1.0 / dz**2
    A = (a * np.eye(Nz)
         + b * (np.diag(np.ones(Nz-1), 1) + np.diag(np.ones(Nz-1), -1)))
    eigs = np.abs(np.linalg.eigvalsh(A))
    return A, float(eigs.max() / eigs.min())

def _build_rhs_row_het(j: int, phi: np.ndarray, f_vals: np.ndarray,
                        dz: float, dr: float,
                        bc_inner: np.ndarray,
                        bc_anode: np.ndarray,
                        bc_cathode: np.ndarray) -> np.ndarray:
    """RHS for j-th radial strip of the HET problem."""
    b = f_vals[:, j].copy()
    if j > 0:
        b -= phi[:, j-1] / dr**2
    else:
        b -= bc_inner / dr**2
    if j < phi.shape[1] - 1:
        b -= phi[:, j+1] / dr**2
    b[0]  -= bc_anode[j]   / dz**2
    b[-1] -= bc_cathode[j] / dz**2
    return b

def sor_2d_het_thomas(Nz: int, Nr: int, f_vals: np.ndarray,
                       dz: float, dr: float,
                       bc_inner: np.ndarray,
                       bc_anode: np.ndarray,
                       bc_cathode: np.ndarray,
                       tol: float = SOR_TOL,
                       max_iter: int = SOR_MAX_ITER,
                       verbose: bool = False
                       ) -> tuple[np.ndarray, int, bool, list[float]]:
    """2D Line-SOR with Thomas inner solver for the HET domain."""
    omega = _sor_omega(min(Nz, Nr))
    A, kappa = _build_row_matrix_het(Nz, dz, dr)
    phi = np.zeros((Nz, Nr))
    deltas: list[float] = []

    if verbose:
        print(f"    Thomas-HET-SOR: Nz={Nz}  Nr={Nr}  "
              f"omega={omega:.4f}  kappa={kappa:.4f}")

    for it in range(max_iter):
        phi_old = phi.copy()
        for j in range(Nr):
            b = _build_rhs_row_het(j, phi, f_vals, dz, dr,
                                    bc_inner, bc_anode, bc_cathode)
            phi_j = _thomas_solve(A, b)
            phi[:, j] = omega * phi_j + (1.0 - omega) * phi[:, j]

        delta = float(np.max(np.abs(phi - phi_old)))
        phi_scale = float(np.max(np.abs(phi)))
        rel_delta = delta / phi_scale if phi_scale > 0.0 else delta
        deltas.append(rel_delta)

        if verbose and (it + 1) % 20 == 0:
            print(f"    Thomas-HET-SOR iter {it+1:4d}  rel_Δ={rel_delta:.3e}")

        if rel_delta < tol:
            return phi, it+1, True, deltas

    return phi, max_iter, False, deltas

def sor_2d_het_quantum(Nz: int, Nr: int, f_vals: np.ndarray,
                        dz: float, dr: float,
                        bc_inner: np.ndarray,
                        bc_anode: np.ndarray,
                        bc_cathode: np.ndarray,
                        phi_exact: np.ndarray,
                        phi_thomas: np.ndarray,
                        solver_name: str,
                        tol: float = SOR_TOL,
                        max_iter: int = SOR_MAX_ITER,
                        print_every: int = 10
                        ) -> tuple:
    """2D Line-SOR with quantum inner solver for the HET domain."""
    omega = _sor_omega(min(Nz, Nr))
    A, kappa = _build_row_matrix_het(Nz, dz, dr)
    adapter  = ADAPTERS[solver_name]
    label    = solver_name.upper()
    is_vqls  = (solver_name == "vqls")

    phi = np.zeros((Nz, Nr))
    diag = {"kappa": kappa, "omega": omega,
            "costs": [], "degrees": [], "depths": [], "prop_consts": []}
    deltas: list[float] = []
    best_delta = float("inf")
    no_improve = 0
    stop_reason = "max_iter"

    print(f"\n  {_C}{'─'*60}{_X}")
    print(f"  {_B}{label}-2D-HET{_X}  Nz={Nz}  Nr={Nr}  "
          f"κ={kappa:.4f}  ω={omega:.4f}")
    print(f"  {'─'*60}")

    for it in range(max_iter):
        phi_old = phi.copy()

        for j in range(Nr):
            b = _build_rhs_row_het(j, phi, f_vals, dz, dr,
                                    bc_inner, bc_anode, bc_cathode)
            result = adapter(A, b)
            u_row, wall_row, extra_row = result

            if u_row is None:
                print(f"  {_R}[FAIL] iter {it+1} strip {j}: "
                      f"{extra_row.get('error','?')}{_X}")
                return phi, it+1, False, deltas, diag, "solver_failure"

            u_row = np.asarray(u_row, dtype=float)
            if u_row.shape != (Nz,):
                print(f"  {_R}[SHAPE] {u_row.shape}{_X}")
                return phi, it+1, False, deltas, diag, "shape_error"

            phi[:, j] = omega * u_row + (1.0 - omega) * phi[:, j]

            if solver_name == "hhl":
                diag["prop_consts"].append(extra_row.get("prop_const", float("nan")))
            elif solver_name == "vqls":
                diag["costs"].append(extra_row.get("final_cost", float("nan")))
            elif solver_name == "qsvt":
                diag["degrees"].append(extra_row.get("polynomial_degree"))
                diag["depths"].append(extra_row.get("circuit_depth"))

        delta = float(np.max(np.abs(phi - phi_old)))
        phi_scale = float(np.max(np.abs(phi)))
        rel_delta = delta / phi_scale if phi_scale > 0.0 else delta
        deltas.append(rel_delta)

        if (it + 1) % print_every == 0:
            err_e = _max_rel(phi, phi_exact)
            err_t = _max_rel(phi, phi_thomas)
            col = _G if err_e < 5.0 else (_Y if err_e < 20.0 else _R)
            print(f"  iter {it+1:4d}  rel_Δ={rel_delta:.3e}  "
                  f"vs_exact={col}{err_e:7.3f}%{_X}  "
                  f"vs_thomas={err_t:7.3f}%")

            if is_vqls and err_t < VQLS_THOMAS_TOL * 100.0:
                print(f"  {_G}[VQLS] vs_thomas={err_t:.3f}% < threshold{_X}")
                return phi, it+1, True, deltas, diag, "vqls_noise_floor"

            if rel_delta < best_delta * (1.0 - EARLY_STOP_MIN_IMPROVEMENT):
                best_delta = rel_delta; no_improve = 0
            else:
                no_improve += 1
                if no_improve >= EARLY_STOP_PATIENCE:
                    print(f"  {_Y}[EARLY STOP]{_X}")
                    converged = _max_rel(phi, phi_thomas) < 2.0
                    return phi, it+1, converged, deltas, diag, "early_stop_stagnation"

        if rel_delta < tol:
            print(f"  {_G}Converged at iter {it+1}{_X}")
            return phi, it+1, True, deltas, diag, "tol_met"

    print(f"  {_R}Max iterations reached. Final rel_Δ={deltas[-1]:.3e}{_X}")
    return phi, max_iter, False, deltas, diag, stop_reason


# ============================================================================
#  Summary printer
# ============================================================================

def print_summary(label: str, phi: Optional[np.ndarray],
                  u_exact: np.ndarray, u_thomas: np.ndarray,
                  n_iters: int, converged: bool,
                  diag: dict, wall_total: float,
                  stop_reason: str) -> None:
    sep = "═" * 60
    print(f"\n  {_B}{sep}{_X}")
    print(f"  {_B}SUMMARY: {label}{_X}")
    print(f"  {sep}")
    status = f"{_G}CONVERGED{_X}" if converged else f"{_R}NOT CONVERGED{_X}"
    print(f"  Status      : {status}  ({n_iters} iters)  {wall_total:.2f}s")
    print(f"  Stop reason : {stop_reason}")
    omega = diag.get("omega")
    if omega is not None:
        print(f"  SOR omega   : {omega:.4f}")

    if phi is not None and not np.allclose(phi, 0.0):
        err_e = _max_rel(phi, u_exact)
        err_t = _max_rel(phi, u_thomas)
        err_thomas_e = _max_rel(u_thomas, u_exact)
        rms_e = float(np.sqrt(np.mean((phi - u_exact)**2)))
        col_e = _G if err_e < 5.0 else (_Y if err_e < 20.0 else _R)
        col_t = _G if err_t < 1.0 else (_Y if err_t < 5.0 else _R)
        print(f"  MaxRelErr vs exact  : {col_e}{err_e:8.3f}%{_X}")
        print(f"  MaxRelErr vs Thomas : {col_t}{err_t:8.3f}%{_X}")
        print(f"  RMS error vs exact  : {rms_e:.3e}")
        print(f"\n  {_B}Error decomposition:{_X}")
        print(f"    Thomas vs exact (h²-discretisation): "
              f"{_Y}{err_thomas_e:.3f}%{_X}  ← irreducible at this N")
        print(f"    {label} vs Thomas (quantum error):   "
              f"{col_t}{err_t:.3f}%{_X}")
        print(f"    {label} vs exact  (total):           "
              f"{col_e}{err_e:.3f}%{_X}")
    else:
        print(f"  {_R}Solution is zero or None.{_X}")

    costs = [c for c in diag.get("costs", [])
             if c is not None and not np.isnan(c)]
    if costs:
        c = np.array(costs)
        col = _G if c.max() < 1e-4 else (_Y if c.max() < 1e-2 else _R)
        print(f"  VQLS cost   : mean={c.mean():.2e}  "
              f"max={col}{c.max():.2e}{_X}")

    degrees = [d for d in diag.get("degrees", []) if d is not None]
    if degrees:
        d = np.array(degrees)
        print(f"  QSVT degree : mean={d.mean():.0f}  max={d.max():.0f}")

    props = [p for p in diag.get("prop_consts", [])
             if p is not None and not np.isnan(p)]
    if props:
        p = np.array(props)
        print(f"  HHL c (prop): mean={p.mean():.4f}  std={p.std():.4f}")

    print(f"  {sep}")


# ============================================================================
#  Plotting
# ============================================================================

def plot_solutions(N, x, y, u_exact, results_dict):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        return
    valid = {k: v for k, v in results_dict.items()
             if v is not None and not np.allclose(v, 0.0)}
    n_cols = 1 + len(valid)
    fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 7))
    if n_cols == 1: axes = axes.reshape(2, 1)
    vmin, vmax = u_exact.min(), u_exact.max()
    im = axes[0,0].pcolormesh(x, y, u_exact, cmap="RdBu_r",
                               vmin=vmin, vmax=vmax, shading="auto")
    axes[0,0].set_title("Exact", fontweight="bold")
    axes[0,0].set_aspect("equal")
    plt.colorbar(im, ax=axes[0,0], shrink=0.8)
    axes[1,0].axis("off")
    for ci, (name, phi) in enumerate(valid.items(), 1):
        im = axes[0,ci].pcolormesh(x, y, phi, cmap="RdBu_r",
                                    vmin=vmin, vmax=vmax, shading="auto")
        axes[0,ci].set_title(name, fontweight="bold")
        axes[0,ci].set_aspect("equal")
        plt.colorbar(im, ax=axes[0,ci], shrink=0.8)
        err = phi - u_exact
        abs_max = max(np.abs(err).max(), 1e-12)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        im2 = axes[1,ci].pcolormesh(x, y, err, cmap="seismic",
                                     norm=norm, shading="auto")
        axes[1,ci].set_title(f"Error ({_max_rel(phi, u_exact):.2f}%)")
        axes[1,ci].set_aspect("equal")
        plt.colorbar(im2, ax=axes[1,ci], shrink=0.8)
    fig.suptitle(f"2D Poisson Debug (SOR) — N={N}", fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / f"debug_2d_solutions_N{N}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  {_G}Solutions: {out}{_X}")
    plt.close(fig)

def plot_convergence(errors_by_solver, N):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colours = {"Thomas": "black", "HHL": "royalblue",
               "VQLS": "darkorange", "QSVT": "crimson"}
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, errs in errors_by_solver.items():
        if errs:
            ax.semilogy(range(1, len(errs)+1), errs,
                        label=name, color=colours.get(name, "grey"), lw=1.8)
    ax.set_xlabel("SOR Iteration")
    ax.set_ylabel("Relative Δ  (delta / max|phi|)")
    ax.set_title(f"2D Line-SOR Convergence — N={N}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / f"debug_2d_convergence_N{N}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  {_G}Convergence: {out}{_X}")
    plt.close(fig)


# ============================================================================
#  HET 2D case runner
# ============================================================================

def run_het_2d_case(Nz: int = 4, Nr: int = 4,
                    solvers_to_run: Optional[list] = None,
                    max_iter: int = 500, tol: float = SOR_TOL,
                    print_every: int = 10, make_plots: bool = False) -> dict:
    if solvers_to_run is None:
        solvers_to_run = ["hhl", "vqls", "qsvt"]

    print(f"\n{_B}{'═'*64}{_X}")
    print(f"{_B}  2D HET MMS BENCHMARK (SOR){_X}")
    print(f"{_B}  Nz={Nz}  Nr={Nr}  Lz={HET_Lz*1e3:.0f}mm  "
          f"Lr={HET_Lr*1e3:.0f}mm  phi0={HET_phi0:.0f}V{_X}")
    print(f"{_B}{'═'*64}{_X}")

    z, r, dz, dr = build_grid_2d_het(Nz, Nr)
    phi_exact = phi_het_mms(z, r)
    f_vals    = f_het_mms(z, r)
    z_pts     = np.arange(1, Nz+1) * dz

    bc_anode   = np.zeros(Nr)
    bc_cathode = np.zeros(Nr)
    bc_inner   = HET_phi0 * np.sin(np.pi * z_pts / HET_Lz)

    print(f"\n  MMS: phi(z,r) = {HET_phi0:.0f}·sin(πz/{HET_Lz*1e3:.0f}mm)"
          f"·cos(πr/{2*HET_Lr*1e3:.0f}mm)")
    print(f"  max|phi_exact| = {np.max(np.abs(phi_exact)):.4f} V")
    print(f"  max|f_source|  = {np.max(np.abs(f_vals)):.4e} V/m²")
    print(f"  dz={dz*1e3:.3f}mm  dr={dr*1e3:.3f}mm")

    print(f"\n{_B}  Thomas-HET-2D-SOR (reference){_X}")
    t0 = time.perf_counter()
    phi_thomas, n_th, conv_th, errs_th = sor_2d_het_thomas(
        Nz, Nr, f_vals, dz, dr, bc_inner, bc_anode, bc_cathode,
        tol=tol, max_iter=max_iter, verbose=True)
    t_th = time.perf_counter() - t0

    err_th = _max_rel(phi_thomas, phi_exact)
    col = _G if err_th < 5.0 else (_Y if err_th < 20.0 else _R)
    print(f"  Thomas-HET: {n_th} iters  "
          f"MaxRelErr={col}{err_th:.3f}%{_X}  "
          f"Time={t_th:.3f}s  Converged={conv_th}")

    Ez = -np.gradient(phi_thomas, dz, axis=0)
    Er = -np.gradient(phi_thomas, dr, axis=1)
    E_peak = np.sqrt(Ez**2 + Er**2).max()
    print(f"  Peak |E| Thomas = {E_peak:.3e} V/m  "
          f"(expected ~{HET_phi0*np.pi/HET_Lz:.3e} V/m)")

    results = {"Thomas": phi_thomas}
    errors_by_solver = {"Thomas": errs_th}

    for sname in solvers_to_run:
        label  = sname.upper()
        t0 = time.perf_counter()
        phi_q, n_q, conv_q, errs_q, diag_q, stop_q = sor_2d_het_quantum(
            Nz, Nr, f_vals, dz, dr, bc_inner, bc_anode, bc_cathode,
            phi_exact, phi_thomas, sname,
            tol=tol, max_iter=max_iter, print_every=print_every)
        wall_q = time.perf_counter() - t0

        print_summary(label+"-HET", phi_q, phi_exact, phi_thomas,
                      n_q, conv_q, diag_q, wall_q, stop_q)

        if phi_q is not None and not np.allclose(phi_q, 0.0):
            Ez_q = -np.gradient(phi_q, dz, axis=0)
            Er_q = -np.gradient(phi_q, dr, axis=1)
            print(f"  Peak |E| {label} = "
                  f"{np.sqrt(Ez_q**2+Er_q**2).max():.3e} V/m")

        results[label] = phi_q
        errors_by_solver[label] = errs_q

    # Final table
    print(f"\n{_B}{'═'*64}{_X}")
    print(f"{_B}  HET-2D FINAL TABLE{_X}")
    print(f"{'─'*64}")
    print(f"  {'Solver':<12} {'Iters':>6} {'MaxRelErr%':>12} {'vs Thomas%':>12}")
    print(f"{'─'*64}")
    for name, phi in results.items():
        if phi is None or np.allclose(phi, 0.0):
            print(f"  {name:<12} {'—':>6} {'FAILED':>12} {'—':>12}")
            continue
        err_e = _max_rel(phi, phi_exact)
        err_t = _max_rel(phi, phi_thomas)
        col_e = _G if err_e < 5.0 else (_Y if err_e < 20.0 else _R)
        n_it  = n_th if name == "Thomas" else "—"
        print(f"  {name:<12} {str(n_it):>6} "
              f"{col_e}{err_e:>11.3f}%{_X} {err_t:>11.3f}%")
    print(f"{'═'*64}\n")

    if make_plots:
        _plot_het_2d(z, r, phi_exact, results, errors_by_solver, Nz, Nr)

    return results

def _plot_het_2d(z, r, phi_exact, results, errors_by_solver, Nz, Nr):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        return
    valid = {k: v for k, v in results.items()
             if v is not None and not np.allclose(v, 0.0)}
    n_cols = 1 + len(valid)
    fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 7))
    if n_cols == 1: axes = axes.reshape(2, 1)
    vmin, vmax = phi_exact.min(), phi_exact.max()
    z_mm = z*1e3; r_mm = r*1e3
    im = axes[0,0].pcolormesh(z_mm, r_mm, phi_exact.T, cmap="RdBu_r",
                               vmin=vmin, vmax=vmax, shading="auto")
    axes[0,0].set_title("MMS Exact", fontweight="bold")
    axes[0,0].set_xlabel("z (mm)"); axes[0,0].set_ylabel("r (mm)")
    plt.colorbar(im, ax=axes[0,0], shrink=0.8, label="φ (V)")
    axes[1,0].axis("off")
    for ci, (name, phi) in enumerate(valid.items(), 1):
        im = axes[0,ci].pcolormesh(z_mm, r_mm, phi.T, cmap="RdBu_r",
                                    vmin=vmin, vmax=vmax, shading="auto")
        axes[0,ci].set_title(name, fontweight="bold")
        axes[0,ci].set_xlabel("z (mm)"); axes[0,ci].set_ylabel("r (mm)")
        plt.colorbar(im, ax=axes[0,ci], shrink=0.8, label="φ (V)")
        err = phi - phi_exact
        abs_max = max(np.abs(err).max(), 1e-12)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        im2 = axes[1,ci].pcolormesh(z_mm, r_mm, err.T, cmap="seismic",
                                     norm=norm, shading="auto")
        axes[1,ci].set_title(f"Error ({_max_rel(phi, phi_exact):.2f}%)")
        axes[1,ci].set_xlabel("z (mm)"); axes[1,ci].set_ylabel("r (mm)")
        plt.colorbar(im2, ax=axes[1,ci], shrink=0.8, label="Δφ (V)")
    fig.suptitle(f"2D HET MMS (SOR) — Nz={Nz}  Nr={Nr}", fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / f"debug_het2d_solutions_Nz{Nz}_Nr{Nr}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  {_G}HET solutions: {out}{_X}")
    plt.close(fig)
    colours = {"Thomas":"black","HHL":"royalblue",
               "VQLS":"darkorange","QSVT":"crimson"}
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, errs in errors_by_solver.items():
        if errs:
            ax.semilogy(range(1, len(errs)+1), errs,
                        label=name, color=colours.get(name,"grey"), lw=1.8)
    ax.set_xlabel("SOR Iteration")
    ax.set_ylabel("Relative Δ")
    ax.set_title(f"2D HET MMS Convergence (SOR) — Nz={Nz}  Nr={Nr}")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / f"debug_het2d_convergence_Nz{Nz}_Nr{Nr}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  {_G}HET convergence: {out}{_X}")
    plt.close(fig)


# ============================================================================
#  Introspection
# ============================================================================

def introspect_solvers():
    print(f"\n{_B}SOLVER INTROSPECTION{_X}\n{'='*60}")
    specs = {"HHL":  ("solvers.quantum.hhl_1d",  "hhl_solve_system"),
             "VQLS": ("solvers.quantum.vqls_1d", "vqls_solve_system"),
             "QSVT": ("solvers.quantum.qsvt_1d", "qsvt_solve_system")}
    N = 4
    A = -4.0*np.eye(N) + np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1)
    b = np.array([0.04, 0.04, 0.04, 0.04])
    for label, (mod, fn) in specs.items():
        print(f"\n{_C}--- {label} ---{_X}")
        try:
            import importlib
            m = importlib.import_module(mod); f = getattr(m, fn)
            print(f"  Signature: {fn}{inspect.signature(f)}")
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = f(A, b) if label != "HHL" else f(A, b, 0.01)
                print(f"  Return: {type(r)}")
                if hasattr(r, "__dict__"):
                    print(f"  Attrs: {list(r.__dict__.keys())}")
                elif isinstance(r, (tuple, list)):
                    for i, v in enumerate(r):
                        print(f"    [{i}] {type(v).__name__} "
                              f"{getattr(v,'shape','')}")
            except Exception as e:
                print(f"  {_R}{e}{_X}")
        except Exception as e:
            print(f"  {_R}{e}{_X}")
    print("\n" + "="*60 + "\n")


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Debug tool for 2D quantum Poisson solvers (SOR version).")
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--tol", type=float, default=SOR_TOL)
    parser.add_argument("--solver", default="all",
                        choices=["all", "hhl", "vqls", "qsvt", "thomas", "het"])
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--introspect", action="store_true")
    args = parser.parse_args()

    if args.introspect:
        introspect_solvers(); return

    N = args.N
    print(f"\n{_B}{'═'*64}{_X}")
    print(f"{_B}  2D QUANTUM POISSON SOLVER DEBUG TOOL  (v5 — SOR){_X}")
    print(f"{_B}  N={N}  tol={args.tol:.0e}  HHL_ε={HHL_EPSILON}{_X}")
    print(f"{_B}  omega_SOR={_sor_omega(N):.4f}{_X}")
    print(f"{_B}  Output: {OUT_DIR}{_X}")
    print(f"{_B}{'═'*64}{_X}")

    _QUANTUM_SOLVERS = ["hhl", "vqls", "qsvt"]

    # -- Generic Poisson (unit square) -----------------------------------------
    if args.solver != "het":
        x, y, dx = build_grid_2d(N)
        f_vals  = f_sin2d(x, y)
        u_exact = u_exact_sin2d(x, y)
        _, kappa = _build_row_matrix_square(N, dx)
        print(f"\n  Grid: {N}×{N}  h={dx:.4f}  κ(A_row)={kappa:.4f}")
        print(f"  max|u_exact|={np.max(np.abs(u_exact)):.6f}")

        print(f"\n{_B}  Thomas-2D-SOR (reference){_X}")
        t0 = time.perf_counter()
        u_thomas, n_th, conv_th, errs_th = sor_2d_thomas(
            N, f_vals, dx, tol=args.tol, max_iter=args.max_iter, verbose=True)
        t_th = time.perf_counter() - t0
        err_th = _max_rel(u_thomas, u_exact)
        col = _G if err_th < 5.0 else _R
        print(f"  Thomas: {n_th} iters  MaxRelErr={col}{err_th:.3f}%{_X}  "
              f"Time={t_th:.3f}s  Converged={conv_th}")

        results_dict = {"Thomas": u_thomas}
        errors_by_solver = {"Thomas": errs_th}

        generic_solvers = (
            _QUANTUM_SOLVERS if args.solver == "all"
            else ([args.solver] if args.solver in _QUANTUM_SOLVERS else [])
        )

        for sname in generic_solvers:
            label = sname.upper()
            t0 = time.perf_counter()
            phi, n_q, conv_q, errs_q, diag_q, stop_q = sor_2d_quantum(
                N, f_vals, dx, sname, u_thomas, u_exact,
                tol=args.tol, max_iter=args.max_iter,
                print_every=args.print_every)
            wall_q = time.perf_counter() - t0
            print_summary(label, phi, u_exact, u_thomas,
                          n_q, conv_q, diag_q, wall_q, stop_q)
            results_dict[label] = phi
            errors_by_solver[label] = errs_q

        # Final table
        print(f"\n{_B}{'═'*64}{_X}")
        print(f"{_B}  FINAL TABLE — Generic Poisson{_X}")
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
                  f"{col_e}{err_e:>11.3f}%{_X} {err_t:>11.3f}% {str(cv):>8}")
        print(f"{'═'*64}\n")

        if args.plot:
            plot_solutions(N, x, y, u_exact, results_dict)
            plot_convergence(errors_by_solver, N)

    # -- HET 2D MMS case -------------------------------------------------------
    if args.solver in ("all", "het"):
        run_het_2d_case(
            Nz=N, Nr=N,
            solvers_to_run=_QUANTUM_SOLVERS,
            max_iter=args.max_iter,
            tol=args.tol,
            print_every=args.print_every,
            make_plots=args.plot,
        )


if __name__ == "__main__":
    main()