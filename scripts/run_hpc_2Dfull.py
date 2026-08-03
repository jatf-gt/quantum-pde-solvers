#!/usr/bin/env python3
"""
Implements
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import multiprocessing as mp
from scipy.interpolate import RegularGridInterpolator

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ============================================================================
#  Output directory and logging
# ============================================================================

RESULTS_DIR = Path("results") / "2Dhpc_run"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = RESULTS_DIR / "run.log"
_IS_MAIN_PROCESS = mp.current_process().name == "MainProcess"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  pid=%(process)-6d  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w" if _IS_MAIN_PROCESS else "a"),
    ],
)
log = logging.getLogger(__name__)

for _noisy in ("qiskit.transpiler", "qiskit_aer", "qiskit_ibm_runtime",
               "stevedore", "qiskit.passmanager"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)


# ============================================================================
#  Sweep configuration
# ============================================================================

N_VALUES_ALL: list[int] = [4, 8, 16, 32, 64]

# QSVT degree caps per N for the 2D row matrix.
# Unit square kappas: 2.36 (N=4), 2.77 (N=8), 2.94 (N=16), 2.98 (N=32), 3.00 (N=64)
# HET kappas:        1.92 (N=4), 1.97 (N=8), 1.99 (N=16), 2.00 (N=32), 2.00 (N=64)
# All kappas ~2-3, so degree is small (~30-60). None = uncapped for N<=16.
QSVT_MAX_DEGREE_2D: dict[int, Optional[int]] = {
    4:  None,
    8:  None,
    16: None,
    32: 500,
    64: 500,
}

VQLS_THOMAS_TOL:          float = 0.005
EARLY_STOP_MIN_IMPROVEMENT: float = 0.01
EARLY_STOP_PATIENCE:        int   = 4
JACOBI_TOL:                 float = 1e-8
JACOBI_MAX_ITER:            int   = 1000
HHL_EPSILON:                float = 0.01
VQLS_SEED:                  int   = 42

# Fourier reference: number of modes and fine-grid resolution
N_FOURIER_MODES: int = 50
# Fine grid for Fourier reference computation (independent of solver N).
# Must be fine enough to resolve the Gaussian peaks (sigma=0.1*Lx=1mm).
# N_FINE=200 gives dx_fine=50um << sigma=1mm: quadrature error < 0.01%.
N_FINE: int = 200

HET_Lz:   float = 0.025
HET_Lr:   float = 0.020
HET_phi0: float = 300.0

MAX_WORKERS_DEFAULT: int = 4
_USE_GPU: bool = os.environ.get("QUANTUM_PDE_USE_GPU", "1") != "0"


# ============================================================================
#  Result dataclass
# ============================================================================

@dataclass
class RunResult2D:
    case:            str
    solver:          str
    N:               int
    kappa_row:       float
    max_rel_err:     Optional[float]
    max_abs_err:     Optional[float]
    residual:        Optional[float]
    wall_time_s:     float
    converged:       bool
    n_jacobi_iters:  int
    notes:           str = ""
    rel_l2_err:      Optional[float] = None
    rms_err:         Optional[float] = None
    vqls_final_cost: Optional[float] = None
    qsvt_degree:     Optional[int]   = None
    qsvt_depth:      Optional[int]   = None
    hhl_scale_c:     Optional[float] = None
    stop_reason:     str = ""


# ============================================================================
#  Logging helpers
# ============================================================================

def _banner(msg: str) -> None:
    sep = "=" * 72
    log.info(sep); log.info("  %s", msg); log.info(sep)

def _section(msg: str) -> None:
    sep = "-" * 72
    log.info(sep); log.info("  %s", msg); log.info(sep)


# ============================================================================
#  Error metrics
# ============================================================================

def _max_rel(u: np.ndarray, ref: np.ndarray, tol: float = 1e-10) -> float:
    mask = np.abs(ref) > tol
    if not np.any(mask): return float("nan")
    return float(np.max(np.abs(u[mask] - ref[mask]) / np.abs(ref[mask])) * 100.0)

def _max_abs(u: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(u - ref)))

def _rel_l2(u: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(u - ref) / (np.linalg.norm(ref) + 1e-300))

def _rms(u: np.ndarray, ref: np.ndarray) -> float:
    return float(np.sqrt(np.mean((u - ref)**2)))

def _accuracy(u: np.ndarray, ref: Optional[np.ndarray]) -> dict:
    if ref is None: return {}
    return {"max_rel_err": _max_rel(u, ref), "max_abs_err": _max_abs(u, ref),
            "rel_l2_err": _rel_l2(u, ref), "rms_err": _rms(u, ref)}


# ============================================================================
#  Grid and matrix builders
# ============================================================================

def _grid_2d(N: int, Lx: float = 1.0, Ly: float = 1.0):
    dx = Lx/(N+1); dy = Ly/(N+1)
    x_pts = np.arange(1, N+1)*dx; y_pts = np.arange(1, N+1)*dy
    x, y = np.meshgrid(x_pts, y_pts, indexing="ij")
    return x, y, dx, dy

def _build_row_matrix(N: int, dx: float, dy: float) -> tuple[np.ndarray, float]:
    a = -2.0*(1.0/dx**2 + 1.0/dy**2); b = 1.0/dx**2
    A = a*np.eye(N) + b*(np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1))
    eigs = np.abs(np.linalg.eigvalsh(A))
    return A, float(eigs.max()/eigs.min())

def _build_rhs_row(j, phi, f_vals, dx, dy,
                   bc_left=None, bc_right=None,
                   bc_bottom=None, bc_top=None):
    b = f_vals[:, j].copy()
    if j > 0:       b -= phi[:, j-1] / dy**2
    else:
        if bc_bottom is not None: b -= bc_bottom / dy**2
    if j < phi.shape[1]-1: b -= phi[:, j+1] / dy**2
    else:
        if bc_top is not None: b -= bc_top / dy**2
    if bc_left  is not None: b[0]  -= bc_left[j]  / dx**2
    if bc_right is not None: b[-1] -= bc_right[j] / dx**2
    return b


# ============================================================================
#  Classical Thomas solver
# ============================================================================

def _thomas_1d_general(A, b):
    N = len(b)
    main = A.diagonal(0).copy(); upper = A.diagonal(1).copy()
    lower = A.diagonal(-1).copy(); d = b.copy()
    for i in range(1, N):
        m = lower[i-1]/main[i-1]; main[i] -= m*upper[i-1]; d[i] -= m*d[i-1]
    u = np.zeros(N); u[-1] = d[-1]/main[-1]
    for i in range(N-2, -1, -1): u[i] = (d[i] - upper[i]*u[i+1])/main[i]
    return u

def jacobi_2d_thomas(N, f_vals, dx, dy,
                     bc_left=None, bc_right=None, bc_bottom=None, bc_top=None,
                     tol=JACOBI_TOL, max_iter=JACOBI_MAX_ITER):
    phi = np.zeros((N, N)); A, _ = _build_row_matrix(N, dx, dy)
    for it in range(max_iter):
        phi_new = phi.copy()
        for j in range(N):
            b = _build_rhs_row(j, phi, f_vals, dx, dy,
                               bc_left, bc_right, bc_bottom, bc_top)
            phi_new[:, j] = _thomas_1d_general(A, b)
        delta = float(np.max(np.abs(phi_new - phi))); phi = phi_new
        if delta < tol: return phi, it+1, True
    return phi, max_iter, False


# ============================================================================
#  Quantum solver wrappers
# ============================================================================

def _get_qsvt_config(N: int, kappa: float):
    """Build QSVTConfig1D with degree cap and log what is being used."""
    from solvers.quantum.qsvt_1d import QSVTConfig1D
    max_deg = QSVT_MAX_DEGREE_2D.get(N, 500)

    try:
        declared = {f.name for f in dataclasses.fields(QSVTConfig1D)}
        kwargs = {k: v for k, v in {
            "epsilon": HHL_EPSILON, "angle_method": "auto",
            "max_degree": max_deg,
        }.items() if k in declared}
        return QSVTConfig1D(**kwargs)
    except Exception:
        return QSVTConfig1D()

def _call_hhl_2d(A, b):
    from solvers.quantum.hhl_1d import hhl_solve_system
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        result = hhl_solve_system(A, b, HHL_EPSILON)
    u = np.asarray(result[0], dtype=float)
    c = float(result[2]) if len(result) > 2 else float("nan")
    return u, c

def _call_vqls_2d(A, b):
    from solvers.quantum.vqls_1d import vqls_solve_system
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        result = vqls_solve_system(A, b)
    return np.asarray(result.u, dtype=float), float(getattr(result, "final_cost", float("nan")))

def _call_qsvt_2d(A, b, N: int, kappa: float):
    from solvers.quantum.qsvt_1d import qsvt_solve_system
    cfg = _get_qsvt_config(N, kappa)
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        result = qsvt_solve_system(A, b, config=cfg)
    u = np.asarray(result.u, dtype=float)
    return u, int(getattr(result, "polynomial_degree", -1)), int(getattr(result, "circuit_depth", -1))


# ============================================================================
#  2D Line-Jacobi with quantum inner solver
# ============================================================================

def jacobi_2d_quantum(N, f_vals, dx, dy, solver_name, phi_thomas,
                      bc_left=None, bc_right=None, bc_bottom=None, bc_top=None,
                      tol=1e-6, max_iter=JACOBI_MAX_ITER):
    phi = np.zeros((N, N)); A, kappa = _build_row_matrix(N, dx, dy)
    is_vqls = (solver_name == "vqls")
    extra = {"kappa_row": kappa, "costs": [], "degrees": [], "depths": [],
             "prop_consts": []}
    best_delta = float("inf"); no_improve = 0; stop_reason = "max_iter"

    for it in range(max_iter):
        phi_new = phi.copy()
        for j in range(N):
            b = _build_rhs_row(j, phi, f_vals, dx, dy,
                               bc_left, bc_right, bc_bottom, bc_top)
            try:
                if solver_name == "hhl":
                    u_row, c = _call_hhl_2d(A, b)
                    extra["prop_consts"].append(c)
                elif solver_name == "vqls":
                    u_row, cost = _call_vqls_2d(A, b)
                    extra["costs"].append(cost)
                elif solver_name == "qsvt":
                    u_row, deg, depth = _call_qsvt_2d(A, b, N, kappa)
                    extra["degrees"].append(deg); extra["depths"].append(depth)
                else:
                    raise ValueError(f"Unknown solver: {solver_name}")
            except Exception as exc:
                log.warning("    %s-2D inner solver failed iter %d row %d: %s",
                            solver_name.upper(), it+1, j, exc)
                return phi, it+1, False, extra, "solver_failure"

            u_row = np.asarray(u_row, dtype=float)
            if u_row.shape != (N,):
                return phi, it+1, False, extra, "shape_error"
            phi_new[:, j] = u_row

        delta = float(np.max(np.abs(phi_new - phi))); phi = phi_new

        if is_vqls and it > 0:
            if _max_rel(phi, phi_thomas) < VQLS_THOMAS_TOL * 100.0:
                return phi, it+1, True, extra, "vqls_noise_floor"

        if delta < best_delta*(1.0 - EARLY_STOP_MIN_IMPROVEMENT):
            best_delta = delta; no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                converged = _max_rel(phi, phi_thomas) < 2.0
                return phi, it+1, converged, extra, "early_stop_stagnation"

        if delta < tol:
            return phi, it+1, True, extra, "tol_met"

    return phi, max_iter, False, extra, stop_reason


# ============================================================================
#  Solution archiving
# ============================================================================

def _save_solution_2d(case, solver, N, x, y, phi, phi_exact, f_vals):
    fname = RESULTS_DIR / f"solutions_{case}_{solver}_N{N}.npz"
    arrays = {"x": x, "y": y, "phi_solver": phi, "f_vals": f_vals}
    if phi_exact is not None: arrays["phi_exact"] = phi_exact
    np.savez_compressed(fname, **arrays)

def _record_2d(results, case_id, solver, N, kappa, x, y,
               phi, phi_ref, f_vals, wall, converged, n_iters,
               notes="", stop_reason="", **extra):
    if phi is None:
        results.append(RunResult2D(
            case=case_id, solver=solver, N=N, kappa_row=kappa,
            max_rel_err=None, max_abs_err=None, residual=None,
            wall_time_s=wall, converged=False, n_jacobi_iters=n_iters,
            notes=notes or "solver_error", stop_reason=stop_reason, **extra))
        return
    acc = _accuracy(phi, phi_ref)
    results.append(RunResult2D(
        case=case_id, solver=solver, N=N, kappa_row=kappa,
        max_rel_err=acc.get("max_rel_err"), max_abs_err=acc.get("max_abs_err"),
        residual=None, wall_time_s=wall, converged=converged,
        n_jacobi_iters=n_iters, notes=notes, stop_reason=stop_reason,
        rel_l2_err=acc.get("rel_l2_err"), rms_err=acc.get("rms_err"), **extra))
    _save_solution_2d(case_id, solver, N, x, y, phi, phi_ref, f_vals)


# ============================================================================
#  Per-case runner
# ============================================================================

def _run_all_2d(case_id, N, x, y, dx, dy, f_vals, phi_exact,
                skip_qsvt, results,
                bc_left=None, bc_right=None, bc_bottom=None, bc_top=None):
    _, kappa = _build_row_matrix(N, dx, dy)

    t0 = time.perf_counter()
    phi_T, n_T, conv_T = jacobi_2d_thomas(N, f_vals, dx, dy,
                                           bc_left, bc_right, bc_bottom, bc_top)
    wall_T = time.perf_counter() - t0
    if phi_exact is not None:
        log.info("    Thomas   MaxRelErr=%7.3f%%  iters=%d  time=%.3fs",
                 _max_rel(phi_T, phi_exact), n_T, wall_T)
    else:
        log.info("    Thomas   iters=%d  time=%.3fs  (reference)", n_T, wall_T)
    _record_2d(results, case_id, "Thomas", N, kappa, x, y,
               phi_T, phi_exact, f_vals, wall_T, conv_T, n_T)

    phi_ref = phi_exact if phi_exact is not None else phi_T
    ref_note = "" if phi_exact is not None else "rel_vs_thomas"

    for sname in ("hhl", "vqls", "qsvt"):
        label = sname.upper()
        if sname == "qsvt" and skip_qsvt:
            log.info("    QSVT-2D  skipped (--skip-qsvt)"); continue

        t0 = time.perf_counter()
        phi_q, n_q, conv_q, diag_q, stop_q = jacobi_2d_quantum(
            N, f_vals, dx, dy, sname, phi_T,
            bc_left, bc_right, bc_bottom, bc_top,
            tol=1e-6, max_iter=JACOBI_MAX_ITER)
        wall_q = time.perf_counter() - t0

        if phi_q is not None:
            log.info("    %s-2D  MaxRelErr=%7.3f%%  iters=%d  time=%.2fs  stop=%s",
                     label, _max_rel(phi_q, phi_ref), n_q, wall_q, stop_q)

        extra_kw: dict = {"stop_reason": stop_q, "notes": ref_note}
        if sname == "vqls" and diag_q.get("costs"):
            extra_kw["vqls_final_cost"] = float(np.mean(diag_q["costs"]))
        if sname == "qsvt" and diag_q.get("degrees"):
            extra_kw["qsvt_degree"] = int(np.mean(diag_q["degrees"]))
            extra_kw["qsvt_depth"]  = int(np.mean(diag_q["depths"]))
        if sname == "hhl" and diag_q.get("prop_consts"):
            extra_kw["hhl_scale_c"] = float(np.mean(diag_q["prop_consts"]))

        _record_2d(results, case_id, label, N, kappa, x, y,
                   phi_q, phi_ref, f_vals, wall_q, conv_q, n_q, **extra_kw)


# ============================================================================
#  Problem definitions
# ============================================================================

# ── Section 1: Sinusoidal source ─────────────────────────────────────────────
def _phi_sin2d(x, y):
    return -np.sin(np.pi*x)*np.sin(np.pi*y) / (2.0*np.pi**2)

def _f_sin2d(x, y):
    return np.sin(np.pi*x)*np.sin(np.pi*y)


# ── Section 2: Two-Gaussian charge density ───────────────────────────────────
def _f_two_gaussian_at(x, y, Lx=0.01, Ly=0.01):
    """f = nabla^2(phi) = -rho/eps0 at arbitrary (x,y) arrays."""
    sigma = 0.1*Lx; n0 = 1e16; e = 1.602e-19; eps0 = 8.854e-12
    rho  = n0*e * np.exp(-((x-0.3*Lx)**2 + (y-0.3*Ly)**2) / (2*sigma**2))
    rho += n0*e * np.exp(-((x-0.7*Lx)**2 + (y-0.7*Ly)**2) / (2*sigma**2))
    return -rho / eps0

def _phi_two_gaussian_reference(Lx=0.01, Ly=0.01,
                                 N_fine=N_FINE, N_modes=N_FOURIER_MODES):
    """
    Compute the Fourier series reference solution on a fine grid.

    Key fix vs v2: the Fourier coefficients are computed on a FINE grid
    (N_fine=200, dx_fine=50um << sigma=1mm) independent of the solver N.
    This eliminates the ~32% quadrature error at N=4 that arose from
    integrating over only 4 points per dimension.

    Returns interpolator: phi_ref(x, y) -> scalar or array.
    """
    dx_f = Lx/(N_fine+1); dy_f = Ly/(N_fine+1)
    x_f = np.arange(1, N_fine+1)*dx_f; y_f = np.arange(1, N_fine+1)*dy_f
    xf, yf = np.meshgrid(x_f, y_f, indexing="ij")
    f_fine = _f_two_gaussian_at(xf, yf, Lx, Ly)

    phi_fine = np.zeros((N_fine, N_fine))
    for n in range(1, N_modes+1):
        for m in range(1, N_modes+1):
            sin_x = np.sin(n*np.pi*xf/Lx)
            sin_y = np.sin(m*np.pi*yf/Ly)
            R_nm = (4.0/(Lx*Ly)) * np.sum(f_fine * sin_x * sin_y) * dx_f * dy_f
            # nabla^2(sin_x*sin_y) = -pi^2*(n^2/Lx^2+m^2/Ly^2)*sin_x*sin_y
            denom = -np.pi**2 * (n**2/Lx**2 + m**2/Ly**2)
            phi_fine += (R_nm/denom) * sin_x * sin_y

    # Return a RegularGridInterpolator so we can evaluate at any solver grid
    return RegularGridInterpolator((x_f, y_f), phi_fine,
                                   method="linear", bounds_error=False,
                                   fill_value=0.0)


# ── Section 3: Single-mode Fourier source ────────────────────────────────────
def _f_single_mode(x, y, n=1, m=1, Lx=1.0, Ly=1.0, R_nm=1.0):
    """
    f = nabla^2(phi_exact) = -R_nm * sin(n*pi*x/Lx) * sin(m*pi*y/Ly).
    Derivation: phi = R_nm/(pi^2*(n^2/Lx^2+m^2/Ly^2)) * sin*sin
                nabla^2(phi) = -pi^2*(n^2/Lx^2+m^2/Ly^2) * phi
                             = -R_nm * sin*sin
    """
    return -R_nm * np.sin(n*np.pi*x/Lx) * np.sin(m*np.pi*y/Ly)

def _phi_single_mode(x, y, n=1, m=1, Lx=1.0, Ly=1.0, R_nm=1.0):
    denom = np.pi**2 * (n**2/Lx**2 + m**2/Ly**2)
    return (R_nm/denom) * np.sin(n*np.pi*x/Lx) * np.sin(m*np.pi*y/Ly)


# ── Section 4: HET MMS manufactured solution ─────────────────────────────────
def _phi_het_mms(z, r):
    return HET_phi0 * np.sin(np.pi*z/HET_Lz) * np.cos(np.pi*r/(2*HET_Lr))

def _f_het_mms(z, r):
    """f = nabla^2(phi_MMS). Negative coefficient. Verified O(h^2) convergence."""
    coeff = -HET_phi0 * np.pi**2 * (1.0/HET_Lz**2 + 1.0/(4.0*HET_Lr**2))
    return coeff * np.sin(np.pi*z/HET_Lz) * np.cos(np.pi*r/(2*HET_Lr))


# ── Section 5: HET sinusoidal source ─────────────────────────────────────────
def _phi_het_sin(x, y, Lx, Ly, phi0):
    return phi0 * np.sin(np.pi*x/Lx) * np.sin(np.pi*y/Ly)

def _f_het_sin(x, y, Lx, Ly, phi0):
    coeff = -phi0 * np.pi**2 * (1.0/Lx**2 + 1.0/Ly**2)
    return coeff * np.sin(np.pi*x/Lx) * np.sin(np.pi*y/Ly)


# ============================================================================
#  Case runners
# ============================================================================

def run_section1_single_N(N, skip_qsvt, results):
    _banner(f"SECTION 1 - Generic Poisson, sinusoidal source, N={N}")
    x, y, dx, dy = _grid_2d(N)
    f_vals = _f_sin2d(x, y); phi_ex = _phi_sin2d(x, y)
    log.info("  N=%d  max|phi_exact|=%.6f  max|f|=%.4f",
             N, np.max(np.abs(phi_ex)), np.max(np.abs(f_vals)))
    _run_all_2d("2D_Poisson_sin_hom", N, x, y, dx, dy, f_vals, phi_ex,
                skip_qsvt, results)

def run_section2_single_N(N, skip_qsvt, results):
    """
    Two-Gaussian PlasmaNet benchmark.
    Reference computed on fine grid (N_FINE=200), interpolated to solver grid.
    Precompute kappas needed:
      Unit square: 2.3586 (N=4), 2.7725 (N=8), 2.9352 (N=16),
                   2.9838 (N=32), 2.9960 (N=64)
    """
    _banner(f"SECTION 2 - Two-Gaussian PlasmaNet benchmark, N={N}")
    Lx = Ly = 0.01
    x, y, dx, dy = _grid_2d(N, Lx, Ly)
    f_vals = _f_two_gaussian_at(x, y, Lx, Ly)

    log.info("  Computing Fourier reference on fine grid (N_fine=%d, N_modes=%d)...",
             N_FINE, N_FOURIER_MODES)
    phi_interp = _phi_two_gaussian_reference(Lx, Ly, N_FINE, N_FOURIER_MODES)
    # Evaluate reference at the solver interior grid points
    pts = np.stack([x.ravel(), y.ravel()], axis=-1)
    phi_ex = phi_interp(pts).reshape(N, N)

    log.info("  N=%d  max|phi_ref|=%.4e V  max|f|=%.4e V/m^2",
             N, np.max(np.abs(phi_ex)), np.max(np.abs(f_vals)))
    _run_all_2d("2D_Poisson_TwoGaussian_PlasmaNet", N, x, y, dx, dy,
                f_vals, phi_ex, skip_qsvt, results)

def run_section3_single_N(N, skip_qsvt, results):
    """
    Single-mode Fourier source (n=1, m=1).
    Precompute kappas: same as Section 1 (unit square).
    """
    _banner(f"SECTION 3 - Single-mode Fourier source (n=1,m=1), N={N}")
    n_mode, m_mode, R_nm = 1, 1, 1.0
    x, y, dx, dy = _grid_2d(N)
    f_vals = _f_single_mode(x, y, n_mode, m_mode, 1.0, 1.0, R_nm)
    phi_ex = _phi_single_mode(x, y, n_mode, m_mode, 1.0, 1.0, R_nm)
    log.info("  N=%d  max|phi_exact|=%.6f  max|f|=%.6f",
             N, np.max(np.abs(phi_ex)), np.max(np.abs(f_vals)))
    _run_all_2d("2D_Poisson_SingleMode_n1m1", N, x, y, dx, dy,
                f_vals, phi_ex, skip_qsvt, results)

def run_section4_single_N(N, skip_qsvt, results):
    """
    HET MMS (SPT-100).
    Precompute kappas (HET domain, dz!=dr):
      1.9228 (N=4), 1.9704 (N=8), 1.9926 (N=16), 1.9982 (N=32), 1.9995 (N=64)
    """
    _banner(f"SECTION 4 - HET MMS (SPT-100), N={N}")
    z, r, dz, dr = _grid_2d(N, HET_Lz, HET_Lr)
    z_pts = np.arange(1, N+1)*dz
    phi_ex = _phi_het_mms(z, r); f_vals = _f_het_mms(z, r)
    bc_bottom = HET_phi0 * np.sin(np.pi*z_pts/HET_Lz)
    log.info("  N=%d  max|phi_exact|=%.4f V  max|f|=%.4e V/m^2",
             N, np.max(np.abs(phi_ex)), np.max(np.abs(f_vals)))
    _run_all_2d("2D_HET_MMS_SPT100", N, z, r, dz, dr, f_vals, phi_ex,
                skip_qsvt, results, bc_bottom=bc_bottom)

def run_section5_single_N(N, skip_qsvt, results):
    """
    HET sinusoidal source (meeting-report case).
    Precompute kappas: same as Section 4 (HET domain).
    """
    _banner(f"SECTION 5 - HET sinusoidal source, N={N}")
    Lx, Ly, phi0 = 0.025, 0.020, 20.0
    x, y, dx, dy = _grid_2d(N, Lx, Ly)
    f_vals = _f_het_sin(x, y, Lx, Ly, phi0)
    phi_ex = _phi_het_sin(x, y, Lx, Ly, phi0)
    log.info("  N=%d  max|phi_exact|=%.4f V", N, np.max(np.abs(phi_ex)))
    _run_all_2d("2D_HET_Sin_MeetingReport", N, x, y, dx, dy,
                f_vals, phi_ex, skip_qsvt, results)


# ============================================================================
#  Result serialisation
# ============================================================================

def _save_results(results):
    json_path = RESULTS_DIR / "results_full.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    log.info("Results saved: %s (%d rows)", json_path, len(results))
    csv_path = RESULTS_DIR / "results_summary.csv"
    if results:
        fieldnames = [f.name for f in dataclasses.fields(RunResult2D)]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results: writer.writerow(asdict(r))
    log.info("CSV saved: %s", csv_path)

def _save_metadata(N_values, skip_qsvt, max_workers):
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": platform.node(), "python": sys.version,
        "numpy": np.__version__, "cpu_count": os.cpu_count(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "N_values": N_values, "skip_qsvt": skip_qsvt,
        "max_workers": max_workers,
        "qsvt_max_degree_2d": {str(k): v for k, v in QSVT_MAX_DEGREE_2D.items()},
        "jacobi_tol": JACOBI_TOL, "jacobi_max_iter": JACOBI_MAX_ITER,
        "hhl_epsilon": HHL_EPSILON, "n_fourier_modes": N_FOURIER_MODES,
        "n_fine_fourier": N_FINE,
        "het_Lz_m": HET_Lz, "het_Lr_m": HET_Lr, "het_phi0_V": HET_phi0,
        "kappas_unit_square": {
            "N4": 2.3586, "N8": 2.7725, "N16": 2.9352,
            "N32": 2.9838, "N64": 2.9960},
        "kappas_het_domain": {
            "N4": 1.9228, "N8": 1.9704, "N16": 1.9926,
            "N32": 1.9982, "N64": 1.9995},
    }
    for mod in ("qiskit", "qiskit_aer", "scipy"):
        try: meta[mod] = __import__(mod).__version__
        except Exception: meta[mod] = "not installed"
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception: meta["git_commit"] = "unknown"
    with open(RESULTS_DIR / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Metadata saved.")


# ============================================================================
#  Work unit dispatch
# ============================================================================

def _execute_work_unit_2d(work_type, N, skip_qsvt):
    results = []
    dispatch = {
        "section1": run_section1_single_N, "section2": run_section2_single_N,
        "section3": run_section3_single_N, "section4": run_section4_single_N,
        "section5": run_section5_single_N,
    }
    fn = dispatch.get(work_type)
    if fn is None: log.error("Unknown work_type '%s'", work_type); return results
    fn(N, skip_qsvt, results)
    return results


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Full 2-D HPC benchmark sweep for quantum PDE solvers.")
    parser.add_argument("--max-n", type=int, default=max(N_VALUES_ALL))
    parser.add_argument("--skip-qsvt", action="store_true")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS_DEFAULT)
    parser.add_argument("--sections", type=str, default="1,2,3,4,5")
    args = parser.parse_args()

    N_values = [n for n in N_VALUES_ALL if n <= args.max_n]
    if not N_values: parser.error(f"--max-n {args.max_n} excludes every N.")
    sections = [f"section{s.strip()}" for s in args.sections.split(",")]

    _banner("QUANTUM PDE SOLVER - FULL 2D HPC BENCHMARK RUN (v3)")
    log.info("  N values   : %s", N_values)
    log.info("  Sections   : %s", sections)
    log.info("  Max workers: %d", args.max_workers)
    log.info("  Output dir : %s", RESULTS_DIR.resolve())

    _save_metadata(N_values, args.skip_qsvt, args.max_workers)

    t_global = time.perf_counter(); results = []
    work_units = [(s, N, args.skip_qsvt)
                  for N in sorted(N_values) for s in sections]

    if args.max_workers == 1:
        log.info("Serial execution mode.")
        for work_type, N, sq in work_units:
            try:
                partial = _execute_work_unit_2d(work_type, N, sq)
                results.extend(partial); _save_results(results)
            except Exception as exc:
                log.error("Work unit failed: %s N=%d - %s",
                          work_type, N, exc, exc_info=True)
    else:
        log.info("Parallel execution: %d units across %d workers.",
                 len(work_units), args.max_workers)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers, max_tasks_per_child=1,
        ) as executor:
            futures = {executor.submit(_execute_work_unit_2d, wt, N, sq): (wt, N)
                       for wt, N, sq in work_units}
            for future in concurrent.futures.as_completed(futures):
                work_type, N = futures[future]
                try:
                    partial = future.result(); results.extend(partial)
                    log.info("Done: %-12s N=%-3d  (%d results total)",
                             work_type, N, len(results))
                    _save_results(results)
                except Exception as exc:
                    log.error("Failed: %s N=%d - %s", work_type, N, exc, exc_info=True)

    _save_results(results)
    elapsed = time.perf_counter() - t_global
    _banner(f"Benchmark complete. Total elapsed time: {elapsed:.1f} s")
    log.info("Results: %s", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()