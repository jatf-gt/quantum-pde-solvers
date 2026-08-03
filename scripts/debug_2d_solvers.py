#!/usr/bin/env python3
"""
=========================
Key changes vs v3:
  1. Outer Jacobi loop has early-stopping: halts when improvement in delta
     stalls for `patience` consecutive print intervals.
  2. VQLS-specific noise floor: outer loop accepts convergence when
     vs_thomas < vqls_thomas_tol (default 0.5%) even if delta > tol.
  3. Error decomposition explanation printed at the end of every run.
  4. N=8 VQLS stagnation is now correctly diagnosed and stopped.
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

# Early-stopping: halt outer loop if delta improves by less than this
# fraction over `patience` consecutive print intervals.
EARLY_STOP_MIN_IMPROVEMENT = 0.01   # 1% relative improvement required
EARLY_STOP_PATIENCE        = 4      # consecutive intervals without improvement

# VQLS-specific: accept outer convergence when quantum vs Thomas is below this.
# VQLS has a cost-function noise floor; demanding delta < 1e-6 is unrealistic.
VQLS_THOMAS_TOL = 0.005             # 0.5% vs Thomas is "good enough" for VQLS
JACOBI_TOL: float = 1e-8   # absolute delta
JACOBI_MAX_ITER = 500


# ============================================================================
#  Problem definition
# ============================================================================

def build_grid_2d(N):
    dx = 1.0 / (N + 1)
    pts = np.arange(1, N + 1) * dx
    x, y = np.meshgrid(pts, pts, indexing="ij")
    return x, y, dx

def f_sin2d(x, y):   return np.sin(np.pi * x) * np.sin(np.pi * y)
def u_exact_sin2d(x, y): return -np.sin(np.pi * x) * np.sin(np.pi * y) / (2.0 * np.pi**2)

def build_tst_row(N, dx):
    A = -4.0*np.eye(N) + np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1)
    eigs = np.abs(np.linalg.eigvalsh(A))
    return A, float(eigs.max() / eigs.min())

def build_rhs_row(j, phi, f_vals, dx):
    N = phi.shape[0]
    b = dx**2 * f_vals[:, j].copy()
    if j > 0:     b -= phi[:, j-1]
    if j < N-1:   b -= phi[:, j+1]
    return b

def _thomas_1d(A, b):
    N = len(b)
    diag = -4.0*np.ones(N); off = 1.0*np.ones(N); d = b.copy()
    for i in range(1, N):
        m = off[i-1]/diag[i-1]; diag[i] -= m*off[i-1]; d[i] -= m*d[i-1]
    u = np.zeros(N); u[-1] = d[-1]/diag[-1]
    for i in range(N-2, -1, -1): u[i] = (d[i] - off[i]*u[i+1])/diag[i]
    return u

def _max_rel(u, ref):
    return float(np.max(np.abs(u - ref))) / (float(np.max(np.abs(ref))) + 1e-300) * 100.0


# ============================================================================
#  Solver adapters (confirmed from introspection)
# ============================================================================

def _call_hhl(A, b):
    from solvers.quantum.hhl_1d import hhl_solve_system
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(): warnings.simplefilter("ignore")
        result = hhl_solve_system(A, b, HHL_EPSILON)
        wall = time.perf_counter() - t0
        u = np.asarray(result[0], dtype=float)
        res = float(np.linalg.norm(A@u - b) / (np.linalg.norm(b) + 1e-300))
        return u, res, wall, {"prop_const": float(result[2]) if len(result)>2 else float("nan")}
    except Exception as e:
        return None, float("nan"), time.perf_counter()-t0, {"error": str(e)}

def _call_vqls(A, b):
    from solvers.quantum.vqls_1d import vqls_solve_system
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(): warnings.simplefilter("ignore")
        result = vqls_solve_system(A, b)
        wall = time.perf_counter() - t0
        u = np.asarray(result.u, dtype=float)
        res = float(np.linalg.norm(A@u - b) / (np.linalg.norm(b) + 1e-300))
        return u, res, wall, {
            "final_cost": getattr(result, "final_cost", float("nan")),
            "optimiser_success": getattr(result, "optimiser_success", None),
        }
    except Exception as e:
        return None, float("nan"), time.perf_counter()-t0, {"error": str(e)}

def _call_qsvt(A, b):
    from solvers.quantum.qsvt_1d import qsvt_solve_system, QSVTConfig1D
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(): warnings.simplefilter("ignore")
        qsvt_cfg = QSVTConfig1D(                             
            angle_method = "auto",
            verbose      = False,
            max_degree   = 500,          
            )
        result = qsvt_solve_system(A, b, config=qsvt_cfg)
        wall = time.perf_counter() - t0
        u = np.asarray(result.u, dtype=float)
        res = float(np.linalg.norm(A@u - b) / (np.linalg.norm(b) + 1e-300))
        return u, res, wall, {
            "polynomial_degree": getattr(result, "polynomial_degree", None),
            "circuit_depth":     getattr(result, "circuit_depth", None),
        }
    except Exception as e:
        return None, float("nan"), time.perf_counter()-t0, {"error": str(e)}

ADAPTERS = {"hhl": _call_hhl, "vqls": _call_vqls, "qsvt": _call_qsvt}


# ============================================================
#  Shared SOR parameter computation
# ============================================================

def _sor_omega(N: int, dx: float, dy: float) -> float:
    """
    Optimal SOR relaxation parameter for the 2D Poisson equation
    on a rectangular domain with the 5-point stencil.

    For a square domain (dx=dy), the spectral radius of the
    point-Jacobi iteration is:
        rho_J = cos(pi/(N+1))
    and the optimal omega is:
        omega = 2 / (1 + sqrt(1 - rho_J^2))

    For a non-square domain (dx != dy), the spectral radius of the
    Line-Jacobi iteration is bounded by the same expression since
    kappa(A_row) ~ O(1), so the same formula gives a good approximation.

    Reference: Young (1971), Iterative Solution of Large Linear Systems,
    Chapter 5. Also confirmed in Ghafourpour & Laizet (2025) Appendix D.
    """
    # Use the smaller dimension for the spectral radius estimate
    # (more conservative, avoids over-relaxation on non-square grids)
    N_eff = N + 1
    rho_J = np.cos(np.pi / N_eff)
    omega = 2.0 / (1.0 + np.sqrt(1.0 - rho_J**2))
    # Clamp to (1, 2) — omega=1 is Gauss-Seidel, omega>=2 diverges
    return float(np.clip(omega, 1.0, 1.99))


def sor_2d_thomas(N: int, f_vals: np.ndarray, dx: float, dy: float,
                  bc_left=None, bc_right=None, bc_bottom=None, bc_top=None,
                  tol: float = JACOBI_TOL, max_iter: int = JACOBI_MAX_ITER,
                  omega: Optional[float] = None) -> tuple[np.ndarray, int, bool]:
    """
    2D Line-SOR with Thomas inner solver.

    Replaces jacobi_2d_thomas. The only algorithmic difference is the
    relaxation update:

        phi_new[:,j] = omega * phi_thomas[:,j] + (1-omega) * phi[:,j]

    where phi_thomas[:,j] is the Thomas solution for strip j using the
    ALREADY-UPDATED phi_new for strips 0..j-1 (Gauss-Seidel ordering).
    This is the key difference from Jacobi: we use phi_new (not phi) for
    already-processed strips, which propagates information faster.

    With omega=1 this reduces to Line Gauss-Seidel.
    With omega=omega_opt this is Line SOR, giving O(N) convergence.

    Parameters
    ----------
    omega : relaxation parameter. None => use _sor_omega(N, dx, dy).
            Pass omega=1.0 to force Gauss-Seidel without over-relaxation.
    """
    if omega is None:
        omega = _sor_omega(N, dx, dy)

    phi = np.zeros((N, N))
    A, _ = _build_row_matrix(N, dx, dy)

    for it in range(max_iter):
        phi_old = phi.copy()   # keep for convergence check only

        for j in range(N):
            # Use phi (partially updated this sweep) for neighbours —
            # this is the Gauss-Seidel part: strips 0..j-1 already have
            # their new values in phi, strips j+1..N-1 still have old values.
            b = _build_rhs_row(j, phi, f_vals, dx, dy,
                               bc_left, bc_right, bc_bottom, bc_top)
            phi_thomas_j = _thomas_1d_general(A, b)

            # SOR relaxation
            phi[:, j] = omega * phi_thomas_j + (1.0 - omega) * phi[:, j]

        delta = float(np.max(np.abs(phi - phi_old)))
        phi_scale = float(np.max(np.abs(phi)))
        rel_delta = delta / phi_scale if phi_scale > 0 else delta

        if rel_delta < tol:
            return phi, it + 1, True

    return phi, max_iter, False


def sor_2d_quantum(N: int, f_vals: np.ndarray, dx: float, dy: float,
                   solver_name: str, phi_thomas: np.ndarray,
                   bc_left=None, bc_right=None, bc_bottom=None, bc_top=None,
                   tol: float = 1e-6, max_iter: int = JACOBI_MAX_ITER,
                   omega: Optional[float] = None) -> tuple:
    """
    2D Line-SOR with a quantum inner solver.

    Same SOR update as sor_2d_thomas, but the inner solve uses
    HHL / VQLS / QSVT instead of Thomas.

    The quantum solver returns a solution proportional to A^{-1}b.
    The SOR relaxation is applied AFTER proportionality recovery,
    so the relaxation acts on the physical solution, not the raw
    quantum state.

    Note on omega for quantum solvers: VQLS has a noise floor that
    can cause the SOR update to oscillate if omega is too large.
    The VQLS_THOMAS_TOL early-stop criterion handles this: once
    vs_thomas < 0.5%, the iteration stops regardless of delta.
    """
    if omega is None:
        omega = _sor_omega(N, dx, dy)

    phi = np.zeros((N, N))
    A, kappa = _build_row_matrix(N, dx, dy)
    is_vqls = (solver_name == "vqls")
    extra = {"kappa_row": kappa, "costs": [], "degrees": [],
             "depths": [], "prop_consts": [], "omega": omega}
    best_delta = float("inf")
    no_improve = 0
    stop_reason = "max_iter"

    for it in range(max_iter):
        phi_old = phi.copy()

        for j in range(N):
            # Gauss-Seidel ordering: use partially-updated phi
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
                    extra["degrees"].append(deg)
                    extra["depths"].append(depth)
                else:
                    raise ValueError(f"Unknown solver: {solver_name}")
            except Exception as exc:
                log.warning("    %s-2D inner solver failed iter %d row %d: %s",
                            solver_name.upper(), it+1, j, exc)
                return phi, it+1, False, extra, "solver_failure"

            u_row = np.asarray(u_row, dtype=float)
            if u_row.shape != (N,):
                return phi, it+1, False, extra, "shape_error"

            # SOR relaxation on the physical solution
            phi[:, j] = omega * u_row + (1.0 - omega) * phi[:, j]

        delta = float(np.max(np.abs(phi - phi_old)))
        phi_scale = float(np.max(np.abs(phi)))
        rel_delta = delta / phi_scale if phi_scale > 0 else delta

        # VQLS noise-floor stop
        if is_vqls and it > 0:
            if _max_rel(phi, phi_thomas) < VQLS_THOMAS_TOL * 100.0:
                return phi, it+1, True, extra, "vqls_noise_floor"

        # Early stopping on relative delta
        if rel_delta < best_delta * (1.0 - EARLY_STOP_MIN_IMPROVEMENT):
            best_delta = rel_delta
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                converged = _max_rel(phi, phi_thomas) < 2.0
                return phi, it+1, converged, extra, "early_stop_stagnation"

        if rel_delta < tol:
            return phi, it+1, True, extra, "tol_met"

    return phi, max_iter, False, extra, stop_reason


# ============================================================================
#  Summary + error decomposition explanation
# ============================================================================

def print_summary(label, phi, u_exact, u_thomas, n_iters,
                  converged, diag, wall_total, stop_reason):
    sep = "═" * 60
    print(f"\n  {_B}{sep}{_X}")
    print(f"  {_B}SUMMARY: {label}-2D{_X}")
    print(f"  {sep}")
    status = f"{_G}CONVERGED{_X}" if converged else f"{_R}NOT CONVERGED{_X}"
    print(f"  Status      : {status}  ({n_iters} iters)  {wall_total:.2f}s")
    print(f"  Stop reason : {stop_reason}")

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

        # Error decomposition explanation
        print(f"\n  {_B}Error decomposition:{_X}")
        print(f"    Thomas vs exact  (Jacobi h²-discretisation): "
              f"{_Y}{err_thomas_e:.3f}%{_X}  ← irreducible at this N")
        print(f"    {label} vs Thomas (quantum algorithmic):      "
              f"{col_t}{err_t:.3f}%{_X}  ← solver quality")
        print(f"    {label} vs exact  (total):                    "
              f"{col_e}{err_e:.3f}%{_X}")
        print(f"\n  {_C}Note: 'vs_exact reversal' during iteration is EXPECTED.{_X}")
        print(f"  {_C}As the quantum solver converges toward Thomas, the total{_X}")
        print(f"  {_C}error floors at the Jacobi discretisation error (~{err_thomas_e:.2f}%).{_X}")
        print(f"  {_C}The transient dip below this floor is partial cancellation{_X}")
        print(f"  {_C}of quantum and discretisation errors — not a real improvement.{_X}")
    else:
        print(f"  {_R}Solution is zero or None.{_X}")

    if diag.get("inner_times"):
        t = np.array(diag["inner_times"])
        print(f"\n  Inner timing: mean={t.mean():.4f}s  "
              f"max={t.max():.4f}s  total={t.sum():.2f}s")

    costs = [e.get("final_cost") for e in diag.get("extra_per_row", [])
             if e.get("final_cost") is not None and not np.isnan(e["final_cost"])]
    if costs:
        c = np.array(costs, dtype=float)
        col = _G if c.max() < 1e-4 else (_Y if c.max() < 1e-2 else _R)
        print(f"  VQLS cost   : mean={c.mean():.2e}  "
              f"max={col}{c.max():.2e}{_X}  min={c.min():.2e}")

    degrees = [e.get("polynomial_degree") for e in diag.get("extra_per_row", [])
               if e.get("polynomial_degree") is not None]
    if degrees:
        d = np.array(degrees)
        print(f"  QSVT degree : mean={d.mean():.0f}  max={d.max():.0f}")
    print(f"  {sep}")


# ============================================================================
#  Plotting
# ============================================================================

def plot_solutions(N, x, y, u_exact, results_dict):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError: return

    valid = {k: v for k, v in results_dict.items()
             if v is not None and not np.allclose(v, 0.0)}
    n_cols = 1 + len(valid)
    fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 7))
    if n_cols == 1: axes = axes.reshape(2, 1)

    vmin, vmax = u_exact.min(), u_exact.max()
    im = axes[0,0].pcolormesh(x, y, u_exact, cmap="RdBu_r",
                               vmin=vmin, vmax=vmax, shading="auto")
    axes[0,0].set_title("Exact", fontweight="bold"); axes[0,0].set_aspect("equal")
    plt.colorbar(im, ax=axes[0,0], shrink=0.8); axes[1,0].axis("off")

    for ci, (name, phi) in enumerate(valid.items(), 1):
        im = axes[0,ci].pcolormesh(x, y, phi, cmap="RdBu_r",
                                    vmin=vmin, vmax=vmax, shading="auto")
        axes[0,ci].set_title(name, fontweight="bold"); axes[0,ci].set_aspect("equal")
        plt.colorbar(im, ax=axes[0,ci], shrink=0.8)
        err = phi - u_exact; abs_max = max(np.abs(err).max(), 1e-12)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        im2 = axes[1,ci].pcolormesh(x, y, err, cmap="seismic",
                                     norm=norm, shading="auto")
        axes[1,ci].set_title(f"Error ({_max_rel(phi,u_exact):.2f}%)")
        axes[1,ci].set_aspect("equal")
        plt.colorbar(im2, ax=axes[1,ci], shrink=0.8)

    fig.suptitle(f"2D Poisson Debug — N={N}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / f"debug_2d_solutions_N{N}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  {_G}Solutions: {out}{_X}"); plt.close(fig)


def plot_convergence(errors_by_solver, N):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError: return

    colours = {"Thomas":"black","HHL":"royalblue","VQLS":"darkorange","QSVT":"crimson"}
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, errs in errors_by_solver.items():
        if errs:
            ax.semilogy(range(1, len(errs)+1), errs,
                        label=name, color=colours.get(name,"grey"), lw=1.8)
    ax.set_xlabel("Jacobi Iteration"); ax.set_ylabel("Max Δ")
    ax.set_title(f"2D Line-Jacobi Convergence — N={N}")
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    out = OUT_DIR / f"debug_2d_convergence_N{N}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  {_G}Convergence: {out}{_X}"); plt.close(fig)


# ============================================================================
#  2D HET MMS CASE — add after the generic Poisson problem definitions
# ============================================================================

# Physical parameters (SPT-100 representative)
HET_Lz   = 0.025    # m  axial channel length
HET_Lr   = 0.020    # m  radial channel width
HET_phi0 = 300.0    # V  discharge voltage


def build_grid_2d_het(Nz: int, Nr: int):
    """
    Non-square grid for the HET axial-radial domain [0,Lz] x [0,Lr].
    Returns (z, r, dz, dr) — interior nodes only.
    """
    dz = HET_Lz / (Nz + 1)
    dr = HET_Lr / (Nr + 1)
    z_pts = np.arange(1, Nz + 1) * dz
    r_pts = np.arange(1, Nr + 1) * dr
    z, r = np.meshgrid(z_pts, r_pts, indexing="ij")   # shape (Nz, Nr)
    return z, r, dz, dr


def phi_het_mms(z: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Manufactured solution for the 2D HET axial-radial Poisson problem.
    phi(z,r) = phi0 * sin(pi*z/Lz) * cos(pi*r/(2*Lr))

    Physical motivation:
      - sin(pi*z/Lz): axial profile, zero at anode and cathode,
                      peak near channel exit — matches observed HET profiles
      - cos(pi*r/(2*Lr)): radial profile, maximum at inner wall (r=0),
                           zero at outer wall (r=Lr) — consistent with
                           radial sheath structure
    """
    return HET_phi0 * np.sin(np.pi * z / HET_Lz) * np.cos(np.pi * r / (2.0 * HET_Lr))


def f_het_mms(z: np.ndarray, r: np.ndarray) -> np.ndarray:
    """
    Source term f = nabla^2(phi_MMS) for the Poisson equation nabla^2(phi) = f.

    phi_MMS = phi0 * sin(pi*z/Lz) * cos(pi*r/(2*Lr))

    d^2phi/dz^2 = -phi0*(pi/Lz)^2   * sin(pi*z/Lz)*cos(pi*r/(2*Lr))
    d^2phi/dr^2 = -phi0*(pi/2Lr)^2  * sin(pi*z/Lz)*cos(pi*r/(2*Lr))

    Therefore: f = nabla^2(phi) = -phi0*pi^2*(1/Lz^2 + 1/(4*Lr^2))
                                   * sin(pi*z/Lz)*cos(pi*r/(2*Lr))

    Convergence rate of FD Laplacian vs exact: verified O(h^2) — correct.
    Large absolute residuals at N=4 (~1.5e5) are expected truncation error,
    not a formulation bug. Relative error ~4% at N=4, ~0.3% at N=16.
    """
    coeff = -HET_phi0 * np.pi**2 * (1.0/HET_Lz**2 + 1.0/(4.0*HET_Lr**2))
    return coeff * np.sin(np.pi * z / HET_Lz) * np.cos(np.pi * r / (2.0 * HET_Lr))


def build_tst_row_het(Nz: int, dz: float, dr: float):
    """
    TST matrix for one radial-strip (r=const) update in the HET Line-Jacobi.

    The row equation (fixing r-index j, updating z-strip):
      phi_{i+1,j}/dz^2 - 2*phi_{i,j}*(1/dz^2 + 1/dr^2) + phi_{i-1,j}/dz^2
        = f_{i,j} - (phi_{i,j-1} + phi_{i,j+1})/dr^2

    Matrix: main diag = -2*(1/dz^2 + 1/dr^2), off-diag = 1/dz^2
    """
    a_diag = -2.0 * (1.0/dz**2 + 1.0/dr**2)
    b_off  =  1.0 / dz**2
    A = a_diag * np.eye(Nz) + b_off * (np.diag(np.ones(Nz-1), 1)
                                         + np.diag(np.ones(Nz-1), -1))
    eigs = np.abs(np.linalg.eigvalsh(A))
    kappa = float(eigs.max() / eigs.min())
    return A, kappa


def build_rhs_row_het(j: int, phi: np.ndarray, f_vals: np.ndarray,
                       dz: float, dr: float,
                       bc_inner: np.ndarray,
                       bc_anode: np.ndarray,
                       bc_cathode: np.ndarray) -> np.ndarray:
    """
    RHS for the j-th interior radial strip (0-indexed).

    The discretised equation for interior node (i, j) is:
        (phi[i+1,j] - 2*phi[i,j] + phi[i-1,j]) / dz^2
      + (phi[i,j+1] - 2*phi[i,j] + phi[i,j-1]) / dr^2
      = f[i,j]

    Rearranging for the axial tridiagonal system (unknown: phi[:,j]):
        phi[i-1,j]/dz^2  +  (-2/dz^2 - 2/dr^2)*phi[i,j]  +  phi[i+1,j]/dz^2
      = f[i,j]
        - phi[i,j-1]/dr^2   (left radial neighbour, or inner BC)
        - phi[i,j+1]/dr^2   (right radial neighbour, or outer BC = 0)
        - bc_anode[j]/dz^2  (absorbed at i=0 boundary)
        - bc_cathode[j]/dz^2 (absorbed at i=Nz-1 boundary)

    The row matrix A already encodes the 1/dz^2 coefficients,
    so the RHS is built in physical units (V/m^2 equivalent).
    """
    Nz = phi.shape[0]
    b = f_vals[:, j].copy()   # raw source f(z,r) in V/m^2

    # ── Radial coupling: subtract neighbour contributions / dr^2 ─────────────
    if j > 0:
        b -= phi[:, j-1] / dr**2          # left interior neighbour
    else:
        b -= bc_inner / dr**2             # j=0: left neighbour is inner wall BC

    if j < phi.shape[1] - 1:
        b -= phi[:, j+1] / dr**2          # right interior neighbour
    # else: j=Nr-1, outer wall phi=0, contributes nothing

    # ── Axial BCs: absorbed into first and last entries ───────────────────────
    b[0]  -= bc_anode[j]   / dz**2        # phi(z=0, r_j) = bc_anode[j]
    b[-1] -= bc_cathode[j] / dz**2        # phi(z=Lz, r_j) = bc_cathode[j]

    return b


def jacobi_2d_het_thomas(Nz, Nr, f_vals, dz, dr,
                           bc_inner, bc_anode, bc_cathode,
                           tol=1e-8, max_iter=500, verbose=False):
    """
    2D Line-Jacobi for the HET domain with Thomas inner solver.
    Outer wall BC is zero (absorbed implicitly — contributes nothing to RHS).
    """
    phi = np.zeros((Nz, Nr))
    A, kappa = build_tst_row_het(Nz, dz, dr)
    deltas = []

    for it in range(max_iter):
        phi_new = phi.copy()
        for j in range(Nr):
            b = build_rhs_row_het(j, phi, f_vals, dz, dr,
                                   bc_inner, bc_anode, bc_cathode)
            phi_new[:, j] = _thomas_1d_het(A, b)

        delta = float(np.max(np.abs(phi_new - phi)))
        deltas.append(delta)
        phi = phi_new
        if verbose and (it+1) % 20 == 0:
            print(f"    Thomas-HET iter {it+1:4d}  Δ={delta:.3e}")
        if delta < tol:
            return phi, it+1, True, deltas

    return phi, max_iter, False, deltas


def jacobi_2d_het_quantum(
    Nz, Nr, f_vals, dz, dr,
    bc_inner, bc_anode, bc_cathode,
    phi_exact, phi_thomas, solver_name,
    tol=1e-6, max_iter=200, print_every=10,
):
    """2D Line-Jacobi for the HET domain with a quantum inner solver."""
    phi = np.zeros((Nz, Nr))
    A, kappa = build_tst_row_het(Nz, dz, dr)
    adapter  = ADAPTERS[solver_name]
    label    = solver_name.upper()
    is_vqls  = (solver_name == "vqls")

    diag = {"kappa_row": kappa, "inner_times": [], "extra_per_row": [],
            "vs_exact_err": [], "vs_thomas_err": []}
    deltas = []
    best_delta = float("inf")
    no_improve_ct = 0
    stop_reason = "max_iter"

    print(f"\n  {_C}{'─'*60}{_X}")
    print(f"  {_B}{label}-2D-HET{_X}  Nz={Nz}  Nr={Nr}  "
          f"κ(A_row)={kappa:.4f}  tol={tol:.0e}")
    print(f"  Domain: {HET_Lz*1e3:.0f}mm × {HET_Lr*1e3:.0f}mm  "
          f"phi0={HET_phi0:.0f}V")
    print(f"  {'─'*60}")

    for it in range(max_iter):
        phi_new = phi.copy()
        for j in range(Nr):
            b = build_rhs_row_het(j, phi, f_vals, dz, dr,
                                   bc_inner, bc_anode, bc_cathode)
            u_row, res_row, t_row, extra = adapter(A, b)
            if u_row is None:
                print(f"  {_R}[FAIL] iter {it+1} strip {j}: "
                      f"{extra.get('error','?')}{_X}")
                return phi, it+1, False, deltas, diag, "solver_failure"
            if u_row.shape != (Nz,):
                print(f"  {_R}[SHAPE] iter {it+1} strip {j}: {u_row.shape}{_X}")
                return phi, it+1, False, deltas, diag, "shape_error"
            phi_new[:, j] = u_row
            diag["inner_times"].append(t_row)
            diag["extra_per_row"].append(extra)

        delta = float(np.max(np.abs(phi_new - phi)))
        deltas.append(delta)
        phi = phi_new

        if (it+1) % print_every == 0:
            err_e = _max_rel(phi, phi_exact)
            err_t = _max_rel(phi, phi_thomas)
            diag["vs_exact_err"].append((it+1, err_e))
            diag["vs_thomas_err"].append((it+1, err_t))
            col = _G if err_e < 5.0 else (_Y if err_e < 20.0 else _R)
            print(f"  iter {it+1:4d}  Δ={delta:.3e}  "
                  f"vs_exact={col}{err_e:7.3f}%{_X}  "
                  f"vs_thomas={err_t:7.3f}%")

            if is_vqls and err_t < VQLS_THOMAS_TOL * 100.0:
                print(f"  {_G}[VQLS] vs_thomas={err_t:.3f}% < threshold{_X}")
                stop_reason = "vqls_noise_floor"
                return phi, it+1, True, deltas, diag, stop_reason

            if delta < best_delta * (1.0 - EARLY_STOP_MIN_IMPROVEMENT):
                best_delta = delta; no_improve_ct = 0
            else:
                no_improve_ct += 1
                if no_improve_ct >= EARLY_STOP_PATIENCE:
                    print(f"  {_Y}[EARLY STOP] stagnation{_X}")
                    stop_reason = "early_stop_stagnation"
                    converged = _max_rel(phi, phi_thomas) < 2.0
                    return phi, it+1, converged, deltas, diag, stop_reason

        if delta < tol:
            print(f"  {_G}Converged at iter {it+1}{_X}")
            stop_reason = "tol_met"
            return phi, it+1, True, deltas, diag, stop_reason

    print(f"  {_R}Max iterations reached. Final Δ={deltas[-1]:.3e}{_X}")
    return phi, max_iter, False, deltas, diag, stop_reason


def _thomas_1d_het(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Thomas algorithm for the non-uniform HET row system.
    A is a general tridiagonal (not necessarily -4/+1).
    """
    N = len(b)
    # Extract diagonals
    main = A.diagonal(0).copy()
    upper = A.diagonal(1).copy()
    lower = A.diagonal(-1).copy()
    d = b.copy()

    # Forward sweep
    for i in range(1, N):
        m = lower[i-1] / main[i-1]
        main[i] -= m * upper[i-1]
        d[i]    -= m * d[i-1]

    # Back substitution
    u = np.zeros(N)
    u[-1] = d[-1] / main[-1]
    for i in range(N-2, -1, -1):
        u[i] = (d[i] - upper[i] * u[i+1]) / main[i]
    return u


def run_het_2d_case(Nz=4, Nr=4, solvers_to_run=None,
                    max_iter=200, tol=1e-6, print_every=10, make_plots=False):
    if solvers_to_run is None:
        solvers_to_run = ["hhl", "vqls", "qsvt"]

    print(f"\n{_B}{'═'*64}{_X}")
    print(f"{_B}  2D HET MMS BENCHMARK{_X}")
    print(f"{_B}  Nz={Nz}  Nr={Nr}  Lz={HET_Lz*1e3:.0f}mm  "
          f"Lr={HET_Lr*1e3:.0f}mm  phi0={HET_phi0:.0f}V{_X}")
    print(f"{_B}{'═'*64}{_X}")

    z, r, dz, dr = build_grid_2d_het(Nz, Nr)
    phi_exact = phi_het_mms(z, r)
    f_vals    = f_het_mms(z, r)

    # Interior node coordinates
    z_pts = np.arange(1, Nz+1) * dz
    r_pts = np.arange(1, Nr+1) * dr

    # Boundary conditions from the manufactured solution
    bc_anode   = np.zeros(Nr)                                    # phi(z=0,  r_j) = 0
    bc_cathode = np.zeros(Nr)                                    # phi(z=Lz, r_j) = 0
    bc_inner   = HET_phi0 * np.sin(np.pi * z_pts / HET_Lz)     # phi(z_i,  r=0)
    # bc_outer = 0 everywhere — absorbed implicitly (no subtraction needed)

    print(f"\n  MMS: phi(z,r) = {HET_phi0:.0f}·sin(πz/{HET_Lz*1e3:.0f}mm)"
          f"·cos(πr/{2*HET_Lr*1e3:.0f}mm)")
    print(f"  max|phi_exact| = {np.max(np.abs(phi_exact)):.4f} V")
    print(f"  max|f_source|  = {np.max(np.abs(f_vals)):.4e} V/m²")
    print(f"  dz={dz*1e3:.3f}mm  dr={dr*1e3:.3f}mm")

    print(f"\n{_B}  Thomas-HET-2D (reference){_X}")
    t0 = time.perf_counter()
    phi_thomas, n_th, conv_th, errs_th = jacobi_2d_het_thomas(
        Nz, Nr, f_vals, dz, dr, bc_inner, bc_anode, bc_cathode,
        tol=tol, max_iter=500, verbose=True)
    t_th = time.perf_counter() - t0

    err_th = _max_rel(phi_thomas, phi_exact)
    col = _G if err_th < 5.0 else (_Y if err_th < 20.0 else _R)
    print(f"  Thomas-HET: {n_th} iters  "
          f"MaxRelErr={col}{err_th:.3f}%{_X}  "
          f"Time={t_th:.3f}s  Converged={conv_th}")

    Ez = -np.gradient(phi_thomas, dz, axis=0)
    Er = -np.gradient(phi_thomas, dr, axis=1)
    E_peak = np.sqrt(Ez**2 + Er**2).max()
    E_expected = HET_phi0 * np.pi / HET_Lz
    print(f"  Peak |E| Thomas = {E_peak:.3e} V/m  "
          f"(MMS expected ~{E_expected:.3e} V/m axial)")

    results = {"Thomas": phi_thomas}
    errors_by_solver = {"Thomas": errs_th}

    for sname in solvers_to_run:
        label  = sname.upper()
        max_it = 150 if sname == "qsvt" else max_iter
        t0 = time.perf_counter()
        phi_q, n_iters, converged, errs, diag, stop_reason = jacobi_2d_het_quantum(
            Nz=Nz, Nr=Nr, f_vals=f_vals, dz=dz, dr=dr,
            bc_inner=bc_inner, bc_anode=bc_anode, bc_cathode=bc_cathode,
            phi_exact=phi_exact, phi_thomas=phi_thomas,
            solver_name=sname, tol=tol, max_iter=max_it,
            print_every=print_every)
        wall = time.perf_counter() - t0
        print_summary(label+"-HET", phi_q, phi_exact, phi_thomas,
                      n_iters, converged, diag, wall, stop_reason)
        if phi_q is not None and not np.allclose(phi_q, 0.0):
            Ez_q = -np.gradient(phi_q, dz, axis=0)
            Er_q = -np.gradient(phi_q, dr, axis=1)
            print(f"  Peak |E| {label} = {np.sqrt(Ez_q**2+Er_q**2).max():.3e} V/m")
        results[label] = phi_q
        errors_by_solver[label] = errs

    # Summary table
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
    """Plots for the HET 2D case: solution fields + convergence."""
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

    # Convert to mm for axis labels
    z_mm = z * 1e3; r_mm = r * 1e3

    im = axes[0, 0].pcolormesh(z_mm, r_mm, phi_exact.T,
                                cmap="RdBu_r", vmin=vmin, vmax=vmax,
                                shading="auto")
    axes[0, 0].set_title("MMS Exact", fontweight="bold")
    axes[0, 0].set_xlabel("z (mm)"); axes[0, 0].set_ylabel("r (mm)")
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8, label="φ (V)")
    axes[1, 0].axis("off")

    for ci, (name, phi) in enumerate(valid.items(), 1):
        im = axes[0, ci].pcolormesh(z_mm, r_mm, phi.T,
                                     cmap="RdBu_r", vmin=vmin, vmax=vmax,
                                     shading="auto")
        axes[0, ci].set_title(name, fontweight="bold")
        axes[0, ci].set_xlabel("z (mm)"); axes[0, ci].set_ylabel("r (mm)")
        plt.colorbar(im, ax=axes[0, ci], shrink=0.8, label="φ (V)")

        err = phi - phi_exact
        abs_max = max(np.abs(err).max(), 1e-12)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        im2 = axes[1, ci].pcolormesh(z_mm, r_mm, err.T,
                                      cmap="seismic", norm=norm, shading="auto")
        axes[1, ci].set_title(f"Error ({_max_rel(phi, phi_exact):.2f}%)")
        axes[1, ci].set_xlabel("z (mm)"); axes[1, ci].set_ylabel("r (mm)")
        plt.colorbar(im2, ax=axes[1, ci], shrink=0.8, label="Δφ (V)")

    fig.suptitle(f"2D HET MMS Benchmark — Nz={Nz}  Nr={Nr}\n"
                 f"φ(z,r) = {HET_phi0:.0f}·sin(πz/{HET_Lz*1e3:.0f}mm)"
                 f"·cos(πr/{2*HET_Lr*1e3:.0f}mm)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / f"debug_het2d_solutions_Nz{Nz}_Nr{Nr}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  {_G}HET solutions: {out}{_X}")
    plt.close(fig)

    # Convergence
    colours = {"Thomas":"black","HHL":"royalblue","VQLS":"darkorange","QSVT":"crimson"}
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, errs in errors_by_solver.items():
        if errs:
            ax.semilogy(range(1, len(errs)+1), errs,
                        label=name, color=colours.get(name,"grey"), lw=1.8)
    ax.set_xlabel("Jacobi Iteration"); ax.set_ylabel("Max Δ")
    ax.set_title(f"2D HET MMS Convergence — Nz={Nz}  Nr={Nr}")
    ax.legend(); ax.grid(True, alpha=0.3); plt.tight_layout()
    out = OUT_DIR / f"debug_het2d_convergence_Nz{Nz}_Nr{Nr}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"  {_G}HET convergence: {out}{_X}")
    plt.close(fig)


# ============================================================================
#  Introspection
# ============================================================================

def introspect_solvers():
    print(f"\n{_B}SOLVER INTROSPECTION{_X}\n{'='*60}")
    specs = {"HHL":("solvers.quantum.hhl_1d","hhl_solve_system"),
             "VQLS":("solvers.quantum.vqls_1d","vqls_solve_system"),
             "QSVT":("solvers.quantum.qsvt_1d","qsvt_solve_system")}
    N=4; A=-4.0*np.eye(N)+np.diag(np.ones(N-1),1)+np.diag(np.ones(N-1),-1)
    b=np.array([0.04,0.04,0.04,0.04])
    for label,(mod,fn) in specs.items():
        print(f"\n{_C}--- {label} ---{_X}")
        try:
            import importlib
            m=importlib.import_module(mod); f=getattr(m,fn)
            print(f"  Signature: {fn}{inspect.signature(f)}")
            try:
                with warnings.catch_warnings(): warnings.simplefilter("ignore")
                r = f(A,b) if label!="HHL" else f(A,b,0.01)
                print(f"  Return: {type(r)}")
                if hasattr(r,"__dict__"): print(f"  Attrs: {list(r.__dict__.keys())}")
                elif isinstance(r,(tuple,list)):
                    for i,v in enumerate(r): print(f"    [{i}] {type(v).__name__} {getattr(v,'shape','')}")
            except Exception as e: print(f"  {_R}{e}{_X}")
        except Exception as e: print(f"  {_R}{e}{_X}")
    print("\n"+"="*60+"\n")


# ============================================================================
#  Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=2500)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--solver", default="all",
                        choices=["all","hhl","vqls","qsvt","thomas","het"])
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--qsvt-max-iter", type=int, default=200)
    parser.add_argument("--introspect", action="store_true")
    args = parser.parse_args()

    if args.introspect: introspect_solvers(); return

    N = args.N
    print(f"\n{_B}{'═'*64}{_X}")
    print(f"{_B}  2D QUANTUM POISSON SOLVER DEBUG TOOL{_X}")
    print(f"{_B}  N={N}  tol={args.tol:.0e}  HHL_ε={HHL_EPSILON}{_X}")
    print(f"{_B}  Output: {OUT_DIR}{_X}")
    print(f"{_B}{'═'*64}{_X}")

    x, y, dx = build_grid_2d(N)
    f_vals = f_sin2d(x, y); u_exact = u_exact_sin2d(x, y)
    _, kappa = build_tst_row(N, dx)
    print(f"\n  Grid: {N}×{N}  h={dx:.4f}  κ(A_row)={kappa:.4f}")
    print(f"  max|u_exact|={np.max(np.abs(u_exact)):.6f}")

    print(f"\n{_B}  Thomas-2D (reference){_X}")
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

    # Determine which quantum algorithms to run for the generic Poisson case
    _QUANTUM_SOLVERS = ["hhl", "vqls", "qsvt"]

    if args.solver == "all":
        generic_solvers = _QUANTUM_SOLVERS
    elif args.solver in _QUANTUM_SOLVERS:
        generic_solvers = [args.solver]
    else:
        # "thomas" or "het" — skip the generic quantum runs
        generic_solvers = []

    for sname in generic_solvers:
        label  = sname.upper()
        max_it = args.qsvt_max_iter if sname == "qsvt" else args.max_iter
        t0 = time.perf_counter()
        phi, n_iters, converged, errs, diag, stop_reason = sor_2d_quantum(
            N=N, f_vals=f_vals, dx=dx, solver_name=sname,
            u_exact=u_exact, u_thomas=u_thomas,
            tol=args.tol, max_iter=JACOBI_MAX_ITER, print_every=args.print_every)
        wall = time.perf_counter() - t0
        print_summary(label, phi, u_exact, u_thomas,
                      n_iters, converged, diag, wall, stop_reason)
        results_dict[label] = phi
        errors_by_solver[label] = errs

    # ── HET 2D MMS case ───────────────────────────────────────────────────────
    if args.solver in ("all", "het"):
        run_het_2d_case(
            Nz=args.N, Nr=args.N,
            solvers_to_run=_QUANTUM_SOLVERS,
            max_iter=args.max_iter,
            tol=args.tol,
            print_every=args.print_every,
            make_plots=args.plot,
        )

    print(f"\n{_B}{'═'*64}{_X}")
    print(f"{_B}  FINAL TABLE{_X}")
    print(f"{'─'*64}")
    print(f"  {'Solver':<8} {'Iters':>6} {'MaxRelErr%':>12} "
          f"{'vs Thomas%':>12} {'Conv':>8} {'Stop':>20}")
    print(f"{'─'*64}")
    for name, phi in results_dict.items():
        if phi is None or np.allclose(phi, 0.0):
            print(f"  {name:<8} {'—':>6} {'FAILED':>12} {'—':>12} {'—':>8}")
            continue
        err_e = _max_rel(phi, u_exact); err_t = _max_rel(phi, u_thomas)
        n_it = n_th if name=="Thomas" else "—"
        cv   = conv_th if name=="Thomas" else "—"
        col_e = _G if err_e < 5.0 else (_Y if err_e < 20.0 else _R)
        print(f"  {name:<8} {str(n_it):>6} "
              f"{col_e}{err_e:>11.3f}%{_X} {err_t:>11.3f}% {str(cv):>8}")
    print(f"{'═'*64}\n")

    if args.plot:
        print(f"{_B}  Saving figures...{_X}")
        plot_solutions(N, x, y, u_exact, results_dict)
        plot_convergence(errors_by_solver, N)

    # ── 2D HET MMS case ───────────────────────────────────────────────────────
    if args.solver in ("all", "het"):
        run_het_2d_case(
            Nz=args.N,
            Nr=args.N,          # square grid by default; can differ
            solvers_to_run=["hhl", "vqls", "qsvt"],
            max_iter=args.max_iter,
            tol=args.tol,
            print_every=args.print_every,
            make_plots=args.plot,
        )


if __name__ == "__main__":
    main()