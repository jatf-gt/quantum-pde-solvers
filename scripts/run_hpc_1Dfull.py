#!/usr/bin/env python3
"""
run_hpc_1Dfull.py
=================
Full 1-D benchmark sweep for Imperial College HPC (CX3, PBS Pro).

Cases
-----
  Section 1  : Generic Poisson, homogeneous Dirichlet BCs, three sources
               (fS = sin, fL = linear, fH = Heaviside).
  Section 1b : Generic Poisson, non-homogeneous Dirichlet BCs (alpha=1, beta=2).
  Section 2  : HET axial Poisson, three sub-cases:
                 3a  linear density profile, homogeneous Dirichlet
                 3b  Gaussian profile, V_d = 300 V anode, Dirichlet
                 3c  Gaussian profile, Neumann(x=0) - Dirichlet(x=1)

Solvers
-------
  Thomas (classical reference), HHL, VQLS, QSVT.

  QSVT runs only on the case families listed in ``QSVT_CASES``. This is a
  deliberate cost control, not an oversight: QSVT phase angles are looked up
  from a disk cache keyed on the matrix condition number, and only the
  standard Dirichlet TST matrix has precomputed phases. See ``QSVT_CASES``.

N range
-------
  N = 4, 8, 16, 32, 64 by default. Use ``--max-n`` to truncate the sweep for
  a fast validation pass (e.g. ``--max-n 16``).

Outputs (all under results/1Dhpc_run/)
--------------------------------------
  results_full.json     Full results table, machine-readable.
  results_summary.csv   Same table, human-readable.
  run_metadata.json     Environment + run configuration provenance.
  solutions_*.npz       Per-(case, solver, N) solution, RHS and residual vector.
  all_solutions.npz     Consolidated solution archive for post-processing.
  run.log               Progress log.

Usage on HPC (after activating the quantum-pde-solvers venv)
------------------------------------------------------------
  python scripts/run_hpc_1Dfull.py                    # full sweep, N = 4..64
  python scripts/run_hpc_1Dfull.py --max-n 16         # quick validation pass
  python scripts/run_hpc_1Dfull.py --skip-qsvt        # omit QSVT entirely
  python scripts/run_hpc_1Dfull.py --max-workers 1    # serial (required on GPU)

Author : Juan Antonio Trobajo Flecha
Date   : August 2026
"""

# -- Standard library ---------------------------------------------------------
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

# -- Third-party --------------------------------------------------------------
import numpy as np
import multiprocessing as mp

# -- Local --------------------------------------------------------------------
# Ensure the repository root is on sys.path regardless of invocation location.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from solvers.backend_factory import get_aer_backend, log_backend_info  # noqa: E402


# ============================================================================
#  Output directory and logging
# ============================================================================

RESULTS_DIR = Path("results") / "1Dhpc_run"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = RESULTS_DIR / "run.log"
# True only in the original parent process -- reliable under fork, spawn,
# or forkserver, unlike checking __name__ (which doesn't distinguish a
# respawned worker from the entry script the same way).
_IS_MAIN_PROCESS = mp.current_process().name == "MainProcess"

logging.basicConfig(
    level=logging.INFO,
    # The process ID is included because work units run in a ProcessPoolExecutor
    # and their log lines interleave in run.log; without it, attributing a line
    # to a particular (case, N) work unit after the fact is guesswork.
    format="%(asctime)s  pid=%(process)-6d  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w" if _IS_MAIN_PROCESS else "a"),
    ],
)
log = logging.getLogger(__name__)

# -- Suppress external library logging noise ----------------------------------
# Qiskit transpiler pass timings, IBM provider plugin errors, and Aer backend
# initialisation messages are irrelevant to benchmark progress monitoring.
for _noisy in (
    "qiskit.transpiler", "qiskit.transpiler.passes", "qiskit_ibm_runtime",
    "qiskit_ibm_provider", "qiskit_aer", "stevedore", "qiskit.passmanager",
    "qiskit.compiler",
):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)


# ============================================================================
#  Sweep configuration
# ============================================================================

# ── N values ─────────────────────────────────────────────────────────────────
# The full sweep. --max-n truncates this list; there is no separate "include
# N=64" flag, because N=64 is now part of the default sweep.
N_VALUES_ALL: list[int] = [4, 8, 16, 32, 64]

# ── QSVT: which N are attempted at all ───────────────────────────────────────
QSVT_MAX_N: int = 64

# ── QSVT: post-hoc wall-time warning threshold (seconds) ─────────────────────
# NOTE: this is a WARNING only. QSVT is not interruptible mid-solve, so this
# cannot abort a long run -- it only flags one in the log after the fact.
# Set to None to silence the warning entirely.
QSVT_TIME_LIMIT_S: Optional[float] = 1800.0   # 30 minutes per QSVT call

# ── QSVT: polynomial degree cap, per N ───────────────────────────────────────
# The QSVT phase cache is keyed on (kappa, epsilon, method, max_degree), so
# these values MUST match what scripts/precompute_qsvt_phases.py was run with,
# or the lookup misses and a from-scratch solve is attempted at runtime (hours
# to days for large kappa, and it will hit the sanity guard in qsp_angles.py).
#
# Current cache contents (results/qsvt_phase_cache/):
#   N =  4, 8, 16 : uncapped        -> cache tag d-1    -> None here
#   N = 32, 64    : capped at 5000  -> cache tag d5000  -> 5000 here
#
# The .get() fallback caps any N not listed (e.g. someone trying N=128), so an
# unlisted N degrades to a capped run rather than a multi-day or OOM attempt.
QSVT_MAX_DEGREE_BY_N: dict[int, Optional[int]] = {
    4: None, 8: None, 16: None,
    32: 5000, 64: 5000,
}
# Cheap cap for any kappa that has NO precomputed entry (checked dynamically
# below, not hardcoded per case) -- e.g. sub-case 3c, whose Neumann row gives
# it a different kappa than the standard TST matrix at the same N (confirmed
# in your run.log: 437.70 at N=16 vs 116.46 everywhere else). 1000 computes
# live in a few seconds regardless of kappa, per the capped-fit path.
QSVT_UNCACHED_FALLBACK_DEGREE: int = 1000
QSVT_MAX_DEGREE_FALLBACK: int = 5000

# ── HHL / VQLS configuration ─────────────────────────────────────────────────
HHL_EPSILON: float = 0.01
VQLS_SEED: int = 42

# ── Timing repeats ───────────────────────────────────────────────────────────
# Repeats give a mean/std rather than a single sample, which is what makes a
# timing number defensible on a shared node. Only the classical solver is
# repeated by default: repeating the quantum solvers would multiply the total
# sweep wall time by the same factor for little statistical gain.
THOMAS_TIMING_REPEATS: int = 10

# ── Parallelisation ──────────────────────────────────────────────────────────
# Each worker process executes one (case_family, N) work unit. Aer simulations
# are already OpenMP-threaded internally, so the worker count should not exceed
# the cores actually requested from PBS -- see the note in main().
MAX_WORKERS_DEFAULT: int = 4

# GPU preference: read from environment so the PBS script can override.
# Set QUANTUM_PDE_USE_GPU=0 to force CPU execution.
_USE_GPU: bool = os.environ.get("QUANTUM_PDE_USE_GPU", "1") != "0"


# ============================================================================
#  Result dataclass
# ============================================================================

@dataclass
class RunResult:
    """
    One row of the results table.

    Fields are grouped by purpose. Everything after `notes` is optional and
    defaults to None so that a solver which cannot supply a given metric
    simply leaves it blank rather than forcing a placeholder.
    """
    # ── Identity ─────────────────────────────────────────────────────────────
    case:           str
    solver:         str
    N:              int
    kappa:          float

    # ── Core accuracy / cost ─────────────────────────────────────────────────
    max_rel_err:    Optional[float]   # % vs reference, near-zero nodes masked
    max_abs_err:    Optional[float]
    residual:       Optional[float]   # ||Au - b|| / ||b||
    wall_time_s:    float
    converged:      bool
    notes:          str = ""

    # ── Additional accuracy norms ────────────────────────────────────────────
    # L2 is the conventional norm for PDE convergence studies; max-norm alone
    # is noisier and unusual to report on its own.
    rel_l2_err:     Optional[float] = None   # ||u-u_ref||_2 / ||u_ref||_2
    rms_err:        Optional[float] = None

    # ── Timing statistics ────────────────────────────────────────────────────
    wall_time_mean_s: Optional[float] = None
    wall_time_std_s:  Optional[float] = None
    n_timing_repeats: int = 1

    # ── Circuit metrics ──────────────────────────────────────────────────────
    # NOT recoverable from a saved solution vector. Populated only where the
    # underlying solver module exposes them; see the note in _run_qsvt.
    n_qubits:            Optional[int]   = None   # data-register qubits
    circuit_depth:       Optional[int]   = None
    circuit_depth_t:     Optional[int]   = None   # after transpilation
    n_gates_total:       Optional[int]   = None
    n_gates_2q:          Optional[int]   = None   # CX count: the cost driver
    success_probability: Optional[float] = None   # post-selection probability

    # ── Solver-specific internals ────────────────────────────────────────────
    qsvt_degree:        Optional[int]   = None    # degree actually solved
    qsvt_max_degree:    Optional[int]   = None    # cap requested (cache key!)
    qsvt_kappa_eff:     Optional[float] = None
    qsvt_phases_cached: Optional[bool]  = None
    vqls_final_cost:    Optional[float] = None
    vqls_n_layers:      Optional[int]   = None
    vqls_n_restarts:    Optional[int]   = None
    hhl_epsilon:        Optional[float] = None
    hhl_scale_c:        Optional[float] = None    # proportionality constant

    # ── Reproducibility ──────────────────────────────────────────────────────
    random_seed:    Optional[int] = None


# ============================================================================
#  Logging helpers
# ============================================================================

def _banner(msg: str) -> None:
    """Print a clearly visible section banner to stdout and the log file."""
    sep = "=" * 72
    log.info(sep)
    log.info(f"  {msg}")
    log.info(sep)


def _section(msg: str) -> None:
    """Print a sub-section separator."""
    sep = "-" * 72
    log.info(sep)
    log.info(f"  {msg}")
    log.info(sep)


# ============================================================================
#  Error metrics
# ============================================================================

def _relative_residual(A: np.ndarray, u: np.ndarray, b: np.ndarray) -> float:
    """||Au - b|| / ||b||. Solver-agnostic and reference-free."""
    return float(np.linalg.norm(A @ u - b) / (np.linalg.norm(b) + 1e-300))


def _max_rel_err(u: np.ndarray, u_ref: np.ndarray, tol: float = 1e-10) -> float:
    """Maximum relative error (%), excluding near-zero reference nodes."""
    mask = np.abs(u_ref) > tol
    if not np.any(mask):
        return float("nan")
    return float(np.max(np.abs(u[mask] - u_ref[mask]) / np.abs(u_ref[mask])) * 100.0)


def _max_abs_err(u: np.ndarray, u_ref: np.ndarray) -> float:
    """Maximum absolute error."""
    return float(np.max(np.abs(u - u_ref)))


def _rel_l2_err(u: np.ndarray, u_ref: np.ndarray) -> float:
    """Relative L2 error -- the standard norm for PDE convergence studies."""
    return float(np.linalg.norm(u - u_ref) / (np.linalg.norm(u_ref) + 1e-300))


def _rms_err(u: np.ndarray, u_ref: np.ndarray) -> float:
    """Root-mean-square error."""
    return float(np.sqrt(np.mean((u - u_ref) ** 2)))


def _accuracy_fields(u: np.ndarray, u_ref: Optional[np.ndarray]) -> dict:
    """
    All reference-based accuracy metrics as a kwargs dict, for splatting into
    RunResult(...). Returns {} when there is no reference (e.g. sub-case 3b's
    Thomas row), so every call site can splat unconditionally.
    """
    if u_ref is None:
        return {}
    return {
        "max_rel_err": _max_rel_err(u, u_ref),
        "max_abs_err": _max_abs_err(u, u_ref),
        "rel_l2_err":  _rel_l2_err(u, u_ref),
        "rms_err":     _rms_err(u, u_ref),
    }


# ============================================================================
#  Solution archiving
# ============================================================================

def _save_solution(
    case:    str,
    solver:  str,
    N:       int,
    x:       np.ndarray,
    u:       np.ndarray,
    u_exact: Optional[np.ndarray],
    b:       Optional[np.ndarray] = None,
    A:       Optional[np.ndarray] = None,
) -> None:
    """
    Save one solution to a compressed NPZ.

    Storing the RHS and the pointwise residual vector alongside the solution
    allows per-node error and residual analysis to be done post-hoc without
    re-running the sweep, which is the whole point of an expensive HPC job.
    """
    fname = RESULTS_DIR / f"solutions_{case}_{solver}_N{N}.npz"
    arrays: dict[str, np.ndarray] = {"x": x, "u_solver": u}
    if u_exact is not None:
        arrays["u_exact"] = u_exact
    if b is not None:
        arrays["b"] = b
    if A is not None and b is not None:
        arrays["residual_vec"] = A @ u - b
    np.savez_compressed(fname, **arrays)


def _record(
    results:       list[RunResult],
    all_solutions: dict,
    case_id:       str,
    solver:        str,
    N:             int,
    kappa:         float,
    x:             np.ndarray,
    u:             Optional[np.ndarray],
    u_ref:         Optional[np.ndarray],
    A:             np.ndarray,
    b:             np.ndarray,
    residual:      float,
    wall:          float,
    converged:     bool,
    notes:         str = "",
    **extra,
) -> None:
    """
    Single funnel for appending a result and archiving its solution.

    Centralising this is what guarantees the accuracy norms, the NPZ archive
    and the consolidated solution dict stay consistent across all ~20 call
    sites -- previously each site repeated the logic and they had drifted
    (some passed A/b to _save_solution, some did not; none populated the L2
    norms at all).

    `u=None` records a failure row with no solution archived.
    """
    if u is None:
        results.append(RunResult(
            case=case_id, solver=solver, N=N, kappa=kappa,
            max_rel_err=None, max_abs_err=None, residual=None,
            wall_time_s=wall, converged=False,
            notes=notes or "solver_error", **extra,
        ))
        return

    # Accuracy metrics (max_rel_err, max_abs_err, rel_l2_err, rms_err) come
    # from _accuracy_fields, which returns {} when there is no reference. They
    # are merged into the kwargs dict rather than passed alongside explicit
    # max_rel_err=/max_abs_err= arguments, which would be a duplicate keyword.
    kwargs: dict = {
        "max_rel_err": None,
        "max_abs_err": None,
    }
    kwargs.update(_accuracy_fields(u, u_ref))
    kwargs.update(extra)

    results.append(RunResult(
        case=case_id, solver=solver, N=N, kappa=kappa,
        residual=residual, wall_time_s=wall, converged=converged,
        notes=notes, **kwargs,
    ))
    _save_solution(case_id, solver, N, x, u, u_ref, b, A)
    all_solutions[f"{case_id}_{solver}_N{N}"] = {
        "x": x, "u": u, "u_exact": u_ref,
    }


def _save_all_solutions(all_solutions: dict) -> None:
    """
    Consolidated archive of every solution in one NPZ.

    The per-case files remain the primary record; this is a convenience for
    post-processing. Previously `all_solutions` was assembled, pickled back
    from every worker process, merged -- and then silently discarded.
    """
    if not all_solutions:
        return
    flat: dict[str, np.ndarray] = {}
    for key, entry in all_solutions.items():
        flat[f"{key}__x"] = entry["x"]
        flat[f"{key}__u"] = entry["u"]
        if entry.get("u_exact") is not None:
            flat[f"{key}__u_exact"] = entry["u_exact"]
    path = RESULTS_DIR / "all_solutions.npz"
    np.savez_compressed(path, **flat)
    log.info("Consolidated solutions saved to %s (%d entries)",
             path, len(all_solutions))


# ============================================================================
#  Problem builders
# ============================================================================

def _build_tst(N: int) -> np.ndarray:
    """N x N Toeplitz symmetric tridiagonal matrix: main diag -2, off-diag +1."""
    return (-2.0 * np.eye(N)
            + np.diag(np.ones(N - 1), 1)
            + np.diag(np.ones(N - 1), -1))


def _grid(N: int) -> tuple[np.ndarray, float]:
    """
    Interior grid for pure-Dirichlet problems: N unknowns at x = h, 2h, ..., Nh
    with h = 1/(N+1). The boundaries x=0 and x=1 are NOT unknowns.

    Sub-case 3c does NOT use this grid -- see the comment in its section.
    """
    dx = 1.0 / (N + 1)
    x = np.arange(1, N + 1) * dx
    return x, dx


def _kappa(A: np.ndarray) -> float:
    """
    Condition number lambda_max / lambda_min.

    np.linalg.eigvalsh reads only one triangle of its argument, so handing it a
    non-symmetric matrix silently returns the spectrum of a DIFFERENT operator.
    The symmetry check and singular-value fallback make that failure loud.
    """
    if np.allclose(A, A.T, atol=1e-12):
        eigs = np.abs(np.linalg.eigvalsh(A))
    else:
        log.warning("_kappa: matrix is not symmetric; using singular values.")
        eigs = np.linalg.svd(A, compute_uv=False)
    return float(eigs.max() / eigs.min())


# ── Generic source functions and their analytical solutions ──────────────────

def f_sin(x): return np.sin(np.pi * x)
def f_lin(x): return 10.0 * x
def f_hev(x): return np.where(x >= 0.5, 1.0, -1.0)


def u_sin(x): return -np.sin(np.pi * x) / np.pi**2
def u_lin(x): return 5.0 * x * (x**2 - 1.0) / 3.0
def u_hev(x):
    return np.where(x < 0.5, -x**2 / 2.0 + x / 4.0,
                              x**2 / 2.0 - 3.0 * x / 4.0 + 1.0 / 4.0)


# ── HET source functions ─────────────────────────────────────────────────────

def _het_gaussian_source(x: np.ndarray, V_d: float = 300.0,
                         L: float = 0.025, sigma: float = 0.005) -> np.ndarray:
    """
    Sub-case 3b source: Gaussian electron density profile,
        n_e(x) = n_0 exp(-(xL - x0)^2 / (2 sigma^2))
        f(x)   = -(e / eps0) n_e

    Ion density is treated as a uniform background (simplified 1-D model).

    NOTE: the returned values are in physical units (order 1e9), NOT normalised.
    An earlier docstring claimed "normalised so that max|f| = 1", which was
    simply false; any normalisation required by a quantum solver interface
    happens inside that solver, not here.
    """
    e    = 1.602e-19
    eps0 = 8.854e-12
    n0   = 1e17            # m^-3, representative channel density
    x0   = 0.6 * L         # peak near the exit plane
    n_e  = n0 * np.exp(-((x * L - x0)**2) / (2 * sigma**2))
    return -(e / eps0) * n_e


def _het_linear_source(x: np.ndarray) -> np.ndarray:
    """Sub-case 3a source: linear density profile, f(x) = 2x - 1."""
    return 2.0 * x - 1.0


def _het_neumann_dirichlet_source(x: np.ndarray, sigma_norm: float = 0.2,
                                  x0: float = 0.6) -> np.ndarray:
    """
    Sub-case 3c source: normalised Gaussian on the unit interval,
        f(x) = -exp(-(x - x0)^2 / (2 sigma_norm^2))

    sigma_norm is ALREADY in normalised (unit-interval) units. An earlier
    version took a physical `sigma` and divided by `L`; called with sigma=0.2
    and L defaulting to 0.025 that gave an effective width of 8 -- an almost
    flat source -- while the analytical reference used 0.2. The discrete and
    exact problems were therefore not the same problem, which was one cause of
    the anomalous ~60% Thomas error. Signature and defaults now match
    _het_neumann_dirichlet_exact exactly.
    """
    return -np.exp(-((x - x0) ** 2) / (2.0 * sigma_norm ** 2))


def _het_neumann_dirichlet_exact(x: np.ndarray, sigma_norm: float = 0.2,
                                 x0: float = 0.6) -> np.ndarray:
    """
    Reference solution for  phi'' = f(x),  phi'(0) = 0,  phi(1) = 0,
    with f(x) = -exp(-(x - x0)^2 / (2 sigma_norm^2)).

    By direct integration of phi'' = f:
        phi'(x) = phi'(0) + int_0^x f(t) dt = int_0^x f(t) dt     [Neumann]
        phi(x)  = phi(0)  + int_0^x phi'(s) ds
    and the additive constant is fixed by phi(1) = 0.

    NOTE ON SIGNS: an earlier docstring wrote both integrals with a leading
    minus sign. That was wrong and the CODE below (no minus signs) is correct
    for phi'' = f -- which is consistent with the discrete system Au = h^2 f.
    Do not "fix" the code to match the old docstring.

    Quadrature is used because the Gaussian has no closed-form finite-limit
    integral; the fine grid is far denser than any N in the sweep, so the
    quadrature error is negligible relative to the discretisation error.
    """
    from scipy.integrate import cumulative_trapezoid

    x_fine = np.linspace(0.0, 1.0, 10000)
    f_fine = -np.exp(-((x_fine - x0)**2) / (2.0 * sigma_norm**2))

    dphi_fine = cumulative_trapezoid(f_fine, x_fine, initial=0.0)   # phi'
    phi_fine  = cumulative_trapezoid(dphi_fine, x_fine, initial=0.0)  # phi
    phi_fine -= phi_fine[-1]                                        # phi(1)=0

    return np.interp(x, x_fine, phi_fine)


def _build_3c_system(N: int, sigma_norm: float = 0.2
                     ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Assemble sub-case 3c: phi'' = f, phi'(0) = 0 (Neumann), phi(1) = 0.

    Returns (x, h, A, b).

    GRID. A Neumann condition at x=0 makes phi(0) an UNKNOWN, so the node set
    must include it. Unknowns are phi_0..phi_{N-1} at x_i = i*h with h = 1/N;
    x_N = 1 is the Dirichlet boundary (phi_N = 0, absorbed into the last row).
    This deliberately differs from _grid(N), the pure-Dirichlet interior grid
    (h = 1/(N+1), starting at h), which has no node at x=0 -- applying a
    Neumann row to index 0 of that grid imposes the condition at x=h, not x=0.

    NEUMANN ROW. Ghost point phi_{-1} = phi_1 from the centred difference
    phi'(0) = (phi_1 - phi_{-1}) / 2h = 0. The node-0 equation
        phi_{-1} - 2 phi_0 + phi_1 = h^2 f_0
    becomes  2 phi_1 - 2 phi_0 = h^2 f_0. Halving that row gives
        -phi_0 + phi_1 = h^2 f_0 / 2
    i.e. A[0,0] = -1, A[0,1] = +1, b[0] = h^2 f_0 / 2, which is SYMMETRIC
    (A[0,1] = A[1,0] = 1). The unhalved form (A[0,1]=2, A[1,0]=1) has the same
    solution but is not Hermitian: HHL and QSVT would not be valid on it, and
    the eigvalsh-based kappa would silently describe a different matrix.
    """
    h = 1.0 / N
    x = np.arange(N) * h

    A = _build_tst(N).astype(float)
    A[0, 0] = -1.0
    A[0, 1] = +1.0

    b = h**2 * _het_neumann_dirichlet_source(x, sigma_norm=sigma_norm)
    b[0] *= 0.5          # the same halving applied to the RHS

    return x, h, A, b


# ============================================================================
#  Solver wrappers
#
#  Every wrapper returns a uniform 3-element core -- (u, residual, wall) --
#  plus solver-specific extras, and every one catches its own exceptions and
#  returns u=None on failure. That last point matters: a wrapper that lets an
#  exception escape takes down the whole work unit, losing the OTHER solvers'
#  results for that (case, N) too.
# ============================================================================

def _run_thomas(A: np.ndarray, b: np.ndarray,
                repeats: int = THOMAS_TIMING_REPEATS
                ) -> tuple[Optional[np.ndarray], float, float, float, float]:
    """
    Thomas algorithm (tridiagonal LU without pivoting).

    Returns (u, residual, wall_best_s, wall_mean_s, wall_std_s).

    Diagonals are extracted FROM the supplied matrix rather than assumed to be
    the standard (-2, +1) TST stencil. The previous version hardcoded -2/+1 and
    ignored `A` entirely, so for sub-case 3c -- whose Neumann row differs -- it
    silently solved the WRONG system and was then compared against 3c's exact
    solution. That was the dominant cause of the anomalous ~60% error.

    Sub- and super-diagonals are tracked separately; the previous version used
    one array for both, which is only correct for symmetric systems.

    The solve is repeated `repeats` times for timing statistics. It is fast
    enough (microseconds) that this is free, and a mean/std is far more
    defensible than a single sample on a shared node.
    """
    try:
        timings: list[float] = []
        u = np.zeros(len(b))

        for _ in range(max(1, repeats)):
            N     = len(b)
            diag  = np.diag(A).astype(float).copy()
            lower = np.diag(A, k=-1).astype(float).copy()
            upper = np.diag(A, k=1).astype(float).copy()
            d     = np.asarray(b, dtype=float).copy()

            t0 = time.perf_counter()
            # Forward elimination
            for i in range(1, N):
                m        = lower[i - 1] / diag[i - 1]
                diag[i] -= m * upper[i - 1]
                d[i]    -= m * d[i - 1]
            # Back substitution
            u = np.zeros(N)
            u[-1] = d[-1] / diag[-1]
            for i in range(N - 2, -1, -1):
                u[i] = (d[i] - upper[i] * u[i + 1]) / diag[i]
            timings.append(time.perf_counter() - t0)

        t_arr = np.array(timings)
        return (u, _relative_residual(A, u, b),
                float(t_arr.min()), float(t_arr.mean()), float(t_arr.std()))

    except Exception as exc:
        log.warning("    Thomas failed: %s", exc)
        return None, float("nan"), 0.0, 0.0, 0.0

def _hhl_worker(A, b, epsilon, q):
    from solvers.quantum.hhl_1d import hhl_solve_system
    q.put(hhl_solve_system(A, b, epsilon))

def _run_hhl(A: np.ndarray, b: np.ndarray, N: int,
             epsilon: float = HHL_EPSILON,
             timeout_s: float = 3600.0,
             ) -> tuple[Optional[np.ndarray], float, float, bool, float]:
    """
    HHL via the project module solvers/quantum/hhl_1d.py, with a HARD wall-clock timeout.

    Returns (u, residual, wall_s, converged, scale_c).

    `scale_c` is the proportionality-recovery constant. It is NOT derivable
    from the returned solution vector afterwards, so it is propagated rather
    than discarded.

    Unlike QSVT's post-hoc warning, this actually terminates the underlying
    process on timeout -- statevector-simulated HHL scales with the clock
    register size (which grows with kappa) and has no existing guard, so a
    large-kappa case can otherwise block a worker indefinitely.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_hhl_worker, args=(A, b, epsilon, q))
    t0 = time.perf_counter()
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        p.terminate()
        p.join()
        log.warning("    HHL: killed after exceeding %.0fs timeout (N=%d).",
                   timeout_s, N)
        return None, float("nan"), time.perf_counter() - t0, False, float("nan")

    try:
        u, x_raw, c = q.get_nowait()

    except Exception as exc:
        log.warning("    HHL failed: %s", exc)
        return None, float("nan"), time.perf_counter() - t0, False, float("nan")

    wall = time.perf_counter() - t0
    return u, _relative_residual(A, u, b), wall, True, float(c)


def _run_vqls(A: np.ndarray, b: np.ndarray, N: int
              ) -> tuple[Optional[np.ndarray], float, float, bool, float, int, int]:
    """
    VQLS via the validated project module solvers/quantum/vqls_1d.py.

    Returns (u, residual, wall_s, converged, final_cost, n_layers, n_restarts).
    """
    try:
        from solvers.quantum.vqls_1d import vqls_solve_system, VQLSConfig1D

        n_qubits   = int(np.log2(N))
        n_layers   = max(6, 2 * n_qubits + 2)
        n_restarts = max(3, 2 * n_qubits)

        vqls_cfg = VQLSConfig1D(
            n_layers    = n_layers,
            max_iter    = 500,
            tol         = 1e-6,
            random_seed = VQLS_SEED,
            verbose     = False,
            n_restarts  = n_restarts,
        )

        t0 = time.perf_counter()
        result = vqls_solve_system(A, b, config=vqls_cfg)
        wall = time.perf_counter() - t0

        final_cost = float(result.final_cost)
        if final_cost > 0.1:
            log.warning(
                "    VQLS cost=%.2e exceeds convergence threshold (0.1); "
                "solution likely non-physical for N=%d. Retained for reporting.",
                final_cost, N,
            )

        return (result.u, _relative_residual(A, result.u, b), wall,
                bool(result.optimiser_success), final_cost,
                n_layers, n_restarts)

    except Exception as exc:
        log.warning("    VQLS failed: %s", exc)
        return None, float("nan"), 0.0, False, float("nan"), -1, -1


def _resolve_qsvt_max_degree(kappa: float, epsilon: float, N: int) -> Optional[int]:
    """
    Use the precomputed phases if they exist for this exact kappa; otherwise
    fall back to a cheap cap rather than skip QSVT or risk an expensive/
    uncapped live solve.

    QSVT_MAX_DEGREE_BY_N assumes every case at a given N shares one kappa --
    true for Sections 1, 1b, 3a, 3b (identical TST matrix), false for 3c
    (Neumann row -> different kappa at the same N). Checking the disk cache
    directly, instead of hardcoding a second per-case table, means this stays
    correct even if 3c's kappa changes, and it needs no special-casing here.
    """
    import solvers.quantum.qsp_angles as qsp_angles

    candidate = QSVT_MAX_DEGREE_BY_N.get(N, QSVT_MAX_DEGREE_FALLBACK)
    key = (round(kappa, 4), round(epsilon, 8), "auto",
           candidate if candidate is not None else -1)
    if qsp_angles._load_disk(key) is not None:
        return candidate                       # cache hit: use the real cap
    return QSVT_UNCACHED_FALLBACK_DEGREE        # cache miss: cheap fallback


def _build_qsvt_config(max_degree: Optional[int]):
    """Construct a QSVTConfig1D, passing only the fields it actually declares."""
    from solvers.quantum.qsvt_1d import QSVTConfig1D

    desired = {
        "epsilon":      HHL_EPSILON,
        "angle_method": "auto",
        "max_degree":   max_degree,
    }
    try:
        declared = {f.name for f in dataclasses.fields(QSVTConfig1D)}
    except TypeError:
        return QSVTConfig1D(**desired)

    accepted = {k: v for k, v in desired.items() if k in declared}
    dropped  = set(desired) - set(accepted)
    if dropped:
        log.warning("QSVTConfig1D does not declare %s; not passed.", dropped)
    return QSVTConfig1D(**accepted)


def _run_qsvt(A: np.ndarray, b: np.ndarray, N: int, kappa: float,
              time_limit: Optional[float]
              ) -> tuple[Optional[np.ndarray], float, float, bool, int, int, Optional[int]]:
    max_deg = _resolve_qsvt_max_degree(kappa, HHL_EPSILON, N)

    if N > QSVT_MAX_N:
        log.info("    QSVT: skipping N=%d > QSVT_MAX_N=%d", N, QSVT_MAX_N)
        return None, float("nan"), 0.0, False, -1, -1, max_deg

    try:
        from solvers.quantum.qsvt_1d import qsvt_solve_system

        cfg = _build_qsvt_config(max_deg)

        t0 = time.perf_counter()
        result = qsvt_solve_system(A, b, config=cfg)
        wall = time.perf_counter() - t0

        if time_limit is not None and wall > time_limit:
            log.warning("    QSVT: completed but exceeded soft time limit "
                        "(%.1fs > %.1fs). Result retained.", wall, time_limit)

        u = result.u
        converged = (bool(getattr(result, "converged", True))
                    and u is not None and not np.any(np.isnan(u)))
        degree = getattr(result, "degree", getattr(result, "polynomial_degree", -1))
        depth  = getattr(result, "circuit_depth", -1)

        return u, _relative_residual(A, u, b), wall, converged, int(degree), int(depth), max_deg

    except Exception as exc:
        log.warning("    QSVT failed: %s", exc)
        return None, float("nan"), 0.0, False, -1, -1, max_deg


# ============================================================================
#  Per-case solver driver
# ============================================================================

def _run_all_solvers(
    case_id:       str,
    N:             int,
    x:             np.ndarray,
    A:             np.ndarray,
    b:             np.ndarray,
    u_exact:       Optional[np.ndarray],
    kappa:         float,
    skip_qsvt:     bool,
    results:       list[RunResult],
    all_solutions: dict,
    reference:     str = "exact",
) -> None:
    """
    Run Thomas -> HHL -> VQLS -> QSVT on one assembled system and record all.

    `reference` selects what the quantum solvers are scored against:
      "exact"  -- the supplied analytical u_exact (most cases)
      "thomas" -- the Thomas solution, for cases with no closed form (3b)

    Factoring this out removes ~15 near-duplicate blocks from the original
    case runners. Those blocks had drifted apart from one another over time,
    which is precisely how the missing accuracy norms and inconsistent NPZ
    archiving crept in.
    """
    n_data_qubits = int(np.log2(N)) if N > 0 and (N & (N - 1)) == 0 else None

    # ── Thomas (classical reference) ─────────────────────────────────────────
    u_T, res_T, t_T, t_T_mean, t_T_std = _run_thomas(A, b)
    if u_T is None:
        log.error("    Thomas FAILED for %s N=%d -- skipping case.", case_id, N)
        return

    # Score Thomas against the analytical solution when one exists.
    ref_for_thomas = u_exact
    if ref_for_thomas is not None:
        log.info("    Thomas  MaxRelErr=%7.3f%%  Residual=%.3e  Time=%.6fs",
                 _max_rel_err(u_T, ref_for_thomas), res_T, t_T)
    else:
        log.info("    Thomas  Residual=%.3e  Time=%.6fs  (reference solution)",
                 res_T, t_T)

    _record(results, all_solutions, case_id, "Thomas", N, kappa,
            x, u_T, ref_for_thomas, A, b, res_T, t_T, True,
            notes="" if ref_for_thomas is not None else "reference_no_exact",
            wall_time_mean_s=t_T_mean, wall_time_std_s=t_T_std,
            n_timing_repeats=THOMAS_TIMING_REPEATS)

    # From here on, the quantum solvers are scored against `u_ref`.
    u_ref = u_T if reference == "thomas" else u_exact
    ref_note = "rel_vs_thomas" if reference == "thomas" else ""

    # ── HHL ──────────────────────────────────────────────────────────────────
    u_H, res_H, t_H, conv_H, c_H = _run_hhl(A, b, N)
    if u_H is not None:
        if u_ref is not None:
            log.info("    HHL     MaxRelErr=%7.3f%%  Residual=%.3e  Time=%.3fs",
                     _max_rel_err(u_H, u_ref), res_H, t_H)
        else:
            log.info("    HHL     Residual=%.3e  Time=%.3fs", res_H, t_H)
    _record(results, all_solutions, case_id, "HHL", N, kappa,
            x, u_H, u_ref, A, b, res_H, t_H, conv_H, notes=ref_note,
            n_qubits=n_data_qubits, hhl_epsilon=HHL_EPSILON,
            hhl_scale_c=c_H if u_H is not None else None)

    # ── VQLS ─────────────────────────────────────────────────────────────────
    u_V, res_V, t_V, conv_V, cost_V, lay_V, res_ct_V = _run_vqls(A, b, N)
    if u_V is not None:
        if u_ref is not None:
            log.info("    VQLS    MaxRelErr=%7.3f%%  Residual=%.3e  Time=%.3fs  "
                     "cost=%.2e", _max_rel_err(u_V, u_ref), res_V, t_V, cost_V)
        else:
            log.info("    VQLS    Residual=%.3e  Time=%.3fs  cost=%.2e",
                     res_V, t_V, cost_V)
    _record(results, all_solutions, case_id, "VQLS", N, kappa,
            x, u_V, u_ref, A, b, res_V, t_V, conv_V, notes=ref_note,
            n_qubits=n_data_qubits, vqls_final_cost=cost_V,
            vqls_n_layers=lay_V if u_V is not None else None,
            vqls_n_restarts=res_ct_V if u_V is not None else None,
            random_seed=VQLS_SEED)

    # ── QSVT ─────────────────────────────────────────────────────────────────
    if skip_qsvt:
        return
    u_Q, res_Q, t_Q, conv_Q, deg_Q, dep_Q, cap_Q = _run_qsvt(
        A, b, N, kappa, QSVT_TIME_LIMIT_S)

    if u_Q is not None and u_ref is not None:
        log.info("    QSVT    MaxRelErr=%7.3f%%  Residual=%.3e  Time=%.1fs  "
                 "deg=%d  depth=%d",
                 _max_rel_err(u_Q, u_ref), res_Q, t_Q, deg_Q, dep_Q)
    _record(results, all_solutions, case_id, "QSVT", N, kappa,
            x, u_Q, u_ref, A, b, res_Q, t_Q, conv_Q,
            notes=ref_note or ("skipped_or_failed" if u_Q is None else ""),
            n_qubits=n_data_qubits,
            circuit_depth=dep_Q if u_Q is not None else None,
            qsvt_degree=deg_Q if u_Q is not None else None,
            qsvt_max_degree=cap_Q)


# ============================================================================
#  Case runners
# ============================================================================

def run_1d_generic_poisson_single_N(
    N: int, skip_qsvt: bool, results: list[RunResult], all_solutions: dict,
) -> None:
    """Section 1: generic Poisson, homogeneous Dirichlet BCs, three sources."""
    _banner(f"SECTION 1 - Generic Poisson (fS, fL, fH), homogeneous BCs, N={N}")

    source_map = {
        "fS": (f_sin, u_sin),
        "fL": (f_lin, u_lin),
        "fH": (f_hev, u_hev),
    }

    for src_key, (src_fn, exact_fn) in source_map.items():
        _section(f"Source: {src_key}  (N={N})")

        x, dx   = _grid(N)
        A       = _build_tst(N)
        b       = dx**2 * src_fn(x)
        kappa   = _kappa(A)
        u_exact = exact_fn(x)
        case_id = f"1D_Poisson_{src_key}_hom"

        log.info("  N=%3d  kappa=%.2f  case=%s", N, kappa, case_id)

        _run_all_solvers(case_id, N, x, A, b, u_exact, kappa,
                         skip_qsvt, results, all_solutions)


def run_1d_generic_poisson_nonhom_single_N(
    N: int, skip_qsvt: bool, results: list[RunResult], all_solutions: dict,
) -> None:
    """
    Section 1b: generic Poisson, fS source, non-homogeneous Dirichlet BCs.

    u'' = sin(pi x),  u(0) = alpha,  u(1) = beta
      =>  u(x) = -sin(pi x)/pi^2 + (beta - alpha) x + alpha
    """
    _banner(f"SECTION 1b - Generic Poisson (fS), non-homogeneous BCs, N={N}")

    alpha, beta = 1.0, 2.0

    x, dx = _grid(N)
    A     = _build_tst(N)
    b     = dx**2 * f_sin(x)
    b[0]  -= alpha      # u(0) = alpha absorbed into the first row
    b[-1] -= beta       # u(1) = beta  absorbed into the last row
    kappa   = _kappa(A)
    u_exact = -np.sin(np.pi * x) / np.pi**2 + (beta - alpha) * x + alpha
    case_id = "1D_Poisson_fS_nonhom"

    log.info("  N=%3d  kappa=%.2f  alpha=%.1f  beta=%.1f", N, kappa, alpha, beta)

    _run_all_solvers(case_id, N, x, A, b, u_exact, kappa,
                     skip_qsvt, results, all_solutions)


def run_1d_het_single_N(
    N: int, skip_qsvt: bool, results: list[RunResult], all_solutions: dict,
) -> None:
    """
    Section 2: HET axial Poisson, three sub-cases.

      3a  linear profile, homogeneous Dirichlet, analytical reference
      3b  Gaussian profile, V_d = 300 V anode, Dirichlet, Thomas reference
      3c  Gaussian profile, Neumann(x=0) - Dirichlet(x=1), quadrature reference
    """
    _banner(f"SECTION 2 - HET Axial Poisson, N={N}")

    # ── Sub-case 3a: linear profile, homogeneous BCs ─────────────────────────
    # u'' = 2x - 1, u(0) = u(1) = 0  =>  u(x) = x^3/3 - x^2/2 + x/6
    _section(f"Sub-case 3a: linear profile, homogeneous BCs  (N={N})")

    x, dx   = _grid(N)
    A       = _build_tst(N)
    b       = dx**2 * _het_linear_source(x)
    kappa   = _kappa(A)
    u_exact = x**3 / 3.0 - x**2 / 2.0 + x / 6.0
    case_id = "HET_1D_3a_linear_hom"

    log.info("  N=%3d  kappa=%.2f  sub-case=3a", N, kappa)
    _run_all_solvers(case_id, N, x, A, b, u_exact, kappa,
                     skip_qsvt, results, all_solutions)

    # ── Sub-case 3b: Gaussian profile, V_d = 300 V ───────────────────────────
    # No closed-form solution, so the Thomas result is the reference.
    _section(f"Sub-case 3b: Gaussian profile, V_d=300V, Dirichlet BCs  (N={N})")

    V_d, L  = 300.0, 0.025
    x, dx   = _grid(N)
    A       = _build_tst(N)
    b       = dx**2 * _het_gaussian_source(x, V_d=V_d, L=L)
    b[0]   -= V_d       # phi(0) = V_d  (anode)
    b[-1]  -= 0.0       # phi(1) = 0    (cathode)
    kappa   = _kappa(A)
    case_id = "HET_1D_3b_gaussian_Vd300"

    log.info("  N=%3d  kappa=%.2f  sub-case=3b  V_d=%.0fV", N, kappa, V_d)
    _run_all_solvers(case_id, N, x, A, b, None, kappa,
                     skip_qsvt, results, all_solutions, reference="thomas")

    # Electric field diagnostic (derived quantity of physical interest).
    thomas_key = f"{case_id}_Thomas_N{N}"
    if thomas_key in all_solutions:
        u_T = all_solutions[thomas_key]["u"]
        log.info("    Peak |E| (Thomas) = %.3e V/m",
                 float(np.max(np.abs(-np.gradient(u_T, x)))))

    # ── Sub-case 3c: Gaussian profile, Neumann-Dirichlet BCs ─────────────────
    _section(f"Sub-case 3c: Gaussian profile, Neumann-Dirichlet BCs  (N={N})")
    log.info("  BCs: phi'(0)=0 (Neumann), phi(1)=0 (Dirichlet)")
    log.info("  Reference: quadrature of the double integral of the source")

    sigma_norm  = 0.2
    x, h, A, b  = _build_3c_system(N, sigma_norm=sigma_norm)
    kappa       = _kappa(A)
    u_exact     = _het_neumann_dirichlet_exact(x, sigma_norm=sigma_norm)
    case_id     = "HET_1D_3c_gaussian_NeumannDirichlet"

    log.info("  N=%3d  kappa=%.2f  sub-case=3c  h=%.5f", N, kappa, h)
    _run_all_solvers(case_id, N, x, A, b, u_exact, kappa,
                     skip_qsvt, results, all_solutions)


# ============================================================================
#  Result serialisation
# ============================================================================

def _save_results(results: list[RunResult]) -> None:
    """Write the results table to JSON and CSV."""
    json_path = RESULTS_DIR / "results_full.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    log.info("Results saved to %s (%d rows)", json_path, len(results))

    csv_path = RESULTS_DIR / "results_summary.csv"
    if results:
        # Field order comes from the dataclass definition, not from results[0],
        # so the header is stable even if the first row is a failure row.
        fieldnames = [f.name for f in dataclasses.fields(RunResult)]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
        log.info("Results saved to %s", csv_path)


def _save_run_metadata(N_values: list[int], skip_qsvt: bool,
                       max_workers: int) -> None:
    """
    Environment and run-configuration provenance.

    Written BEFORE the sweep starts so that it survives a crash or a walltime
    kill; without it a partial result set is not reproducible.
    """
    meta: dict = {
        # -- Environment ------------------------------------------------------
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname":    platform.node(),
        "platform":    platform.platform(),
        "python":      sys.version,
        "numpy":       np.__version__,
        "cpu_count":   os.cpu_count(),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "use_gpu":     _USE_GPU,
        "pbs_jobid":   os.environ.get("PBS_JOBID"),
        "pbs_queue":   os.environ.get("PBS_QUEUE"),
        # -- Run configuration (not derivable from the results table) ---------
        "N_values":            N_values,
        "skip_qsvt":           skip_qsvt,
        "max_workers":         max_workers,
        "qsvt_max_n":          QSVT_MAX_N,
        "qsvt_time_limit_s":   QSVT_TIME_LIMIT_S,
        "qsvt_max_degree_by_N": {str(k): v for k, v in QSVT_MAX_DEGREE_BY_N.items()},
        "qsvt_uncached_fallback_degree": QSVT_UNCACHED_FALLBACK_DEGREE,
        "qsvt_max_degree_fallback": QSVT_MAX_DEGREE_FALLBACK,
        "hhl_epsilon":         HHL_EPSILON,
        "vqls_seed":           VQLS_SEED,
        "thomas_timing_repeats": THOMAS_TIMING_REPEATS,
    }

    for mod in ("qiskit", "qiskit_aer", "pyqsp", "scipy"):
        try:
            meta[mod] = __import__(mod).__version__
        except Exception:
            meta[mod] = "not installed"

    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        meta["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip())
    except Exception:
        meta["git_commit"] = "unknown"
        meta["git_dirty"]  = None

    with open(RESULTS_DIR / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Run metadata saved to %s", RESULTS_DIR / "run_metadata.json")


# ============================================================================
#  Work unit dispatch
# ============================================================================

def _execute_work_unit(work_type: str, N: int, skip_qsvt: bool
                       ) -> tuple[list[RunResult], dict]:
    """
    Execute one (case_family, N) work unit and return its results.

    Designed to be called from a ProcessPoolExecutor worker. Each invocation is
    self-contained: Qiskit circuit objects and Aer backend state are not safely
    shareable across processes, so results are accumulated locally and returned
    to the parent for merging.
    """
    results: list[RunResult] = []
    solutions: dict = {}

    dispatch = {
        "generic_poisson":        run_1d_generic_poisson_single_N,
        "generic_poisson_nonhom": run_1d_generic_poisson_nonhom_single_N,
        "het_1d":                 run_1d_het_single_N,
    }

    fn = dispatch.get(work_type)
    if fn is None:
        log.error("Unknown work_type '%s'; skipping.", work_type)
        return results, solutions

    fn(N, skip_qsvt, results, solutions)
    return results, solutions


# ============================================================================
#  Main entry point
# ============================================================================

def main() -> None:
    """
    Entry point for the full HPC benchmark sweep.

    Execution strategy
    ------------------
    Independent (case_family, N) combinations are dispatched to a
    ProcessPoolExecutor. Each worker builds its own Aer backend.

    IMPORTANT -- worker count vs requested cores. Aer simulations are already
    OpenMP-threaded internally, so `--max-workers` should not exceed the ncpus
    actually requested in the PBS script. The default here (4) matches
    `#PBS -l select=1:ncpus=4` in hpc/submit_hpc_1D.sh. Raising one without raising
    the other oversubscribes the node and slows the sweep down.

    On a GPU node, use `--max-workers 1` with QUANTUM_PDE_USE_GPU=1 to
    serialise execution through the single GPU and avoid CUDA context
    conflicts between processes.
    """
    parser = argparse.ArgumentParser(
        description="Full 1-D HPC benchmark sweep for quantum PDE solvers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--max-n", type=int, default=max(N_VALUES_ALL),
        help=f"Largest N to include. The full sweep is {N_VALUES_ALL}; "
             f"use e.g. --max-n 16 for a fast validation pass "
             f"(default: {max(N_VALUES_ALL)}, i.e. the whole sweep).",
    )
    parser.add_argument(
        "--skip-qsvt", action="store_true",
        help="Omit QSVT from all cases. Use for a rapid validation sweep or "
             "if the QSVT module is unavailable.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=MAX_WORKERS_DEFAULT,
        help=f"Parallel worker processes (default: {MAX_WORKERS_DEFAULT}). "
             f"Must not exceed the ncpus requested from PBS. Set to 1 for "
             f"serial execution (required on GPU).",
    )
    args = parser.parse_args()

    N_values = [n for n in N_VALUES_ALL if n <= args.max_n]
    if not N_values:
        parser.error(f"--max-n {args.max_n} excludes every N in {N_VALUES_ALL}.")

    # -- Resolve backend and report configuration ----------------------------
    backend = get_aer_backend(prefer_gpu=_USE_GPU)

    _banner("QUANTUM PDE SOLVER - FULL 1D HPC BENCHMARK RUN")
    log.info("  N values      : %s", N_values)
    log.info("  QSVT deg caps : %s", QSVT_MAX_DEGREE_BY_N)
    log.info("  Max workers   : %d", args.max_workers)
    log.info("  Output dir    : %s", RESULTS_DIR.resolve())
    log.info("  Python        : %s", sys.version.split()[0])
    log.info("  PID           : %d", os.getpid())
    log_backend_info(backend)

    if args.max_workers > (os.cpu_count() or 1):
        log.warning("--max-workers (%d) exceeds visible CPU count (%s); "
                    "this will oversubscribe the node.",
                    args.max_workers, os.cpu_count())

    # Provenance first, so it survives a crash or walltime kill.
    _save_run_metadata(N_values, args.skip_qsvt, args.max_workers)

    t_global_start = time.perf_counter()
    results: list[RunResult] = []
    all_solutions: dict = {}

    # -- Build the work unit list --------------------------------------------
    # Smallest N first: with only a handful of workers and a large spread in
    # per-unit cost (HHL/QSVT scale badly with kappa -- see Problem 2 below),
    # dispatching largest-N units first saturates every worker on the slowest
    # cases immediately and starves the fast, informative small-N validation
    # runs behind them. Ascending order guarantees N=4/8/16 complete and are
    # written to disk before any worker can get tied up on N=32/64.
    work_units = [
        (family, N, args.skip_qsvt)
        for N in sorted(N_values)
        for family in ("generic_poisson", "generic_poisson_nonhom", "het_1d")
    ]

    if args.max_workers == 1:
        log.info("Serial execution mode (max_workers=1).")
        for work_type, N, skip_qsvt in work_units:
            try:
                partial_results, partial_solutions = _execute_work_unit(
                    work_type, N, skip_qsvt)
                results.extend(partial_results)
                all_solutions.update(partial_solutions)
            except Exception as exc:
                log.error("Work unit failed: type=%s N=%d - %s",
                          work_type, N, exc, exc_info=True)
    else:
        log.info("Parallel execution: %d work units across %d workers.",
                 len(work_units), args.max_workers)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers,
            max_tasks_per_child=1,   # fresh process per work unit
        ) as executor:
            futures = {
                executor.submit(_execute_work_unit, wt, N, sq): (wt, N)
                for wt, N, sq in work_units
            }
            for future in concurrent.futures.as_completed(futures):
                work_type, N = futures[future]
                try:
                    partial_results, partial_solutions = future.result()
                    results.extend(partial_results)
                    all_solutions.update(partial_solutions)
                    log.info("Work unit done: type=%-24s N=%-3d "
                             "(%d results so far).",
                             work_type, N, len(results))
                except Exception as exc:
                    log.error("Work unit failed: type=%s N=%d - %s",
                              work_type, N, exc, exc_info=True)

    # -- Persist everything --------------------------------------------------
    _save_results(results)
    _save_all_solutions(all_solutions)

    elapsed = time.perf_counter() - t_global_start
    _banner(f"Benchmark complete. Total elapsed time: {elapsed:.1f} s")
    log.info("Results written to: %s", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()
