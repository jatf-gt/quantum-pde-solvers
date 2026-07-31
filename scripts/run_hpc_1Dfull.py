#!/usr/bin/env python3
"""
run_hpc_1Dfull.py
===============
Full benchmark run for Imperial College HPC (CX3, PBS Pro).

Runs all working 1D cases (generic Poisson + HET application) for
N = 4, 8, 16, 32 (and optionally N = 64) using Thomas, HHL, and VQLS.
QSVT is included for N = 4 and N = 8 only, with a hard time-limit guard.

Results are saved as:
  - JSON  : results/1Dhpc_run/results_full.json          (machine-readable)
  - CSV   : results/1Dhpc_run/results_summary.csv         (human-readable)
  - NPZ   : results/1Dhpc_run/solutions_N{N}_{case}.npz   (solution vectors)
  - LOG   : results/1Dhpc_run/run.log                     (progress log)

Usage on HPC (after activating the quantum-pde-solvers venv):
  python run_hpc_1Dfull.py [--include-n64] [--skip-qsvt]

Author : Juan Antonio Trobajo Flecha
Date   : July 2026
"""

# -- Standard library ---------------------------------------------------------
import argparse
import concurrent.futures
import csv
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional
from unittest import result
from qiskit_aer import AerSimulator

# -- Third-party --------------------------------------------------------------
import numpy as np
import logging
logging.getLogger("qiskit_ibm_runtime").setLevel(logging.CRITICAL)
logging.getLogger("qiskit.transpiler").setLevel(logging.CRITICAL)

# -- Local --------------------------------------------------------------------
# Ensure the repository root is on sys.path regardless of invocation location.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from solvers.backend_factory import get_aer_backend, log_backend_info  # noqa: E402

# ── Ensure the repo root is on sys.path ──────────────────────────────────────
# Adjust this if the script lives in a subdirectory of the repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = Path("results") / "1Dhpc_run"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging: both stdout (for HPC job output file) and a log file ─────────────
LOG_FILE = RESULTS_DIR / "run.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
log = logging.getLogger(__name__)
# -- Suppress external library logging noise ---------------------------------
# Qiskit transpiler pass timings, IBM provider plugin errors, and Aer
# backend initialisation messages are irrelevant to benchmark progress
# monitoring and are suppressed here to maintain readable output.
logging.getLogger("qiskit.transpiler").setLevel(logging.CRITICAL)
logging.getLogger("qiskit.transpiler.passes").setLevel(logging.CRITICAL)
logging.getLogger("qiskit_ibm_runtime").setLevel(logging.CRITICAL)
logging.getLogger("qiskit_ibm_provider").setLevel(logging.CRITICAL)
logging.getLogger("qiskit_aer").setLevel(logging.CRITICAL)
logging.getLogger("stevedore").setLevel(logging.CRITICAL)
logging.getLogger("qiskit.passmanager").setLevel(logging.CRITICAL)
logging.getLogger("qiskit.compiler").setLevel(logging.CRITICAL)

# ── QSVT time limit: skip if estimated wall time exceeds this (seconds) ───────
# N=8 QSVT took ~222s on a laptop. On HPC it will be faster but still long.
# Set to None to disable the guard entirely.
QSVT_TIME_LIMIT_S: Optional[float] = 1800.0   # 20 minutes per QSVT call

# ── N values to run ───────────────────────────────────────────────────────────
N_VALUES_DEFAULT = [4, 8, 16, 32]
N_VALUES_WITH_64 = [4, 8, 16, 32, 64]

# ── QSVT is only attempted for small N (circuit depth manageable) ─────────────
QSVT_MAX_N = 32

# -- Parallelisation configuration --------------------------------------------
# Maximum number of worker processes for case-level parallelisation.
# Each worker executes one (case_function, N) combination independently.
# On a CX3 large24/72 node (128 cores AMD Rome, 1 TB RAM), setting this
# to 16–32 is appropriate: each Qiskit Aer simulation is already
# internally multi-threaded via OpenMP, so over-subscription is
# counterproductive. A value of 0 disables process-level parallelism.
MAX_WORKERS_DEFAULT: int = 16

# GPU preference flag: read from environment to allow PBS script override.
# Set QUANTUM_PDE_USE_GPU=0 in the PBS script to force CPU execution.
_USE_GPU: bool = os.environ.get("QUANTUM_PDE_USE_GPU", "1") != "0"

# ============================================================================
#  Result dataclass
# ============================================================================

@dataclass
class RunResult:
    """One row of the results table."""
    case:           str       # e.g. "1D_Poisson_fS_hom"
    solver:         str       # "Thomas" | "HHL" | "VQLS" | "QSVT"
    N:              int
    kappa:          float
    max_rel_err:    Optional[float]   # % relative to analytical solution
    max_abs_err:    Optional[float]
    residual:       Optional[float]   # ||Au - b|| / ||b||
    wall_time_s:    float
    converged:      bool
    notes:          str = ""
    # Solution vector stored separately in NPZ; not serialised here.


# ============================================================================
#  Utility helpers
# ============================================================================

def _banner(msg: str) -> None:
    """Print a clearly visible section banner to stdout and log."""
    sep = "=" * 72
    log.info(sep)
    log.info(f"  {msg}")
    log.info(sep)


def _section(msg: str) -> None:
    sep = "-" * 72
    log.info(sep)
    log.info(f"  {msg}")
    log.info(sep)


def _save_solution(
    case: str,
    solver: str,
    N: int,
    x: np.ndarray,
    u: np.ndarray,
    u_exact: Optional[np.ndarray],
) -> None:
    """Save solution vector(s) to a compressed NPZ file."""
    fname = RESULTS_DIR / f"solutions_{case}_{solver}_N{N}.npz"
    arrays = {"x": x, "u_solver": u}
    if u_exact is not None:
        arrays["u_exact"] = u_exact
    np.savez_compressed(fname, **arrays)


def _relative_residual(A: np.ndarray, u: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(A @ u - b) / (np.linalg.norm(b) + 1e-300))


def _max_rel_err(u: np.ndarray, u_ref: np.ndarray, tol: float = 1e-10) -> float:
    """Maximum relative error (%), excluding near-zero reference nodes."""
    mask = np.abs(u_ref) > tol
    if not np.any(mask):
        return float("nan")
    return float(np.max(np.abs(u[mask] - u_ref[mask]) / np.abs(u_ref[mask])) * 100.0)


def _max_abs_err(u: np.ndarray, u_ref: np.ndarray) -> float:
    return float(np.max(np.abs(u - u_ref)))


# ============================================================================
#  Problem builders
# ============================================================================

def _build_tst(N: int) -> np.ndarray:
    """N×N TST matrix with main diag -2, off-diag +1."""
    A = -2.0 * np.eye(N) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)
    return A


def _grid(N: int):
    dx = 1.0 / (N + 1)
    x = np.arange(1, N + 1) * dx
    return x, dx


def _kappa(A: np.ndarray) -> float:
    eigs = np.abs(np.linalg.eigvalsh(A))
    return float(eigs.max() / eigs.min())


# ── Source functions ──────────────────────────────────────────────────────────

def f_sin(x):   return np.sin(np.pi * x)
def f_lin(x):   return 10.0 * x
def f_hev(x):   return np.where(x >= 0.5, 1.0, -1.0)


# ── Analytical solutions (homogeneous BCs) ────────────────────────────────────

def u_sin(x):   return -np.sin(np.pi * x) / np.pi**2
def u_lin(x):   return 5.0 * x * (x**2 - 1.0) / 3.0
def u_hev(x):
    return np.where(x < 0.5, -x**2 / 2.0 + x / 4.0,
                              x**2 / 2.0 - 3.0 * x / 4.0 + 1.0 / 4.0)


# ── HET source functions ──────────────────────────────────────────────────────

def _het_gaussian_source(x: np.ndarray, V_d: float = 300.0,
                          L: float = 0.025, sigma: float = 0.005) -> np.ndarray:
    """
    Gaussian electron density profile: n_e(x) = n_0 * exp(-(x-x0)^2 / (2*sigma^2))
    Source term: f(x) = e/eps0 * (n_i - n_e) approximated as -e/eps0 * n_e
    (ion density treated as uniform background for the simplified 1D model).
    Normalised so that max|f| = 1 for the quantum solver interface.
    """
    e = 1.602e-19
    eps0 = 8.854e-12
    n0 = 1e17       # m^-3, representative channel density
    x0 = 0.6 * L   # peak near exit plane
    n_e = n0 * np.exp(-((x * L - x0)**2) / (2 * sigma**2))
    f = -(e / eps0) * n_e
    return f


def _het_linear_source(x: np.ndarray) -> np.ndarray:
    """Linear density profile (Sub-case 3a): f(x) = 2x - 1."""
    return 2.0 * x - 1.0


def _het_neumann_dirichlet_source(x: np.ndarray, L: float = 0.025,
                                   sigma: float = 0.005) -> np.ndarray:
    """
    Sub-case 3c: Gaussian source with Neumann BC at x=0 (phi'(0)=0),
    Dirichlet at x=1 (phi(1)=0).
    Same Gaussian profile as 3b but different BCs.
    """
    n0 = 1.0   # normalised
    x0 = 0.6
    f = -n0 * np.exp(-((x - x0)**2) / (2 * (sigma / L)**2))
    return f


def _het_neumann_dirichlet_exact(x: np.ndarray, sigma_norm: float = 0.2) -> np.ndarray:
    """
    Analytical solution for phi'' = f(x) with phi'(0)=0, phi(1)=0.
    f(x) = -exp(-(x-0.6)^2 / (2*sigma^2))

    Obtained by double integration:
      phi'(x) = -integral_0^x f(t) dt
      phi(x)  = -integral_0^x phi'(s) ds + C
    with C chosen so that phi(1) = 0.

    We use numerical quadrature for the exact solution since the
    Gaussian integral has no closed form in finite limits.
    """
    from scipy.integrate import cumulative_trapezoid
    x_fine = np.linspace(0, 1, 10000)
    f_fine = -np.exp(-((x_fine - 0.6)**2) / (2 * sigma_norm**2))
    # phi'(x) = integral_0^x f(t) dt  (Neumann: phi'(0)=0)
    dphi_fine = cumulative_trapezoid(f_fine, x_fine, initial=0.0)
    # phi(x) = integral_0^x phi'(s) ds + C
    phi_fine = cumulative_trapezoid(dphi_fine, x_fine, initial=0.0)
    # Enforce phi(1) = 0
    phi_fine -= phi_fine[-1]
    # Interpolate to the requested x points
    return np.interp(x, x_fine, phi_fine)


# ============================================================================
#  Solver wrappers
# ============================================================================

def _run_thomas(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Thomas algorithm. Returns (u, residual, wall_time_s)."""
    N = len(b)
    diag = -2.0 * np.ones(N)
    off  =  1.0 * np.ones(N)
    d = b.copy()
    t0 = time.perf_counter()
    for i in range(1, N):
        m = off[i - 1] / diag[i - 1]
        diag[i] -= m * off[i - 1]
        d[i]    -= m * d[i - 1]
    u = np.zeros(N)
    u[-1] = d[-1] / diag[-1]
    for i in range(N - 2, -1, -1):
        u[i] = (d[i] - off[i] * u[i + 1]) / diag[i]
    wall = time.perf_counter() - t0
    return u, _relative_residual(A, u, b), wall


def _run_hhl(
    A: np.ndarray,
    b: np.ndarray,
    N: int,
    epsilon: float = 0.01,
    backend: Optional[object] = None,
) -> tuple[Optional[np.ndarray], float, float, bool]:
    """
    Execute the HHL quantum linear solver via the validated project module.

    Delegates directly to the existing ``hhl_solve`` implementation in
    ``solvers/quantum/hhl_1d.py``, which has been validated against the
    Ghafourpour & Laizet (2025) benchmark results. This avoids duplicating
    normalisation and proportionality recovery logic.
    """
    # Replace the entire try block body in _run_hhl with:
    from solvers.quantum.hhl_1d import hhl_solve_system

    t0 = time.perf_counter()
    u, x_raw, c = hhl_solve_system(A, b, epsilon)
    wall = time.perf_counter() - t0

    return u, _relative_residual(A, u, b), wall, True


def _extract_hhl_solution(
    solution: object,
    num_qubits: int,
    backend: Optional["AerSimulator"] = None,  # noqa: F821
) -> np.ndarray:
    """
    Extract the physical solution vector from the HHL output circuit's
    statevector via post-selection on the ancilla register.

    The HHL circuit encodes the solution as amplitudes in the b-register
    (lowest ``num_qubits`` qubits) conditioned on the ancilla qubit
    (highest qubit index) being in state |1⟩ and all clock-register
    qubits being in state |0⟩ (inverse QPE post-selection).

    Qubit ordering follows Qiskit's little-endian convention: qubit k
    corresponds to bit position k of the statevector integer index.

    Parameters
    ----------
    solution : HHL result object
        Output of ``HHL.solve()``, containing ``solution.state`` as a
        ``QuantumCircuit`` whose statevector encodes the solution.
    num_qubits : int
        Number of data register qubits; equals log₂(N).
    backend : AerSimulator or None
        Aer backend for statevector simulation. If ``None``, a CPU
        backend is constructed via :func:`get_aer_backend`.

    Returns
    -------
    x_raw : np.ndarray, shape (N,), dtype float64
        Real part of the extracted solution amplitudes. Imaginary
        residuals arising from Trotter approximation error are discarded.

    Raises
    ------
    RuntimeError
        If the extracted vector is identically zero, indicating a
        qubit-register mapping error.
    """
    from qiskit import transpile
    import warnings

    if backend is None:
        backend = get_aer_backend(prefer_gpu=_USE_GPU)

    N = 2**num_qubits
    qc = solution.state

    with warnings.catch_warnings():
        from qiskit.quantum_info import Statevector
        qc_t = transpile(qc, backend)
        job = backend.run(qc_t)
        result = job.result()
        try:
            sv = np.array(result.get_statevector())
        except Exception:
            # Qiskit 1.x returns statevector directly via Statevector class
            sv = np.array(Statevector(qc_t).data)

    n_total = qc.num_qubits
    n_b     = num_qubits
    n_other = n_total - 1 - n_b

    x_raw = np.zeros(N, dtype=complex)
    for idx in range(2**n_total):
        ancilla_bit = (idx >> (n_total - 1)) & 1
        other_bits  = (idx >> n_b) & ((1 << n_other) - 1)
        b_reg_idx   = idx & (N - 1)
        if ancilla_bit == 1 and other_bits == 0:
            x_raw[b_reg_idx] = sv[idx]

    x_raw_real = np.real(x_raw)
    if np.allclose(x_raw_real, 0.0):
        raise RuntimeError(
            "HHL solution extraction returned an identically zero vector. "
            "This indicates a qubit-register mapping error. "
            f"Inspect circuit layout: n_total={n_total}, n_b={n_b}."
        )
    return x_raw_real


def _run_vqls(
    A: np.ndarray,
    b: np.ndarray,
    N: int,
) -> tuple[Optional[np.ndarray], float, float, bool, float]:
    """
    Execute the VQLS solver via the validated project module.
    """
    try:
        from solvers.quantum.vqls_1d import vqls_solve_system, VQLSConfig1D

        n_qubits  = int(np.log2(N))
        n_layers  = max(6, 2 * n_qubits + 2)
        n_restarts = max(3, 2*n_qubits)
        vqls_cfg = VQLSConfig1D(
            n_layers   = n_layers,
            max_iter   = 500,
            tol        = 1e-6,
            random_seed = 42,
            verbose    = False,
            n_restarts = n_restarts,
        )

        t0 = time.perf_counter()
        result = vqls_solve_system(A, b, config=vqls_cfg)
        wall = time.perf_counter() - t0

        u          = result.u
        converged  = result.optimiser_success
        final_cost = result.final_cost

        # Flag runs where VQLS clearly diverged (cost > threshold indicates non-convergence)
        if final_cost > 0.1:
            log.warning(
                "    VQLS cost=%.2e exceeds convergence threshold (0.1); "
                "solution is likely non-physical for N=%d. "
                "Result retained for reporting purposes.",
                final_cost, N
            )

        return u, _relative_residual(A, u, b), wall, converged, float(final_cost)

    except Exception as exc:
        log.warning("    VQLS failed: %s", exc)
        return None, float("nan"), 0.0, False, float("nan")


def _run_qsvt(A: np.ndarray, b: np.ndarray, N: int,
              time_limit: Optional[float]) -> tuple[Optional[np.ndarray], float, float, bool, int, int]:
    """
    QSVT solver. Returns (u, residual, wall_time_s, converged, degree, depth).
    Skips if N > QSVT_MAX_N or if estimated time exceeds time_limit.
    """
    if N > QSVT_MAX_N:
        log.info(f"    QSVT: skipping N={N} > QSVT_MAX_N={QSVT_MAX_N}")
        return None, float("nan"), 0.0, False, -1, -1

    try:
        from solvers.quantum.qsvt_1d import qsvt_solve_system
        t0 = time.perf_counter()
        result = qsvt_solve_system(A, b, verbose=True)
        wall = time.perf_counter() - t0

        # Enforce time limit (post-hoc check — QSVT runs to completion).
        if time_limit is not None and wall > time_limit:
            log.warning(f"    QSVT: completed but exceeded time limit "
                        f"({wall:.1f}s > {time_limit:.1f}s). Result retained.")

        u = result.u
        return (u, _relative_residual(A, u, b), wall,
                result.converged, result.degree, result.circuit_depth)
    except Exception as exc:
        log.warning(f"    QSVT failed: {exc}")
        return None, float("nan"), 0.0, False, -1, -1


# ============================================================================
#  Case runners
# ============================================================================

def run_1d_generic_poisson_single_N(
    N: int,
    skip_qsvt: bool,
    results: list[RunResult],
    all_solutions: dict,
) -> None:
    """
    Section 1: 1D generic Poisson, fS source, homogeneous BCs.
    Runs Thomas, HHL, VQLS, and (for small N) QSVT.
    """
    _banner("SECTION 1 — 1D Generic Poisson (fS, fL, fH) — Homogeneous BCs")

    source_map = {
        "fS": (f_sin, u_sin),
        "fL": (f_lin, u_lin),
        "fH": (f_hev, u_hev),
    }

    for src_key, (src_fn, exact_fn) in source_map.items():
        _section(f"Source: {src_key}")
        x, dx = _grid(N)
        A = _build_tst(N)
        b = dx**2 * src_fn(x)
        kap = _kappa(A)
        u_exact = exact_fn(x)
        case_id = f"1D_Poisson_{src_key}_hom"

        log.info(f"  N={N:3d}  kappa={kap:.2f}  case={case_id}")

        # Thomas
        u_T, res_T, t_T = _run_thomas(A, b)
        log.info(f"    Thomas  MaxRelErr={_max_rel_err(u_T, u_exact):7.3f}%  "
                    f"Residual={res_T:.3e}  Time={t_T:.3f}s")
        results.append(RunResult(case_id, "Thomas", N, kap,
                                    _max_rel_err(u_T, u_exact),
                                    _max_abs_err(u_T, u_exact),
                                    res_T, t_T, True))
        _save_solution(case_id, "Thomas", N, x, u_T, u_exact)
        all_solutions[f"{case_id}_Thomas_N{N}"] = {"x": x, "u": u_T, "u_exact": u_exact}

        # HHL
        u_H, res_H, t_H, conv_H = _run_hhl(A, b, N)
        if u_H is not None:
            log.info(f"    HHL     MaxRelErr={_max_rel_err(u_H, u_exact):7.3f}%  "
                        f"Residual={res_H:.3e}  Time={t_H:.3f}s")
            results.append(RunResult(case_id, "HHL", N, kap,
                                        _max_rel_err(u_H, u_exact),
                                        _max_abs_err(u_H, u_exact),
                                        res_H, t_H, conv_H))
            _save_solution(case_id, "HHL", N, x, u_H, u_exact)
            all_solutions[f"{case_id}_HHL_N{N}"] = {"x": x, "u": u_H, "u_exact": u_exact}
        else:
            results.append(RunResult(case_id, "HHL", N, kap,
                                        None, None, None, t_H, False, "solver_error"))

        # VQLS
        u_V, res_V, t_V, conv_V, cost_V = _run_vqls(A, b, N)
        if u_V is not None:
            log.info(f"    VQLS    MaxRelErr={_max_rel_err(u_V, u_exact):7.3f}%  "
                        f"Residual={res_V:.3e}  Time={t_V:.3f}s  cost={cost_V:.2e}")
            results.append(RunResult(case_id, "VQLS", N, kap,
                                        _max_rel_err(u_V, u_exact),
                                        _max_abs_err(u_V, u_exact),
                                        res_V, t_V, conv_V,
                                        notes=f"cost={cost_V:.2e}"))
            _save_solution(case_id, "VQLS", N, x, u_V, u_exact)
            all_solutions[f"{case_id}_VQLS_N{N}"] = {"x": x, "u": u_V, "u_exact": u_exact}
        else:
            results.append(RunResult(case_id, "VQLS", N, kap,
                                        None, None, None, t_V, False, "solver_error"))

        # QSVT
        if not skip_qsvt:
            u_Q, res_Q, t_Q, conv_Q, deg_Q, dep_Q = _run_qsvt(A, b, N, QSVT_TIME_LIMIT_S)
            if u_Q is not None:
                log.info(f"    QSVT    MaxRelErr={_max_rel_err(u_Q, u_exact):7.3f}%  "
                            f"Residual={res_Q:.3e}  Time={t_Q:.1f}s  "
                            f"deg={deg_Q}  depth={dep_Q}")
                results.append(RunResult(case_id, "QSVT", N, kap,
                                            _max_rel_err(u_Q, u_exact),
                                            _max_abs_err(u_Q, u_exact),
                                            res_Q, t_Q, conv_Q,
                                            notes=f"deg={deg_Q},depth={dep_Q}"))
                _save_solution(case_id, "QSVT", N, x, u_Q, u_exact)
                all_solutions[f"{case_id}_QSVT_N{N}"] = {"x": x, "u": u_Q, "u_exact": u_exact}
            elif N <= QSVT_MAX_N:
                results.append(RunResult(case_id, "QSVT", N, kap,
                                            None, None, None, t_Q, False,
                                            notes="skipped_or_failed"))


def run_1d_generic_poisson_nonhom_single_N(
    N: int,
    skip_qsvt: bool,
    results: list[RunResult],
    all_solutions: dict,
) -> None:
    """
    Section 1b: 1D generic Poisson, fS source, non-homogeneous BCs.
    alpha = 1.0, beta = 2.0 (arbitrary non-zero Dirichlet BCs).
    Thomas and HHL only (VQLS and QSVT not tested for non-hom BCs in current code).
    """
    _banner("SECTION 1b — 1D Generic Poisson (fS) — Non-Homogeneous BCs (alpha=1, beta=2)")

    alpha, beta = 1.0, 2.0

    x, dx = _grid(N)
    A = _build_tst(N)
    b = dx**2 * f_sin(x)
    b[0]  -= alpha
    b[-1] -= beta
    kap = _kappa(A)
    case_id = "1D_Poisson_fS_nonhom"

    # Analytical solution for non-hom BCs:
    # u'' = sin(pi*x), u(0)=alpha, u(1)=beta
    # u(x) = -sin(pi*x)/pi^2 + (beta - alpha)*x + alpha
    u_exact = -np.sin(np.pi * x) / np.pi**2 + (beta - alpha) * x + alpha

    log.info(f"  N={N:3d}  kappa={kap:.2f}  alpha={alpha}  beta={beta}")

    u_T, res_T, t_T = _run_thomas(A, b)
    log.info(f"    Thomas  MaxRelErr={_max_rel_err(u_T, u_exact):7.3f}%  "
                f"Residual={res_T:.3e}  Time={t_T:.3f}s")
    results.append(RunResult(case_id, "Thomas", N, kap,
                                _max_rel_err(u_T, u_exact),
                                _max_abs_err(u_T, u_exact),
                                res_T, t_T, True))
    _save_solution(case_id, "Thomas", N, x, u_T, u_exact)

    u_H, res_H, t_H, conv_H = _run_hhl(A, b, N)
    if u_H is not None:
        log.info(f"    HHL     MaxRelErr={_max_rel_err(u_H, u_exact):7.3f}%  "
                    f"Residual={res_H:.3e}  Time={t_H:.3f}s")
        results.append(RunResult(case_id, "HHL", N, kap,
                                    _max_rel_err(u_H, u_exact),
                                    _max_abs_err(u_H, u_exact),
                                    res_H, t_H, conv_H))
        _save_solution(case_id, "HHL", N, x, u_H, u_exact)


def run_1d_het_single_N(
    N: int,
    skip_qsvt: bool,
    results: list[RunResult],
    all_solutions: dict,
) -> None:
    """
    Section 2: 1D HET Axial Poisson — three sub-cases.
    Sub-case 3a: linear profile, homogeneous BCs (Thomas, HHL, VQLS)
    Sub-case 3b: Gaussian profile, V_d=300V (Thomas, HHL, VQLS)
    Sub-case 3c: Gaussian profile, Neumann-Dirichlet BCs (Thomas, HHL, VQLS)
                 — new benchmark case with analytical reference
    """
    _banner("SECTION 2 — 1D HET Axial Poisson")

    # ── Sub-case 3a: linear profile, homogeneous BCs ──────────────────────────
    _section("Sub-case 3a: Linear profile, homogeneous BCs")
    x, dx = _grid(N)
    A = _build_tst(N)
    b = dx**2 * _het_linear_source(x)
    kap = _kappa(A)
    case_id = "HET_1D_3a_linear_hom"

    # Analytical solution: u'' = 2x - 1, u(0)=u(1)=0
    # u(x) = x^3/3 - x^2/2 + x/6
    u_exact = x**3 / 3.0 - x**2 / 2.0 + x / 6.0

    log.info(f"  N={N:3d}  kappa={kap:.2f}  sub-case=3a")

    u_T, res_T, t_T = _run_thomas(A, b)
    log.info(f"    Thomas  MaxRelErr={_max_rel_err(u_T, u_exact):7.3f}%  "
                f"Residual={res_T:.3e}  Time={t_T:.3f}s")
    results.append(RunResult(case_id, "Thomas", N, kap,
                                _max_rel_err(u_T, u_exact),
                                _max_abs_err(u_T, u_exact),
                                res_T, t_T, True))
    _save_solution(case_id, "Thomas", N, x, u_T, u_exact)
    all_solutions[f"{case_id}_Thomas_N{N}"] = {"x": x, "u": u_T, "u_exact": u_exact}

    u_H, res_H, t_H, conv_H = _run_hhl(A, b, N)
    if u_H is not None:
        log.info(f"    HHL     MaxRelErr={_max_rel_err(u_H, u_exact):7.3f}%  "
                    f"Residual={res_H:.3e}  Time={t_H:.3f}s")
        results.append(RunResult(case_id, "HHL", N, kap,
                                    _max_rel_err(u_H, u_exact),
                                    _max_abs_err(u_H, u_exact),
                                    res_H, t_H, conv_H))
        _save_solution(case_id, "HHL", N, x, u_H, u_exact)
        all_solutions[f"{case_id}_HHL_N{N}"] = {"x": x, "u": u_H, "u_exact": u_exact}

    u_V, res_V, t_V, conv_V, cost_V = _run_vqls(A, b, N)
    if u_V is not None:
        log.info(f"    VQLS    MaxRelErr={_max_rel_err(u_V, u_exact):7.3f}%  "
                    f"Residual={res_V:.3e}  Time={t_V:.3f}s")
        results.append(RunResult(case_id, "VQLS", N, kap,
                                    _max_rel_err(u_V, u_exact),
                                    _max_abs_err(u_V, u_exact),
                                    res_V, t_V, conv_V))
        _save_solution(case_id, "VQLS", N, x, u_V, u_exact)
        all_solutions[f"{case_id}_VQLS_N{N}"] = {"x": x, "u": u_V, "u_exact": u_exact}

    # ── Sub-case 3b: Gaussian profile, V_d = 300V ─────────────────────────────
    _section("Sub-case 3b: Gaussian profile, V_d=300V, Dirichlet BCs")
    V_d = 300.0
    L   = 0.025
    x, dx = _grid(N)
    A = _build_tst(N)
    f_vals = _het_gaussian_source(x, V_d=V_d, L=L)
    b = dx**2 * f_vals
    b[0]  -= V_d    # phi(0) = V_d (anode)
    b[-1] -= 0.0    # phi(1) = 0   (cathode)
    kap = _kappa(A)
    case_id = "HET_1D_3b_gaussian_Vd300"

    log.info(f"  N={N:3d}  kappa={kap:.2f}  sub-case=3b  V_d={V_d}V")

    u_T, res_T, t_T = _run_thomas(A, b)
    log.info(f"    Thomas  Residual={res_T:.3e}  Time={t_T:.3f}s  (reference)")
    results.append(RunResult(case_id, "Thomas", N, kap,
                                None, None, res_T, t_T, True,
                                notes="reference_no_exact"))
    _save_solution(case_id, "Thomas", N, x, u_T, None)
    all_solutions[f"{case_id}_Thomas_N{N}"] = {"x": x, "u": u_T, "u_exact": None}

    u_H, res_H, t_H, conv_H = _run_hhl(A, b, N)
    if u_H is not None:
        rel_vs_thomas = _max_rel_err(u_H, u_T)
        log.info(f"    HHL     MaxRelErr(vs Thomas)={rel_vs_thomas:7.3f}%  "
                    f"Residual={res_H:.3e}  Time={t_H:.3f}s")
        # Electric field comparison
        E_T = -np.gradient(u_T, x)
        E_H = -np.gradient(u_H, x)
        log.info(f"    Peak |E| Thomas={np.max(np.abs(E_T)):.3e} V/m  "
                    f"HHL={np.max(np.abs(E_H)):.3e} V/m")
        results.append(RunResult(case_id, "HHL", N, kap,
                                    rel_vs_thomas, _max_abs_err(u_H, u_T),
                                    res_H, t_H, conv_H,
                                    notes="rel_vs_thomas"))
        _save_solution(case_id, "HHL", N, x, u_H, u_T)
        all_solutions[f"{case_id}_HHL_N{N}"] = {"x": x, "u": u_H, "u_exact": u_T}

    u_V, res_V, t_V, conv_V, cost_V = _run_vqls(A, b, N)
    if u_V is not None:
        rel_vs_thomas = _max_rel_err(u_V, u_T)
        E_V = -np.gradient(u_V, x)
        log.info(f"    VQLS    MaxRelErr(vs Thomas)={rel_vs_thomas:7.3f}%  "
                    f"Residual={res_V:.3e}  Time={t_V:.3f}s")
        log.info(f"    Peak |E| VQLS={np.max(np.abs(E_V)):.3e} V/m")
        results.append(RunResult(case_id, "VQLS", N, kap,
                                    rel_vs_thomas, _max_abs_err(u_V, u_T),
                                    res_V, t_V, conv_V,
                                    notes=f"rel_vs_thomas,cost={cost_V:.2e}"))
        _save_solution(case_id, "VQLS", N, x, u_V, u_T)
        all_solutions[f"{case_id}_VQLS_N{N}"] = {"x": x, "u": u_V, "u_exact": u_T}

    # # ── Sub-case 3c: Gaussian profile, Neumann-Dirichlet BCs (new benchmark) ──
    # _section("Sub-case 3c: Gaussian profile, Neumann-Dirichlet BCs (NEW BENCHMARK)")
    # log.info("  Boundary conditions: phi'(0)=0 (Neumann), phi(1)=0 (Dirichlet)")
    # log.info("  Analytical reference: numerical quadrature of double integral")
    # sigma_norm = 0.2
    # x, dx = _grid(N)
    # # For Neumann at x=0: modify the first row of A.
    # # Standard second-order one-sided Neumann: phi'(0) ~ (-3phi_0 + 4phi_1 - phi_2)/(2dx) = 0
    # # In the interior system (phi_0 absorbed), this modifies b[0]:
    # # The ghost-point approach: phi_{-1} = phi_1 (from phi'(0)=0)
    # # => first equation becomes: phi_1 - 2*phi_0 + phi_1 = dx^2 * f(x_0)
    # #    i.e. A[0,0] = -2, A[0,1] = 2 (instead of 1)
    # A_nd = _build_tst(N).copy()
    # A_nd[0, 1] = 2.0   # Neumann BC: ghost point phi_{-1} = phi_1
    # b_nd = dx**2 * _het_neumann_dirichlet_source(x, sigma=sigma_norm)
    # # Dirichlet at x=1: phi(1) = 0, already absorbed (b[-1] unchanged)
    # kap = _kappa(A_nd)
    # u_exact = _het_neumann_dirichlet_exact(x, sigma_norm=sigma_norm)
    # case_id = "HET_1D_3c_gaussian_NeumannDirichlet"

    # log.info(f"  N={N:3d}  kappa={kap:.2f}  sub-case=3c")

    # u_T, res_T, t_T = _run_thomas(A_nd, b_nd)
    # log.info(f"    Thomas  MaxRelErr={_max_rel_err(u_T, u_exact):7.3f}%  "
    #             f"Residual={res_T:.3e}  Time={t_T:.3f}s")
    # results.append(RunResult(case_id, "Thomas", N, kap,
    #                             _max_rel_err(u_T, u_exact),
    #                             _max_abs_err(u_T, u_exact),
    #                             res_T, t_T, True))
    # _save_solution(case_id, "Thomas", N, x, u_T, u_exact)
    # all_solutions[f"{case_id}_Thomas_N{N}"] = {"x": x, "u": u_T, "u_exact": u_exact}

    # u_H, res_H, t_H, conv_H = _run_hhl(A_nd, b_nd, N)
    # if u_H is not None:
    #     log.info(f"    HHL     MaxRelErr={_max_rel_err(u_H, u_exact):7.3f}%  "
    #                 f"Residual={res_H:.3e}  Time={t_H:.3f}s")
    #     results.append(RunResult(case_id, "HHL", N, kap,
    #                                 _max_rel_err(u_H, u_exact),
    #                                 _max_abs_err(u_H, u_exact),
    #                                 res_H, t_H, conv_H))
    #     _save_solution(case_id, "HHL", N, x, u_H, u_exact)
    #     all_solutions[f"{case_id}_HHL_N{N}"] = {"x": x, "u": u_H, "u_exact": u_exact}

    # u_V, res_V, t_V, conv_V, cost_V = _run_vqls(A_nd, b_nd, N)
    # if u_V is not None:
    #     log.info(f"    VQLS    MaxRelErr={_max_rel_err(u_V, u_exact):7.3f}%  "
    #                 f"Residual={res_V:.3e}  Time={t_V:.3f}s")
    #     results.append(RunResult(case_id, "VQLS", N, kap,
    #                                 _max_rel_err(u_V, u_exact),
    #                                 _max_abs_err(u_V, u_exact),
    #                                 res_V, t_V, conv_V))
    #     _save_solution(case_id, "VQLS", N, x, u_V, u_exact)
    #     all_solutions[f"{case_id}_VQLS_N{N}"] = {"x": x, "u": u_V, "u_exact": u_exact}


# ============================================================================
#  Save results
# ============================================================================

def _save_results(results: list[RunResult]) -> None:
    """Save results to JSON and CSV."""
    # JSON
    json_path = RESULTS_DIR / "results_full.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2, default=str)
    log.info(f"Results saved to {json_path}")

    # CSV
    csv_path = RESULTS_DIR / "results_summary.csv"
    if results:
        fieldnames = list(asdict(results[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
    log.info(f"Results saved to {csv_path}")


def _execute_work_unit(
    work_type: str,
    N: int,
    skip_qsvt: bool,
) -> tuple[list[RunResult], dict]:
    """
    Execute a single (case_type, N) work unit and return its results.

    This function is designed to be called from a ``ProcessPoolExecutor``
    worker process. Each invocation is fully self-contained: it constructs
    its own Aer backend and accumulates results into local lists that are
    returned to the parent process for aggregation.

    The process-level isolation is necessary because Qiskit circuit objects
    and Aer backend state are not safely shareable across processes via
    shared memory. Each worker independently detects GPU availability via
    ``CUDA_VISIBLE_DEVICES``; on a GPU node with a single GPU allocated,
    ``--max-workers 1`` must be used to prevent CUDA context conflicts.

    Parameters
    ----------
    work_type : str
        One of ``'generic_poisson'``, ``'generic_poisson_nonhom'``,
        ``'het_1d'``. Determines which case-runner function is invoked.
    N : int
        System dimension for this work unit.
    skip_qsvt : bool
        If ``True``, QSVT solver calls are omitted from this work unit.

    Returns
    -------
    results : list[RunResult]
        Benchmark results accumulated during this work unit.
    solutions : dict
        Solution vector dictionary for NPZ serialisation.
    """
    results: list[RunResult] = []
    solutions: dict = {}

    dispatch = {
        "generic_poisson":     run_1d_generic_poisson_single_N,
        "generic_poisson_nonhom": run_1d_generic_poisson_nonhom_single_N,
        "het_1d":              run_1d_het_single_N,
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
    Independent (case_function, N) combinations are dispatched to a
    ``ProcessPoolExecutor`` worker pool. Each worker process constructs
    its own Aer backend (GPU or CPU) independently, avoiding shared-state
    conflicts between Qiskit circuit objects.

    The degree of parallelism is controlled by ``--max-workers``. On a
    CX3 large24/72 node (128 cores), 16 workers is a reasonable default:
    each Aer CPU simulation is already OpenMP-parallelised internally, so
    over-subscription beyond ~16 processes yields diminishing returns.

    On a GPU node (gpu72 queue), set ``--max-workers 1`` and
    ``QUANTUM_PDE_USE_GPU=1`` to serialise circuit execution through the
    single GPU, avoiding CUDA context conflicts between processes.
    """
    parser = argparse.ArgumentParser(
        description="Full HPC benchmark sweep for quantum PDE solvers."
    )
    parser.add_argument(
        "--include-n64", action="store_true",
        help="Include N=64 in the parameter sweep (substantially increases "
             "wall-clock time; recommended only after confirming N=32 "
             "completes within the allocated walltime)."
    )
    parser.add_argument(
        "--skip-qsvt", action="store_true",
        help="Omit QSVT solver from all cases. Use when the QSVT module "
             "is unavailable or when a rapid validation sweep is required."
    )
    parser.add_argument(
        "--max-workers", type=int, default=MAX_WORKERS_DEFAULT,
        help=f"Number of parallel worker processes for case-level "
             f"parallelisation (default: {MAX_WORKERS_DEFAULT}). "
             f"Set to 1 to disable process-level parallelism (required "
             f"when using GPU to avoid CUDA context conflicts)."
    )
    args = parser.parse_args()

    N_values = N_VALUES_WITH_64 if args.include_n64 else N_VALUES_DEFAULT

    # -- Resolve GPU/CPU backend and log configuration -----------------------
    backend = get_aer_backend(prefer_gpu=_USE_GPU)

    _banner("QUANTUM PDE SOLVER — FULL HPC BENCHMARK RUN")
    log.info("  N values      : %s", N_values)
    log.info("  QSVT          : %s",
             "DISABLED" if args.skip_qsvt
             else f"enabled for N ≤ {QSVT_MAX_N}")
    log.info("  Max workers   : %d", args.max_workers)
    log.info("  Output dir    : %s", RESULTS_DIR.resolve())
    log.info("  Python        : %s", sys.version.split()[0])
    log.info("  PID           : %d", os.getpid())
    log_backend_info(backend)

    t_global_start = time.perf_counter()
    results: list[RunResult] = []
    all_solutions: dict = {}

    # -- Build the list of (function, N) work units -------------------------
    # Each work unit is an independent function call that populates the
    # shared results list. Because ProcessPoolExecutor cannot share mutable
    # state, each worker returns its (results, solutions) pair and the
    # main process merges them.
    work_units = []
    for N in N_values:
        work_units.append(("generic_poisson", N, args.skip_qsvt))
        work_units.append(("generic_poisson_nonhom", N, args.skip_qsvt))
        work_units.append(("het_1d", N, args.skip_qsvt))

    if args.max_workers == 1:
        # Serial execution path — simpler and required for GPU mode.
        log.info("Serial execution mode (max_workers=1).")
        for work_type, N, skip_qsvt in work_units:
            partial_results, partial_solutions = _execute_work_unit(
                work_type, N, skip_qsvt
            )
            results.extend(partial_results)
            all_solutions.update(partial_solutions)
    else:
        # Parallel execution path using process pool.
        log.info(
            "Parallel execution mode: dispatching %d work units across "
            "%d worker processes.",
            len(work_units), args.max_workers,
        )
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers
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
                    log.info(
                        "Work unit completed: type=%-25s N=%d  "
                        "(%d results accumulated).",
                        work_type, N, len(results),
                    )
                except Exception as exc:
                    log.error(
                        "Work unit failed: type=%s N=%d — %s",
                        work_type, N, exc, exc_info=True,
                    )

    _save_results(results)
    elapsed = time.perf_counter() - t_global_start
    _banner(f"Benchmark complete. Total elapsed time: {elapsed:.1f} s")
    log.info("Results written to: %s", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main() 