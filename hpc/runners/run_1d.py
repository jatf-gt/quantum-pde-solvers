#!/usr/bin/env python3
"""
run_1d.py
=========
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
  python hpc/runners/run_1d.py                    # full sweep, N = 4..64
  python hpc/runners/run_1d.py --max-n 16         # quick validation pass
  python hpc/runners/run_1d.py --skip-qsvt        # omit QSVT entirely
  python hpc/runners/run_1d.py --max-workers 1    # serial (required on GPU)

Note on imports
---------------
The quantum solver imports in this module are deferred into the functions that
use them, contrary to the usual PEP 8 placement, for two reasons that both
apply here. The solver workers execute in spawned subprocesses, which re-import
the module and must not pay the Qiskit and PennyLane import cost until the
work is dispatched; and the classical and cost-estimate paths (`--skip-qsvt`,
Thomas-only runs, the wall-time projection) must remain runnable on a node with
no quantum backend installed. Case definitions - including sub-case 3c's SciPy
quadrature reference - now live in `core/cases.py`, which applies the same
lazy-import discipline itself.

Date   : August 2026
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import concurrent.futures
import csv
import dataclasses
import fnmatch
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

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import multiprocessing as mp

# ── Local ─────────────────────────────────────────────────────────────────────
# Ensure the repository root is on sys.path regardless of invocation location.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import cases  # noqa: E402

from solvers.backend_factory import get_aer_backend, log_backend_info  # noqa: E402


# ── Output directory and logging ──────────────────────────────────────────────

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
        # Append, never truncate. A single PBS job invokes this runner several
        # times in sequence (one step per resolution band or solver group), and
        # under mode="w" each invocation destroyed its predecessor's log: a job
        # that ran five steps ended holding only the fifth. The history is the
        # sole record of which rows a walltime-killed step had reached, so it is
        # exactly what must not be discarded. Sessions are delimited by the
        # banner written by `_log_session_header` instead.
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger(__name__)


def _log_session_header(phase_tag: Optional[str] = None) -> None:
    """
    Delimit one invocation within an appended log.

    Called once from the parent process after the output directory and any phase
    tag are resolved. Workers do not call it: `_IS_MAIN_PROCESS` is false in a
    spawned child, and a banner per work unit would defeat the purpose.

    Parameters
    ----------
    phase_tag : str, optional
        Step label supplied by the submission script (e.g. ``wave1_het_small``),
        recorded so a row can be attributed to the step that produced it.
    """
    if not _IS_MAIN_PROCESS:
        return
    log.info("=" * 78)
    log.info("SESSION START  %s  pid=%d  phase=%s  argv=%s",
             time.strftime("%Y-%m-%d %H:%M:%S"), os.getpid(),
             phase_tag or "(untagged)", " ".join(sys.argv[1:]) or "(no arguments)")
    log.info("=" * 78)


def _redirect_log_file(path: Path) -> None:
    """
    Point the root logger's file handler at `path`, preserving its formatter.

    Required because the output directory depends on `--order`, which is known
    only after argument parsing, whereas the handler is installed at import time.

    The replacement inherits the original formatter rather than accepting
    logging's bare ``"%(message)s"`` default, which stripped the timestamp and PID
    from every 4th-order log line and left the interleaved output of parallel work
    units unattributable. The new handler appends, for the reason given where the
    first one is installed.

    Parameters
    ----------
    path : Path
        Destination log file. Its parent directory must already exist.
    """
    logger = logging.getLogger()
    fmt = None
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            fmt = fmt or handler.formatter
            handler.close()
            logger.removeHandler(handler)
    new_handler = logging.FileHandler(path, mode="a")
    if fmt is not None:
        new_handler.setFormatter(fmt)
    logger.addHandler(new_handler)

# ── Suppress external library logging noise ───────────────────────────────────
# Qiskit transpiler pass timings, IBM provider plugin errors, and Aer backend
# initialisation messages are irrelevant to benchmark progress monitoring.
for _noisy in (
    "qiskit.transpiler", "qiskit.transpiler.passes", "qiskit_ibm_runtime",
    "qiskit_ibm_provider", "qiskit_aer", "stevedore", "qiskit.passmanager",
    "qiskit.compiler",
):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)


# ── Sweep configuration ───────────────────────────────────────────────────────

# ── N values ──────────────────────────────────────────────────────────────────
# The full sweep. --max-n truncates this list; there is no separate "include
# N=64" flag, because N=64 is now part of the default sweep.
N_VALUES_ALL: list[int] = [4, 8, 16, 32, 64]

# ── QSVT: which N are attempted at all ────────────────────────────────────────
QSVT_MAX_N: int = 64

# ── QSVT: post-hoc wall-time warning threshold (seconds) ──────────────────────
# NOTE: this is a WARNING only. QSVT is not interruptible mid-solve, so this
# cannot abort a long run -- it only flags one in the log after the fact.
# Set to None to silence the warning entirely.
QSVT_TIME_LIMIT_S: Optional[float] = 1800.0   # 30 minutes per QSVT call

# ── QSVT: polynomial degree cap, per N ────────────────────────────────────────
# The QSVT phase cache is keyed on (kappa, epsilon, method, max_degree), so
# these values MUST match what hpc/runners/precompute_phases.py was run with,
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
QSVT_UNCACHED_FALLBACK_DEGREE: int = 5000
QSVT_MAX_DEGREE_FALLBACK: int = 5000

# ── HHL / VQLS configuration ──────────────────────────────────────────────────
HHL_EPSILON: float = 0.01
VQLS_SEED: int = 42

# Hard per-solve wall-clock budget for HHL, in seconds. Named rather than left as
# a default argument because the gap analysis needs the same number to recognise a
# timed-out row as a terminal measurement rather than a fault: see `_run_hhl`.
HHL_TIMEOUT_S: float = 3600.0

# ── Solver and case families ──────────────────────────────────────────────────
# The quantum solvers that `--solvers` selects among. Thomas is deliberately
# absent: it is always executed, both because it costs microseconds and because
# it is the reference solution for sub-case 3b, which has no closed form.
QUANTUM_SOLVERS_1D: tuple[str, ...] = ("hhl", "vqls", "qsvt")

# Section label -> work-unit family, as dispatched in `_execute_work_unit`. The
# labels match the section headings in the log and in README section 5, so a
# gap-analysis row can be traced to a `--sections` argument without a lookup.
SECTION_FAMILIES: dict[str, str] = {
    "1":  "generic_poisson",
    "1b": "generic_poisson_nonhom",
    "2":  "het_1d",
}

# ── Timing repeats ────────────────────────────────────────────────────────────
# Repeats give a mean/std rather than a single sample, which is what makes a
# timing number defensible on a shared node. Only the classical solver is
# repeated by default: repeating the quantum solvers would multiply the total
# sweep wall time by the same factor for little statistical gain.
THOMAS_TIMING_REPEATS: int = 10

# ── Parallelisation ───────────────────────────────────────────────────────────
# Each worker process executes one (case_family, N) work unit. Aer simulations
# are already OpenMP-threaded internally, so the worker count should not exceed
# the cores actually requested from PBS -- see the note in main().
MAX_WORKERS_DEFAULT: int = 4

# GPU preference: read from environment so the PBS script can override.
# Set QUANTUM_PDE_USE_GPU=0 to force CPU execution.
_USE_GPU: bool = os.environ.get("QUANTUM_PDE_USE_GPU", "1") != "0"


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RunResult:
    """
    One row of the results table.

    Fields are grouped by purpose. Everything after `notes` is optional and
    defaults to None so that a solver which cannot supply a given metric
    simply leaves it blank rather than forcing a placeholder.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    case:           str
    solver:         str
    N:              int
    kappa:          float

    # ── Core accuracy / cost ──────────────────────────────────────────────────
    max_rel_err:    Optional[float]   # % vs reference, near-zero nodes masked
    max_abs_err:    Optional[float]
    residual:       Optional[float]   # ||Au - b|| / ||b||
    wall_time_s:    float
    converged:      bool
    notes:          str = ""

    # ── Additional accuracy norms ─────────────────────────────────────────────
    # L2 is the conventional norm for PDE convergence studies; max-norm alone
    # is noisier and unusual to report on its own.
    rel_l2_err:     Optional[float] = None   # ||u-u_ref||_2 / ||u_ref||_2
    rms_err:        Optional[float] = None

    # ── Timing statistics ─────────────────────────────────────────────────────
    wall_time_mean_s: Optional[float] = None
    wall_time_std_s:  Optional[float] = None
    n_timing_repeats: int = 1

    # ── Circuit metrics ───────────────────────────────────────────────────────
    # NOT recoverable from a saved solution vector. Populated only where the
    # underlying solver module exposes them; see the note in _run_qsvt.
    n_qubits:            Optional[int]   = None   # data-register qubits
    circuit_depth:       Optional[int]   = None
    circuit_depth_t:     Optional[int]   = None   # after transpilation
    n_gates_total:       Optional[int]   = None
    n_gates_2q:          Optional[int]   = None   # CX count: the cost driver
    success_probability: Optional[float] = None   # post-selection probability

    # ── Solver-specific internals ─────────────────────────────────────────────
    qsvt_degree:        Optional[int]   = None    # degree actually solved
    qsvt_max_degree:    Optional[int]   = None    # cap requested (cache key!)
    qsvt_kappa_eff:     Optional[float] = None
    qsvt_phases_cached: Optional[bool]  = None
    vqls_final_cost:    Optional[float] = None
    vqls_n_layers:      Optional[int]   = None
    vqls_n_restarts:    Optional[int]   = None
    hhl_epsilon:        Optional[float] = None
    hhl_scale_c:        Optional[float] = None    # proportionality constant

    # ── Reproducibility ───────────────────────────────────────────────────────
    random_seed:    Optional[int] = None


# ── Logging helpers ───────────────────────────────────────────────────────────

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


# ── Error metrics ─────────────────────────────────────────────────────────────

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


# ── Solution archiving ────────────────────────────────────────────────────────

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

    Any archive already present is read back and merged, with this invocation's
    entries superseding same-key predecessors. A scope-restricted run (`--cases`,
    `--n-values`) holds only its own solutions in memory, so a plain overwrite
    would have replaced a complete archive with a handful of entries — the same
    class of loss that `--append` prevents in the summary table.

    Parameters
    ----------
    all_solutions : dict
        Mapping of ``"<case>_<solver>_N<N>"`` to a dict carrying ``x``, ``u`` and
        optionally ``u_exact``, as accumulated by `_record`.
    """
    if not all_solutions:
        return
    path = RESULTS_DIR / "all_solutions.npz"

    flat: dict[str, np.ndarray] = {}
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as prior:
                flat.update({k: prior[k] for k in prior.files})
            log.info("Merging %d prior entry array(s) from %s", len(flat), path)
        except Exception as exc:
            log.warning("Could not read %s (%s); it will be rewritten from this "
                        "run's solutions only.", path, exc)
            flat = {}

    for key, entry in all_solutions.items():
        flat[f"{key}__x"] = entry["x"]
        flat[f"{key}__u"] = entry["u"]
        if entry.get("u_exact") is not None:
            flat[f"{key}__u_exact"] = entry["u_exact"]
    np.savez_compressed(path, **flat)
    log.info("Consolidated solutions saved to %s (%d entries this run, "
             "%d arrays total)", path, len(all_solutions), len(flat))


# ──────────────────────────────────────────────────────────────────────────────
#  Solver wrappers
#
#  Every wrapper returns a uniform 3-element core -- (u, residual, wall) --
#  plus solver-specific extras, and every one catches its own exceptions and
#  returns u=None on failure. That last point matters: a wrapper that lets an
#  exception escape takes down the whole work unit, losing the OTHER solvers'
#  results for that (case, N) too.
# ──────────────────────────────────────────────────────────────────────────────

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
            # If the matrix is not strictly tridiagonal (e.g. 4th order pentadiagonal),
            # fallback to a general solver since Thomas only works for tridiagonal.
            if np.any(np.abs(np.triu(A, 2)) > 1e-12) or np.any(np.abs(np.tril(A, -2)) > 1e-12):
                u = np.linalg.solve(A, d)
                timings.append(time.perf_counter() - t0)
                continue
                
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
             timeout_s: float = HHL_TIMEOUT_S,
             ) -> tuple[Optional[np.ndarray], float, float, bool, float, str]:
    """
    HHL via the project module solvers/quantum/hhl_1d.py, with a HARD wall-clock timeout.

    Returns (u, residual, wall_s, converged, scale_c, failure_note).

    `scale_c` is the proportionality-recovery constant. It is NOT derivable
    from the returned solution vector afterwards, so it is propagated rather
    than discarded.

    `failure_note` is empty on success, ``"hhl_timeout"`` when the budget expired
    and ``"hhl_error"`` when the solve raised. The distinction is load-bearing for
    the gap analysis, not cosmetic. Both outcomes previously yielded ``u=None`` and
    hence the single note ``"solver_error"``, which made a *measured* result
    indistinguishable from a *defect*: at N=32 and N=64 the 1-D operator reaches
    kappa ~ 1.7e3, HHL's clock register grows with kappa, and the solve genuinely
    does not finish inside an hour. That is the benchmark's finding. Classified as
    an error, those thirteen rows were scheduled for recomputation, which would
    have spent 13 h of cluster time reproducing thirteen identical timeouts.

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
        log.warning("    HHL: killed after exceeding %.0fs timeout (N=%d). "
                    "This is a recorded outcome, not a fault: kappa=%.3g here.",
                    timeout_s, N, float(np.linalg.cond(A)))
        return (None, float("nan"), time.perf_counter() - t0, False,
                float("nan"), "hhl_timeout")

    try:
        u, x_raw, c = q.get_nowait()

    except Exception as exc:
        log.warning("    HHL failed: %s", exc)
        return (None, float("nan"), time.perf_counter() - t0, False,
                float("nan"), "hhl_error")

    wall = time.perf_counter() - t0
    return u, _relative_residual(A, u, b), wall, True, float(c), ""


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
    # Cache miss: fallback to an N-dependent cap
    fallback = min(QSVT_UNCACHED_FALLBACK_DEGREE, int(kappa * 15))
    return max(100, fallback)


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


# ── Run selection ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunSelection:
    """
    Which (case, solver) combinations this invocation is to execute.

    Replaces the former `skip_qsvt` boolean, which could express exactly one of
    the restrictions a gap-fill run needs. The 2026-08-09 gap analysis put the
    1D outstanding work at 33 rows out of 140 — 20 of them a single case,
    `HET_1D_3b_gaussian_Vd300`, at five resolutions — against a driver whose only
    scope control was `--max-n`. Re-running the whole sweep to reach 33 rows would
    have recomputed 107 sound rows, several of them multi-hour HHL solves at
    N=32/64.

    Frozen so that it is hashable and unambiguously picklable across the
    `ProcessPoolExecutor` boundary, and so that no worker can mutate the parent's
    scope.

    Attributes
    ----------
    solvers : tuple of str
        Quantum solvers to execute, drawn from ("hhl", "vqls", "qsvt"). Thomas is
        always executed and recorded regardless: it costs microseconds, and it is
        the reference solution against which sub-case 3b's quantum errors are
        measured, so excluding it would silently invalidate that case.
    cases : tuple of str
        Case-identifier patterns. An empty tuple selects every case. A pattern
        containing a glob metacharacter (``*``, ``?``, ``[``) is matched with
        `fnmatch`; any other pattern is treated as a case-insensitive substring,
        so ``--cases 3b`` selects `HET_1D_3b_gaussian_Vd300` without the caller
        having to spell out the full identifier.
    """

    solvers: tuple[str, ...] = QUANTUM_SOLVERS_1D
    cases:   tuple[str, ...] = ()

    def wants_solver(self, solver: str) -> bool:
        """
        Report whether `solver` is in scope.

        Parameters
        ----------
        solver : str
            Solver key, lower case (e.g. ``"hhl"``).

        Returns
        -------
        bool
            True if the solver is to be executed and recorded.
        """
        return solver.lower() in self.solvers

    def wants_case(self, case_id: str) -> bool:
        """
        Report whether `case_id` is in scope.

        Parameters
        ----------
        case_id : str
            Case identifier as recorded in the results (e.g.
            ``"HET_1D_3b_gaussian_Vd300"``).

        Returns
        -------
        bool
            True if the case is to be executed.
        """
        if not self.cases:
            return True
        target = case_id.lower()
        for pattern in self.cases:
            p = pattern.lower()
            if any(ch in p for ch in "*?["):
                if fnmatch.fnmatch(target, p):
                    return True
            elif p in target:
                return True
        return False


# ── Per-case solver driver ────────────────────────────────────────────────────

def _run_all_solvers(
    case_id:       str,
    N:             int,
    x:             np.ndarray,
    A:             np.ndarray,
    b:             np.ndarray,
    u_exact:       Optional[np.ndarray],
    kappa:         float,
    sel:           RunSelection,
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

    A solver excluded by `sel` is neither executed *nor recorded*. The distinction
    matters under `--append`: a placeholder row for a skipped solver would
    supersede the sound row already on disk, converting a scope restriction into
    data loss.
    """
    if not sel.wants_case(case_id):
        log.info("    %s N=%d excluded by --cases; skipping.", case_id, N)
        return

    n_data_qubits = int(np.log2(N)) if N > 0 and (N & (N - 1)) == 0 else None

    # ── Thomas (classical reference) ──────────────────────────────────────────
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

    # ── HHL ───────────────────────────────────────────────────────────────────
    if sel.wants_solver("hhl"):
        u_H, res_H, t_H, conv_H, c_H, note_H = _run_hhl(A, b, N)
        if u_H is not None:
            if u_ref is not None:
                log.info("    HHL     MaxRelErr=%7.3f%%  Residual=%.3e  Time=%.3fs",
                         _max_rel_err(u_H, u_ref), res_H, t_H)
            else:
                log.info("    HHL     Residual=%.3e  Time=%.3fs", res_H, t_H)
        # The reference marker and the failure mode are both recorded; a timed-out
        # 3b row carries "rel_vs_thomas;hhl_timeout" rather than losing one to the
        # other, which is what previously hid the timeouts behind the reference note.
        _record(results, all_solutions, case_id, "HHL", N, kappa,
                x, u_H, u_ref, A, b, res_H, t_H, conv_H,
                notes=";".join(filter(None, (ref_note, note_H))),
                n_qubits=n_data_qubits, hhl_epsilon=HHL_EPSILON,
                hhl_scale_c=c_H if u_H is not None else None)

    # ── VQLS ──────────────────────────────────────────────────────────────────
    if sel.wants_solver("vqls"):
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

    # ── QSVT ──────────────────────────────────────────────────────────────────
    if not sel.wants_solver("qsvt"):
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


# ── Case runners ──────────────────────────────────────────────────────────────

def run_1d_generic_poisson_single_N(
    N: int, sel: RunSelection, results: list[RunResult], all_solutions: dict,
    order: int,
) -> None:
    """Section 1: generic Poisson, homogeneous Dirichlet BCs, three sources."""
    _banner(f"SECTION 1 - Generic Poisson (fS, fL, fH), homogeneous BCs, N={N}")

    case_map = {
        "fS": "poisson_1d_fS_hom",
        "fL": "poisson_1d_fL_hom",
        "fH": "poisson_1d_fH_hom",
    }

    for src_key, case_name in case_map.items():
        _section(f"Source: {src_key}  (N={N})")

        built = cases.get(case_name).build(N)
        if order == 4:
            from problems.poisson_1d_4th import PoissonProblem1D4th
            import dataclasses
            dx = built.spacings[0]
            alpha = float(dx**2 * built.f_values[0] - built.b[0])
            beta = float(dx**2 * built.f_values[-1] - built.b[-1])
            prob_4th = PoissonProblem1D4th(
                N=N,
                f_vals=built.f_values,
                alpha=alpha,
                beta=beta
            )
            built = dataclasses.replace(built, A=prob_4th.A, b=prob_4th.b, kappa=prob_4th.kappa)
        x, A, b, u_exact, kappa = built.coords[0], built.A, built.b, built.exact, built.kappa
        case_id = f"1D_Poisson_{src_key}_hom"

        log.info("  N=%3d  kappa=%.2f  case=%s", N, kappa, case_id)

        _run_all_solvers(case_id, N, x, A, b, u_exact, kappa,
                         sel, results, all_solutions)


def run_1d_generic_poisson_nonhom_single_N(
    N: int, sel: RunSelection, results: list[RunResult], all_solutions: dict,
    order: int,
) -> None:
    """
    Section 1b: generic Poisson, fS source, non-homogeneous Dirichlet BCs.

    u'' = sin(pi x),  u(0) = alpha,  u(1) = beta
      =>  u(x) = -sin(pi x)/pi^2 + (beta - alpha) x + alpha
    """
    _banner(f"SECTION 1b - Generic Poisson (fS), non-homogeneous BCs, N={N}")

    built = cases.get("poisson_1d_fS_nonhom").build(N)
    if order == 4:
        from problems.poisson_1d_4th import PoissonProblem1D4th
        import dataclasses
        dx = built.spacings[0]
        alpha = float(dx**2 * built.f_values[0] - built.b[0])
        beta = float(dx**2 * built.f_values[-1] - built.b[-1])
        prob_4th = PoissonProblem1D4th(
            N=N,
            f_vals=built.f_values,
            alpha=alpha,
            beta=beta
        )
        built = dataclasses.replace(built, A=prob_4th.A, b=prob_4th.b, kappa=prob_4th.kappa)
    x, A, b, u_exact, kappa = built.coords[0], built.A, built.b, built.exact, built.kappa
    case_id = "1D_Poisson_fS_nonhom"

    log.info("  N=%3d  kappa=%.2f  case=%s", N, kappa, case_id)

    _run_all_solvers(case_id, N, x, A, b, u_exact, kappa,
                     sel, results, all_solutions)


def run_1d_het_single_N(
    N: int, sel: RunSelection, results: list[RunResult], all_solutions: dict,
    order: int,
) -> None:
    """
    Section 2: HET axial Poisson, three sub-cases.

      3a  linear profile, homogeneous Dirichlet, analytical reference
      3b  Gaussian profile, V_d = 300 V anode, Dirichlet, Thomas reference
      3c  Gaussian profile, Neumann(x=0) - Dirichlet(x=1), quadrature reference

    All three cases are read from core.cases (het_1d_3a_linear,
    het_1d_3b_gaussian_Vd300, het_1d_3c_neumann), rather than built locally,
    since Phase 1 of the entry-point consolidation proved every one of them
    bit-identical to what this file used to construct by hand.
    """
    _banner(f"SECTION 2 - HET Axial Poisson, N={N}")

    def apply_4th_order(built):
        if order != 4:
            return built
        from problems.poisson_1d_4th import PoissonProblem1D4th
        import dataclasses
        dx = built.spacings[0]
        alpha = float(dx**2 * built.f_values[0] - built.b[0])
        beta = float(dx**2 * built.f_values[-1] - built.b[-1])
        prob_4th = PoissonProblem1D4th(
            N=N,
            f_vals=built.f_values,
            alpha=alpha,
            beta=beta
        )
        return dataclasses.replace(built, A=prob_4th.A, b=prob_4th.b, kappa=prob_4th.kappa)

    # ── Sub-case 3a: linear profile, homogeneous BCs ──────────────────────────
    _section(f"Sub-case 3a: linear profile, homogeneous BCs  (N={N})")

    built = cases.get("het_1d_3a_linear").build(N)
    built = apply_4th_order(built)
    x, A, b, u_exact, kappa = built.coords[0], built.A, built.b, built.exact, built.kappa
    case_id = "HET_1D_3a_linear_hom"

    log.info("  N=%3d  kappa=%.2f  sub-case=3a", N, kappa)
    _run_all_solvers(case_id, N, x, A, b, u_exact, kappa,
                     sel, results, all_solutions)

    # ── Sub-case 3b: Gaussian profile, V_d = 300 V ────────────────────────────
    # No closed-form solution, so the Thomas result is the reference.
    _section(f"Sub-case 3b: Gaussian profile, V_d=300V, Dirichlet BCs  (N={N})")

    built = cases.get("het_1d_3b_gaussian_Vd300").build(N)
    built = apply_4th_order(built)
    x, A, b, kappa = built.coords[0], built.A, built.b, built.kappa
    case_id = "HET_1D_3b_gaussian_Vd300"

    log.info("  N=%3d  kappa=%.2f  sub-case=3b  V_d=300V", N, kappa)
    _run_all_solvers(case_id, N, x, A, b, None, kappa,
                     sel, results, all_solutions, reference="thomas")

    # Electric field diagnostic (derived quantity of physical interest).
    thomas_key = f"{case_id}_Thomas_N{N}"
    if thomas_key in all_solutions:
        u_T = all_solutions[thomas_key]["u"]
        log.info("    Peak |E| (Thomas) = %.3e V/m",
                 float(np.max(np.abs(-np.gradient(u_T, x)))))

    # ── Sub-case 3c: Gaussian profile, Neumann-Dirichlet BCs ──────────────────
    _section(f"Sub-case 3c: Gaussian profile, Neumann-Dirichlet BCs  (N={N})")
    log.info("  BCs: phi'(0)=0 (Neumann), phi(1)=0 (Dirichlet)")
    log.info("  Reference: quadrature of the double integral of the source")

    built = cases.get("het_1d_3c_neumann").build(N)
    if order == 4:
        log.warning("Sub-case 3c (Neumann) not supported by PoissonProblem1D4th. Skipping.")
    else:
        x, A, b, u_exact, kappa = built.coords[0], built.A, built.b, built.exact, built.kappa
        case_id = "HET_1D_3c_gaussian_NeumannDirichlet"

        log.info("  N=%3d  kappa=%.2f  sub-case=3c  h=%.5f", N, kappa, built.spacings[0])
        _run_all_solvers(case_id, N, x, A, b, u_exact, kappa,
                         sel, results, all_solutions)


# ── Result serialisation ──────────────────────────────────────────────────────

def _load_existing_results(path: Path) -> list[RunResult]:
    """
    Load rows from a previous invocation for `--append` to build on.

    Unknown fields in the JSON (e.g. from an older schema) are dropped rather than
    raising, and a missing or unparsable file yields an empty list rather than
    aborting: `--append` must never be the reason a sweep fails to start.

    Parameters
    ----------
    path : Path
        Location of a previous ``results_full.json``.

    Returns
    -------
    list of RunResult
        Rows recovered, oldest first. Empty if the file is absent or unreadable.
    """
    if not path.exists():
        return []
    try:
        with open(path) as fh:
            rows = json.load(fh)
    except Exception as exc:
        log.warning("Could not parse %s (%s); starting without prior results.",
                    path, exc)
        return []
    valid = {f.name for f in dataclasses.fields(RunResult)}
    out: list[RunResult] = []
    for d in rows:
        try:
            out.append(RunResult(**{k: v for k, v in d.items() if k in valid}))
        except Exception as exc:
            log.warning("Skipping unreadable prior row: %s", exc)
    return out


def _dedupe_results(results: list[RunResult]) -> list[RunResult]:
    """
    Collapse superseded rows, retaining the most recent for each identity.

    A sweep is uniquely indexed by (case, solver, N); a second row for that triple
    is a recomputation of the first, not an additional datum. `--append` merges the
    previous ``results_full.json`` ahead of the rows produced by this invocation,
    so the later occurrence of any key is by construction the newer measurement.

    Parameters
    ----------
    results : list of RunResult
        Rows in chronological order, oldest first.

    Returns
    -------
    list of RunResult
        Deduplicated rows, in first-seen order so the file's layout stays stable
        across appends rather than reshuffling on every save.
    """
    keep: dict = {}
    for r in results:
        keep[(r.case, r.solver, r.N)] = r
    if len(keep) != len(results):
        log.info("Superseded %d duplicate row(s) on (case, solver, N).",
                 len(results) - len(keep))
    return list(keep.values())


def _save_results(results: list[RunResult]) -> None:
    """Write the results table to JSON and CSV."""
    results = _dedupe_results(results)
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


def _save_run_metadata(N_values: list[int], sel: RunSelection,
                       max_workers: int, order: int = 2,
                       sections: Optional[list[str]] = None,
                       phase_tag: Optional[str] = None) -> None:
    """
    Environment and run-configuration provenance.

    Written BEFORE the sweep starts so that it survives a crash or a walltime
    kill; without it a partial result set is not reproducible.

    Parameters
    ----------
    N_values : list of int
        Resolutions in scope for this invocation.
    sel : RunSelection
        Solver and case scope, recorded so a gap-fill run's restriction is part of
        the provenance rather than being inferable only from the absent rows.
    max_workers : int
        Worker process count.
    order : int, optional
        Spatial discretisation order, 2 or 4. Previously recorded nowhere in 1D,
        which left the two orders' metadata indistinguishable.
    sections : list of str, optional
        Section labels in scope (see `SECTION_FAMILIES`).
    phase_tag : str, optional
        Step label; when given, the record is additionally written to
        ``run_metadata_<tag>.json`` so successive steps of one job do not
        overwrite each other's provenance.
    """
    meta: dict = {
        # ── Environment ───────────────────────────────────────────────────────
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
        # ── Run configuration (not derivable from the results table) ──────────
        "N_values":            N_values,
        "order":               order,
        "sections":            sections if sections is not None else list(SECTION_FAMILIES),
        "solvers":             list(sel.solvers),
        "cases_filter":        list(sel.cases),
        "phase_tag":           phase_tag,
        # Retained under its historical name so existing readers of
        # run_metadata.json continue to resolve it; it is now derived from the
        # solver selection rather than being an independent flag.
        "skip_qsvt":           not sel.wants_solver("qsvt"),
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

    for name in filter(None, ("run_metadata.json",
                              f"run_metadata_{phase_tag}.json" if phase_tag else None)):
        with open(RESULTS_DIR / name, "w") as f:
            json.dump(meta, f, indent=2)
        log.info("Run metadata saved to %s", RESULTS_DIR / name)


# ── Work unit dispatch ────────────────────────────────────────────────────────

def _init_worker(results_dir: Path, qsvt_fallback_degree: int) -> None:
    """
    Propagates the main process's resolved configuration into a worker.

    This is load-bearing, not defensive. `ProcessPoolExecutor` is constructed with
    ``max_tasks_per_child=1``, and CPython selects the **spawn** start method
    whenever that argument is given without an explicit context. A spawned worker
    re-imports this module from scratch, so it sees the module-level defaults rather
    than the values `main` assigned under ``global`` — and `_save_solution` reads
    `RESULTS_DIR` from inside the worker.

    The consequence was silent and destructive: an ``--order 4`` sweep wrote its
    summary to ``results/1Dhpc_run_4th`` from the parent whilst every per-solution
    archive went to ``results/1Dhpc_run`` from the workers, overwriting the
    2nd-order fields with 4th-order ones. The summaries were left describing
    solutions that no longer matched the archives beside them.

    Parameters
    ----------
    results_dir : Path
        Output directory resolved by `main`, including the 4th-order variant.
    qsvt_fallback_degree : int
        Degree used when no cached QSVT phase set matches, likewise resolved by
        `main` and likewise previously lost on spawn.
    """
    global RESULTS_DIR, QSVT_UNCACHED_FALLBACK_DEGREE
    RESULTS_DIR = results_dir
    QSVT_UNCACHED_FALLBACK_DEGREE = qsvt_fallback_degree


def _execute_work_unit(work_type: str, N: int, sel: RunSelection, order: int
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

    # The dispatch table and SECTION_FAMILIES must stay in step, otherwise a
    # --sections argument resolves to a family this function cannot execute and
    # the unit is silently skipped with a single log line.
    assert set(dispatch) == set(SECTION_FAMILIES.values()), \
        "dispatch table and SECTION_FAMILIES have diverged"

    fn = dispatch.get(work_type)
    if fn is None:
        log.error("Unknown work_type '%s'; skipping.", work_type)
        return results, solutions

    fn(N, sel, results, solutions, order)
    return results, solutions


# ── Main entry point ──────────────────────────────────────────────────────────

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
    `#PBS -l select=1:ncpus=4` in hpc/jobs/submit_hpc_1D.sh. Raising one without raising
    the other oversubscribes the node and slows the sweep down.

    On a GPU node, use `--max-workers 1` with QUANTUM_PDE_USE_GPU=1 to
    serialise execution through the single GPU and avoid CUDA context
    conflicts between processes.
    """
    parser = argparse.ArgumentParser(
        description="Full 1-D HPC benchmark sweep for quantum PDE solvers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--order", type=int, choices=[2, 4], default=2,
                        help="Spatial discretisation order (2 or 4).")
    parser.add_argument(
        "--max-n", type=int, default=max(N_VALUES_ALL),
        help=f"Largest N to include. The full sweep is {N_VALUES_ALL}; "
             f"use e.g. --max-n 16 for a fast validation pass "
             f"(default: {max(N_VALUES_ALL)}, i.e. the whole sweep).",
    )
    parser.add_argument(
        "--n-values", type=str, default=None,
        help="Comma-separated resolutions to run, e.g. --n-values 32,64. Takes "
             "precedence over --max-n. Required for a gap-fill run, where the "
             "outstanding resolutions are not a prefix of the sweep.",
    )
    parser.add_argument(
        "--sections", type=str, default=",".join(SECTION_FAMILIES),
        help=f"Comma-separated section labels to run, from "
             f"{sorted(SECTION_FAMILIES)} (default: all).",
    )
    parser.add_argument(
        "--solvers", type=str, default=",".join(QUANTUM_SOLVERS_1D),
        help=f"Comma-separated quantum solvers to run, from "
             f"{list(QUANTUM_SOLVERS_1D)} (default: all). Thomas always runs: it "
             f"costs microseconds and is sub-case 3b's reference solution.",
    )
    parser.add_argument(
        "--cases", type=str, default=None,
        help="Comma-separated case filters. A value containing *, ? or [ is "
             "treated as a glob against the case identifier; anything else as a "
             "case-insensitive substring. E.g. --cases 3b selects only "
             "HET_1D_3b_gaussian_Vd300.",
    )
    parser.add_argument(
        "--skip-qsvt", action="store_true",
        help="Omit QSVT from all cases. Use for a rapid validation sweep or "
             "if the QSVT module is unavailable. Equivalent to removing qsvt "
             "from --solvers; retained because every existing submission script "
             "passes it.",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Merge with the rows already in results_full.json instead of "
             "replacing them. Rows are superseded on (case, solver, N), so a "
             "re-run of a triple replaces it rather than duplicating it. Required "
             "for any partial or gap-fill run - without it, a run restricted to "
             "33 outstanding rows would discard the other 107.",
    )
    parser.add_argument(
        "--phase-tag", default=None,
        help="Label for this step, recorded in the log session banner and in "
             "run_metadata_<tag>.json. Lets a multi-step PBS job attribute each "
             "row to the step that produced it.",
    )
    parser.add_argument(
        "--max-workers", type=int, default=MAX_WORKERS_DEFAULT,
        help=f"Parallel worker processes (default: {MAX_WORKERS_DEFAULT}). "
             f"Must not exceed the ncpus requested from PBS. Set to 1 for "
             f"serial execution (required on GPU).",
    )
    args = parser.parse_args()

    # ── Resolve scope ─────────────────────────────────────────────────────────
    if args.n_values:
        try:
            N_values = [int(tok) for tok in args.n_values.split(",") if tok.strip()]
        except ValueError:
            parser.error(f"--n-values {args.n_values!r} is not a comma-separated "
                         f"list of integers.")
        unknown = [n for n in N_values if n not in N_VALUES_ALL]
        if unknown:
            parser.error(f"--n-values contains {unknown}, which are not in the "
                         f"sweep {N_VALUES_ALL}.")
    else:
        N_values = [n for n in N_VALUES_ALL if n <= args.max_n]
        if not N_values:
            parser.error(f"--max-n {args.max_n} excludes every N in {N_VALUES_ALL}.")

    sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    unknown_sections = [s for s in sections if s not in SECTION_FAMILIES]
    if unknown_sections:
        parser.error(f"--sections contains {unknown_sections}; valid labels are "
                     f"{sorted(SECTION_FAMILIES)}.")

    solvers = tuple(s.strip().lower() for s in args.solvers.split(",") if s.strip())
    unknown_solvers = [s for s in solvers if s not in QUANTUM_SOLVERS_1D]
    if unknown_solvers:
        parser.error(f"--solvers contains {unknown_solvers}; valid solvers are "
                     f"{list(QUANTUM_SOLVERS_1D)}. Thomas is always run and is "
                     f"not selectable.")
    if args.skip_qsvt:
        solvers = tuple(s for s in solvers if s != "qsvt")

    sel = RunSelection(
        solvers=solvers,
        cases=tuple(c.strip() for c in (args.cases or "").split(",") if c.strip()),
    )

    # ── Resolve backend and report configuration ──────────────────────────────
    backend = get_aer_backend(prefer_gpu=_USE_GPU)

    global RESULTS_DIR, LOG_FILE, QSVT_UNCACHED_FALLBACK_DEGREE
    
    if args.order == 4:
        # Avoid overwriting 2nd-order results
        RESULTS_DIR = Path("results") / "1Dhpc_run_4th"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        LOG_FILE = RESULTS_DIR / "run.log"
        QSVT_UNCACHED_FALLBACK_DEGREE = 5000
        
        _redirect_log_file(LOG_FILE)

    _log_session_header(args.phase_tag)
    _banner("QUANTUM PDE SOLVERS - 1D HPC BENCHMARK")
    log.info("  Order         : %d", args.order)
    log.info("  N values      : %s", N_values)
    log.info("  Sections      : %s", sections)
    log.info("  Solvers       : Thomas + %s", list(sel.solvers) or "(none)")
    log.info("  Case filter   : %s", list(sel.cases) or "(all cases)")
    log.info("  Append        : %s", args.append)
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
    _save_run_metadata(N_values, sel, args.max_workers, order=args.order,
                       sections=sections, phase_tag=args.phase_tag)

    t_global_start = time.perf_counter()
    results: list[RunResult] = []
    all_solutions: dict = {}

    # Prior rows are merged ahead of anything this invocation produces, so
    # `_dedupe_results` resolves a repeated (case, solver, N) in favour of the
    # new measurement. Without --append a scope-restricted run would write only
    # its own rows and discard every sound row already on disk.
    if args.append:
        prior = _load_existing_results(RESULTS_DIR / "results_full.json")
        if prior:
            log.info("--append: merging with %d prior row(s).", len(prior))
        results.extend(prior)

    # ── Build the work unit list ──────────────────────────────────────────────
    # Smallest N first: with only a handful of workers and a large spread in
    # per-unit cost (HHL/QSVT scale badly with kappa -- see Problem 2 below),
    # dispatching largest-N units first saturates every worker on the slowest
    # cases immediately and starves the fast, informative small-N validation
    # runs behind them. Ascending order guarantees N=4/8/16 complete and are
    # written to disk before any worker can get tied up on N=32/64.
    work_units = [
        (SECTION_FAMILIES[section], N)
        for N in sorted(N_values)
        for section in sections
    ]

    if args.max_workers == 1:
        log.info("Serial execution mode (max_workers=1).")
        for work_type, N in work_units:
            try:
                partial_results, partial_solutions = _execute_work_unit(
                    work_type, N, sel, args.order)
                results.extend(partial_results)
                all_solutions.update(partial_solutions)
                # Written after every unit, not only at the end. A walltime kill
                # previously lost the whole summary table whilst leaving the
                # per-solution archives on disk, so the sweep's own record of what
                # it had achieved had to be reconstructed from the .npz filenames.
                _save_results(results)
            except Exception as exc:
                log.error("Work unit failed: type=%s N=%d - %s",
                          work_type, N, exc, exc_info=True)
    else:
        log.info("Parallel execution: %d work units across %d workers.",
                 len(work_units), args.max_workers)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.max_workers,
            max_tasks_per_child=1,   # fresh process per work unit; forces spawn
            initializer=_init_worker,
            initargs=(RESULTS_DIR, QSVT_UNCACHED_FALLBACK_DEGREE),
        ) as executor:
            futures = {
                executor.submit(_execute_work_unit, wt, N, sel, args.order): (wt, N)
                for wt, N in work_units
            }
            for future in concurrent.futures.as_completed(futures):
                work_type, N = futures[future]
                try:
                    partial_results, partial_solutions = future.result()
                    results.extend(partial_results)
                    all_solutions.update(partial_solutions)
                    _save_results(results)
                    log.info("Work unit done: type=%-24s N=%-3d "
                             "(%d results so far).",
                             work_type, N, len(results))
                except Exception as exc:
                    log.error("Work unit failed: type=%s N=%d - %s",
                              work_type, N, exc, exc_info=True)

    # ── Persist everything ────────────────────────────────────────────────────
    _save_results(results)
    _save_all_solutions(all_solutions)

    elapsed = time.perf_counter() - t_global_start
    _banner(f"Benchmark complete. Total elapsed time: {elapsed:.1f} s")
    log.info("Results written to: %s", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()
