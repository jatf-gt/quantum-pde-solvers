"""
Sweep drivers for the benchmarking framework.

This module provides the top-level orchestration functions that wire together
problem construction, solver invocation, metric collection, and result
persistence. It is the primary entry point for HPC runner scripts.

Sweep catalogue
---------------
  run_primary_1d          Primary 1D benchmark: all solvers × all N.
  run_equal_accuracy_1d   Equal-accuracy protocol for 1D cases.
  run_sensitivity_1d      OAT sensitivity sweep for 1D cases (N ∈ {4, 8}).
  run_primary_4th_1d      Primary 1D benchmark with 4th-order discretisation.

Each function returns a list of BenchmarkResult objects and optionally
writes to a SweepArchive. Incremental writing is used throughout: each
result is persisted immediately after the solver returns, so partial
progress survives a walltime kill on HPC.

Design principles
-----------------
  1. Solvers are called via the existing (A, b) → SolverResult interface
     in solvers/quantum/. This module does not import solver internals.
  2. The Thomas algorithm is always run first and its solution stored as
     the reference for all subsequent quantum solver comparisons.
  3. All metric computation is delegated to benchmark/equal_accuracy.py
     (_build_base_result) to ensure a single point of truth.
  4. Circuit metric extraction is optional (extract_circuits flag) to
     allow fast sweeps without the transpilation overhead.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

from benchmark.metrics import BenchmarkResult, compute_residual
from benchmark.equal_accuracy import (
    _build_base_result,
    sweep_hhl_equal_accuracy,
    sweep_vqls_equal_accuracy,
    sweep_qsvt_equal_accuracy,
    EqualAccuracyResult,
    DEFAULT_R_TARGET,
    DEFAULT_BAND_FACTOR,
    HHL_EPSILON_GRID,
    VQLS_NLAYERS_GRID,
    QSVT_MAXDEGREE_GRID,
)
from benchmark.sensitivity import (
    run_all_sensitivity_sweeps,
    SensitivitySweepResult,
)
from benchmark.results_io import SweepArchive

log = logging.getLogger(__name__)


# -- Problem builders ----------------------------------------------------------

def _build_problem_2nd(
    N: int,
    source_fn: str,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
    """
    Build the 2nd-order 1D Poisson problem.

    Returns (A, b, x, kappa, u_exact).
    u_exact is None if no analytical solution exists.
    """
    from problems.poisson_1d import PoissonProblem1D
    from core.config import SimConfig1D
    from core.exact_solutions import EXACT_SOLUTIONS_1D

    cfg = SimConfig1D(N=N, epsilon=0.01, source_fn=source_fn,
                      alpha=alpha, beta=beta)
    prob = PoissonProblem1D(cfg)

    exact_fn = EXACT_SOLUTIONS_1D.get(source_fn)
    u_exact = exact_fn(prob.x) if (exact_fn and alpha == 0.0 and beta == 0.0) else None

    return prob.A, prob.b, prob.x, prob.kappa, u_exact


def _build_problem_4th(
    N: int,
    source_fn: str,
    alpha: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, Optional[np.ndarray]]:
    """
    Build the 4th-order 1D Poisson problem.

    Returns (A, b, x, kappa, u_exact).
    """
    from problems.poisson_1d_4th import PoissonProblem1D4th

    prob = PoissonProblem1D4th(N=N, source_fn=source_fn,
                               alpha=alpha, beta=beta)
    u_exact = prob.exact_solution()
    return prob.A, prob.b, prob.x, prob.kappa, u_exact


# -- Thomas reference solver ---------------------------------------------------

def _run_thomas(
    A: np.ndarray,
    b: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Solve using NumPy's direct solver as the Thomas reference.

    Returns (u_thomas, residual, wall_time_s).
    NumPy's solver is used rather than the Thomas algorithm module to
    handle both tridiagonal and pentadiagonal systems without branching.
    """
    t0 = time.perf_counter()
    u = np.linalg.solve(A, b)
    wall = time.perf_counter() - t0
    return u, compute_residual(A, u, b), wall


# -- Quantum solver wrappers ---------------------------------------------------

def _run_hhl(
    A: np.ndarray,
    b: np.ndarray,
    epsilon: float = 0.01,
    extract_circuits: bool = True,
) -> tuple[Optional[np.ndarray], float, float, dict]:
    """
    Run HHL and return (u, residual, wall_time_s, extra_fields).
    extra_fields contains all HHL-specific BenchmarkResult fields.
    """
    from solvers.quantum.hhl_1d import hhl_solve_system, HHLConfig1D
    from benchmark.metrics import extract_circuit_metrics

    cfg = HHLConfig1D(epsilon=epsilon)
    t0 = time.perf_counter()
    try:
        result = hhl_solve_system(A, b, config=cfg)
        wall = time.perf_counter() - t0
        u = np.array(result.solution)

        extra: dict = {
            "hhl_epsilon":       epsilon,
            "hhl_trotter_steps": int(np.ceil(1.0 / epsilon)),
        }

        # Proportionality recovery residual
        if hasattr(result, "raw_state") and result.raw_state is not None:
            Ax_raw = A @ result.raw_state
            c_val  = float(np.dot(b, Ax_raw) / (np.dot(Ax_raw, Ax_raw) + 1e-300))
            extra["proportionality_residual"] = compute_residual(
                A, c_val * result.raw_state, b
            )

        if extract_circuits and hasattr(result, "circuit"):
            try:
                extra["circuit_metrics"] = extract_circuit_metrics(
                    result.circuit, optimisation_level=1
                )
            except Exception as e:
                log.warning("  HHL circuit extraction failed: %s", e)

        return u, compute_residual(A, u, b), wall, extra

    except Exception as exc:
        log.warning("  HHL failed (epsilon=%.4f): %s", epsilon, exc)
        return None, float("nan"), time.perf_counter() - t0, {}


def _run_vqls(
    A: np.ndarray,
    b: np.ndarray,
    n_layers: int = 2,
    n_restarts: int = 3,
    extract_circuits: bool = True,
) -> tuple[Optional[np.ndarray], float, float, dict]:
    """
    Run VQLS and return (u, residual, wall_time_s, extra_fields).
    """
    from solvers.quantum.vqls_1d import vqls_solve_system, VQLSConfig1D
    from benchmark.metrics import extract_circuit_metrics

    cfg = VQLSConfig1D(n_layers=n_layers, n_restarts=n_restarts)
    t0 = time.perf_counter()
    try:
        result = vqls_solve_system(A, b, config=cfg)
        wall = time.perf_counter() - t0
        u = np.array(result.solution)

        extra: dict = {
            "vqls_n_layers":      n_layers,
            "vqls_n_restarts":    n_restarts,
            "vqls_cost_final":    float(result.final_cost),
            "vqls_n_evaluations": getattr(result, "n_evaluations", None),
            "vqls_converged":     getattr(result, "converged", None),
        }

        if extract_circuits and hasattr(result, "circuit"):
            try:
                extra["circuit_metrics"] = extract_circuit_metrics(
                    result.circuit, optimisation_level=1
                )
            except Exception as e:
                log.warning("  VQLS circuit extraction failed: %s", e)

        return u, compute_residual(A, u, b), wall, extra

    except Exception as exc:
        log.warning("  VQLS failed (n_layers=%d): %s", n_layers, exc)
        return None, float("nan"), time.perf_counter() - t0, {}


def _run_qsvt(
    A: np.ndarray,
    b: np.ndarray,
    max_degree: Optional[int] = None,
    epsilon: float = 0.01,
    extract_circuits: bool = True,
) -> tuple[Optional[np.ndarray], float, float, dict]:
    """
    Run QSVT and return (u, residual, wall_time_s, extra_fields).
    """
    from solvers.quantum.qsvt_1d import qsvt_solve_system, QSVTConfig1D
    from benchmark.metrics import extract_circuit_metrics

    cfg = QSVTConfig1D(epsilon=epsilon, max_degree=max_degree,
                       angle_method="auto")
    t0 = time.perf_counter()
    try:
        result = qsvt_solve_system(A, b, config=cfg)
        wall = time.perf_counter() - t0
        u = np.array(result.solution)

        alpha_sub = float(np.linalg.norm(A, ord=2))
        kappa_val = float(
            np.abs(np.linalg.eigvalsh(A)).max()
            / np.abs(np.linalg.eigvalsh(A)).min()
        )

        extra: dict = {
            "qsvt_polynomial_degree": getattr(result, "degree", None),
            "qsvt_max_degree_cap":    max_degree,
            "qsvt_subnormalisation":  alpha_sub,
            "qsvt_kappa_eff":         kappa_val * alpha_sub,
            "qsvt_angle_method":      "auto",
            "qsvt_phase_from_cache":  getattr(result, "phase_from_cache", None),
            "phase_lookup_time_s":    getattr(result, "phase_lookup_time_s", None),
        }

        if hasattr(result, "raw_state") and result.raw_state is not None:
            Ax_raw = A @ result.raw_state
            c_val  = float(np.dot(b, Ax_raw) / (np.dot(Ax_raw, Ax_raw) + 1e-300))
            extra["proportionality_residual"] = compute_residual(
                A, c_val * result.raw_state, b
            )

        if extract_circuits and hasattr(result, "circuit"):
            try:
                extra["circuit_metrics"] = extract_circuit_metrics(
                    result.circuit, optimisation_level=1
                )
            except Exception as e:
                log.warning("  QSVT circuit extraction failed: %s", e)

        return u, compute_residual(A, u, b), wall, extra

    except Exception as exc:
        log.warning("  QSVT failed (max_degree=%s): %s", max_degree, exc)
        return None, float("nan"), time.perf_counter() - t0, {}


# -- Primary 1D sweep ----------------------------------------------------------

def run_primary_1d(
    cases: list[dict],
    N_values: list[int],
    solvers: list[str],
    archive: Optional[SweepArchive] = None,
    extract_circuits: bool = True,
    hhl_epsilon: float = 0.01,
    vqls_n_layers: int = 2,
    vqls_n_restarts: int = 3,
    qsvt_max_degree: Optional[int] = None,
    qsvt_epsilon: float = 0.01,
    discretisation_order: int = 2,
    backend_name: str = "aer_statevector",
    hardware_run: bool = False,
    backend_shots: Optional[int] = None,
) -> list[BenchmarkResult]:
    """
    Run the primary 1D benchmark sweep.

    For each (case, N, solver) combination:
      1. Build the problem (2nd or 4th order).
      2. Run the Thomas algorithm and store as reference.
      3. Run each quantum solver and compute all metrics.
      4. Write results to the archive incrementally.

    Parameters
    ----------
    cases : list[dict]
        Each dict must contain: 'case_id', 'source_fn', 'alpha', 'beta'.
    N_values : list[int]
        Problem sizes to sweep.
    solvers : list[str]
        Solvers to run: any subset of ['thomas', 'hhl', 'vqls', 'qsvt'].
    archive : SweepArchive, optional
        If provided, results are written incrementally after each solve.
    extract_circuits : bool
        If True, extract circuit metrics (adds transpilation overhead).
    hhl_epsilon : float
        HHL QPE precision parameter for the primary sweep.
    vqls_n_layers : int
        VQLS ansatz layers for the primary sweep.
    vqls_n_restarts : int
        VQLS restart count for the primary sweep.
    qsvt_max_degree : int or None
        QSVT polynomial degree cap. None = uncapped.
    qsvt_epsilon : float
        QSVT approximation precision.
    discretisation_order : int
        Spatial discretisation order: 2 or 4.
    backend_name : str
        Backend identifier for metadata.
    hardware_run : bool
        True if running on real hardware.
    backend_shots : int or None
        Shot count for hardware runs.

    Returns
    -------
    list[BenchmarkResult]
        All results from the sweep.
    """
    build_fn = _build_problem_2nd if discretisation_order == 2 else _build_problem_4th
    all_results: list[BenchmarkResult] = []

    for case in cases:
        case_id   = case["case_id"]
        source_fn = case["source_fn"]
        alpha     = case.get("alpha", 0.0)
        beta      = case.get("beta",  0.0)

        for N in N_values:
            log.info(
                "Primary sweep: case=%s  N=%d  order=%d",
                case_id, N, discretisation_order,
            )

            try:
                A, b, x, kappa, u_exact = build_fn(N, source_fn, alpha, beta)
            except Exception as exc:
                log.error("  Problem build failed: %s", exc)
                continue

            # Thomas reference (always run first)
            u_thomas, r_thomas, t_thomas = _run_thomas(A, b)
            rec_thomas = _build_base_result(
                case_id=case_id, solver="thomas", N=N, kappa=kappa,
                source_fn=source_fn, alpha_bc=alpha, beta_bc=beta,
                discretisation_order=discretisation_order,
                u_solver=u_thomas, A=A, b=b,
                u_thomas=u_thomas, u_exact=u_exact,
                wall_time_s=t_thomas, r_target=None,
                backend_name="numpy_direct", hardware_run=False,
            )
            if "thomas" in solvers:
                all_results.append(rec_thomas)
                if archive:
                    archive.write_solution(
                        case_id, "thomas", N, x, u_thomas, u_exact,
                        discretisation_order=discretisation_order,
                    )
                    archive.append_primary([rec_thomas])

            # Quantum solvers
            solver_dispatch = {
                "hhl":  lambda: _run_hhl(A, b, epsilon=hhl_epsilon,
                                         extract_circuits=extract_circuits),
                "vqls": lambda: _run_vqls(A, b, n_layers=vqls_n_layers,
                                          n_restarts=vqls_n_restarts,
                                          extract_circuits=extract_circuits),
                "qsvt": lambda: _run_qsvt(A, b, max_degree=qsvt_max_degree,
                                          epsilon=qsvt_epsilon,
                                          extract_circuits=extract_circuits),
            }

            for solver in solvers:
                if solver == "thomas":
                    continue
                if solver not in solver_dispatch:
                    log.warning("  Unknown solver '%s'; skipping.", solver)
                    continue

                log.info("  Running %s...", solver.upper())
                u_sol, residual, wall, extra = solver_dispatch[solver]()

                if u_sol is None:
                    log.warning("  %s returned no solution; skipping.", solver.upper())
                    continue

                rec = _build_base_result(
                    case_id=case_id, solver=solver, N=N, kappa=kappa,
                    source_fn=source_fn, alpha_bc=alpha, beta_bc=beta,
                    discretisation_order=discretisation_order,
                    u_solver=u_sol, A=A, b=b,
                    u_thomas=u_thomas, u_exact=u_exact,
                    wall_time_s=wall, r_target=None,
                    backend_name=backend_name,
                    hardware_run=hardware_run,
                    backend_shots=backend_shots,
                )

                # Apply extra fields from solver wrapper
                for field_name, field_val in extra.items():
                    if hasattr(rec, field_name):
                        setattr(rec, field_name, field_val)

                all_results.append(rec)
                log.info(
                    "  %s: residual=%.4e  err_vs_exact=%s%%  time=%.2fs",
                    solver.upper(), rec.residual,
                    f"{rec.max_rel_err_vs_exact:.3f}"
                    if rec.max_rel_err_vs_exact else "N/A",
                    wall,
                )

                if archive:
                    archive.write_solution(
                        case_id, solver, N, x, u_sol, u_exact, u_thomas,
                        discretisation_order=discretisation_order,
                    )
                    archive.append_primary([rec])

    return all_results


# -- Equal-accuracy sweep ------------------------------------------------------

def run_equal_accuracy_1d(
    cases: list[dict],
    N_values: list[int],
    solvers: list[str],
    archive: Optional[SweepArchive] = None,
    r_target: float = DEFAULT_R_TARGET,
    band_factor: float = DEFAULT_BAND_FACTOR,
    hhl_epsilon_grid: list[float] = HHL_EPSILON_GRID,
    vqls_n_layers_grid: list[int] = VQLS_NLAYERS_GRID,
    qsvt_max_degree_grid: list[Optional[int]] = QSVT_MAXDEGREE_GRID,
    extract_circuits: bool = True,
    discretisation_order: int = 2,
    backend_name: str = "aer_statevector",
    hardware_run: bool = False,
    backend_shots: Optional[int] = None,
) -> list[EqualAccuracyResult]:
    """
    Run the equal-accuracy protocol for all (case, N, solver) combinations.

    Parameters
    ----------
    cases : list[dict]
        Problem cases (same format as run_primary_1d).
    N_values : list[int]
        Problem sizes. Recommended: [4, 8] for equal-accuracy runs
        (the parameter sweep multiplies the solver call count by 5–10×).
    solvers : list[str]
        Quantum solvers to sweep: subset of ['hhl', 'vqls', 'qsvt'].
        Thomas is excluded (it achieves machine-precision residuals that
        are unreachable by quantum solvers).
    archive : SweepArchive, optional
        If provided, results are written after each solver sweep.
    r_target : float
        Target relative residual.
    band_factor : float
        Acceptance band multiplier.

    Returns
    -------
    list[EqualAccuracyResult]
        One result per (case, N, solver) combination.
    """
    build_fn = _build_problem_2nd if discretisation_order == 2 else _build_problem_4th
    all_ea: list[EqualAccuracyResult] = []

    for case in cases:
        case_id   = case["case_id"]
        source_fn = case["source_fn"]
        alpha     = case.get("alpha", 0.0)
        beta      = case.get("beta",  0.0)

        for N in N_values:
            log.info(
                "Equal-accuracy sweep: case=%s  N=%d  r_target=%.2e",
                case_id, N, r_target,
            )

            try:
                A, b, x, kappa, u_exact = build_fn(N, source_fn, alpha, beta)
            except Exception as exc:
                log.error("  Problem build failed: %s", exc)
                continue

            u_thomas, _, _ = _run_thomas(A, b)

            common_kwargs = dict(
                A=A, b=b, u_thomas=u_thomas, u_exact=u_exact,
                case_id=case_id, N=N, kappa=kappa,
                source_fn=source_fn, alpha_bc=alpha, beta_bc=beta,
                discretisation_order=discretisation_order,
                r_target=r_target, band_factor=band_factor,
                extract_circuits=extract_circuits,
                backend_name=backend_name,
                hardware_run=hardware_run,
                backend_shots=backend_shots,
            )

            sweep_dispatch = {
                "hhl":  lambda: sweep_hhl_equal_accuracy(
                    **common_kwargs, epsilon_grid=hhl_epsilon_grid
                ),
                "vqls": lambda: sweep_vqls_equal_accuracy(
                    **common_kwargs, n_layers_grid=vqls_n_layers_grid
                ),
                "qsvt": lambda: sweep_qsvt_equal_accuracy(
                    **common_kwargs, max_degree_grid=qsvt_max_degree_grid
                ),
            }

            for solver in solvers:
                if solver == "thomas":
                    continue
                if solver not in sweep_dispatch:
                    log.warning("  Unknown solver '%s'; skipping.", solver)
                    continue

                try:
                    ear = sweep_dispatch[solver]()
                    all_ea.append(ear)
                    log.info(
                        "  %s: best_r=%.4e  in_band=%s  calls=%d",
                        solver.upper(), ear.best_result.residual,
                        ear.in_band, ear.n_solver_calls,
                    )
                    if archive:
                        archive.write_equal_accuracy(all_ea)
                except Exception as exc:
                    log.error(
                        "  Equal-accuracy sweep failed for %s: %s", solver, exc
                    )

    return all_ea


# -- Sensitivity sweep ---------------------------------------------------------

def run_sensitivity_1d(
    cases: list[dict],
    N_values: list[int],
    solvers: list[str],
    archive: Optional[SweepArchive] = None,
    extract_circuits: bool = True,
    discretisation_order: int = 2,
    backend_name: str = "aer_statevector",
    hardware_run: bool = False,
    backend_shots: Optional[int] = None,
) -> dict[str, list[SensitivitySweepResult]]:
    """
    Run OAT sensitivity sweeps for all specified solvers.

    Sensitivity analysis is restricted to N ∈ {4, 8} in practice.
    This function does not enforce this restriction, but the caller
    should pass only small N values to keep runtime tractable.

    Parameters
    ----------
    cases : list[dict]
        Problem cases. Sensitivity sweeps use the first case only
        (typically the canonical fS homogeneous case).
    N_values : list[int]
        Problem sizes. Recommended: [4, 8].
    solvers : list[str]
        Quantum solvers to sweep.
    archive : SweepArchive, optional
        If provided, results are written after each solver sweep.

    Returns
    -------
    dict[str, list[SensitivitySweepResult]]
        Mapping solver -> list of sweep results (one per parameter).
    """
    build_fn = _build_problem_2nd if discretisation_order == 2 else _build_problem_4th

    # Use only the first case for sensitivity analysis
    case = cases[0]
    case_id   = case["case_id"]
    source_fn = case["source_fn"]
    alpha     = case.get("alpha", 0.0)
    beta      = case.get("beta",  0.0)

    all_sensitivity: dict[str, list[SensitivitySweepResult]] = {}

    for N in N_values:
        log.info("Sensitivity sweep: case=%s  N=%d", case_id, N)

        try:
            A, b, x, kappa, u_exact = build_fn(N, source_fn, alpha, beta)
        except Exception as exc:
            log.error("  Problem build failed: %s", exc)
            continue

        u_thomas, _, _ = _run_thomas(A, b)

        for solver in solvers:
            if solver == "thomas":
                continue
            log.info("  Solver: %s", solver.upper())

            try:
                sweeps = run_all_sensitivity_sweeps(
                    solver=solver,
                    A=A, b=b,
                    u_thomas=u_thomas, u_exact=u_exact,
                    case_id=case_id, N=N, kappa=kappa,
                    source_fn=source_fn, alpha_bc=alpha, beta_bc=beta,
                    discretisation_order=discretisation_order,
                    extract_circuits=extract_circuits,
                    backend_name=backend_name,
                    hardware_run=hardware_run,
                    backend_shots=backend_shots,
                )
                key = f"{solver}_N{N}"
                all_sensitivity[key] = sweeps
                if archive:
                    archive.write_sensitivity(f"{solver}_N{N}", sweeps)
            except Exception as exc:
                log.error("  Sensitivity sweep failed for %s: %s", solver, exc)

    return all_sensitivity