"""
Equal-accuracy benchmarking protocol for quantum linear system solvers.

Motivation
──────────
A naive comparison of HHL, VQLS, and QSVT at nominally equal precision
parameters (e.g. ε = 0.01 for all) is methodologically unsound because:

  1. The VQLS cost function C is not the residual r. The bound
     C ≥ r²/κ² (Bravo-Prieto et al. 2023) means a cost of 10⁻⁶ guarantees
     only r ≤ κ × 10⁻³, which for κ ≈ 9.5 (N=4) gives r ≤ 0.95%.

  2. The HHL QPE precision ε and Trotter step count n_T are coupled in the
     current implementation (n_T = ceil(1/ε)). Reducing ε improves QPE
     resolution but also increases n_T, which changes the Hamiltonian
     simulation error independently.

  3. The QSVT residual is not monotone in polynomial degree due to
     oscillatory Chebyshev approximation error.

  4. The Thomas algorithm achieves machine-precision residuals (~10⁻¹⁴),
     which are unachievable by any of the quantum solvers. Thomas is
     therefore used as the *reference solution*, not as a competitor in
     the equal-accuracy comparison.

Protocol
────────
For each quantum solver and each problem (N, case_id):

  1. Define a target residual r_target (e.g. 1e-3).
  2. Sweep the solver's primary precision parameter over a predefined grid.
  3. Run the solver at each grid point and record the achieved residual.
  4. Select the result whose residual is closest to r_target within the
     acceptance band [r_target / band_factor, r_target * band_factor].
  5. If no grid point achieves a residual within the band, record the
     closest result and flag it as 'out_of_band'.

The output is a set of BenchmarkResult objects, one per solver, all at
approximately equal residual, enabling a fair comparison of circuit depth,
qubit count, and wall time.

Parameter grids
───────────────
HHL:
  Primary parameter: epsilon ∈ {0.1, 0.05, 0.01, 0.005, 0.001}
  Note: trotter_steps = max(1, ceil(1/epsilon)) is coupled to epsilon.
  This coupling is documented in the result but not broken, since
  decoupling requires modifying the TridiagonalToeplitz constructor.

VQLS:
  Primary parameter: n_layers ∈ {1, 2, 3, 4, 5}
  Secondary sweep (if primary fails): n_restarts ∈ {1, 2, 3, 5}
  Note: COBYLA tolerance is fixed at 1e-8 throughout. The cost function
  value is NOT used as the convergence criterion for equal-accuracy
  purposes; only the measured residual determines acceptance.

QSVT:
  Primary parameter: max_degree ∈ {None, 5000, 2000, 1000, 500}
  None means uncapped (full-precision phases). The grid is ordered from
  highest to lowest accuracy so that the first result within the band
  is the most resource-efficient one that achieves the target.

References
──────────
  Bravo-Prieto et al. (2023) Quantum 7, 1188.  doi:10.22331/q-2023-11-22-1188
  Ghafourpour & Laizet (2025) Phys. Rev. Applied 24, 024032.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from benchmark.metrics import (
    BenchmarkResult,
    CircuitMetrics,
    compute_max_abs_err,
    compute_max_rel_err,
    compute_residual,
    extract_circuit_metrics,
)

log = logging.getLogger(__name__)

# ── Default parameter grids ───────────────────────────────────────────────────

HHL_EPSILON_GRID: list[float] = [0.1, 0.05, 0.01, 0.005, 0.001]

VQLS_NLAYERS_GRID: list[int] = [1, 2, 3, 4, 5]
VQLS_NRESTARTS_GRID: list[int] = [1, 2, 3, 5]

# Ordered highest-to-lowest accuracy so the first in-band result is the
# most resource-efficient one that meets the target.
QSVT_MAXDEGREE_GRID: list[Optional[int]] = [None, 5000, 2000, 1000, 500]

# Default equal-accuracy target and acceptance band
DEFAULT_R_TARGET: float = 1.0e-3
DEFAULT_BAND_FACTOR: float = 3.0   # accept r ∈ [r_target/3, r_target*3]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class EqualAccuracyResult:
    """
    Result of an equal-accuracy sweep for a single solver.

    Attributes
    ----------
    solver : str
        Algorithm name.
    r_target : float
        Target residual specified for this sweep.
    band_factor : float
        Acceptance band multiplier. Accepted if r ∈ [r_target/band_factor,
        r_target * band_factor].
    in_band : bool
        True if the best result falls within the acceptance band.
    best_result : BenchmarkResult
        The BenchmarkResult with residual closest to r_target.
    all_results : list[BenchmarkResult]
        All results from the parameter sweep, ordered by parameter value.
    n_solver_calls : int
        Total number of solver invocations in this sweep.
    total_sweep_time_s : float
        Total wall time [s] for the entire sweep.
    notes : str
        Human-readable notes, e.g. 'out_of_band: best residual = 0.0312'.
    """

    solver:            str
    r_target:          float
    band_factor:       float
    in_band:           bool
    best_result:       BenchmarkResult
    all_results:       list[BenchmarkResult]
    n_solver_calls:    int
    total_sweep_time_s: float
    notes:             str = ""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_base_result(
    case_id: str,
    solver: str,
    N: int,
    kappa: float,
    source_fn: str,
    alpha_bc: float,
    beta_bc: float,
    discretisation_order: int,
    u_solver: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    u_thomas: np.ndarray,
    u_exact: Optional[np.ndarray],
    wall_time_s: float,
    r_target: Optional[float],
    backend_name: str = "aer_statevector",
    hardware_run: bool = False,
    backend_shots: Optional[int] = None,
) -> BenchmarkResult:
    """
    Construct a BenchmarkResult from raw solver output.

    This function computes all accuracy metrics from the actual solution
    vector, never from solver parameters. It is the single point of truth
    for metric computation in this framework.

    Parameters
    ----------
    u_solver : np.ndarray, shape (N,)
        Solution vector returned by the quantum solver.
    u_thomas : np.ndarray, shape (N,)
        Thomas algorithm reference solution (always available).
    u_exact : np.ndarray or None, shape (N,)
        Analytical solution, if available.
    """
    residual = compute_residual(A, u_solver, b)

    # Accuracy vs exact (only where analytical solution exists)
    if u_exact is not None:
        mre_exact = compute_max_rel_err(u_solver, u_exact) * 100.0
        mae_exact = compute_max_abs_err(u_solver, u_exact)
        err_disc  = compute_max_rel_err(u_thomas, u_exact) * 100.0
        err_alg   = max(0.0, mre_exact - err_disc)
    else:
        mre_exact = None
        mae_exact = None
        err_disc  = None
        err_alg   = None

    # Accuracy vs Thomas (always available)
    mre_thomas = compute_max_rel_err(u_solver, u_thomas) * 100.0
    mae_thomas = compute_max_abs_err(u_solver, u_thomas)

    return BenchmarkResult(
        case_id=case_id,
        solver=solver,
        N=N,
        discretisation_order=discretisation_order,
        kappa=kappa,
        source_fn=source_fn,
        alpha_bc=alpha_bc,
        beta_bc=beta_bc,
        residual=residual,
        max_rel_err_vs_exact=mre_exact,
        max_abs_err_vs_exact=mae_exact,
        max_rel_err_vs_thomas=mre_thomas,
        max_abs_err_vs_thomas=mae_thomas,
        err_disc=err_disc,
        err_alg=err_alg,
        proportionality_residual=None,   # set by caller for HHL/QSVT
        wall_time_s=wall_time_s,
        phase_lookup_time_s=None,        # set by caller for QSVT
        circuit_metrics=None,            # set by caller
        hhl_epsilon=None,
        hhl_trotter_steps=None,
        vqls_n_layers=None,
        vqls_n_restarts=None,
        vqls_cost_final=None,
        vqls_n_evaluations=None,
        vqls_converged=None,
        qsvt_polynomial_degree=None,
        qsvt_max_degree_cap=None,
        qsvt_subnormalisation=None,
        qsvt_kappa_eff=None,
        qsvt_angle_method=None,
        qsvt_phase_from_cache=None,
        sensitivity_param=None,
        sensitivity_value=None,
        r_target=r_target,
        backend_name=backend_name,
        backend_shots=backend_shots,
        hardware_run=hardware_run,
    )


def _select_best(
    results: list[BenchmarkResult],
    r_target: float,
    band_factor: float,
) -> tuple[BenchmarkResult, bool]:
    """
    Select the result whose residual is closest to r_target.

    Returns the best result and a boolean indicating whether it falls
    within the acceptance band [r_target/band_factor, r_target*band_factor].
    """
    if not results:
        raise ValueError("Cannot select from an empty result list.")

    best = min(results, key=lambda r: abs(r.residual - r_target))
    r_lo = r_target / band_factor
    r_hi = r_target * band_factor
    in_band = r_lo <= best.residual <= r_hi
    return best, in_band


# ── HHL equal-accuracy sweep ──────────────────────────────────────────────────

def sweep_hhl_equal_accuracy(
    A: np.ndarray,
    b: np.ndarray,
    u_thomas: np.ndarray,
    u_exact: Optional[np.ndarray],
    case_id: str,
    N: int,
    kappa: float,
    source_fn: str,
    alpha_bc: float,
    beta_bc: float,
    discretisation_order: int,
    epsilon_grid: list[float] = HHL_EPSILON_GRID,
    r_target: float = DEFAULT_R_TARGET,
    band_factor: float = DEFAULT_BAND_FACTOR,
    extract_circuits: bool = True,
    backend_name: str = "aer_statevector",
    hardware_run: bool = False,
    backend_shots: Optional[int] = None,
) -> EqualAccuracyResult:
    """
    Equal-accuracy sweep for the HHL algorithm.

    Sweeps epsilon over epsilon_grid, runs the HHL solver at each value,
    and selects the result closest to r_target. The Trotter step count is
    coupled to epsilon as n_T = max(1, ceil(1/epsilon)) per the current
    TridiagonalToeplitz implementation.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        System matrix (TST or pentadiagonal).
    b : np.ndarray, shape (N,)
        Right-hand side vector.
    u_thomas : np.ndarray, shape (N,)
        Thomas algorithm reference solution.
    u_exact : np.ndarray or None, shape (N,)
        Analytical solution, if available.
    epsilon_grid : list[float]
        HHL QPE precision values to sweep.
    r_target : float
        Target relative residual.
    band_factor : float
        Acceptance band multiplier.
    extract_circuits : bool
        If True, extract circuit metrics (depth, qubit count) for each run.
        Adds transpilation overhead; set False for quick sweeps.

    Returns
    -------
    EqualAccuracyResult
        Sweep result with best BenchmarkResult and all intermediate results.
    """
    # `hhl_solve_system` takes epsilon positionally and returns the plain tuple
    # (u, raw_state, prop_const); there is no HHLConfig1D in this codebase. An
    # earlier draft of this module assumed a config object and an attribute-bearing
    # result, matching neither, so the HHL half of the protocol raised on import.
    from solvers.quantum.hhl_1d import hhl_solve_system

    results: list[BenchmarkResult] = []
    t_sweep_start = time.perf_counter()

    for eps in epsilon_grid:
        log.info(
            "  HHL equal-accuracy sweep: N=%d  epsilon=%.4f  r_target=%.2e",
            N, eps, r_target,
        )
        try:
            t0 = time.perf_counter()
            u_sol, raw_state, _prop_const = hhl_solve_system(A, b, eps)
            wall = time.perf_counter() - t0

            u_sol = np.array(u_sol)
            n_trotter = int(np.ceil(1.0 / eps))

            # Proportionality recovery residual (HHL-specific)
            raw_state = np.asarray(raw_state, dtype=float)
            Ax_raw = A @ raw_state
            c_val  = float(
                np.dot(b, Ax_raw) / (np.dot(Ax_raw, Ax_raw) + 1.0e-300)
            )
            prop_residual = compute_residual(A, c_val * raw_state, b)

            rec = _build_base_result(
                case_id=case_id, solver="hhl", N=N, kappa=kappa,
                source_fn=source_fn, alpha_bc=alpha_bc, beta_bc=beta_bc,
                discretisation_order=discretisation_order,
                u_solver=u_sol, A=A, b=b,
                u_thomas=u_thomas, u_exact=u_exact,
                wall_time_s=wall, r_target=r_target,
                backend_name=backend_name, hardware_run=hardware_run,
                backend_shots=backend_shots,
            )
            rec.hhl_epsilon = eps
            rec.hhl_trotter_steps = n_trotter
            rec.proportionality_residual = prop_residual
            rec.sensitivity_param = "epsilon"
            rec.sensitivity_value = eps

            # `hhl_solve_system` returns only (u, raw_state, prop_const), so there
            # is no circuit object to measure here; the other two solvers return a
            # result carrying one. Circuit metrics for HHL come from the primary
            # sweep, which records them per row.
            results.append(rec)
            log.info(
                "    residual=%.4e  max_rel_err_vs_exact=%s%%  time=%.2fs",
                rec.residual,
                f"{rec.max_rel_err_vs_exact:.3f}" if rec.max_rel_err_vs_exact else "N/A",
                wall,
            )

        except Exception as exc:
            log.warning("  HHL failed at epsilon=%.4f: %s", eps, exc)

    total_time = time.perf_counter() - t_sweep_start

    if not results:
        raise RuntimeError(
            f"HHL equal-accuracy sweep produced no valid results for "
            f"case_id={case_id}, N={N}."
        )

    best, in_band = _select_best(results, r_target, band_factor)
    notes = "" if in_band else (
        f"out_of_band: best residual = {best.residual:.4e}, "
        f"target = {r_target:.2e}, "
        f"band = [{r_target/band_factor:.2e}, {r_target*band_factor:.2e}]"
    )

    return EqualAccuracyResult(
        solver="hhl",
        r_target=r_target,
        band_factor=band_factor,
        in_band=in_band,
        best_result=best,
        all_results=results,
        n_solver_calls=len(results),
        total_sweep_time_s=total_time,
        notes=notes,
    )


# ── VQLS equal-accuracy sweep ─────────────────────────────────────────────────

def sweep_vqls_equal_accuracy(
    A: np.ndarray,
    b: np.ndarray,
    u_thomas: np.ndarray,
    u_exact: Optional[np.ndarray],
    case_id: str,
    N: int,
    kappa: float,
    source_fn: str,
    alpha_bc: float,
    beta_bc: float,
    discretisation_order: int,
    n_layers_grid: list[int] = VQLS_NLAYERS_GRID,
    n_restarts_grid: list[int] = VQLS_NRESTARTS_GRID,
    r_target: float = DEFAULT_R_TARGET,
    band_factor: float = DEFAULT_BAND_FACTOR,
    extract_circuits: bool = True,
    backend_name: str = "aer_statevector",
    hardware_run: bool = False,
    backend_shots: Optional[int] = None,
) -> EqualAccuracyResult:
    """
    Equal-accuracy sweep for the VQLS algorithm.

    Primary sweep: n_layers ∈ n_layers_grid (n_restarts fixed at 3).
    Secondary sweep: if no result falls within the band after the primary
    sweep, n_restarts is also varied.

    Critical note on VQLS convergence
    ───────────────────────────────────
    The COBYLA optimiser tolerance is fixed at 1e-8 throughout. The cost
    function value C is recorded but is NOT used as the acceptance criterion.
    Only the measured residual r = ‖Aû - b‖₂/‖b‖₂ determines acceptance.
    This is essential because C ≥ r²/κ² (Bravo-Prieto et al. 2023), so a
    small C does not guarantee a small r without knowledge of κ.

    Parameters
    ----------
    n_layers_grid : list[int]
        Ansatz layer counts to sweep (primary parameter).
    n_restarts_grid : list[int]
        COBYLA restart counts to sweep (secondary parameter).
    """
    from solvers.quantum.vqls_1d import vqls_solve_system, VQLSConfig1D

    results: list[BenchmarkResult] = []
    t_sweep_start = time.perf_counter()

    # Primary sweep: vary n_layers, fix n_restarts=3
    _fixed_restarts = 3
    for n_lay in n_layers_grid:
        log.info(
            "  VQLS equal-accuracy sweep: N=%d  n_layers=%d  "
            "n_restarts=%d  r_target=%.2e",
            N, n_lay, _fixed_restarts, r_target,
        )
        cfg = VQLSConfig1D(n_layers=n_lay, n_restarts=_fixed_restarts)

        try:
            t0 = time.perf_counter()
            solver_result = vqls_solve_system(A, b, config=cfg)
            wall = time.perf_counter() - t0

            u_sol = np.array(solver_result.u)

            rec = _build_base_result(
                case_id=case_id, solver="vqls", N=N, kappa=kappa,
                source_fn=source_fn, alpha_bc=alpha_bc, beta_bc=beta_bc,
                discretisation_order=discretisation_order,
                u_solver=u_sol, A=A, b=b,
                u_thomas=u_thomas, u_exact=u_exact,
                wall_time_s=wall, r_target=r_target,
                backend_name=backend_name, hardware_run=hardware_run,
                backend_shots=backend_shots,
            )
            rec.vqls_n_layers = n_lay
            rec.vqls_n_restarts = _fixed_restarts
            rec.vqls_cost_final = float(solver_result.final_cost)
            rec.vqls_n_evaluations = getattr(solver_result, "n_evaluations", None)
            rec.vqls_converged = getattr(solver_result, "converged", None)
            rec.sensitivity_param = "n_layers"
            rec.sensitivity_value = float(n_lay)

            if extract_circuits and hasattr(solver_result, "circuit"):
                try:
                    rec.circuit_metrics = extract_circuit_metrics(
                        solver_result.circuit, optimisation_level=1
                    )
                except Exception as e:
                    log.warning("  Circuit metric extraction failed: %s", e)

            results.append(rec)
            log.info(
                "    residual=%.4e  cost=%.4e  converged=%s  time=%.2fs",
                rec.residual,
                rec.vqls_cost_final,
                rec.vqls_converged,
                wall,
            )

        except Exception as exc:
            log.warning("  VQLS failed at n_layers=%d: %s", n_lay, exc)

    # Check if primary sweep achieved the target
    if results:
        _, in_band_primary = _select_best(results, r_target, band_factor)
    else:
        in_band_primary = False

    # Secondary sweep: vary n_restarts if primary sweep did not achieve target
    if not in_band_primary:
        log.info(
            "  VQLS primary sweep did not achieve r_target=%.2e; "
            "initiating secondary sweep over n_restarts.",
            r_target,
        )
        _fixed_layers = n_layers_grid[-1]   # use deepest ansatz from primary
        for n_rest in n_restarts_grid:
            if n_rest == _fixed_restarts:
                continue   # already run in primary sweep
            log.info(
                "  VQLS secondary sweep: N=%d  n_layers=%d  n_restarts=%d",
                N, _fixed_layers, n_rest,
            )
            cfg = VQLSConfig1D(n_layers=_fixed_layers, n_restarts=n_rest)
            try:
                t0 = time.perf_counter()
                solver_result = vqls_solve_system(A, b, config=cfg)
                wall = time.perf_counter() - t0

                u_sol = np.array(solver_result.u)
                rec = _build_base_result(
                    case_id=case_id, solver="vqls", N=N, kappa=kappa,
                    source_fn=source_fn, alpha_bc=alpha_bc, beta_bc=beta_bc,
                    discretisation_order=discretisation_order,
                    u_solver=u_sol, A=A, b=b,
                    u_thomas=u_thomas, u_exact=u_exact,
                    wall_time_s=wall, r_target=r_target,
                    backend_name=backend_name, hardware_run=hardware_run,
                    backend_shots=backend_shots,
                )
                rec.vqls_n_layers = _fixed_layers
                rec.vqls_n_restarts = n_rest
                rec.vqls_cost_final = float(solver_result.final_cost)
                rec.vqls_n_evaluations = getattr(solver_result, "n_evaluations", None)
                rec.vqls_converged = getattr(solver_result, "converged", None)
                rec.sensitivity_param = "n_restarts"
                rec.sensitivity_value = float(n_rest)
                results.append(rec)

            except Exception as exc:
                log.warning("  VQLS failed at n_restarts=%d: %s", n_rest, exc)

    total_time = time.perf_counter() - t_sweep_start

    if not results:
        raise RuntimeError(
            f"VQLS equal-accuracy sweep produced no valid results for "
            f"case_id={case_id}, N={N}."
        )

    best, in_band = _select_best(results, r_target, band_factor)
    notes = "" if in_band else (
        f"out_of_band: best residual = {best.residual:.4e}, "
        f"target = {r_target:.2e}"
    )

    return EqualAccuracyResult(
        solver="vqls",
        r_target=r_target,
        band_factor=band_factor,
        in_band=in_band,
        best_result=best,
        all_results=results,
        n_solver_calls=len(results),
        total_sweep_time_s=total_time,
        notes=notes,
    )


# ── QSVT equal-accuracy sweep ─────────────────────────────────────────────────

def sweep_qsvt_equal_accuracy(
    A: np.ndarray,
    b: np.ndarray,
    u_thomas: np.ndarray,
    u_exact: Optional[np.ndarray],
    case_id: str,
    N: int,
    kappa: float,
    source_fn: str,
    alpha_bc: float,
    beta_bc: float,
    discretisation_order: int,
    max_degree_grid: list[Optional[int]] = QSVT_MAXDEGREE_GRID,
    r_target: float = DEFAULT_R_TARGET,
    band_factor: float = DEFAULT_BAND_FACTOR,
    extract_circuits: bool = True,
    backend_name: str = "aer_statevector",
    hardware_run: bool = False,
    backend_shots: Optional[int] = None,
) -> EqualAccuracyResult:
    """
    Equal-accuracy sweep for the QSVT algorithm.

    Sweeps max_degree from highest to lowest accuracy. The grid is ordered
    so that the first in-band result is the most resource-efficient one.

    Critical note on QSVT residual monotonicity
    ─────────────────────────────────────────────
    The QSVT residual is NOT guaranteed to decrease monotonically with
    increasing polynomial degree due to oscillatory Chebyshev approximation
    error. The sweep therefore evaluates all grid points and selects the
    result closest to r_target, rather than stopping at the first in-band
    result.

    Parameters
    ----------
    max_degree_grid : list[int or None]
        Polynomial degree caps to sweep. None means uncapped (full precision).
        Ordered highest-to-lowest accuracy (None first).
    """
    from solvers.quantum.qsvt_1d import qsvt_solve_system, QSVTConfig1D

    results: list[BenchmarkResult] = []
    t_sweep_start = time.perf_counter()

    for max_deg in max_degree_grid:
        deg_label = "uncapped" if max_deg is None else str(max_deg)
        log.info(
            "  QSVT equal-accuracy sweep: N=%d  max_degree=%s  r_target=%.2e",
            N, deg_label, r_target,
        )
        cfg = QSVTConfig1D(epsilon=0.01, max_degree=max_deg, angle_method="auto")

        try:
            t_phase_start = time.perf_counter()
            solver_result = qsvt_solve_system(A, b, config=cfg)
            wall = time.perf_counter() - t_phase_start

            u_sol = np.array(solver_result.u)

            # Proportionality recovery residual (QSVT-specific)
            prop_residual = None
            if hasattr(solver_result, "raw_state") and solver_result.raw_state is not None:
                Ax_raw = A @ solver_result.raw_state
                c_val  = float(
                    np.dot(b, Ax_raw) / (np.dot(Ax_raw, Ax_raw) + 1.0e-300)
                )
                prop_residual = compute_residual(
                    A, c_val * solver_result.raw_state, b
                )

            # Phase lookup time (subset of wall time)
            phase_time = getattr(solver_result, "phase_lookup_time_s", None)

            alpha_sub = float(np.linalg.norm(A, ord=2))   # subnormalisation
            kappa_eff = kappa * alpha_sub

            rec = _build_base_result(
                case_id=case_id, solver="qsvt", N=N, kappa=kappa,
                source_fn=source_fn, alpha_bc=alpha_bc, beta_bc=beta_bc,
                discretisation_order=discretisation_order,
                u_solver=u_sol, A=A, b=b,
                u_thomas=u_thomas, u_exact=u_exact,
                wall_time_s=wall, r_target=r_target,
                backend_name=backend_name, hardware_run=hardware_run,
                backend_shots=backend_shots,
            )
            rec.proportionality_residual = prop_residual
            rec.phase_lookup_time_s = phase_time
            rec.qsvt_polynomial_degree = getattr(solver_result, "degree", None)
            rec.qsvt_max_degree_cap = max_deg
            rec.qsvt_subnormalisation = alpha_sub
            rec.qsvt_kappa_eff = kappa_eff
            rec.qsvt_angle_method = cfg.angle_method
            rec.qsvt_phase_from_cache = getattr(
                solver_result, "phase_from_cache", None
            )
            rec.sensitivity_param = "max_degree"
            rec.sensitivity_value = float(max_deg) if max_deg is not None else -1.0

            if extract_circuits and hasattr(solver_result, "circuit"):
                try:
                    rec.circuit_metrics = extract_circuit_metrics(
                        solver_result.circuit, optimisation_level=1
                    )
                except Exception as e:
                    log.warning("  Circuit metric extraction failed: %s", e)

            results.append(rec)
            log.info(
                "    residual=%.4e  degree=%s  kappa_eff=%.2f  time=%.2fs",
                rec.residual,
                rec.qsvt_polynomial_degree,
                kappa_eff,
                wall,
            )

        except Exception as exc:
            log.warning("  QSVT failed at max_degree=%s: %s", deg_label, exc)

    total_time = time.perf_counter() - t_sweep_start

    if not results:
        raise RuntimeError(
            f"QSVT equal-accuracy sweep produced no valid results for "
            f"case_id={case_id}, N={N}."
        )

    best, in_band = _select_best(results, r_target, band_factor)
    notes = "" if in_band else (
        f"out_of_band: best residual = {best.residual:.4e}, "
        f"target = {r_target:.2e}"
    )

    return EqualAccuracyResult(
        solver="qsvt",
        r_target=r_target,
        band_factor=band_factor,
        in_band=in_band,
        best_result=best,
        all_results=results,
        n_solver_calls=len(results),
        total_sweep_time_s=total_time,
        notes=notes,
    )