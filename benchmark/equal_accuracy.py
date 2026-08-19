"""
Equal-accuracy benchmarking protocol for quantum linear system solvers.

Motivation
----------
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
--------
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
---------------
HHL:
  Primary parameter: epsilon ∈ {0.1, 0.05, 0.01, 0.005, 0.001}
  epsilon is the algorithm's overall precision. HHL apportions eps/6 of it to
  the Hamiltonian simulation and derives the Trotter step count from that and
  the evolution time it fixes from the spectral bounds. The count is therefore
  coupled to epsilon but is *not* ceil(1/epsilon): at eps = 0.01 the derived
  count is 7, not 100. The realised value is read back from the solve rather
  than inferred, and the two can be decoupled where wanted — see the
  `trotter_steps` argument of `solvers/quantum/hhl_1d.hhl_solve_system`.

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
----------
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
    compute_rel_l2_err,
    compute_residual,
    extract_circuit_metrics,
)

log = logging.getLogger(__name__)

# -- Default parameter grids ---------------------------------------------------

HHL_EPSILON_GRID: list[float] = [0.1, 0.05, 0.01, 0.005, 0.001]

VQLS_NLAYERS_GRID: list[int] = [1, 2, 3, 4, 5]
VQLS_NRESTARTS_GRID: list[int] = [1, 2, 3, 5]

# Ordered highest-to-lowest accuracy so the first in-band result is the
# most resource-efficient one that meets the target.
#
# The grid extends down to degree 20 because the protocol is meaningless
# otherwise. QSVT's accuracy is governed by the ratio d/kappa: the residual runs
# from ~0.9 at d/kappa ~ 0.6, through 3e-2 at ~3 and 5e-5 at ~11, to 1e-11 by
# ~26. At N=8 (kappa = 32.2) the shallowest previous entry, degree 500, already
# gives d/kappa = 15.5 and a residual near 5e-7 — three orders below the 1e-3
# target, so every configuration on the grid overshot and the sweep compared
# cost at *unequal* accuracy while reporting it as equal. Degrees 20 to 200 put
# d/kappa in [0.6, 6.2] at that resolution, which is where the target lives.
#
# The additional points are close to free: QSVT's cost is linear in the degree,
# so the four new entries together cost less than the single degree-500 entry
# they sit below.
QSVT_MAXDEGREE_GRID: list[Optional[int]] = [
    None, 5000, 2000, 1000, 500, 200, 100, 50, 20,
]

# Default equal-accuracy target and acceptance band
DEFAULT_R_TARGET: float = 1.0e-3
DEFAULT_BAND_FACTOR: float = 3.0   # accept r ∈ [r_target/3, r_target*3]


# -- Result dataclass ----------------------------------------------------------

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


# -- Internal helpers ----------------------------------------------------------

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
    u_thomas: np.ndarray,
    u_exact: Optional[np.ndarray],
    wall_time_s: float,
    r_target: Optional[float],
    A: Optional[np.ndarray] = None,
    b: Optional[np.ndarray] = None,
    residual: Optional[float] = None,
    backend_name: str = "aer_statevector",
    hardware_run: bool = False,
    backend_shots: Optional[int] = None,
) -> BenchmarkResult:
    """
    Construct a BenchmarkResult from raw solver output.

    This function computes all accuracy metrics from the actual solution
    vector, never from solver parameters. It is the single point of truth
    for metric computation in this framework.

    The residual may be supplied either as an assembled system, via ``A`` and
    ``b``, or directly as a precomputed scalar. Both routes exist because the
    1-D solvers own a dense operator while the 2-D and 3-D ones never assemble
    one: `solvers/outer` applies the stencil matrix-free and reports the outer
    residual on `OuterResult`. Requiring ``(A, b)`` there would mean fabricating
    an N²×N² dense system — 16.8 M entries at N=64 — solely to recompute a number
    already in hand. The invariant that matters is unaffected: every metric is
    still measured from the solution, never inferred from solver parameters.

    Parameters
    ----------
    u_solver : np.ndarray
        Solution returned by the quantum solver; length-N in 1-D, an (N, N) or
        (N, N, N) field in 2-D/3-D.
    u_thomas : np.ndarray
        Classical reference on the same mesh, same shape as `u_solver`.
    u_exact : np.ndarray or None
        Analytical solution where the case has one, same shape.
    A : np.ndarray, optional
        NxN system matrix. Required only when `residual` is not given.
    b : np.ndarray, optional
        Length-N right-hand side. Required only when `residual` is not given.
    residual : float, optional
        Precomputed relative residual ‖Au − b‖/‖b‖. Takes precedence over
        ``(A, b)``.

    Raises
    ------
    ValueError
        If neither `residual` nor both of `A` and `b` are supplied.
    """
    if residual is None:
        if A is None or b is None:
            raise ValueError(
                "_build_base_result needs either a precomputed `residual` or "
                "both `A` and `b`; received neither."
            )
        residual = compute_residual(A, u_solver, b)

    # Accuracy vs exact (only where analytical solution exists)
    if u_exact is not None:
        mre_exact = compute_max_rel_err(u_solver, u_exact) * 100.0
        mae_exact = compute_max_abs_err(u_solver, u_exact)
        err_disc  = compute_max_rel_err(u_thomas, u_exact) * 100.0
    else:
        mre_exact = None
        mae_exact = None
        err_disc  = None

    # Accuracy vs Thomas (always available)
    mre_thomas = compute_max_rel_err(u_solver, u_thomas) * 100.0
    mae_thomas = compute_max_abs_err(u_solver, u_thomas)

    # The algorithmic error is measured, not inferred by subtraction. The solver
    # is asked for the solution of A u = b, whose exact answer on this mesh is
    # the Thomas field; the deviation from it is the solver's own error, and
    # err_disc = ‖u_thomas − u_exact‖ is the discretisation error beside it. The
    # two bound the total error against u_exact above, by the triangle
    # inequality.
    #
    # The former definition, max(0, mre_exact − err_disc), was a difference of
    # two nearly equal max-relative errors and lost all precision exactly where
    # the solver was good: the clamp returned 0.0 whenever the solve beat the
    # truncation estimate, deleting the most accurate points from every curve.
    # Measured on the 1-D N=8 Trotter sweep it gave 32.3, **0.0**, 0.203, 0.0628,
    # 0.0174, 0.00446 across n_T = 1…32, against a clean 1/n_T² decay; on
    # `fH_hom` three of six points were annihilated.
    #
    # The L2 ratio is used rather than the pointwise `mre_thomas`, which is
    # unbounded near a node of the reference field and reports 1.2e8 % on the
    # 3-D HET manufactured case for a solve accurate to a fraction of a per cent.
    err_alg = compute_rel_l2_err(u_solver, u_thomas) * 100.0

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


# -- HHL equal-accuracy sweep --------------------------------------------------

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
    and selects the result closest to r_target. The Trotter step count follows
    from epsilon through the library's own derivation and is recorded as
    realised, not inferred.

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
    # hhl_solve_system accepts epsilon positionally and returns the tuple
    # (u, raw_state, prop_const). A dedicated HHLConfig1D class is absent from this
    # architecture, necessitating direct parameter passing to avoid interface mismatches.
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
            # `diagnostics` carries back the step count actually simulated;
            # ceil(1/eps) is not it. See `solvers/quantum/hhl_1d.py`.
            diag: dict = {}
            u_sol, raw_state, _prop_const = hhl_solve_system(
                A, b, eps, diagnostics=diag)
            wall = time.perf_counter() - t0

            u_sol = np.array(u_sol)
            n_trotter = diag.get("trotter_steps")

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


# -- VQLS equal-accuracy sweep -------------------------------------------------

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
    -----------------------------------
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


# -- QSVT equal-accuracy sweep -------------------------------------------------

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
    ---------------------------------------------
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

# -- Equal accuracy in 2-D and 3-D ---------------------------------------------

# The inner-solver option each solver's precision knob is spelled as, in the
# registry `solvers/outer/inner.py` validates against. The registry rejects an
# unknown key rather than ignoring it, so a name that drifts from this mapping
# fails loudly at the first solve rather than silently sweeping nothing.
OUTER_PRECISION_KNOB: dict[str, str] = {
    "hhl":  "epsilon",
    "vqls": "n_layers",
    "qsvt": "max_degree",
}

# Grids for the knobs above, ordered highest-accuracy first so that the first
# in-band result is also the most economical one meeting the target. These mirror
# the 1-D grids, except that VQLS is swept on n_layers alone: in 2-D and 3-D each
# outer iteration performs N (or N²) inner solves, so a two-dimensional
# layer/restart grid would multiply an already expensive sweep by four.
OUTER_EQUAL_ACCURACY_GRIDS: dict[str, list] = {
    "hhl":  [0.1, 0.05, 0.01, 0.005],
    "vqls": [1, 2, 3, 4, 5],
    # Extended below 50 for the same reason as the 1-D grid, and further,
    # because the strip operator is far better conditioned: kappa_row is bounded
    # by 3 in 2-D and by 2 in 3-D, so degree 50 already gives d/kappa ~ 17 and
    # every previous entry sat at machine precision. Degrees 5 to 20 put the
    # ratio in [1.7, 6.7], which is the range in which the target is reachable.
    "qsvt": [None, 500, 200, 100, 50, 20, 10, 5],
}


def sweep_outer_equal_accuracy(
    problem,
    u_thomas: np.ndarray,
    u_exact: Optional[np.ndarray],
    case_id: str,
    N: int,
    kappa: float,
    source_fn: str,
    discretisation_order: int,
    solver: str,
    scheme: str = "fmg",
    grid: Optional[list] = None,
    r_target: float = DEFAULT_R_TARGET,
    band_factor: float = DEFAULT_BAND_FACTOR,
    scheme_options: Optional[dict] = None,
    backend_name: str = "aer_statevector",
) -> EqualAccuracyResult:
    """
    Equal-accuracy sweep for one solver on a 2-D or 3-D line problem.

    The 1-D entry points above each drive a solver directly on an assembled
    ``(A, b)``. In 2-D and 3-D there is no assembled operator: the domain is
    decomposed into strips and `solvers.outer.solve` runs an outer iteration
    whose inner solves are 1-D. The precision knob is therefore not a solver
    argument but an `inner_options` entry, and the residual to compare against
    `r_target` is the OUTER residual after convergence — the quantity that
    actually characterises the coupled scheme, and the one `OuterResult` reports
    without any dense operator being formed.

    One function serves all three solvers here, where 1-D needs three, because
    every solver reaches the outer layer through the same `solve()` signature and
    differs only in which `inner_options` key carries its knob.

    Parameters
    ----------
    problem : LineProblem2D
        Assembled problem satisfying the 2-D/3-D protocol.
    u_thomas : np.ndarray
        Classical reference field on the same mesh, from an identical outer solve
        with ``inner="thomas"``, so that the comparison isolates the inner solver
        rather than the scheme.
    u_exact : np.ndarray or None
        Analytical field where the case has one.
    case_id, N, kappa, source_fn, discretisation_order
        Recorded on every row, as in the 1-D sweeps. `kappa` is the strip
        condition number κ_row, not that of any global operator.
    solver : {'hhl', 'vqls', 'qsvt'}
        Inner solver to sweep.
    scheme : str
        Outer scheme, passed through to `solve`.
    grid : list, optional
        Values for this solver's knob. Defaults to `OUTER_EQUAL_ACCURACY_GRIDS`.
    r_target : float
        Common outer-residual target.
    band_factor : float
        Acceptance band, r ∈ [r_target/band_factor, r_target·band_factor].
    scheme_options : dict, optional
        Forwarded to `solve` as keyword scheme options, e.g. ``max_wall_s``.

    Returns
    -------
    EqualAccuracyResult
        Best in-band result and every intermediate one.

    Raises
    ------
    ValueError
        If `solver` is not one of the three swept here.
    RuntimeError
        If no grid point produced a usable result.
    """
    from solvers.outer import solve

    if solver not in OUTER_PRECISION_KNOB:
        raise ValueError(
            f"solver must be one of {sorted(OUTER_PRECISION_KNOB)}, "
            f"received {solver!r}.")

    knob = OUTER_PRECISION_KNOB[solver]
    values = grid if grid is not None else OUTER_EQUAL_ACCURACY_GRIDS[solver]
    scheme_options = dict(scheme_options or {})

    # The outer tolerance IS the equal-accuracy target here, and setting it is what
    # makes the protocol mean anything in more than one dimension.
    #
    # The outer iteration drives the residual to whatever tolerance it is given,
    # largely independently of the inner solver's precision, because inner error
    # is re-annihilated on the next sweep. Left at its default, every grid point
    # converges to the same residual and the comparison is vacuous: measured at
    # N=8 on the sinusoid, QSVT returned 6.2852e-09 at max_degree 500, 200, 100
    # AND 50, while the wall time fell from 154.6 s to 15.7 s.
    #
    # Fixing the outer tolerance at r_target formalises the comparative analysis:
    # ensuring uniform terminal accuracy permits direct evaluation of computational
    # cost. It furthermore identifies the threshold at which inner-solver imprecision
    # stalls outer convergence entirely, manifesting as a non-converged state
    # rather than an inflated residual.
    scheme_options.setdefault("tol", r_target)

    results: list[BenchmarkResult] = []
    t_sweep_start = time.perf_counter()

    for val in values:
        log.info("  %s equal-accuracy (outer): N=%d  %s=%s  r_target=%.2e",
                 solver.upper(), N, knob, val, r_target)
        # A None entry means "no cap": the option is omitted rather than passed
        # as None, since the registry validates values as well as keys.
        inner_options = {} if val is None else {knob: val}

        try:
            t0 = time.perf_counter()
            res = solve(problem, inner=solver, scheme=scheme,
                        inner_options=inner_options, **scheme_options)
            wall = time.perf_counter() - t0

            rec = _build_base_result(
                case_id=case_id, solver=solver, N=N, kappa=kappa,
                source_fn=source_fn, alpha_bc=0.0, beta_bc=0.0,
                discretisation_order=discretisation_order,
                u_solver=np.asarray(res.u), u_thomas=u_thomas, u_exact=u_exact,
                residual=res.residual, wall_time_s=wall, r_target=r_target,
                backend_name=backend_name,
            )
            rec.sensitivity_param = knob
            rec.sensitivity_value = None if val is None else float(val)
            if solver == "hhl":
                rec.hhl_epsilon = float(val)
                # Left unset rather than filled with ceil(1/eps), which is not
                # the count the library simulates. This path drives the solver
                # through the outer iteration, which returns no per-strip
                # diagnostics, so the realised count is genuinely unavailable
                # here; recording a wrong number is worse than recording none.
                rec.hhl_trotter_steps = None
            elif solver == "vqls":
                rec.vqls_n_layers = int(val)
                rec.vqls_cost_final = res.diagnostics.get("final_cost_mean")
            else:
                rec.qsvt_max_degree_cap = val
                degree = res.diagnostics.get("polynomial_degree_mean")
                rec.qsvt_polynomial_degree = (None if degree is None
                                              else int(degree))

            results.append(rec)
            log.info("    outer residual=%.4e  n_outer=%d  stop=%s  time=%.1fs",
                     res.residual, res.n_outer, res.stop_reason, wall)

        except Exception as exc:
            log.warning("  %s failed at %s=%s: %s", solver.upper(), knob, val, exc)

    if not results:
        raise RuntimeError(
            f"{solver.upper()} equal-accuracy sweep produced no valid results "
            f"for case_id={case_id}, N={N}.")

    total_time = time.perf_counter() - t_sweep_start

    # Selection differs from the 1-D protocol, and must. There, each grid point
    # lands at a different residual and the best is the one nearest the target.
    # Here the outer tolerance pins every converged point to the same residual, so
    # "nearest the target" would pick arbitrarily among ties; the meaningful
    # answer is the CHEAPEST configuration that still reaches the target. Points
    # that failed to reach it are excluded rather than ranked, their inner solver
    # being too imprecise for the outer iteration to close -- which is itself the
    # result being sought, and is recorded on the row.
    reached = [r for r in results
               if r.residual is not None and r.residual <= r_target * band_factor]
    in_band = bool(reached)
    best = (min(reached, key=lambda r: r.wall_time_s) if reached
            else min(results, key=lambda r: (r.residual is None, r.residual or 0.0)))
    notes = "" if in_band else (
        f"no configuration reached r_target: best residual = "
        f"{best.residual:.4e}, target = {r_target:.2e}")
    if in_band and len(reached) < len(results):
        notes = (f"{len(results) - len(reached)} of {len(results)} "
                 f"configuration(s) did not reach r_target")

    return EqualAccuracyResult(
        solver=solver,
        r_target=r_target,
        band_factor=band_factor,
        in_band=in_band,
        best_result=best,
        all_results=results,
        n_solver_calls=len(results),
        total_sweep_time_s=total_time,
        notes=notes,
    )
