"""
One-at-a-time (OAT) sensitivity analysis for quantum linear system solvers.

Design rationale
────────────────
A sensitivity study varies one parameter at a time while holding all others
at a fixed baseline. This design is chosen over a full factorial sweep for
two reasons:

  1. Computational tractability: a full factorial sweep over K parameters
     each with M values requires M^K solver calls. For K=3, M=5, this is
     125 calls per (N, case) combination. OAT requires only K×M = 15 calls.

  2. Interpretability: OAT results can be presented as K separate curves,
     each showing the effect of a single parameter. Full factorial results
     require multi-dimensional visualisation that is difficult to interpret
     in a thesis context.

The limitation of OAT is that interaction effects between parameters are
not captured. For the algorithms studied here, the primary parameters are
largely independent (VQLS n_layers and n_restarts affect the optimisation
landscape independently; HHL epsilon and Trotter steps are coupled by
design but treated as a single parameter), so OAT is appropriate.

Scope
─────
Sensitivity analysis is restricted to N ∈ {4, 8} to keep total runtime
tractable. At N=4, a single QSVT call takes ~0.5s; at N=8, ~220s. A full
OAT sweep at N=8 with 5 QSVT parameter values therefore takes ~18 minutes,
which is acceptable for an overnight HPC run but not for an interactive
session.

Parameter definitions
─────────────────────
HHL:
  epsilon: QPE precision. Range [0.1, 0.001]. Coupled to Trotter steps.
  Note: the coupling n_T = ceil(1/epsilon) means this is effectively a
  joint (epsilon, n_T) sweep. This is documented in the output.

VQLS:
  n_layers: Ansatz depth. Range [1, 5]. Primary cost driver.
  n_restarts: COBYLA restart count. Range [1, 5]. Affects escape from
    local minima but not the circuit depth.
  cobyla_tol: COBYLA convergence tolerance. Range [1e-4, 1e-10].
    Note: reducing cobyla_tol does NOT guarantee a smaller residual;
    it only changes when the optimiser stops. The residual is always
    measured from the returned solution.

QSVT:
  max_degree: Polynomial degree cap. Range [500, uncapped].
    Note: residual is not monotone in degree (see equal_accuracy.py).
  epsilon: Approximation precision for phase-angle computation.
    Range [0.1, 0.001]. Affects the polynomial degree required.

References
──────────
  Saltelli et al. (2008) Global Sensitivity Analysis: The Primer. Wiley.
  Bravo-Prieto et al. (2023) Quantum 7, 1188.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from benchmark.metrics import (
    BenchmarkResult,
    compute_max_abs_err,
    compute_max_rel_err,
    compute_residual,
    extract_circuit_metrics,
)
from benchmark.equal_accuracy import _build_base_result

log = logging.getLogger(__name__)


# ── Baseline parameter definitions ────────────────────────────────────────────

HHL_BASELINE: dict = {
    "epsilon": 0.01,
}

VQLS_BASELINE: dict = {
    "n_layers":   2,
    "n_restarts": 3,
    "cobyla_tol": 1.0e-8,
}

QSVT_BASELINE: dict = {
    "max_degree": None,    # uncapped
    "epsilon":    0.01,
    "angle_method": "auto",
}

# ── Sensitivity parameter grids ───────────────────────────────────────────────

HHL_SENSITIVITY_GRIDS: dict[str, list] = {
    "epsilon": [0.1, 0.05, 0.01, 0.005, 0.001],
}

VQLS_SENSITIVITY_GRIDS: dict[str, list] = {
    "n_layers":   [1, 2, 3, 4, 5],
    "n_restarts": [1, 2, 3, 5, 8],
    "cobyla_tol": [1.0e-4, 1.0e-6, 1.0e-8, 1.0e-10],
}

QSVT_SENSITIVITY_GRIDS: dict[str, list] = {
    "max_degree": [500, 1000, 2000, 5000, None],
    "epsilon":    [0.1, 0.05, 0.01, 0.005, 0.001],
}


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SensitivitySweepResult:
    """
    Result of a one-at-a-time sensitivity sweep for a single parameter.

    Attributes
    ----------
    solver : str
        Algorithm name.
    param_name : str
        Name of the parameter being varied.
    param_values : list
        Values of the parameter that were swept.
    results : list[BenchmarkResult]
        One BenchmarkResult per parameter value, in the same order as
        param_values. May be shorter if some solver calls failed.
    baseline_config : dict
        Fixed parameter values used for all runs in this sweep.
    n_solver_calls : int
        Number of successful solver invocations.
    total_sweep_time_s : float
        Total wall time [s] for the entire sweep.
    """

    solver:             str
    param_name:         str
    param_values:       list
    results:            list[BenchmarkResult]
    baseline_config:    dict
    n_solver_calls:     int
    total_sweep_time_s: float


# ── HHL sensitivity sweep ─────────────────────────────────────────────────────

def sensitivity_sweep_hhl(
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
    param_name: str,
    param_values: Optional[list] = None,
    baseline: Optional[dict] = None,
    extract_circuits: bool = True,
    backend_name: str = "aer_statevector",
    hardware_run: bool = False,
    backend_shots: Optional[int] = None,
) -> SensitivitySweepResult:
    """
    One-at-a-time sensitivity sweep for the HHL algorithm.

    Parameters
    ----------
    param_name : str
        Parameter to vary: 'epsilon'.
    param_values : list, optional
        Values to sweep. Defaults to HHL_SENSITIVITY_GRIDS[param_name].
    baseline : dict, optional
        Fixed parameter values. Defaults to HHL_BASELINE.

    Returns
    -------
    SensitivitySweepResult
        Sweep results with one BenchmarkResult per parameter value.

    Raises
    ------
    ValueError
        If param_name is not a recognised HHL sensitivity parameter.
    """
    # `hhl_solve_system` takes epsilon positionally and returns the plain tuple
    # (u, raw_state, prop_const); there is no HHLConfig1D in this codebase. An
    # earlier draft assumed a config object and an attribute-bearing result,
    # matching neither, so this sweep raised on import.
    from solvers.quantum.hhl_1d import hhl_solve_system

    if param_name not in HHL_SENSITIVITY_GRIDS:
        raise ValueError(
            f"Unknown HHL sensitivity parameter '{param_name}'. "
            f"Valid options: {list(HHL_SENSITIVITY_GRIDS)}"
        )

    if param_values is None:
        param_values = HHL_SENSITIVITY_GRIDS[param_name]
    if baseline is None:
        baseline = HHL_BASELINE.copy()

    results: list[BenchmarkResult] = []
    t_start = time.perf_counter()

    for val in param_values:
        cfg_kwargs = {**baseline, param_name: val}
        log.info(
            "  HHL sensitivity: N=%d  %s=%.4g  (baseline=%s)",
            N, param_name, val, baseline,
        )
        eps_val = float(cfg_kwargs["epsilon"])

        try:
            t0 = time.perf_counter()
            u_sol, raw_state, _prop_const = hhl_solve_system(A, b, eps_val)
            wall = time.perf_counter() - t0

            u_sol = np.array(u_sol)
            n_trotter = int(np.ceil(1.0 / eps_val))

            prop_residual = None
            if raw_state is not None:
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
                wall_time_s=wall, r_target=None,
                backend_name=backend_name, hardware_run=hardware_run,
                backend_shots=backend_shots,
            )
            rec.hhl_epsilon = eps_val
            rec.hhl_trotter_steps = n_trotter
            rec.proportionality_residual = prop_residual
            rec.sensitivity_param = param_name
            rec.sensitivity_value = float(val)

            # `hhl_solve_system` returns only (u, raw_state, prop_const), so there
            # is no circuit object to measure here; the other two solvers return a
            # result carrying one. Circuit metrics for HHL come from the primary
            # sweep, which records them per row.
            results.append(rec)
            log.info(
                "    residual=%.4e  err_vs_exact=%s%%  time=%.2fs",
                rec.residual,
                f"{rec.max_rel_err_vs_exact:.3f}" if rec.max_rel_err_vs_exact else "N/A",
                wall,
            )

        except Exception as exc:
            log.warning("  HHL sensitivity failed at %s=%s: %s", param_name, val, exc)

    return SensitivitySweepResult(
        solver="hhl",
        param_name=param_name,
        param_values=param_values,
        results=results,
        baseline_config=baseline,
        n_solver_calls=len(results),
        total_sweep_time_s=time.perf_counter() - t_start,
    )


# ── VQLS sensitivity sweep ────────────────────────────────────────────────────

def sensitivity_sweep_vqls(
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
    param_name: str,
    param_values: Optional[list] = None,
    baseline: Optional[dict] = None,
    extract_circuits: bool = True,
    backend_name: str = "aer_statevector",
    hardware_run: bool = False,
    backend_shots: Optional[int] = None,
) -> SensitivitySweepResult:
    """
    One-at-a-time sensitivity sweep for the VQLS algorithm.

    Parameters
    ----------
    param_name : str
        Parameter to vary: 'n_layers' | 'n_restarts' | 'cobyla_tol'.

    Notes on cobyla_tol
    ───────────────────
    Reducing cobyla_tol does NOT guarantee a smaller residual. The COBYLA
    optimiser may converge to a local minimum regardless of the tolerance.
    The residual is always measured from the returned solution vector.
    The cobyla_tol sweep therefore measures the *optimiser effort* required
    to reach a given cost value, not the achievable solution accuracy.
    """
    from solvers.quantum.vqls_1d import vqls_solve_system, VQLSConfig1D

    if param_name not in VQLS_SENSITIVITY_GRIDS:
        raise ValueError(
            f"Unknown VQLS sensitivity parameter '{param_name}'. "
            f"Valid options: {list(VQLS_SENSITIVITY_GRIDS)}"
        )

    if param_values is None:
        param_values = VQLS_SENSITIVITY_GRIDS[param_name]
    if baseline is None:
        baseline = VQLS_BASELINE.copy()

    results: list[BenchmarkResult] = []
    t_start = time.perf_counter()

    for val in param_values:
        cfg_kwargs = {**baseline, param_name: val}
        log.info(
            "  VQLS sensitivity: N=%d  %s=%s  (baseline=%s)",
            N, param_name, val, baseline,
        )
        # `VQLSConfig1D` names the optimiser tolerance `tol`. It is called
        # `cobyla_tol` throughout this module, and in the recorded parameter name,
        # because `tol` alone is ambiguous once a row is read next to a residual
        # tolerance or an outer-iteration tolerance -- so the descriptive name is
        # kept in the study and translated to the constructor's name here rather
        # than renamed at either end.
        cfg_kwargs = dict(cfg_kwargs)
        if "cobyla_tol" in cfg_kwargs:
            cfg_kwargs["tol"] = cfg_kwargs.pop("cobyla_tol")
        cfg = VQLSConfig1D(**cfg_kwargs)

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
                wall_time_s=wall, r_target=None,
                backend_name=backend_name, hardware_run=hardware_run,
                backend_shots=backend_shots,
            )
            rec.vqls_n_layers = cfg.n_layers
            rec.vqls_n_restarts = cfg.n_restarts
            rec.vqls_cost_final = float(solver_result.final_cost)
            rec.vqls_n_evaluations = getattr(solver_result, "n_evaluations", None)
            rec.vqls_converged = getattr(solver_result, "converged", None)
            rec.sensitivity_param = param_name
            rec.sensitivity_value = float(val)

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
                rec.residual, rec.vqls_cost_final, rec.vqls_converged, wall,
            )

        except Exception as exc:
            log.warning(
                "  VQLS sensitivity failed at %s=%s: %s", param_name, val, exc
            )

    return SensitivitySweepResult(
        solver="vqls",
        param_name=param_name,
        param_values=param_values,
        results=results,
        baseline_config=baseline,
        n_solver_calls=len(results),
        total_sweep_time_s=time.perf_counter() - t_start,
    )


# ── QSVT sensitivity sweep ────────────────────────────────────────────────────

def sensitivity_sweep_qsvt(
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
    param_name: str,
    param_values: Optional[list] = None,
    baseline: Optional[dict] = None,
    extract_circuits: bool = True,
    backend_name: str = "aer_statevector",
    hardware_run: bool = False,
    backend_shots: Optional[int] = None,
) -> SensitivitySweepResult:
    """
    One-at-a-time sensitivity sweep for the QSVT algorithm.

    Parameters
    ----------
    param_name : str
        Parameter to vary: 'max_degree' | 'epsilon'.

    Notes on max_degree
    ───────────────────
    The QSVT residual is not monotone in polynomial degree. A higher degree
    does not guarantee a lower residual. The sweep evaluates all grid points
    and records the residual at each.

    Notes on epsilon
    ────────────────
    The epsilon parameter governs the target approximation error for the
    1/x polynomial. A smaller epsilon requires a higher polynomial degree,
    which increases circuit depth and wall time. The achieved residual may
    not improve proportionally with epsilon due to the oscillatory nature
    of the Chebyshev approximation.
    """
    from solvers.quantum.qsvt_1d import qsvt_solve_system, QSVTConfig1D

    if param_name not in QSVT_SENSITIVITY_GRIDS:
        raise ValueError(
            f"Unknown QSVT sensitivity parameter '{param_name}'. "
            f"Valid options: {list(QSVT_SENSITIVITY_GRIDS)}"
        )

    if param_values is None:
        param_values = QSVT_SENSITIVITY_GRIDS[param_name]
    if baseline is None:
        baseline = QSVT_BASELINE.copy()

    results: list[BenchmarkResult] = []
    t_start = time.perf_counter()

    for val in param_values:
        cfg_kwargs = {**baseline, param_name: val}
        deg_label = "uncapped" if val is None else str(val)
        log.info(
            "  QSVT sensitivity: N=%d  %s=%s  (baseline=%s)",
            N, param_name, deg_label, baseline,
        )
        cfg = QSVTConfig1D(**cfg_kwargs)

        try:
            t0 = time.perf_counter()
            solver_result = qsvt_solve_system(A, b, config=cfg)
            wall = time.perf_counter() - t0

            u_sol = np.array(solver_result.u)

            prop_residual = None
            if hasattr(solver_result, "raw_state") and solver_result.raw_state is not None:
                Ax_raw = A @ solver_result.raw_state
                c_val  = float(
                    np.dot(b, Ax_raw) / (np.dot(Ax_raw, Ax_raw) + 1.0e-300)
                )
                prop_residual = compute_residual(
                    A, c_val * solver_result.raw_state, b
                )

            alpha_sub = float(np.linalg.norm(A, ord=2))
            kappa_eff = kappa * alpha_sub

            rec = _build_base_result(
                case_id=case_id, solver="qsvt", N=N, kappa=kappa,
                source_fn=source_fn, alpha_bc=alpha_bc, beta_bc=beta_bc,
                discretisation_order=discretisation_order,
                u_solver=u_sol, A=A, b=b,
                u_thomas=u_thomas, u_exact=u_exact,
                wall_time_s=wall, r_target=None,
                backend_name=backend_name, hardware_run=hardware_run,
                backend_shots=backend_shots,
            )
            rec.proportionality_residual = prop_residual
            rec.phase_lookup_time_s = getattr(
                solver_result, "phase_lookup_time_s", None
            )
            rec.qsvt_polynomial_degree = getattr(solver_result, "degree", None)
            rec.qsvt_max_degree_cap = val if param_name == "max_degree" else baseline.get("max_degree")
            rec.qsvt_subnormalisation = alpha_sub
            rec.qsvt_kappa_eff = kappa_eff
            rec.qsvt_angle_method = cfg.angle_method
            rec.qsvt_phase_from_cache = getattr(
                solver_result, "phase_from_cache", None
            )
            rec.sensitivity_param = param_name
            rec.sensitivity_value = float(val) if val is not None else -1.0

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
            log.warning(
                "  QSVT sensitivity failed at %s=%s: %s", param_name, val, exc
            )

    return SensitivitySweepResult(
        solver="qsvt",
        param_name=param_name,
        param_values=param_values,
        results=results,
        baseline_config=baseline,
        n_solver_calls=len(results),
        total_sweep_time_s=time.perf_counter() - t_start,
    )


# ── Convenience: run all sensitivity sweeps for a given solver ────────────────

def run_all_sensitivity_sweeps(
    solver: str,
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
    extract_circuits: bool = True,
    backend_name: str = "aer_statevector",
    hardware_run: bool = False,
    backend_shots: Optional[int] = None,
) -> list[SensitivitySweepResult]:
    """
    Run all OAT sensitivity sweeps for the specified solver.

    Iterates over all parameters defined in the solver's sensitivity grid
    and runs a separate sweep for each. Results are returned as a list of
    SensitivitySweepResult objects, one per parameter.

    Parameters
    ----------
    solver : str
        Algorithm name: 'hhl' | 'vqls' | 'qsvt'.

    Returns
    -------
    list[SensitivitySweepResult]
        One result per sensitivity parameter.

    Raises
    ------
    ValueError
        If solver is not recognised.
    """
    dispatch = {
        "hhl":  (sensitivity_sweep_hhl,  HHL_SENSITIVITY_GRIDS),
        "vqls": (sensitivity_sweep_vqls, VQLS_SENSITIVITY_GRIDS),
        "qsvt": (sensitivity_sweep_qsvt, QSVT_SENSITIVITY_GRIDS),
    }
    if solver not in dispatch:
        raise ValueError(
            f"Unknown solver '{solver}'. Valid options: {list(dispatch)}"
        )

    sweep_fn, grids = dispatch[solver]
    all_results: list[SensitivitySweepResult] = []

    for param_name in grids:
        log.info(
            "Running %s sensitivity sweep: param=%s  N=%d  case=%s",
            solver.upper(), param_name, N, case_id,
        )
        result = sweep_fn(
            A=A, b=b, u_thomas=u_thomas, u_exact=u_exact,
            case_id=case_id, N=N, kappa=kappa,
            source_fn=source_fn, alpha_bc=alpha_bc, beta_bc=beta_bc,
            discretisation_order=discretisation_order,
            param_name=param_name,
            extract_circuits=extract_circuits,
            backend_name=backend_name,
            hardware_run=hardware_run,
            backend_shots=backend_shots,
        )
        all_results.append(result)

    return all_results

# ── Sensitivity in 2-D and 3-D ────────────────────────────────────────────────

# Parameters swept per solver, identically named as in the `solvers/outer/inner.py`
# registry. The registry rejects an unknown key rather than ignoring it; therefore,
# a parameter name deviating from this mapping causes a failure at the initial
# solve rather than executing a null sweep. `cobyla_tol` is specified as `tol` here
# for consistency.
OUTER_SENSITIVITY_GRIDS: dict[str, dict[str, list]] = {
    "hhl":  {"epsilon": [0.1, 0.05, 0.01, 0.005]},
    "vqls": {"n_layers": [1, 2, 3, 4, 5],
             "n_restarts": [1, 2, 3, 5]},
    "qsvt": {"max_degree": [50, 100, 200, 500, None]},
}


def sensitivity_sweep_outer(
    problem,
    u_thomas: np.ndarray,
    u_exact: Optional[np.ndarray],
    case_id: str,
    N: int,
    kappa: float,
    source_fn: str,
    discretisation_order: int,
    solver: str,
    param_name: str,
    scheme: str = "fmg",
    param_values: Optional[list] = None,
    scheme_options: Optional[dict] = None,
    backend_name: str = "aer_statevector",
) -> SensitivitySweepResult:
    """
    One-at-a-time sensitivity sweep for one solver on a 2-D or 3-D line problem.

    The 1-D sweeps vary a solver config and re-solve an assembled ``(A, b)``. Here
    the parameter is an `inner_options` entry and the quantity measured is the
    outer residual after convergence, because the inner solver is executed once per
    strip per outer iteration, and the study focuses on this coupled behaviour.
    No dense operator is formed at any point.

    The baseline is implicit rather than declared: all options other than the one
    swept remain unset; consequently, each inner solver executes using its default
    parameters. That is the configuration the primary sweep records, which ensures
    the sensitivity curve is comparable against it.

    Parameters
    ----------
    problem : LineProblem2D
        Assembled problem satisfying the 2-D/3-D protocol.
    u_thomas : np.ndarray
        Classical reference field from an identical outer solve with
        ``inner="thomas"``, isolating the inner solver from the scheme.
    u_exact : np.ndarray or None
        Analytical field where the case has one.
    case_id, N, kappa, source_fn, discretisation_order
        Recorded on every row. `kappa` is the strip condition number κ_row.
    solver : {'hhl', 'vqls', 'qsvt'}
        Inner solver to sweep.
    param_name : str
        Option to vary; must appear in `OUTER_SENSITIVITY_GRIDS[solver]`.
    scheme : str
        Outer scheme, passed through to `solve`.
    param_values : list, optional
        Values to sweep. Defaults to the declared grid.
    scheme_options : dict, optional
        Forwarded to `solve`, e.g. ``max_wall_s``.

    Returns
    -------
    SensitivitySweepResult
        One BenchmarkResult per grid point that completed.

    Raises
    ------
    ValueError
        If `solver` or `param_name` is not one this function sweeps.
    """
    from benchmark.equal_accuracy import _build_base_result
    from solvers.outer import solve

    if solver not in OUTER_SENSITIVITY_GRIDS:
        raise ValueError(
            f"solver must be one of {sorted(OUTER_SENSITIVITY_GRIDS)}, "
            f"received {solver!r}.")
    grids = OUTER_SENSITIVITY_GRIDS[solver]
    if param_name not in grids:
        raise ValueError(
            f"Unknown {solver.upper()} sensitivity parameter {param_name!r}. "
            f"Valid options: {list(grids)}")

    values = param_values if param_values is not None else grids[param_name]
    scheme_options = dict(scheme_options or {})

    results: list[BenchmarkResult] = []
    t_start = time.perf_counter()

    for val in values:
        log.info("  %s sensitivity (outer): N=%d  %s=%s",
                 solver.upper(), N, param_name, val)
        # A None entry means "unset": the option is omitted rather than passed as
        # None, since the registry validates values as well as keys.
        inner_options = {} if val is None else {param_name: val}

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
                residual=res.residual, wall_time_s=wall, r_target=None,
                backend_name=backend_name,
            )
            rec.sensitivity_param = param_name
            rec.sensitivity_value = None if val is None else float(val)
            if solver == "vqls":
                rec.vqls_cost_final = res.diagnostics.get("final_cost_mean")
            elif solver == "qsvt":
                degree = res.diagnostics.get("polynomial_degree_mean")
                rec.qsvt_polynomial_degree = (None if degree is None
                                              else int(degree))

            results.append(rec)
            log.info("    outer residual=%.4e  n_outer=%d  stop=%s  time=%.1fs",
                     res.residual, res.n_outer, res.stop_reason, wall)

        except Exception as exc:
            log.warning("  %s failed at %s=%s: %s",
                        solver.upper(), param_name, val, exc)

    return SensitivitySweepResult(
        solver=solver,
        param_name=param_name,
        param_values=list(values),
        results=results,
        baseline_config={"scheme": scheme, "inner": solver,
                         "note": "all other options at inner-solver defaults"},
        n_solver_calls=len(results),
        total_sweep_time_s=time.perf_counter() - t_start,
    )
