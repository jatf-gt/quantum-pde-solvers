"""
Calculates error metrics and defines result structures for the Poisson benchmarks.

This module maintains strict independence from quantum circuit and solver 
libraries, executing purely classical post-processing arithmetic. This 
isolation ensures computational efficiency and facilitates independent 
verification of statistical error metrics for both 1D and 2D domains.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from core.config import SimConfig1D, SimConfig2D
from core.exact_solutions import EXACT_SOLUTIONS
from problems.poisson_1d import PoissonProblem1D
from problems.poisson_2d import PoissonProblem2D
from solvers.quantum.result import SolverResult, SolverResult2D


# ── Tolerance Thresholds ──────────────────────────────────────────────────────

# Threshold below which an analytical value is classified as effectively zero.
# Nodes satisfying |u_exact| < _NEAR_ZERO_TOL are systematically excluded from 
# relative error calculations to prevent artificial divergence. This approach 
# implicitly reflects the methodology applied to the central nodes of the f_H 
# source function within Section IV A of the reference literature.
_NEAR_ZERO_TOL = 1e-10


# ── 1D Result Container ───────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """
    Encapsulates the comprehensive error metrics for a single 1D solver execution.

    Attributes
    ----------
    config : SimConfig1D
        Configuration parameters governing the simulation instance.
    solver : str
        Identifier for the employed solver algorithm ('Thomas' or 'HHL').
    x : np.ndarray
        Spatial coordinates of the interior grid nodes.
    u_solver : np.ndarray
        Numerical solution vector extracted from the applied solver.
    u_exact : Optional[np.ndarray]
        Analytical solution evaluated at interior nodes. Assumes a value of 
        None when closed-form solutions are unavailable (e.g., non-homogeneous 
        boundary conditions lacking defined antiderivatives).
    u_thomas : Optional[np.ndarray]
        Classical Thomas solution vector. Retained alongside HHL outputs to 
        enable direct node-by-node comparative benchmarking.
    rel_error : Optional[np.ndarray]
        Pointwise relative error vector, formatted as percentages. Nodes where 
        |u_exact| < _NEAR_ZERO_TOL are assigned NaN values.
    abs_error : np.ndarray
        Pointwise absolute error vector.
    max_rel_error : Optional[float]
        Supremum of the relative error vector, excluding near-zero nodes.
    avg_rel_error : Optional[float]
        Arithmetic mean of the relative error vector, excluding near-zero nodes.
    max_abs_error : float
        Supremum of the absolute error vector.
    avg_abs_error : float
        Arithmetic mean of the absolute error vector.
    euclidean_residual : Optional[float]
        Relative Euclidean residual computed as ||Au - b||_2 / ||b||_2.
    prop_const : Optional[float]
        Proportionality constant extracted during HHL post-selection. Assigned 
        None for classical solvers.
    """
    config:             SimConfig1D
    solver:             str
    x:                  np.ndarray
    u_solver:           np.ndarray
    u_exact:            Optional[np.ndarray]
    u_thomas:           Optional[np.ndarray]
    rel_error:          Optional[np.ndarray]
    abs_error:          np.ndarray
    max_rel_error:      Optional[float]
    avg_rel_error:      Optional[float]
    max_abs_error:      float
    avg_abs_error:      float
    euclidean_residual: Optional[float]
    prop_const:         Optional[float] = None


# ── 1D Error Computation ──────────────────────────────────────────────────────

def compute_errors(
    problem: PoissonProblem1D,
    result: SolverResult,
    u_thomas: Optional[np.ndarray] = None,
) -> BenchmarkResult:
    """
    Evaluates statistical error metrics for a given 1D solver execution.

    For systems constrained by homogeneous Dirichlet boundary conditions 
    (alpha = beta = 0.0), the analytical solution is retrieved from 
    EXACT_SOLUTIONS to compute relative errors. Conversely, systems subject 
    to non-homogeneous boundary conditions default to absolute error evaluation 
    against the classical Thomas reference, as the primary literature omits 
    closed-form solutions for these configurations.

    Parameters
    ----------
    problem : PoissonProblem1D
        Discretised problem instance defining the linear system.
    result : SolverResult
        Numerical output yielded by the selected solver algorithm.
    u_thomas : Optional[np.ndarray], default=None
        Reference solution vector derived via the Thomas algorithm. Provided 
        during HHL evaluation to facilitate comparative error assessment.
        
    Returns
    -------
    BenchmarkResult
        Populated data structure containing all calculated deviation metrics.
    """
    cfg = problem.config
    x   = problem.x
    u   = result.u

    # ── Phase 1: Analytical Solution Formulation ──────────────────────────────
    has_exact = (
        cfg.alpha == 0.0 and 
        cfg.beta == 0.0 and
        cfg.source_fn in EXACT_SOLUTIONS
    )

    if has_exact:
        u_exact = EXACT_SOLUTIONS[cfg.source_fn](x)
    else:
        u_exact = None

    # ── Phase 2: Absolute Error Computation ───────────────────────────────────
    if u_exact is not None:
        abs_error = np.abs(u - u_exact)
    elif u_thomas is not None:
        abs_error = np.abs(u - u_thomas)
    else:
        abs_error = np.zeros_like(u)

    max_abs = float(np.max(abs_error))
    avg_abs = float(np.mean(abs_error))

    # ── Phase 3: Relative Error Computation ───────────────────────────────────
    if u_exact is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_error = np.where(
                np.abs(u_exact) > _NEAR_ZERO_TOL,
                np.abs(u - u_exact) / np.abs(u_exact) * 100.0,
                np.nan,
            )
        valid = rel_error[~np.isnan(rel_error)]
        max_rel = float(np.max(valid))  if valid.size > 0 else None
        avg_rel = float(np.mean(valid)) if valid.size > 0 else None
    else:
        rel_error = None
        max_rel   = None
        avg_rel   = None

    return BenchmarkResult(
        config=cfg,
        solver=result.solver,
        x=x,
        u_solver=u,
        u_exact=u_exact,
        u_thomas=u_thomas,
        rel_error=rel_error,
        abs_error=abs_error,
        max_rel_error=max_rel,
        avg_rel_error=avg_rel,
        max_abs_error=max_abs,
        avg_abs_error=avg_abs,
        euclidean_residual=result.euclidean_residual,
        prop_const=result.prop_const,
    )


# ── 2D Result Container ───────────────────────────────────────────────────────

@dataclass
class BenchmarkResult2D:
    """
    Encapsulates the comprehensive error metrics for a single 2D solver execution.

    Attributes
    ----------
    config : SimConfig2D
        Configuration parameters governing the 2D simulation instance.
    solver : str
        Identifier for the employed solver algorithm ('Thomas-2D' or 'HHL-2D').
    X : np.ndarray
    Y : np.ndarray
        (N, N) spatial coordinate matrices corresponding to interior grid nodes.
    u_solver : np.ndarray
        (N, N) numerical solution field extracted from the applied iterative solver.
    u_reference : Optional[np.ndarray]
        (N, N) high-fidelity reference solution derived via classical direct 
        resolution or refined Thomas methodologies. Assumes None if unavailable.
    abs_error : np.ndarray
        (N, N) pointwise absolute error matrix.
    rel_error : Optional[np.ndarray]
        (N, N) pointwise relative error matrix, formatted as percentages. Nodes 
        satisfying |u_reference| < _NEAR_ZERO_TOL are systematically assigned NaN values.
    max_rel_error : Optional[float]
        Supremum of the relative error matrix, excluding near-zero nodes.
    avg_rel_error : Optional[float]
        Arithmetic mean of the relative error matrix, excluding near-zero nodes.
    max_abs_error : float
        Supremum of the absolute error matrix.
    avg_abs_error : float
        Arithmetic mean of the absolute error matrix.
    iterations : int
        Total number of line-Jacobi iterations executed to achieve convergence.
    converged : bool
        Boolean indicator designating whether the solver successfully satisfied 
        the iteration tolerance threshold.
    iteration_errors : list[float]
        Sequential list tracking the supremum norm of the iterative difference 
        (max|u^{n+1} - u^n|) at each step. Retained explicitly for convergence 
        profile reconstruction.
    euclidean_residual : Optional[float]
        Relative Euclidean residual evaluated against the fully coupled N²×N² system.
    """
    config:             SimConfig2D
    solver:             str
    X:                  np.ndarray
    Y:                  np.ndarray
    u_solver:           np.ndarray
    u_reference:        Optional[np.ndarray]
    abs_error:          np.ndarray
    rel_error:          Optional[np.ndarray]
    max_rel_error:      Optional[float]
    avg_rel_error:      Optional[float]
    max_abs_error:      float
    avg_abs_error:      float
    iterations:         int
    converged:          bool
    iteration_errors:   list[float]
    euclidean_residual: Optional[float]


# ── 2D Error Computation ──────────────────────────────────────────────────────

def compute_errors_2d(
    problem:     PoissonProblem2D,
    result:      SolverResult2D,
    u_reference: Optional[np.ndarray] = None,
) -> BenchmarkResult2D:
    """
    Evaluates statistical error metrics for a given 2D line-Jacobi solver execution.

    The high-fidelity reference solution is typically derived via the classical 
    methodology (`problem.classical_reference_solve()`). It is injected directly 
    from the execution orchestrator to preclude redundant computational overhead 
    across successive solver evaluations for identical configurations.

    In accordance with Section IV F of the primary reference literature, relative 
    errors are predominantly utilised for homogeneous boundary conditions, whereas 
    absolute errors are prioritised for non-homogeneous constraints. This routine 
    computes both metrics simultaneously, deferring the contextual selection to 
    the downstream reporting layer.

    Parameters
    ----------
    problem : PoissonProblem2D
        Discretised 2D problem instance defining the linear system.
    result : SolverResult2D
        Numerical output yielded by the selected 2D solver algorithm.
    u_reference : Optional[np.ndarray], default=None
        (N, N) reference solution matrix utilised for baseline deviation analysis.

    Returns
    -------
    BenchmarkResult2D
        Populated data structure containing all calculated deviation metrics.
    """
    u = result.u

    # ── Phase 1: Absolute Error Computation ───────────────────────────────────
    if u_reference is not None:
        abs_error = np.abs(u - u_reference)
    else:
        abs_error = np.zeros_like(u)

    max_abs = float(np.max(abs_error))
    avg_abs = float(np.mean(abs_error))

    # ── Phase 2: Relative Error Computation ───────────────────────────────────
    # Masked where |u_reference| approaches zero to prevent artificial division 
    # singularities, mirroring the primary reference's protocol for near-zero nodes.
    if u_reference is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_error = np.where(
                np.abs(u_reference) > _NEAR_ZERO_TOL,
                np.abs(u - u_reference) / np.abs(u_reference) * 100.0,
                np.nan,
            )
        valid   = rel_error[~np.isnan(rel_error)]
        max_rel = float(np.max(valid))  if valid.size > 0 else None
        avg_rel = float(np.mean(valid)) if valid.size > 0 else None
    else:
        rel_error = None
        max_rel   = None
        avg_rel   = None

    return BenchmarkResult2D(
        config=problem.config,
        solver=result.solver,
        X=problem.X,
        Y=problem.Y,
        u_solver=u,
        u_reference=u_reference,
        abs_error=abs_error,
        rel_error=rel_error,
        max_rel_error=max_rel,
        avg_rel_error=avg_rel,
        max_abs_error=max_abs,
        avg_abs_error=avg_abs,
        iterations=result.iterations,
        converged=result.converged,
        iteration_errors=result.iteration_errors,
        euclidean_residual=result.euclidean_residual,
    )