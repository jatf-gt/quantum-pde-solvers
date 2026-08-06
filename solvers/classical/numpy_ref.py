"""
Direct dense solver built on the NumPy linear algebra backend.

Provides a reference baseline independent of the Thomas implementation, retained
to validate matrix construction and right-hand side assembly before any quantum
circuit is evaluated. Where Thomas exploits the tridiagonal structure at O(N)
cost, this routine performs a general LAPACK dense solve at O(N³); agreement
between the two is therefore a check on the assembly, not on either algorithm.
"""
from __future__ import annotations

import numpy as np

from problems.poisson_1d import PoissonProblem1D
from solvers.quantum.result import SolverResult


# ── NumPy Direct Solver ───────────────────────────────────────────────────────

def numpy_solve(problem: PoissonProblem1D) -> SolverResult:
    """
    Solves the linear system Au = b using NumPy's direct dense solver.

    Parameters
    ----------
    problem : PoissonProblem1D
        Discretised 1D problem supplying the N×N operator and length-N
        right-hand side.

    Returns
    -------
    result : SolverResult
        Solution vector, solver label and relative Euclidean residual.
    """
    u = np.linalg.solve(problem.A, problem.b)
    return SolverResult(
        u=u,
        solver="NumPy",
        euclidean_residual=_relative_residual(problem.A, u, problem.b),
    )


# ── Private Utility Methods ───────────────────────────────────────────────────

def _relative_residual(A: np.ndarray, u: np.ndarray, b: np.ndarray) -> float:
    """Computes the relative Euclidean residual ‖Au - b‖₂ / ‖b‖₂."""
    return float(np.linalg.norm(A @ u - b) / np.linalg.norm(b))
