"""
Implements the classical Thomas algorithm for tridiagonal linear systems.

This module provides an exact classical resolution technique for the 1D Poisson 
equation, serving as the primary baseline for the theoretical temporal complexity 
analysis in the reference literature.
"""
from __future__ import annotations

import numpy as np

from problems.poisson_1d import PoissonProblem1D
from solvers.quantum.result import SolverResult


# ── Thomas Algorithm ──────────────────────────────────────────────────────────

def thomas_solve(problem: PoissonProblem1D) -> SolverResult:
    """
    Resolves the tridiagonal system Au = b utilising the Thomas algorithm.

    This method executes in O(N) temporal complexity and yields a solution 
    accurate to machine precision. It corresponds directly to Algorithm 1 
    in the primary reference literature.
    """
    N   = problem.config.N
    b_d = -2.0 * np.ones(N)   # Main diagonal (modified in place)
    a_d =  1.0 * np.ones(N)   # Sub-diagonal (a_d[0] unused)
    c_d =  1.0 * np.ones(N)   # Super-diagonal (c_d[-1] unused)
    d   = problem.b.copy()    # Right-hand side vector

    # Forward sweep
    for i in range(1, N):
        m      = a_d[i] / b_d[i - 1]
        b_d[i] -= m * c_d[i - 1]
        d[i]   -= m * d[i - 1]

    # Back substitution
    u = np.zeros(N)
    u[-1] = d[-1] / b_d[-1]
    for i in range(N - 2, -1, -1):
        u[i] = (d[i] - c_d[i] * u[i + 1]) / b_d[i]

    return SolverResult(
        u=u,
        solver="Thomas",
        euclidean_residual=_relative_residual(problem.A, u, problem.b),
    )


# ── Private Utility Methods ───────────────────────────────────────────────────

def _relative_residual(A: np.ndarray, u: np.ndarray, b: np.ndarray) -> float:
    """Computes the relative Euclidean residual ||Au - b||_2 / ||b||_2."""
    return float(np.linalg.norm(A @ u - b) / np.linalg.norm(b))