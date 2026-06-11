"""
Implements a direct numerical solver utilising the NumPy linear algebra backend.

This module provides a robust computational baseline, retained primarily to 
validate matrix construction and right-hand side vector assembly prior to 
quantum circuit evaluation.
"""
from __future__ import annotations

import numpy as np

from problems.poisson_1d import PoissonProblem1D
from solvers.quantum.result import SolverResult


# ── NumPy Direct Solver ───────────────────────────────────────────────────────

def numpy_solve(problem: PoissonProblem1D) -> SolverResult:
    """
    Resolves the linear system Au = b employing NumPy's native direct solver.
    """
    u = np.linalg.solve(problem.A, problem.b)
    return SolverResult(
        u=u,
        solver="NumPy",
        euclidean_residual=_relative_residual(problem.A, u, problem.b),
    )


# ── Private Utility Methods ───────────────────────────────────────────────────

def _relative_residual(A: np.ndarray, u: np.ndarray, b: np.ndarray) -> float:
    """Computes the relative Euclidean residual ||Au - b||_2 / ||b||_2."""
    return float(np.linalg.norm(A @ u - b) / np.linalg.norm(b))