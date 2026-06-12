"""
Implements the classical Thomas algorithm for tridiagonal linear systems.

This module provides the exact classical resolution technique utilised as the 
primary baseline reference. It is employed both as a standalone direct solver 
for 1D configurations and as the core sub-routine within the 2D line-Jacobi 
iterative loop.
"""
from __future__ import annotations

import numpy as np

from problems.poisson_1d import PoissonProblem1D
from solvers.quantum.result import SolverResult


# ── 1D Benchmark Wrapper ──────────────────────────────────────────────────────

def thomas_solve(problem: PoissonProblem1D) -> SolverResult:
    """
    Resolves the 1D Poisson system Au = b utilising the Thomas algorithm.
    
    Serves as a procedural wrapper around the core `thomas_solve_system` routine, 
    packaging the numerical array output into a standardised `SolverResult` object 
    for the 1D benchmark suite.
    """
    u = thomas_solve_system(problem.A, problem.b)
    residual = float(
        np.linalg.norm(problem.A @ u - problem.b)
        / np.linalg.norm(problem.b)
    )
    return SolverResult(
        u=u,
        solver="Thomas",
        euclidean_residual=residual,
    )


# ── Core Algorithm ────────────────────────────────────────────────────────────

def thomas_solve_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Resolves an arbitrary tridiagonal system Au = b employing the Thomas algorithm.

    This routine processes any tridiagonal matrix passed as a dense N×N array. 
    By extracting the principal and sub/super-diagonals dynamically, the function 
    maintains strict compatibility with both the 1D Poisson operator (main diagonal a=-2) 
    and the 2D line-Jacobi operator (main diagonal a=-4).

    This core function acts as the classical analog to the `hhl_solve_system` 
    sub-routine, and is sequentially invoked by the 2D Thomas line-Jacobi solver 
    for each interior spatial row.

    Parameters
    ----------
    A : np.ndarray
        Dense N×N tridiagonal system matrix.
    b : np.ndarray
        Right-hand side vector of length N.

    Returns
    -------
    u : np.ndarray
        Numerical solution vector of length N.
    """
    N = len(b)

    # Dynamic diagonal extraction ensures compatibility across arbitrary tridiagonal inputs.
    b_diag = A.diagonal(0).copy()       # Principal diagonal
    c_diag = A.diagonal(1).copy()       # Super-diagonal (length N-1)
    a_diag = A.diagonal(-1).copy()      # Sub-diagonal (length N-1)
    d      = b.copy()

    # Forward elimination sweep
    for i in range(1, N):
        m        = a_diag[i - 1] / b_diag[i - 1]
        b_diag[i] -= m * c_diag[i - 1]
        d[i]      -= m * d[i - 1]

    # Back substitution phase
    u = np.zeros(N)
    u[-1] = d[-1] / b_diag[-1]
    for i in range(N - 2, -1, -1):
        u[i] = (d[i] - c_diag[i] * u[i + 1]) / b_diag[i]

    return u