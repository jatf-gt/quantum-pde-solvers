"""
Implements the classical Thomas algorithm for tridiagonal linear systems.

The exact classical solution method used as the primary baseline reference. It
serves both as a standalone direct solver for 1D configurations and, through the
inner-solver registry in `solvers/outer/inner.py`, as the per-strip sub-routine
of every 2D and 3D outer iteration.

The algorithm is Gaussian elimination specialised to a tridiagonal matrix,
costing O(N) operations and O(N) memory — against O(N³) and O(N²) for a general
dense solve. It is unconditionally stable for the diagonally dominant operators
assembled here.
"""
from __future__ import annotations

import numpy as np

from problems.poisson_1d import PoissonProblem1D
from solvers.quantum.result import SolverResult


# ── 1D Benchmark Wrapper ──────────────────────────────────────────────────────

def thomas_solve(problem: PoissonProblem1D) -> SolverResult:
    """
    Solves the 1D Poisson system Au = b by the Thomas algorithm.

    A wrapper around `thomas_solve_system` that packages the solution vector
    into the standardised `SolverResult` used by the 1D benchmark suite,
    including the relative Euclidean residual ‖Au - b‖₂ / ‖b‖₂.

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
    Solves an arbitrary tridiagonal system Au = b by the Thomas algorithm.

    Accepts any tridiagonal matrix supplied as a dense N×N array. The three
    diagonals are extracted at call time rather than assumed, so the routine
    applies unchanged to the 1D Poisson operator (main diagonal -2) and to the
    line-decomposed strip operator of `PoissonLine2D`/`PoissonLine3D` (main
    diagonal -2(1/dx² + 1/dy²), or its h²-scaled equivalent -4 on a square mesh).

    This is the classical counterpart to `hhl_solve_system`, `vqls_solve_system`
    and `qsvt_solve_system`, and is the default inner solver of the outer
    iteration, invoked once per strip per sweep.

    Parameters
    ----------
    A : np.ndarray
        Dense N×N tridiagonal system matrix.
    b : np.ndarray
        Length-N right-hand side vector.

    Returns
    -------
    u : np.ndarray
        Length-N solution vector.

    Notes
    -----
    Cost is O(N) time and O(N) memory: one forward elimination sweep followed by
    one back substitution, with no pivoting. Inputs are copied, so neither A nor
    b is modified in place.
    """
    N = len(b)

    # Diagonals are extracted at call time, for compatibility with any
    # tridiagonal operator rather than one specific discretisation.
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
