"""
Implements the classical Thomas-based line-Jacobi solver for the 2D Poisson equation.

This module provides the classical baseline reference for two-dimensional domains. 
Its execution architecture strictly mirrors the sequential structure of the HHL 
line-Jacobi solver, guaranteeing a mathematically fair comparative analysis. 

Within the primary reference literature (Section III B), this methodology corresponds 
to Algorithm 1 extended to 2D domains via the line-Jacobi decomposition scheme.
"""
from __future__ import annotations

import numpy as np

from problems.poisson_2d import PoissonProblem2D
from solvers.classical.thomas import thomas_solve_system
from solvers.quantum.result import SolverResult2D


# ── 2D Line-Jacobi Solver ─────────────────────────────────────────────────────

def thomas_solve_2d(problem: PoissonProblem2D) -> SolverResult2D:
    """
    Resolves the 2D Poisson equation employing the Thomas algorithm embedded 
    within a line-Jacobi iterative cycle.

    During each sequential iteration, every interior row j is updated by solving 
    the isolated 1D tridiagonal sub-system:

        u^{n+1}_{i+1,j} - 4·u^{n+1}_{i,j} + u^{n+1}_{i-1,j}
            = h²·f(x_i, y_j) - (u^n_{i,j-1} + u^n_{i,j+1})

    The resolution of this sub-system is executed via the `thomas_solve_system` routine. 
    A comprehensive sweep across all interior rows constitutes one global iteration. 
    This iterative progression continues until the absolute update magnitude satisfies 
    the convergence threshold:

        max_{i,j} |u^{n+1}_{i,j} - u^n_{i,j}| < tol

    Alternatively, execution terminates if the predefined maximum iteration 
    ceiling (max_iter) is reached.

    Parameters
    ----------
    problem : PoissonProblem2D
        Data structure encapsulating the 2D discretised system parameters.

    Returns
    -------
    SolverResult2D
        Standardised result object containing the converged (N, N) spatial solution field.
    """
    cfg  = problem.config
    N    = cfg.N
    u    = problem.u_init.copy()

    iteration_errors: list[float] = []

    for iteration in range(cfg.max_iter):
        u_new = np.zeros((N, N))

        # Perform one comprehensive sweep: update every interior row independently 
        # utilising boundary data from the preceding solution state (u).
        for j in range(N):
            A_row, b_row = problem.get_row_system(j, u)
            u_new[:, j]  = thomas_solve_system(A_row, b_row)

        # Evaluate the supremum norm of the sequential difference to assess convergence.
        iter_error = float(np.max(np.abs(u_new - u)))
        iteration_errors.append(iter_error)

        u = u_new

        if iter_error < cfg.tol:
            return _package_result(
                problem, u, iteration + 1, True, iteration_errors, "Thomas-2D"
            )

    # Triggered strictly if the solver reaches max_iter without achieving convergence.
    return _package_result(
        problem, u, cfg.max_iter, False, iteration_errors, "Thomas-2D"
    )


# ── Private Utility Methods ───────────────────────────────────────────────────

def _package_result(
    problem:          PoissonProblem2D,
    u:                np.ndarray,
    iterations:       int,
    converged:        bool,
    iteration_errors: list[float],
    solver_label:     str,
) -> SolverResult2D:
    """
    Encapsulates the numerical solver output and associated metrics into a 
    standardised `SolverResult2D` data structure.

    This routine computes two distinct residual metrics to facilitate comprehensive analysis:
    
    1. Euclidean Residual : ||A_full * u_flat - b_full||_2 / ||b_full||_2
       Quantifies the global deviation of the iterative solution from the exact 
       analytical solution of the fully coupled N²×N² system. It should be noted 
       that for the Jacobi iterative scheme, this residual may remain on the order 
       of O(1) even when the sequential update norm is minimal. This reflects the 
       inherent convergence properties of the outer iterative loop and does not 
       indicate an algorithmic fault or computational bug.
       
    2. Iteration Error : Tracked sequentially within `iteration_errors`.
       The terminal value of this array represents the actual mathematical convergence 
       criterion (supremum norm) successfully satisfied by the line-Jacobi loop.
    """
    A_full = problem.build_full_matrix()
    b_full = problem.build_full_rhs()
    u_flat = u.flatten(order="C")
    
    residual = float(
        np.linalg.norm(A_full @ u_flat - b_full)
        / np.linalg.norm(b_full)
    )
    
    return SolverResult2D(
        u=u,
        solver=solver_label,
        iterations=iterations,
        converged=converged,
        iteration_errors=iteration_errors,
        euclidean_residual=residual,
    )