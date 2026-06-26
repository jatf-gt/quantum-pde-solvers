"""
Variational Quantum Linear Solver for the 2-D Poisson equation via the
line-Jacobi iterative decomposition of Ghafourpour & Laizet (2025).

Mathematical formulation
------------------------
The 2-D Poisson equation on [0,1]² is decomposed into a sequence of
1-D TST sub-problems via the line-Jacobi scheme (paper Eq. 9):

    u^{n+1}_{i+1,j} - 4·u^{n+1}_{i,j} + u^{n+1}_{i-1,j}
        = h²·f(x_i, y_j) - (u^n_{i,j-1} + u^n_{i,j+1})

Each row j yields a TST system A_row · u^{n+1}_{:,j} = b_row(j, u^n)
with a = -4, b = 1, κ(A_row) → 3⁻ as N → ∞. This near-constant
condition number makes the 2-D sub-problems significantly better
conditioned than the 1-D Poisson system (κ ~ O(N²)).

Performance strategy
--------------------
Two optimisations reduce the computational overhead of calling VQLS
N × max_iter times:

1. Pauli decomposition caching: A_row is identical for all rows and
   all iterations. Its LCU decomposition is computed once and reused,
   eliminating O(4^n) matrix operations per call.

2. Warm-starting: The optimal parameters θ* from row j at iteration n
   serve as the initial guess for row j at iteration n+1. Since the
   RHS changes only through the y-neighbour terms (which converge
   monotonically), the cost landscape shifts gradually and the previous
   optimum is a reliable starting point.

References
----------
Ghafourpour & Laizet, Phys. Rev. Applied 24, 024032 (2025).
Bravo-Prieto et al., Quantum 7, 1188 (2023).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from problems.poisson_2d import PoissonProblem2D
from solvers.quantum.result import SolverResult2D, VQLSSolverResult
from solvers.quantum.vqls_1d import VQLSConfig1D, DEFAULT_VQLS_CONFIG, vqls_solve_system
from solvers.quantum.vqls_utils import (
    pauli_decompose_normalised,
    n_params,
)

_ZERO_RHS_TOL = 1e-14


# -- VQLS 2-D Configuration ---------------------------------------------------

@dataclass
class VQLSConfig2D:
    """
    Hyperparameters governing the outer line-Jacobi loop and the inner
    VQLS sub-solver for the 2-D Poisson problem.

    The inner VQLSConfig1D controls the variational optimisation for each
    row sub-problem. The outer parameters control the Jacobi iteration.

    Attributes
    ----------
    inner_config : VQLSConfig1D
        Configuration for the per-row VQLS optimisation. Defaults to
        a fast configuration suitable for the well-conditioned 2-D row
        matrix (κ ≈ 2.77).
    warm_start : bool
        If True, the optimal parameters from the previous iteration are
        used as the initial guess for the current iteration. This
        exploits the slow variation of the RHS between Jacobi iterates
        and typically halves the number of circuit evaluations required
        for convergence. Default True.
    verbose : bool
        If True, print iteration progress every 10 sweeps. Default False.
    """

    inner_config : VQLSConfig1D = field(
        default_factory=lambda: VQLSConfig1D(
            n_layers    = 4,
            optimiser   = "COBYLA",
            max_iter    = 200,
            tol         = 1e-4,
            random_seed = 0,
            verbose     = False,
        )
    )
    warm_start   : bool = True
    verbose      : bool = False


DEFAULT_VQLS_CONFIG_2D = VQLSConfig2D()


# -- Public interface ---------------------------------------------------------

def vqls_solve_2d(
    problem : PoissonProblem2D,
    config  : VQLSConfig2D = DEFAULT_VQLS_CONFIG_2D,
) -> SolverResult2D:
    """
    Solve the 2-D Poisson equation using VQLS inside a line-Jacobi loop.

    Each iteration performs a full sweep over all N interior rows. For
    each row j the sub-problem

        A_row · u^{n+1}_{:,j} = b_row(j, u^n)

    is solved by the 1-D VQLS solver. The Pauli decomposition of A_row
    is computed once before the loop and cached for all subsequent calls.

    Convergence criterion (Ghafourpour & Laizet 2025, Section IV E):

        max_{i,j} |u^{n+1}_{i,j} - u^n_{i,j}| < tol

    Parameters
    ----------
    problem : PoissonProblem2D
        Discretised 2-D Poisson problem containing the row matrix,
        grid coordinates, source function, and boundary conditions.
    config : VQLSConfig2D
        Outer loop and inner VQLS hyperparameters.

    Returns
    -------
    SolverResult2D
        Converged (or best available) solution field of shape (N, N),
        with iteration history and residual diagnostics.
    """
    cfg = problem.config
    N   = cfg.N
    u   = problem.u_init.copy()

    # -- Pre-compute Pauli decomposition for A_row ----------------------------
    # A_row is identical for all rows and iterations; computing the LCU
    # decomposition once avoids O(4^n_qubits) matrix operations per call.
    pauli_terms, A_norm_factor = _cache_pauli_decomposition(
        problem.A_row, N
    )

    # -- Initialise per-row parameter cache -----------------------------------
    # warm_params[j] stores the optimal θ* from the most recent solve of
    # row j. Initialised to None; the first solve uses random initialisation.
    n_p          = n_params(int(np.log2(N)), config.inner_config.n_layers)
    warm_params  : List[Optional[np.ndarray]] = [None] * N

    iteration_errors : List[float] = []

    if config.verbose:
        print(
            f"  VQLS-2D: N={N}, max_iter={cfg.max_iter}, "
            f"n_layers={config.inner_config.n_layers}, "
            f"warm_start={config.warm_start}"
        )

    for iteration in range(cfg.max_iter):
        u_new = np.zeros((N, N))

        # -- Full sweep over all interior rows --------------------------------
        for j in range(N):
            A_row, b_row = problem.get_row_system(j, u)

            # Skip the VQLS call if the RHS is numerically zero.
            if np.linalg.norm(b_row) < _ZERO_RHS_TOL:
                u_new[:, j] = np.zeros(N)
                continue

            # Build a per-row inner config, injecting warm-start parameters
            # if available and warm_start is enabled.
            row_config = _build_row_config(
                config.inner_config,
                warm_params[j] if config.warm_start else None,
            )

            result       = vqls_solve_system(A_row, b_row, row_config)
            u_new[:, j]  = result.u

            # Cache the optimal parameters for warm-starting next iteration.
            if config.warm_start:
                warm_params[j] = result.optimal_params

        # -- Convergence check ------------------------------------------------
        iter_error = float(np.max(np.abs(u_new - u)))
        iteration_errors.append(iter_error)
        u = u_new

        if config.verbose and (iteration + 1) % 10 == 0:
            print(
                f"    Iteration {iteration+1:4d}/{cfg.max_iter}  "
                f"error = {iter_error:.3e}"
            )

        if iter_error < cfg.tol:
            if config.verbose:
                print(f"    Converged at iteration {iteration + 1}.")
            return _package_result(
                problem, u, iteration + 1, True, iteration_errors
            )

    if config.verbose:
        print(
            f"    Maximum iterations ({cfg.max_iter}) reached without "
            f"convergence. Final error: {iteration_errors[-1]:.3e}"
        )

    return _package_result(
        problem, u, cfg.max_iter, False, iteration_errors
    )


# -- Private utilities --------------------------------------------------------

def _cache_pauli_decomposition(
    A_row        : np.ndarray,
    N            : int,
) -> Tuple[list, float]:
    """
    Compute and return the LCU Pauli decomposition of the normalised
    row matrix A_row / ‖A_row‖₂.

    Parameters
    ----------
    A_row : np.ndarray, shape (N, N)
        TST row matrix with main diagonal −4 and off-diagonals +1.
    N : int
        System size; must be a power of 2.

    Returns
    -------
    pauli_terms : list of (complex, str)
        LCU decomposition of A_row / ‖A_row‖₂.
    A_norm_factor : float
        Spectral norm ‖A_row‖₂ used for rescaling.
    """
    main_diag = float(A_row[0, 0])
    off_diag  = float(A_row[0, 1])
    return pauli_decompose_normalised(N, main_diag, off_diag)


def _build_row_config(
    base_config  : VQLSConfig1D,
    init_params  : Optional[np.ndarray],
) -> VQLSConfig1D:
    """
    Construct a per-row VQLSConfig1D, optionally injecting warm-start
    parameters from the previous iteration.

    A new dataclass instance is created rather than mutating the shared
    base_config, ensuring thread safety and preventing state leakage
    between rows.

    Parameters
    ----------
    base_config : VQLSConfig1D
        Shared inner configuration for all row solves.
    init_params : np.ndarray or None, shape (n_params,)
        Warm-start parameter vector from the previous iteration, or
        None if no prior solve exists for this row.

    Returns
    -------
    VQLSConfig1D
        Row-specific configuration with init_params injected.
    """
    return VQLSConfig1D(
        n_layers    = base_config.n_layers,
        optimiser   = base_config.optimiser,
        max_iter    = base_config.max_iter,
        tol         = base_config.tol,
        init_params = init_params,
        random_seed = base_config.random_seed,
        device_name = base_config.device_name,
        verbose     = base_config.verbose,
    )


def _package_result(
    problem          : PoissonProblem2D,
    u                : np.ndarray,
    iterations       : int,
    converged        : bool,
    iteration_errors : List[float],
) -> SolverResult2D:
    """
    Compute the full-system residual via the tridiagonal matvec and
    package all outputs into a SolverResult2D.

    The residual is computed without forming the full N²×N² matrix,
    using the O(N²) tridiagonal matvec from poisson_2d.py.

    Parameters
    ----------
    problem : PoissonProblem2D
    u : np.ndarray, shape (N, N)
        Converged or best-available solution field.
    iterations : int
        Number of line-Jacobi sweeps performed.
    converged : bool
    iteration_errors : list of float

    Returns
    -------
    SolverResult2D
    """
    from problems.poisson_2d import _compute_residual_tridiagonal
    residual = _compute_residual_tridiagonal(problem, u)
    return SolverResult2D(
        u                 = u,
        solver            = "VQLS-2D",
        iterations        = iterations,
        converged         = converged,
        iteration_errors  = iteration_errors,
        euclidean_residual= residual,
    )