"""
Quantum Singular Value Transformation (QSVT) solver for the 2-D Poisson
equation via the line-Jacobi iterative decomposition.

Mathematical foundation
-----------------------
The 2-D Poisson equation on [0,1]^2 is decomposed into a sequence of
1-D TST sub-problems via the line-Jacobi scheme (Ghafourpour & Laizet
2025, Eq. 9):

    u^{n+1}_{i+1,j} - 4*u^{n+1}_{i,j} + u^{n+1}_{i-1,j}
        = h^2*f(x_i, y_j) - (u^n_{i,j-1} + u^n_{i,j+1})

Each row j yields a TST system A_row * u^{n+1}_{:,j} = b_row(j, u^n)
with a = -4, b = 1. The condition number of A_row satisfies:

    kappa(A_row) -> 3^-  as N -> infinity

This near-constant condition number is the key advantage of QSVT in 2-D:
the polynomial degree required for matrix inversion is:

    d = O(kappa_row * log(kappa_row/epsilon))
    For kappa_row ~ 2.36 (N=4) and epsilon=0.1: d ~ 33 (pyqsp estimate)
    For kappa_row -> 3 (large N) and epsilon=0.1: d ~ 35 (pyqsp estimate)
    This is constant in N, confirming the scaling advantage over 1-D.

which is essentially independent of N, unlike the 1-D case where
d = O(N^2 * log(1/epsilon)).

Performance optimisations
--------------------------
Two optimisations reduce the computational overhead of calling QSVT
N * max_iter times:

1. Pre-computation and caching: A_row is identical for all rows and
   all iterations. Its block encoding circuit and QSP phase angles are
   computed once before the loop and reused for every row solve.
   This eliminates O(N * max_iter) block encoding constructions and
   O(N * max_iter) phase angle computations.

2. Optional parallelisation: The N row solves within each iteration are
   mutually independent (they share only the read-only u_prev array).
   On multi-core hardware, they can be executed in parallel using
   Python's ProcessPoolExecutor. This is controlled by the n_workers
   parameter and is particularly effective on HPC nodes with many cores.
   On a laptop, n_workers=1 (sequential) is recommended to avoid the
   overhead of process spawning for small circuits.

HPC deployment note
--------------------
On Imperial College's HPC cluster (PBS/SLURM), the recommended strategy
is to parallelise at the sweep level rather than the row level: submit
one job per benchmark configuration (N, epsilon, source function) and
run each job sequentially within a single node. This avoids the Python
multiprocessing overhead for small N whilst fully utilising the cluster's
job scheduler. For N >= 16, row-level parallelisation with n_workers
equal to the number of available cores becomes beneficial.

2-D extension design
---------------------
This module mirrors the structure of hhl_2d.py and vqls_2d.py exactly:
    - qsvt_solve_2d(problem, config) is the public interface
    - _package_result uses _compute_residual_tridiagonal (no full matrix)
    - The inner row solve calls qsvt_solve_system from qsvt_1d.py

References
----------
Ghafourpour & Laizet, Phys. Rev. Applied 24, 024032 (2025).
Gilyen, A., Su, Y., Low, G. H. & Wiebe, N. (2019). Quantum singular
    value transformation and beyond. STOC 2019, pp. 193-204.
Martyn, J. M., Rossi, Z. M., Tan, A. K. & Chuang, I. L. (2021). Grand
    unification of quantum algorithms. PRX Quantum, 2, 040203.
"""
from __future__ import annotations

import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from problems.poisson_2d import PoissonProblem2D
from solvers.quantum.block_encoding import build_tst_block_encoding
from solvers.quantum.qsp_angles import compute_inversion_angles
from solvers.quantum.qsvt_1d import QSVTConfig1D, qsvt_solve_system
from solvers.quantum.result import SolverResult2D

_ZERO_RHS_TOL = 1e-14


# -- QSVT 2-D configuration ---------------------------------------------------

@dataclass
class QSVTConfig2D:
    """
    Configuration for the QSVT 2-D line-Jacobi Poisson solver.

    Attributes
    ----------
    epsilon : float
        Target approximation error for the matrix inversion polynomial.
        For the 2-D row matrix with kappa_row ~ 3, the polynomial degree
        is d ~ 3 * log(1/epsilon), so epsilon=0.1 gives d ~ 7 and
        epsilon=0.01 gives d ~ 14. Default 0.1 for laptop tractability.
    angle_method : str
        Method for QSP phase angle computation. One of 'auto', 'pyqsp',
        'chebyshev'. Default 'auto'.
    max_degree : int
        Maximum allowed polynomial degree. Default 50 (sufficient for
        kappa_row ~ 3 and epsilon >= 1e-4).
    n_workers : int
        Number of parallel worker processes for row-level parallelisation.
        Set to 1 for sequential execution (recommended on laptops).
        Set to the number of available CPU cores for HPC deployment.
        Default 1.
    verbose : bool
        If True, print iteration progress and per-row timing. Default False.
    """

    epsilon     : float = 0.1
    angle_method: str   = "auto"
    max_degree  : int   = 50
    n_workers   : int   = 1
    verbose     : bool  = False


DEFAULT_QSVT_CONFIG_2D = QSVTConfig2D()


# -- Cached row solver state --------------------------------------------------

@dataclass
class _RowSolverCache:
    """
    Pre-computed quantities shared across all row solves.

    Caching these avoids redundant computation of the block encoding
    and QSP phase angles, which are identical for every row and every
    iteration since A_row never changes.

    Attributes
    ----------
    be_circuit : object
        Pre-built block encoding QuantumCircuit for A_row.
    angles : np.ndarray, shape (d+1,)
        Pre-computed QSP phase angles for the inversion polynomial.
    degree : int
        Polynomial degree d.
    alpha : float
        Block encoding subnormalisation factor (spectral norm of A_row).
    kappa_eff : float
        Effective condition number: kappa(A_row) (equals kappa since
        alpha = ||A_row||_2 for the Sz.-Nagy encoding).
    inner_config : QSVTConfig1D
        Inner 1-D QSVT configuration with pre-computed angles injected.
        Passed directly to qsvt_solve_system to skip recomputation.
    """

    be_circuit  : object
    angles      : np.ndarray
    degree      : int
    alpha       : float
    kappa_eff   : float
    inner_config: QSVTConfig1D


# -- Public interface ---------------------------------------------------------

def qsvt_solve_2d(
    problem : PoissonProblem2D,
    config  : QSVTConfig2D = DEFAULT_QSVT_CONFIG_2D,
) -> SolverResult2D:
    """
    Solve the 2-D Poisson equation using QSVT inside a line-Jacobi loop.

    Each iteration performs a full sweep over all N interior rows. For
    each row j the sub-problem

        A_row * u^{n+1}_{:,j} = b_row(j, u^n)

    is solved by the 1-D QSVT solver. The block encoding circuit and
    QSP phase angles are pre-computed once and reused for all rows.

    Convergence criterion (Ghafourpour & Laizet 2025, Section IV E):

        max_{i,j} |u^{n+1}_{i,j} - u^n_{i,j}| < tol

    Parameters
    ----------
    problem : PoissonProblem2D
        Discretised 2-D Poisson problem.
    config : QSVTConfig2D
        Outer loop and inner QSVT hyperparameters.

    Returns
    -------
    SolverResult2D
        Converged (or best available) solution field of shape (N, N),
        with iteration history and residual diagnostics.
    """
    cfg = problem.config
    N   = cfg.N
    u   = problem.u_init.copy()

    # -- Pre-compute block encoding and phase angles once --------------------
    cache = _build_row_cache(problem.A_row, N, config)

    if config.verbose:
        print(
            f"  QSVT-2D: N={N}, kappa_row={cache.kappa_eff:.4f}, "
            f"degree={cache.degree}, alpha={cache.alpha:.4f}, "
            f"max_iter={cfg.max_iter}, n_workers={config.n_workers}"
        )

    iteration_errors: list[float] = []

    for iteration in range(cfg.max_iter):

        if config.n_workers == 1:
            # -- Sequential row sweep ----------------------------------------
            u_new = _sweep_sequential(problem, u, cache, N)
        else:
            # -- Parallel row sweep ------------------------------------------
            u_new = _sweep_parallel(problem, u, cache, N, config.n_workers)

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
            f"    Maximum iterations ({cfg.max_iter}) reached. "
            f"Final error: {iteration_errors[-1]:.3e}"
        )

    return _package_result(
        problem, u, cfg.max_iter, False, iteration_errors
    )


# -- Private helpers ----------------------------------------------------------

def _build_row_cache(
    A_row  : np.ndarray,
    N      : int,
    config : QSVTConfig2D,
) -> _RowSolverCache:
    """
    Pre-compute the block encoding circuit and QSP phase angles for the
    row matrix A_row.

    These quantities are identical for every row and every iteration,
    so computing them once before the loop eliminates O(N * max_iter)
    redundant computations.

    Parameters
    ----------
    A_row : np.ndarray, shape (N, N)
        TST row matrix with main diagonal -4 and off-diagonals +1.
    N : int
        System size.
    config : QSVTConfig2D

    Returns
    -------
    _RowSolverCache
        Pre-computed quantities for injection into the inner solver.
    """
    main_diag = float(A_row[0, 0])
    off_diag  = float(A_row[0, 1])

    # Build block encoding circuit.
    be_circuit, alpha = build_tst_block_encoding(N, main_diag, off_diag)

    # Effective condition number: kappa_eff = kappa(A_row) since
    # alpha = ||A_row||_2 for the Sz.-Nagy encoding.
    eigs      = np.abs(np.linalg.eigvalsh(A_row))
    kappa_eff = float(eigs.max() / eigs.min())

    # Estimate polynomial degree and check against max_degree.
    from solvers.quantum.qsp_angles import polynomial_degree_estimate
    est_degree = polynomial_degree_estimate(kappa_eff, config.epsilon)
    if est_degree > config.max_degree:
        warnings.warn(
            f"Estimated polynomial degree {est_degree} exceeds "
            f"max_degree={config.max_degree}. Capping at {config.max_degree}. "
            f"Solution accuracy may be reduced.",
            RuntimeWarning,
        )
        est_degree = config.max_degree

    # Compute QSP phase angles.
    angles, degree = compute_inversion_angles(
        kappa   = kappa_eff,
        epsilon = config.epsilon,
        method  = config.angle_method,
        max_degree = config.max_degree,
    )

    # Build an inner QSVTConfig1D with the angles pre-injected.
    # The inner solver will use these angles directly rather than
    # recomputing them, bypassing the angle computation step.
    inner_config = _QSVTConfig1DWithAngles(
        epsilon      = config.epsilon,
        angle_method = config.angle_method,
        max_degree   = config.max_degree,
        verbose      = False,
        _precomputed_angles = angles,
        _precomputed_degree = degree,
        _precomputed_alpha  = alpha,
    )

    return _RowSolverCache(
        be_circuit   = be_circuit,
        angles       = angles,
        degree       = degree,
        alpha        = alpha,
        kappa_eff    = kappa_eff,
        inner_config = inner_config,
    )


def _sweep_sequential(
    problem : PoissonProblem2D,
    u_prev  : np.ndarray,
    cache   : _RowSolverCache,
    N       : int,
) -> np.ndarray:
    """
    Perform one full sequential line-Jacobi sweep using QSVT.

    Iterates over all N interior rows, solving each 1-D sub-problem
    with the pre-cached block encoding and phase angles.

    Parameters
    ----------
    problem : PoissonProblem2D
    u_prev : np.ndarray, shape (N, N)
        Solution field from the previous iteration.
    cache : _RowSolverCache
    N : int

    Returns
    -------
    u_new : np.ndarray, shape (N, N)
    """
    u_new = np.zeros((N, N))

    for j in range(N):
        A_row, b_row = problem.get_row_system(j, u_prev)

        if np.linalg.norm(b_row) < _ZERO_RHS_TOL:
            u_new[:, j] = np.zeros(N)
            continue

        result       = qsvt_solve_system(A_row, b_row, cache.inner_config)
        u_new[:, j]  = result.u

    return u_new


def _sweep_parallel(
    problem   : PoissonProblem2D,
    u_prev    : np.ndarray,
    cache     : _RowSolverCache,
    N         : int,
    n_workers : int,
) -> np.ndarray:
    """
    Perform one full parallel line-Jacobi sweep using QSVT.

    The N row solves are submitted as independent tasks to a
    ProcessPoolExecutor. Each worker receives a copy of the row
    system (A_row, b_row) and the inner QSVT configuration.

    This parallelisation is effective when N is large relative to the
    process spawning overhead, typically N >= 8 on HPC hardware.
    On a laptop with N=4, sequential execution is faster.

    Parameters
    ----------
    problem : PoissonProblem2D
    u_prev : np.ndarray, shape (N, N)
    cache : _RowSolverCache
    N : int
    n_workers : int

    Returns
    -------
    u_new : np.ndarray, shape (N, N)
    """
    u_new = np.zeros((N, N))

    # Collect all row systems before submitting to workers.
    # This avoids passing the full PoissonProblem2D object (which may
    # not be picklable) to the worker processes.
    row_systems = []
    for j in range(N):
        A_row, b_row = problem.get_row_system(j, u_prev)
        row_systems.append((j, A_row, b_row))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                _solve_row_worker,
                j, A_row, b_row, cache.inner_config,
            ): j
            for j, A_row, b_row in row_systems
        }

        for future in as_completed(futures):
            j        = futures[future]
            u_row    = future.result()
            u_new[:, j] = u_row

    return u_new


def _solve_row_worker(
    j            : int,
    A_row        : np.ndarray,
    b_row        : np.ndarray,
    inner_config : QSVTConfig1D,
) -> np.ndarray:
    """
    Worker function for parallel row solving.

    Designed to be picklable and stateless so it can be submitted to
    a ProcessPoolExecutor. Returns the solution vector for row j.

    Parameters
    ----------
    j : int
        Row index (used only for zero-RHS detection logging).
    A_row : np.ndarray, shape (N, N)
    b_row : np.ndarray, shape (N,)
    inner_config : QSVTConfig1D

    Returns
    -------
    u_row : np.ndarray, shape (N,)
    """
    if np.linalg.norm(b_row) < _ZERO_RHS_TOL:
        return np.zeros(len(b_row))

    result = qsvt_solve_system(A_row, b_row, inner_config)
    return result.u


def _package_result(
    problem          : PoissonProblem2D,
    u                : np.ndarray,
    iterations       : int,
    converged        : bool,
    iteration_errors : list[float],
) -> SolverResult2D:
    """
    Compute the full-system residual via the tridiagonal matvec and
    package all outputs into a SolverResult2D.

    The residual is computed without forming the full N^2 x N^2 matrix,
    using the O(N^2) tridiagonal matvec from poisson_2d.py.

    Parameters
    ----------
    problem : PoissonProblem2D
    u : np.ndarray, shape (N, N)
    iterations : int
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
        solver            = "QSVT-2D",
        iterations        = iterations,
        converged         = converged,
        iteration_errors  = iteration_errors,
        euclidean_residual= residual,
    )


# -- Extended QSVTConfig1D with pre-computed angles -----------------------------

@dataclass
class _QSVTConfig1DWithAngles(QSVTConfig1D):
    """
    QSVTConfig extended with pre-computed phase angles and block encoding
    parameters.

    This internal subclass is used to pass pre-computed quantities from
    the row cache into qsvt_solve_system, bypassing the angle computation
    and block encoding construction steps that would otherwise be
    redundantly repeated for every row.

    Attributes
    ----------
    _precomputed_angles : np.ndarray or None
        Pre-computed QSP phase angles. If not None, qsvt_solve_system
        uses these directly rather than calling compute_inversion_angles.
    _precomputed_degree : int or None
        Pre-computed polynomial degree corresponding to _precomputed_angles.
    _precomputed_alpha : float or None
        Pre-computed subnormalisation factor from the block encoding.
    """

    _precomputed_angles : Optional[np.ndarray] = field(
        default=None, repr=False
    )
    _precomputed_degree : Optional[int]         = None
    _precomputed_alpha  : Optional[float]       = None