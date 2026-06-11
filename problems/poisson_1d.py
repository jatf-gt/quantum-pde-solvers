"""
Constructs the discretised 1D Poisson boundary value problem.

This module generates the Toeplitz Symmetric Tridiagonal (TST) system matrix 
and the right-hand side vector for the one-dimensional Poisson equation, 
adhering strictly to the formulation provided in Equation (5) of the primary 
reference. It encapsulates the grid generation and system assembly into a 
unified data structure required by subsequent classical and quantum solvers.
"""
from __future__ import annotations

import numpy as np

from core.config import SimConfig1D
from core.source_functions import SOURCE_FUNCTIONS

# ── Grid construction ─────────────────────────────────────────────────────────

def build_grid(N: int) -> tuple[np.ndarray, float]:
    """
    Computes the interior node coordinates and the spatial mesh spacing, Δx.

    The continuous domain spans [0, 1]. Boundary nodes at x=0 and x=1 are 
    excluded, as their contributions are assimilated into the right-hand side 
    vector via Dirichlet boundary conditions.

    Parameters
    ----------
    N : int
        Number of interior nodes (must be a power of 2, per SimConfig1D).

    Returns
    -------
    x : np.ndarray
        Interior node coordinates x_i = i·Δx, for i = 1, …, N.
    dx : float
        Mesh spacing, defined as Δx = 1 / (N + 1).
    """
    dx = 1.0 / (N + 1)
    x = np.arange(1, N + 1) * dx
    return x, dx


# ── TST matrix ────────────────────────────────────────────────────────────────

def build_tst_matrix(N: int) -> np.ndarray:
    """
    Constructs the N×N Toeplitz Symmetric Tridiagonal (TST) matrix for the
    1D Poisson operator utilising second-order centred finite differences.

    The matrix possesses a main diagonal of -2 and off-diagonals of +1 
    (a = -2, b = 1 in the reference notation). The 1/Δx² scaling factor is 
    omitted from the matrix and instead integrated into the right-hand side.

    The output is a dense array. System dimensions are sufficiently constrained 
    (N <= 32) to render sparse matrices unnecessary, precluding the need for 
    subsequent sparse-to-dense data conversions.
    """
    diag_main = -2.0 * np.ones(N)
    diag_off  =  1.0 * np.ones(N - 1)
    A = (
        np.diag(diag_main)
        + np.diag(diag_off, k=1)
        + np.diag(diag_off, k=-1)
    )
    return A


# ── Right-hand side ───────────────────────────────────────────────────────────

def build_rhs(
    x: np.ndarray,
    dx: float,
    source_fn_key: str,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Assembles the right-hand side vector b corresponding to Equation (5).

    The interior equations are given by:
        u_{i+1} - 2u_i + u_{i-1} = Δx² f(x_i)

    Following the incorporation of Dirichlet boundary conditions u(0) = α 
    and u(1) = β, the boundary-adjacent nodes necessitate a correction term:
        b_1   = Δx² f(x_1) - α
        b_N   = Δx² f(x_N) - β
        b_i   = Δx² f(x_i)   for 2 <= i <= N-1

    Parameters
    ----------
    x : np.ndarray
        Interior node coordinates derived from build_grid.
    dx : float
        Mesh spacing derived from build_grid.
    source_fn_key : str
        Dictionary key specifying the analytical forcing function.
    alpha : float
        Dirichlet boundary condition evaluated at x = 0.
    beta : float
        Dirichlet boundary condition evaluated at x = 1.
    """
    f = SOURCE_FUNCTIONS[source_fn_key]
    b = dx**2 * f(x)

    # Absorb boundary conditions into the terminal entries.
    b[0]  -= alpha
    b[-1] -= beta

    return b


# ── Condition number utility ──────────────────────────────────────────────────

def condition_number(A: np.ndarray) -> float:
    """
    Computes the 2-norm condition number, κ(A) = |λ_max| / |λ_min|.

    For the 1D Poisson TST matrix, this parameter scales asymptotically as 
    O(N²). It represents the primary metric dictating the requisite depth 
    of the HHL quantum circuit (l-register width).
    """
    eigenvalues = np.linalg.eigvalsh(A)
    abs_eigs = np.abs(eigenvalues)
    return float(abs_eigs.max() / abs_eigs.min())


# ── Packaged problem ──────────────────────────────────────────────────────────

class PoissonProblem1D:
    """
    Data structure encapsulating all discretised parameters for a singular 
    benchmark execution.

    Attributes
    ----------
    config : SimConfig1D
        Base configuration parameters defining the problem instance.
    x : np.ndarray
        Interior node spatial coordinates.
    dx : float
        Uniform mesh spacing.
    A : np.ndarray
        TST system matrix structured as an N×N dense array.
    b : np.ndarray
        Assembled right-hand side vector of length N.
    kappa : float
        Calculated 2-norm condition number of matrix A.
    """

    def __init__(self, cfg: SimConfig1D) -> None:
        self.config = cfg
        self.x, self.dx = build_grid(cfg.N)
        self.A = build_tst_matrix(cfg.N)
        self.b = build_rhs(
            self.x, self.dx,
            cfg.source_fn, cfg.alpha, cfg.beta,
        )
        self.kappa = condition_number(self.A)

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Generates a concise summary string detailing the system configuration."""
        return (
            f"N={self.config.N}, f={self.config.source_fn}, "
            f"α={self.config.alpha}, β={self.config.beta}, "
            f"ε={self.config.epsilon:.4g}, "
            f"κ(A)={self.kappa:.2f}"
        )