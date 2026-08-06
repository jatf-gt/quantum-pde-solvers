"""
Constructs the discretised 1D Poisson boundary value problem.

Generates the Toeplitz Symmetric Tridiagonal (TST) system matrix and the
right-hand side vector for the one-dimensional Poisson equation, following the
formulation of Equation (5) of the primary reference. Grid generation and system
assembly are packaged into the single data structure required by the classical
and quantum solvers.
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
    excluded, their contributions being absorbed into the right-hand side vector
    via the Dirichlet boundary conditions.

    Parameters
    ----------
    N : int
        Number of interior nodes (must be a power of 2, per SimConfig1D).

    Returns
    -------
    x : np.ndarray
        Length-N vector of interior node coordinates x_i = i·Δx, for i = 1, …, N.
    dx : float
        Mesh spacing, Δx = 1 / (N + 1).
    """
    dx = 1.0 / (N + 1)
    x = np.arange(1, N + 1) * dx
    return x, dx


# ── TST matrix ────────────────────────────────────────────────────────────────

def build_tst_matrix(N: int) -> np.ndarray:
    """
    Constructs the N×N Toeplitz Symmetric Tridiagonal (TST) matrix for the 1D
    Poisson operator, using second-order centred finite differences.

    The matrix has a main diagonal of -2 and off-diagonals of +1 (a = -2, b = 1
    in the reference notation). The 1/Δx² scaling factor is omitted here and
    folded into the right-hand side instead.

    Parameters
    ----------
    N : int
        System dimension (number of interior nodes).

    Returns
    -------
    A : np.ndarray
        N×N dense TST matrix.

    Notes
    -----
    The output is deliberately dense. The benchmark sweeps reach N = 64, at
    which the matrix occupies 32 kB, so sparse storage would buy nothing and
    would force sparse-to-dense conversions at every quantum solver interface.
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

    The interior equations are

        u_{i+1} - 2u_i + u_{i-1} = Δx² f(x_i)

    and, once the Dirichlet conditions u(0) = α and u(1) = β are incorporated,
    the boundary-adjacent nodes carry a correction term:

        b_1 = Δx² f(x_1) - α
        b_N = Δx² f(x_N) - β
        b_i = Δx² f(x_i)        for 2 ≤ i ≤ N-1

    Parameters
    ----------
    x : np.ndarray
        Length-N vector of interior node coordinates, from build_grid.
    dx : float
        Mesh spacing, from build_grid.
    source_fn_key : str
        Registry key selecting the analytical forcing function.
    alpha : float
        Dirichlet boundary condition at x = 0.
    beta : float
        Dirichlet boundary condition at x = 1.

    Returns
    -------
    b : np.ndarray
        Length-N right-hand side vector with boundary data absorbed.
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

    Parameters
    ----------
    A : np.ndarray
        N×N symmetric matrix.

    Returns
    -------
    kappa : float
        Spectral condition number.

    Notes
    -----
    For the 1D Poisson TST matrix κ scales as O(N²), and it is the primary
    quantity dictating the required depth of the HHL quantum circuit (the width
    of the l-register). This unfavourable growth is what motivates the
    line-decomposed 2D/3D formulation, where the strip operator instead has
    κ → 3⁻ (2D) or κ → 2⁻ (3D) as N → ∞.
    """
    eigenvalues = np.linalg.eigvalsh(A)
    abs_eigs = np.abs(eigenvalues)
    return float(abs_eigs.max() / abs_eigs.min())


# ── Packaged problem ──────────────────────────────────────────────────────────

class PoissonProblem1D:
    """
    Bundles the discretised system for a single 1D benchmark instance.

    Attributes
    ----------
    config : SimConfig1D
        Configuration parameters defining the problem instance.
    x : np.ndarray
        Length-N vector of interior node coordinates.
    dx : float
        Uniform mesh spacing, Δx = 1/(N+1).
    A : np.ndarray
        N×N dense TST system matrix.
    b : np.ndarray
        Length-N assembled right-hand side vector.
    kappa : float
        2-norm condition number of A.
    """

    def __init__(self, cfg: SimConfig1D) -> None:
        """
        Assembles the grid, system matrix and right-hand side from a config.

        Parameters
        ----------
        cfg : SimConfig1D
            Validated configuration for this benchmark instance.
        """
        self.config = cfg
        self.x, self.dx = build_grid(cfg.N)
        self.A = build_tst_matrix(cfg.N)
        self.b = build_rhs(
            self.x, self.dx,
            cfg.source_fn, cfg.alpha, cfg.beta,
        )
        self.kappa = condition_number(self.A)

    def summary(self) -> str:
        """Returns a one-line summary of the discretised system."""
        return (
            f"N={self.config.N}, f={self.config.source_fn}, "
            f"α={self.config.alpha}, β={self.config.beta}, "
            f"ε={self.config.epsilon:.4g}, "
            f"κ(A)={self.kappa:.2f}"
        )
