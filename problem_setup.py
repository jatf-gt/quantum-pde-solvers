"""
problem_setup.py
----------------
Builds the TST matrix A and the right-hand-side vector b for the
1D Poisson equation, following the paper's Eq. (5) exactly.

Also provides the grid coordinates and a thin wrapper that packages
everything the solvers need in one place.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import diags

from config import SimConfig, SOURCE_FUNCTIONS


# ── Grid construction ─────────────────────────────────────────────────────────

def build_grid(N: int) -> tuple[np.ndarray, float]:
    """
    Return the N interior node coordinates and the mesh spacing Δx.

    The domain is [0, 1].  Boundary nodes x=0 and x=1 are NOT included
    because they are absorbed into the RHS via the Dirichlet BCs.

    Parameters
    ----------
    N : int
        Number of interior nodes (must be a power of 2 per SimConfig).

    Returns
    -------
    x : np.ndarray, shape (N,)
        Interior node coordinates x_i = i·Δx, i = 1, …, N.
    dx : float
        Mesh spacing Δx = 1 / (N + 1).
    """
    dx = 1.0 / (N + 1)
    x = np.arange(1, N + 1) * dx  # x_1, x_2, ..., x_N
    return x, dx


# ── TST matrix ────────────────────────────────────────────────────────────────

def build_tst_matrix(N: int) -> np.ndarray:
    """
    Build the N×N Toeplitz Symmetric Tridiagonal (TST) matrix for the
    1D Poisson operator with second-order centred finite differences.

    The matrix has main diagonal −2 and off-diagonals +1, i.e.
    a = −2, b = 1 in the paper's notation.  It does NOT include the
    1/Δx² prefactor; that is folded into the RHS instead (see build_rhs).

    Returns a dense NumPy array — N is small enough (≤ 32) that dense
    storage is fine and avoids sparse/dense conversion headaches later.
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
    Assemble the RHS vector b following the paper's Eq. (5).

    The interior equations are:
        u_{i+1} - 2u_i + u_{i-1} = Δx² f(x_i)

    After incorporating the Dirichlet BCs u(0) = α, u(1) = β, the
    first and last entries pick up a correction term:
        b_1   = Δx² f(x_1) - α
        b_N   = Δx² f(x_N) - β
        b_i   = Δx² f(x_i)   for 2 ≤ i ≤ N-1

    Parameters
    ----------
    x            : interior node coordinates from build_grid
    dx           : mesh spacing from build_grid
    source_fn_key: one of 'fS', 'fL', 'fH'
    alpha        : Dirichlet BC at x = 0
    beta         : Dirichlet BC at x = 1
    """
    f = SOURCE_FUNCTIONS[source_fn_key]
    b = dx**2 * f(x)

    # Absorb boundary conditions into the first and last entries.
    b[0]  -= alpha
    b[-1] -= beta

    return b


# ── Condition number utility ──────────────────────────────────────────────────

def condition_number(A: np.ndarray) -> float:
    """
    Return the 2-norm condition number κ(A) = |λ_max| / |λ_min|.

    For the 1D Poisson TST matrix this scales as O(N²), which is the
    key quantity governing HHL circuit depth (l-register width).
    """
    eigenvalues = np.linalg.eigvalsh(A)
    abs_eigs = np.abs(eigenvalues)
    return float(abs_eigs.max() / abs_eigs.min())


# ── Packaged problem ──────────────────────────────────────────────────────────

class PoissonProblem1D:
    """
    Convenience container that holds all discretised quantities for one
    benchmark run.  Constructed directly from a SimConfig.

    Attributes
    ----------
    config  : SimConfig
    x       : interior node coordinates
    dx      : mesh spacing
    A       : TST system matrix (N×N dense array)
    b       : RHS vector (length N)
    kappa   : condition number of A
    """

    def __init__(self, cfg: SimConfig) -> None:
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
        """One-line summary for logging."""
        return (
            f"N={self.config.N}, f={self.config.source_fn}, "
            f"α={self.config.alpha}, β={self.config.beta}, "
            f"ε={self.config.epsilon:.4g}, "
            f"κ(A)={self.kappa:.2f}"
        )