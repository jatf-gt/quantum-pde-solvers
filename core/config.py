"""
Configuration structures for the quantum and classical simulation benchmarks.

Defines the execution parameters for an individual benchmark instance in 1D and
2D, together with validation of the grid dimension, convergence criteria and
algorithmic precision. Validation is sited in ``__post_init__`` rather than at
the call sites, so that an invalid configuration is rejected at construction and
cannot propagate into matrix assembly or circuit generation.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.source_functions import SOURCE_FUNCTIONS, SOURCE_FUNCTIONS_2D


# ── 1D Quantum Configuration ──────────────────────────────────────────────────

@dataclass
class SimConfig1D:
    """
    Configuration parameters for a one-dimensional quantum benchmark simulation.

    Attributes
    ----------
    N : int
        System matrix dimension (number of interior spatial nodes). Must be a
        power of two to accommodate quantum amplitude encoding via log₂(N) qubits.
    epsilon : float
        Precision parameter governing the Trotter approximation within the
        Hamiltonian simulation phase. Smaller values reduce discretisation
        error at the cost of increased circuit depth.
    source_fn : str
        Identifier for the analytical source function (e.g., 'fS', 'fL', 'fH').
    alpha : float
        Dirichlet boundary condition at the left boundary, u(0). Default is 0.0.
    beta : float
        Dirichlet boundary condition at the right boundary, u(1). Default is 0.0.
    """

    N: int
    epsilon: float
    source_fn: str
    alpha: float = 0.0
    beta: float = 0.0

    def __post_init__(self) -> None:
        """Validates parameter constraints upon instantiation."""
        if self.N <= 0 or (self.N & (self.N - 1)) != 0:
            raise ValueError(
                f"System dimension N must be a positive power of 2, received {self.N}."
            )

        if self.source_fn not in SOURCE_FUNCTIONS:
            raise ValueError(
                f"Unrecognised source function '{self.source_fn}'. "
                f"Permitted identifiers: {list(SOURCE_FUNCTIONS.keys())}."
            )

        if self.epsilon <= 0:
            raise ValueError(
                f"Precision parameter epsilon must be strictly positive, "
                f"received {self.epsilon}."
            )


# ── 2D Quantum Configuration ──────────────────────────────────────────────────

@dataclass
class SimConfig2D:
    """
    Configuration parameters for a two-dimensional quantum benchmark simulation.

    Attributes
    ----------
    N : int
        System resolution parameter governing the interior mesh. The domain
        [0, 1]² is discretised into (N+1) intervals along each spatial axis,
        yielding N² interior unknowns. N must be a positive power of two to
        accommodate quantum amplitude encoding, analogous to the 1D formulation.
    epsilon : float
        Precision parameter governing the Trotter approximation within each
        one-dimensional HHL sub-solve of the line-Jacobi iterative cycle.
    source_fn : str
        Identifier for the 2D analytical source function.
    tol : float
        Convergence threshold for the line-Jacobi iterative method, defined by
        the supremum norm of the sequential difference: max|u⁽ⁿ⁺¹⁾ - u⁽ⁿ⁾| < tol.
        The primary reference specifies tol = 1e-8 throughout Section IV E.
    max_iter : int
        Absolute ceiling on the line-Jacobi iteration count, bounding the cost of
        pathological configurations that fail to converge — notably systems
        governed by the discontinuous fH source function.
    bc_x0, bc_x1, bc_y0, bc_y1 : float
        Dirichlet boundary conditions applied to the respective domain edges
        (x=0, x=1, y=0, y=1). Held as scalar floats for homogeneous
        configurations; handling of callables in non-homogeneous regimes is the
        responsibility of the 2D problem class.

    Notes
    -----
    The line-Jacobi terminology is deliberate: the sweeps that replicate the
    published 2D figures drive ``solvers.outer.solve`` with ``scheme="jacobi"``,
    which reproduces the original line-Jacobi loop exactly. ``tol`` and
    ``max_iter`` are the parameters of that outer iteration, not of any inner
    1D solve.
    """
    N:          int
    epsilon:    float
    source_fn:  str
    tol:        float = 1e-8
    max_iter:   int   = 500
    bc_x0:      float = 0.0   # u(0, y) = bc_x0
    bc_x1:      float = 0.0   # u(1, y) = bc_x1
    bc_y0:      float = 0.0   # u(x, 0) = bc_y0
    bc_y1:      float = 0.0   # u(x, 1) = bc_y1

    def __post_init__(self) -> None:
        """Validates parameter constraints upon instantiation."""
        if self.N <= 0 or (self.N & (self.N - 1)) != 0:
            raise ValueError(
                f"System resolution N must be a positive power of 2, received {self.N}."
            )
        if self.source_fn not in SOURCE_FUNCTIONS_2D:
            raise ValueError(
                f"Unrecognised 2D source function '{self.source_fn}'. "
                f"Permitted identifiers: {list(SOURCE_FUNCTIONS_2D.keys())}."
            )
        if self.epsilon <= 0:
            raise ValueError(
                f"Precision parameter epsilon must be strictly positive, "
                f"received {self.epsilon}."
            )
        if self.tol <= 0:
            raise ValueError(
                f"Convergence tolerance tol must be strictly positive, "
                f"received {self.tol}."
            )
