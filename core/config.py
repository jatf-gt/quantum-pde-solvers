"""
Defines the configuration structures for the quantum and classical simulation benchmarks.

This module encapsulates the execution parameters required for singular 
benchmark instances across 1D and 2D domains. It ensures rigorous input 
validation for grid dimensions, convergence criteria, and algorithmic 
precision prior to matrix instantiation or circuit generation.
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
            raise ValueError(f"System dimension N must be a positive power of 2, received {self.N}.")
        
        if self.source_fn not in SOURCE_FUNCTIONS:
            raise ValueError(
                f"Unrecognised source function '{self.source_fn}'. "
                f"Permitted identifiers: {list(SOURCE_FUNCTIONS.keys())}."
            )
            
        if self.epsilon <= 0:
            raise ValueError(f"Precision parameter epsilon must be strictly positive, received {self.epsilon}.")


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
        one-dimensional HHL sub-resolution phase of the line-Jacobi iterative cycle.
    source_fn : str
        Identifier for the 2D analytical source function.
    tol : float
        Convergence threshold for the line-Jacobi iterative method, defined by 
        the supremum norm of the sequential difference: max|u^{n+1} - u^n| < tol. 
        The primary reference specifies tol = 1e-8 throughout Section IV E.
    max_iter : int
        Absolute ceiling on the line-Jacobi iteration count. This parameter 
        prevents infinite divergence in pathological configurations, notably 
        systems governed by the discontinuous fH source function.
    bc_x0, bc_x1, bc_y0, bc_y1 : float
        Dirichlet boundary conditions applied to the respective domain edges 
        (x=0, x=1, y=0, y=1). Maintained as scalar floats for homogeneous 
        configurations; the 2D problem class assumes responsibility for handling 
        callables in non-homogeneous regimes.
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
            raise ValueError(f"N must be a positive power of 2, got N={self.N}.")
        if self.source_fn not in SOURCE_FUNCTIONS_2D:
            raise ValueError(
                f"Unknown 2D source function '{self.source_fn}'. "
                f"Valid options: {list(SOURCE_FUNCTIONS_2D.keys())}"
            )
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}.")
        if self.tol <= 0:
            raise ValueError(f"tol must be positive, got {self.tol}.")


# ── 2D Classical Configuration ────────────────────────────────────────────────

@dataclass
class ClassicalConfig2D:
    """
    Configuration parameters for purely classical two-dimensional resolutions.

    This structure defines system parameters explicitly for classical 
    execution, fully bypassing the HHL algorithm pipeline. Consequently, the 
    spatial resolution dimension (N) is not constrained to powers of two, as 
    amplitude encoding is omitted. 
    
    This configuration is strictly utilised for:
      1. Generating high-fidelity reference solutions via classical methodologies.
      2. Conducting condition number scaling analyses across arbitrary mesh sizes.
      3. Facilitating subsequent classical solvers independent of quantum feedback.

    Attributes
    ----------
    N : int
        Number of interior nodes spanning each spatial direction. Permitted 
        to be any strictly positive integer.
    source_fn : str
        Identifier for the 2D analytical source function.
    tol : float
        Convergence threshold for the line-Jacobi iterative method.
    max_iter : int
        Absolute ceiling on the line-Jacobi iteration count.
    bc_x0, bc_x1, bc_y0, bc_y1 : float
        Dirichlet boundary conditions applied to the respective domain edges.
    epsilon : float
        Retained exclusively for interface polymorphism with SimConfig2D, 
        ensuring the underlying 2D problem class processes either configuration 
        structure transparently. Unused computationally.
    """
    N:         int
    source_fn: str
    tol:       float = 1e-10
    max_iter:  int   = 5000
    bc_x0:     float = 0.0
    bc_x1:     float = 0.0
    bc_y0:     float = 0.0
    bc_y1:     float = 0.0
    epsilon:   float = 0.01

    def __post_init__(self) -> None:
        """Validates parameter constraints upon instantiation."""
        if self.N <= 0:
            raise ValueError(f"N must be a positive integer, got N={self.N}.")
        if self.source_fn not in SOURCE_FUNCTIONS_2D:
            raise ValueError(
                f"Unknown 2D source function '{self.source_fn}'. "
                f"Valid options: {list(SOURCE_FUNCTIONS_2D.keys())}"
            )
        if self.tol <= 0:
            raise ValueError(f"tol must be positive, got {self.tol}.")