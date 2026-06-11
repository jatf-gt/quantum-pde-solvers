"""
Defines the configuration structures for the quantum simulation benchmarks.

This module encapsulates the execution parameters required for a single 
benchmark instance, ensuring rigorous input validation for grid dimensions 
and algorithmic precision prior to matrix instantiation or circuit generation.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.source_functions import SOURCE_FUNCTIONS


@dataclass
class SimConfig1D:
    """
    Configuration parameters for a one-dimensional benchmark simulation.

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