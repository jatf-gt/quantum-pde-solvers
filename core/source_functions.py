"""
Defines the source functions (forcing terms) for the Poisson equation benchmarks.

This module implements the analytical forcing functions specified in Section IV 
of the primary reference literature. These functions are utilised to generate 
the right-hand side vector of the discretised linear system prior to classical 
or quantum resolution.
"""
from __future__ import annotations

from typing import Callable
import numpy as np


def f_sin(x: np.ndarray) -> np.ndarray:
    """Evaluates the sinusoidal source function, f_S(x) = sin(πx)."""
    return np.sin(np.pi * x)


def f_linear(x: np.ndarray) -> np.ndarray:
    """Evaluates the linear source function, f_L(x) = 10x."""
    return 10.0 * x


def f_heaviside(x: np.ndarray) -> np.ndarray:
    """
    Evaluates the modified Heaviside source function.
    
    Defined as a shifted unit step centred at the domain midpoint: 
    f_H(x) = 2H(x - 0.5) - 1.
    """
    return np.where(x >= 0.5, 1.0, -1.0)


# Dictionary mapping benchmark nomenclature to the corresponding functional implementations.
SOURCE_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "fS": f_sin,
    "fL": f_linear,
    "fH": f_heaviside,
}