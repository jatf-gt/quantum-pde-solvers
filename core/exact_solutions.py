"""
Defines the analytical solutions for the 1D Poisson equation benchmarks.

This module implements the exact analytical solutions corresponding to the
source functions defined in Section IV of the primary reference literature,
assuming homogeneous Dirichlet boundary conditions, u(0) = u(1) = 0.
"""
from __future__ import annotations

from typing import Callable
import numpy as np


def u_fS(x: np.ndarray) -> np.ndarray:
    """Evaluates the exact solution for the sinusoidal source: u(x) = -sin(πx) / π²."""
    return -np.sin(np.pi * x) / np.pi**2


def u_fL(x: np.ndarray) -> np.ndarray:
    """Evaluates the exact solution for the linear source: u(x) = 5x(x² - 1) / 3.
    u'' = 10x  →  u = 5x(x² − 1) / 3.
    Derived by integrating twice and enforcing u(0) = u(1) = 0."""
    return 5.0 * x * (x**2 - 1.0) / 3.0


def u_fH(x: np.ndarray) -> np.ndarray:
    """
    Evaluates the exact piecewise solution for the modified Heaviside source.
    
    x < 0.5  : u(x) = -x²/2 + x/4
    x >= 0.5 : u(x) = x²/2 - 3x/4 + 1/4

    Obtained by integrating each piece of fH, then matching continuity of u
    and u' at x = 0.5, together with u(0) = u(1) = 0.
    """
    return np.where(
        x < 0.5,
        -x**2 / 2.0 + x / 4.0,
        x**2 / 2.0 - 3.0 * x / 4.0 + 1.0 / 4.0,
    )


# Dictionary mapping benchmark nomenclature to the corresponding analytical solutions.
EXACT_SOLUTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "fS": u_fS,
    "fL": u_fL,
    "fH": u_fH,
}