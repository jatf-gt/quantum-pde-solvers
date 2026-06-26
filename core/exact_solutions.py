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


# ── HET Analytical Solutions (Homogeneous Boundaries) ─────────────────────────

def het_phi_linear(
    x:     np.ndarray,
    rho_0: float,
    alpha: float,
) -> np.ndarray:
    """
    Evaluates the exact analytical solution for the linear HET charge density 
    profile under homogeneous Dirichlet boundary conditions (φ(0) = φ(1) = 0).

    Mathematical Derivation:
        Governing PDE: d²φ/dx² = -α · ρ_0 · x
        
        Integrating sequentially and applying φ(0) = 0:
            φ'(x) = -α · ρ_0 · x² / 2 + C_1
            φ(x)  = -α · ρ_0 · x³ / 6 + C_1 · x
            
        Applying the terminal boundary constraint φ(1) = 0:
            0 = -α · ρ_0 / 6 + C_1  =>  C_1 = α · ρ_0 / 6
            
        Yielding the final closed-form spatial potential:
            φ(x) = α · ρ_0 · x · (1 - x²) / 6
    """
    return alpha * rho_0 * x * (1.0 - x**2) / 6.0


# Dictionary mapping HET exact solution models.
HET_EXACT_SOLUTIONS = {
    "linear": het_phi_linear,
}