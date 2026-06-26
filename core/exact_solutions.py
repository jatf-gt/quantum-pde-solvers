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


# -- 2-D analytical solutions -------------------------------------------------

def phi_2d_sinusoidal(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Analytical solution for the 2-D Poisson equation with sinusoidal
    source term and homogeneous Dirichlet boundary conditions.

    Problem:
        ∂²φ̃/∂x̃² + ∂²φ̃/∂ỹ² = -2π² sin(πx̃) sin(πỹ)
        φ̃ = 0 on ∂[0,1]²

    Solution:
        φ̃(x̃, ỹ) = sin(πx̃) sin(πỹ)

    Derived by substitution: the Laplacian of sin(πx̃)sin(πỹ) is
    -π²sin(πx̃)sin(πỹ) - π²sin(πx̃)sin(πỹ) = -2π²sin(πx̃)sin(πỹ).

    Physical interpretation for HET modelling
    ------------------------------------------
    With the source term interpreted as -α·δñ(x̃,ỹ), this corresponds
    to a separable charge density profile:

        δñ(x̃,ỹ) = (2π²/α) sin(πx̃) sin(πỹ)

    which peaks at the domain centre and vanishes at all boundaries,
    approximating the quasi-neutral bulk plasma in a 2-D cross-section
    of the discharge channel.

    Parameters
    ----------
    X : np.ndarray, shape (N, N)
        Meshgrid of x-coordinates at interior nodes.
    Y : np.ndarray, shape (N, N)
        Meshgrid of y-coordinates at interior nodes.

    Returns
    -------
    phi : np.ndarray, shape (N, N)
        Analytical potential field at interior nodes.
    """
    return np.sin(np.pi * X) * np.sin(np.pi * Y)


def E_field_2d_sinusoidal(
    X    : np.ndarray,
    Y    : np.ndarray,
    phi_0: float,
    L_x  : float,
    L_y  : float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Analytical electric field components for the sinusoidal 2-D Poisson
    solution, converted to physical units.

    Derivation:
        Ẽ_x = -∂φ̃/∂x̃ = -π cos(πx̃) sin(πỹ)
        Ẽ_y = -∂φ̃/∂ỹ = -π sin(πx̃) cos(πỹ)

    Physical conversion:
        E_x [V/m] = Ẽ_x · φ_0 / L_x
        E_y [V/m] = Ẽ_y · φ_0 / L_y

    Parameters
    ----------
    X, Y : np.ndarray, shape (N, N)
        Meshgrid coordinates at interior nodes.
    phi_0 : float
        Thermal voltage [V]: φ_0 = T_e [eV].
    L_x : float
        Axial channel length [m].
    L_y : float
        Radial channel height [m].

    Returns
    -------
    E_x : np.ndarray, shape (N, N)
        Axial electric field component [V/m].
    E_y : np.ndarray, shape (N, N)
        Radial electric field component [V/m].
    """
    E_x = -np.pi * np.cos(np.pi * X) * np.sin(np.pi * Y) * phi_0 / L_x
    E_y = -np.pi * np.sin(np.pi * X) * np.cos(np.pi * Y) * phi_0 / L_y
    return E_x, E_y


EXACT_SOLUTIONS_2D = {
    "sinusoidal": phi_2d_sinusoidal,
}