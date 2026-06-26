"""
Defines the source functions (forcing terms) for the Poisson equation benchmarks.

This module implements the analytical forcing functions specified in Section IV 
of the primary reference literature for both 1D and 2D configurations. These 
functions are utilised to generate the right-hand side vector of the discretised 
linear system prior to classical or quantum resolution.
"""
from __future__ import annotations

from typing import Callable
import numpy as np


# ── 1D Source Functions ───────────────────────────────────────────────────────

def f_sin(x: np.ndarray) -> np.ndarray:
    """Evaluates the 1D sinusoidal source function: f_S(x) = sin(πx)."""
    return np.sin(np.pi * x)


def f_linear(x: np.ndarray) -> np.ndarray:
    """Evaluates the 1D linear source function: f_L(x) = 10x."""
    return 10.0 * x


def f_heaviside(x: np.ndarray) -> np.ndarray:
    """
    Evaluates the 1D modified Heaviside source function.
    
    Defined as a shifted unit step centred at the domain midpoint: 
    f_H(x) = 2H(x - 0.5) - 1.
    """
    return np.where(x >= 0.5, 1.0, -1.0)


# Dictionary mapping 1D benchmark nomenclature to corresponding functional implementations.
SOURCE_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "fS": f_sin,
    "fL": f_linear,
    "fH": f_heaviside,
}


# ── 2D Source Functions ───────────────────────────────────────────────────────

def f_sin_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluates the 2D sinusoidal source function: f_S(x,y) = 10 sin(2πx) cos(2πy)."""
    return 10.0 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)


def f_linear_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Evaluates the 2D linear source function: f_L(x,y) = 10x."""
    return 10.0 * x


def f_heaviside_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Evaluates the 2D modified Heaviside source function.

    Defined as: f_H(x,y) = 4 - 8·H(x - 0.5, y), where the discontinuous 
    Heaviside step evaluates solely along the x-coordinate (i.e., H(x,y) = H(x)).
    """
    return np.where(x >= 0.5, -4.0, 4.0)


# Dictionary mapping 2D benchmark nomenclature to corresponding functional implementations.
SOURCE_FUNCTIONS_2D: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "fS": f_sin_2d,
    "fL": f_linear_2d,
    "fH": f_heaviside_2d,
}


# ── HET Plasma Charge Density Profiles ────────────────────────────────────────

def het_gaussian(
    x:     np.ndarray,
    rho_0: float,
    x_ion: float,
    sigma: float,
    alpha: float,
) -> np.ndarray:
    """
    Evaluates a smooth Gaussian charge density profile.

    This function models a well-behaved, localised ionisation region within 
    the Hall Effect Thruster (HET) discharge channel.

    Formulation:
        f(x) = -α · ρ_0 · exp(-(x - x_ion)² / σ²)

    The negative sign adheres to the standard Poisson convention for plasmas:
        d²φ/dx² = -α(n_i - n_e) = f(x)

    The dimensionless prefactor α = (L/λ_D)² dictates the scaling between 
    macroscopic channel dimensions and the local Debye length, heavily 
    dominating the condition number of the discretised operator.
    """
    return -alpha * rho_0 * np.exp(-((x - x_ion)**2) / sigma**2)


def het_linear(
    x:     np.ndarray,
    rho_0: float,
    alpha: float,
) -> np.ndarray:
    """
    Evaluates a linear charge density profile.

    Formulation:
        f(x) = -α · ρ_0 · x

    This simplified profile provides a rigorous baseline for algorithmic 
    verification, as it possesses a closed-form analytical solution under 
    homogeneous Dirichlet boundary conditions:
        φ(x) = α · ρ_0 · x(1 - x²) / 6
    """
    return -alpha * rho_0 * x


def het_step(
    x:     np.ndarray,
    rho_0: float,
    x_ion: float,
    alpha: float,
) -> np.ndarray:
    """
    Evaluates a discontinuous step charge density profile.

    Formulation:
        f(x) = -α · ρ_0 · sign(x - x_ion)

    This distribution provides a physically motivated representation of a 
    sharp ionisation boundary, characterising a net positive charge upstream 
    of the primary ionisation zone and a net negative charge downstream.
    """
    return -alpha * rho_0 * np.sign(x - x_ion)


# Dictionary mapping HET benchmark nomenclature to corresponding implementations.
HET_SOURCE_FUNCTIONS = {
    "gaussian": het_gaussian,
    "linear":   het_linear,
    "step":     het_step,
}