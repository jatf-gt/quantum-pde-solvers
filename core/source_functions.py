"""
Source functions (forcing terms) for the Poisson equation benchmarks.

Implements the analytical forcing functions specified in Section IV of the
primary reference literature, for both the 1D and 2D configurations, together
with the charge density profiles of the Hall Effect Thruster (HET) application.
These generate the right-hand side vector of the discretised linear system prior
to any classical or quantum solve.

Each family is exposed through a registry dictionary (``SOURCE_FUNCTIONS``,
``SOURCE_FUNCTIONS_2D``, ``HET_SOURCE_FUNCTIONS``) keyed by the benchmark
nomenclature; ``core.config`` validates configuration identifiers against these
keys, so a source added here becomes selectable without further registration.
"""
from __future__ import annotations

from typing import Callable
import numpy as np


# -- 1D Source Functions -------------------------------------------------------

def f_sin(x: np.ndarray) -> np.ndarray:
    """
    Evaluates the 1D sinusoidal source function, f_S(x) = sin(πx).

    Parameters
    ----------
    x : np.ndarray
        Length-N vector of spatial coordinates on [0, 1].

    Returns
    -------
    f : np.ndarray
        Length-N vector of source values.
    """
    return np.sin(np.pi * x)


def f_linear(x: np.ndarray) -> np.ndarray:
    """
    Evaluates the 1D linear source function, f_L(x) = 10x.

    Parameters
    ----------
    x : np.ndarray
        Length-N vector of spatial coordinates on [0, 1].

    Returns
    -------
    f : np.ndarray
        Length-N vector of source values.
    """
    return 10.0 * x


def f_heaviside(x: np.ndarray) -> np.ndarray:
    """
    Evaluates the 1D modified Heaviside source function.

    A shifted unit step centred at the domain midpoint:

        f_H(x) = 2H(x - 0.5) - 1

    Parameters
    ----------
    x : np.ndarray
        Length-N vector of spatial coordinates on [0, 1].

    Returns
    -------
    f : np.ndarray
        Length-N vector of source values, taking -1 upstream of x = 0.5 and
        +1 downstream. The discontinuity limits the attainable order of
        convergence of the discretisation.
    """
    return np.where(x >= 0.5, 1.0, -1.0)


# Maps 1D benchmark nomenclature to the corresponding implementation.
SOURCE_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "fS": f_sin,
    "fL": f_linear,
    "fH": f_heaviside,
}


# -- 2D Source Functions -------------------------------------------------------

def f_sin_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Evaluates the 2D sinusoidal source function.

        f_S(x, y) = 10 sin(2πx) cos(2πy)

    Parameters
    ----------
    x, y : np.ndarray
        (N, N) meshgrid coordinate arrays over the interior nodes.

    Returns
    -------
    f : np.ndarray
        (N, N) source field.
    """
    return 10.0 * np.sin(2.0 * np.pi * x) * np.cos(2.0 * np.pi * y)


def f_linear_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Evaluates the 2D linear source function, f_L(x, y) = 10x.

    Parameters
    ----------
    x, y : np.ndarray
        (N, N) meshgrid coordinate arrays over the interior nodes. The source
        is independent of y, which is accepted only for signature uniformity
        across the registry.

    Returns
    -------
    f : np.ndarray
        (N, N) source field.
    """
    return 10.0 * x


def f_heaviside_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Evaluates the 2D modified Heaviside source function.

        f_H(x, y) = 4 - 8·H(x - 0.5)

    The discontinuous step is evaluated solely along the x-coordinate, so that
    H(x, y) ≡ H(x) and the source is constant along each y-line.

    Parameters
    ----------
    x, y : np.ndarray
        (N, N) meshgrid coordinate arrays over the interior nodes.

    Returns
    -------
    f : np.ndarray
        (N, N) source field, taking +4 upstream of x = 0.5 and -4 downstream.
    """
    return np.where(x >= 0.5, -4.0, 4.0)


# Maps 2D benchmark nomenclature to the corresponding implementation.
SOURCE_FUNCTIONS_2D: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "fS": f_sin_2d,
    "fL": f_linear_2d,
    "fH": f_heaviside_2d,
}


# -- HET Plasma Charge Density Profiles ----------------------------------------

def het_gaussian(
    x:     np.ndarray,
    rho_0: float,
    x_ion: float,
    sigma: float,
    alpha: float,
) -> np.ndarray:
    """
    Evaluates a smooth Gaussian charge density profile.

    Models a well-behaved, localised ionisation region within the Hall Effect
    Thruster discharge channel:

        f(x) = -α · ρ_0 · exp(-(x - x_ion)² / σ²)

    The negative sign adheres to the standard Poisson convention for plasmas:

        d²φ/dx² = -α(n_i - n_e) = f(x)

    Parameters
    ----------
    x : np.ndarray
        Length-N vector of non-dimensional axial coordinates on [0, 1].
    rho_0 : float
        Non-dimensional peak charge density of the ionisation region.
    x_ion : float
        Non-dimensional axial location of the ionisation peak.
    sigma : float
        Non-dimensional Gaussian width of the ionisation region.
    alpha : float
        Non-dimensional source scaling parameter, α = (L / λ_D)².

    Returns
    -------
    f : np.ndarray
        Length-N vector of source values.

    Notes
    -----
    α sets the scaling between the macroscopic channel dimension and the local
    Debye length. For the default ``HETConfig`` (L = 2.5 cm, T_e = 20 eV,
    n_0 = 1×10¹⁷ m⁻³) it evaluates to α ≈ 5.65×10⁴, so the physically scaled
    right-hand side is large in magnitude even where φ remains O(1). This is
    why the HHL implementation recovers its proportionality constant against
    the normalised system A/‖A‖₂ rather than against A directly.
    """
    return -alpha * rho_0 * np.exp(-((x - x_ion)**2) / sigma**2)


def het_linear(
    x:     np.ndarray,
    rho_0: float,
    alpha: float,
) -> np.ndarray:
    """
    Evaluates a linear charge density profile.

        f(x) = -α · ρ_0 · x

    This simplified profile provides a baseline for algorithmic verification,
    possessing a closed-form analytical solution under homogeneous Dirichlet
    boundary conditions:

        φ(x) = α · ρ_0 · x(1 - x²) / 6

    implemented as ``core.exact_solutions.het_phi_linear``.

    Parameters
    ----------
    x : np.ndarray
        Length-N vector of non-dimensional axial coordinates on [0, 1].
    rho_0 : float
        Non-dimensional charge density scale of the linear profile.
    alpha : float
        Non-dimensional source scaling parameter, α = (L / λ_D)².

    Returns
    -------
    f : np.ndarray
        Length-N vector of source values.
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

        f(x) = -α · ρ_0 · sign(x - x_ion)

    A physically motivated representation of a sharp ionisation boundary,
    characterising a net positive charge upstream of the primary ionisation
    zone and a net negative charge downstream.

    Parameters
    ----------
    x : np.ndarray
        Length-N vector of non-dimensional axial coordinates on [0, 1].
    rho_0 : float
        Non-dimensional charge density magnitude either side of the boundary.
    x_ion : float
        Non-dimensional axial location of the ionisation boundary.
    alpha : float
        Non-dimensional source scaling parameter, α = (L / λ_D)².

    Returns
    -------
    f : np.ndarray
        Length-N vector of source values.
    """
    return -alpha * rho_0 * np.sign(x - x_ion)


# Maps HET profile nomenclature to the corresponding implementation.
HET_SOURCE_FUNCTIONS = {
    "gaussian": het_gaussian,
    "linear":   het_linear,
    "step":     het_step,
}
