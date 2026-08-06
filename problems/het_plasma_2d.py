"""
Assembles the discretised 2D Hall Effect Thruster plasma Poisson problem.

Physical model
--------------
The non-dimensionalised 2D Poisson equation for the electrostatic potential φ̃
in the discharge channel reads

    ∂²φ̃/∂x̃² + ∂²φ̃/∂ỹ² = −α · δñ(x̃, ỹ)

where

    x̃ = x/L_x      axial coordinate (normalised by channel length)
    ỹ = y/L_y      radial coordinate (normalised by channel height)
    α  = L_x²/λ_D²  dimensionless Debye scaling parameter
    δñ = (n_i − n_e)/n_0   non-dimensional net charge density

Dirichlet data are imposed on all four edges:

    φ̃(0,  ỹ) = α_bc = V_discharge/φ_0   (anode)
    φ̃(1,  ỹ) = 0                         (cathode)
    φ̃(x̃, 0) = 0                         (inner wall, grounded)
    φ̃(x̃, 1) = 0                         (outer wall, grounded)

The physical parameterisation, including the analytical charge density profile
δñ, lives in `core/het_config.py::HETConfig2D`. This module contains only the
discretisation: it evaluates that profile on a mesh and packages the result as
a `PoissonLine2D`, the line-decomposed problem type consumed by every outer
scheme in `solvers/outer`. Both the classical and the quantum solvers therefore
see the HET channel through exactly the same interface as the generic Poisson
benchmarks — the sole difference is the source term and the anode boundary.

Two cases are provided:

    build_het_problem   — the Boeuf-Garrigues charge density with physical
                          Dirichlet data (V_d = 300 V at the anode). No closed
                          form exists; certify against a fine-mesh reference.
    build_het_sinusoidal — a manufactured separable source admitting the exact
                          solution φ̃ = sin(πx̃)·sin(πỹ) under homogeneous
                          Dirichlet data, enabling quantitative error assessment
                          independent of any classical reference solver.

Scaling convention
------------------
`PoissonLine2D` uses the physical (unscaled) form, in which the operator
carries the 1/dx² and 1/dy² factors and the right-hand side is f itself rather
than h²·f. The source assembled here is therefore −α·δñ directly, with the
spacings dx = 1/(N+1) and dy = 1/(Nr+1) arising from the non-dimensional unit
square. On a square mesh this is algebraically identical to the h²-scaled form
(diagonal −4, right-hand side h²f) used in the reference literature.

References
----------
Boeuf & Garrigues, J. Appl. Phys. 84(7), 3541-3554 (1998).
Ghafourpour & Laizet, Phys. Rev. Applied 24, 024032 (2025).
"""
from __future__ import annotations

import numpy as np

from core.exact_solutions import E_field_2d_sinusoidal, phi_2d_sinusoidal
from core.het_config import HETConfig2D
from problems.poisson_line_2d import PoissonLine2D


# ── Boeuf-Garrigues Physical Case ─────────────────────────────────────────────

def build_het_problem(
    cfg: HETConfig2D,
    N:   int,
    Nr:  int | None = None,
) -> PoissonLine2D:
    """
    Assembles the 2D HET channel with the Boeuf-Garrigues charge density.

    The source term is the non-dimensional Poisson right-hand side −α·δñ,
    evaluated at every interior node. Dirichlet data place the anode at the
    non-dimensional discharge potential α_bc and hold the cathode plane and
    both radial walls at ground.

    Parameters
    ----------
    cfg : HETConfig2D
        Physical parameterisation of the discharge channel.
    N : int
        Number of interior nodes along the axial direction. Must be a power of
        two when the resulting strips are handed to a quantum inner solver,
        which encodes a length-N strip on log₂(N) qubits; the constraint is not
        enforced here because the classical path and the fine-mesh reference
        solves are unconstrained.
    Nr : int, optional
        Number of interior nodes along the radial direction. Defaults to N.

    Returns
    -------
    problem : PoissonLine2D
        (N, Nr) line-decomposed problem, with κ(A_row) → 3⁻ as N → ∞.

    Notes
    -----
    The strip operator is the same tridiagonal matrix for every strip, so
    κ(A_row) is bounded irrespective of N — in sharp contrast with the O(N²)
    growth of the corresponding 1D Poisson operator. This bound is the reason
    the line decomposition is viable for the quantum solvers at all.
    """
    return PoissonLine2D(
        cfg.poisson_source_at(*cfg.grid(N, Nr)),
        Lx=1.0, Ly=1.0,
        bc_x0=cfg.alpha_bc,   # Anode  : φ̃(0, ỹ) = α_bc
        bc_x1=0.0,            # Cathode: φ̃(1, ỹ) = 0
        bc_y0=0.0,            # Inner wall, grounded
        bc_y1=0.0,            # Outer wall, grounded
    )


# ── Manufactured Sinusoidal Case ──────────────────────────────────────────────

def build_het_sinusoidal(
    cfg: HETConfig2D,
    N:   int,
    Nr:  int | None = None,
) -> PoissonLine2D:
    """
    Assembles the 2D HET channel with a manufactured sinusoidal source.

    The source

        f(x̃, ỹ) = −2π² sin(πx̃) sin(πỹ)

    corresponds to the charge density δñ = (2π²/α) sin(πx̃) sin(πỹ), which is
    separable, peaks at the channel centre and vanishes on all four boundaries.
    Under homogeneous Dirichlet data the exact solution is

        φ̃(x̃, ỹ) = sin(πx̃) sin(πỹ)

    so solver error can be quantified against a closed form rather than against
    another numerical solution — the only configuration in the 2D suite for
    which this is possible.

    Parameters
    ----------
    cfg : HETConfig2D
        Physical parameterisation. Used solely to convert the resulting
        potential and field to dimensional units; the discrete system itself is
        independent of it, since the α prefactor cancels between the charge
        density and the source term.
    N : int
        Number of interior nodes along the axial direction.
    Nr : int, optional
        Number of interior nodes along the radial direction. Defaults to N.

    Returns
    -------
    problem : PoissonLine2D
        (N, Nr) line-decomposed problem with homogeneous Dirichlet data.
    """
    X, Y = cfg.grid(N, Nr)
    f = -2.0 * np.pi**2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    return PoissonLine2D(f, Lx=1.0, Ly=1.0)


def sinusoidal_solution(
    cfg: HETConfig2D,
    N:   int,
    Nr:  int | None = None,
) -> np.ndarray:
    """
    Evaluates the exact solution of the manufactured sinusoidal case.

    Parameters
    ----------
    cfg : HETConfig2D
        Physical parameterisation, supplying the mesh convention.
    N : int
        Number of interior nodes along the axial direction.
    Nr : int, optional
        Number of interior nodes along the radial direction. Defaults to N.

    Returns
    -------
    phi_exact : np.ndarray
        (N, Nr) non-dimensional potential φ̃ = sin(πx̃)·sin(πỹ) at the interior
        nodes. Multiply by φ_0 [V] to recover the dimensional potential.
    """
    X, Y = cfg.grid(N, Nr)
    return phi_2d_sinusoidal(X, Y)


def sinusoidal_electric_field(
    cfg: HETConfig2D,
    N:   int,
    Nr:  int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluates the exact electric field of the manufactured sinusoidal case.

    The field follows from E = −∇φ with the chain rule supplying the
    dimensional scaling: ∂/∂x = (1/L_x)·∂/∂x̃, so that E_x carries φ_0/L_x and
    E_y carries φ_0/L_y.

    Parameters
    ----------
    cfg : HETConfig2D
        Physical parameterisation, supplying φ_0, L_x and L_y.
    N : int
        Number of interior nodes along the axial direction.
    Nr : int, optional
        Number of interior nodes along the radial direction. Defaults to N.

    Returns
    -------
    E_x : np.ndarray
        (N, Nr) axial electric field [V/m].
    E_y : np.ndarray
        (N, Nr) radial electric field [V/m].
    """
    X, Y = cfg.grid(N, Nr)
    return E_field_2d_sinusoidal(X, Y, cfg.phi_0, cfg.L_x, cfg.L_y)
