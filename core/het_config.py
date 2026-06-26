"""
Defines the physical and numerical configuration parameters for the 1D Hall 
Effect Thruster (HET) plasma Poisson benchmarks.

This system is non-dimensionalised utilising the Debye length λ_D and the 
electron thermal voltage φ_0 = k_B T_e / e prior to discretisation. This 
analytical scaling guarantees that the quantum solver processes a well-conditioned 
operator matrix irrespective of the specific physical parameter regime. 

Physical parameters are instantiated in standard SI units; the non-dimensional 
scaling transformations are explicitly applied within `problems/het_plasma_1d.py`.

Reference: Laizet (PhD thesis), Hall Effect Thruster plasma modelling with 
           Poisson equation for the electrostatic potential.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


# ── Physical Constants ────────────────────────────────────────────────────────

E_CHARGE    = 1.602176634e-19   # Elementary charge [C]
EPS_0       = 8.854187817e-12   # Vacuum permittivity [F/m]
K_BOLTZMANN = 1.380649e-23      # Boltzmann constant [J/K]
EV_TO_J     = E_CHARGE          # Conversion factor: 1 eV in Joules


# ── Configuration Structure ───────────────────────────────────────────────────

@dataclass
class HETConfig:
    """
    Encapsulates the parameters governing a 1D HET plasma Poisson benchmark execution.

    Physical Parameters
    -------------------
    L : float
        Axial channel length [m].
    T_e_eV : float
        Electron temperature [eV].
    n_0 : float
        Reference plasma density [m⁻³].
    V_discharge : float
        Discharge voltage [V] applied to the system, establishing the anode 
        boundary condition.
    rho_profile : Literal["gaussian", "linear", "step"]
        Identifier dictating the analytical charge density distribution.
    rho_0 : float
        Peak non-dimensional charge density amplitude.
    x_ion : float
        Spatial centre of the ionisation zone, mapped to the non-dimensional 
        domain [0, 1]. Exclusively utilised by 'step' and 'gaussian' profiles.
    sigma : float
        Non-dimensional distribution width for the 'gaussian' profile.

    Numerical Parameters
    --------------------
    N : int
        System matrix dimension (number of interior spatial nodes). Must be a 
        strict power of two to accommodate quantum amplitude encoding.
    epsilon : float
        Precision parameter governing the Trotterisation or VQLS tolerance thresholds.

    Derived Quantities (Computed upon instantiation)
    ------------------------------------------------
    lambda_D : float
        System Debye length [m].
    phi_0 : float
        Thermal voltage corresponding to the electron temperature, k_B T_e / e [V].
    alpha : float
        Dimensionless scaling ratio (L² / λ_D²), functioning as the primary 
        source scaling parameter in the Poisson equation.
    alpha_bc : float
        Non-dimensional anode potential, V_discharge / phi_0.
    """

    # Physical parameters
    L:           float = 0.025          # 2.5 cm channel length
    T_e_eV:      float = 20.0           # 20 eV electron temperature
    n_0:         float = 1e17           # 10^17 m⁻³ reference density
    V_discharge: float = 300.0          # 300 V discharge voltage
    rho_profile: Literal["gaussian", "linear", "step"] = "gaussian"
    rho_0:       float = 1.0            # Peak charge density amplitude
    x_ion:       float = 0.5            # Ionisation zone spatial centre
    sigma:       float = 0.1            # Gaussian distribution width

    # Numerical parameters
    N:       int   = 8
    epsilon: float = 0.01

    # Derived parameters — systematically populated by __post_init__
    lambda_D:  float = field(init=False, repr=True)
    phi_0:     float = field(init=False, repr=True)
    alpha:     float = field(init=False, repr=True)
    alpha_bc:  float = field(init=False, repr=True)

    def __post_init__(self) -> None:
        """Validates parameter constraints and populates derived physical constants."""
        if self.N <= 0 or (self.N & (self.N - 1)) != 0:
            raise ValueError(
                f"System dimension N must be a positive power of 2, received N={self.N}."
            )
        if self.T_e_eV <= 0:
            raise ValueError(f"T_e_eV must be strictly positive, received {self.T_e_eV}.")
        if self.n_0 <= 0:
            raise ValueError(f"n_0 must be strictly positive, received {self.n_0}.")

        # Debye length calculation: λ_D = sqrt(ε_0 k_B T_e / (e² n_0))
        T_e_J         = self.T_e_eV * EV_TO_J
        self.lambda_D = float(np.sqrt(
            EPS_0 * T_e_J / (E_CHARGE**2 * self.n_0)
        ))

        # Thermal voltage calculation: φ_0 = k_B T_e / e = T_e [eV]
        self.phi_0    = float(self.T_e_eV)   # Units: Volts

        # Dimensionless scaling parameter: α = L² / λ_D²
        self.alpha    = float((self.L / self.lambda_D)**2)

        # Non-dimensional boundary constraint at the anode.
        self.alpha_bc = float(self.V_discharge / self.phi_0)

    def summary(self) -> str:
        """Generates a concise execution summary detailing core physical scalings."""
        return (
            f"HET: L={self.L*100:.1f}cm, T_e={self.T_e_eV}eV, "
            f"n_0={self.n_0:.1e}m⁻³, V_d={self.V_discharge}V | "
            f"λ_D={self.lambda_D*1e6:.2f}μm, α={self.alpha:.1f}, "
            f"φ_0={self.phi_0:.1f}V, N={self.N}, ε={self.epsilon}"
        )