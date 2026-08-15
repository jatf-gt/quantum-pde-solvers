"""
Physical configuration parameters for the 1D and 2D Hall Effect Thruster (HET)
plasma Poisson benchmarks.

The system is non-dimensionalised by the Debye length λ_D and the electron
thermal voltage φ_0 = k_B T_e / e prior to discretisation, which reduces the
governing equation to a single dimensionless group α = (L/λ_D)² and renders the
operator independent of the physical parameter regime. Note that this does not
by itself improve the conditioning of the discretised operator, whose condition
number remains O(N²) for the 1D Poisson matrix; what it removes is the
dependence of the *matrix* on the device parameters, which instead enter
through α on the right-hand side.

Physical parameters are declared in SI units; the non-dimensional scaling
transformations are applied in the corresponding problem assembly modules,
`problems/het_plasma_1d.py` and `problems/het_plasma_2d.py`.

Three configuration structures are provided:

    HETConfig          — 1D benchmark with a parameterised analytical charge
                         density profile ('gaussian', 'linear', 'step').
    HETPhysicalConfig  — 1D Boeuf-Garrigues device, with an analytical sheath
                         model for the net charge density.
    HETConfig2D        — 2D axial-radial channel, extending the Boeuf-Garrigues
                         parameterisation with a radial wall sheath.

References
----------
Boeuf & Garrigues, J. Appl. Phys. 84(7), 3541-3554 (1998).
Laizet (PhD thesis), Hall Effect Thruster plasma modelling with the Poisson
equation for the electrostatic potential.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from core import het_geometry as geom


# -- Physical Constants --------------------------------------------------------

E_CHARGE    = 1.602176634e-19   # Elementary charge [C]
EPS_0       = 8.854187817e-12   # Vacuum permittivity [F/m]
EV_TO_J     = E_CHARGE          # Conversion factor: 1 eV in Joules


# -- Configuration Structure ---------------------------------------------------

@dataclass
class HETConfig:
    """
    Parameters governing a 1D HET plasma Poisson benchmark.

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
        Identifier selecting the analytical charge density distribution.
    rho_0 : float
        Peak non-dimensional charge density amplitude.
    x_ion : float
        Spatial centre of the ionisation zone, mapped to the non-dimensional
        domain [0, 1]. Used only by the 'step' and 'gaussian' profiles.
    sigma : float
        Non-dimensional distribution width for the 'gaussian' profile.

    Numerical Parameters
    --------------------
    N : int
        System matrix dimension (number of interior spatial nodes). Must be a
        power of two to accommodate quantum amplitude encoding.
    epsilon : float
        Precision parameter governing the Trotterisation step count and the
        VQLS convergence tolerance.

    Derived Quantities (Computed upon instantiation)
    ------------------------------------------------
    lambda_D : float
        Debye length [m]: λ_D = sqrt(ε_0 k_B T_e / (e² n_0)).
    phi_0 : float
        Electron thermal voltage [V]: φ_0 = k_B T_e / e = T_e [eV].
    alpha : float
        Dimensionless scaling ratio α = L² / λ_D², the primary source scaling
        parameter of the Poisson equation. Evaluates to α ≈ 5.65×10⁴ for the
        defaults below.
    alpha_bc : float
        Non-dimensional anode potential, α_bc = V_discharge / φ_0.
    """

    # Physical parameters
    L:           float = geom.L_Z       # Axial channel length [m], see core/het_geometry.py
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
        """
        Validates parameter constraints and populates the derived quantities.

        Raises
        ------
        ValueError
            If N is not a positive power of two, or if T_e_eV or n_0 is
            non-positive.
        """
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
        """Returns a one-line summary of the physical and derived scalings."""
        return (
            f"HET: L={self.L*100:.1f}cm, T_e={self.T_e_eV}eV, "
            f"n_0={self.n_0:.1e}m⁻³, V_d={self.V_discharge}V | "
            f"λ_D={self.lambda_D*1e6:.2f}μm, α={self.alpha:.1f}, "
            f"φ_0={self.phi_0:.1f}V, N={self.N}, ε={self.epsilon}"
        )


# -- Physical Configuration (Boeuf-Garrigues) ----------------------------------

@dataclass
class HETPhysicalConfig:
    """
    Encapsulates the physical configuration parameters for the Boeuf-Garrigues
    1D Hall Effect Thruster (HET) benchmark.

    The parameters follow Table 1 of Boeuf & Garrigues (1998), J. Appl. Phys.
    84(7), 3541-3554. The spatial plasma density profile is approximated by a
    Gaussian distribution centred near the exit plane.

    The net charge density, δn = n_i - n_e, is prescribed analytically to
    represent the sheath boundaries at the anode and cathode, following the
    quasi-neutral bulk approximation with the sheath corrections of Hagelaar
    et al. (2002), Phys. Rev. E 62(1). This is a generic, physically motivated
    stand-in carrying Boeuf & Garrigues' scalar parameters (T_e, n_0,
    V_discharge, channel geometry) — it is not their reported potential/field
    profile, because their own quasineutral model does not solve Poisson's
    equation for the field at all (their Sec. III.G is explicit about this).
    For a case actually anchored to their reported profile (Fig. 5(a) of the
    paper), see `core.cases.get("het_1d_bg1998_fig5_profile")`.

    Scaling of the charge separation amplitude δ_0 is load-bearing and is
    therefore derived rather than prescribed. In a quasi-neutral HET plasma the
    departure from charge neutrality satisfies

        (n_i - n_e)/n_0 ~ (λ_D/L)² = 1/α ≈ 3.54×10⁻⁶

    for the default parameters below, so δ_0 must be of order 1/α if the space
    charge term is to remain a physically realistic small perturbation on the
    applied voltage. δ_0 is accordingly set as δ_0 = δ_0_factor / α with
    δ_0_factor = 5, an amplitude that produces a visible but realistic space
    charge effect. A fixed δ_0 of order 10⁻² — that is, O(10⁴) times too large —
    drives the solution an order of magnitude above the correct scale, which is
    why the factor is applied to 1/α rather than chosen directly.

    Attributes
    ----------
    Physical Parameters (SI Units)
        L : float
            Axial channel length [m].
        V_discharge : float
            Applied discharge voltage establishing the anode potential [V].
        T_e_eV : float
            Uniform electron temperature approximation [eV].
        n_0 : float
            Reference bulk plasma density [m⁻³].

    Density Profile Parameters
        x_peak : float
            Non-dimensional spatial location of the density peak, mapped to [0,1].
        sigma_n : float
            Non-dimensional Gaussian distribution width.
        n_min : float
            Minimum boundary density expressed as a fractional scalar of n_0.

    Charge Separation Parameters (Sheath Modelling)
        delta_0 : float
            Non-dimensional peak charge separation amplitude.
        sigma_anode : float
            Non-dimensional characteristic thickness of the anode sheath.
        sigma_cath : float
            Non-dimensional characteristic thickness of the cathode sheath.

    Numerical Parameters
        N : int
            System matrix dimension (interior nodes, must be a power of 2).
        epsilon : float
            Precision threshold for HHL Trotterisation and VQLS tolerance.
    """
    # Physical parameters — Boeuf & Garrigues (1998) Table 1.
    L:            float = geom.L_Z       # Axial channel length [m], see core/het_geometry.py
    V_discharge:  float = 300.0          # 300 V discharge potential
    T_e_eV:       float = 20.0           # 20 eV uniform electron temperature
    n_0:          float = 5e17           # 5×10¹⁷ m⁻³ reference density

    # Density profile parameters.
    x_peak:       float = 0.75           # Peak density proximity to exit plane
    sigma_n:      float = 0.20           # Spatial profile width
    n_min:        float = 0.05           # Minimum boundary density fraction

    # Charge separation parameters (Analytical sheath model).
    # Set delta_0_factor to control the amplitude: delta_0 = factor/alpha.
    delta_0_factor: float = 5.0          # dimensionless, O(1)
    sigma_anode:  float = 0.08           # Anode sheath exponential thickness
    sigma_cath:   float = 0.06           # Cathode sheath exponential thickness

    # Numerical execution parameters.
    N:            int   = 8
    epsilon:      float = 0.01

    # Derived constants — systematically populated by __post_init__.
    lambda_D:     float = field(init=False, repr=True)
    phi_0:        float = field(init=False, repr=True)
    alpha:        float = field(init=False, repr=True)
    alpha_bc:     float = field(init=False, repr=True)
    delta_0:      float = field(init=False, repr=True)

    def __post_init__(self) -> None:
        """
        Validates numerical constraints and computes the dimensionless scalings.

        Raises
        ------
        ValueError
            If N is not a positive power of two.
        """
        if self.N <= 0 or (self.N & (self.N - 1)) != 0:
            raise ValueError(
                f"System dimension N must be a positive power of 2, received {self.N}."
            )

        T_e_J         = self.T_e_eV * EV_TO_J
        self.lambda_D = float(np.sqrt(
            EPS_0 * T_e_J / (E_CHARGE**2 * self.n_0)
        ))
        self.phi_0    = float(self.T_e_eV)
        self.alpha    = float((self.L / self.lambda_D)**2)
        self.alpha_bc = float(self.V_discharge / self.phi_0)

        # Physical charge separation amplitude: delta_0 = factor / alpha.
        # This ensures alpha * delta_0 = factor ~ O(1), so the space charge
        # contribution to the potential is of order factor, which is a small
        # fraction of alpha_bc = V_d/phi_0 = 15.
        self.delta_0 = self.delta_0_factor / self.alpha

    def summary(self) -> str:
        """Returns a one-line summary of the physical and derived scalings."""
        return (
            f"HET (Boeuf-Garrigues 1998): "
            f"L={self.L*1e3:.1f}mm, V_d={self.V_discharge}V, "
            f"T_e={self.T_e_eV}eV, n_0={self.n_0:.1e}m⁻³ | "
            f"λ_D={self.lambda_D*1e6:.2f}μm, α={self.alpha:.1f}, "
            f"α_bc={self.alpha_bc:.1f}, δ_0={self.delta_0:.2e}, "
            f"N={self.N}"
        )


# -- Two-Dimensional Physical Configuration (Boeuf-Garrigues) ------------------

@dataclass
class HETConfig2D:
    """
    Encapsulates the physical parameterisation of the 2D Hall Effect Thruster
    (HET) discharge channel, resolved in the axial-radial (x, y) plane.

    Boeuf & Garrigues (1998) is a 1D axial model; there is no 2D result in the
    paper to reproduce. This structure extends their scalar parameters and the
    axial charge-separation idea of `HETPhysicalConfig` with a second, radial
    wall-sheath factor that is this codebase's own addition, not the paper's —
    see `HETPhysicalConfig`'s docstring for the corresponding caveat in 1D.

    This structure is the two-dimensional sibling of `HETPhysicalConfig` and
    shares its non-dimensionalisation exactly: lengths are scaled by the axial
    channel length L_x, potentials by the electron thermal voltage φ_0, and the
    net charge density by the reference plasma density n_0. The governing
    non-dimensional Poisson equation is

        ∂²φ̃/∂x̃² + ∂²φ̃/∂ỹ² = −α · δñ(x̃, ỹ)

    with x̃ = x/L_x the axial coordinate, ỹ = y/L_y the radial coordinate,
    α = L_x²/λ_D² the Debye scaling parameter, and δñ = (n_i − n_e)/n_0 the
    non-dimensional net charge density.

    The structure carries physical parameters exclusively. Discretisation
    parameters (N), quantum precision parameters (ε) and outer-iteration
    controls (tol, max_iter) are deliberately absent: they belong to the solver
    invocation, not to the description of the device, and holding them here
    previously forced a single configuration object to be rebuilt whenever the
    mesh or the solver changed. Mesh-dependent quantities are obtained by
    passing N to `charge_density`, and the corresponding `PoissonLine2D`
    instances are assembled in `problems/het_plasma_2d.py`.

    Attributes
    ----------
    Physical Parameters (SI Units)
        L_x : float
            Axial channel length [m]. Default 40 mm, from `core.het_geometry`
            (Boeuf & Garrigues 1998, §II and §IV.A).
        L_y : float
            Radial channel height [m]. Default 20 mm, from `core.het_geometry`
            (Boeuf & Garrigues 1998, §II and §IV.A).
        V_discharge : float
            Applied discharge voltage establishing the anode potential [V].
        T_e_eV : float
            Uniform electron temperature approximation [eV].
        n_0 : float
            Reference bulk plasma density [m⁻³].

    Density Profile Parameters
        x_peak : float
            Non-dimensional axial location of the density peak, mapped to [0,1].
        sigma_n : float
            Non-dimensional axial width of the density profile.
        n_min : float
            Minimum boundary density expressed as a fractional scalar of n_0.

        These three parameters describe the bulk plasma density profile of the
        Boeuf-Garrigues device and are retained for parity with the 1D
        configuration. The 2D charge-separation model implemented in
        `charge_density` is formulated directly in terms of the sheath
        parameters below and does not presently consume them.

    Charge Separation Parameters (Sheath Modelling)
        delta_0_factor : float
            Dimensionless O(1) amplitude of the charge separation, from which
            δ_0 = delta_0_factor / α. Scaling by 1/α is physically mandatory
            rather than cosmetic: in a quasi-neutral HET plasma the departure
            from charge neutrality satisfies (n_i − n_e)/n_0 ~ (λ_D/L_x)² = 1/α,
            so that α·δ_0 = delta_0_factor = O(1) remains a small perturbation
            on the applied potential α_bc = V_discharge/φ_0 ≈ 15.
        sigma_anode : float
            Non-dimensional characteristic thickness of the anode sheath.
        sigma_cath : float
            Non-dimensional characteristic thickness of the cathode sheath.
        sigma_wall : float
            Non-dimensional characteristic thickness of the radial wall sheath.

    Derived Quantities (Computed upon instantiation)
        lambda_D : float
            Debye length [m]: λ_D = sqrt(ε_0 k_B T_e / (e² n_0)).
        phi_0 : float
            Electron thermal voltage [V]: φ_0 = k_B T_e / e = T_e [eV].
        alpha : float
            Dimensionless Debye scaling: α = L_x² / λ_D².
        alpha_bc : float
            Non-dimensional anode potential: α_bc = V_discharge / φ_0.
        delta_0 : float
            Charge separation amplitude: δ_0 = delta_0_factor / α.

    References
    ----------
    Boeuf & Garrigues, J. Appl. Phys. 84(7), 3541-3554 (1998), Table 1.
    Hagelaar et al., Phys. Rev. E 62(1) (2002) — sheath corrections.
    """

    # Physical parameters — Boeuf & Garrigues (1998) Table 1.
    L_x:            float = geom.L_Z     # Axial channel length [m], see core/het_geometry.py
    L_y:            float = geom.L_R     # Radial channel width [m], see core/het_geometry.py
    V_discharge:    float = 300.0        # 300 V discharge potential
    T_e_eV:         float = 20.0         # 20 eV uniform electron temperature
    n_0:            float = 5e17         # 5×10¹⁷ m⁻³ reference density

    # Bulk density profile parameters.
    x_peak:         float = 0.75         # Peak density proximity to exit plane
    sigma_n:        float = 0.20         # Axial profile width
    n_min:          float = 0.05         # Minimum boundary density fraction

    # Charge separation parameters (analytical sheath model).
    delta_0_factor: float = 5.0          # Dimensionless, O(1)
    sigma_anode:    float = 0.08         # Anode sheath exponential thickness
    sigma_cath:     float = 0.06         # Cathode sheath exponential thickness
    sigma_wall:     float = 0.10         # Radial wall sheath thickness

    # Derived constants — systematically populated by __post_init__.
    lambda_D:  float = field(init=False, repr=True)
    phi_0:     float = field(init=False, repr=True)
    alpha:     float = field(init=False, repr=True)
    alpha_bc:  float = field(init=False, repr=True)
    delta_0:   float = field(init=False, repr=True)

    def __post_init__(self) -> None:
        """Validates physical constraints and computes dimensionless scalings."""
        if self.L_x <= 0 or self.L_y <= 0:
            raise ValueError(
                f"Channel extents must be strictly positive, received "
                f"L_x={self.L_x}, L_y={self.L_y}."
            )
        if self.T_e_eV <= 0:
            raise ValueError(f"T_e_eV must be strictly positive, received {self.T_e_eV}.")
        if self.n_0 <= 0:
            raise ValueError(f"n_0 must be strictly positive, received {self.n_0}.")

        T_e_J         = self.T_e_eV * EV_TO_J
        self.lambda_D = float(np.sqrt(
            EPS_0 * T_e_J / (E_CHARGE**2 * self.n_0)
        ))
        self.phi_0    = float(self.T_e_eV)
        self.alpha    = float((self.L_x / self.lambda_D)**2)
        self.alpha_bc = float(self.V_discharge / self.phi_0)
        self.delta_0  = self.delta_0_factor / self.alpha

    # -- Mesh-Dependent Physical Fields ----------------------------------------

    def grid(self, N: int, Nr: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Constructs the non-dimensional interior coordinate matrices (x̃, ỹ).

        The mesh is vertex-centred with boundary nodes excluded, matching the
        convention of `PoissonLine2D`: x̃_i = i/(N+1) for i = 1 … N, and
        likewise for ỹ. Coordinates are non-dimensional and therefore span the
        unit square irrespective of the physical extents L_x and L_y; the
        physical aspect ratio enters through the operator, not the coordinates.

        Parameters
        ----------
        N : int
            Number of interior nodes along the axial direction.
        Nr : int, optional
            Number of interior nodes along the radial direction. Defaults to N,
            yielding a square mesh.

        Returns
        -------
        X, Y : np.ndarray
            (N, Nr) non-dimensional coordinate matrices in 'ij' indexing order.
        """
        Nr = N if Nr is None else Nr
        x = np.arange(1, N + 1) / (N + 1)
        y = np.arange(1, Nr + 1) / (Nr + 1)
        return np.meshgrid(x, y, indexing="ij")

    def charge_density(self, N: int, Nr: int | None = None) -> np.ndarray:
        """
        Evaluates the non-dimensional net charge density δñ(x̃, ỹ) on the mesh.

        The profile is separable, constructed as the outer product of the axial
        charge separation inherited from the 1D HET model and a radial
        modulation representing the near-wall sheaths:

            δñ(x̃, ỹ) = δñ_axial(x̃) · g_radial(ỹ)

            δñ_axial(x̃) = δ_0 · [exp(−x̃/σ_a) − exp(−(1 − x̃)²/σ_c²)]
            g_radial(ỹ) = 1 − exp(−ỹ/σ_w) − exp(−(1 − ỹ)/σ_w)

        The axial factor is positive in the anode sheath, where ions dominate,
        and negative approaching the cathode plane, reproducing the bipolar
        space-charge structure of Boeuf & Garrigues (1998). The radial
        modulation is unity throughout the quasi-neutral core and decays to
        zero at both walls (ỹ → 0 and ỹ → 1), where the sheath terminates the
        charge separation.

        Parameters
        ----------
        N : int
            Number of interior nodes along the axial direction.
        Nr : int, optional
            Number of interior nodes along the radial direction. Defaults to N.

        Returns
        -------
        delta_n : np.ndarray
            (N, Nr) non-dimensional net charge density δñ = (n_i − n_e)/n_0 at
            every interior node. Dimensionless; multiply by n_0 [m⁻³] to
            recover the physical net density.

        See Also
        --------
        charge_density_at : the same profile at arbitrary coordinates, required
            when the mesh is not this configuration's own (e.g. the refined mesh
            of a reference solve).
        """
        return self.charge_density_at(*self.grid(N, Nr))

    def charge_density_at(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Evaluates the net charge density δñ at arbitrary non-dimensional points.

        This is the analytical profile itself, decoupled from any particular
        mesh. Separating it from `charge_density` is what allows the identical
        physical source to be evaluated on a refined mesh: a fine-mesh reference
        solve needs δñ at N_fine² points that this configuration never sees, and
        interpolating a coarse evaluation would contaminate the reference with
        precisely the error it exists to measure.

        Parameters
        ----------
        X, Y : np.ndarray
            Non-dimensional coordinate matrices of identical shape, with
            x̃ = x/L_x ∈ [0,1] axial and ỹ = y/L_y ∈ [0,1] radial.

        Returns
        -------
        delta_n : np.ndarray
            Net charge density δñ = (n_i − n_e)/n_0, of the same shape as X and
            Y. Dimensionless.
        """
        delta_axial = self.delta_0 * (
            np.exp(-X / self.sigma_anode)
            - np.exp(-((1.0 - X)**2) / self.sigma_cath**2)
        )

        g_radial = (
            1.0
            - np.exp(-Y / self.sigma_wall)
            - np.exp(-(1.0 - Y) / self.sigma_wall)
        )

        return delta_axial * g_radial

    def poisson_source_at(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Evaluates the non-dimensional Poisson right-hand side −α·δñ(x̃, ỹ).

        Supplied as a single callable so that it can be handed directly to
        `benchmark.reference_2d.fine_mesh_reference`, which evaluates the source
        on whatever mesh it selects.

        Parameters
        ----------
        X, Y : np.ndarray
            Non-dimensional coordinate matrices of identical shape.

        Returns
        -------
        f : np.ndarray
            Right-hand side of ∇²φ̃ = −α·δñ, of the same shape as X and Y.
        """
        return -self.alpha * self.charge_density_at(X, Y)

    def summary(self) -> str:
        """Returns a one-line summary of the physical and derived scalings."""
        return (
            f"HET-2D (Boeuf-Garrigues 1998): "
            f"L_x={self.L_x*1e3:.1f}mm, L_y={self.L_y*1e3:.1f}mm, "
            f"V_d={self.V_discharge}V, T_e={self.T_e_eV}eV, "
            f"n_0={self.n_0:.1e}m⁻³ | "
            f"λ_D={self.lambda_D*1e6:.2f}μm, α={self.alpha:.1f}, "
            f"α_bc={self.alpha_bc:.1f}, δ_0={self.delta_0:.2e}"
        )