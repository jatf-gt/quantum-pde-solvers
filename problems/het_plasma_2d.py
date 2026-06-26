"""
2-D Hall Effect Thruster plasma Poisson problem for quantum solver
benchmarking.

Physical model
--------------
The non-dimensionalised 2-D Poisson equation for the electrostatic
potential φ̃ in the discharge channel is:

    ∂²φ̃/∂x̃² + ∂²φ̃/∂ỹ² = -α · δñ(x̃, ỹ)

where:
    x̃ = x/L_x    axial coordinate (normalised by channel length)
    ỹ = y/L_y    radial coordinate (normalised by channel height)
    α = L_x²/λ_D²  dimensionless Debye scaling parameter
    δñ = (n_i - n_e)/n_0  non-dimensional net charge density

Boundary conditions (Dirichlet on all four edges):
    φ̃(0,  ỹ) = α_bc = V_discharge/φ_0   (anode)
    φ̃(1,  ỹ) = 0                          (cathode)
    φ̃(x̃, 0) = 0                          (inner wall, grounded)
    φ̃(x̃, 1) = 0                          (outer wall, grounded)

The line-Jacobi decomposition (Ghafourpour & Laizet 2025, Eq. 9)
reduces this to a sequence of 1-D TST sub-problems with a = -4, b = 1,
identical in structure to the generic 2-D Poisson case. The only
difference is the HET source term δñ(x̃, ỹ).

The 2-D charge density profile is constructed as the outer product of
the axial profile (from the 1-D HET model) and a radial profile that
models the near-wall sheath regions.

References
----------
Boeuf & Garrigues, J. Appl. Phys. 84(7), 3541-3554 (1998).
Ghafourpour & Laizet, Phys. Rev. Applied 24, 024032 (2025).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.het_config import HETPhysicalConfig
from problems.poisson_2d import PoissonProblem2D, build_grid_2d, build_row_tst_matrix
from core.config import SimConfig2D


# -- HET 2-D configuration ----------------------------------------------------

@dataclass
class HETConfig2D:
    """
    Configuration for the 2-D HET plasma Poisson benchmark.

    Extends the physical parameters of HETPhysicalConfig with a radial
    dimension and a 2-D charge density profile model.

    Attributes
    ----------
    L_x : float
        Axial channel length [m]. Default 25 mm (Boeuf & Garrigues 1998).
    L_y : float
        Radial channel height [m]. Default 20 mm (typical SPT-100 geometry).
    V_discharge : float
        Discharge voltage [V]. Sets the anode boundary condition.
    T_e_eV : float
        Electron temperature [eV]. Used to compute λ_D and φ_0.
    n_0 : float
        Reference plasma density [m⁻³].
    x_peak : float
        Axial location of peak plasma density (non-dimensional, ∈ [0,1]).
    sigma_n : float
        Axial width of the density profile (non-dimensional).
    n_min : float
        Minimum plasma density as a fraction of n_0.
    delta_0_factor : float
        Dimensionless amplitude of the charge separation:
        δ_0 = delta_0_factor / α. Ensures α·δ_0 = O(1) << α_bc.
    sigma_anode : float
        Axial anode sheath thickness (non-dimensional).
    sigma_cath : float
        Axial cathode sheath thickness (non-dimensional).
    sigma_wall : float
        Radial wall sheath thickness (non-dimensional). Models the
        near-wall charge separation in the radial direction.
    N : int
        Number of interior nodes in each direction. Must be a power of 2.
    epsilon : float
        Trotter / VQLS tolerance for the quantum sub-solvers.
    tol : float
        Line-Jacobi convergence tolerance.
    max_iter : int
        Maximum number of line-Jacobi iterations.

    Derived attributes (populated by __post_init__)
    -----------------------------------------------
    lambda_D : float
        Debye length [m]: λ_D = sqrt(ε_0 k_B T_e / (e² n_0)).
    phi_0 : float
        Thermal voltage [V]: φ_0 = T_e [eV].
    alpha : float
        Dimensionless scaling: α = L_x² / λ_D².
    alpha_bc : float
        Non-dimensional anode potential: α_bc = V_discharge / φ_0.
    delta_0 : float
        Charge separation amplitude: δ_0 = delta_0_factor / α.
    """

    from core.het_config import EPS_0, E_CHARGE, EV_TO_J

    L_x:            float = 0.025
    L_y:            float = 0.020
    V_discharge:    float = 300.0
    T_e_eV:         float = 20.0
    n_0:            float = 5e17
    x_peak:         float = 0.75
    sigma_n:        float = 0.20
    n_min:          float = 0.05
    delta_0_factor: float = 5.0
    sigma_anode:    float = 0.08
    sigma_cath:     float = 0.06
    sigma_wall:     float = 0.10
    N:              int   = 8
    epsilon:        float = 0.01
    tol:            float = 1e-8
    max_iter:       int   = 500

    lambda_D:  float = field(init=False, repr=True)
    phi_0:     float = field(init=False, repr=True)
    alpha:     float = field(init=False, repr=True)
    alpha_bc:  float = field(init=False, repr=True)
    delta_0:   float = field(init=False, repr=True)

    def __post_init__(self) -> None:
        from core.het_config import EPS_0, E_CHARGE, EV_TO_J
        if self.N <= 0 or (self.N & (self.N - 1)) != 0:
            raise ValueError(
                f"N must be a positive power of 2, received N={self.N}."
            )
        T_e_J         = self.T_e_eV * EV_TO_J
        self.lambda_D = float(np.sqrt(
            EPS_0 * T_e_J / (E_CHARGE**2 * self.n_0)
        ))
        self.phi_0    = float(self.T_e_eV)
        self.alpha    = float((self.L_x / self.lambda_D)**2)
        self.alpha_bc = float(self.V_discharge / self.phi_0)
        self.delta_0  = self.delta_0_factor / self.alpha

    def summary(self) -> str:
        return (
            f"HET-2D (Boeuf-Garrigues 1998): "
            f"L_x={self.L_x*1e3:.1f}mm, L_y={self.L_y*1e3:.1f}mm, "
            f"V_d={self.V_discharge}V, T_e={self.T_e_eV}eV | "
            f"λ_D={self.lambda_D*1e6:.2f}μm, α={self.alpha:.1f}, "
            f"α_bc={self.alpha_bc:.1f}, δ_0={self.delta_0:.2e}, N={self.N}"
        )


# -- HET 2-D problem ----------------------------------------------------------

class HETPoissonProblem2D(PoissonProblem2D):
    """
    2-D HET plasma Poisson problem, inheriting the line-Jacobi
    infrastructure of PoissonProblem2D.

    The class overrides the source function evaluation to use the
    physically motivated 2-D HET charge density profile, whilst
    retaining the full get_row_system / build_full_rhs interface
    required by the Thomas-2D and VQLS-2D solvers.

    Attributes
    ----------
    het_config : HETConfig2D
        Physical and numerical parameters for the HET problem.
    delta_n_2d : np.ndarray, shape (N, N)
        Non-dimensional net charge density δñ(x̃_i, ỹ_j) at all
        interior nodes, precomputed for efficiency.
    """

    def __init__(self, cfg: HETConfig2D) -> None:
        # Construct a SimConfig2D wrapper so the parent __init__ receives
        # the expected interface. The source function key is set to 'fS'
        # as a placeholder; the actual source is overridden via delta_n_2d.
        # Inline import required to avoid circular dependency at module level.
        sim_cfg = SimConfig2D(
            N        = cfg.N,
            epsilon  = cfg.epsilon,
            source_fn= "fS",       # placeholder; overridden below
            tol      = cfg.tol,
            max_iter = cfg.max_iter,
            bc_x0    = cfg.alpha_bc,  # anode: φ̃(0, ỹ) = α_bc
            bc_x1    = 0.0,           # cathode: φ̃(1, ỹ) = 0
            bc_y0    = 0.0,           # inner wall: grounded
            bc_y1    = 0.0,           # outer wall: grounded
        )
        super().__init__(sim_cfg)
        self.het_config = cfg

        # Pre-compute the 2-D charge density profile.
        self.delta_n_2d = self._build_charge_density()

    def _build_charge_density(self) -> np.ndarray:
        """
        Construct the 2-D non-dimensional net charge density field.

        The profile is the outer product of the axial charge separation
        (from the 1-D HET model) and a radial modulation that models
        the near-wall sheath regions:

            δñ(x̃, ỹ) = δñ_axial(x̃) · g_radial(ỹ)

        where:
            δñ_axial(x̃) = δ_0 · [exp(-x̃/σ_a) - exp(-(1-x̃)²/σ_c²)]
            g_radial(ỹ)  = 1 - exp(-ỹ/σ_w) - exp(-(1-ỹ)/σ_w)

        The radial modulation g_radial is unity in the bulk and
        approaches zero near the walls (ỹ = 0 and ỹ = 1), modelling
        the quasi-neutral core with sheath regions at both walls.

        Returns
        -------
        delta_n : np.ndarray, shape (N, N)
            δñ(x̃_i, ỹ_j) at all interior nodes.
        """
        cfg = self.het_config
        x   = self.X[:, 0]   # axial coordinates, shape (N,)
        y   = self.Y[0, :]   # radial coordinates, shape (N,)

        # Axial charge separation (same as 1-D HET model).
        delta_axial = cfg.delta_0 * (
            np.exp(-x / cfg.sigma_anode)
            - np.exp(-((1.0 - x)**2) / cfg.sigma_cath**2)
        )

        # Radial modulation: unity in bulk, zero at walls.
        g_radial = (
            1.0
            - np.exp(-y / cfg.sigma_wall)
            - np.exp(-(1.0 - y) / cfg.sigma_wall)
        )

        # Outer product: shape (N, N).
        return np.outer(delta_axial, g_radial)

    def get_row_system(
        self,
        j      : int,
        u_prev : np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (A_row, b_row) for the line-Jacobi update of row j,
        using the HET charge density as the source term.

        Overrides the parent method to substitute the HET source term
        δñ(x̃, ỹ_j) in place of the generic source function f(x, y).

        Parameters
        ----------
        j : int
            Row index (0-indexed), 0 ≤ j ≤ N−1.
        u_prev : np.ndarray, shape (N, N)
            Solution field from the previous iteration.

        Returns
        -------
        A_row : np.ndarray, shape (N, N)
            TST row matrix (a=−4, b=1), shared across all rows.
        b_row : np.ndarray, shape (N,)
            RHS vector for row j incorporating the HET source term
            and y-neighbour contributions from u_prev.
        """
        cfg = self.het_config
        N   = cfg.N
        h   = self.h

        # HET source term for row j: -α · δñ(x̃_i, ỹ_j).
        f_row = -cfg.alpha * self.delta_n_2d[:, j]
        b_row = h**2 * f_row

        # Subtract y-neighbour contributions (line-Jacobi coupling).
        if j == 0:
            b_row -= cfg.alpha_bc * np.ones(N)   # inner wall BC
        else:
            b_row -= u_prev[:, j - 1]

        if j == N - 1:
            b_row -= 0.0                          # outer wall BC (zero)
        else:
            b_row -= u_prev[:, j + 1]

        # Incorporate x-direction Dirichlet BCs.
        b_row[0]  -= cfg.alpha_bc   # anode
        b_row[-1] -= 0.0            # cathode (zero contribution)

        return self.A_row, b_row

    def build_full_rhs(self) -> np.ndarray:
        """
        Assemble the full N²-length RHS vector for the classical
        reference solve, using the HET source term.

        Returns
        -------
        rhs : np.ndarray, shape (N²,)
            Flattened RHS in C-order (row j occupies entries j·N to
            (j+1)·N − 1).
        """
        cfg = self.het_config
        N   = cfg.N
        h   = self.h
        rhs = np.zeros(N * N)

        for j in range(N):
            b_col       = h**2 * (-cfg.alpha * self.delta_n_2d[:, j])
            b_col[0]   -= cfg.alpha_bc   # anode BC
            # cathode BC is zero; no subtraction required
            if j == 0:
                b_col -= cfg.alpha_bc    # inner wall BC
            if j == N - 1:
                pass                     # outer wall BC is zero
            rhs[j * N:(j + 1) * N] = b_col

        return rhs

    def summary(self) -> str:
        return (
            f"{self.het_config.summary()} | "
            f"κ(A_row)={self.kappa_row:.4f}"
        )