"""
Assembles the discretised 1D Hall Effect Thruster (HET) plasma Poisson boundary
value problem.

Physical Problem Formulation
----------------------------
The electrostatic potential φ(x) within the 1D axial discharge channel satisfies
the non-dimensionalised Poisson equation

    d²φ̃/dx̃² = f̃(x̃)

where x̃ = x/L ∈ [0,1], φ̃ = φ/φ_0, and the source term f̃ encodes the net charge
density scaled by the parameter α = L²/λ_D²:

    f̃(x̃) = -α · (ñ_i(x̃) - ñ_e(x̃))

Boundary constraints:

    φ̃(0) = V_discharge / φ_0   (anode, non-dimensional discharge voltage)
    φ̃(1) = 0                   (cathode, grounded)

Discretising with N interior nodes and mesh spacing Δx̃ = 1/(N+1), the linear
system Au = b is constructed exactly as in the generic 1D Poisson case. The
identical TST operator (a = -2, b = 1) is used, with the physical source term
and the boundary constraints absorbed into the right-hand side vector.

The resulting system is consumed by the HHL and VQLS solvers without algorithmic
modification; the physics resides entirely in the system assembly.

Reference: Laizet (PhD thesis), Hall Effect Thruster plasma modelling.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from core.het_config import HETConfig, HETPhysicalConfig
from core.source_functions import HET_SOURCE_FUNCTIONS
from core.exact_solutions import HET_EXACT_SOLUTIONS


# -- Problem Container ---------------------------------------------------------

@dataclass
class HETPoissonProblem1D:
    """
    The discretised 1D HET plasma Poisson system.

    Mirrors the `PoissonProblem1D` interface, so that the quantum solvers
    (`hhl_solve`, `vqls_solve`, `qsvt_solve`) accept a plasma configuration
    without architectural modification.

    Attributes
    ----------
    config : HETConfig
        Physical and numerical parameters of the instance.
    x : np.ndarray
        Length-N vector of interior node coordinates on the non-dimensional
        domain [0, 1].
    dx : float
        Uniform mesh spacing, Δx̃ = 1/(N+1).
    A : np.ndarray
        N×N TST system operator (a = -2, b = 1), structurally identical to the
        generic 1D Poisson operator.
    b : np.ndarray
        Length-N right-hand side incorporating the scaled source term and the
        Dirichlet data.
    kappa : float
        2-norm condition number of A.
    b_phys : np.ndarray
        Length-N right-hand side projected back into dimensional units [V],
        retained for diagnostics.
    """
    config: HETConfig
    x:      np.ndarray
    dx:     float
    A:      np.ndarray
    b:      np.ndarray
    kappa:  float
    b_phys: np.ndarray

    def __init__(self, cfg: HETConfig) -> None:
        """
        Assembles the grid, TST operator and right-hand side from a config.

        Parameters
        ----------
        cfg : HETConfig
            Validated physical and numerical configuration.
        """
        self.config = cfg
        N           = cfg.N
        self.dx     = 1.0 / (N + 1)
        self.x      = np.arange(1, N + 1) * self.dx

        # -- System Matrix Construction ----------------------------------------
        # The TST operator mirrors the generic 1D Poisson case; the plasma
        # physics enters exclusively through the right-hand side vector.
        self.A = (
            -2.0 * np.diag(np.ones(N))
            +  1.0 * np.diag(np.ones(N - 1), k=1)
            +  1.0 * np.diag(np.ones(N - 1), k=-1)
        )

        # -- Right-Hand Side Assembly ------------------------------------------
        self.b      = self._build_rhs()
        self.b_phys = self.b * cfg.phi_0   # Dimensional projection [V]

        # -- Condition Number Evaluation ---------------------------------------
        eigs       = np.abs(np.linalg.eigvalsh(self.A))
        self.kappa = float(eigs.max() / eigs.min())

    def _build_rhs(self) -> np.ndarray:
        """
        Assembles the non-dimensional right-hand side vector.

        The discretised equation at interior node i is

            φ̃_{i+1} - 2φ̃_i + φ̃_{i-1} = Δx̃² · f̃(x̃_i)

        and, once the Dirichlet constraints are incorporated,

            b[0]   = Δx̃² · f̃(x̃_1) - φ̃_anode
            b[i]   = Δx̃² · f̃(x̃_i)              for 1 < i < N-1
            b[N-1] = Δx̃² · f̃(x̃_N) - φ̃_cathode

        The cathode constraint φ̃(1) = 0 contributes nothing; the anode
        constraint enforces φ̃(0) = V_discharge / φ_0 = alpha_bc.

        Returns
        -------
        b : np.ndarray
            Length-N non-dimensional right-hand side vector.

        Raises
        ------
        ValueError
            If the configured rho_profile is not a recognised identifier.
        """
        cfg = self.config
        dx  = self.dx
        x   = self.x

        # Evaluate the selected analytical source across the interior nodes.
        f_fn = HET_SOURCE_FUNCTIONS[cfg.rho_profile]

        if cfg.rho_profile == "gaussian":
            f_vals = f_fn(x, cfg.rho_0, cfg.x_ion, cfg.sigma, cfg.alpha)
        elif cfg.rho_profile == "linear":
            f_vals = f_fn(x, cfg.rho_0, cfg.alpha)
        elif cfg.rho_profile == "step":
            f_vals = f_fn(x, cfg.rho_0, cfg.x_ion, cfg.alpha)
        else:
            raise ValueError(f"Unrecognised rho_profile identifier: {cfg.rho_profile}")

        b = dx**2 * f_vals

        # Anode constraint: φ̃(0) = alpha_bc, subtracted from the first entry.
        b[0] -= cfg.alpha_bc

        # Cathode constraint: φ̃(1) = 0, contributing nothing to the last entry.

        return b

    def analytical_solution(self) -> np.ndarray | None:
        """
        Evaluates the analytical solution at the interior nodes, where one exists.

        Returns
        -------
        phi : np.ndarray or None
            Length-N exact potential, or None if the configuration admits no
            closed form. Defined only for the linear charge distribution under
            homogeneous boundary conditions (V_discharge = 0).
        """
        cfg = self.config
        if cfg.rho_profile == "linear" and abs(cfg.alpha_bc) < 1e-10:
            fn = HET_EXACT_SOLUTIONS["linear"]
            return fn(self.x, cfg.rho_0, cfg.alpha)
        return None

    def summary(self) -> str:
        """Returns a one-line summary of the configuration and operator."""
        return (
            f"{self.config.summary()} | "
            f"κ(A)={self.kappa:.2f}"
        )


# -- Physical Problem Container ------------------------------------------------

class HETPhysicalProblem1D:
    """
    The discretised 1D HET plasma Poisson system using the Boeuf-Garrigues
    (1998) density profile as the prescribed analytical source.

    The non-dimensionalised Poisson equation is

        d²φ̃/dx̃² = -α · δñ(x̃)

    where δñ = (n_i - n_e)/n_0 is the prescribed non-dimensional net charge
    density and α = L²/λ_D² the dimensionless physical scaling.

    Dirichlet boundary constraints:

        φ̃(0) = V_discharge / φ_0   (anode)
        φ̃(1) = 0                   (cathode, grounded)

    The physical electric field is recovered by differentiation,

        Ẽ(x̃) = -dφ̃/dx̃ ≈ -(φ̃_{i+1} - φ̃_{i-1}) / (2Δx̃)

    and projected into physical units as E(x) = Ẽ · φ_0 / L [V/m].

    Attributes
    ----------
    config : HETPhysicalConfig
        Physical and numerical parameters of the instance.
    x : np.ndarray
        Length-N vector of interior node coordinates on [0, 1].
    dx : float
        Uniform mesh spacing, Δx̃ = 1/(N+1).
    A : np.ndarray
        N×N TST system operator (a = -2, b = 1).
    n_profile : np.ndarray
        Length-N Gaussian bulk density profile, n(x̃)/n_0.
    delta_n : np.ndarray
        Length-N prescribed net charge density, δñ = (n_i - n_e)/n_0.
    b : np.ndarray
        Length-N non-dimensional right-hand side.
    kappa : float
        2-norm condition number of A.

    References
    ----------
    Boeuf & Garrigues (1998), J. Appl. Phys. 84(7), 3541-3554.
    Hagelaar et al. (2002), Phys. Rev. E 62(1).
    """

    def __init__(self, cfg: "HETPhysicalConfig") -> None:
        """
        Assembles the operator, the prescribed profiles and the right-hand side.

        Parameters
        ----------
        cfg : HETPhysicalConfig
            Validated Boeuf-Garrigues physical configuration.

        Warns
        -----
        UserWarning
            If the assembled right-hand side implies a solution scale more than
            three times the expected α_bc + δ_0_factor, which indicates δ_0 has
            been set far from its physical value of order 1/α.
        """
        self.config = cfg
        N           = cfg.N
        self.dx     = 1.0 / (N + 1)
        self.x      = np.arange(1, N + 1) * self.dx

        # -- System Matrix Construction ----------------------------------------
        # Identical TST operator to the generic 1D Poisson case.
        self.A = (
            -2.0 * np.diag(np.ones(N))
            +  1.0 * np.diag(np.ones(N - 1), k=1)
            +  1.0 * np.diag(np.ones(N - 1), k=-1)
        )

        # -- Prescribed Analytical Distributions -------------------------------
        self.n_profile    = self._density_profile()
        self.delta_n      = self._charge_separation()

        # -- Right-Hand Side Assembly ------------------------------------------
        self.b = self._build_rhs()

        # -- Sanity check: solution scale should be O(alpha_bc) ----------------
        # The dominant contribution to the potential is the applied voltage,
        # so the solution norm should be comparable to alpha_bc = V_d/phi_0.
        # The space charge contribution is alpha * delta_0 = delta_0_factor.
        # Total expected scale: alpha_bc + delta_0_factor.
        expected_scale = cfg.alpha_bc + cfg.delta_0_factor
        rhs_scale      = np.max(np.abs(self.b)) / (cfg.alpha * self.dx**2 + 1e-14)

        # Warn if the RHS-implied solution scale is more than 3x the expected.
        if rhs_scale > 3.0 * expected_scale and expected_scale > 0:
            warnings.warn(
                f"RHS scale ({rhs_scale:.2f}) is {rhs_scale/expected_scale:.1f}x "
                f"larger than expected ({expected_scale:.2f}). "
                f"Check delta_0 ({cfg.delta_0:.2e}) — it may be too large. "
                f"Physical requirement: delta_0 ~ 1/alpha = {1/cfg.alpha:.2e}.",
                UserWarning,
            )

        # -- Condition Number Evaluation ---------------------------------------
        eigs       = np.abs(np.linalg.eigvalsh(self.A))
        self.kappa = float(eigs.max() / eigs.min())

    def _density_profile(self) -> np.ndarray:
        """
        Evaluates the Gaussian bulk plasma density profile approximating the
        empirical data of Boeuf & Garrigues (1998).

            n(x̃)/n_0 = n_min + (1 - n_min) · exp(-(x̃ - x_peak)² / (2σ_n²))

        The distribution peaks near the thruster exit plane (x̃ ≈ 0.75) and
        decays to n_min at the axial boundaries.

        Returns
        -------
        n : np.ndarray
            Length-N non-dimensional density profile.
        """
        cfg = self.config
        return (
            cfg.n_min
            + (1.0 - cfg.n_min)
            * np.exp(-((self.x - cfg.x_peak)**2) / (2.0 * cfg.sigma_n**2))
        )

    def _charge_separation(self) -> np.ndarray:
        """
        Evaluates the prescribed net charge density, δñ = (n_i - n_e)/n_0.

            δñ(x̃) = δ_0 · [exp(-x̃/σ_a) - exp(-(1-x̃)²/σ_c²)]

        This models the plasma sheath regions analytically: positive space
        charge near the anode (x̃ → 0), negative space charge near the cathode
        (x̃ → 1), separated by a quasi-neutral bulk.

        Returns
        -------
        delta_n : np.ndarray
            Length-N non-dimensional net charge density.
        """
        cfg = self.config
        anode_term   = np.exp(-self.x / cfg.sigma_anode)
        cathode_term = np.exp(-((1.0 - self.x)**2) / cfg.sigma_cath**2)
        return cfg.delta_0 * (anode_term - cathode_term)

    def _build_rhs(self) -> np.ndarray:
        """
        Assembles the non-dimensional right-hand side vector.

        At the interior nodes,

            b[i] = -α · Δx̃² · δñ(x̃_i)

        The fixed anode potential is subtracted from the first entry b[0];
        the grounded cathode contributes nothing to the last entry b[-1].

        Returns
        -------
        b : np.ndarray
            Length-N non-dimensional right-hand side vector.
        """
        cfg = self.config
        b   = -cfg.alpha * self.dx**2 * self.delta_n
        b[0]  -= cfg.alpha_bc    # Anode constraint: φ̃(0) = V_d/φ_0
        return b

    def electric_field(
        self,
        phi_nondim: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the physical electric field E(x) [V/m] from the non-dimensional
        potential φ̃.

        Second-order centred differences are used at the interior nodes and
        first-order one-sided differences at the axial boundaries. The boundary
        constraints φ̃(0) = alpha_bc and φ̃(1) = 0 are applied before
        differentiation, so the returned arrays span the closed domain including
        both boundary nodes.

        Field recovery: E = -dφ/dx = -(φ_0/L) · dφ̃/dx̃.

        Parameters
        ----------
        phi_nondim : np.ndarray
            Length-N non-dimensional potential at the interior nodes.

        Returns
        -------
        x_full : np.ndarray
            Length-(N+2) non-dimensional coordinates spanning [0, 1] inclusive
            of both boundary nodes.
        E_phys : np.ndarray
            Length-(N+2) dimensional electric field [V/m].
        """
        cfg = self.config
        N   = cfg.N
        dx  = self.dx

        # Augment the solution vector with the explicit boundary constraints.
        phi_full        = np.zeros(N + 2)
        phi_full[0]     = cfg.alpha_bc   # Anode
        phi_full[1:N+1] = phi_nondim
        phi_full[N+1]   = 0.0            # Cathode

        # Extended coordinate and field arrays.
        x_full   = np.linspace(0, 1, N + 2)
        E_nondim = np.zeros(N + 2)

        # Interior nodes: second-order centred difference.
        E_nondim[1:-1] = -(phi_full[2:] - phi_full[:-2]) / (2.0 * dx)

        # Boundary nodes: first-order one-sided difference.
        E_nondim[0]  = -(phi_full[1]  - phi_full[0])  / dx
        E_nondim[-1] = -(phi_full[-1] - phi_full[-2]) / dx

        # Rescale to dimensional physical units [V/m].
        E_phys = E_nondim * cfg.phi_0 / cfg.L

        return x_full, E_phys

    def summary(self) -> str:
        """Returns a one-line summary of the configuration and operator."""
        return f"{self.config.summary()} | κ(A)={self.kappa:.2f}"
