"""
Assembles the discretised 1D Hall Effect Thruster (HET) plasma Poisson boundary 
value problem.

Physical Problem Formulation
----------------------------
The electrostatic potential φ(x) within the 1D axial discharge channel 
satisfies the non-dimensionalised Poisson equation:

    d²φ̃/dx̃² = f̃(x̃)

where x̃ = x/L ∈ [0,1], φ̃ = φ/φ_0, and the source term f̃ encodes 
the net charge density linearly scaled by the parameter α = L²/λ_D²:

    f̃(x̃) = -α · (ñ_i(x̃) - ñ_e(x̃))

Boundary Constraints:
    φ̃(0) = V_discharge / φ_0   (Anode, non-dimensional discharge voltage)
    φ̃(1) = 0                   (Cathode, grounded potential)

Upon continuous domain discretisation utilizing N interior nodes and spatial 
mesh spacing Δx̃ = 1/(N+1), the linear system Au = b is constructed exactly 
as within the generic 1D Poisson methodology. The identical TST operator 
(a = -2, b = 1) is employed, with the physical source terms and boundary 
constraints fully assimilated into the right-hand side vector.

The formulated linear system is seamlessly processed by the established 
HHL and VQLS quantum solvers without necessitating algorithmic modifications; 
the physical characteristics reside entirely within the system assembly phase.

Reference: Laizet (PhD thesis), Hall Effect Thruster plasma modelling.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.het_config import HETConfig, HETPhysicalConfig
from core.source_functions import HET_SOURCE_FUNCTIONS
from core.exact_solutions import HET_EXACT_SOLUTIONS


# ── Problem Container ─────────────────────────────────────────────────────────

@dataclass
class HETPoissonProblem1D:
    """
    Encapsulates the discretised 1D HET plasma Poisson system, structured for 
    seamless integration with quantum solvers.

    This data structure mirrors the `PoissonProblem1D` interface, ensuring that 
    downstream quantum algorithmic functions (e.g., `hhl_solve`, `vqls_solve`) 
    can process the plasma configurations without architectural modification.

    Attributes
    ----------
    config : HETConfig
        Configuration structure encapsulating all physical and numerical parameters.
    x : np.ndarray
        Interior spatial node coordinates mapped to the non-dimensional domain [0,1].
    dx : float
        Uniform mesh spacing, Δx̃ = 1/(N+1).
    A : np.ndarray
        N×N TST system operator matrix (a = -2, b = 1). Structurally identical 
        to the standard 1D Poisson configuration.
    b : np.ndarray
        Right-hand side vector incorporating the scaled source term and Dirichlet boundaries.
    kappa : float
        Calculated 2-norm condition number of matrix A.
    b_phys : np.ndarray
        Right-hand side vector projected back into dimensional physical units 
        (facilitates independent diagnostics).
    """
    config: HETConfig
    x:      np.ndarray
    dx:     float
    A:      np.ndarray
    b:      np.ndarray
    kappa:  float
    b_phys: np.ndarray

    def __init__(self, cfg: HETConfig) -> None:
        self.config = cfg
        N           = cfg.N
        self.dx     = 1.0 / (N + 1)
        self.x      = np.arange(1, N + 1) * self.dx

        # ── System Matrix Construction ────────────────────────────────────────
        # The underlying TST operator mirrors the generic 1D Poisson methodology. 
        # The specific plasma physics are assimilated exclusively via the RHS vector.
        self.A = (
            -2.0 * np.diag(np.ones(N))
            +  1.0 * np.diag(np.ones(N - 1), k=1)
            +  1.0 * np.diag(np.ones(N - 1), k=-1)
        )

        # ── Right-Hand Side Assembly ──────────────────────────────────────────
        self.b      = self._build_rhs()
        self.b_phys = self.b * cfg.phi_0   # Dimensional projection for telemetry

        eigs       = np.abs(np.linalg.eigvalsh(self.A))
        self.kappa = float(eigs.max() / eigs.min())

        # ── Condition Number Evaluation ───────────────────────────────────────
        eigs       = np.abs(np.linalg.eigvalsh(self.A))
        self.kappa = float(eigs.max() / eigs.min())

    def _build_rhs(self) -> np.ndarray:
        """
        Assembles the non-dimensional right-hand side vector.

        The discretised equation evaluated at interior node i corresponds to:
            φ̃_{i+1} - 2φ̃_i + φ̃_{i-1} = Δx̃² · f̃(x̃_i)

        Following the incorporation of Dirichlet boundary constraints:
            b[0]   = Δx̃² · f̃(x̃_1) - φ̃_anode
            b[i]   = Δx̃² · f̃(x̃_i)         for 1 < i < N-1
            b[N-1] = Δx̃² · f̃(x̃_N) - φ̃_cathode

        The cathode constraint establishes φ̃(1) = 0, contributing a null value.
        The anode constraint enforces φ̃(0) = V_discharge / φ_0 = alpha_bc.
        """
        cfg = self.config
        dx  = self.dx
        x   = self.x

        # Evaluate the chosen analytical source distribution across interior coordinates.
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

        # Anode Constraint: φ̃(0) = alpha_bc (Subtracts from the primary entry)
        b[0] -= cfg.alpha_bc

        # Cathode Constraint: φ̃(1) = 0 (Contributes nothing to the terminal entry)

        return b

    def analytical_solution(self) -> np.ndarray | None:
        """
        Evaluates the analytical mathematical solution at the interior nodes.

        Presently defined exclusively for the linear charge distribution under 
        strictly homogeneous boundary conditions (V_discharge = 0). 
        Yields None for unmapped configurations.
        """
        cfg = self.config
        if cfg.rho_profile == "linear" and abs(cfg.alpha_bc) < 1e-10:
            fn = HET_EXACT_SOLUTIONS["linear"]
            return fn(self.x, cfg.rho_0, cfg.alpha)
        return None

    def summary(self) -> str:
        """Generates a concise execution string detailing the configuration and matrix status."""
        return (
            f"{self.config.summary()} | "
            f"κ(A)={self.kappa:.2f}"
        )
    

# ── Physical Problem Container ────────────────────────────────────────────────

class HETPhysicalProblem1D:
    """
    Constructs the discretised 1D HET plasma Poisson system, utilising the 
    Boeuf-Garrigues (1998) density profile as the prescribed analytical source term.

    The non-dimensionalised Poisson equation is formulated as:
        d²φ̃/dx̃² = -α · δñ(x̃)

    where δñ = (n_i - n_e)/n_0 characterises the prescribed non-dimensional net 
    charge density, and α = L²/λ_D² constitutes the dimensionless physical scaling.

    Dirichlet Boundary Constraints:
        φ̃(0) = V_discharge / φ_0   (Anode)
        φ̃(1) = 0                   (Cathode, grounded potential)

    The macroscopic physical electric field is recovered via differentiation:
        Ẽ(x̃) = -dφ̃/dx̃ ≈ -(φ̃_{i+1} - φ̃_{i-1}) / (2Δx̃)
    Projected into physical units as: 
        E(x) = Ẽ · φ_0 / L [V/m]

    References
    ----------
    - Boeuf & Garrigues (1998), J. Appl. Phys. 84(7), 3541-3554.
    - Hagelaar et al. (2002), Phys. Rev. E 62(1).
    """

    def __init__(self, cfg: "HETPhysicalConfig") -> None:
        from core.het_config import HETPhysicalConfig
        self.config = cfg
        N           = cfg.N
        self.dx     = 1.0 / (N + 1)
        self.x      = np.arange(1, N + 1) * self.dx

        # ── System Matrix Construction ────────────────────────────────────────
        # Identical TST operator to the generic 1D Poisson methodology.
        self.A = (
            -2.0 * np.diag(np.ones(N))
            +  1.0 * np.diag(np.ones(N - 1), k=1)
            +  1.0 * np.diag(np.ones(N - 1), k=-1)
        )

        # ── Prescribed Analytical Distributions ───────────────────────────────
        self.n_profile    = self._density_profile()
        self.delta_n      = self._charge_separation()

        # ── Right-Hand Side Assembly ──────────────────────────────────────────
        self.b = self._build_rhs()

        # ── Sanity check: solution scale should be O(alpha_bc) ───────────────
        # The dominant contribution to the potential is the applied voltage,
        # so the solution norm should be comparable to alpha_bc = V_d/phi_0.
        # The space charge contribution is alpha * delta_0 = delta_0_factor.
        # Total expected scale: alpha_bc + delta_0_factor.
        expected_scale = cfg.alpha_bc + cfg.delta_0_factor
        rhs_scale      = np.max(np.abs(self.b)) / (cfg.alpha * self.dx**2 + 1e-14)

        # Warn if the RHS-implied solution scale is more than 3x the expected.
        if rhs_scale > 3.0 * expected_scale and expected_scale > 0:
            import warnings
            warnings.warn(
                f"RHS scale ({rhs_scale:.2f}) is {rhs_scale/expected_scale:.1f}x "
                f"larger than expected ({expected_scale:.2f}). "
                f"Check delta_0 ({cfg.delta_0:.2e}) — it may be too large. "
                f"Physical requirement: delta_0 ~ 1/alpha = {1/cfg.alpha:.2e}.",
                UserWarning,
            )

        # ── Condition Number Evaluation ───────────────────────────────────────
        eigs       = np.abs(np.linalg.eigvalsh(self.A))
        self.kappa = float(eigs.max() / eigs.min())

    def _density_profile(self) -> np.ndarray:
        """
        Evaluates the Gaussian plasma density profile approximating the empirical 
        data from Boeuf-Garrigues (1998).

        Formulation:
            n(x̃)/n_0 = n_min + (1 - n_min) · exp(-(x̃ - x_peak)² / (2σ_n²))

        The distribution peaks in proximity to the thruster exit plane (x̃ ≈ 0.75) 
        and decays to n_min at the defined axial boundaries.
        """
        cfg = self.config
        return (
            cfg.n_min
            + (1.0 - cfg.n_min)
            * np.exp(-((self.x - cfg.x_peak)**2) / (2.0 * cfg.sigma_n**2))
        )

    def _charge_separation(self) -> np.ndarray:
        """
        Evaluates the prescribed non-dimensional net charge density, δñ = (n_i - n_e)/n_0.

        This distribution analytically models the plasma sheath regions:
          - Positive space charge accumulation near the anode (x̃ → 0)
          - Negative space charge accumulation near the cathode (x̃ → 1)
        Interspersed by a quasi-neutral bulk plasma regime.

        Formulation:
            δñ(x̃) = δ_0 · [exp(-x̃/σ_a) - exp(-(1-x̃)²/σ_c²)]
        """
        cfg = self.config
        anode_term   = np.exp(-self.x / cfg.sigma_anode)
        cathode_term = np.exp(-((1.0 - self.x)**2) / cfg.sigma_cath**2)
        return cfg.delta_0 * (anode_term - cathode_term)

    def _build_rhs(self) -> np.ndarray:
        """
        Assembles the non-dimensional right-hand side vector encoding the 
        physical source topology and boundary states.

        Evaluation at interior nodes:
            b[i] = -α · Δx̃² · δñ(x̃_i)

        The fixed anode potential is subtracted from the primary entry b[0], 
        whilst the grounded cathode contributes nothing to the terminal entry b[-1].
        """
        cfg = self.config
        b   = -cfg.alpha * self.dx**2 * self.delta_n
        b[0]  -= cfg.alpha_bc    # Anode constraint: φ̃(0) = V_d/φ_0
        return b

    def electric_field(self, phi_nondim: np.ndarray) -> np.ndarray:
        """
        Computes the macroscopic physical electric field E(x) [V/m] derived from 
        the non-dimensional spatial potential array φ̃.

        The derivation employs second-order centred differences across interior 
        nodes and strictly first-order one-sided differences at the axial boundaries.

        Boundary Constraints applied prior to differentiation:
            φ̃(0) = alpha_bc,  φ̃(1) = 0

        Physical Field Recovery:
            E = -dφ/dx = -(φ_0/L) · dφ̃/dx̃
        """
        cfg = self.config
        N   = cfg.N
        dx  = self.dx

        # Augment the solution vector with explicit boundary constraints.
        phi_full        = np.zeros(N + 2)
        phi_full[0]     = cfg.alpha_bc   # Anode
        phi_full[1:N+1] = phi_nondim
        phi_full[N+1]   = 0.0            # Cathode

        # Instantiate extended arrays.
        x_full   = np.linspace(0, 1, N + 2)
        E_nondim = np.zeros(N + 2)

        # Interior Nodes: Second-order centred difference.
        E_nondim[1:-1] = -(phi_full[2:] - phi_full[:-2]) / (2.0 * dx)

        # Boundary Nodes: First-order one-sided difference.
        E_nondim[0]  = -(phi_full[1]  - phi_full[0])  / dx
        E_nondim[-1] = -(phi_full[-1] - phi_full[-2]) / dx

        # Rescale differentiated field to dimensional physical units [V/m].
        E_phys = E_nondim * cfg.phi_0 / cfg.L

        return x_full, E_phys

    def summary(self) -> str:
        """Generates a concise execution string detailing configuration and matrix status."""
        return f"{self.config.summary()} | κ(A)={self.kappa:.2f}"