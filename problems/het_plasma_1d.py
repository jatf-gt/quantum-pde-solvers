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

from core.het_config import HETConfig
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