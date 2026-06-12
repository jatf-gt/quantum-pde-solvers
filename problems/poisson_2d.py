"""
Assembles the discretised two-dimensional Poisson boundary value problem.

This module formulates the 2D Poisson equation on the domain [0,1]² employing a 
uniform Cartesian mesh and Dirichlet boundary conditions across all perimeters. 
Utilising a line-Jacobi decomposition, the 2D system is reduced to a sequence of 
1D Toeplitz Symmetric Tridiagonal (TST) sub-problems, each structurally compatible 
with the Harrow-Hassidim-Lloyd (HHL) quantum solver pipeline.

Reference: Ghafourpour & Laizet (2025), Section III B and Equation (9).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from core.config import SimConfig2D, ClassicalConfig2D
from core.source_functions import SOURCE_FUNCTIONS_2D


# ── Grid Construction ─────────────────────────────────────────────────────────

def build_grid_2d(N: int) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Constructs the two-dimensional interior spatial mesh for an N×N system.

    The continuous domain spans [0,1]². Boundary nodes are explicitly excluded 
    from the coordinate arrays, as their mathematical contributions are assimilated 
    into the right-hand side vector via Dirichlet boundary conditions.

    Parameters
    ----------
    N : int
        Number of interior nodes along each spatial dimension.

    Returns
    -------
    X : np.ndarray
        (N, N) matrix of x-coordinates corresponding to interior nodes.
    Y : np.ndarray
        (N, N) matrix of y-coordinates corresponding to interior nodes.
    h : float
        Uniform spatial mesh spacing, identical for both axes.
    """
    h = 1.0 / (N + 1)
    coords = np.arange(1, N + 1) * h
    X, Y   = np.meshgrid(coords, coords, indexing="ij")
    return X, Y, h


# ── Row-Wise System Assembly ──────────────────────────────────────────────────

def build_row_tst_matrix(N: int) -> np.ndarray:
    """
    Constructs the N×N TST matrix governing a singular row within the line-Jacobi update.

    The 2D line-Jacobi stencil dictates diagonal parameters a = -4 and off-diagonal 
    parameters b = 1. This distinguishes it structurally from the pure 1D Poisson 
    operator (where a = -2). 

    The output is maintained as a dense NumPy array, as the constrained system 
    dimensions (N <= 32 per sub-problem) render sparse storage architectures unnecessary.
    """
    A = (
        -4.0 * np.diag(np.ones(N))
        +  1.0 * np.diag(np.ones(N - 1), k=1)
        +  1.0 * np.diag(np.ones(N - 1), k=-1)
    )
    return A


def build_row_rhs(
    j:        int,
    u_prev:   np.ndarray,
    X:        np.ndarray,
    Y:        np.ndarray,
    h:        float,
    cfg:      SimConfig2D,
) -> np.ndarray:
    """
    Assembles the right-hand side vector for the line-Jacobi update of row j.

    The iterative update equation for interior row j (0-indexed, corresponding 
    to physical coordinate y_j = (j+1)·h) is given by:

        u^{n+1}_{i+1,j} - 4·u^{n+1}_{i,j} + u^{n+1}_{i-1,j} 
            = h²·f(x_i, y_j) - (u^n_{i,j-1} + u^n_{i,j+1})

    Dirichlet boundary conditions applied along the x-axis enforce:
        u(0, y_j)   = bc_x0   -> subtracts from b[0]
        u(1, y_j)   = bc_x1   -> subtracts from b[N-1]

    Dirichlet boundary conditions applied along the y-axis enforce:
        j = 0       -> u^n_{i, -1}  = bc_y0  (Bottom perimeter)
        j = N-1     -> u^n_{i,  N}  = bc_y1  (Top perimeter)

    Parameters
    ----------
    j : int
        Zero-indexed row identifier.
    u_prev : np.ndarray
        (N, N) physical solution array extracted from the preceding Jacobi iteration.
    X, Y : np.ndarray
        Coordinate matrices generated via build_grid_2d.
    h : float
        Uniform spatial mesh spacing.
    cfg : SimConfig2D
        Configuration structure containing analytical source mappings and boundary constraints.
    """
    N  = cfg.N
    f  = SOURCE_FUNCTIONS_2D[cfg.source_fn]

    # Isolate the analytical source term contribution for the specified row.
    rhs = h**2 * f(X[:, j], Y[:, j])

    # Assimilate y-axis neighbour contributions from the preceding iterative step.
    if j == 0:
        rhs -= cfg.bc_y0 * np.ones(N)
    else:
        rhs -= u_prev[:, j - 1]

    if j == N - 1:
        rhs -= cfg.bc_y1 * np.ones(N)
    else:
        rhs -= u_prev[:, j + 1]

    # Assimilate x-axis Dirichlet boundary conditions into the terminal vector entries.
    rhs[0]  -= cfg.bc_x0
    rhs[-1] -= cfg.bc_x1

    return rhs


def condition_number_2d(N: int) -> float:
    """
    Computes the 2-norm condition number, κ(A), for the 2D line-Jacobi TST matrix.

    Analytical derivation (Appendix B.1 of the primary reference) dictates that 
    κ(A) approaches 3 asymptotically as N -> ∞. This demonstrates substantially 
    superior conditioning compared to the O(N²) scaling inherent to the 1D system, 
    constituting the primary mathematical justification for the efficiency of the 
    sub-problem quantum resolution.
    """
    A = build_row_tst_matrix(N)
    eigs = np.abs(np.linalg.eigvalsh(A))
    return float(eigs.max() / eigs.min())


# ── Packaged Problem Container ────────────────────────────────────────────────

@dataclass
class PoissonProblem2D:
    """
    Encapsulates all discretised parameters and operators for a 2D benchmark execution.

    Attributes
    ----------
    config : SimConfig2D | ClassicalConfig2D
        Configuration parameters governing the problem instance.
    X, Y : np.ndarray
        (N, N) spatial coordinate matrices for all interior nodes.
    h : float
        Uniform spatial mesh spacing.
    A_row : np.ndarray
        The N×N TST matrix (a=-4, b=1) universally applied across all row 
        sub-problems due to spatial operator uniformity.
    kappa_row : float
        Condition number corresponding to A_row.
    u_init : np.ndarray
        (N, N) zero-initialised matrix serving as the iterative cold-start parameter.

    Note
    ----
    The full N²×N² block-banded discrete system matrix is intentionally excluded 
    from standard instantiation to preserve memory efficiency. It is constructed 
    exclusively on-demand via `build_full_matrix()` for condition number diagnostics 
    and classical reference benchmarking.
    """
    config:    Union[SimConfig2D, ClassicalConfig2D]
    X:         np.ndarray
    Y:         np.ndarray
    h:         float
    A_row:     np.ndarray
    kappa_row: float
    u_init:    np.ndarray

    def __init__(self, cfg: Union[SimConfig2D, ClassicalConfig2D]) -> None:
        self.config    = cfg
        self.X, self.Y, self.h = build_grid_2d(cfg.N)
        self.A_row     = build_row_tst_matrix(cfg.N)
        self.kappa_row = condition_number_2d(cfg.N)
        self.u_init    = np.zeros((cfg.N, cfg.N))

    def get_row_system(
        self,
        j:      int,
        u_prev: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the governing operator and right-hand side vector for row j.

        This constitutes the primary interface accessed continuously by the 2D HHL 
        solver during the iterative cycle. The operator A_row remains static, whereas 
        b_row updates dynamically relative to the preceding solution state.
        """
        b_row = build_row_rhs(
            j, u_prev, self.X, self.Y, self.h, self.config
        )
        return self.A_row, b_row

    def build_full_matrix(self) -> np.ndarray:
        """
        Constructs the comprehensive N²×N² block-banded system matrix.

        Formulated according to Equation (8) of the primary reference:
            A_full = (1/h²) * [T  I  0  ...]
                              [I  T  I  ...]
                              [0  I  T  ...]
                              [     ...    ]
        Where T represents the N×N TST operator (a=-4, b=1), and I represents 
        the N×N identity matrix.

        This construction is strictly reserved for classical reference resolutions 
        and holistic condition number analysis. The line-Jacobi methodology 
        expressly bypasses the formation of this global operator.
        """
        N  = self.config.N
        T  = self.A_row.copy()
        I  = np.eye(N)
        A_full = np.zeros((N * N, N * N))

        for j in range(N):
            row_start = j * N
            row_end   = row_start + N
            
            # Populate the principal diagonal block
            A_full[row_start:row_end, row_start:row_end] = T
            
            # Populate the off-diagonal identity blocks (inter-row coupling)
            if j > 0:
                col_start = (j - 1) * N
                A_full[row_start:row_end, col_start:col_start + N] = I
            if j < N - 1:
                col_start = (j + 1) * N
                A_full[row_start:row_end, col_start:col_start + N] = I

        # The 1/h² prefactor is analytically absorbed into the right-hand side 
        # evaluation to maintain structural parity with the 1D methodology.
        return A_full

    def build_full_rhs(self) -> np.ndarray:
        """
        Assembles the comprehensive N²-dimensional right-hand side vector.

        Data is ordered via column-major sequence (j varies slowest, i varies fastest), 
        aligning strictly with the block structure formulated in build_full_matrix.
        """
        N   = self.config.N
        f   = SOURCE_FUNCTIONS_2D[self.config.source_fn]
        h   = self.h
        rhs = np.zeros(N * N)

        for j in range(N):
            b_row = h**2 * f(self.X[:, j], self.Y[:, j])
            b_row[0]  -= self.config.bc_x0
            b_row[-1] -= self.config.bc_x1
            
            if j == 0:
                b_row -= self.config.bc_y0
            if j == N - 1:
                b_row -= self.config.bc_y1
                
            rhs[j * N:(j + 1) * N] = b_row

        return rhs

    # TODO: choose the most appropiate benchmark(s) moving forward (as f(project direction))
    # def classical_reference_solve(self) -> np.ndarray:
    #     """
    #     Solve the full 2D system with NumPy's direct solver.

    #     Returns the solution as an (N, N) array, matching the shape
    #     of the iterative solver output.

    #     This is the ground truth that the line-Jacobi HHL result is
    #     compared against in the benchmark — equivalent to the Thomas
    #     algorithm on the refined mesh used in the paper (Section IV E).
    #     """
    #     A_full = self.build_full_matrix()
    #     b_full = self.build_full_rhs()
    #     u_flat = np.linalg.solve(A_full, b_full)
    #     return u_flat.reshape((self.config.N, self.config.N), order="C")
    
    def classical_reference_solve(
        self,
        refine_factor: int   = 19,
        target_h:      float = None,
        analytical_fn        = None,
    ) -> np.ndarray:
        """
        Computes the high-fidelity reference solution and extracts the coarse spatial nodes.

        Execution is governed by a strict hierarchical priority system:

        Mode 1: Analytical Solution (Highest Priority)
            Evaluates the precise mathematical solution directly at the coarse nodes. 
            Utilised exclusively when a closed-form derivation is known. The provided 
            `analytical_fn` must accept (X, Y) coordinate matrices.

        Mode 2: Targeted Resolution
            Enforces a specific fine mesh spacing parameter (`target_h`). Discarding 
            the power-of-two operational constraint, it computes N_fine = round(1/target_h) - 1. 
            Facilitates direct replication of literature-specific mesh sizes (e.g., h=1/153).

        Mode 3: Factorial Refinement (Default)
            Scales the operational mesh by an integer multiplier: N_fine = N * refine_factor. 
            The integer scaling ensures that coarse spatial nodes align perfectly with fine 
            nodal intersections, precluding the necessity for interpolation. 
            Mathematically verified via:
                x_i^{coarse} = i * h_coarse = i * refine_factor * h_fine = x_{i*refine_factor}^{fine}

        Parameters
        ----------
        refine_factor : int
            Integer multiplier applied to the coarse mesh density (Mode 3).
        target_h : float, optional
            Absolute target value for the fine mesh spacing (Mode 2).
        analytical_fn : Callable, optional
            Function mapping (X, Y) coordinate matrices to exact solution states (Mode 1).

        Returns
        -------
        u_coarse : np.ndarray
            (N, N) matrix representing the reference solution evaluated at the coarse coordinates.
        """
        # ── Mode 1: Analytical Solution ───────────────────────────────────────
        if analytical_fn is not None:
            return analytical_fn(self.X, self.Y)

        # ── Mode 2 / 3: Numerical Fine-Mesh Reference ─────────────────────────
        if target_h is not None:
            N_fine = max(self.config.N, int(round(1.0 / target_h)) - 1)
        else:
            N_fine = self.config.N * refine_factor

        if N_fine < self.config.N:
            raise ValueError(
                f"Computed fine mesh N_fine={N_fine} is inferior to the base resolution "
                f"N={self.config.N}. Augment refine_factor or diminish target_h."
            )

        # Instantiate a purely classical configuration (bypassing quantum constraints)
        from core.config import ClassicalConfig2D
        cfg_fine = ClassicalConfig2D(
            N=N_fine,
            source_fn=self.config.source_fn,
            tol=1e-10,
            max_iter=5000,
            bc_x0=self.config.bc_x0,
            bc_x1=self.config.bc_x1,
            bc_y0=self.config.bc_y0,
            bc_y1=self.config.bc_y1,
        )

        # Delayed import to circumvent circular dependency resolution errors
        from solvers.classical.thomas_2d import thomas_solve_2d

        prob_fine   = PoissonProblem2D(cfg_fine)
        result_fine = thomas_solve_2d(prob_fine)

        if not result_fine.converged:
            import warnings
            warnings.warn(
                f"Fine-mesh reference solver failed to converge within "
                f"{cfg_fine.max_iter} iterations "
                f"(Terminal residual = {result_fine.iteration_errors[-1]:.2e}). "
                f"Adjust max_iter limits or relax the convergence threshold.",
                RuntimeWarning,
            )

        u_fine = result_fine.u

        # Coordinate Extraction Protocol
        # A coarse node 'i' possesses the coordinate x_i = i * (N_fine+1)/(N+1) * h_fine. 
        # Precise nodal alignment requires (N_fine+1)/(N+1) to resolve to an integer.
        stride = (N_fine + 1) / (self.config.N + 1)

        if abs(stride - round(stride)) < 1e-9:
            s = int(round(stride))
            coarse_indices = np.array([s * i - 1 for i in range(1, self.config.N + 1)])
            u_coarse = u_fine[np.ix_(coarse_indices, coarse_indices)]
        else:
            u_coarse = _bilinear_interpolate(
                u_fine, self.config.N, N_fine
            )

        return u_coarse

    def coarse_direct_solve(self) -> np.ndarray:
        """
        Computes the analytical resolution of the unrefined N²×N² discrete system.

        This protocol evaluates the precise solution of the coarse matrix geometry, 
        serving exclusively as a diagnostic mechanism to verify line-Jacobi convergence 
        stability and absolute discrete residuals. It does not represent the refined 
        reference methodology utilised in the primary literature evaluation.
        """
        A_full = self.build_full_matrix()
        b_full = self.build_full_rhs()
        u_flat = np.linalg.solve(A_full, b_full)
        return u_flat.reshape((self.config.N, self.config.N), order="C")

    def summary(self) -> str:
        """Generates a concise execution summary string detailing the current system configuration."""
        cfg = self.config
        return (
            f"2D N={cfg.N}, f={cfg.source_fn}, "
            f"BCs=({cfg.bc_x0},{cfg.bc_x1},{cfg.bc_y0},{cfg.bc_y1}), "
            f"ε={cfg.epsilon:.4g}, tol={cfg.tol:.1e}, "
            f"κ(A_row)={self.kappa_row:.4f}"
        )


# ── Private Utility Methods ───────────────────────────────────────────────────

def _bilinear_interpolate(
    u_fine:   np.ndarray,
    N_coarse: int,
    N_fine:   int,
) -> np.ndarray:
    """
    Executes bilinear interpolation to map data from a high-resolution mesh 
    onto a coarse analytical grid.

    This sub-routine is triggered exclusively when the refined spatial mesh 
    spacing does not partition the coarse boundaries uniformly, resulting in 
    a misalignment between coarse spatial coordinates and fine nodal intersections.
    """
    h_fine   = 1.0 / (N_fine   + 1)
    h_coarse = 1.0 / (N_coarse + 1)

    u_coarse = np.zeros((N_coarse, N_coarse))

    for i in range(N_coarse):
        for j in range(N_coarse):
            xc = (i + 1) * h_coarse
            yc = (j + 1) * h_coarse

            fi = xc / h_fine - 1.0
            fj = yc / h_fine - 1.0

            i0 = int(np.floor(fi))
            j0 = int(np.floor(fj))
            i1 = min(i0 + 1, N_fine - 1)
            j1 = min(j0 + 1, N_fine - 1)
            i0 = max(i0, 0)
            j0 = max(j0, 0)

            tx = fi - np.floor(fi)
            ty = fj - np.floor(fj)

            u_coarse[i, j] = (
                (1 - tx) * (1 - ty) * u_fine[i0, j0]
                +      tx * (1 - ty) * u_fine[i1, j0]
                + (1 - tx) * ty * u_fine[i0, j1]
                +      tx * ty * u_fine[i1, j1]
            )

    return u_coarse