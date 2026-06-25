"""
Implements the Variational Quantum Linear Solver (VQLS) for 1D Poisson systems 
characterised by a Toeplitz Symmetric Tridiagonal (TST) matrix structure.

Public Interface
----------------
vqls_solve(problem) : 
    High-level wrapper accepting a `PoissonProblem1D` object, returning a 
    standardised `VQLSSolverResult`.
vqls_solve_system(A, b, config) : 
    Low-level core routine accepting raw numerical arrays. Architected to 
    allow the 2D line-Jacobi solver (Phase 3) to bypass problem container 
    instantiation, significantly reducing computational overhead.

Methodology
-----------
The solver methodology adheres strictly to the formulation detailed by 
Bravo-Prieto et al. (2023):
  1. Decompose the system matrix A into a sum of Pauli unitaries (LCU).
  2. Prepare a hardware-efficient parameterised ansatz |x(θ)>.
  3. Minimise the globally normalised cost function: C(θ) = 1 - |<b|A|x>|² / <x|A†A|x>.
  4. Recover the physical solution dimensionality via proportionality constant projection.

Framework: PennyLane
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from problems.poisson_1d import PoissonProblem1D
from solvers.quantum.result import SolverResult
from solvers.quantum.vqls_utils import (
    pauli_decompose_normalised,
    build_cost_function,
    recover_solution,
    n_params,
)


# ── VQLS Configuration ────────────────────────────────────────────────────────

@dataclass
class VQLSConfig:
    """
    Encapsulates the hyperparameters governing the VQLS variational optimisation.

    Attributes
    ----------
    n_layers : int
        Total number of entangling layers comprising the ansatz. Increased depth 
        enhances expressivity at the cost of optimisation complexity. Empirical 
        baselines suggest 3-5 layers for N=8 (3 qubits) and 5-8 layers for N=16 (4 qubits).
    optimiser : str
        Identifier for the SciPy optimisation algorithm. 'COBYLA' provides a 
        gradient-free, robust approach for constrained system dimensions. 
        'L-BFGS-B' accelerates convergence for larger parameter spaces but 
        necessitates gradient evaluations.
    max_iter : int
        Absolute ceiling on the optimiser iteration count.
    tol : float
        Convergence tolerance threshold applied to the objective cost function.
    init_params : Optional[np.ndarray], default=None
        Initial variational parameter vector. If None, parameters are randomly 
        initialised within [0, 2π] constrained by `random_seed`.
    random_seed : int
        Seed value ensuring reproducible pseudo-random initialisation.
    device_name : str
        Designated PennyLane simulation device (e.g., 'default.qubit' for exact 
        statevector simulation).
    verbose : bool
        Boolean flag dictating the standard output of iterative cost trajectory diagnostics.
    """
    n_layers:    int   = 4
    optimiser:   str   = "COBYLA"
    max_iter:    int   = 500
    tol:         float = 1e-6
    init_params: Optional[np.ndarray] = field(default=None, repr=False)
    random_seed: int   = 42
    device_name: str   = "default.qubit"
    verbose:     bool  = False


# More layers and iterations needed for physically scaled problems.
DEFAULT_VQLS_CONFIG = VQLSConfig(
    n_layers  = 6,
    optimiser = "COBYLA",
    max_iter  = 300,    # per restart — 3 restarts = 900 total
    tol       = 1e-6,
    random_seed = 42,
)

# ── Extended Result Container ─────────────────────────────────────────────────

@dataclass
class VQLSSolverResult(SolverResult):
    """
    Standardised solver output extended with VQLS-specific optimisation diagnostics.

    Inherits core attributes (u, solver, raw_state, prop_const, euclidean_residual) 
    from `SolverResult` and appends variational telemetry.

    Attributes
    ----------
    final_cost : float
        Terminal value of the objective function C(θ) upon convergence (0 = optimal).
    n_circuit_evals : int
        Aggregate count of cost function evaluations performed during optimisation.
    optimiser_success : bool
        Boolean flag indicating successful convergence reported by the SciPy optimiser.
    cost_history : List[float]
        Sequential trajectory of cost evaluations at each iteration.
    optimal_params : Optional[np.ndarray]
        The optimally converged parameter vector, θ*.
    n_layers : int
        Total number of entangling ansatz layers utilised.
    n_parameters : int
        Total dimensionality of the variational parameter space.
    """
    final_cost:        float            = 0.0
    n_circuit_evals:   int              = 0
    optimiser_success: bool             = False
    cost_history:      List[float]      = field(default_factory=list, repr=False)
    optimal_params:    Optional[np.ndarray] = field(default=None, repr=False)
    n_layers:          int              = 0
    n_parameters:      int              = 0


# ── Public High-Level Interface ───────────────────────────────────────────────

def vqls_solve(
    problem: PoissonProblem1D,
    config:  VQLSConfig = DEFAULT_VQLS_CONFIG,
) -> VQLSSolverResult:
    """
    Resolves the 1D Poisson system Au = b utilising the VQLS algorithm.

    Operates as a procedural wrapper around the core `vqls_solve_system` 
    sub-routine, unpacking the `PoissonProblem1D` data structure and 
    packaging the numerical outputs into a unified `VQLSSolverResult`.

    Parameters
    ----------
    problem : PoissonProblem1D
        Discretised 1D problem instance defining the linear system.
    config : VQLSConfig, default=DEFAULT_VQLS_CONFIG
        Hyperparameter structure governing the variational optimisation.

    Returns
    -------
    VQLSSolverResult
        Standardised result object containing the physical solution and 
        associated optimisation diagnostics.
    """
    return vqls_solve_system(
        A       = problem.A,
        b       = problem.b,
        config  = config,
    )


# ── Core Algorithmic Sub-Routine ──────────────────────────────────────────────

def vqls_solve_system(
    A:      np.ndarray,
    b:      np.ndarray,
    config: VQLSConfig = DEFAULT_VQLS_CONFIG,
) -> VQLSSolverResult:
    """
    Resolves the linear system Au = b employing the VQLS algorithm directly 
    on raw NumPy arrays.

    Incorporates an automated restart heuristic: should the initial optimisation 
    phase fail to satisfy the target cost tolerance, the solver iteratively 
    restarts from the optimal parameter vector discovered, employing progressively 
    reduced step sizes. This architecture constitutes a critical enhancement for 
    ill-conditioned systems (e.g., severe condition numbers, elevated RHS norms, 
    non-homogeneous boundary constraints).
    
    Parameters
    ----------
    A : np.ndarray
        N×N TST system matrix (must be Hermitian).
    b : np.ndarray
        Right-hand side target vector of length N.
    config : VQLSConfig, default=DEFAULT_VQLS_CONFIG
        Hyperparameter structure governing the variational optimisation.

    Returns
    -------
    VQLSSolverResult
        Populated data structure encapsulating the solution and full telemetry.
    """
    N = len(b)
    n_qubits = int(np.log2(N))

    if 2**n_qubits != N:
        raise ValueError(
            f"System size N={N} must be a power of 2 for amplitude encoding."
        )

    _validate_system(A, b)

    b_norm_factor = float(np.linalg.norm(b))
    if b_norm_factor < 1e-14:
        raise ValueError("RHS vector b is numerically zero.")
    b_norm = b / b_norm_factor

    # ── Pauli Decomposition ───────────────────────────────────────────────────
    pauli_terms, A_norm_factor = pauli_decompose_normalised(
        N         = N,
        main_diag = A[0, 0],
        off_diag  = A[0, 1],
    )

    if config.verbose:
        print(
            f"  VQLS: N={N}, n_qubits={n_qubits}, "
            f"n_layers={config.n_layers}, "
            f"LCU terms={len(pauli_terms)}, "
            f"||A||_2={A_norm_factor:.4f}, "
            f"||b||={b_norm_factor:.4e}"
        )

    # ── Parameter Initialisation ──────────────────────────────────────────────
    n_p = n_params(n_qubits, config.n_layers)

    if config.init_params is not None:
        if len(config.init_params) != n_p:
            raise ValueError(
                f"init_params has length {len(config.init_params)} but "
                f"ansatz requires {n_p} parameters."
            )
        theta_init = config.init_params.copy()
    else:
        rng        = np.random.default_rng(config.random_seed)
        theta_init = rng.uniform(0, 2 * np.pi, size=n_p)

    # ── Objective Function Construction ───────────────────────────────────────
    cost_fn = build_cost_function(
        pauli_terms  = pauli_terms,
        b_norm       = b_norm,
        n_qubits     = n_qubits,
        n_layers     = config.n_layers,
        device_name  = config.device_name,
    )

    # ── Optimisation with Restart Heuristics ──────────────────────────────────
    # Execute up to n_restarts sequential optimisation phases. Each phase 
    # initiates from the optimal parameter configuration identified previously, 
    # applying a monotonically decreasing COBYLA step size (rhobeg) to refine 
    # the solution vector.
    # This protocol is crucial for complex landscapes: the initial pass with 
    # a large rhobeg explores the global topography, whereas subsequent passes 
    # with a diminished rhobeg refine the targeted local minima.
    n_restarts     = 3
    rhobeg_values  = [0.5, 0.1, 0.01]
    cost_history:  list[float] = []
    total_evals    = 0
    best_params    = theta_init.copy()
    best_cost      = float(cost_fn(theta_init))

    for restart_idx in range(n_restarts):
        rhobeg = rhobeg_values[restart_idx]

        if config.verbose:
            print(
                f"  Restart {restart_idx+1}/{n_restarts}: "
                f"rhobeg={rhobeg}, starting cost={best_cost:.6f}"
            )

        opt_result = minimize(
            fun     = cost_fn,
            x0      = best_params,
            method  = "COBYLA",
            tol     = config.tol,
            options = {
                "maxiter": config.max_iter,
                "rhobeg":  rhobeg,
            },
        )

        total_evals += int(opt_result.nfev)
        current_cost = float(opt_result.fun)
        cost_history.append(current_cost)

        if current_cost < best_cost:
            best_cost   = current_cost
            best_params = opt_result.x.copy()

        if config.verbose:
            print(
                f"    → cost={current_cost:.6f}, "
                f"evals={opt_result.nfev}, "
                f"success={opt_result.success}"
            )

        # Terminate heuristic loops prematurely if the target tolerance is achieved.
        if best_cost <= config.tol:
            break

    optimal_params    = best_params
    final_cost        = best_cost
    optimiser_success = final_cost <= config.tol * 10

    if config.verbose:
        print(
            f"  VQLS final: cost={final_cost:.6f}, "
            f"total_evals={total_evals}, "
            f"converged={optimiser_success}"
        )

    # ── Dimensionality Recovery ───────────────────────────────────────────────
    u, c = recover_solution(
        params      = optimal_params,
        A           = A,
        b           = b,
        n_qubits    = n_qubits,
        n_layers    = config.n_layers,
        device_name = config.device_name,
    )

    residual = float(
        np.linalg.norm(A @ u - b) / np.linalg.norm(b)
    )

    return VQLSSolverResult(
        u                 = u,
        solver            = "VQLS",
        raw_state         = None,
        prop_const        = c,
        euclidean_residual= residual,
        final_cost        = final_cost,
        n_circuit_evals   = total_evals,
        optimiser_success = optimiser_success,
        cost_history      = cost_history,
        optimal_params    = optimal_params,
        n_layers          = config.n_layers,
        n_parameters      = n_p,
    )


# ── Private Utility Methods ───────────────────────────────────────────────────

def _validate_system(A: np.ndarray, b: np.ndarray) -> None:
    """
    Validates the algebraic integrity of the specified linear system prior 
    to VQLS execution.

    Validation Criteria:
      - A must be a square matrix.
      - A must exhibit strict Hermitian symmetry (essential for real Pauli decomposition).
      - Dimensionality alignment between A and the target vector b.
      - System dimension N must constitute a strict power of 2.
    """
    N = len(b)

    if A.shape != (N, N):
        raise ValueError(
            f"Matrix A possesses shape {A.shape}, conflicting with vector b length {N}."
        )

    if not np.allclose(A, A.conj().T, atol=1e-10):
        raise ValueError(
            "Matrix A must be strictly Hermitian for VQLS operator decomposition. "
            f"Maximum asymmetry observed: {np.max(np.abs(A - A.conj().T)):.2e}"
        )

    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError(
            f"System size N={N} must constitute a positive power of 2."
        )