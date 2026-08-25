"""
Variational Quantum Linear Solver (VQLS) for 1D Poisson systems with a
Toeplitz Symmetric Tridiagonal (TST) matrix structure.

Mathematical formulation
------------------------
VQLS minimises the normalised cost function:

    C(θ) = 1 − |⟨b|A|x(θ)⟩|² / ⟨x(θ)|A†A|x(θ)⟩

over the variational parameters θ of a hardware-efficient ansatz
|x(θ)⟩ = V(θ)|0⟩, where V(θ) is a layered circuit of RY rotations
and nearest-neighbour CNOT gates. The matrix A is decomposed into a
Linear Combination of Unitaries (LCU):

    A = Σ_l c_l U_l

via Pauli string decomposition, enabling efficient evaluation of the
cost function on a statevector simulator.

Public Interface
----------------
vqls_solve(problem) :
    High-level wrapper accepting a `PoissonProblem1D`, returning a standardised
    `VQLSSolverResult`.
vqls_solve_system(A, b, config) :
    Low-level routine accepting raw NumPy arrays. Kept separate so that the
    outer iteration in `solvers/outer` can solve each strip without
    constructing a problem container per sub-problem.

Methodology
-----------
Following Bravo-Prieto et al. (2023):

  1. Decompose the system matrix A into a sum of Pauli unitaries (LCU).
  2. Prepare a hardware-efficient parameterised ansatz |x(θ)⟩.
  3. Minimise the globally normalised cost function
     C(θ) = 1 - |⟨b|A|x⟩|² / ⟨x|A†A|x⟩.
  4. Recover the physical solution scale by proportionality constant
     projection.

The ansatz circuit and its statevector are provided by PennyLane
(`vqls_utils.py`); the cost function itself is then evaluated by direct
statevector arithmetic over the Pauli LCU terms rather than by executing a
Hadamard test, which is exact and considerably cheaper at these system sizes.

References
----------
Bravo-Prieto et al., "Variational Quantum Linear Solver",
    Quantum 7, 1188 (2023).
Ghafourpour & Laizet, Phys. Rev. Applied 24, 024032 (2025).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import minimize

from problems.poisson_1d import PoissonProblem1D
from solvers.quantum.result import SolverResult, VQLSSolverResult
from solvers.quantum.vqls_utils import (
    pauli_decompose_normalised,
    build_cost_function,
    recover_solution,
    n_params,
)


# -- VQLS Configuration --------------------------------------------------------

@dataclass
class VQLSConfig1D:
    """
    Hyperparameters governing the VQLS variational optimisation.

    Attributes
    ----------
    n_layers : int
        Number of entangling layers in the ansatz. Greater depth increases
        expressivity at the cost of a harder optimisation landscape. Empirical
        baselines are 3–5 layers for N=8 (3 qubits) and 5–8 for N=16 (4 qubits).
    optimiser : str
        SciPy optimisation algorithm. 'COBYLA' is gradient-free and robust at
        these system dimensions; 'L-BFGS-B' converges faster over larger
        parameter spaces but requires gradient evaluations.
    max_iter : int
        Ceiling on the optimiser iteration count, applied per restart.
    tol : float
        Convergence tolerance on the cost function.
    init_params : np.ndarray or None
        Initial variational parameter vector of length n_qubits·(n_layers+1).
        If None, parameters are drawn uniformly from [0, 2π] under
        `random_seed`.
    random_seed : int
        Seed giving reproducible pseudo-random initialisation.
    device_name : str
        PennyLane simulation device for the ansatz statevector, e.g.
        'default.qubit' for exact simulation.
    verbose : bool
        If True, print the iterative cost trajectory.
    n_restarts : int
        Number of independent optimisation runs; the lowest final cost is
        retained. Each run performs one exploration pass at rhobeg = 0.5
        followed by two polishing passes at rhobeg = 0.1 and 0.01, this
        three-stage schedule being what makes COBYLA reliable here.
    """
    n_layers:    int   = 4
    optimiser:   str   = "COBYLA"
    max_iter:    int   = 1000
    tol:         float = 1e-6
    init_params: Optional[np.ndarray] = field(default=None, repr=False)
    random_seed: int   = 42
    device_name: str   = "default.qubit"
    verbose:     bool  = False
    n_restarts  : int   = 3      # number of random restarts; best cost is kept


# More layers and iterations needed for physically scaled problems.
DEFAULT_VQLS_CONFIG = VQLSConfig1D(
    n_layers  = 6,
    optimiser = "COBYLA",
    max_iter  = 300,    # per restart — 3 restarts = 900 total
    tol       = 1e-6,
    random_seed = 42,
)


# -- Public High-Level Interface -----------------------------------------------

def vqls_solve(
    problem: PoissonProblem1D,
    config:  VQLSConfig1D = DEFAULT_VQLS_CONFIG,
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
    config : VQLSConfig1D, default=DEFAULT_VQLS_CONFIG
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


# -- Core Algorithmic Sub-Routine ----------------------------------------------

def _vqls_single_run(
    A           : np.ndarray,
    b_norm      : np.ndarray,
    b_norm_factor: float,
    A_norm_factor: float,
    pauli_terms : list,
    n_qubits    : int,
    config      : "VQLSConfig1D",
    seed        : int,
) -> tuple[np.ndarray, float, list[float], int]:
    """
    Execute a single independent VQLS optimisation run from a fresh random
    parameter initialisation, followed by sequential refinement passes.

    This function constitutes the inner workhorse of the multi-restart
    strategy implemented in vqls_solve_system. Each call explores an
    independent basin of the cost landscape by starting from a randomly
    drawn parameter vector, then applies a cascade of COBYLA refinement
    passes with progressively reduced step sizes (rhobeg) to polish the
    local minimum found during exploration.

    The separation of exploration (this function, called multiple times
    with different seeds) from selection (performed in vqls_solve_system
    by comparing final costs) is the key architectural difference from
    the previous single-run-with-refinement approach. Exploration with
    diverse seeds is necessary because the COBYLA landscape for N=8
    (3 qubits, n_params ~ 21) contains multiple local minima of similar
    depth, some of which correspond to asymmetric solutions that achieve
    low cost but do not respect the spatial symmetry of the problem.
    Selecting the run with the globally lowest cost across all seeds
    reliably identifies the symmetric global minimum.

    Parameters
    ----------
    A : np.ndarray
        N×N TST system matrix (Hermitian).
    b_norm : np.ndarray
        Normalised RHS vector b / ||b||, length N.
    b_norm_factor : float
        ||b||, used for residual computation.
    A_norm_factor : float
        ||A||_2, used for Pauli decomposition normalisation.
    pauli_terms : list
        Pre-computed Pauli decomposition of A (shared across all restarts
        to avoid redundant decomposition).
    n_qubits : int
        Number of qubits log2(N).
    config : VQLSConfig1D
        Hyperparameter structure. n_restarts is ignored here (single run).
    seed : int
        Random seed for parameter initialisation. Different seeds produce
        different starting points and therefore explore different basins.

    Returns
    -------
    optimal_params : np.ndarray
        Best parameter vector found in this run.
    final_cost : float
        Cost function value at optimal_params.
    cost_history : list[float]
        Cost at the end of each refinement pass (length = n_refinement_passes).
    total_evals : int
        Total number of cost function evaluations in this run.
    """
    n_p = n_params(n_qubits, config.n_layers)

    # -- Fresh Random Initialisation -------------------------------------------
    # Each independent run draws a new theta_init from a different seed.
    # This fundamentally distinguishes it from sequential refinement: it explores
    # a new basin in the landscape rather than refining within the current one.
    rng        = np.random.default_rng(seed)
    theta_init = rng.uniform(0, 2 * np.pi, size=n_p)

    # -- Objective Function ----------------------------------------------------
    cost_fn = build_cost_function(
        pauli_terms  = pauli_terms,
        b_norm       = b_norm,
        n_qubits     = n_qubits,
        n_layers     = config.n_layers,
        device_name  = config.device_name,
    )

    # -- Exploration / Optimisation Pass ---------------------------------------
    if config.optimiser == "COBYLA":
        # COBYLA Cascade
        opt_result = minimize(
            fun     = cost_fn,
            x0      = theta_init,
            method  = "COBYLA",
            tol     = config.tol,
            options = {
                "maxiter": config.max_iter,
                "rhobeg":  0.5,
            },
        )
        best_params  = opt_result.x.copy()
        best_cost    = float(opt_result.fun)
        cost_history = [best_cost]
        total_evals  = int(opt_result.nfev)

        if best_cost <= config.tol:
            return best_params, best_cost, cost_history, total_evals

        for rhobeg in (0.1, 0.01):
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
            total_evals  += int(opt_result.nfev)
            current_cost  = float(opt_result.fun)
            cost_history.append(current_cost)

            if current_cost < best_cost:
                best_cost   = current_cost
                best_params = opt_result.x.copy()

            if best_cost <= config.tol:
                break
    else:
        # e.g., L-BFGS-B or others
        opt_result = minimize(
            fun     = cost_fn,
            x0      = theta_init,
            method  = config.optimiser,
            tol     = config.tol,
            options = {
                "maxiter": config.max_iter,
            },
        )
        best_params  = opt_result.x.copy()
        best_cost    = float(opt_result.fun)
        cost_history = [best_cost]
        total_evals  = int(opt_result.nfev)

    return best_params, best_cost, cost_history, total_evals


def vqls_solve_system(
    A:      np.ndarray,
    b:      np.ndarray,
    config: "VQLSConfig1D" = DEFAULT_VQLS_CONFIG,
) -> "VQLSSolverResult":
    """
    Resolve the linear system Au = b employing the VQLS algorithm directly
    on raw NumPy arrays.

    Implements a two-phase multi-restart strategy to mitigate the local-minimum
    problem inherent to variational quantum optimisation:

    Phase 1 — Independent Exploration:
        n_restarts independent optimisation runs are launched, each from a
        fresh random parameter initialisation drawn from a different seed.
        Each run performs one exploration pass (large rhobeg=0.5) followed
        by two refinement passes (rhobeg=0.1, 0.01). The runs are independent
        — they do not share parameter vectors between them. This diversity is
        essential for N=8 (3 qubits) where the COBYLA landscape contains
        multiple local minima of similar depth. Without independent restarts,
        a single run with seed=42 consistently finds an asymmetric local
        minimum that produces a monotonically varying relative error profile
        despite the problem being spatially symmetric.

    Phase 2 — Global Refinement of Best Result:
        The run achieving the lowest final cost is selected. One additional
        refinement pass with rhobeg=0.001 is applied to the best parameter
        vector to achieve the tightest possible convergence.

    The selection criterion (lowest cost) reliably identifies the symmetric
    global minimum because: (a) the symmetric solution satisfies the cost
    function exactly (cost → 0), while asymmetric local minima have cost
    ~ 10^{-5} to 10^{-3}; (b) the gap between the global minimum and the
    best asymmetric local minimum is large enough (~5 orders of magnitude
    at N=8) to make the selection unambiguous.

    Parameters
    ----------
    A : np.ndarray
        N×N TST system matrix (must be Hermitian).
    b : np.ndarray
        Right-hand side target vector of length N.
    config : VQLSConfig1D, default=DEFAULT_VQLS_CONFIG
        Hyperparameter structure governing the variational optimisation.
        The n_restarts field controls the number of independent exploration
        runs in Phase 1. Default is 3; 5 is recommended for N=8.

    Returns
    -------
    VQLSSolverResult
        Populated data structure encapsulating the solution and full telemetry.
        The cost_history field contains the concatenated cost histories from
        all restarts, with the best-run history listed first.
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

    # -- Pauli Decomposition ---------------------------------------------------
    # Computed once and shared across all restarts to avoid redundant work.
    # The decomposition depends only on A, not on the parameter vector.
    pauli_terms, A_norm_factor = pauli_decompose_normalised(A)

    n_p = n_params(n_qubits, config.n_layers)

    if config.verbose:
        print(
            f"  VQLS: N={N}, n_qubits={n_qubits}, "
            f"n_layers={config.n_layers}, n_params={n_p}, "
            f"LCU terms={len(pauli_terms)}, "
            f"||A||_2={A_norm_factor:.4f}, "
            f"||b||={b_norm_factor:.4e}, "
            f"n_restarts={config.n_restarts}"
        )

    # -- Phase 1: Independent Exploration Restarts -----------------------------
    # Generate n_restarts independent seeds from the master seed. Using a
    # seeded RNG to generate the child seeds ensures full reproducibility:
    # the same master seed always produces the same set of child seeds and
    # therefore the same final result, regardless of n_restarts.
    master_rng   = np.random.default_rng(config.random_seed)
    child_seeds  = [
        int(master_rng.integers(0, 2**31))
        for _ in range(config.n_restarts)
    ]

    best_params_global  = None
    best_cost_global    = float("inf")
    all_cost_histories  : list[list[float]] = []
    total_evals_global  = 0

    for restart_idx, seed in enumerate(child_seeds):
        if config.verbose:
            print(
                f"  Restart {restart_idx + 1}/{config.n_restarts}: "
                f"seed={seed}"
            )

        params, cost, history, evals = _vqls_single_run(
            A             = A,
            b_norm        = b_norm,
            b_norm_factor = b_norm_factor,
            A_norm_factor = A_norm_factor,
            pauli_terms   = pauli_terms,
            n_qubits      = n_qubits,
            config        = config,
            seed          = seed,
        )

        all_cost_histories.append(history)
        total_evals_global += evals

        if config.verbose:
            print(
                f"    → final cost={cost:.6e}, evals={evals}"
            )

        if cost < best_cost_global:
            best_cost_global   = cost
            best_params_global = params.copy()

        # Early exit across restarts if the global tolerance is already met.
        # This avoids running remaining restarts when the global minimum has
        # already been found (common at N=4 where the landscape is simple).
        if best_cost_global <= config.tol:
            if config.verbose:
                print(
                    f"  Tolerance {config.tol:.0e} met at restart "
                    f"{restart_idx + 1}. Skipping remaining restarts."
                )
            # Pad remaining history slots with the achieved cost for
            # consistent telemetry length.
            for _ in range(config.n_restarts - restart_idx - 1):
                all_cost_histories.append([best_cost_global])
            break

    # -- Phase 2: Global Refinement of Best Result -----------------------------
    # Apply one final tight refinement pass to the best parameter vector
    # found across all restarts. This polishes the solution within the
    # identified global minimum basin without risking escape to a worse basin
    # (rhobeg=0.001 is too small to cross basin boundaries).
    refined = False
    if best_cost_global > config.tol:
        cost_fn_final = build_cost_function(
            pauli_terms  = pauli_terms,
            b_norm       = b_norm,
            n_qubits     = n_qubits,
            n_layers     = config.n_layers,
            device_name  = config.device_name,
        )

        opt_final = minimize(
            fun     = cost_fn_final,
            x0      = best_params_global,
            method  = "COBYLA",
            tol     = config.tol,
            options = {
                "maxiter": config.max_iter,
                "rhobeg":  0.001,
            },
        )

        total_evals_global += int(opt_final.nfev)
        final_cost_refined  = float(opt_final.fun)
        all_cost_histories.append([final_cost_refined])
        refined = True

        if final_cost_refined < best_cost_global:
            best_cost_global   = final_cost_refined
            best_params_global = opt_final.x.copy()

        if config.verbose:
            print(
                f"  Global refinement: cost={best_cost_global:.6e}, "
                f"evals={opt_final.nfev}"
            )

    optimal_params    = best_params_global
    final_cost        = best_cost_global
    optimiser_success = final_cost <= config.tol * 10

    # Partition the telemetry into exploration restarts and the optional Phase 2
    # entry. The refinement pass is executed only when the tolerance was not
    # already met, so the trailing entry exists conditionally and the split must
    # be made on `refined` rather than by an unconditional [:-1] slice. With
    # n_restarts = 1 the early-exit branch pads nothing and Phase 2 is skipped,
    # leaving a single history; the unconditional slice then yielded an empty
    # sequence and np.argmin raised ValueError. Under the outer iteration that
    # exception is absorbed by the classical fallback in
    # `solvers.outer.inner.InnerSolverWrapper`, so every strip returned the
    # Thomas solution and the failure presented as a flawless VQLS result.
    restart_histories  = all_cost_histories[:-1] if refined else all_cost_histories
    refinement_history = [all_cost_histories[-1]] if refined else []

    if config.verbose:
        print(
            f"  VQLS final: cost={final_cost:.6e}, "
            f"total_evals={total_evals_global}, "
            f"converged={optimiser_success}, "
            f"best_seed_idx={np.argmin([h[-1] for h in restart_histories])}"
        )

    # -- Solution Recovery -----------------------------------------------------
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

    # Flatten cost history: best-run history first, then remaining runs,
    # then global refinement. This preserves the full optimisation trajectory
    # for diagnostic purposes whilst keeping the primary history accessible.
    best_run_idx     = int(np.argmin([h[-1] for h in restart_histories]))
    ordered_histories = (
        [restart_histories[best_run_idx]]
        + [h for i, h in enumerate(restart_histories) if i != best_run_idx]
        + refinement_history
    )
    flat_cost_history = [c for h in ordered_histories for c in h]

    return VQLSSolverResult(
        u                 = u,
        solver            = "VQLS",
        raw_state         = None,
        prop_const        = c,
        euclidean_residual= residual,
        final_cost        = final_cost,
        n_circuit_evals   = total_evals_global,
        optimiser_success = optimiser_success,
        cost_history      = flat_cost_history,
        optimal_params    = optimal_params,
        n_layers          = config.n_layers,
        n_parameters      = n_p,
    )


# -- Private Utility Methods ---------------------------------------------------

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