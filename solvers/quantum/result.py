"""
Result container dataclasses for all quantum and classical solver outputs.

This module is the single authoritative source for all solver result types.
Placing all result containers here prevents circular import dependencies:
  - vqls_1d.py imports VQLSSolverResult from this module
  - This module has no dependency on vqls_1d.py or hhl_1d.py

Result hierarchy
────────────────
  SolverResult       : 1-D solver output (HHL, Thomas, NumPy, VQLS base)
  VQLSSolverResult   : SolverResult extended with VQLS-specific diagnostics
  SolverResult2D     : 2-D line-Jacobi solver output (HHL-2D, Thomas-2D)

References
──────────
Bravo-Prieto et al., "Variational Quantum Linear Solver",
    Quantum 7, 1188 (2023).
Ghafourpour & Laizet, Phys. Rev. Applied 24, 024032 (2025).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ── 1-D solver result ─────────────────────────────────────────────────────────

@dataclass
class SolverResult:
    """
    Output container for a single 1-D solver run.

    Attributes
    ----------
    u : np.ndarray, shape (N,)
        Recovered physical solution vector in non-dimensional units.
    solver : str
        Human-readable solver identifier, e.g. ``'Thomas'``, ``'HHL'``,
        ``'VQLS'``, or ``'NumPy'``.
    raw_state : np.ndarray or None, shape (N,)
        Raw b-register amplitudes extracted from the quantum statevector
        prior to proportionality scaling. Populated for HHL and VQLS;
        ``None`` for classical solvers.
    prop_const : float or None
        Proportionality constant c satisfying c·A·raw_state ≈ b.
        Populated for quantum solvers; ``None`` for classical solvers.
    euclidean_residual : float or None
        Relative Euclidean residual ‖Au − b‖ / ‖b‖, computed after
        proportionality recovery.
    """

    u:                    np.ndarray
    solver:               str
    raw_state:            Optional[np.ndarray] = field(default=None, repr=False)
    prop_const:           Optional[float]      = None
    euclidean_residual:   Optional[float]      = None


# ── VQLS extended result ──────────────────────────────────────────────────────

@dataclass
class VQLSSolverResult(SolverResult):
    """
    SolverResult extended with VQLS-specific optimisation diagnostics.

    Inherits all fields from SolverResult and appends the variational
    optimisation metadata required for convergence analysis and
    benchmarking against HHL.

    Attributes
    ----------
    final_cost : float
        Value of the normalised cost function C(θ) at termination.
        C = 0 indicates exact alignment of A|x(θ)⟩ with |b⟩;
        C = 1 indicates complete misalignment.
    n_circuit_evals : int
        Total number of cost function evaluations across all restarts.
        Proportional to the number of quantum circuit executions.
    optimiser_success : bool
        ``True`` if the optimiser reported convergence within the
        prescribed tolerance; ``False`` if max_iter was exhausted.
    cost_history : list of float
        Sequence of cost values recorded at each restart termination.
        Used to reproduce convergence plots for the thesis.
    optimal_params : np.ndarray or None, shape (n_params,)
        Optimised variational parameter vector θ* at termination.
        n_params = n_qubits × (n_layers + 1).
    n_layers : int
        Number of entangling layers in the hardware-efficient ansatz.
    n_parameters : int
        Total number of variational parameters: n_qubits × (n_layers + 1).
    """

    final_cost:        float                    = 0.0
    n_circuit_evals:   int                      = 0
    optimiser_success: bool                     = False
    cost_history:      List[float]              = field(
                           default_factory=list, repr=False
                       )
    optimal_params:    Optional[np.ndarray]     = field(default=None, repr=False)
    n_layers:          int                      = 0
    n_parameters:      int                      = 0


# ── QSVT result container ────────────────────────────────────────────────────

@dataclass
class QSVTSolverResult(SolverResult):
    """
    SolverResult extended with QSVT-specific circuit diagnostics.

    Inherits all fields from SolverResult and appends the circuit
    complexity metadata required for the thesis benchmarking analysis.

    Attributes
    ----------
    polynomial_degree : int
        Degree d of the QSP polynomial approximation to 1/x.
        Determines the number of block encoding oracle calls: O(d).
    n_angles : int
        Number of QSP phase angles: d + 1.
    circuit_depth : int
        Total gate depth of the QSVT circuit as reported by Qiskit.
        Includes state preparation, QSVT sequence, and ancilla management.
    n_qubits : int
        Total qubit count: n (data) + n_a (block encoding ancilla)
        + 1 (QSVT signal qubit).
    alpha : float
        Block encoding subnormalisation factor: alpha = |a| + 2|b|
        for a TST matrix with main diagonal a and off-diagonal b.
    kappa_effective : float
        Effective condition number after subnormalisation:
        kappa_eff = alpha * kappa(A) / ||A||_2.
        This is the condition number seen by the QSVT polynomial and
        determines the polynomial degree requirement.
    angles : np.ndarray or None, shape (d+1,)
        QSP phase angles phi_0, ..., phi_d. Stored for reproducibility
        and for circuit reconstruction without recomputation.
    """

    polynomial_degree : int                     = 0
    n_angles          : int                     = 0
    circuit_depth     : int                     = 0
    n_qubits          : int                     = 0
    alpha             : float                   = 0.0
    kappa_effective   : float                   = 0.0
    angles            : Optional[np.ndarray]    = field(
                            default=None, repr=False
                        )


# ── 2-D solver result ─────────────────────────────────────────────────────────

@dataclass
class SolverResult2D:
    """
    Output container for a single 2-D line-Jacobi solver run.

    The line-Jacobi scheme decomposes the 2-D Poisson problem into a
    sequence of 1-D TST sub-problems (one per interior row per iteration),
    solved iteratively until the update norm falls below the prescribed
    tolerance. This container records both the converged solution field
    and the full iteration history for convergence analysis.

    Attributes
    ----------
    u : np.ndarray, shape (N, N)
        Recovered (N, N) solution field in non-dimensional units,
        indexed as u[i, j] where i is the x-index and j the y-index.
    solver : str
        Human-readable solver identifier, e.g. ``'Thomas-2D'`` or
        ``'HHL-2D'``.
    iterations : int
        Number of complete line-Jacobi sweeps performed before
        convergence or exhaustion of max_iter.
    converged : bool
        ``True`` if max|u^{n+1} − u^n| < tol was satisfied;
        ``False`` if max_iter was reached without convergence.
    iteration_errors : list of float
        Sequence of max|u^{n+1} − u^n| values at each iteration.
        Used to reproduce the convergence history plots of
        Ghafourpour & Laizet (2025), Fig. 15.
    euclidean_residual : float or None
        Relative residual ‖A_full·u_flat − b_full‖ / ‖b_full‖
        computed via the tridiagonal matvec (no full matrix allocation).
        Measures proximity to the exact solution of the full coupled
        system; O(1) values are expected for Jacobi iterates that have
        not fully converged.
    """

    u:                  np.ndarray
    solver:             str
    iterations:         int
    converged:          bool
    iteration_errors:   list
    euclidean_residual: Optional[float] = None