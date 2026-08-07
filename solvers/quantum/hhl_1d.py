"""
Harrow-Hassidim-Lloyd (HHL) quantum solver for 1D Poisson systems with a
Toeplitz Symmetric Tridiagonal (TST) matrix structure.

Public Interface
----------------
hhl_solve(problem) :
    High-level wrapper accepting a `PoissonProblem1D`, returning a standardised
    `SolverResult`.
hhl_solve_system(A, b, eps) :
    Low-level routine accepting raw NumPy arrays. Kept separate so that the
    outer iteration in `solvers/outer` can solve each strip without
    constructing a problem container per sub-problem.

Hamiltonian simulation is provided by the vendored `quantum_linear_solvers`
implementation of the TST evolution operator (Vázquez et al.), and all circuits
are evaluated by deterministic statevector simulation — there is no shot noise.
"""
from __future__ import annotations

import warnings

import numpy as np

from quantum_linear_solvers.linear_solvers.hhl import HHL
from quantum_linear_solvers.linear_solvers.matrices.tridiagonal_toeplitz import (
    TridiagonalToeplitz,
)
from qiskit.quantum_info import Statevector

from core.execution import default_executor, hhl_spec

from problems.poisson_1d import PoissonProblem1D
from solvers.quantum.result import SolverResult


# ── Public High-Level Interface ───────────────────────────────────────────────

def hhl_solve(problem: PoissonProblem1D) -> SolverResult:
    """
    Solves the 1D Poisson system Au = b by the HHL algorithm.

    A wrapper around `hhl_solve_system` that unpacks the `PoissonProblem1D`
    container and packages the outputs into a `SolverResult`.

    Parameters
    ----------
    problem : PoissonProblem1D
        Discretised 1D problem supplying the N×N operator, the length-N
        right-hand side and the precision parameter ε.

    Returns
    -------
    result : SolverResult
        Solution vector, raw b-register amplitudes, proportionality constant
        and relative Euclidean residual.
    """
    u, x_raw, c = hhl_solve_system(
        problem.A,
        problem.b,
        problem.config.epsilon,
    )
    return SolverResult(
        u=u,
        solver="HHL",
        raw_state=x_raw,
        prop_const=c,
        euclidean_residual=_relative_residual(problem.A, u, problem.b),
    )


# ── Core Algorithmic Sub-Routine ──────────────────────────────────────────────

def hhl_solve_system(
    A:       np.ndarray,
    b:       np.ndarray,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solves the linear system Au = b by the HHL algorithm, operating directly on
    raw NumPy arrays.

    This is the primary quantum execution routine. It is invoked once per strip
    per sweep by the outer iteration in `solvers/outer`, and holds no dependency
    on any problem container class.

    Parameters
    ----------
    A : np.ndarray
        N×N TST system matrix, assumed Hermitian. Spectrally normalised
        internally so that its eigenvalues lie within (-1, 1].
    b : np.ndarray
        Length-N right-hand side vector.
    epsilon : float
        Precision parameter governing the Trotter approximation. Sets the
        internal `trotter_steps` allocation as ceil(1/ε).

    Returns
    -------
    u : np.ndarray
        Length-N recovered physical solution vector.
    x_raw : np.ndarray
        Length-N raw quantum state amplitudes extracted from the b-register.
    c : float
        Proportionality constant satisfying c·A·x_raw ≈ b.

    Raises
    ------
    ValueError
        If the right-hand side b is numerically zero, precluding state
        normalisation. Callers in the outer layer must detect this and assign a
        zero-vector solution directly.
    RuntimeError
        If statevector extraction yields a null vector under the strict
        post-selection criteria, or if proportionality recovery degenerates.
    """
    N = len(b)

    # ── Phase 1: Spectral Normalisation ───────────────────────────────────────
    A_norm_factor = float(np.linalg.norm(A, ord=2))
    b_norm_factor = float(np.linalg.norm(b))

    if b_norm_factor < 1e-14:
        raise ValueError(
            "RHS vector b is numerically zero; state normalisation for amplitude "
            "encoding cannot proceed. The calling namespace must detect this "
            "condition and assign a zero-vector solution directly."
        )

    b_norm = b / b_norm_factor
    a_norm = A[0, 0] / A_norm_factor       # Principal diagonal of normalised A
    b_off  = A[0, 1] / A_norm_factor       # Off-diagonal of normalised A

    # ── Phase 2: Operator Construction ────────────────────────────────────────
    num_qubits    = int(np.log2(N))
    trotter_steps = max(1, int(np.ceil(1.0 / epsilon)))

    matrix = TridiagonalToeplitz(
        num_state_qubits=num_qubits,
        main_diag=a_norm,
        off_diag=b_off,
        trotter_steps=trotter_steps,
    )

    # ── Phase 3: Algorithm Execution ──────────────────────────────────────────
    hhl = HHL()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solution = hhl.solve(matrix, b_norm)

    # ── Phase 4: Statevector Extraction ───────────────────────────────────────
    x_raw = _extract_solution_statevector(solution.state, num_qubits)

    # ── Phase 5: Dimensionality Recovery ──────────────────────────────────────
    # The scaling constant is recovered against the *normalised* system, so that
    # quantum error is not amplified by the geometric factor ||b||_2 / ||A||_2.
    # This matters for physically scaled domains — notably the HET case, where
    # ||b|| >> 1 owing to the alpha = L^2 / lambda_D^2 prefactor.
    #
    # The normalised geometric relation is
    #   A_norm . x_raw ~ c_norm . b_norm
    # with A_norm = A / ||A||_2 and b_norm = b / ||b||_2, from which the
    # physical solution is recovered as
    #   u = c_norm . x_raw . (||b||_2 / ||A||_2)

    A_norm = A / A_norm_factor
    b_norm_vec = b / b_norm_factor

    Ax_norm = A_norm @ x_raw
    denom   = float(np.dot(Ax_norm, Ax_norm))

    if denom < 1e-14:
        raise RuntimeError(
            "HHL proportionality recovery failed: ||A_norm·x_raw||² "
            "is numerically zero."
        )

    c_norm = float(np.dot(b_norm_vec, Ax_norm) / denom)
    scale  = b_norm_factor / A_norm_factor
    u      = c_norm * scale * x_raw
    c      = c_norm * scale

    return u, x_raw, c


# ── Private Utility Methods ───────────────────────────────────────────────────

def _relative_residual(
    A: np.ndarray,
    u: np.ndarray,
    b: np.ndarray,
) -> float:
    """Computes the relative Euclidean residual ‖Au - b‖₂ / ‖b‖₂."""
    return float(np.linalg.norm(A @ u - b) / np.linalg.norm(b))


def _extract_solution_statevector(
    circuit,
    num_qubits: int,
) -> np.ndarray:
    """
    Extracts the solution vector from the HHL output quantum circuit.

    Register layout (from circuit.qregs, Qiskit little-endian ordering):

        qregs[0] : b-register (solution), n_b qubits, indices [0, n_b - 1]
        qregs[1] : l-register (clock),    n_l qubits, indices [n_b, n_b+n_l-1]
        qregs[2] : MCMT ancilla,          n_a qubits
        qregs[3] : flag qubit (ancilla),  index [n_total - 1]

    Strict post-selection criteria:

        - flag qubit evaluates to |1⟩
        - clock (l-register) is cleared to |0…0⟩
        - MCMT ancillae are returned to |0…0⟩

    Parameters
    ----------
    circuit : QuantumCircuit
        HHL output circuit carrying the solution register.
    num_qubits : int
        Number of b-register qubits; N = 2^num_qubits.

    Returns
    -------
    x_raw : np.ndarray
        Length-N real part of the post-selected b-register amplitudes.

    Raises
    ------
    RuntimeError
        If post-selection yields a null vector. A diagnostic table of the ten
        dominant statevector amplitudes is emitted first, to identify which
        register failed to clear.
    """
    # Routed through core.execution so that the same solver body runs under
    # exact statevector evolution (the default and the thesis baseline), under
    # a shot-based noisy simulator, or on hardware. The default executor
    # reproduces the previous inline masking and diagnostics bit-for-bit; the
    # register layout documented above is now expressed by ``hhl_spec``.
    x_raw_real, _record = default_executor().extract(
        circuit,
        hhl_spec(circuit, num_qubits),
    )
    return x_raw_real
