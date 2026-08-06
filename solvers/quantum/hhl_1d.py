"""
Implements the Harrow-Hassidim-Lloyd (HHL) quantum solver for 1D Poisson systems 
characterised by a Toeplitz Symmetric Tridiagonal (TST) matrix structure.

Public Interface
----------------
hhl_solve(problem) : 
    High-level wrapper accepting a `PoissonProblem1D` object, returning a 
    standardised `SolverResult`.
hhl_solve_system(A, b, eps) : 
    Low-level core routine accepting raw numerical arrays. Architected to 
    allow the 2D line-Jacobi solver to bypass problem container instantiation 
    for every row sub-problem, significantly reducing computational overhead.
"""
from __future__ import annotations

import warnings

import numpy as np

from quantum_linear_solvers.linear_solvers.hhl import HHL
from quantum_linear_solvers.linear_solvers.matrices.tridiagonal_toeplitz import (
    TridiagonalToeplitz,
)
from qiskit.quantum_info import Statevector

from problems.poisson_1d import PoissonProblem1D
from solvers.quantum.result import SolverResult


# ── Public High-Level Interface ───────────────────────────────────────────────

def hhl_solve(problem: PoissonProblem1D) -> SolverResult:
    """
    Resolves the 1D Poisson system Au = b utilising the HHL algorithm.

    Operates as a procedural wrapper around the core `hhl_solve_system` 
    sub-routine, unpacking the `PoissonProblem1D` data structure and 
    packaging the numerical outputs into a unified `SolverResult`.
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
    Resolves the linear system Au = b employing the HHL algorithm directly 
    on raw NumPy arrays.

    This decoupled function serves as the primary quantum execution engine. 
    It is invoked iteratively by the 2D line-Jacobi loop for each row sub-problem, 
    maintaining strict independence from any problem container classes.

    Parameters
    ----------
    A : np.ndarray
        N×N TST system matrix. Assumed to be Hermitian. Will be spectrally 
        normalised internally such that eigenvalues reside within (-1, 1].
    b : np.ndarray
        Right-hand side vector of length N.
    epsilon : float
        Precision parameter governing the Trotter approximation. Determines 
        the internal `trotter_steps` allocation via ceil(1/epsilon).

    Returns
    -------
    u : np.ndarray
        Recovered physical solution vector.
    x_raw : np.ndarray
        Raw quantum state amplitudes extracted directly from the b-register.
    c : float
        Proportionality constant satisfying the relation c * A * x_raw ≈ b.

    Raises
    ------
    ValueError
        Triggered if the RHS vector `b` evaluates to a numerical zero, precluding 
        state normalisation. (Downstream 2D algorithms must handle this edge case).
    RuntimeError
        Triggered if statevector extraction yields a null vector under strict 
        post-selection criteria.
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
    # Recover the scaling constant against the normalised system to mitigate 
    # the amplification of quantum noise by the geometric factor ||b||_2 / ||A||_2. 
    # This regularisation is critical for physically scaled domains (e.g., 
    # Heterogeneous configurations) where ||b|| ≫ 1 due to the α = L²/λ_D² prefactor.
    #
    # The normalised geometric relation is formulated as: 
    #   A_norm · x_raw ≈ c_norm · b_norm
    # where A_norm = A / ||A||_2, and b_norm = b / ||b||_2.
    #
    # The physical solution dimensionality is subsequently recovered via: 
    #   u = c_norm · x_raw · (||b||_2 / ||A||_2)

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
    """Computes the relative Euclidean residual ||Au - b||_2 / ||b||_2."""
    return float(np.linalg.norm(A @ u - b) / np.linalg.norm(b))


def _extract_solution_statevector(
    circuit,
    num_qubits: int,
) -> np.ndarray:
    """
    Extracts the solution vector from the HHL output quantum circuit.

    Register Layout (Derived from circuit.qregs, Qiskit little-endian ordering):
        qregs[0] : b-register (solution), n_b qubits, indices [0, n_b - 1]
        qregs[1] : l-register (clock), n_l qubits, indices [n_b, n_b + n_l - 1]
        qregs[2] : MCMT ancilla, n_a qubits
        qregs[3] : Flag qubit (ancilla), index [n_total - 1]

    Strict Post-Selection Criteria:
        - Flag qubit evaluates to |1⟩.
        - Clock (l-register) is cleared to |0...0⟩.
        - MCMT ancillary hardware is returned to |0...0⟩.
    """
    N       = 2 ** num_qubits
    n_total = circuit.num_qubits
    n_b     = circuit.qregs[0].size
    n_l     = circuit.qregs[1].size
    n_ancilla = n_total - 1 - n_b - n_l

    flag_bit_pos          = n_total - 1
    clock_start           = n_b
    ancilla_start         = n_b + n_l
    non_b_non_flag_mask   = (
        ((1 << n_l)       - 1) << clock_start
        | ((1 << n_ancilla) - 1) << ancilla_start
    )

    sv    = Statevector(circuit).data
    x_raw = np.zeros(N, dtype=complex)

    for idx in range(2 ** n_total):
        flag_bit    = (idx >> flag_bit_pos) & 1
        middle_bits = idx & non_b_non_flag_mask
        b_reg_idx   = idx & (N - 1)
        
        if flag_bit == 1 and middle_bits == 0:
            x_raw[b_reg_idx] = sv[idx]

    x_raw_real = np.real(x_raw)

    if np.allclose(x_raw_real, 0.0, atol=1e-12):
        reg_info = [(r.name, r.size) for r in circuit.qregs]
        print("\nDEBUG — Dominant statevector amplitudes by magnitude:")
        magnitudes  = np.abs(sv)
        top_indices = np.argsort(magnitudes)[::-1][:10]
        for idx in top_indices:
            print(
                f"  idx={idx:5d}  "
                f"|amp|={magnitudes[idx]:.6f}  "
                f"flag={(idx >> flag_bit_pos) & 1}  "
                f"clock={(idx & (((1 << n_l) - 1) << n_b)) >> n_b:0{n_l}b}  "
                f"b_reg={idx & (N - 1)}"
            )
        raise RuntimeError(
            f"HHL extraction returned a null vector under strict post-selection.\n"
            f"Registers: {reg_info}\n"
            f"Parameters: n_total={n_total}, n_b={n_b}, n_l={n_l}, "
            f"n_ancilla={n_ancilla}."
        )

    return x_raw_real