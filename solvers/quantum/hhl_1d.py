"""
Implements the quantum solver for the 1D Poisson equation utilising the 
Harrow-Hassidim-Lloyd (HHL) algorithm.

This module leverages the external `quantum_linear_solvers` library to 
instantiate the quantum circuit. It manages the complete execution pipeline: 
pre-processing (spectral normalisation), circuit parameterisation, statevector 
extraction via post-selection, and the final recovery of physical dimensions 
through proportionality scaling.
"""
from __future__ import annotations

import warnings
import numpy as np

from qiskit.quantum_info import Statevector

# External algorithm dependencies (must strictly originate from quantum_linear_solvers)
from quantum_linear_solvers.linear_solvers.hhl import HHL
from quantum_linear_solvers.linear_solvers.matrices.tridiagonal_toeplitz import (
    TridiagonalToeplitz,
)

from problems.poisson_1d import PoissonProblem1D
from solvers.quantum.result import SolverResult


# ── Quantum Solver Implementation ─────────────────────────────────────────────

def hhl_solve(problem: PoissonProblem1D) -> SolverResult:
    """
    Resolves the linear system Au = b employing the HHL algorithm, subsequently 
    recovering the physical solution via least-squares proportionality scaling.

    Algorithm Pipeline
    ------------------
    Initialisation : 
        The system matrix A is scaled by its spectral norm to constrain its 
        eigenvalues to the interval (-1, 1]. The right-hand side vector b is 
        normalised to a unit vector to facilitate amplitude encoding.
    Matrix Instantiation : 
        The `TridiagonalToeplitz` operator is constructed, utilising the 
        specialised Hamiltonian simulation detailed by Vázquez et al.
    Circuit Execution : 
        The HHL routine is executed. The library's `solve()` method processes 
        the raw numerical vector, internally generating the requisite state 
        preparation isometry.
    State Extraction : 
        The resulting `QuantumCircuit` is simulated via Qiskit's `Statevector`. 
        The solution is extracted by strictly post-selecting on the success 
        flag qubit and the cleared ancillary/clock registers.
    Dimensional Recovery : 
        The extracted quantum state satisfies c * A * x_raw ≈ b. The scalar c 
        is recovered via least-squares projection, mapping the normalised 
        state back into the physical domain.

    Implementation Details
    ----------------------
    - The `HHL()` constructor in the current library version accepts no arguments, 
      inheriting directly from the base `LinearSolver`.
    - Trotterisation precision is governed by the `trotter_steps` attribute 
      on the matrix object. The continuous precision parameter (epsilon) is 
      mapped via ceil(1/epsilon) to ensure consistency with discrete step sweeps.
    - Solution extraction necessitates local statevector masking rather than 
      measurement sampling to eliminate statistical shot noise during baseline 
      algorithmic verification.
    """
    cfg = problem.config
    A   = problem.A
    b   = problem.b
    N   = cfg.N

    # ── Phase 1: Spectral Normalisation ───────────────────────────────────────
    # Scale A by its spectral norm to guarantee eigenvalues reside within (-1, 1].
    # The right-hand side vector is normalised internally by the HHL instance.
    A_norm_factor = float(np.linalg.norm(A, ord=2))
    b_norm_factor = float(np.linalg.norm(b))

    b_norm = b / b_norm_factor   # Unit vector for amplitude encoding
    a_norm = -2.0 / A_norm_factor
    b_off  =  1.0 / A_norm_factor

    # ── Phase 2: Operator Construction ────────────────────────────────────────
    # The precision mapping ceil(1/epsilon) aligns a continuous epsilon sweep 
    # (e.g., 0.01 -> 100 steps) with the discrete Trotter steps parameter.
    num_qubits    = int(np.log2(N))
    trotter_steps = max(1, int(np.ceil(1.0 / cfg.epsilon)))

    matrix = TridiagonalToeplitz(
        num_state_qubits=num_qubits,
        main_diag=a_norm,
        off_diag=b_off,
        trotter_steps=trotter_steps,
    )

    # ── Phase 3: Algorithm Execution ──────────────────────────────────────────
    # Instantiate the solver and supply the raw NumPy array; internal methods 
    # handle the state preparation isometry.
    hhl = HHL()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solution = hhl.solve(matrix, b_norm)

    # ── Phase 4: Statevector Extraction ───────────────────────────────────────
    x_raw = _extract_solution_statevector(solution.state, num_qubits)

    # ── Phase 5: Proportionality Recovery ─────────────────────────────────────
    # The extracted vector x_raw is proportional to A^{-1} b_norm, subject to 
    # Trotterisation error. The physical scaling constant is computed via:
    #   c * A * x_raw ≈ b  =>  c = (b · A x_raw) / ||A x_raw||^2
    Ax = A @ x_raw
    c  = float(np.dot(b, Ax) / np.dot(Ax, Ax))
    u  = c * x_raw

    return SolverResult(
        u=u,
        solver="HHL",
        raw_state=x_raw,
        prop_const=c,
        euclidean_residual=_relative_residual(A, u, b),
    )


# ── Private Utility Methods ───────────────────────────────────────────────────

def _relative_residual(A: np.ndarray, u: np.ndarray, b: np.ndarray) -> float:
    """Computes the relative Euclidean residual ||Au - b||_2 / ||b||_2."""
    return float(np.linalg.norm(A @ u - b) / np.linalg.norm(b))


def _extract_solution_statevector(
    circuit,
    num_qubits: int,
) -> np.ndarray:
    """
    Extracts the solution vector from the HHL output quantum circuit via 
    deterministic statevector simulation and rigorous post-selection.

    Register Layout (Qiskit little-endian ordering):
        qregs[0] (q0) : b-register (solution), n_b qubits, indices [0, n_b - 1]
        qregs[1] (q1) : l-register (clock), n_l qubits, indices [n_b, n_b + n_l - 1]
        qregs[2] (a1) : MCMT ancilla, n_a qubits, indices [n_b + n_l, n_total - 2]
        qregs[3] (q2) : Flag qubit, 1 qubit, index [n_total - 1]

    Post-Selection Criteria:
        1. Flag qubit = |1⟩ (indicates successful controlled rotation).
        2. l-register = |0...0⟩ (clock register successfully cleared by inverse QPE).
        3. MCMT ancilla = |0...0⟩ (ancillary hardware returned to ground state).

    Amplitudes are only retained if the computational basis state simultaneously 
    satisfies all three conditions.
    """
    N       = 2 ** num_qubits
    n_total = circuit.num_qubits

    # Extract register dimensions directly from the circuit instance to 
    # maintain robustness against upstream library modifications.
    n_b = circuit.qregs[0].size   # Solution register
    n_l = circuit.qregs[1].size   # Clock register

    # Determine the number of ancillary qubits situated between the clock 
    # register and the terminal flag qubit.
    n_ancilla = n_total - 1 - n_b - n_l

    # Bit positions within the little-endian statevector integer index:
    #   b-register : bits [0, n_b - 1]
    #   l-register : bits [n_b, n_b + n_l - 1]
    #   MCMT anc.  : bits [n_b + n_l, n_b + n_l + n_ancilla - 1]
    #   Flag qubit : bit  [n_total - 1]
    flag_bit_pos  = n_total - 1
    clock_start   = n_b
    ancilla_start = n_b + n_l

    # Construct a composite bitmask to isolate the l-register and MCMT ancilla; 
    # a valid solution state requires these masked bits to evaluate to zero.
    non_b_non_flag_mask = (
        ((1 << n_l)       - 1) << clock_start
        | ((1 << n_ancilla) - 1) << ancilla_start
    )

    # Execute local statevector simulation, bypassing Aer backend overhead.
    sv = Statevector(circuit).data

    x_raw = np.zeros(N, dtype=complex)

    for idx in range(2 ** n_total):
        flag_bit    = (idx >> flag_bit_pos) & 1
        middle_bits = idx & non_b_non_flag_mask
        b_reg_idx   = idx & (N - 1)   # Isolates the lowest n_b bits

        # Enforce post-selection: flag must be set, and intermediate registers cleared.
        if flag_bit == 1 and middle_bits == 0:
            x_raw[b_reg_idx] = sv[idx]

    # The analytical Poisson solution is strictly real; imaginary components 
    # manifest exclusively as Trotterisation artefacts and are discarded.
    x_raw_real = np.real(x_raw)

    # Diagnostic routine: Identifies dominant statevector components if the 
    # extraction yields a null vector, assisting in basis state debugging.
    if np.allclose(x_raw_real, 0.0, atol=1e-12):
        print("\nDEBUG — Dominant statevector amplitudes by magnitude:")
        magnitudes = np.abs(sv)
        top_indices = np.argsort(magnitudes)[::-1][:10]
        for idx in top_indices:
            bits = format(idx, f"0{n_total}b")[::-1]   # LSB first
            print(
                f"  idx={idx:5d}  bits(LSB first)={bits}  "
                f"|amp|={magnitudes[idx]:.6f}  "
                f"flag={(idx >> flag_bit_pos)&1}  "
                f"clock={(idx & (((1<<n_l)-1)<<n_b))>>n_b:0{n_l}b}  "
                f"b_reg={idx & (N-1)}"
            )
        raise RuntimeError(
            f"HHL extraction returned a null vector under strict post-selection.\n"
            f"Circuit Registers: {[(r.name, r.size) for r in circuit.qregs]}\n"
            f"System Parameters: n_total={n_total}, n_b={n_b}, n_l={n_l}, n_ancilla={n_ancilla}.\n"
            f"Consult the DEBUG diagnostic output for dominant vector states."
        )

    return x_raw_real