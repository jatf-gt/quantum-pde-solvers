"""
solvers.py
----------
Two solvers for the 1D Poisson system Au = b:

  1. thomas_solve   — classical Thomas algorithm (tridiagonal direct solver)
  2. hhl_solve      — quantum HHL solver via the quantum_linear_solvers library

IMPORTANT — correct imports:
    The library lives at quantum_linear_solvers.linear_solvers, NOT at
    qiskit.algorithms.linear_solvers.  If both are installed, Python may
    silently resolve to the wrong one, causing "HHL() takes no arguments".
"""
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Correct imports from the anedumla/quantum_linear_solvers library ──────────
# These must come from quantum_linear_solvers, NOT from qiskit.algorithms.
from quantum_linear_solvers.linear_solvers.hhl import HHL
from quantum_linear_solvers.linear_solvers.matrices.tridiagonal_toeplitz import (
    TridiagonalToeplitz,
)

# Qiskit statevector utility — used for solution extraction.
from qiskit.quantum_info import Statevector

from problem_setup import PoissonProblem1D


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class SolverResult:
    """
    Holds everything produced by a single solver run.

    Attributes
    ----------
    u                  : recovered physical solution vector (length N)
    solver             : 'Thomas', 'HHL', or 'NumPy'
    raw_state          : raw quantum state amplitudes from the b-register
                         (HHL only, None otherwise)
    prop_const         : proportionality constant c s.t. c·A·raw_state ≈ b
                         (HHL only, None otherwise)
    euclidean_residual : ||Au - b|| / ||b||
    """
    u:                    np.ndarray
    solver:               str
    raw_state:            Optional[np.ndarray] = field(default=None, repr=False)
    prop_const:           Optional[float]      = None
    euclidean_residual:   Optional[float]      = None


# ── 1. Thomas algorithm ───────────────────────────────────────────────────────

def thomas_solve(problem: PoissonProblem1D) -> SolverResult:
    """
    Solve the tridiagonal system Au = b using the Thomas algorithm
    (Algorithm 1 in the paper).  O(N) time, exact to machine precision.
    """
    N     = problem.config.N
    b_d   = -2.0 * np.ones(N)   # main diagonal (modified in place)
    a_d   =  1.0 * np.ones(N)   # sub-diagonal  (a_d[0] unused)
    c_d   =  1.0 * np.ones(N)   # super-diagonal (c_d[-1] unused)
    d     = problem.b.copy()     # RHS

    # Forward sweep
    for i in range(1, N):
        m     = a_d[i] / b_d[i - 1]
        b_d[i] -= m * c_d[i - 1]
        d[i]   -= m * d[i - 1]

    # Back substitution
    u = np.zeros(N)
    u[-1] = d[-1] / b_d[-1]
    for i in range(N - 2, -1, -1):
        u[i] = (d[i] - c_d[i] * u[i + 1]) / b_d[i]

    return SolverResult(
        u=u,
        solver="Thomas",
        euclidean_residual=_relative_residual(problem.A, u, problem.b),
    )


# ── 2. NumPy reference solver ─────────────────────────────────────────────────

def numpy_solve(problem: PoissonProblem1D) -> SolverResult:
    """Direct solve via NumPy — useful for debugging matrix/RHS assembly."""
    u = np.linalg.solve(problem.A, problem.b)
    return SolverResult(
        u=u,
        solver="NumPy",
        euclidean_residual=_relative_residual(problem.A, u, problem.b),
    )


# ── 3. HHL solver ─────────────────────────────────────────────────────────────

def hhl_solve(problem: PoissonProblem1D) -> SolverResult:
    """
    Solve Au = b using the HHL algorithm from the quantum_linear_solvers
    library, then recover the physical solution via proportionality scaling.

    Four-stage pipeline
    -------------------
    Stage 1 — Normalise A and b.
        Divide A by its spectral norm so eigenvalues lie in (−1, 1].
        Normalise b to a unit vector for amplitude encoding.

    Stage 2 — Build the TridiagonalToeplitz matrix object.
        Uses the TST-specialised Hamiltonian simulation from Vázquez et al.
        The trotter_steps property is set separately after construction
        (the constructor does not accept it as a keyword argument in all
        versions of the library).

    Stage 3 — Prepare the b-register circuit and run HHL.
        The library's solve() requires the RHS as a QuantumCircuit that
        prepares |b_norm⟩ via isometry, not a raw numpy array.

    Stage 4 — Extract the solution vector from the statevector.
        Simulate the returned QuantumCircuit with Qiskit's Statevector
        class, then post-select on the ancilla qubit being |1⟩.

    Stage 5 — Recover the proportionality constant.
        The extracted vector x_raw satisfies c·A·x_raw ≈ b for some c.
        Recover c via least-squares projection against the original A and b
        so the returned u is directly in physical units.

    Key implementation notes
    ------------------------
    - HHL() takes NO constructor arguments in this library version.
      The outer HHL class inherits __init__ from LinearSolver which takes
      no arguments.  The epsilon parameter lives on an unreachable inner
      class — ignore it at construction time.

    - Trotter precision is controlled via matrix.trotter_steps on the
      TridiagonalToeplitz object.  We map cfg.epsilon → trotter_steps
      via ceil(1/epsilon) so the paper's epsilon sweep still works.

    - The vector b can be passed as a raw numpy array to hhl.solve().
      The library's construct_circuit() handles normalisation and
      isometry preparation internally.

    - Solution extraction uses Qiskit's Statevector class directly on
      the returned QuantumCircuit, with post-selection on ancilla = |1>
      and clock register = |0...0>.
    """
    cfg = problem.config
    A   = problem.A
    b   = problem.b
    N   = cfg.N

    # ── Stage 1: normalise A ──────────────────────────────────────────────────
    # Divide A by its spectral norm so eigenvalues lie in (−1, 1].
    # The library normalises b internally, so we only need to normalise A here.
    A_norm_factor = float(np.linalg.norm(A, ord=2))
    b_norm_factor = float(np.linalg.norm(b))

    b_norm = b / b_norm_factor   # unit vector for amplitude encoding
    a_norm = -2.0 / A_norm_factor
    b_off  =  1.0 / A_norm_factor

    # ── Stage 2: build TridiagonalToeplitz ────────────────────────────────────
    # trotter_steps controls the Trotter approximation precision.
    # ceil(1/epsilon) maps epsilon=0.01 → 100 steps, 0.001 → 1000 steps,
    # matching the paper's epsilon sweep in Section IV.
    num_qubits    = int(np.log2(N))
    trotter_steps = max(1, int(np.ceil(1.0 / cfg.epsilon)))

    matrix = TridiagonalToeplitz(
        num_state_qubits=num_qubits,
        main_diag=a_norm,
        off_diag=b_off,
        trotter_steps=trotter_steps,   # accepted in constructor per source
    )

    # ── Stage 3: run HHL ──────────────────────────────────────────────────────
    # HHL() takes no arguments — the outer class inherits LinearSolver.__init__.
    # Pass b_norm as a raw numpy array; the library handles state preparation
    # internally via construct_circuit().
    hhl = HHL()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solution = hhl.solve(matrix, b_norm)

    # ── Stage 4: extract the solution vector ──────────────────────────────────
    x_raw = _extract_solution_statevector(solution.state, num_qubits)

    # ── Stage 5: recover the proportionality constant ─────────────────────────
    # x_raw is proportional to A^{-1} b_norm (up to Trotter error).
    # Recover c against the original A and b so u is in physical units:
    #   c · A · x_raw ≈ b
    #   c = (b · A·x_raw) / ||A·x_raw||²
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


# ── Private helpers ───────────────────────────────────────────────────────────

def _relative_residual(A: np.ndarray, u: np.ndarray, b: np.ndarray) -> float:
    """Relative Euclidean residual ||Au - b|| / ||b||."""
    return float(np.linalg.norm(A @ u - b) / np.linalg.norm(b))


def _extract_solution_statevector(
    circuit,
    num_qubits: int,
) -> np.ndarray:
    """
    Extract the solution vector from the HHL output QuantumCircuit.

    Register layout (from circuit.qregs, Qiskit little-endian ordering):
        qregs[0]  q0  : b-register, n_b qubits, qubit indices 0 … n_b-1
        qregs[1]  q1  : l-register, n_l qubits, qubit indices n_b … n_b+n_l-1
        qregs[2]  a1  : MCMT ancilla, n_a qubits, indices n_b+n_l … n_b+n_l+n_a-1
        qregs[3]  q2  : flag qubit, 1 qubit, index n_total-1

    Post-selection condition for a valid solution state:
        - flag qubit  = 1   (ancilla measured in |1>, HHL succeeded)
        - l-register  = 0   (clock register cleared by inverse QPE)
        - MCMT ancilla = 0  (ancilla register returned to |0>)

    Only the b-register amplitudes satisfying all three conditions are kept.
    """
    from qiskit.quantum_info import Statevector

    N       = 2 ** num_qubits
    n_total = circuit.num_qubits

    # Read register sizes directly from the circuit so this is robust to
    # any changes in the library's qubit count.
    n_b = circuit.qregs[0].size   # b-register  (solution)
    n_l = circuit.qregs[1].size   # l-register  (clock)

    # Everything between the clock and the flag qubit is ancilla.
    # n_total - 1 is the flag qubit index; everything from n_b+n_l to
    # n_total-2 inclusive is ancilla (MCMT etc.).
    n_ancilla = n_total - 1 - n_b - n_l   # e.g. 2 for the a1 register

    # Bit positions in the statevector integer index (little-endian):
    #   b-register : bits 0 … n_b-1
    #   l-register : bits n_b … n_b+n_l-1
    #   MCMT anc.  : bits n_b+n_l … n_b+n_l+n_ancilla-1
    #   flag qubit : bit  n_total-1

    # Build masks for the registers we need to check.
    # A mask selects the relevant bits from the integer index.
    flag_bit_pos   = n_total - 1
    clock_start    = n_b
    ancilla_start  = n_b + n_l

    # Mask that covers the l-register AND the MCMT ancilla — both must be 0.
    non_b_non_flag_mask = (
        ((1 << n_l)       - 1) << clock_start    # l-register bits
        | ((1 << n_ancilla) - 1) << ancilla_start  # MCMT ancilla bits
    )

    # Simulate the statevector — no Aer backend required.
    sv = Statevector(circuit).data   # complex array, length 2^n_total

    x_raw = np.zeros(N, dtype=complex)

    for idx in range(2 ** n_total):
        flag_bit      = (idx >> flag_bit_pos) & 1
        middle_bits   = idx & non_b_non_flag_mask
        b_reg_idx     = idx & (N - 1)   # lowest n_b bits

        # Keep only states where flag=1 and all non-b, non-flag bits are 0.
        if flag_bit == 1 and middle_bits == 0:
            x_raw[b_reg_idx] = sv[idx]

    # The Poisson solution is real; imaginary parts are Trotter noise.
    x_raw_real = np.real(x_raw)

    # Diagnostic: if still all-zero, print the largest-amplitude states
    # to help identify which basis states carry the solution.
    if np.allclose(x_raw_real, 0.0, atol=1e-12):
        print("\nDEBUG — top 10 statevector amplitudes by magnitude:")
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
            f"HHL extraction returned an all-zero vector after corrected masking.\n"
            f"Registers: {[(r.name, r.size) for r in circuit.qregs]}\n"
            f"n_total={n_total}, n_b={n_b}, n_l={n_l}, n_ancilla={n_ancilla}.\n"
            f"See DEBUG output above for the dominant statevector components."
        )

    return x_raw_real