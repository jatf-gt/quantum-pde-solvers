"""
Block encoding circuits for Hermitian matrices arising from finite-difference
discretisation of the Poisson equation.

Mathematical foundation
-----------------------
An (α, n_a, 0)-block encoding of a matrix A ∈ ℂ^{N×N} is a unitary U_A acting on
(n_a + n) qubits such that

    (⟨0^{n_a}| ⊗ I_n) U_A (|0^{n_a}⟩ ⊗ I_n) = A / α

where α ≥ ‖A‖₂ is the subnormalisation factor and n_a the number of ancilla
qubits.

Implementation: Sz.-Nagy unitary dilation
-----------------------------------------
For the small N encountered in this project (N ∈ {4, 8, 16}), the most reliable
and numerically exact block encoding is the Sz.-Nagy dilation. Given a matrix
M = A/α satisfying ‖M‖₂ ≤ 1, the 2N × 2N unitary

    U = [[M,        √(I - M²)],
         [√(I - M²),       -M]]

satisfies (⟨0|_anc ⊗ I_n) U (|0⟩_anc ⊗ I_n) = M = A/α exactly, using a single
ancilla qubit (n_a = 1).

The matrix square root √(I - M²) is computed via eigendecomposition:

    I - M²   = V diag(1 - λ_k²) V†
    √(I - M²) = V diag(√(1 - λ_k²)) V†

This approach is exact to numerical precision (no Trotter approximation),
requires only n + 1 qubits in total, is valid for any Hermitian matrix with
‖A‖₂ ≤ α, and extends unchanged to the line-decomposed strip operator.

Note that the circuit actually built here uses the *Wx* convention rather than
the dilation shown above — see `_sznagy_dilation`, where the distinction is
documented in full and is load-bearing.

Subnormalisation factor
-----------------------
For the 1D Poisson TST matrix (a = -2, b = 1):

    α = ‖A‖₂ = |λ_max| = 2 + 2cos(π/(N+1)) < 4

For the h²-scaled 2D strip matrix (a = -4, b = 1):

    α = ‖A‖₂ = |λ_max| = 4 + 2cos(π/(N+1)) < 6

Taking α to be the spectral norm gives the tightest valid subnormalisation, so
the effective condition number κ_eff = α·κ/‖A‖₂ reduces to κ exactly.

References
----------
Gilyén, A., Su, Y., Low, G. H. & Wiebe, N. (2019). Quantum singular value
    transformation and beyond. STOC 2019, pp. 193-204.
Camps, D., Lin, L., Van Beeumen, R. & Yang, C. (2022). Explicit quantum circuits
    for block encodings of certain sparse matrices. SIAM J. Matrix Anal. Appl.,
    43(3), 1183-1207.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

# Number of ancilla qubits for the Sz.-Nagy block encoding.
_N_ANCILLA_BE = 1


# ── Public Interface ──────────────────────────────────────────────────────────

def build_tst_block_encoding(
    N         : int,
    main_diag : float,
    off_diag  : float,
) -> tuple[QuantumCircuit, float]:
    """
    Constructs a block encoding circuit for the N×N TST Poisson matrix via the
    Sz.-Nagy unitary dilation.

    The circuit implements

        (⟨0_anc| ⊗ I_n) U_A (|0_anc⟩ ⊗ I_n) = A / α

    where α = ‖A‖₂ is the subnormalisation factor.

    Parameters
    ----------
    N : int
        System size; must be a power of 2.
    main_diag : float
        Main diagonal value (e.g. -2 for 1D Poisson, -4 for the h²-scaled 2D
        strip operator).
    off_diag : float
        Off-diagonal value (+1 for both Poisson formulations).

    Returns
    -------
    circuit : QuantumCircuit
        Block encoding circuit on n_a + n qubits, with register layout
        (Qiskit little-endian): qubits 0…n-1 the data register, qubit n the
        ancilla register.
    alpha : float
        Subnormalisation factor, the spectral norm of A.

    Raises
    ------
    ValueError
        If N is not a positive power of 2.
    """
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError(
            f"N must be a positive power of 2, received N={N}."
        )

    n = int(np.log2(N))

    # Build the TST matrix explicitly.
    A = (
        main_diag * np.eye(N)
        + off_diag  * np.diag(np.ones(N - 1), k=1)
        + off_diag  * np.diag(np.ones(N - 1), k=-1)
    )

    # Subnormalisation: spectral norm = largest singular value.
    # For a symmetric matrix this equals the largest absolute eigenvalue.
    eigs  = np.linalg.eigvalsh(A)
    alpha = float(np.max(np.abs(eigs)))

    # Normalised matrix: M = A / alpha, ||M||_2 = 1.
    M = A / alpha

    # Sz.-Nagy dilation: construct the 2N x 2N unitary.
    U_2N = _sznagy_dilation(M)

    # Embed into a quantum circuit.
    # Total qubits: n (data) + 1 (ancilla).
    # The 2N x 2N unitary acts on (n+1) qubits.
    data = QuantumRegister(n,              name="data")
    anc  = AncillaRegister(_N_ANCILLA_BE,  name="anc_be")
    qc   = QuantumCircuit(data, anc, name="BlockEnc_TST")

    # In Qiskit's convention, the circuit acts on qubits [data[0], ...,
    # data[n-1], anc[0]], corresponding to the statevector ordering where
    # data[0] is the LSB and anc[0] is the MSB.
    # The UnitaryGate is applied to all n+1 qubits in this order.
    gate = UnitaryGate(U_2N, label="U_BE")
    qc.append(gate, list(range(n)) + [n])

    return qc, alpha


def block_encoding_matrix(
    circuit : QuantumCircuit,
    n       : int,
) -> np.ndarray:
    """
    Extracts the top-left N×N block of the block encoding unitary.

    This block equals A/α to within numerical precision, and the function exists
    to verify that. The extraction post-selects on the ancilla qubit (index n,
    the MSB in Qiskit's little-endian convention) being in state |0⟩.

    Parameters
    ----------
    circuit : QuantumCircuit
        Block encoding circuit with n data qubits and 1 ancilla qubit.
    n : int
        Number of data qubits; N = 2ⁿ.

    Returns
    -------
    block : np.ndarray, shape (N, N)
        The A/α block of the unitary matrix.
    """
    N     = 2**n
    U     = Operator(circuit).data   # shape (2N, 2N)

    # In Qiskit's little-endian statevector convention:
    #   qubit 0 (data[0]) is bit 0 of the index (LSB)
    #   qubit n (anc[0])  is bit n of the index (MSB)
    #
    # Basis states with ancilla = 0: indices where bit n = 0,
    # i.e. indices 0, 1, ..., N-1 (the lower half of the 2N-dim space).
    # Basis states with ancilla = 1: indices N, N+1, ..., 2N-1.
    #
    # The block encoding condition is:
    #   U[anc=0 output, anc=0 input] = A/alpha
    # which corresponds to the top-left N x N submatrix of U.
    return U[:N, :N]


def subnormalisation_factor(
    main_diag : float,
    off_diag  : float,
    N         : int = 4,
) -> float:
    """
    Returns the spectral norm of the TST matrix, used as the subnormalisation
    factor α for the Sz.-Nagy block encoding.

    For large N the spectral norm approaches 4 for the 1D Poisson operator
    (a = -2, b = 1) and 6 for the h²-scaled 2D strip operator (a = -4, b = 1).

    Parameters
    ----------
    main_diag : float
        Main diagonal value of the TST matrix.
    off_diag : float
        Off-diagonal value of the TST matrix.
    N : int
        System size at which the exact spectral norm is evaluated. Default 4,
        the smallest non-trivial case.

    Returns
    -------
    alpha : float
        Spectral norm ‖A‖₂.
    """
    A    = (
        main_diag * np.eye(N)
        + off_diag  * np.diag(np.ones(N - 1), k=1)
        + off_diag  * np.diag(np.ones(N - 1), k=-1)
    )
    eigs = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(eigs)))


# ── Private Helpers ───────────────────────────────────────────────────────────

def _sznagy_dilation(M: np.ndarray) -> np.ndarray:
    """
    Constructs the Sz.-Nagy unitary dilation of a Hermitian matrix M satisfying
    ‖M‖₂ ≤ 1, using the Wx signal convention.

    The Wx convention dilation is the 2N × 2N unitary

        U = [[M,           i√(I - M²)],
             [i√(I - M²),           M]]

    which matches the Wx signal processing unitary used by pyqsp sym_qsp,

        W_x = [[x,          i√(1 - x²)],
               [i√(1 - x²),          x]]

    at the scalar level, ensuring that the QSP phase angles computed by pyqsp
    for the Wx convention are correctly applied by the circuit.

    The original Sz.-Nagy dilation uses -M in the bottom-right block (the Rx
    convention). This causes sign alternations in the garbage space that
    interfere destructively for multi-eigenvector inputs, making QSVT fail for
    asymmetric right-hand sides (e.g. the HET linear profile) whilst
    accidentally succeeding for symmetric ones (e.g. the generic Poisson fS
    source, which is approximately a single eigenvector).

    Parameters
    ----------
    M : np.ndarray, shape (N, N)
        Hermitian matrix with ‖M‖₂ ≤ 1.

    Returns
    -------
    U : np.ndarray, shape (2N, 2N), complex
        Unitary dilation in the Wx convention.
    """
    N   = M.shape[0]
    I   = np.eye(N)

    # Compute i * sqrt(I - M^2) via eigendecomposition.
    # Since M is Hermitian and ||M||_2 <= 1, all eigenvalues of I - M^2
    # are non-negative, so the square root is real and PSD.
    ImM2     = I - M @ M
    eigs, V  = np.linalg.eigh(ImM2)
    eigs_pos = np.clip(eigs, 0.0, None)
    sqrtImM2 = V @ np.diag(np.sqrt(eigs_pos)) @ V.conj().T

    # Assemble the 2N x 2N Wx-convention dilation.
    # Both diagonal blocks are +M (not M and -M as in the Rx convention).
    # Off-diagonal blocks carry a factor of i.
    U = np.zeros((2 * N, 2 * N), dtype=complex)
    U[:N, :N] =  M                  # top-left:     ancilla 0 -> ancilla 0
    U[:N, N:] =  1j * sqrtImM2      # top-right:    ancilla 1 -> ancilla 0
    U[N:, :N] =  1j * sqrtImM2      # bottom-left:  ancilla 0 -> ancilla 1
    U[N:, N:] =  M                  # bottom-right: ancilla 1 -> ancilla 1

    return U
