"""
Block encoding circuits for Hermitian matrices arising from
finite-difference discretisation of the Poisson equation.

Mathematical foundation
-----------------------
A (alpha, n_a, 0)-block encoding of a matrix A in C^{N x N} is a
unitary U_A acting on (n_a + n) qubits such that:

    (<0^{n_a}| x I_n) U_A (|0^{n_a}> x I_n) = A / alpha

where alpha >= ||A||_2 is the subnormalisation factor and n_a is the
number of ancilla qubits.

Implementation: Sz.-Nagy unitary dilation
------------------------------------------
For small N (as encountered in this project with N in {4, 8, 16}), the
most reliable and numerically exact block encoding is the Sz.-Nagy
dilation. Given a matrix M = A/alpha satisfying ||M||_2 <= 1, the
2N x 2N unitary:

    U = [[M,          sqrt(I - M^2)],
         [sqrt(I - M^2), -M        ]]

satisfies (<0|_anc x I_n) U (|0>_anc x I_n) = M = A/alpha exactly,
using a single ancilla qubit (n_a = 1).

The matrix square root sqrt(I - M^2) is computed via eigendecomposition:
    I - M^2 = V diag(1 - lambda_k^2) V^dagger
    sqrt(I - M^2) = V diag(sqrt(1 - lambda_k^2)) V^dagger

This approach is:
  - Exact to numerical precision (no Trotter approximation)
  - Requires only n + 1 total qubits (one ancilla)
  - Valid for any Hermitian matrix with ||A||_2 <= alpha
  - Straightforwardly extensible to the 2-D row matrix (a=-4, b=1)

Subnormalisation factor
-----------------------
For the 1-D Poisson TST matrix (a=-2, b=1):
    alpha = ||A||_2 = |lambda_max| = 2 + 2*cos(pi/(N+1)) < 4

For the 2-D line-Jacobi row matrix (a=-4, b=1):
    alpha = ||A||_2 = |lambda_max| = 4 + 2*cos(pi/(N+1)) < 6

Using alpha = spectral norm ensures the tightest valid subnormalisation,
minimising the effective condition number kappa_eff = alpha * kappa / ||A||_2
= kappa exactly (since alpha = ||A||_2).

References
----------
Gilyen, A., Su, Y., Low, G. H. & Wiebe, N. (2019). Quantum singular
    value transformation and beyond. STOC 2019, pp. 193-204.
Camps, D., Lin, L., Van Beeumen, R. & Yang, C. (2022). Explicit
    quantum circuits for block encodings of certain sparse matrices.
    SIAM J. Matrix Anal. Appl., 43(3), 1183-1207.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

# Number of ancilla qubits for the Sz.-Nagy block encoding.
_N_ANCILLA_BE = 1


# -- Public interface ---------------------------------------------------------

def build_tst_block_encoding(
    N         : int,
    main_diag : float,
    off_diag  : float,
) -> tuple[QuantumCircuit, float]:
    """
    Construct a block encoding circuit for the N x N TST Poisson matrix
    via the Sz.-Nagy unitary dilation.

    The circuit implements:

        (<0_anc| x I_n) U_A (|0_anc> x I_n) = A / alpha

    where alpha = ||A||_2 (spectral norm) is the subnormalisation factor.

    Parameters
    ----------
    N : int
        System size; must be a power of 2.
    main_diag : float
        Main diagonal value (e.g. -2 for 1-D Poisson, -4 for 2-D row).
    off_diag : float
        Off-diagonal value (e.g. +1 for both Poisson formulations).

    Returns
    -------
    circuit : QuantumCircuit
        Block encoding circuit on n_a + n qubits.
        Register layout (Qiskit little-endian):
            qubits 0..n-1  : data register
            qubit  n       : ancilla register
    alpha : float
        Subnormalisation factor (spectral norm of A).

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
    Extract the top-left N x N block of the block encoding unitary.

    This block should equal A / alpha to within numerical precision.
    Used for verification of the block encoding construction.

    The extraction post-selects on the ancilla qubit (qubit index n,
    the MSB in Qiskit's little-endian convention) being in state |0>.

    Parameters
    ----------
    circuit : QuantumCircuit
        Block encoding circuit with n data qubits and 1 ancilla qubit.
    n : int
        Number of data qubits; N = 2^n.

    Returns
    -------
    block : np.ndarray, shape (N, N)
        The A/alpha block of the unitary matrix.
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
    Return the spectral norm of the TST matrix as the subnormalisation
    factor alpha for the Sz.-Nagy block encoding.

    For large N, the spectral norm approaches:
        1-D Poisson (a=-2, b=1): alpha -> 4
        2-D row matrix (a=-4, b=1): alpha -> 6

    Parameters
    ----------
    main_diag : float
    off_diag  : float
    N : int
        System size used to compute the exact spectral norm.
        Default 4 (smallest non-trivial case).

    Returns
    -------
    alpha : float
        Spectral norm ||A||_2.
    """
    A    = (
        main_diag * np.eye(N)
        + off_diag  * np.diag(np.ones(N - 1), k=1)
        + off_diag  * np.diag(np.ones(N - 1), k=-1)
    )
    eigs = np.linalg.eigvalsh(A)
    return float(np.max(np.abs(eigs)))


# -- Private helpers ----------------------------------------------------------

def _sznagy_dilation(M: np.ndarray) -> np.ndarray:
    """
    Construct the Sz.-Nagy unitary dilation of a Hermitian matrix M
    satisfying ||M||_2 <= 1, using the Wx signal convention.

    The Wx convention dilation is the 2N x 2N unitary:

        U = [[M,               i * sqrt(I - M^2)],
             [i * sqrt(I - M^2), M              ]]

    This matches the Wx signal processing unitary used by pyqsp sym_qsp:

        W_x = [[x,          i * sqrt(1 - x^2)],
               [i * sqrt(1 - x^2), x         ]]

    at the scalar level, ensuring that the QSP phase angles computed by
    pyqsp for the Wx convention are correctly applied by the circuit.

    The original Sz.-Nagy dilation uses -M in the bottom-right block
    (the Rx convention). This causes sign alternations in the garbage
    space that interfere destructively for multi-eigenvector inputs,
    causing QSVT to fail for asymmetric RHS vectors (e.g. the HET
    linear profile) while accidentally working for symmetric inputs
    (e.g. the generic Poisson fS source, which is approximately a
    single eigenvector).

    Parameters
    ----------
    M : np.ndarray, shape (N, N)
        Hermitian matrix with ||M||_2 <= 1.

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