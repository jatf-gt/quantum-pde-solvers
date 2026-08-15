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

# Relative tolerance on max|A - A†| / max|A| before A is rejected as non-Hermitian.
# Sufficient to accommodate a matrix assembled in floating-point arithmetic from a symmetric
# stencil, which accumulates round-off asymmetry of order the machine epsilon multiplied by
# the number of accumulated terms; sufficiently stringent to reject a genuinely asymmetric
# operator, such as a one-sided boundary row intended to be symmetrised.
_HERMITIAN_TOL = 1e-10

# Absolute tolerance below which ‖A‖₂ is treated as zero. Subnormalising by a value
# at this scale would amplify round-off into the block encoding without limit.
_ZERO_MATRIX_TOL = 1e-14

# Relative tolerance on entries outside the tridiagonal band, enforced by
# `assert_tridiagonal`. Established well above round-off but significantly below any physically
# meaningful stencil coefficient: the 4th-order operator's ±2 band is 1/12 of its
# main diagonal, roughly eleven orders of magnitude above this threshold, ensuring the guard
# does not conflate a structural band with numerical noise.
_BAND_TOL = 1e-12


# -- Public Interface ----------------------------------------------------------

def is_toeplitz_tridiagonal(A: np.ndarray) -> bool:
    """
    Whether A is exactly what a reconstruction from ``A[0,0]`` and ``A[0,1]`` gives.

    That reconstruction requires two properties, and callers have historically
    checked only the first: the matrix must carry no band beyond |i−j| ≤ 1, AND
    each of those three diagonals must be constant. A tridiagonal operator with a
    varying diagonal — a boundary-modified stencil such as the Neumann sub-case 3c
    — is silently replaced by a uniform one, which is the same corruption as a
    discarded ±2 band arriving by a different route.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Candidate operator.

    Returns
    -------
    bool
        True when the two-scalar fast path reproduces A to within `_BAND_TOL`
        relative to max|A|; False when the dense encoding is required.
    """
    try:
        assert_tridiagonal(A, "probe")
    except ValueError:
        return False
    return True


def assert_tridiagonal(A: np.ndarray, solver: str) -> None:
    """
    Rejects a matrix carrying any band beyond the tridiagonal.

    Guards the two-scalar fast paths in `solvers/quantum/hhl_1d.py` and
    `solvers/quantum/qsvt_1d.py`, both of which reconstruct their operator from
    ``A[0,0]`` and ``A[0,1]`` alone. That reconstruction is exact for the 2nd-order
    TST operator and is retained because it reproduces the published figures
    bit-for-bit — but for the 4th-order pentadiagonal operator it silently discarded
    the ±2 band and solved a *different, tridiagonal* system. Nothing failed: the
    solve converged, the residual was computed against the truncated matrix, and the
    row looked entirely healthy. Every 4th-order HHL and QSVT result produced before
    2026-08-10 is invalid for this reason.

    Raising here converts that silent corruption into an immediate, named error, and
    points the caller at the dense encoding that does handle the operator.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Candidate operator.
    solver : str
        Solver name, for the message (e.g. ``"HHL"``).

    Raises
    ------
    ValueError
        If any entry outside the three central diagonals exceeds
        `_BAND_TOL` relative to the largest entry of A.
    """
    A = np.asarray(A)
    N = A.shape[0]
    if N < 3:
        return

    scale = float(np.max(np.abs(A))) or 1.0
    # Everything outside |i - j| <= 1. Built as a mask rather than by inspecting
    # the k=+-2 diagonals alone, so a stencil wider still cannot slip through.
    idx      = np.arange(N)
    off_band = np.abs(idx[:, None] - idx[None, :]) > 1
    leak     = float(np.max(np.abs(A[off_band]))) if off_band.any() else 0.0

    if leak / scale > _BAND_TOL:
        raise ValueError(
            f"{solver} received a matrix with non-zero entries outside the "
            f"tridiagonal band: max|off-band| / max|A| = {leak / scale:.3e}. "
            f"This code path reconstructs the operator from A[0,0] and A[0,1] "
            f"alone, so those entries would be silently discarded and a different "
            f"system solved. For the 4th-order pentadiagonal operator use the "
            f"order-4 solver ({solver.lower()}_4th), which block encodes A in full "
            f"via build_dense_block_encoding."
        )

    # Being within the band is necessary but not sufficient. The reconstruction
    # is Toeplitz — one scalar per diagonal — so a tridiagonal matrix whose
    # diagonals are not constant is corrupted just as completely as a wider
    # stencil, and by exactly the same mechanism.
    #
    # This is not hypothetical. Sub-case 3c carries a Neumann row at x=0 whose
    # halved form gives A[0,0] = -1 against -2 everywhere else. Reconstruction
    # from A[0,0] therefore built tridiag(1, -1, 1) — a uniformly shifted
    # operator, not the Neumann one — and HHL and QSVT solved that instead, at
    # every N and every degree, returning ~100 % error against 3c's true
    # solution while matching the surrogate's solution to machine precision.
    # The band check above passes 3c cleanly, which is why this went unseen.
    deviation = 0.0
    for k in (-1, 0, 1):
        diag = np.diag(A, k)
        if diag.size > 1:
            deviation = max(deviation,
                            float(np.max(np.abs(diag - diag[0]))))

    if deviation / scale > _BAND_TOL:
        raise ValueError(
            f"{solver} received a tridiagonal matrix that is not Toeplitz: "
            f"max deviation along a diagonal / max|A| = {deviation / scale:.3e}. "
            f"This code path reconstructs the operator from A[0,0] and A[0,1] "
            f"alone, which assumes every diagonal is constant, so a varying "
            f"diagonal would be silently replaced by its first entry and a "
            f"different system solved. Boundary-modified operators such as the "
            f"Neumann sub-case 3c fall here. Use build_dense_block_encoding, "
            f"which encodes A in full at identical asymptotic cost."
        )


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

    # Normalised matrix: M = A / alpha, ‖M‖₂ = 1.
    M = A / alpha

    return _assemble_dilation_circuit(M, alpha, name="BlockEnc_TST")


def build_dense_block_encoding(
    A : np.ndarray,
) -> tuple[QuantumCircuit, float]:
    """
    Constructs a block encoding circuit for an arbitrary Hermitian matrix A.

    The circuit implements

        (⟨0_anc| ⊗ I_n) U_A (|0_anc⟩ ⊗ I_n) = A / α

    with α = ‖A‖₂, identically to `build_tst_block_encoding` — the only difference
    is that A is supplied in full rather than reconstructed from two scalars.

    Purpose
    -------
    The banded structure of A is irrelevant to the Sz.-Nagy dilation, which acts on
    a general Hermitian contraction. `build_tst_block_encoding` is TST-specific only
    in its *constructor*: it rebuilds A from `main_diag` and `off_diag`, and so
    cannot represent an operator with any further bands. Applied to the 4th-order
    pentadiagonal operator that constructor silently discarded the ±2 band, block
    encoding a *different, tridiagonal* matrix. The solve then converged neatly to
    the answer to the wrong problem, which is why the defect survived several runs
    undetected.

    Complexity
    ----------
    The dilation is dense: O(N³) for the eigendecomposition of I - M², and the
    resulting `UnitaryGate` on n+1 qubits synthesises to O(4ⁿ) = O(N²) two-qubit
    gates. This is the same asymptotic cost `build_tst_block_encoding` already pays
    — neither is a structure-exploiting encoding, and neither claims to be. Both
    exist to make the *algorithmic* behaviour of HHL and QSVT measurable at small N,
    not to demonstrate an asymptotic advantage in the encoding itself.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Hermitian matrix, N a power of 2. Any bandwidth, including dense.

    Returns
    -------
    circuit : QuantumCircuit
        Block encoding circuit on n + 1 qubits, with register layout
        (Qiskit little-endian): qubits 0…n-1 the data register, qubit n the
        ancilla register.
    alpha : float
        Subnormalisation factor, the spectral norm of A.

    Raises
    ------
    ValueError
        If A is not square, if its dimension is not a positive power of 2, if it is
        not Hermitian to within `_HERMITIAN_TOL`, or if it is numerically zero.
    """
    A = np.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be a square matrix, received shape {A.shape}.")

    N = A.shape[0]
    if N <= 0 or (N & (N - 1)) != 0:
        raise ValueError(
            f"A's dimension must be a positive power of 2, received N={N}. "
            f"Amplitude encoding addresses the N basis states with log2(N) qubits."
        )

    asymmetry = float(np.max(np.abs(A - A.conj().T))) if N else 0.0
    scale     = float(np.max(np.abs(A))) or 1.0
    if asymmetry / scale > _HERMITIAN_TOL:
        raise ValueError(
            f"A must be Hermitian: max|A - A†| / max|A| = {asymmetry / scale:.3e} "
            f"exceeds {_HERMITIAN_TOL:.1e}. The Sz.-Nagy dilation assumes a "
            f"Hermitian contraction, and a non-Hermitian argument yields a "
            f"non-unitary dilation rather than an error at circuit level."
        )

    eigs  = np.linalg.eigvalsh(A)
    alpha = float(np.max(np.abs(eigs)))
    if alpha < _ZERO_MATRIX_TOL:
        raise ValueError(
            "A is numerically zero; there is no subnormalisation factor and the "
            "linear system is degenerate."
        )

    return _assemble_dilation_circuit(A / alpha, alpha, name="BlockEnc_Dense")


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


# -- Private Helpers -----------------------------------------------------------

def _assemble_dilation_circuit(
    M     : np.ndarray,
    alpha : float,
    name  : str,
) -> tuple[QuantumCircuit, float]:
    """
    Wraps the Sz.-Nagy dilation of a normalised matrix in a circuit.

    Shared by `build_tst_block_encoding` and `build_dense_block_encoding` so that
    the register layout and qubit ordering are defined once. The two constructors
    differ only in how they obtain M; if the embedding conventions were duplicated,
    a QSVT phase sequence calibrated against one would silently misapply to the
    other.

    Parameters
    ----------
    M : np.ndarray, shape (N, N)
        Hermitian matrix with ‖M‖₂ ≤ 1, i.e. A / α.
    alpha : float
        Subnormalisation factor, returned unchanged for the caller's convenience.
    name : str
        Circuit name, identifying the constructor in a drawn circuit.

    Returns
    -------
    circuit : QuantumCircuit
        Block encoding circuit on n + 1 qubits.
    alpha : float
        As supplied.
    """
    N = M.shape[0]
    n = int(np.log2(N))

    # Sz.-Nagy dilation: construct the 2N x 2N unitary.
    U_2N = _sznagy_dilation(M)

    # Embed into a quantum circuit.
    # Total qubits: n (data) + 1 (ancilla).
    # The 2N x 2N unitary acts on (n+1) qubits.
    data = QuantumRegister(n,              name="data")
    anc  = AncillaRegister(_N_ANCILLA_BE,  name="anc_be")
    qc   = QuantumCircuit(data, anc, name=name)

    # In Qiskit's convention, the circuit acts on qubits [data[0], ...,
    # data[n-1], anc[0]], corresponding to the statevector ordering where
    # data[0] is the LSB and anc[0] is the MSB.
    # The UnitaryGate is applied to all n+1 qubits in this order.
    gate = UnitaryGate(U_2N, label="U_BE")
    qc.append(gate, list(range(n)) + [n])

    return qc, alpha


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

    # Compute i√(I − M²) via eigendecomposition.
    # Since M is Hermitian and ||M||_2 <= 1, all eigenvalues of I - M^2
    # are non-negative, so the square root is real and PSD.
    ImM2     = I - M @ M
    eigs, V  = np.linalg.eigh(ImM2)
    eigs_pos = np.clip(eigs, 0.0, None)
    sqrtImM2 = V @ np.diag(np.sqrt(eigs_pos)) @ V.conj().T

    # Assemble the 2N × 2N Wx-convention dilation.
    # Both diagonal blocks are +M (not M and -M as in the Rx convention).
    # Off-diagonal blocks carry a factor of i.
    U = np.zeros((2 * N, 2 * N), dtype=complex)
    U[:N, :N] =  M                  # top-left:     ancilla 0 -> ancilla 0
    U[:N, N:] =  1j * sqrtImM2      # top-right:    ancilla 1 -> ancilla 0
    U[N:, :N] =  1j * sqrtImM2      # bottom-left:  ancilla 0 -> ancilla 1
    U[N:, N:] =  M                  # bottom-right: ancilla 1 -> ancilla 1

    return U
