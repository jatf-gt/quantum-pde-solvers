"""
Provides utility functions for the Variational Quantum Linear Solver (VQLS).

This module implements the foundational components required for VQLS execution, 
including the Pauli decomposition of the Toeplitz Symmetric Tridiagonal (TST) 
operator (Linear Combination of Unitaries representation), hardware-efficient 
ansatz construction, cost function evaluation via Hadamard test equivalents, 
and the final physical dimensionality recovery.

Circuit construction is entirely delegated to PennyLane, whereas classical 
linear algebraic operations rely on NumPy.

Reference: Bravo-Prieto et al., "Variational Quantum Linear Solver",
           Quantum 7, 1188 (2023).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pennylane as qml


# ── Pauli Decomposition ───────────────────────────────────────────────────────

def pauli_decompose_tst(
    N:         int,
    main_diag: float,
    off_diag:  float,
) -> List[Tuple[complex, str]]:
    """
    Computes the decomposition of the N×N TST matrix into a Linear Combination 
    of Unitaries (LCU) via Pauli strings.

    The system matrix is represented as:
        A = sum_l c_l * P_l

    where each P_l constitutes a tensor product of single-qubit Pauli operators 
    {I, X, Y, Z} acting across n = log2(N) qubits.

    The decomposition coefficients are analytically derived via the projection:
        c_l = (1/N) * Tr(A · P_l)

    For the TST matrix characterised by `main_diag` and `off_diag`, only O(n) 
    Pauli strings possess non-zero coefficients, rendering this an efficient 
    representational schema.

    Parameters
    ----------
    N : int
        System dimension (must be a positive power of 2).
    main_diag : float
        Magnitude of the principal diagonal elements (e.g., -2.0 for 1D Poisson).
    off_diag : float
        Magnitude of the adjacent off-diagonal elements (e.g., 1.0 for 1D Poisson).

    Returns
    -------
    List[Tuple[complex, str]]
        A collection of (coefficient, pauli_string) pairs, omitting elements 
        with zero coefficients. Strings adhere to the PennyLane convention: 
        the rightmost character acts upon qubit 0 (Least Significant Bit).
    """
    n = int(np.log2(N))

    # Build the full TST matrix explicitly for the decomposition.
    # N is small (≤ 32) so this is fine.
    A = (
        main_diag * np.eye(N)
        + off_diag  * np.diag(np.ones(N - 1), k=1)
        + off_diag  * np.diag(np.ones(N - 1), k=-1)
    )

    # Generate all n-qubit Pauli strings and compute their coefficients.
    pauli_labels = ["I", "X", "Y", "Z"]
    pauli_matrices = {
        "I": np.eye(2,      dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }

    terms = []

    # Iterate over all 4^n Pauli strings.
    for idx in range(4**n):
        # Decode idx into a Pauli string of length n.
        pauli_str = ""
        tmp = idx
        for _ in range(n):
            pauli_str = pauli_labels[tmp % 4] + pauli_str
            tmp //= 4

        # Build the n-qubit Pauli matrix as a tensor product.
        P = np.array([[1.0 + 0j]])
        for char in pauli_str:
            P = np.kron(P, pauli_matrices[char])

        # Coefficient: c_l = (1/N) * Tr(A · P_l)
        coeff = complex(np.trace(A @ P) / N)

        if abs(coeff) > 1e-12:
            terms.append((coeff, pauli_str))

    return terms


def pauli_decompose_normalised(
    N:         int,
    main_diag: float,
    off_diag:  float,
) -> Tuple[List[Tuple[complex, str]], float]:
    """
    Decomposes the spectrally normalised TST matrix, A / ||A||_2, into Pauli strings.

    Returns both the decomposition coefficients and the spectral normalisation 
    factor, ensuring the physical solution dimensionality can be accurately 
    recovered following the variational optimisation.

    Parameters
    ----------
    N : int
        System dimension.
    main_diag : float
        Magnitude of the principal diagonal elements.
    off_diag : float
        Magnitude of the adjacent off-diagonal elements.

    Returns
    -------
    terms : List[Tuple[complex, str]]
        The Pauli decomposition of the normalised operator, A_norm.
    norm_factor : float
        The spectral norm of the original matrix A, ||A||_2.
    """
    # Build A to compute its spectral norm.
    A = (
        main_diag * np.eye(N)
        + off_diag  * np.diag(np.ones(N - 1), k=1)
        + off_diag  * np.diag(np.ones(N - 1), k=-1)
    )
    norm_factor = float(np.linalg.norm(A, ord=2))

    terms = pauli_decompose_tst(
        N,
        main_diag / norm_factor,
        off_diag  / norm_factor,
    )
    return terms, norm_factor


# ── Ansatz Construction ───────────────────────────────────────────────────────

def build_ansatz(
    params: np.ndarray,
    n_qubits: int,
    n_layers: int,
) -> None:
    """
    Constructs and applies a hardware-efficient layered ansatz to the active 
    PennyLane quantum circuit.

    Structural topology per layer:
        1. Unparameterised RY(θ_{d,i}) rotations acting on each qubit i.
        2. A linear CNOT entangling chain: qubit 0 → 1 → 2 → … → n-1.

    The terminal layer applies final RY rotations devoid of a subsequent 
    CNOT chain. The aggregate parameter count equates to n_qubits * (n_layers + 1).

    Note: This routine is strictly invoked within a PennyLane QNode context. 
    It sequentially applies quantum operations and yields no return variables.

    Parameters
    ----------
    params : np.ndarray
        Flattened array of variational parameters of length n_qubits * (n_layers + 1).
    n_qubits : int
        Total number of qubits comprising the active register.
    n_layers : int
        Total number of entangling layers to construct.
    """
    params_2d = params.reshape(n_layers + 1, n_qubits)

    for d in range(n_layers):
        # Rotation layer.
        for i in range(n_qubits):
            qml.RY(params_2d[d, i], wires=i)
        # Entangling layer: linear CNOT chain.
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

    # Final rotation layer (no CNOT after the last layer).
    for i in range(n_qubits):
        qml.RY(params_2d[n_layers, i], wires=i)


def n_params(n_qubits: int, n_layers: int) -> int:
    """Computes the total number of variational parameters necessitated by the ansatz."""
    return n_qubits * (n_layers + 1)


# ── Cost Function Evaluation ──────────────────────────────────────────────────

def build_cost_function(
    pauli_terms:  List[Tuple[complex, str]],
    b_norm:       np.ndarray,
    n_qubits:     int,
    n_layers:     int,
    device_name:  str = "default.qubit",
) -> callable:
    """
    Constructs and returns the VQLS global cost function as a callable routine.

    The global cost function evaluates the objective landscape:

        C(θ) = 1 - |<b|A|x(θ)>|² / <x(θ)|A†A|x(θ)>

    utilising a mathematical equivalent of the Hadamard test decomposition 
    over the specified LCU terms.

    Within this specific implementation, both the numerator and denominator 
    are computed strictly via statevector arithmetic rather than explicit 
    circuit-level Hadamard tests. This ensures exact analytical evaluation 
    on classical simulators while circumventing the profound computational 
    overhead associated with constructing O(L) distinct circuits per evaluation.
    (On physical hardware, explicit Hadamard tests would be necessitated).

    Parameters
    ----------
    pauli_terms : List[Tuple[complex, str]]
        The LCU decomposition of the system operator.
    b_norm : np.ndarray
        The unit-normalised right-hand side target vector.
    n_qubits : int
        Total number of qubits comprising the data register.
    n_layers : int
        Total number of entangling ansatz layers.
    device_name : str, default="default.qubit"
        The designated PennyLane simulation device.

    Returns
    -------
    Callable[[np.ndarray], float]
        A function mapping a variational parameter array to a scalar cost 
        value bounded within [0, 1].
    """
    dev = qml.device(device_name, wires=n_qubits + 1)
    # Ancilla qubit is the last wire (index n_qubits).
    ancilla = n_qubits

    # ── Numerator: <b|A|x(θ)> ────────────────────────────────────────────────
    # We compute this directly using statevector inner products rather than
    # Hadamard tests, which is valid on a statevector simulator and avoids
    # the overhead of constructing O(L) separate circuits.
    # On real hardware, Hadamard tests would be required.

    @qml.qnode(qml.device(device_name, wires=n_qubits), interface="autograd")
    def state_x(params):
        """Prepares the variational ansatz state |x(θ)>."""
        build_ansatz(params, n_qubits, n_layers)
        return qml.state()

    def cost_fn(params: np.ndarray) -> float:
        """
        Evaluates the objective C(θ) = 1 - |<b|A|x(θ)>|² / <x(θ)|A†A|x(θ)>.

        Both components are computed via exact statevector algebra, constituting 
        an idealised execution environment exclusive to classical simulation.
        """
        # Get the current ansatz state as a complex vector.
        x_state = np.array(state_x(params), dtype=complex)

        # Reconstruct A from the Pauli decomposition and apply to |x>.
        N = 2 ** n_qubits
        Ax = np.zeros(N, dtype=complex)
        for coeff, pstr in pauli_terms:
            P  = _pauli_string_to_matrix(pstr)
            Ax += coeff * (P @ x_state)

        # Numerator: |<b|A|x>|²
        numerator = abs(np.dot(b_norm.conj(), Ax)) ** 2

        # Denominator: <x|A†A|x> = ||A|x>||²
        denominator = float(np.real(np.dot(Ax.conj(), Ax)))

        if denominator < 1e-14:
            return 1.0   # degenerate case — return maximum cost

        return float(1.0 - numerator / denominator)

    return cost_fn


# ── Dimensionality Recovery ───────────────────────────────────────────────────

def recover_solution(
    params:       np.ndarray,
    A:            np.ndarray,
    b:            np.ndarray,
    n_qubits:     int,
    n_layers:     int,
    device_name:  str = "default.qubit",
) -> Tuple[np.ndarray, float]:
    """
    Extracts the physically dimensioned solution vector from the optimally 
    trained ansatz parameters.

    The converged ansatz state |x(θ*)> satisfies the proportional relation:
        A_norm |x(θ*)> ∝ |b_norm>

    where A_norm = A / ||A||_2 and b_norm = b / ||b||_2.

    The proportionality constant is recovered against the normalised system 
    to constrain all numerical quantities to O(1), followed by a rescaling 
    projection to the physical domain:

        u = c_norm * x_raw * (||b||_2 / ||A||_2)

    This methodology explicitly mitigates the amplification of quantum noise 
    driven by the scaling factor ||b||_2 / ||A||_2. Such regularisation is 
    critical for physically scaled domains (e.g., the Heterogeneous Poisson 
    equation) where ||b|| ~ α·h² ≫ 1.

    Parameters
    ----------
    params : np.ndarray
        Optimised variational parameters yielding the minimal cost state.
    A : np.ndarray
        Original (unnormalised) dense system matrix.
    b : np.ndarray
        Original (unnormalised) right-hand side target vector.
    n_qubits : int
        Total number of qubits comprising the data register.
    n_layers : int
        Total number of entangling ansatz layers.
    device_name : str, default="default.qubit"
        The designated PennyLane simulation device.

    Returns
    -------
    u : np.ndarray
        Recovered physical solution array.
    c_phys : float
        Effective proportionality constant projected into physical units.
    """
    import pennylane as qml

    @qml.qnode(qml.device(device_name, wires=n_qubits), interface="autograd")
    def state_circuit(p):
        build_ansatz(p, n_qubits, n_layers)
        return qml.state()

    x_raw = np.real(np.array(state_circuit(params), dtype=complex))

    # Normalise A and b — recover c against the normalised system.
    A_norm_factor = float(np.linalg.norm(A, ord=2))
    b_norm_factor = float(np.linalg.norm(b))

    A_norm = A / A_norm_factor
    b_norm = b / b_norm_factor

    # Least-squares recovery against normalised system:
    #   c_norm · A_norm · x_raw ≈ b_norm
    #   c_norm = (b_norm · A_norm·x_raw) / ||A_norm·x_raw||²
    Ax_norm = A_norm @ x_raw
    denom   = float(np.dot(Ax_norm, Ax_norm))

    if denom < 1e-14:
        raise RuntimeError(
            "Proportionality recovery failed: ||A_norm·x_raw||² is "
            "numerically zero.  The ansatz may not have converged."
        )

    c_norm = float(np.dot(b_norm, Ax_norm) / denom)

    # Rescale to physical units.
    # u = c_norm · x_raw · ||b|| / ||A||_2
    scale = b_norm_factor / A_norm_factor
    u     = c_norm * scale * x_raw

    # Effective proportionality constant in physical units.
    c_phys = c_norm * scale

    return u, c_phys


# ── Private Utility Methods ───────────────────────────────────────────────────

def _pauli_string_to_matrix(pauli_str: str) -> np.ndarray:
    """
    Translates a sequential Pauli string (e.g., 'XZI') into its explicit 
    dense matrix representation.

    The input string adheres to a big-endian structural convention (the 
    leftmost character operates upon the highest index qubit), preserving 
    parity with the output geometry of `pauli_decompose_tst`.
    """
    pauli_matrices = {
        "I": np.eye(2,      dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    result = np.array([[1.0 + 0j]])
    for char in pauli_str:
        result = np.kron(result, pauli_matrices[char])
    return result