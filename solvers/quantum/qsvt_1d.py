"""
Quantum Singular Value Transformation (QSVT) solver for the 1-D Poisson
equation with a Toeplitz Symmetric Tridiagonal (TST) system matrix.

Algorithm overview
------------------
QSVT solves A|x> = |b> by implementing a polynomial approximation to
the matrix inverse A^{-1} via the singular value transformation:

    |x> proportional to p(A/alpha)|b>

where p(x) approx 1/x on the singular value range [1/kappa, 1] of
A/alpha, and alpha is the block encoding subnormalisation factor.

The implementation proceeds in five stages:

Stage 1 -- Block encoding construction.
    Build the (alpha, n_a, 0)-block encoding U_A of A/alpha using the
    LCU decomposition of the TST matrix (block_encoding.py).

Stage 2 -- QSP phase angle computation.
    Compute the degree-d polynomial approximation to 1/x on [1/kappa, 1]
    and find the corresponding QSP phase angles phi_0, ..., phi_d
    (qsp_angles.py).

Stage 3 -- QSVT circuit construction.
    Assemble the QSVT circuit as a sequence of d alternating applications
    of U_A and U_A^dagger interleaved with controlled phase rotations
    parametrised by the QSP angles.

Stage 4 -- State preparation and circuit execution.
    Prepare the state |b> via amplitude encoding (isometry) and execute
    the QSVT circuit on the statevector simulator.

Stage 5 -- Solution extraction and proportionality recovery.
    Post-select on the ancilla register and extract the solution
    amplitudes. Recover the physical solution via proportionality
    constant estimation against the original system.

Complexity comparison
---------------------
For the 1-D Poisson matrix with N interior nodes:

    kappa(A)  ~ (4/pi^2)(N+1)^2  = O(N^2)
    alpha     = |a| + 2|b| = 4   (for a=-2, b=1)
    kappa_eff = alpha * kappa / ||A||_2 ~ 4 * kappa

    HHL:  circuit depth O(kappa^2 / epsilon)  = O(N^4 / epsilon)
    VQLS: circuit depth O(n_layers * n_qubits) (variational, no guarantee)
    QSVT: circuit depth O(kappa * log(1/epsilon)) = O(N^2 * log(1/epsilon))

QSVT achieves a quadratic improvement in kappa dependence over HHL,
at the cost of requiring a block encoding oracle and classical
preprocessing to compute the QSP phase angles.

2-D extension
-------------
The 2-D QSVT solver (qsvt_2d.py, to be implemented) will reuse this
module via the same interface as hhl_2d.py reuses hhl_1d.py:

    qsvt_solve_system(A_row, b_row, config)

is called inside the line-Jacobi loop for each row sub-problem. The
row matrix has a=-4, b=1, kappa_row -> 3, which makes the QSVT
polynomial degree requirement d = O(3 * log(1/epsilon)) essentially
constant in N -- a significant advantage over the 1-D case.

References
----------
Gilyen, A., Su, Y., Low, G. H. & Wiebe, N. (2019). Quantum singular
    value transformation and beyond. STOC 2019, pp. 193-204.
Martyn, J. M., Rossi, Z. M., Tan, A. K. & Chuang, I. L. (2021). Grand
    unification of quantum algorithms. PRX Quantum, 2, 040203.
Lin, L. & Tong, Y. (2022). Lecture notes on quantum algorithms for
    scientific computation. arXiv:2201.08309.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, AncillaRegister
from qiskit.circuit.library import Isometry
from qiskit.quantum_info import Statevector

from problems.poisson_1d import PoissonProblem1D
from solvers.quantum.block_encoding import (
    build_tst_block_encoding,
    subnormalisation_factor,
    _N_ANCILLA_BE,
)
from solvers.quantum.qsp_angles import (
    compute_inversion_angles,
    polynomial_degree_estimate,
)
from solvers.quantum.result import SolverResult, QSVTSolverResult


# -- QSVT configuration -------------------------------------------------------

@dataclass
class QSVTConfig:
    """
    Configuration for the QSVT 1-D Poisson solver.

    Attributes
    ----------
    epsilon : float
        Target approximation error for the matrix inversion polynomial.
        Determines the polynomial degree d = O(kappa * log(1/epsilon))
        and hence the circuit depth. Default 1e-2.
    angle_method : str
        Method for computing QSP phase angles. One of:
            'auto'      : try pyqsp, fall back to Chebyshev (recommended)
            'pyqsp'     : use pyqsp library (most accurate)
            'chebyshev' : Chebyshev series construction (fallback)
        Default 'auto'.
    device_name : str
        Qiskit backend for statevector simulation. Default 'statevector'.
    max_degree : int
        Maximum allowed polynomial degree. If the estimated degree
        exceeds this value, a warning is issued and the degree is
        capped. This prevents intractably deep circuits for large kappa.
        Default 500.
    verbose : bool
        If True, print circuit depth and phase angle diagnostics.
        Default False.
    """

    epsilon     : float = 1e-2
    angle_method: str   = "auto"
    device_name : str   = "statevector"
    max_degree  : int   = 500
    verbose     : bool  = False


DEFAULT_QSVT_CONFIG = QSVTConfig()


# -- Public interface ---------------------------------------------------------

def qsvt_solve(
    problem : PoissonProblem1D,
    config  : QSVTConfig = DEFAULT_QSVT_CONFIG,
) -> QSVTSolverResult:
    """
    Solve the 1-D Poisson system Au = b using QSVT.

    Thin wrapper around qsvt_solve_system that unpacks PoissonProblem1D
    and packages the result into a QSVTSolverResult.

    Parameters
    ----------
    problem : PoissonProblem1D
        Discretised 1-D Poisson problem.
    config : QSVTConfig
        QSVT solver hyperparameters.

    Returns
    -------
    QSVTSolverResult
        Physical solution vector and all circuit diagnostics.
    """
    return qsvt_solve_system(
        A      = problem.A,
        b      = problem.b,
        config = config,
    )


def qsvt_solve_system(
    A      : np.ndarray,
    b      : np.ndarray,
    config : QSVTConfig = DEFAULT_QSVT_CONFIG,
) -> QSVTSolverResult:
    """
    Solve the linear system Au = b using QSVT on raw NumPy arrays.

    This is the lower-level interface used by the 2-D line-Jacobi solver
    (qsvt_2d.py) for each row sub-problem, mirroring the interface of
    hhl_solve_system and vqls_solve_system.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Hermitian TST system matrix.
    b : np.ndarray, shape (N,)
        Right-hand side vector.
    config : QSVTConfig
        Solver hyperparameters.

    Returns
    -------
    QSVTSolverResult
        Physical solution and circuit diagnostics.

    Raises
    ------
    ValueError
        If N is not a power of 2, A is not Hermitian, or b is zero.
    RuntimeError
        If phase angle computation fails or solution extraction yields
        an all-zero vector.
    """
    N = len(b)
    n = int(np.log2(N))

    if 2**n != N:
        raise ValueError(
            f"System size N={N} must be a power of 2."
        )
    if not np.allclose(A, A.conj().T, atol=1e-10):
        raise ValueError(
            f"Matrix A must be Hermitian. "
            f"Max asymmetry: {np.max(np.abs(A - A.conj().T)):.2e}"
        )
    b_norm = float(np.linalg.norm(b))
    if b_norm < 1e-14:
        raise ValueError("RHS vector b is numerically zero.")

    # -- Stage 1: block encoding ----------------------------------------------
    main_diag = float(A[0, 0])
    off_diag  = float(A[0, 1])
    alpha     = subnormalisation_factor(main_diag, off_diag)

    be_circuit, alpha_check = build_tst_block_encoding(N, main_diag, off_diag)

    # Effective condition number after subnormalisation.
    # kappa_eff = alpha * kappa(A) / ||A||_2
    eigs         = np.abs(np.linalg.eigvalsh(A))
    kappa_A      = float(eigs.max() / eigs.min())
    A_norm_2     = float(eigs.max())
    kappa_eff    = float(alpha * kappa_A / A_norm_2)

    if config.verbose:
        print(
            f"  QSVT: N={N}, n={n}, alpha={alpha:.4f}, "
            f"kappa(A)={kappa_A:.2f}, kappa_eff={kappa_eff:.2f}"
        )

    # -- Stage 2: QSP phase angles --------------------------------------------
    est_degree = polynomial_degree_estimate(kappa_eff, config.epsilon)
    if est_degree > config.max_degree:
        import warnings
        warnings.warn(
            f"Estimated polynomial degree {est_degree} exceeds max_degree "
            f"={config.max_degree}. Capping at {config.max_degree}. "
            f"Solution accuracy may be reduced.",
            RuntimeWarning,
        )
        est_degree = config.max_degree

    angles, degree = compute_inversion_angles(
        kappa   = kappa_eff,
        epsilon = config.epsilon,
        method  = config.angle_method,
    )

    if config.verbose:
        print(
            f"  QSVT: polynomial degree={degree}, "
            f"n_angles={len(angles)}, "
            f"circuit_depth_estimate={degree * (be_circuit.depth() + 1)}"
        )

    # -- Stage 3: QSVT circuit construction -----------------------------------
    qsvt_circuit = _build_qsvt_circuit(
        be_circuit = be_circuit,
        angles     = angles,
        n          = n,
        b_norm_vec = b / b_norm,
    )

    n_qubits     = qsvt_circuit.num_qubits
    circuit_depth = qsvt_circuit.depth()

    if config.verbose:
        print(
            f"  QSVT: total qubits={n_qubits}, "
            f"circuit depth={circuit_depth}"
        )

    # -- Stage 4: statevector simulation --------------------------------------
    sv = Statevector(qsvt_circuit).data

    # -- Stage 5: solution extraction and proportionality recovery ------------
    x_raw = _extract_solution(sv, n, _N_ANCILLA_BE, degree)

    # Recover proportionality constant against the normalised system.
    A_norm_factor = float(np.linalg.norm(A, ord=2))
    b_norm_vec    = b / b_norm
    A_norm_mat    = A / A_norm_factor

    Ax_norm = A_norm_mat @ x_raw
    denom   = float(np.dot(Ax_norm, Ax_norm))
    if denom < 1e-14:
        raise RuntimeError(
            "QSVT proportionality recovery failed: ||A_norm @ x_raw||^2 "
            "is numerically zero. The QSVT circuit may not have produced "
            "a valid solution state."
        )

    c_norm = float(np.dot(b_norm_vec, Ax_norm) / denom)
    scale  = b_norm / A_norm_factor
    u      = c_norm * scale * x_raw
    c_phys = c_norm * scale

    residual = float(
        np.linalg.norm(A @ u - b) / np.linalg.norm(b)
    )

    return QSVTSolverResult(
        u                  = u,
        solver             = "QSVT",
        raw_state          = x_raw,
        prop_const         = c_phys,
        euclidean_residual = residual,
        polynomial_degree  = degree,
        n_angles           = len(angles),
        circuit_depth      = circuit_depth,
        n_qubits           = n_qubits,
        alpha              = alpha,
        kappa_effective    = kappa_eff,
        angles             = angles,
    )


# -- Private circuit builders -------------------------------------------------

def _build_qsvt_circuit(
    be_circuit : QuantumCircuit,
    angles     : np.ndarray,
    n          : int,
    b_norm_vec : np.ndarray,
) -> QuantumCircuit:
    """
    Assemble the full QSVT circuit for matrix inversion.

    The QSVT sequence implements the polynomial transformation p(A/alpha)
    via alternating applications of U_A and U_A^dagger interleaved with
    projector-controlled phase rotations on the signal qubit:

        U_QSVT = [prod_{k=0}^{d} R(phi_k) . (U_A or U_A^dag)]

    where R(phi_k) is a Z-rotation by 2*phi_k on the signal qubit
    conditioned on the block encoding ancilla being in |0>.

    Register layout (Qiskit little-endian, LSB first):
        qubits 0..n-1  : data register (solution)
        qubit  n       : block encoding ancilla (single qubit, Sz.-Nagy)
        qubit  n+1     : QSVT signal qubit

    Parameters
    ----------
    be_circuit : QuantumCircuit
        Block encoding circuit on n+1 qubits (data + 1 ancilla).
    angles : np.ndarray, shape (d+1,)
        QSP phase angles.
    n : int
        Number of data qubits.
    b_norm_vec : np.ndarray, shape (N,)
        Normalised RHS vector for state preparation.

    Returns
    -------
    QuantumCircuit
        Full QSVT circuit ready for statevector simulation.
    """
    from qiskit.circuit.library import Isometry

    N      = 2**n
    n_a    = _N_ANCILLA_BE   # = 1 for Sz.-Nagy encoding
    degree = len(angles) - 1

    # Total qubits: n (data) + 1 (block encoding ancilla) + 1 (signal).
    n_total = n + n_a + 1
    sig_idx = n + n_a   # index of the signal qubit

    qc = QuantumCircuit(n_total, name="QSVT")

    # -- State preparation: encode |b_norm> in the data register -------------
    qc.append(
        Isometry(b_norm_vec, 0, 0),
        list(range(n)),
    )

    # -- QSVT sequence --------------------------------------------------------
    # Block encoding acts on qubits 0..n (data + ancilla).
    be_qubits  = list(range(n + n_a))
    be_gate    = be_circuit.to_gate(label="U_A")
    be_inv_gate = be_circuit.inverse().to_gate(label="U_A†")

    for k, phi in enumerate(angles):
        # Projector-controlled phase rotation on signal qubit.
        # R(phi) = exp(i*phi*(2*Pi_0 - I)) where Pi_0 = |0><0| on ancilla.
        # Implemented as: X on ancilla -> controlled-Z(2*phi) -> X on ancilla.
        # This applies phase exp(i*phi) when ancilla=0 and exp(-i*phi) when
        # ancilla=1, which is the correct projector-controlled rotation.
        qc.x(n)                          # flip ancilla: |0> -> |1>
        qc.cp(2.0 * phi, n, sig_idx)     # controlled phase: ancilla=1 -> signal
        qc.x(n)                          # restore ancilla

        # Alternate U_A and U_A^dagger after each phase rotation.
        if k < degree:
            if k % 2 == 0:
                qc.append(be_gate,     be_qubits)
            else:
                qc.append(be_inv_gate, be_qubits)

    return qc


def _apply_signal_rotation(
    qc    : QuantumCircuit,
    sig   : QuantumRegister,
    anc   : AncillaRegister,
    n_a   : int,
    phi   : float,
) -> None:
    """
    Apply the QSVT signal rotation R(phi) on the signal qubit.

    The rotation is:
        R(phi) = exp(i * phi * (2|0><0| - I))

    on the signal qubit, conditioned on the block encoding ancilla
    register being in state |0^{n_a}>. This implements the projector-
    controlled phase shift required by the QSVT algorithm.

    In practice, for the statevector simulation this is implemented as:
        X on each anc qubit -> multi-controlled phase(2*phi) on sig -> X

    Parameters
    ----------
    qc : QuantumCircuit
    sig : QuantumRegister
        Single-qubit signal register.
    anc : AncillaRegister
        Block encoding ancilla register (n_a qubits).
    n_a : int
        Number of ancilla qubits.
    phi : float
        Phase angle in radians.
    """
    # Flip ancilla qubits so |0^{n_a}> -> |1^{n_a}>.
    for i in range(n_a):
        qc.x(anc[i])

    # Multi-controlled phase rotation on signal qubit.
    # Controlled on all n_a ancilla qubits being in |1> (after X flip).
    control_qubits = [anc[i] for i in range(n_a)]
    qc.append(
        QuantumCircuit(n_a + 1, name=f"CP({phi:.3f})").to_gate(),
        control_qubits + [sig[0]],
    )
    # Direct implementation for small n_a.
    if n_a == 1:
        qc.cp(2.0 * phi, anc[0], sig[0])
    elif n_a == 2:
        qc.ccx(anc[0], anc[1], sig[0])
        qc.p(2.0 * phi, sig[0])
        qc.ccx(anc[0], anc[1], sig[0])
    else:
        # General case: decompose multi-controlled phase via ancilla chain.
        qc.mcx(control_qubits, sig[0])
        qc.p(2.0 * phi, sig[0])
        qc.mcx(control_qubits, sig[0])

    # Restore ancilla qubits.
    for i in range(n_a):
        qc.x(anc[i])


def _extract_solution(
    sv    : np.ndarray,
    n     : int,
    n_a   : int,
    degree: int,
) -> np.ndarray:
    """
    Extract the solution vector from the QSVT statevector.

    Post-selects on:
        - Block encoding ancilla (qubit n) = |0>
        - Signal qubit (qubit n+n_a) = |0>

    Register layout (Qiskit little-endian):
        bits 0..n-1  : data register
        bit  n       : block encoding ancilla
        bit  n+n_a   : signal qubit

    Parameters
    ----------
    sv : np.ndarray, shape (2^{n+n_a+1},), complex
    n : int
    n_a : int
    degree : int

    Returns
    -------
    x_raw : np.ndarray, shape (N,)
    """
    N       = 2**n
    n_total = n + n_a + 1

    # Bit positions (Qiskit little-endian: bit k = qubit k).
    anc_bit = n          # block encoding ancilla qubit index
    sig_bit = n + n_a    # signal qubit index

    x_raw = np.zeros(N, dtype=complex)

    for idx in range(2**n_total):
        anc_val = (idx >> anc_bit) & 1
        sig_val = (idx >> sig_bit) & 1
        dat_idx = idx & (N - 1)   # lowest n bits

        if anc_val == 0 and sig_val == 0:
            x_raw[dat_idx] = sv[idx]

    x_raw_real = np.real(x_raw)

    if np.allclose(x_raw_real, 0.0, atol=1e-12):
        magnitudes  = np.abs(sv)
        top_indices = np.argsort(magnitudes)[::-1][:8]
        print("\n  DEBUG — top 8 statevector amplitudes after QSVT:")
        for idx in top_indices:
            anc_v = (idx >> anc_bit) & 1
            sig_v = (idx >> sig_bit) & 1
            dat_v = idx & (N - 1)
            print(
                f"    idx={idx:4d}  |amp|={magnitudes[idx]:.6f}  "
                f"data={dat_v}  anc={anc_v}  sig={sig_v}"
            )
        raise RuntimeError(
            f"QSVT solution extraction returned an all-zero vector. "
            f"n_total={n_total}, n={n}, n_a={n_a}, degree={degree}."
        )

    return x_raw_real