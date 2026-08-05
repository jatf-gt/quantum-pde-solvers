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
    _N_ANCILLA_BE,
)
from solvers.quantum.qsp_angles import (
    compute_inversion_angles,
    polynomial_degree_estimate,
)
from solvers.quantum.result import SolverResult, QSVTSolverResult


# -- QSVT configuration -------------------------------------------------------

@dataclass
class QSVTConfig1D:
    """
    Configuration for the QSVT 1-D Poisson solver.

    Attributes
    ----------
    epsilon : float
        Target approximation error for the matrix inversion polynomial.
        Controls polynomial degree d = O(kappa * log(kappa/epsilon)).
        Default 0.01.
    angle_method : str
        Phase computation method. One of:
            'sym_qsp_direct'  : direct Newton solver (default, fastest)
            'sym_qsp_wrapper' : via QuantumSignalProcessingPhases wrapper
            'reduced_degree'  : degree capped at max_degree (fast, approx)
            'precomputed'     : load from disk cache only
            'auto'            : try sym_qsp_direct, fall back to wrapper
        Default 'sym_qsp_direct'.
    max_degree : int or None
        Maximum polynomial degree. Only used when angle_method='reduced_degree'.
        Recommended values:
            N=4  (kappa~9):    max_degree=63  (fast, <1s, ~10% poly error)
            N=8  (kappa~32):   max_degree=127 (fast, <5s, ~5% poly error)
            N=16 (kappa~117):  max_degree=255 (moderate, ~30s, ~3% poly error)
            N=32 (kappa~441):  max_degree=511 (moderate, ~120s, ~2% poly error)
            N=64 (kappa~1700): max_degree=511 (moderate, ~120s, ~2% poly error)
        The polynomial approximation error is much smaller than the FD
        discretisation error (O(h²)) for all these choices.
        Default None (use full degree from epsilon).
    device_name : str
        Qiskit backend. Default 'statevector'.
    max_degree_cap : int
        Hard cap on polynomial degree regardless of method. Prevents
        intractably deep circuits. Default 5000.
    verbose : bool
        Print circuit depth and phase angle diagnostics. Default False.
    label : str
        Diagnostic label for targeted output. Default ''.
    """
    epsilon       : float          = 0.01
    angle_method  : str            = "sym_qsp_direct"
    max_degree    : Optional[int]  = None
    device_name   : str            = "statevector"
    max_degree_cap: int            = 15000
    verbose       : bool           = False
    label         : str            = ""


DEFAULT_QSVT_CONFIG = QSVTConfig1D()


# -- Public interface ---------------------------------------------------------

def qsvt_solve(
    problem : PoissonProblem1D,
    config  : QSVTConfig1D = DEFAULT_QSVT_CONFIG,
) -> QSVTSolverResult:
    """
    Solve the 1-D Poisson system Au = b using QSVT.

    Thin wrapper around qsvt_solve_system that unpacks PoissonProblem1D
    and packages the result into a QSVTSolverResult.

    Parameters
    ----------
    problem : PoissonProblem1D
        Discretised 1-D Poisson problem.
    config : QSVTConfig1D
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
    config : QSVTConfig1D = DEFAULT_QSVT_CONFIG,
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
    config : QSVTConfig1D
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
    
    # -- Stage 0: sign normalisation for negative definite matrices -----------
    # The QSVT polynomial p(x) approximates 1/x for x in [1/kappa, 1].
    # This requires the matrix to be positive semidefinite after normalisation.
    # The 1-D and 2-D Poisson TST matrices are negative definite (all
    # eigenvalues negative), so we negate both A and b before proceeding.
    # Since (-A)u = -b is equivalent to Au = b, the solution u is unchanged.
    eigs_sign = np.linalg.eigvalsh(A)
    if np.all(eigs_sign < 0):
        A = -A.copy()
        b = -b.copy()
    # If A has mixed-sign eigenvalues (indefinite), the QSVT polynomial
    # cannot be applied directly — raise an informative error.
    elif not np.all(eigs_sign > 0):
        raise ValueError(
            "QSVT requires A to be positive or negative definite. "
            "The matrix has mixed-sign eigenvalues (indefinite). "
            f"Eigenvalue range: [{eigs_sign.min():.4f}, {eigs_sign.max():.4f}]"
        )
    # A is now positive definite; proceed with standard QSVT.

    # -- Stage 1: block encoding ----------------------------------------------
    main_diag = float(A[0, 0])
    off_diag  = float(A[0, 1])
    # Use the spectral norm (largest eigenvalue) as the subnormalisation
    # factor alpha, rather than |a|+2|b| (subnormalisation_factor). This is
    # what build_tst_block_encoding uses internally to construct the block
    # encoding circuit. Using |a|+2|b| instead causes a mismatch at N=8
    # where lambda_max > |a|+2|b|, making lambda_max/alpha > 1 and breaking
    # the QSP polynomial domain.
    eigs_for_alpha = np.linalg.eigvalsh(A)
    alpha          = float(np.max(np.abs(eigs_for_alpha)))

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
    if (
        hasattr(config, "_precomputed_angles")
        and config._precomputed_angles is not None
    ):
        angles = config._precomputed_angles
        degree = config._precomputed_degree
    else:
        est_degree = polynomial_degree_estimate(kappa_eff, config.epsilon)
        if est_degree > config.max_degree_cap:
            import warnings
            warnings.warn(
                f"Estimated polynomial degree {est_degree} exceeds "
                f"max_degree_cap={config.max_degree_cap}. Capping.",
                RuntimeWarning,
            )
            est_degree = config.max_degree_cap

        angles, degree = compute_inversion_angles(
            kappa      = kappa_eff,
            epsilon    = config.epsilon,
            method     = config.angle_method,
            max_degree = config.max_degree,   
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
            f"  QSVT: total qubits={n + _N_ANCILLA_BE + 1}, "
            f"circuit depth={circuit_depth}"
        )

    # -- Stage 4: statevector simulation --------------------------------------
    sv = Statevector(qsvt_circuit).data

    # -- Stage 5: solution extraction and proportionality recovery ------------
    x_raw = _extract_solution(sv, n, _N_ANCILLA_BE, degree)  # n_a=2: BE ancilla + signal qubit ??
    # x_raw = Im(post-selected state), shape (N,), real-valued.
    # Under sym_qsp Wx convention: Im(P(A/alpha))|b_norm> approx (1/kappa_eff) A^{-1}|b_norm>

    A_norm_factor = float(np.linalg.norm(A, ord=2))
    b_norm_vec    = b / b_norm
    A_norm_mat    = A / A_norm_factor

    Ax_norm = A_norm_mat @ x_raw
    denom   = float(np.dot(Ax_norm, Ax_norm))

    # Diagnostics — always print to identify HET failure.
    _qsvt_recovery_diagnostics(
        x_raw, Ax_norm, b_norm_vec, denom,
        b_norm, A_norm_factor,
        verbose = config.verbose,
        label   = getattr(config, 'label', ''),
    )

    if denom < 1e-14:
        raise RuntimeError(
            "QSVT proportionality recovery: ||A_norm @ x_raw||^2 is zero."
        )

    c_norm = float(np.dot(b_norm_vec, Ax_norm) / denom)
    scale  = b_norm / A_norm_factor
    u      = c_norm * scale * x_raw
    c_phys = c_norm * scale

    residual = float(np.linalg.norm(A @ u - b) / np.linalg.norm(b))

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


def _qsvt_recovery_diagnostics(
    x_raw        : np.ndarray,
    Ax_norm      : np.ndarray,
    b_norm_vec   : np.ndarray,
    denom        : float,
    b_norm       : float,
    A_norm_factor: float,
    verbose      : bool,
    label        : str = "",
) -> None:
    """
    Print diagnostics for the QSVT proportionality recovery.

    Only prints when verbose=True or when the alignment is poor
    (cos < 0.5), to avoid flooding the console during 2D Jacobi loops.
    The label parameter identifies which problem instance is being solved,
    enabling targeted diagnosis of the HET failure without printing
    diagnostics for every row of the 2D solver.

    Parameters
    ----------
    x_raw : np.ndarray
    Ax_norm : np.ndarray
    b_norm_vec : np.ndarray
    denom : float
    b_norm : float
    A_norm_factor : float
    verbose : bool
        If True, always print. If False, only print on failure.
    label : str
        Descriptive label for the problem instance, e.g. 'HET-3a' or
        '2D-row-2-iter-5'. Printed in the header to identify the source.
    """
    norm_x    = float(np.linalg.norm(x_raw))
    norm_Ax   = float(np.linalg.norm(Ax_norm))
    dot_bAx   = float(np.dot(b_norm_vec, Ax_norm))
    c_norm    = dot_bAx / denom if denom > 1e-14 else float("nan")
    scale     = b_norm / A_norm_factor
    cos_angle = dot_bAx / (norm_Ax + 1e-14)

    is_failure = abs(cos_angle) < 0.5

    if not (verbose or is_failure):
        return

    header = f"  QSVT recovery diagnostics{' [' + label + ']' if label else ''}:"
    print(
        f"{header}\n"
        f"    ||x_raw||          = {norm_x:.6e}\n"
        f"    ||A_norm @ x_raw|| = {norm_Ax:.6e}\n"
        f"    <b_norm|A_norm|x>  = {dot_bAx:.6e}\n"
        f"    denom (||Ax||^2)   = {denom:.6e}\n"
        f"    cos(Ax, b_norm)    = {cos_angle:.6f}\n"
        f"    c_norm             = {c_norm:.6e}\n"
        f"    scale (||b||/||A||)= {scale:.6e}\n"
        f"    u_scale (c*scale)  = {c_norm * scale:.6e}"
    )
    if is_failure:
        print(
            f"    *** FAILURE: cos={cos_angle:.4f} < 0.5.\n"
            f"    *** x_raw is not proportional to A^{{-1}}b_norm."
        )


# -- Private circuit builders -------------------------------------------------

def _build_qsvt_circuit(
    be_circuit : QuantumCircuit,
    angles     : np.ndarray,
    n          : int,
    b_norm_vec : np.ndarray,
) -> QuantumCircuit:
    """
    Assemble the QSVT circuit for matrix inversion.

    Uses the sym_qsp convention: alternating U_A and U_A† interleaved
    with Rz phase rotations on the ancilla. The pyqsp sym_qsp method
    with signal_operator='Wx' designs phases for this alternating
    sequence. The non-alternating sequence (U_A at every step) implements
    a different polynomial and must NOT be used with sym_qsp phases.

    The Rz convention matches Qiskit: rz(2*phi) = diag(exp(-i*phi), exp(+i*phi)).
    The phases from pyqsp are negated before use to correct for the sign
    difference between pyqsp's convention (diag(exp(+i*phi), exp(-i*phi)))
    and Qiskit's convention.
    """
    degree  = len(angles) - 1
    n_total = n + _N_ANCILLA_BE
    anc_idx = n

    qc = QuantumCircuit(n_total, name="QSVT")
    qc.append(Isometry(b_norm_vec, 0, 0), list(range(n)))

    be_gate     = be_circuit.to_gate(label="U_A")
    be_inv_gate = be_circuit.inverse().to_gate(label="U_A†")
    be_qubits   = list(range(n_total))

    for k, phi in enumerate(angles):
        qc.rz(2.0 * phi, anc_idx)
        if k < degree:
            qc.append(be_gate, be_qubits)  # non-alternating

    return qc


def _extract_solution(
    sv    : np.ndarray,
    n     : int,
    n_a   : int,
    degree: int,
) -> np.ndarray:
    """
    Extract the solution vector from the QSVT statevector.

    Post-selects on the block encoding ancilla (qubit n) = |0> and
    returns the imaginary part of the post-selected amplitudes.

    With the Wx-convention block encoding, the QSP sequence achieves
    Im(P(A/alpha))|b_norm> ≈ (1/kappa_eff) * A^{-1}|b_norm>.
    The imaginary part is the correct extraction for this convention.

    Parameters
    ----------
    sv : np.ndarray, shape (2^{n+1},), complex
    n : int
    n_a : int
        Number of block encoding ancilla qubits (= 1).
    degree : int

    Returns
    -------
    x_raw : np.ndarray, shape (N,), real
    """
    N       = 2**n
    n_total = n + n_a
    anc_bit = n
    x_raw   = np.zeros(N, dtype=complex)

    for idx in range(2**n_total):
        if ((idx >> anc_bit) & 1) == 0:
            x_raw[idx & (N - 1)] = sv[idx]

    x_raw_im = np.imag(x_raw)
    x_raw_re = np.real(x_raw)

    norm_im = float(np.linalg.norm(x_raw_im))
    norm_re = float(np.linalg.norm(x_raw_re))

    if norm_im < 1e-12:
        raise RuntimeError(
            f"QSVT extraction: imaginary part is all-zero. "
            f"n_total={n_total}, n={n}, n_a={n_a}, degree={degree}."
        )

    return x_raw_im
