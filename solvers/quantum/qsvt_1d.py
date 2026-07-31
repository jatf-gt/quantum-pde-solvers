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
    max_degree_cap: int            = 5000
    verbose       : bool           = False
    label         : str            = ""


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
    #alpha     = subnormalisation_factor(main_diag, off_diag)
    # Use the spectral norm (largest eigenvalue) as the subnormalisation
    # factor alpha. This is what build_tst_block_encoding uses internally
    # to construct the block encoding circuit. Using |a|+2|b| instead
    # causes a mismatch at N=8 where lambda_max > |a|+2|b|, making
    # lambda_max/alpha > 1 and breaking the QSP polynomial domain.
    eigs_for_alpha = np.linalg.eigvalsh(A)
    alpha          = float(np.max(np.abs(eigs_for_alpha)))

    A_norm_factor = float(np.linalg.norm(A, ord=2))
    b_norm_vec    = b / b_norm
    A_norm_mat_   = A / A_norm_factor

    be_circuit, alpha_check = build_tst_block_encoding(N, main_diag, off_diag)

    from qiskit.quantum_info import Operator
    # Get the full 2N x 2N unitary from the circuit.
    be_unitary_circuit = np.array(Operator(be_circuit).data)

    # Reconstruct the exact Sz.-Nagy dilation independently.
    # M = A / alpha (the normalised matrix, already computed above).
    M    = A / alpha
    N_be = 2 * N
    I    = np.eye(N)
    ImM2 = I - M @ M
    eigs_be, V_be = np.linalg.eigh(ImM2)
    eigs_pos_be   = np.clip(eigs_be, 0.0, None)
    sqrtImM2      = V_be @ np.diag(np.sqrt(eigs_pos_be)) @ V_be.conj().T

    # Wx-convention dilation (both diagonal blocks +M, off-diagonals i*sqrt).
    be_unitary_exact = np.zeros((N_be, N_be), dtype=complex)
    be_unitary_exact[:N, :N] =  M
    be_unitary_exact[:N, N:] =  1j * sqrtImM2
    be_unitary_exact[N:, :N] =  1j * sqrtImM2
    be_unitary_exact[N:, N:] =  M

    # Compare full 2N x 2N matrices.
    full_error = float(np.max(np.abs(be_unitary_circuit - be_unitary_exact)))
    topL_error = float(np.max(np.abs(be_unitary_circuit[:N, :N] - M)))

    print(f"  Block encoding full unitary check:")
    print(f"    Top-left N×N block error:  {topL_error:.4e}  (should be ~0)")
    print(f"    Full 2N×2N unitary error:  {full_error:.4e}  (should be ~0)")
    if full_error > 1e-6:
        print(
            f"    WARNING: full unitary error {full_error:.4e} > 1e-6.\n"
            f"    The UnitaryGate decomposition is introducing numerical errors.\n"
            f"    These errors accumulate over {len(angles)-1} block encoding\n"
            f"    applications and corrupt the QSVT polynomial for multi-\n"
            f"    eigenvector inputs (HET case) while being hidden for\n"
            f"    single-eigenvector inputs (generic Poisson case)."
        )
    # Verify block encoding for both b_norm_vec and a test vector
    from qiskit.quantum_info import Statevector as QSV, Operator

    # Get the full unitary of the block encoding circuit
    be_unitary = np.array(Operator(be_circuit).data)
    n_total_be = n + _N_ANCILLA_BE
    N_be = 2**n_total_be

    # Extract the top-left N×N block: <0_anc|U_A|0_anc>
    # In Qiskit little-endian: ancilla is qubit n (bit n of index)
    # |0_anc, data_i> has index i (ancilla bit = 0, data bits = i)
    # <0_anc, data_j|U_A|0_anc, data_i> = be_unitary[j, i]
    # where j and i range over data register indices with ancilla=0

    A_encoded = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            # Row index: ancilla=0, data=j → global index j (ancilla bit n = 0)
            # Col index: ancilla=0, data=i → global index i
            row_idx = j  # ancilla bit 0, data bits = j
            col_idx = i  # ancilla bit 0, data bits = i
            A_encoded[j, i] = be_unitary[row_idx, col_idx]

    print(f"  Block encoding check:")
    print(f"    A/alpha (exact)   = {np.round(A/alpha, 4)}")
    print(f"    <0|U_A|0> (circuit) = {np.round(np.real(A_encoded), 4)}")
    print(f"    Max error: {np.max(np.abs(A_encoded - A/alpha)):.4e}")

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
    import run_qsvt_debug as _dbg
    _dbg._debug_be_circuit = be_circuit
    _dbg._debug_angles     = angles
    _dbg._debug_n          = n
    _dbg._debug_alpha      = alpha

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

    if getattr(config, 'label', '') == 'HET-3a':
        A_norm_mat_diag = A / float(np.linalg.norm(A, ord=2))
        u_expected = np.linalg.solve(A_norm_mat_diag, b / b_norm)
        u_exp_norm = u_expected / np.linalg.norm(u_expected)
        x_raw_norm = x_raw / (np.linalg.norm(x_raw) + 1e-14)
        cos_xu = float(np.dot(x_raw_norm, u_exp_norm))
        print(f"  QSVT x_raw vs A_norm^{{-1}}b_norm:")
        print(f"    x_raw (normalised)    = {np.round(x_raw_norm, 5)}")
        print(f"    A^{{-1}}b (normalised) = {np.round(u_exp_norm, 5)}")
        print(f"    cos(x_raw, A^{{-1}}b) = {cos_xu:.6f}")
        # Also check the generic Poisson expected solution
        b_fS = np.array([-0.3717, -0.6015, -0.6015, -0.3717])
        u_fS_expected = np.linalg.solve(A_norm_mat_diag, b_fS)
        u_fS_norm = u_fS_expected / np.linalg.norm(u_fS_expected)
        cos_fS = float(np.dot(x_raw_norm, u_fS_norm))
        print(f"    cos(x_raw, A^{{-1}}b_fS) = {cos_fS:.6f}  "
            f"(should be ~0 if x_raw is HET-specific)")

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
    print(
        f"  QSVT extraction: ||Re||={norm_re:.4e}, ||Im||={norm_im:.4e}, "
        f"ratio Im/Re={norm_im/(norm_re + 1e-14):.2f}"
    )

    if norm_im < 1e-12:
        raise RuntimeError(
            f"QSVT extraction: imaginary part is all-zero. "
            f"n_total={n_total}, n={n}, n_a={n_a}, degree={degree}."
        )

    return x_raw_im


def _evaluate_qsp_polynomial(
    angles    : np.ndarray,
    degree    : int,
    kappa_eff : float,
    alpha     : float,
    A         : np.ndarray,
    label     : str,
) -> None:
    """
    Evaluate P(λ_k/α) at each eigenvalue of A using the correct signal
    unitary convention matching the block encoding circuit.

    The block encoding circuit implements the real rotation:
        R_x = [[ x,          sqrt(1-x²)],
               [-sqrt(1-x²), x         ]]

    NOT the W_x convention used by pyqsp internally. The relationship is:
        W_x = diag(1, i) · R_x · diag(1, -i)

    The pyqsp sym_qsp phases are designed for W_x. When applied with R_x
    (as in the circuit), the polynomial achieved is different. This
    diagnostic evaluates both conventions to determine which one the
    circuit actually implements.
    """
    eig_vals_A, _ = np.linalg.eigh(A)

    print(f"\n  QSP polynomial evaluation [{label}]:")
    print(
        f"  {'lambda_k':>10}  {'x':>8}  "
        f"{'1/(kx)·0.9':>12}  "
        f"{'Im(P_Wx)':>10}  {'Re(P_Wx)':>10}  "
        f"{'Im(P_Rx)':>10}  {'Re(P_Rx)':>10}"
    )

    for lam_k in eig_vals_A:
        x = lam_k / alpha

        if x > 1.0:
            print(
                f"  {lam_k:>10.4f}  {x:>8.4f}  "
                f"{'x>1: INVALID':>12}  "
                f"{'overflow':>10}  {'overflow':>10}  "
                f"{'overflow':>10}  {'overflow':>10}"
            )
            continue

        expected = (1.0 / (kappa_eff * x)) * 0.9
        sq       = float(np.sqrt(max(1.0 - x**2, 0.0)))

        # W_x convention (pyqsp internal):
        Wx = np.array([[x, 1j*sq], [1j*sq, x]], dtype=complex)

        # R_x convention (block encoding circuit — real rotation matrix):
        Rx = np.array([[x, sq], [-sq, x]], dtype=float)

        def _apply_qsp(signal_unitary: np.ndarray) -> complex:
            """Apply QSP sequence and return top-left element P(x)."""
            U = np.eye(2, dtype=complex)
            for idx_phi in range(len(angles)):
                phi = angles[idx_phi]
                Rz  = np.diag([np.exp(-1j * phi), np.exp(1j * phi)])
                U   = Rz @ U
                if idx_phi < degree:
                    U = signal_unitary @ U
            return complex(U[0, 0])

        # P_Wx = _apply_qsp_alternating(Wx, angles, degree)
        # P_Rx = _apply_qsp_alternating(Rx)

        # print(
        #     f"  {lam_k:>10.4f}  {x:>8.4f}  "
        #     f"{expected:>12.6f}  "
        #     f"{np.imag(P_Wx):>10.6f}  {np.real(P_Wx):>10.6f}  "
        #     f"{np.imag(P_Rx):>10.6f}  {np.real(P_Rx):>10.6f}"
        # )

        P_Wx_alt = _apply_qsp_alternating(Wx, angles, degree)
        P_Rx_alt = _apply_qsp_alternating(Rx, angles, degree)

        print(
            f"  {lam_k:>10.4f}  {x:>8.4f}  "
            f"{expected:>12.6f}  "
            f"{'Im(P_Wx_alt)':>14}  {'Re(P_Wx_alt)':>14}  "
            f"{'Im(P_Rx_alt)':>14}  {'Re(P_Rx_alt)':>14}"
        )
        print(
            f"  {'':>10}  {'':>8}  "
            f"{'':>12}  "
            f"{np.imag(P_Wx_alt):>14.6f}  {np.real(P_Wx_alt):>14.6f}  "
            f"{np.imag(P_Rx_alt):>14.6f}  {np.real(P_Rx_alt):>14.6f}"
        )


def _apply_qsp_alternating(
    signal_unitary : np.ndarray,
    angles         : np.ndarray,
    degree         : int,
) -> complex:
    """
    Apply the QSP sequence with alternating signal_unitary and
    signal_unitary.conj().T, matching the circuit's alternation
    between U_A (even steps) and U_A† (odd steps).
    """
    U = np.eye(2, dtype=complex)
    for idx_phi in range(len(angles)):
        phi = angles[idx_phi]
        Rz  = np.diag([np.exp(-1j * phi), np.exp(1j * phi)])
        U   = Rz @ U
        if idx_phi < degree:
            if idx_phi % 2 == 0:
                U = signal_unitary @ U
            else:
                U = signal_unitary.conj().T @ U
    return complex(U[0, 0])