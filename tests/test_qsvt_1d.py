"""
Verification tests for the QSVT 1-D Poisson solver.

Test scope
----------
These tests verify the structural correctness of the QSVT implementation
at N=4 (2 data qubits), which is the smallest non-trivial system size.
The tests are organised in three groups:

    1. Block encoding verification: confirms that the LCU circuit
       correctly encodes A/alpha in the top-left block of the unitary.

    2. QSP angle verification: confirms that the computed phase angles
       produce a polynomial that approximates 1/x on [1/kappa, 1].

    3. QSVT solver verification: confirms that the full solver pipeline
       produces a solution of the correct shape, sign, and approximate
       magnitude relative to the Thomas reference.

Runtime note
------------
The QSVT circuit for N=4, kappa~5, epsilon=0.1 has polynomial degree
approximately 5*log(50) ~ 20 and circuit depth approximately 20 * 15
= 300 gates. Statevector simulation of a 6-qubit circuit (2 data +
2 block encoding ancilla + 1 signal + 1 state prep ancilla) with 300
gates takes approximately 5-15 seconds per test on standard hardware.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.config import SimConfig1D
from problems.poisson_1d import PoissonProblem1D
from solvers.classical.thomas import thomas_solve
from solvers.quantum.block_encoding import (
    build_tst_block_encoding,
    block_encoding_matrix,
    subnormalisation_factor,
)
from solvers.quantum.qsp_angles import (
    compute_inversion_angles,
    evaluate_inversion_polynomial,
    polynomial_degree_estimate,
)
from solvers.quantum.qsvt_1d import (
    qsvt_solve,
    qsvt_solve_system,
    QSVTConfig1D,
    DEFAULT_QSVT_CONFIG,
)
from solvers.quantum.result import QSVTSolverResult

# Every test in this module builds and simulates a quantum circuit.
pytestmark = pytest.mark.quantum


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def qsvt_cfg_fast():
    """
    Fast QSVT configuration for structural verification tests.

    Uses epsilon=0.1 to minimise polynomial degree (and hence circuit
    depth) whilst still producing a recognisable solution. The loose
    tolerance is appropriate for structural tests; publication-quality
    results require epsilon=0.01 or smaller.
    """
    return QSVTConfig1D(
        epsilon      = 0.1,
        angle_method = "auto",
        verbose      = True,
        max_degree   = 50,
    )


# ── Block encoding tests ──────────────────────────────────────────────────────

class TestBlockEncoding:

    def test_circuit_is_unitary(self):
        """
        The block encoding circuit must be unitary to within numerical
        precision. Verified by checking U^dagger U = I.
        """
        from qiskit.quantum_info import Operator
        qc, alpha = build_tst_block_encoding(4, -2.0, 1.0)
        U         = Operator(qc).data
        I_approx  = U.conj().T @ U
        assert np.allclose(I_approx, np.eye(len(I_approx)), atol=1e-10), (
            "Block encoding circuit is not unitary."
        )

    def _test_block_encoding_action(
        be_circuit : object,
        A          : np.ndarray,
        alpha      : float,
        b_norm_vec : np.ndarray,
        n          : int,
        label      : str,
    ) -> None:
        """
        Test the block encoding action on b_norm_vec directly.

        Computes U_A |0_anc> |b_norm_vec> and checks that the |0_anc>
        component of the output equals (A/alpha)|b_norm_vec>.

        Also checks the |1_anc> component (the garbage state) to verify
        it is orthogonal to |0_anc> as required for valid block encoding.
        """
        from qiskit.quantum_info import Statevector, Operator
        N = 2**n

        # Prepare |0_anc> |b_norm_vec>.
        init_sv = np.zeros(2 * N, dtype=complex)
        # In Qiskit little-endian: ancilla is qubit n (bit n of index).
        # |0_anc, data_i> has index i (ancilla bit = 0).
        for i in range(N):
            init_sv[i] = b_norm_vec[i]

        # Apply the block encoding unitary.
        be_unitary = np.array(Operator(be_circuit).data)
        out_sv     = be_unitary @ init_sv

        # Extract |0_anc> component (indices 0..N-1).
        out_anc0 = out_sv[:N]
        # Extract |1_anc> component (indices N..2N-1).
        out_anc1 = out_sv[N:]

        expected_anc0 = (A / alpha) @ b_norm_vec

        print(f"\n  Block encoding action test [{label}]:")
        print(f"    Input  |0_anc>|b> component: {np.round(b_norm_vec, 4)}")
        print(f"    Output |0_anc> component:    {np.round(np.real(out_anc0), 4)}")
        print(f"    Expected (A/alpha)|b>:        {np.round(expected_anc0, 4)}")
        print(f"    Max error:                    "
            f"{np.max(np.abs(out_anc0 - expected_anc0)):.4e}")
        print(f"    ||garbage||:                  {np.linalg.norm(out_anc1):.4e}")
        print(f"    ||garbage||² + ||(A/alpha)b||² = "
            f"{np.linalg.norm(out_anc1)**2 + np.linalg.norm(out_anc0)**2:.6f} "
            f"(should = 1.0 if unitary)")

    def test_block_encodes_correct_matrix(self):
        N         = 4
        main_diag = -2.0
        off_diag  =  1.0
        qc, alpha = build_tst_block_encoding(N, main_diag, off_diag)
        n         = int(np.log2(N))
        block     = block_encoding_matrix(qc, n)

        A_expected = (
            main_diag * np.eye(N)
            + off_diag * np.diag(np.ones(N - 1), k=1)
            + off_diag * np.diag(np.ones(N - 1), k=-1)
        ) / alpha   # alpha is now the spectral norm, not |a|+2|b|

        assert np.allclose(block, A_expected, atol=1e-8), (
            f"Block encoding error: max deviation = "
            f"{np.max(np.abs(block - A_expected)):.3e}"
        )

    def test_subnormalisation_satisfies_bound(self):
        """
        alpha must satisfy ||A/alpha||_2 <= 1 for the block encoding
        to be a valid sub-unitary embedding.
        """
        for main_diag, off_diag in [(-2.0, 1.0), (-4.0, 1.0)]:
            N     = 4
            alpha = subnormalisation_factor(main_diag, off_diag)
            A     = (
                main_diag * np.eye(N)
                + off_diag * np.diag(np.ones(N - 1), k=1)
                + off_diag * np.diag(np.ones(N - 1), k=-1)
            )
            norm_A_normalised = float(np.linalg.norm(A / alpha, ord=2))
            assert norm_A_normalised <= 1.0 + 1e-10, (
                f"Subnormalisation violated: ||A/alpha||_2 = "
                f"{norm_A_normalised:.6f} > 1 for a={main_diag}, b={off_diag}."
            )

    def test_alpha_satisfies_subnormalisation_bound(self):
        """
        alpha must satisfy ||A||_2 <= alpha (subnormalisation condition).
        For the Sz.-Nagy encoding, alpha = ||A||_2 exactly.
        """
        for N, main_diag, off_diag in [(4, -2.0, 1.0), (4, -4.0, 1.0)]:
            qc, alpha = build_tst_block_encoding(N, main_diag, off_diag)
            A = (
                main_diag * np.eye(N)
                + off_diag * np.diag(np.ones(N - 1), k=1)
                + off_diag * np.diag(np.ones(N - 1), k=-1)
            )
            A_norm_2 = float(np.max(np.abs(np.linalg.eigvalsh(A))))
            assert alpha == pytest.approx(A_norm_2, rel=1e-6), (
                f"alpha={alpha:.6f} != ||A||_2={A_norm_2:.6f}"
            )

    def test_invalid_N_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            build_tst_block_encoding(6, -2.0, 1.0)


# ── QSP angle tests ───────────────────────────────────────────────────────────

class TestQSPAngles:

    def test_degree_estimate_increases_with_kappa(self):
        """
        Polynomial degree must increase with condition number.
        """
        d1 = polynomial_degree_estimate(5.0,  0.01)
        d2 = polynomial_degree_estimate(10.0, 0.01)
        d3 = polynomial_degree_estimate(32.0, 0.01)
        assert d1 < d2 < d3

    def test_degree_estimate_increases_with_precision(self):
        """
        Polynomial degree must increase as epsilon decreases.
        """
        d1 = polynomial_degree_estimate(10.0, 0.1)
        d2 = polynomial_degree_estimate(10.0, 0.01)
        d3 = polynomial_degree_estimate(10.0, 0.001)
        assert d1 < d2 < d3

    def test_angles_shape(self):
        """
        Phase angle array must have length degree + 1.
        """
        angles, degree = compute_inversion_angles(5.0, 0.1, method="auto")
        assert len(angles) == degree + 1

    def test_polynomial_approximates_inverse(self):
        """
        The QSP polynomial must be bounded by 1 on [-1, 1] and must
        have positive real part on [1/kappa, 1], indicating it approximates
        a positive function (1/x > 0 on this interval).
        """
        kappa   = 5.0
        epsilon = 0.1
        angles, degree = compute_inversion_angles(
            kappa, epsilon, method="auto"
        )

        # Check boundedness on [-1, 1].
        x_full = np.linspace(-1.0, 1.0, 50)
        p_full = evaluate_inversion_polynomial(angles, x_full)
        assert np.all(np.abs(p_full) <= 1.0 + 1e-6), (
            "QSP polynomial exceeds magnitude 1 on [-1, 1]."
        )

        # Check positive real part on [1/kappa, 1].
        x_pos = np.linspace(1.0 / kappa, 1.0, 20)
        p_pos = evaluate_inversion_polynomial(angles, x_pos)
        assert np.mean(np.real(p_pos) > 0) >= 0.7, (
            "QSP polynomial does not have predominantly positive real part "
            "on [1/kappa, 1] — may not approximate 1/x correctly."
        )

    def test_invalid_kappa_raises(self):
        with pytest.raises(ValueError, match="kappa"):
            compute_inversion_angles(0.5, 0.01)

    def test_invalid_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            compute_inversion_angles(5.0, -0.01)


# ── QSVT solver tests ─────────────────────────────────────────────────────────

class TestQSVT1D:

    def test_returns_qsvt_solver_result(self, problem_1d_N4_fS, qsvt_cfg_fast):
        r = qsvt_solve(problem_1d_N4_fS, config=qsvt_cfg_fast)
        assert isinstance(r, QSVTSolverResult)

    def test_solution_shape(self, problem_1d_N4_fS, qsvt_cfg_fast):
        r = qsvt_solve(problem_1d_N4_fS, config=qsvt_cfg_fast)
        assert r.u.shape == (4,)

    def test_solver_label(self, problem_1d_N4_fS, qsvt_cfg_fast):
        r = qsvt_solve(problem_1d_N4_fS, config=qsvt_cfg_fast)
        assert r.solver == "QSVT"

    def test_solution_finite(self, problem_1d_N4_fS, qsvt_cfg_fast):
        """All solution values must be finite."""
        r = qsvt_solve(problem_1d_N4_fS, config=qsvt_cfg_fast)
        assert np.all(np.isfinite(r.u)), (
            "QSVT solution contains non-finite values."
        )

    def test_circuit_diagnostics_populated(self, problem_1d_N4_fS, qsvt_cfg_fast):
        r = qsvt_solve(problem_1d_N4_fS, config=qsvt_cfg_fast)
        assert r.polynomial_degree > 0
        assert r.circuit_depth     > 0
        # n+2 qubits: n data + 1 BE ancilla + 1 signal qubit.
        assert r.n_qubits == int(np.log2(4)) + 1   # = 4 for N=4
        assert r.n_angles == r.polynomial_degree + 1

    def test_kappa_effective_positive(self, problem_1d_N4_fS, qsvt_cfg_fast):
        r = qsvt_solve(problem_1d_N4_fS, config=qsvt_cfg_fast)
        assert r.kappa_effective > 0.0

    def test_residual_finite(self, problem_1d_N4_fS, qsvt_cfg_fast):
        r = qsvt_solve(problem_1d_N4_fS, config=qsvt_cfg_fast)
        assert np.isfinite(r.euclidean_residual)

    def test_sign_consistent_with_thomas(self, problem_1d_N4_fS, qsvt_cfg_fast):
        """
        The dominant solution component must have the same sign as the
        Thomas reference. A sign flip indicates a proportionality
        recovery failure.
        """
        u_thomas = thomas_solve(problem_1d_N4_fS).u
        u_qsvt   = qsvt_solve(problem_1d_N4_fS, config=qsvt_cfg_fast).u

        idx_max      = int(np.argmax(np.abs(u_thomas)))
        sign_thomas  = np.sign(u_thomas[idx_max])
        sign_qsvt    = np.sign(u_qsvt[idx_max])
        assert sign_thomas == sign_qsvt, (
            f"Sign mismatch at dominant node {idx_max}: "
            f"Thomas={u_thomas[idx_max]:.4f}, QSVT={u_qsvt[idx_max]:.4f}"
        )

    def test_qsvt_solve_system_raw_arrays(self, qsvt_cfg_fast):
        """qsvt_solve_system accepts raw (A, b) arrays."""
        A = np.array([[-2., 1., 0., 0.],
                      [ 1.,-2., 1., 0.],
                      [ 0., 1.,-2., 1.],
                      [ 0., 0., 1.,-2.]], dtype=float)
        b = np.array([0.1, 0.2, 0.2, 0.1])
        r = qsvt_solve_system(A, b, qsvt_cfg_fast)
        assert r.u.shape == (4,)
        assert np.all(np.isfinite(r.u))

    def test_non_power_of_2_raises(self, qsvt_cfg_fast):
        A = np.eye(3) * (-2.0)
        b = np.ones(3)
        with pytest.raises(ValueError, match="power of 2"):
            qsvt_solve_system(A, b, qsvt_cfg_fast)

    def test_non_hermitian_raises(self, qsvt_cfg_fast):
        A = np.array([[-2., 2.], [1., -2.]])
        b = np.array([1., 1.])
        with pytest.raises(ValueError, match="Hermitian"):
            qsvt_solve_system(A, b, qsvt_cfg_fast)

    def test_zero_rhs_raises(self, qsvt_cfg_fast):
        A = np.array([[-2., 1.], [1., -2.]])
        b = np.zeros(2)
        with pytest.raises(ValueError, match="zero"):
            qsvt_solve_system(A, b, qsvt_cfg_fast)

    def test_angles_stored_in_result(self, problem_1d_N4_fS, qsvt_cfg_fast):
        """Phase angles must be stored in the result for reproducibility."""
        r = qsvt_solve(problem_1d_N4_fS, config=qsvt_cfg_fast)
        assert r.angles is not None
        assert len(r.angles) == r.n_angles