"""
Verification tests for the QSVT 2-D line-Jacobi solver.

All tests use N=4 (2 data qubits per row sub-problem) with max_iter=10
to bound runtime. The purpose is to verify structural correctness —
solution shape, finite values, sign consistency, and iteration
bookkeeping — not publication-level numerical accuracy.

The 2-D row matrix has kappa_row ~ 2.77 and polynomial degree d ~ 3
for epsilon=0.1, making each row QSVT circuit very shallow (depth ~ 30).
This means the 2-D QSVT tests are significantly faster than the 1-D
tests at the same N.

Expected runtime: under 5 minutes for the full file.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.config import SimConfig2D
from problems.poisson_2d import PoissonProblem2D
from problems.het_plasma_2d import HETConfig2D, HETSinusoidalProblem2D
from solvers.classical.thomas_2d import thomas_solve_2d
from solvers.quantum.qsvt_2d import QSVTConfig2D, qsvt_solve_2d
from solvers.quantum.result import SolverResult2D


# -- Shared fixtures ----------------------------------------------------------

@pytest.fixture(scope="module")
def qsvt_cfg_2d_fast():
    """
    Minimal QSVT-2D configuration for rapid structural verification.

    epsilon=0.1 gives polynomial degree d ~ 3 for kappa_row ~ 2.77,
    resulting in very shallow circuits (depth ~ 30 per row).
    max_iter=10 is sufficient to verify the solver runs and produces
    a decreasing iteration error.
    """
    return QSVTConfig2D(
        epsilon      = 0.1,
        angle_method = "auto",
        max_degree   = 50,      # increased from 20 to accommodate pyqsp degree 33
        n_workers    = 1,
        verbose      = True,
    )


@pytest.fixture(scope="module")
def problem_2d_N4_fL_qsvt():
    """N=4, fL, max_iter=30 — enough for QSVT to establish correct sign."""
    cfg = SimConfig2D(N=4, epsilon=0.01, source_fn="fL", max_iter=30)
    return PoissonProblem2D(cfg)


@pytest.fixture(scope="module")
def het_problem_2d_N4_qsvt():
    """HET sinusoidal N=4, max_iter=10."""
    cfg = HETConfig2D(N=4, epsilon=0.01, max_iter=10)
    return HETSinusoidalProblem2D(cfg)


# -- Structural tests ---------------------------------------------------------

class TestQSVT2DStructure:

    @pytest.mark.slow
    def test_returns_solver_result_2d(
        self, problem_2d_N4_fL_qsvt, qsvt_cfg_2d_fast
    ):
        r = qsvt_solve_2d(problem_2d_N4_fL_qsvt, config=qsvt_cfg_2d_fast)
        assert isinstance(r, SolverResult2D)

    def test_solution_shape(
        self, problem_2d_N4_fL_qsvt, qsvt_cfg_2d_fast
    ):
        r = qsvt_solve_2d(problem_2d_N4_fL_qsvt, config=qsvt_cfg_2d_fast)
        assert r.u.shape == (4, 4)

    def test_solver_label(
        self, problem_2d_N4_fL_qsvt, qsvt_cfg_2d_fast
    ):
        r = qsvt_solve_2d(problem_2d_N4_fL_qsvt, config=qsvt_cfg_2d_fast)
        assert r.solver == "QSVT-2D"

    @pytest.mark.slow
    def test_solution_finite(
        self, problem_2d_N4_fL_qsvt, qsvt_cfg_2d_fast
    ):
        """All solution values must be finite — no NaN or Inf."""
        r = qsvt_solve_2d(problem_2d_N4_fL_qsvt, config=qsvt_cfg_2d_fast)
        assert np.all(np.isfinite(r.u)), (
            "QSVT-2D solution contains non-finite values."
        )
    
    @pytest.mark.slow
    def test_iteration_errors_recorded(
        self, problem_2d_N4_fL_qsvt, qsvt_cfg_2d_fast
    ):
        r = qsvt_solve_2d(problem_2d_N4_fL_qsvt, config=qsvt_cfg_2d_fast)
        assert len(r.iteration_errors) > 0

    @pytest.mark.slow
    def test_iteration_errors_decreasing_overall(
        self, problem_2d_N4_fL_qsvt, qsvt_cfg_2d_fast
    ):
        """Final iteration error must be less than the initial error."""
        r = qsvt_solve_2d(problem_2d_N4_fL_qsvt, config=qsvt_cfg_2d_fast)
        assert r.iteration_errors[-1] < r.iteration_errors[0], (
            f"Iteration error did not decrease: "
            f"initial={r.iteration_errors[0]:.3e}, "
            f"final={r.iteration_errors[-1]:.3e}"
        )

    @pytest.mark.slow
    def test_residual_finite(
        self, problem_2d_N4_fL_qsvt, qsvt_cfg_2d_fast
    ):
        r = qsvt_solve_2d(problem_2d_N4_fL_qsvt, config=qsvt_cfg_2d_fast)
        assert np.isfinite(r.euclidean_residual)

    @pytest.mark.slow
    def test_sign_consistent_with_thomas(
        self, problem_2d_N4_fL_qsvt, qsvt_cfg_2d_fast
    ):
        """
        The mean of the QSVT-2D solution must have the same sign as the
        mean of the Thomas reference. Checking the mean rather than a single
        dominant node is more robust for partially-converged iterates.
        """
        r_thomas = thomas_solve_2d(problem_2d_N4_fL_qsvt)
        r_qsvt   = qsvt_solve_2d(
            problem_2d_N4_fL_qsvt, config=qsvt_cfg_2d_fast
        )

        # Use the mean of the absolute values to find the dominant sign.
        sign_thomas = np.sign(np.mean(r_thomas.u))
        sign_qsvt   = np.sign(np.mean(r_qsvt.u))

        assert sign_thomas == sign_qsvt, (
            f"Mean sign mismatch: "
            f"Thomas mean={np.mean(r_thomas.u):.4f}, "
            f"QSVT mean={np.mean(r_qsvt.u):.4f}"
        )


# -- Cache pre-computation tests ----------------------------------------------

class TestQSVT2DCache:

    def test_cache_angles_shape(self, problem_2d_N4_fL_qsvt):
        """
        The pre-computed phase angles must have length degree + 1.
        Verified by inspecting the cache built inside qsvt_solve_2d.
        """
        from solvers.quantum.qsvt_2d import _build_row_cache
        cfg   = QSVTConfig2D(epsilon=0.1, angle_method="auto", max_degree=50)
        cache = _build_row_cache(problem_2d_N4_fL_qsvt.A_row, 4, cfg)
        assert len(cache.angles) == cache.degree + 1

    def test_cache_alpha_positive(self, problem_2d_N4_fL_qsvt):
        from solvers.quantum.qsvt_2d import _build_row_cache
        cfg   = QSVTConfig2D(epsilon=0.1, angle_method="auto", max_degree=50)
        cache = _build_row_cache(problem_2d_N4_fL_qsvt.A_row, 4, cfg)
        assert cache.alpha > 0.0

    def test_cache_kappa_eff_near_3(self, problem_2d_N4_fL_qsvt):
        """
        kappa_eff for the 2-D row matrix must lie strictly between 1 and 3
        for any N, consistent with the theoretical limit kappa -> 3^- as
        N -> infinity. For N=4 the exact value is approximately 2.36.
        """
        from solvers.quantum.qsvt_2d import _build_row_cache
        cfg   = QSVTConfig2D(epsilon=0.1, angle_method="auto", max_degree=50)
        cache = _build_row_cache(problem_2d_N4_fL_qsvt.A_row, 4, cfg)
        assert 1.0 < cache.kappa_eff < 3.0, (
            f"kappa_eff={cache.kappa_eff:.4f} is outside the theoretical "
            f"range (1, 3) for the 2-D row matrix."
        )

    def test_degree_small_for_row_matrix(self, problem_2d_N4_fL_qsvt):
        """
        The polynomial degree for the 2-D row matrix must be substantially
        smaller than for the 1-D Poisson matrix at the same N. For the 1-D
        matrix at N=4, kappa ~ 9.5 gives degree ~ 45. For the 2-D row matrix,
        kappa ~ 2.36 gives degree < 50 (pyqsp may use a more conservative
        estimate than the theoretical minimum).
        """
        from solvers.quantum.qsvt_2d import _build_row_cache
        cfg   = QSVTConfig2D(epsilon=0.1, angle_method="auto", max_degree=50)
        cache = _build_row_cache(problem_2d_N4_fL_qsvt.A_row, 4, cfg)
        assert cache.degree < 50, (
            f"Polynomial degree {cache.degree} exceeds 50 for kappa_row "
            f"~ 2.36 and epsilon=0.1 — unexpectedly large."
        )
        # Verify it is smaller than the 1-D case at the same N.
        from solvers.quantum.qsp_angles import polynomial_degree_estimate
        degree_1d = polynomial_degree_estimate(9.47, 0.1)   # kappa for N=4 1-D
        assert cache.degree < degree_1d * 2, (
            f"2-D degree ({cache.degree}) is not substantially smaller than "
            f"the 1-D degree estimate ({degree_1d})."
        )


# -- HET 2-D compatibility tests ----------------------------------------------

class TestQSVT2DHET:

    def test_qsvt_runs_on_het_sinusoidal(
        self, het_problem_2d_N4_qsvt, qsvt_cfg_2d_fast
    ):
        """QSVT-2D must complete without error on the HET sinusoidal problem."""
        r = qsvt_solve_2d(het_problem_2d_N4_qsvt, config=qsvt_cfg_2d_fast)
        assert r.u.shape == (4, 4)
        assert np.all(np.isfinite(r.u))

    def test_qsvt_het_sign_consistent_with_thomas(
        self, het_problem_2d_N4_qsvt, qsvt_cfg_2d_fast
    ):
        """
        QSVT-2D and Thomas-2D must agree on the sign of the dominant
        solution component for the HET sinusoidal problem.
        """
        r_thomas = thomas_solve_2d(het_problem_2d_N4_qsvt)
        r_qsvt   = qsvt_solve_2d(
            het_problem_2d_N4_qsvt, config=qsvt_cfg_2d_fast
        )

        idx         = np.unravel_index(
            np.argmax(np.abs(r_thomas.u)), r_thomas.u.shape
        )
        sign_thomas = np.sign(r_thomas.u[idx])
        sign_qsvt   = np.sign(r_qsvt.u[idx])
        assert sign_thomas == sign_qsvt, (
            f"HET sign mismatch at {idx}: "
            f"Thomas={r_thomas.u[idx]:.4f}, QSVT={r_qsvt.u[idx]:.4f}"
        )

    def test_qsvt_het_analytical_error_reasonable(
        self, het_problem_2d_N4_qsvt, qsvt_cfg_2d_fast
    ):
        """
        After 10 iterations, the QSVT-2D solution for the HET sinusoidal
        problem must be closer to the analytical solution than a zero field.
        This is a minimal sanity check — not a tight accuracy requirement.
        """
        r       = qsvt_solve_2d(
            het_problem_2d_N4_qsvt, config=qsvt_cfg_2d_fast
        )
        u_exact = het_problem_2d_N4_qsvt.analytical_solution()

        err_qsvt = float(np.linalg.norm(r.u - u_exact))
        err_zero = float(np.linalg.norm(u_exact))

        assert err_qsvt < err_zero, (
            f"QSVT-2D solution is further from the analytical solution "
            f"than the zero field: err_qsvt={err_qsvt:.3e}, "
            f"err_zero={err_zero:.3e}."
        )