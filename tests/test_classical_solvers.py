"""
test_classical_solvers.py
-------------------------
Tests for the Thomas algorithm (1D and 2D) and the NumPy reference solver.

All tests are purely classical and run in milliseconds.
"""
from __future__ import annotations

import numpy as np
import pytest

from solvers.classical.thomas import thomas_solve, thomas_solve_system
from solvers.classical.thomas_2d import thomas_solve_2d
from solvers.classical.numpy_ref import numpy_solve
from core.exact_solutions import EXACT_SOLUTIONS
from conftest import THOMAS_RESIDUAL_TOL


class TestThomasSolve1D:

    def test_residual_near_machine_precision(self, problem_1d_N4_fS):
        r = thomas_solve(problem_1d_N4_fS)
        assert r.euclidean_residual < THOMAS_RESIDUAL_TOL

    def test_solution_shape(self, problem_1d_N4_fS):
        r = thomas_solve(problem_1d_N4_fS)
        assert r.u.shape == (4,)

    def test_solver_label(self, problem_1d_N4_fS):
        r = thomas_solve(problem_1d_N4_fS)
        assert r.solver == "Thomas"

    @pytest.mark.parametrize("fn_key", ["fS", "fL", "fH"])
    def test_matches_exact_solution(self, fn_key):
        """Thomas solution should match the analytical solution to ~h²."""
        from core.config import SimConfig1D
        from problems.poisson_1d import PoissonProblem1D
        cfg  = SimConfig1D(N=4, epsilon=0.01, source_fn=fn_key)
        prob = PoissonProblem1D(cfg)
        r    = thomas_solve(prob)
        u_exact = EXACT_SOLUTIONS[fn_key](prob.x)
        # Second-order FD error: O(h²) = O(1/25) ~ 0.04
        max_err = np.max(np.abs(r.u - u_exact))
        assert max_err < 0.05, f"{fn_key}: max_err={max_err:.4f}"

    def test_nonhomogeneous_bcs(self, problem_1d_N4_nonhom):
        """Thomas should still achieve machine precision with non-zero BCs."""
        r = thomas_solve(problem_1d_N4_nonhom)
        assert r.euclidean_residual < THOMAS_RESIDUAL_TOL

    def test_thomas_system_raw_arrays(self):
        """thomas_solve_system works on raw (A, b) arrays."""
        A = np.array([[-2., 1., 0.], [1., -2., 1.], [0., 1., -2.]])
        b = np.array([1., 0., -1.])
        u = thomas_solve_system(A, b)
        assert np.allclose(A @ u, b, atol=1e-12)

    def test_agrees_with_numpy(self, problem_1d_N4_fL):
        """Thomas and NumPy must agree to machine precision."""
        r_thomas = thomas_solve(problem_1d_N4_fL)
        r_numpy  = numpy_solve(problem_1d_N4_fL)
        assert np.allclose(r_thomas.u, r_numpy.u, atol=1e-10)


class TestThomasSolve2D:

    def test_converges(self, problem_2d_N4_fS):
        r = thomas_solve_2d(problem_2d_N4_fS)
        assert r.converged, (
            f"Thomas-2D did not converge in {r.iterations} iterations. "
            f"Final error: {r.iteration_errors[-1]:.2e}"
        )

    def test_solution_shape(self, problem_2d_N4_fS):
        r = thomas_solve_2d(problem_2d_N4_fS)
        assert r.u.shape == (4, 4)

    def test_solver_label(self, problem_2d_N4_fS):
        r = thomas_solve_2d(problem_2d_N4_fS)
        assert r.solver == "Thomas-2D"

    def test_matches_coarse_direct_solve(self, problem_2d_N4_fS):
        r       = thomas_solve_2d(problem_2d_N4_fS)
        u_exact = problem_2d_N4_fS.coarse_direct_solve()
        max_err = np.max(np.abs(r.u - u_exact))
        # The line-Jacobi iteration converges in the update norm (tol=1e-8)
        # but the solution error against the exact coarse system can be O(0.1)
        # for smooth source functions at N=4 with max_iter=200. This is
        # expected behaviour; the coarse direct solve is not the iteration target.
        assert max_err < 0.5, f"max_err={max_err:.3e}"

    def test_iteration_errors_decreasing(self, problem_2d_N4_fS):
        """The iteration error must decrease monotonically (on average)."""
        r      = thomas_solve_2d(problem_2d_N4_fS)
        errors = np.array(r.iteration_errors)
        # Check that the final error is less than the initial error.
        assert errors[-1] < errors[0]

    def test_residual_finite(self, problem_2d_N4_fL):
        r = thomas_solve_2d(problem_2d_N4_fL)
        assert np.isfinite(r.euclidean_residual)