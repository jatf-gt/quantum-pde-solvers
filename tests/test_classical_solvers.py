"""
test_classical_solvers.py
-------------------------
Tests for the 1D Thomas algorithm and the NumPy reference solver.

All tests are purely classical and run in milliseconds.

The 2D Thomas coverage that formerly lived here exercised the retired
``thomas_solve_2d``. Its replacement is ``solvers.outer.solve(problem,
inner="thomas")``, whose coverage belongs with the rest of the outer-iteration
layer rather than here; that is outstanding work.
"""
from __future__ import annotations

import numpy as np
import pytest

from solvers.classical.thomas import thomas_solve, thomas_solve_system
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
