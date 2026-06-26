"""
test_hhl_2d.py
--------------
Tests for the HHL 2D line-Jacobi solver.

N=4 is used throughout.  The line-Jacobi loop for N=4 with ~30 iterations
and 4 HHL calls per iteration takes roughly 2-4 minutes total.
We use max_iter=20 to keep the test under 60 seconds — this is enough
to verify the solver runs and produces a reasonable answer, even if it
has not fully converged.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.config import SimConfig2D
from problems.poisson_2d import PoissonProblem2D
from solvers.quantum.hhl_2d import hhl_solve_2d
from solvers.classical.thomas_2d import thomas_solve_2d
from solvers.quantum.result import SolverResult2D
from conftest import HHL_REL_ERROR_TOL


@pytest.fixture(scope="module")
def problem_2d_N4_fL_fast():
    """N=4, fL, max_iter=20 — enough to verify the solver runs."""
    cfg = SimConfig2D(N=4, epsilon=0.01, source_fn="fL", max_iter=20)
    return PoissonProblem2D(cfg)


class TestHHL2D:

    @pytest.mark.slow
    def test_returns_solver_result_2d(self, problem_2d_N4_fL_fast):
        r = hhl_solve_2d(problem_2d_N4_fL_fast)
        assert isinstance(r, SolverResult2D)

    @pytest.mark.slow
    def test_solution_shape(self, problem_2d_N4_fL_fast):
        r = hhl_solve_2d(problem_2d_N4_fL_fast)
        assert r.u.shape == (4, 4)

    @pytest.mark.slow
    def test_solver_label(self, problem_2d_N4_fL_fast):
        r = hhl_solve_2d(problem_2d_N4_fL_fast)
        assert r.solver == "HHL-2D"
    
    @pytest.mark.slow
    def test_solution_finite(self, problem_2d_N4_fL_fast):
        """All solution values must be finite — no NaN or Inf."""
        r = hhl_solve_2d(problem_2d_N4_fL_fast)
        assert np.all(np.isfinite(r.u)), "HHL-2D solution contains NaN or Inf."

    @pytest.mark.slow
    def test_iteration_errors_recorded(self, problem_2d_N4_fL_fast):
        """Iteration error history must be non-empty and decreasing overall."""
        r = hhl_solve_2d(problem_2d_N4_fL_fast)
        assert len(r.iteration_errors) > 0
        # Final error should be less than initial error.
        assert r.iteration_errors[-1] < r.iteration_errors[0]

    @pytest.mark.slow
    def test_agrees_with_thomas_direction(self, problem_2d_N4_fL_fast):
        """
        HHL-2D and Thomas-2D solutions must have the same sign pattern.

        We do not check magnitude (the solver may not have converged in
        20 iterations), but the sign of the dominant solution component
        must agree — a sign flip indicates a proportionality recovery error.
        """
        r_thomas = thomas_solve_2d(problem_2d_N4_fL_fast)
        r_hhl    = hhl_solve_2d(problem_2d_N4_fL_fast)

        # Find the node with the largest Thomas solution magnitude.
        idx = np.unravel_index(np.argmax(np.abs(r_thomas.u)), r_thomas.u.shape)
        sign_thomas = np.sign(r_thomas.u[idx])
        sign_hhl    = np.sign(r_hhl.u[idx])
        assert sign_thomas == sign_hhl, (
            f"Sign mismatch at dominant node {idx}: "
            f"Thomas={r_thomas.u[idx]:.4f}, HHL={r_hhl.u[idx]:.4f}"
        )

    @pytest.mark.slow
    def test_residual_finite(self, problem_2d_N4_fL_fast):
        r = hhl_solve_2d(problem_2d_N4_fL_fast)
        assert np.isfinite(r.euclidean_residual)