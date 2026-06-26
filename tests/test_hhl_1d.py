"""
test_hhl_1d.py
--------------
Tests for the HHL 1D solver.

All tests use N=4 (2 qubits) to keep runtime under ~30 seconds per test.
Tolerances are loose (20%) — we are checking correctness, not accuracy.
"""
from __future__ import annotations

import numpy as np
import pytest

from solvers.quantum.hhl_1d import hhl_solve, hhl_solve_system
from solvers.classical.thomas import thomas_solve
from conftest import HHL_REL_ERROR_TOL


def _rel_err(u: np.ndarray, ref: np.ndarray) -> float:
    """Max relative error excluding near-zero nodes."""
    scale = np.max(np.abs(ref))
    mask  = np.abs(ref) > 1e-6 * scale
    if not mask.any():
        return 0.0
    return float(np.max(np.abs((u - ref)[mask]) / np.abs(ref[mask])))


class TestHHL1D:

    def test_returns_solver_result(self, problem_1d_N4_fS):
        from solvers.quantum.result import SolverResult
        r = hhl_solve(problem_1d_N4_fS)
        assert isinstance(r, SolverResult)

    def test_solution_shape(self, problem_1d_N4_fS):
        r = hhl_solve(problem_1d_N4_fS)
        assert r.u.shape == (4,)

    def test_solver_label(self, problem_1d_N4_fS):
        r = hhl_solve(problem_1d_N4_fS)
        assert r.solver == "HHL"

    def test_prop_const_nonzero(self, problem_1d_N4_fS):
        """Proportionality constant must be non-zero and finite."""
        r = hhl_solve(problem_1d_N4_fS)
        assert r.prop_const is not None
        assert np.isfinite(r.prop_const)
        assert abs(r.prop_const) > 1e-10

    def test_residual_finite(self, problem_1d_N4_fS):
        r = hhl_solve(problem_1d_N4_fS)
        assert np.isfinite(r.euclidean_residual)

    @pytest.mark.parametrize("fn_key", ["fS", "fL", "fH"])
    def test_agrees_with_thomas_loose(self, fn_key):
        """
        HHL solution must agree with Thomas to within HHL_REL_ERROR_TOL.

        This is a loose check (20%) — the Trotter approximation at
        epsilon=0.01 introduces errors of order 1-5% for N=4.
        """
        from core.config import SimConfig1D
        from problems.poisson_1d import PoissonProblem1D
        cfg      = SimConfig1D(N=4, epsilon=0.01, source_fn=fn_key)
        prob     = PoissonProblem1D(cfg)
        u_thomas = thomas_solve(prob).u
        u_hhl    = hhl_solve(prob).u
        err      = _rel_err(u_hhl, u_thomas)
        assert err < HHL_REL_ERROR_TOL, (
            f"HHL vs Thomas rel err = {err*100:.2f}% > "
            f"{HHL_REL_ERROR_TOL*100:.0f}% for {fn_key}"
        )

    def test_solution_has_correct_sign(self, problem_1d_N4_fS):
        """
        For fS with homogeneous BCs, the solution is negative (u = -sin(πx)/π²).
        The proportionality recovery must get the sign right.
        """
        from core.exact_solutions import EXACT_SOLUTIONS
        r       = hhl_solve(problem_1d_N4_fS)
        u_exact = EXACT_SOLUTIONS["fS"](problem_1d_N4_fS.x)
        # Check that the signs agree at the majority of nodes.
        sign_agreement = np.mean(np.sign(r.u) == np.sign(u_exact))
        assert sign_agreement >= 0.75, (
            f"Sign agreement = {sign_agreement:.0%} — "
            f"proportionality recovery may have wrong sign."
        )

    def test_hhl_solve_system_raw_arrays(self, problem_1d_N4_fS):
        """hhl_solve_system works on raw (A, b, epsilon) inputs."""
        prob = problem_1d_N4_fS
        u, x_raw, c = hhl_solve_system(prob.A, prob.b, 0.01)
        assert u.shape == (4,)
        assert x_raw.shape == (4,)
        assert np.isfinite(c)

    def test_zero_rhs_raises(self):
        """A zero RHS should raise ValueError before calling HHL."""
        A = np.array([[-2., 1.], [1., -2.]])
        b = np.zeros(2)
        with pytest.raises(ValueError, match="zero"):
            hhl_solve_system(A, b, 0.01)