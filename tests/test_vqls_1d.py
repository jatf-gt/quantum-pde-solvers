"""
test_vqls_1d.py
---------------
Tests for the VQLS 1D solver.

Uses N=4 (2 qubits) and a fast VQLSConfig1D (n_layers=3, max_iter=150)
to keep each test under ~15 seconds.  Tolerances are loose (15%).
"""
from __future__ import annotations

import numpy as np
import pytest

from solvers.quantum.vqls_1d import (
    vqls_solve, vqls_solve_system, VQLSConfig1D, VQLSSolverResult,
)
from solvers.classical.thomas import thomas_solve
from conftest import VQLS_REL_ERROR_TOL, VQLS_COST_TOL, rel_err

# Every test in this module builds and simulates a quantum circuit.
pytestmark = pytest.mark.quantum



class TestVQLSSolve1D:

    def test_returns_vqls_solver_result(self, problem_1d_N4_fS, vqls_cfg_fast):
        """
        Validates that the solver returns a correctly typed VQLSSolverResult object.
        """
        r = vqls_solve(problem_1d_N4_fS, config=vqls_cfg_fast)
        assert isinstance(r, VQLSSolverResult)

    def test_solution_shape(self, problem_1d_N4_fS, vqls_cfg_fast):
        """
        Ensures that the computed solution vector matches the expected dimensions of the problem space.
        """
        r = vqls_solve(problem_1d_N4_fS, config=vqls_cfg_fast)
        assert r.u.shape == (4,)

    def test_solver_label(self, problem_1d_N4_fS, vqls_cfg_fast):
        """
        Confirms the solver metadata correctly identifies as VQLS.
        """
        r = vqls_solve(problem_1d_N4_fS, config=vqls_cfg_fast)
        assert r.solver == "VQLS"

    def test_cost_converged(self, problem_1d_N4_fS, vqls_cfg_fast):
        """
        Validates that the optimization cost function converges below the specified tolerance threshold.
        Ensures the optimizer operates correctly without stagnation.
        """
        r = vqls_solve(problem_1d_N4_fS, config=vqls_cfg_fast)
        assert r.final_cost < VQLS_COST_TOL, (
            f"VQLS cost={r.final_cost:.4f} > threshold={VQLS_COST_TOL}. "
            f"Optimiser may be stuck. evals={r.n_circuit_evals}"
        )

    def test_prop_const_nonzero(self, problem_1d_N4_fS, vqls_cfg_fast):
        """
        Confirms that the proportionality constant of the quantum state is finite and non-zero.
        """
        r = vqls_solve(problem_1d_N4_fS, config=vqls_cfg_fast)
        assert r.prop_const is not None
        assert np.isfinite(r.prop_const)
        assert abs(r.prop_const) > 1e-10

    def test_agrees_with_thomas_loose(self, problem_1d_N4_fS, vqls_cfg_fast):
        """
        Validates that the VQLS solution broadly agrees with the classical Thomas algorithm baseline within an acceptable relative error margin.
        """
        u_thomas = thomas_solve(problem_1d_N4_fS).u
        u_vqls   = vqls_solve(problem_1d_N4_fS, config=vqls_cfg_fast).u
        err      = rel_err(u_vqls, u_thomas)
        assert err < VQLS_REL_ERROR_TOL, (
            f"VQLS vs Thomas rel err = {err*100:.2f}% > "
            f"{VQLS_REL_ERROR_TOL*100:.0f}%"
        )

    def test_n_circuit_evals_positive(self, problem_1d_N4_fS, vqls_cfg_fast):
        """
        Ensures that the optimization process executes a positive number of quantum circuit evaluations.
        """
        r = vqls_solve(problem_1d_N4_fS, config=vqls_cfg_fast)
        assert r.n_circuit_evals > 0

    def test_cost_history_nonempty(self, problem_1d_N4_fS, vqls_cfg_fast):
        """
        Validates that the solver correctly logs the history of cost evaluations over the optimization trajectory.
        """
        r = vqls_solve(problem_1d_N4_fS, config=vqls_cfg_fast)
        assert len(r.cost_history) > 0

    def test_optimal_params_shape(self, problem_1d_N4_fS, vqls_cfg_fast):
        """
        Confirms that the derived optimal parameters align structurally with the ansatz depth and qubit count.
        """
        from solvers.quantum.vqls_utils import n_params
        r        = vqls_solve(problem_1d_N4_fS, config=vqls_cfg_fast)
        expected = n_params(2, vqls_cfg_fast.n_layers)   # 2 qubits for N=4
        assert r.optimal_params.shape == (expected,)

    def test_residual_finite(self, problem_1d_N4_fS, vqls_cfg_fast):
        """
        Ensures the computed Euclidean residual is a valid, finite numerical quantity.
        """
        r = vqls_solve(problem_1d_N4_fS, config=vqls_cfg_fast)
        assert np.isfinite(r.euclidean_residual)

    def test_solution_finite(self, problem_1d_N4_fS, vqls_cfg_fast):
        """
        Validates that all elements of the computed solution vector are finite.
        """
        r = vqls_solve(problem_1d_N4_fS, config=vqls_cfg_fast)
        assert np.all(np.isfinite(r.u))

    def test_vqls_solve_system_raw_arrays(self, vqls_cfg_fast):
        """
        Confirms that the primitive array interface successfully processes and solves raw system matrices.
        """
        A = np.array([[-2., 1., 0., 0.],
                      [ 1.,-2., 1., 0.],
                      [ 0., 1.,-2., 1.],
                      [ 0., 0., 1.,-2.]], dtype=float)
        b = np.array([0.1, 0.2, 0.2, 0.1])
        r = vqls_solve_system(A, b, vqls_cfg_fast)
        assert r.u.shape == (4,)
        assert np.isfinite(r.final_cost)

    def test_non_power_of_2_raises(self, vqls_cfg_fast):
        """
        Ensures that the solver enforces structural prerequisites by rejecting systems whose dimensionality is not a power of 2.
        """
        A = np.eye(3) * (-2)
        b = np.ones(3)
        with pytest.raises(ValueError, match="power of 2"):
            vqls_solve_system(A, b, vqls_cfg_fast)

    def test_non_hermitian_raises(self, vqls_cfg_fast):
        """
        Validates that the solver explicitly rejects non-Hermitian system matrices, maintaining algorithmic assumptions.
        """
        A = np.array([[-2., 2.], [1., -2.]])   # asymmetric
        b = np.array([1., 1.])
        with pytest.raises(ValueError, match="Hermitian"):
            vqls_solve_system(A, b, vqls_cfg_fast)

    def test_reproducible_with_same_seed(self, problem_1d_N4_fS):
        """
        Confirms that identical pseudo-random initialization seeds yield perfectly deterministic solution vectors.
        """
        cfg = VQLSConfig1D(n_layers=2, max_iter=50, tol=1e-2,
                         random_seed=99, verbose=False)
        r1 = vqls_solve(problem_1d_N4_fS, config=cfg)
        r2 = vqls_solve(problem_1d_N4_fS, config=cfg)
        assert np.allclose(r1.u, r2.u, atol=1e-10)