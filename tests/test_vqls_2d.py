"""
Verification tests for the VQLS 2-D line-Jacobi solver.

All tests use N=4 (2 qubits per row sub-problem) with max_iter=15 to
bound runtime. The purpose is to verify structural correctness —
solution shape, finite values, sign consistency, and iteration
bookkeeping — not publication-level numerical accuracy.

Expected runtime: under 3 minutes for the full file.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.config import SimConfig2D
from problems.poisson_2d import PoissonProblem2D
from solvers.quantum.vqls_2d import vqls_solve_2d, VQLSConfig2D
from solvers.quantum.vqls_1d import VQLSConfig1D
from solvers.classical.thomas_2d import thomas_solve_2d
from solvers.quantum.result import SolverResult2D


# -- Shared fast fixture ------------------------------------------------------

@pytest.fixture(scope="module")
def vqls_cfg_2d_fast():
    """
    Minimal VQLS-2D configuration for rapid structural verification.

    Inner solver: 3 layers, 100 iterations per restart, tol=1e-2.
    Outer loop:   max_iter=15 sweeps.
    At N=4, each row solve takes ~5s; 15 iterations × 4 rows = ~5 min
    worst case, typically less with warm-starting.
    """
    inner = VQLSConfig1D(
        n_layers    = 3,
        optimiser   = "COBYLA",
        max_iter    = 100,
        tol         = 1e-2,
        random_seed = 0,
        verbose     = False,
    )
    return VQLSConfig2D(
        inner_config = inner,
        warm_start   = True,
        verbose      = True,
    )


@pytest.fixture(scope="module")
def problem_2d_N4_fL_vqls():
    """N=4, fL, max_iter=15 — sufficient to verify solver operation."""
    cfg = SimConfig2D(N=4, epsilon=0.01, source_fn="fL", max_iter=15)
    return PoissonProblem2D(cfg)


# -- Structural tests ---------------------------------------------------------

class TestVQLS2DStructure:

    def test_returns_solver_result_2d(
        self, problem_2d_N4_fL_vqls, vqls_cfg_2d_fast
    ):
        r = vqls_solve_2d(problem_2d_N4_fL_vqls, config=vqls_cfg_2d_fast)
        assert isinstance(r, SolverResult2D)

    def test_solution_shape(
        self, problem_2d_N4_fL_vqls, vqls_cfg_2d_fast
    ):
        r = vqls_solve_2d(problem_2d_N4_fL_vqls, config=vqls_cfg_2d_fast)
        assert r.u.shape == (4, 4)

    def test_solver_label(
        self, problem_2d_N4_fL_vqls, vqls_cfg_2d_fast
    ):
        r = vqls_solve_2d(problem_2d_N4_fL_vqls, config=vqls_cfg_2d_fast)
        assert r.solver == "VQLS-2D"

    def test_solution_finite(
        self, problem_2d_N4_fL_vqls, vqls_cfg_2d_fast
    ):
        """All solution values must be finite — no NaN or Inf."""
        r = vqls_solve_2d(problem_2d_N4_fL_vqls, config=vqls_cfg_2d_fast)
        assert np.all(np.isfinite(r.u)), (
            "VQLS-2D solution contains non-finite values."
        )

    def test_iteration_errors_recorded(
        self, problem_2d_N4_fL_vqls, vqls_cfg_2d_fast
    ):
        """Iteration error history must be non-empty."""
        r = vqls_solve_2d(problem_2d_N4_fL_vqls, config=vqls_cfg_2d_fast)
        assert len(r.iteration_errors) > 0

    def test_iteration_errors_decreasing_overall(
        self, problem_2d_N4_fL_vqls, vqls_cfg_2d_fast
    ):
        """Final iteration error must be less than the initial error."""
        r = vqls_solve_2d(problem_2d_N4_fL_vqls, config=vqls_cfg_2d_fast)
        assert r.iteration_errors[-1] < r.iteration_errors[0], (
            f"Iteration error did not decrease: "
            f"initial={r.iteration_errors[0]:.3e}, "
            f"final={r.iteration_errors[-1]:.3e}"
        )

    def test_residual_finite(
        self, problem_2d_N4_fL_vqls, vqls_cfg_2d_fast
    ):
        r = vqls_solve_2d(problem_2d_N4_fL_vqls, config=vqls_cfg_2d_fast)
        assert np.isfinite(r.euclidean_residual)

    def test_sign_consistent_with_thomas(
        self, problem_2d_N4_fL_vqls, vqls_cfg_2d_fast
    ):
        """
        The dominant solution component must have the same sign as the
        Thomas reference. A sign flip indicates a proportionality
        recovery failure.
        """
        r_thomas = thomas_solve_2d(problem_2d_N4_fL_vqls)
        r_vqls   = vqls_solve_2d(problem_2d_N4_fL_vqls, config=vqls_cfg_2d_fast)

        idx           = np.unravel_index(
            np.argmax(np.abs(r_thomas.u)), r_thomas.u.shape
        )
        sign_thomas   = np.sign(r_thomas.u[idx])
        sign_vqls     = np.sign(r_vqls.u[idx])
        assert sign_thomas == sign_vqls, (
            f"Sign mismatch at dominant node {idx}: "
            f"Thomas={r_thomas.u[idx]:.4f}, VQLS={r_vqls.u[idx]:.4f}"
        )

    def test_warm_start_does_not_raise(self, problem_2d_N4_fL_vqls):
        """Warm-start mode must complete without raising any exception."""
        inner = VQLSConfig1D(
            n_layers=2, max_iter=50, tol=1e-1,
            random_seed=7, verbose=False,
        )
        cfg = VQLSConfig2D(inner_config=inner, warm_start=True, verbose=False)
        cfg_prob = SimConfig2D(N=4, epsilon=0.01, source_fn="fL", max_iter=5)
        prob     = PoissonProblem2D(cfg_prob)
        r = vqls_solve_2d(prob, config=cfg)
        assert r.u.shape == (4, 4)

    def test_no_warm_start_does_not_raise(self, problem_2d_N4_fL_vqls):
        """Cold-start mode (warm_start=False) must also complete cleanly."""
        inner = VQLSConfig1D(
            n_layers=2, max_iter=50, tol=1e-1,
            random_seed=7, verbose=False,
        )
        cfg = VQLSConfig2D(inner_config=inner, warm_start=False, verbose=False)
        cfg_prob = SimConfig2D(N=4, epsilon=0.01, source_fn="fL", max_iter=5)
        prob     = PoissonProblem2D(cfg_prob)
        r = vqls_solve_2d(prob, config=cfg)
        assert r.u.shape == (4, 4)