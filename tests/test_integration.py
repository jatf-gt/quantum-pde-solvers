"""
test_integration.py
-------------------
End-to-end integration tests: problem → solver → benchmark metrics.

These tests verify that the full pipeline from SimConfig to BenchmarkResult
works without error and that the metrics are internally consistent.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.config import SimConfig1D
from problems.poisson_1d import PoissonProblem1D
from solvers.classical.thomas import thomas_solve
from solvers.quantum.hhl_1d import hhl_solve
from solvers.quantum.vqls_1d import vqls_solve
from benchmark.metrics import compute_errors, BenchmarkResult
from conftest import VQLS_COST_TOL


class TestFullPipeline1D:

    def test_thomas_pipeline(self, problem_1d_N4_fS):
        """Thomas: problem → solve → compute_errors → BenchmarkResult."""
        sr = thomas_solve(problem_1d_N4_fS)
        br = compute_errors(problem_1d_N4_fS, sr)
        assert isinstance(br, BenchmarkResult)
        assert br.solver == "Thomas"
        assert br.u_exact is not None   # fS has analytical solution
        assert br.max_abs_error < 0.05  # O(h²) discretisation error

    def test_hhl_pipeline(self, problem_1d_N4_fS):
        """HHL: problem → solve → compute_errors → BenchmarkResult."""
        sr = hhl_solve(problem_1d_N4_fS)
        br = compute_errors(problem_1d_N4_fS, sr)
        assert isinstance(br, BenchmarkResult)
        assert br.solver == "HHL"
        assert br.euclidean_residual is not None
        assert np.isfinite(br.euclidean_residual)

    def test_vqls_pipeline(self, problem_1d_N4_fS, vqls_cfg_fast):
        """VQLS: problem → solve → compute_errors → BenchmarkResult."""
        sr = vqls_solve(problem_1d_N4_fS, config=vqls_cfg_fast)
        br = compute_errors(problem_1d_N4_fS, sr)
        assert isinstance(br, BenchmarkResult)
        assert br.solver == "VQLS"

    def test_benchmark_result_fields_consistent(self, problem_1d_N4_fS):
        """
        BenchmarkResult fields must be internally consistent:
          - max_abs_error >= avg_abs_error
          - rel_error and abs_error have the same length as the solution
          - max_rel_error >= avg_rel_error
        """
        sr = thomas_solve(problem_1d_N4_fS)
        br = compute_errors(problem_1d_N4_fS, sr)
        assert br.max_abs_error >= br.avg_abs_error
        assert len(br.abs_error) == problem_1d_N4_fS.config.N
        if br.max_rel_error is not None and br.avg_rel_error is not None:
            assert br.max_rel_error >= br.avg_rel_error

    def test_hhl_better_than_random(self, problem_1d_N4_fS):
        """
        HHL solution must be closer to Thomas than a random vector of
        the same norm.  This is a basic sanity check that the solver
        is doing something useful.
        """
        u_thomas = thomas_solve(problem_1d_N4_fS).u
        u_hhl    = hhl_solve(problem_1d_N4_fS).u

        rng      = np.random.default_rng(0)
        u_random = rng.normal(size=4)
        u_random *= np.linalg.norm(u_thomas) / np.linalg.norm(u_random)

        err_hhl    = np.linalg.norm(u_hhl    - u_thomas)
        err_random = np.linalg.norm(u_random - u_thomas)
        assert err_hhl < err_random, (
            f"HHL error ({err_hhl:.4f}) >= random error ({err_random:.4f}). "
            f"Solver may not be working."
        )

    def test_three_source_functions_all_pass(self, vqls_cfg_fast):
        """All three source functions must produce valid results for all solvers."""
        for fn_key in ("fS", "fL", "fH"):
            cfg  = SimConfig1D(N=4, epsilon=0.01, source_fn=fn_key)
            prob = PoissonProblem1D(cfg)

            sr_thomas = thomas_solve(prob)
            sr_hhl    = hhl_solve(prob)
            sr_vqls   = vqls_solve(prob, config=vqls_cfg_fast)

            assert np.all(np.isfinite(sr_thomas.u)), f"Thomas NaN for {fn_key}"
            assert np.all(np.isfinite(sr_hhl.u)),    f"HHL NaN for {fn_key}"
            assert np.all(np.isfinite(sr_vqls.u)),   f"VQLS NaN for {fn_key}"


class TestBenchmarkMetrics:

    def test_compute_errors_homogeneous_has_exact(self, problem_1d_N4_fS):
        """Homogeneous BC problems must have u_exact populated."""
        sr = thomas_solve(problem_1d_N4_fS)
        br = compute_errors(problem_1d_N4_fS, sr)
        assert br.u_exact is not None
        assert br.rel_error is not None
        assert br.max_rel_error is not None

    def test_compute_errors_nonhomogeneous_no_exact(self, problem_1d_N4_nonhom):
        """Non-homogeneous BC problems have no analytical solution."""
        sr = thomas_solve(problem_1d_N4_nonhom)
        br = compute_errors(problem_1d_N4_nonhom, sr)
        assert br.u_exact is None
        assert br.max_rel_error is None

    def test_thomas_zero_error_against_itself(self, problem_1d_N4_fS):
        """
        When Thomas is used as both the solver and the u_thomas reference,
        the absolute error against itself must be identically zero.
        This test uses a non-homogeneous BC problem so that u_exact is None
        and abs_error is computed against u_thomas rather than the analytical
        solution.
        """
        from core.config import SimConfig1D
        cfg  = SimConfig1D(N=4, epsilon=0.01, source_fn="fS", alpha=0.5, beta=-0.5)
        prob = PoissonProblem1D(cfg)
        sr   = thomas_solve(prob)
        br   = compute_errors(prob, sr, u_thomas=sr.u)
        assert br.max_abs_error == pytest.approx(0.0, abs=1e-12)