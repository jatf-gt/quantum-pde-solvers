"""
test_outer.py
-------------
Tests for the outer-iteration layer, `solvers/outer`.

This package is the sole 2D/3D architecture: every multi-dimensional result in
the project, classical and quantum, is produced by `solvers.outer.solve`. It had
no test coverage prior to this file despite being what the HPC sweeps run.

Coverage is organised by the four modules of the package:

    core.py        WorkLog, StagnationMonitor, strip_sweep
    inner.py       the option-validated strip solver registry
    stationary.py  line Jacobi / Gauss-Seidel / SOR
    multigrid.py   transfer operators, hierarchy construction, V-cycle and FMG

plus the `solve` / `solve_staged` entry points.

Two properties are load-bearing and tested explicitly rather than incidentally:

  * `scheme="jacobi"` with `criterion="delta"` must reproduce the original
    line-Jacobi loop *exactly*. The published 2D figures were produced by that
    loop, and the retired implementation is no longer present to compare
    against, so the reference is reconstructed here from first principles.

  * The option registry must reject unknown keys. A registry that silently
    absorbed them would accept `qsvt_max_degrees=500`, ignore it, and let an
    HPC run cost an order of magnitude more than intended whilst appearing to
    honour the setting.

Every test uses the direct Thomas strip solver or the `perturbed` surrogate, so
the file runs in seconds and needs no quantum backend.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import build_cube_3d, build_periodic_3d, build_square_2d
from problems.poisson_line_2d import PoissonLine2D
from solvers.outer import (InnerConfig, available_inner, available_options,
                           available_schemes, build_hierarchy, describe_inner,
                           describe_scheme, get_inner, interpolation_1d,
                           interpolation_1d_periodic, optimal_omega,
                           resolve_options, solve, solve_multigrid,
                           solve_staged, solve_stationary, strip_sweep)
from solvers.outer.core import OuterResult, StagnationMonitor, WorkLog
from solvers.outer.multigrid import restriction_from


# -- Work accounting -----------------------------------------------------------

class TestWorkLog:

    def test_counts_by_strip_size(self):
        w = WorkLog()
        w.add(64, 3)
        w.add(32)
        w.add(64)
        assert w.solves_by_size == {64: 4, 32: 1}
        assert w.total == 5

    def test_weighted_cost_is_in_finest_solve_units(self):
        """
        Coarse strips are exponentially cheaper, so a plain solve count
        understates multigrid's advantage. With alpha = 0 every solve costs the
        same and the weighted cost degenerates to the raw total.
        """
        w = WorkLog()
        w.add(64, 2)
        w.add(32, 4)
        assert w.weighted_cost(0.0) == pytest.approx(6.0)
        # alpha = 1: the half-length strips cost half as much each.
        assert w.weighted_cost(1.0) == pytest.approx(2.0 + 4.0 * 0.5)
        # alpha = 2: a quarter each.
        assert w.weighted_cost(2.0) == pytest.approx(2.0 + 4.0 * 0.25)

    def test_weighted_cost_of_empty_log_is_zero(self):
        assert WorkLog().weighted_cost(2.35) == 0.0

    def test_merge_accumulates(self):
        a, b = WorkLog(), WorkLog()
        a.add(16, 2)
        b.add(16, 3)
        b.add(8, 1)
        a.merge(b)
        assert a.solves_by_size == {16: 5, 8: 1}

    def test_summary_mentions_every_size(self):
        w = WorkLog()
        w.add(16, 2)
        w.add(8, 1)
        s = w.summary()
        assert "3 solves" in s and "n=16:2" in s and "n=8:1" in s


# -- Stagnation detection ------------------------------------------------------

class TestStagnationMonitor:

    def test_silent_before_the_window_fills(self):
        m = StagnationMonitor(window=10)
        assert not any(m.update(1.0) for _ in range(9))

    def test_non_finite_residual_stops_immediately(self):
        m = StagnationMonitor(window=10)
        assert m.update(float("nan")) is True
        assert StagnationMonitor(window=10).update(float("inf")) is True

    def test_flat_residual_is_detected_as_a_floor(self):
        m = StagnationMonitor(window=10, min_improvement=0.01)
        stopped = [m.update(1e-6) for _ in range(20)]
        assert any(stopped)

    def test_healthy_geometric_convergence_is_not_stopped(self):
        """
        Even slow convergence must survive: line-SOR has rho → 1 as N grows, so
        a per-iteration test would eventually mistake healthy convergence for
        stagnation. Over a 10-iteration window rho = 0.995 still yields ~5 %
        improvement, comfortably above the 1 % threshold.
        """
        m = StagnationMonitor(window=10, min_improvement=0.01)
        r = 1.0
        for _ in range(60):
            r *= 0.995
            assert not m.update(r)

    def test_transient_dip_does_not_trigger_a_stop(self):
        """
        Line-SOR residuals are not monotone: a sharp transient dip occurs where
        truncation and iteration error briefly cancel. Comparing medians of the
        two half-windows ignores the outlier, whereas a test built on the raw
        residual would read the recovery as a stall.
        """
        m = StagnationMonitor(window=20, min_improvement=0.01)
        r = 1.0
        stopped = False
        for i in range(40):
            r *= 0.97
            value = r * 0.02 if i == 25 else r      # a single 50x dip
            stopped |= m.update(value)
        assert not stopped

    def test_best_records_the_lowest_residual_seen(self):
        m = StagnationMonitor(window=5)
        for v in (1.0, 0.1, 0.5):
            m.update(v)
        assert m.best == pytest.approx(0.1)


# -- Inner solver registry -----------------------------------------------------

class TestInnerRegistry:

    def test_expected_solvers_are_registered(self):
        assert {"thomas", "perturbed", "hhl", "vqls", "qsvt"} <= set(available_inner())

    def test_unknown_solver_raises_and_lists_alternatives(self):
        with pytest.raises(ValueError, match="Unknown inner solver"):
            get_inner("definitely_not_a_solver")

    def test_unknown_option_raises_rather_than_being_ignored(self):
        """
        The single most important property of the registry. Silently absorbing
        an unrecognised key would let a mistyped option be accepted, ignored,
        and change the cost of an HPC run without any indication.
        """
        with pytest.raises(ValueError, match="Unknown option"):
            get_inner("qsvt", max_degrees=500)

    def test_error_message_names_the_valid_options(self):
        with pytest.raises(ValueError, match="max_degree"):
            get_inner("qsvt", max_degrees=500)

    def test_declared_defaults_are_applied(self):
        assert resolve_options("perturbed", None) == {"delta": 0.0, "seed": 0}

    def test_unset_options_defer_to_the_underlying_solver(self):
        """
        An option with no declared default must be omitted entirely, so the
        wrapper cannot silently re-specify VQLSConfig1D's own defaults.
        """
        assert resolve_options("vqls", None) == {}

    def test_options_are_type_coerced(self):
        resolved = resolve_options("perturbed", {"delta": "0.25", "seed": "7"})
        assert resolved == {"delta": 0.25, "seed": 7}
        assert isinstance(resolved["seed"], int)

    def test_uncoercible_option_raises(self):
        with pytest.raises(ValueError, match="expected float"):
            resolve_options("perturbed", {"delta": "not_a_number"})

    def test_boolean_options_accept_string_spellings(self):
        assert resolve_options("vqls", {"verbose": "yes"}) == {"verbose": True}
        assert resolve_options("vqls", {"verbose": "off"}) == {"verbose": False}

    def test_available_options_rejects_unknown_solver(self):
        with pytest.raises(ValueError, match="Unknown inner solver"):
            available_options("nope")

    def test_describe_is_introspectable(self):
        text = describe_inner("qsvt")
        assert "max_degree" in text and "epsilon" in text

    def test_inner_config_routes_per_solver_sections(self):
        cfg = InnerConfig(qsvt={"max_degree": 50}, hhl={"epsilon": 0.05})
        assert cfg.for_solver("qsvt") == {"max_degree": 50}
        assert cfg.for_solver("vqls") == {}


class TestThomasInnerSolver:

    def test_solves_a_tridiagonal_system_exactly(self):
        A = np.array([[-2.0, 1.0, 0.0],
                      [1.0, -2.0, 1.0],
                      [0.0, 1.0, -2.0]])
        b = np.array([1.0, -2.0, 3.0])
        x = get_inner("thomas")(A, b)
        assert np.allclose(A @ x, b)

    def test_zero_rhs_short_circuits_without_calling_the_solver(self):
        """
        A strip whose right-hand side is numerically zero has the zero solution.
        Skipping it avoids a pointless — and for VQLS ill-posed — quantum call.
        """
        calls = []

        def failing(A, b):
            calls.append(1)
            raise AssertionError("must not be called for a zero RHS")

        from solvers.outer.inner import InnerSolverWrapper
        wrapper = InnerSolverWrapper("probe", failing)
        assert np.allclose(wrapper(np.eye(4), np.zeros(4)), 0.0)
        assert not calls

    def test_failure_falls_back_and_is_counted(self):
        """A substituted solve must never be silent."""
        from solvers.outer.inner import InnerSolverWrapper, _FACTORIES

        def failing(A, b):
            raise RuntimeError("simulated backend failure")

        wrapper = InnerSolverWrapper("probe", failing,
                                     fallback=_FACTORIES["thomas"]())
        A = np.array([[-2.0, 1.0], [1.0, -2.0]])
        x = wrapper(A, np.array([1.0, 1.0]))
        assert np.allclose(A @ x, [1.0, 1.0])
        assert wrapper.failures == 1
        assert wrapper.records[-1]["fallback"] is True

    def test_summary_reports_call_statistics(self):
        inner = get_inner("thomas")
        A = np.array([[-2.0, 1.0], [1.0, -2.0]])
        for _ in range(3):
            inner(A, np.array([1.0, 2.0]))
        s = inner.summary()
        assert s["inner_calls"] == 3
        assert s["inner_failures"] == 0
        assert s["inner_total_s"] >= 0.0


class TestPerturbedInnerSolver:

    def test_zero_delta_is_an_exact_solve(self):
        A = np.array([[-2.0, 1.0], [1.0, -2.0]])
        b = np.array([1.0, 2.0])
        x = get_inner("perturbed", delta=0.0)(A, b)
        assert np.allclose(A @ x, b)

    def test_perturbation_is_deterministic(self):
        """
        The surrogate models a *systematic* approximation error, so repeated
        solves of the same system must return the same answer.
        """
        A = np.array([[-2.0, 1.0], [1.0, -2.0]])
        b = np.array([1.0, 2.0])
        a = get_inner("perturbed", delta=0.05, seed=3)(A, b)
        c = get_inner("perturbed", delta=0.05, seed=3)(A, b)
        assert np.allclose(a, c)

    def test_larger_delta_increases_the_error(self):
        A = np.array([[-2.0, 1.0], [1.0, -2.0]])
        b = np.array([1.0, 2.0])
        exact = np.linalg.solve(A, b)
        small = np.linalg.norm(get_inner("perturbed", delta=0.01)(A, b) - exact)
        large = np.linalg.norm(get_inner("perturbed", delta=0.10)(A, b) - exact)
        assert large > small


# -- Strip sweep ---------------------------------------------------------------

class TestStripSweep:

    def test_single_strip_problem_is_solved_in_one_sweep(self):
        """With one strip there is no transverse coupling, so one sweep is exact."""
        prob = PoissonLine2D(np.ones((8, 1)))
        u = np.zeros(prob.shape)
        strip_sweep(prob, u, prob.rhs(), get_inner("thomas"), WorkLog())
        assert prob.residual(u) < 1e-12

    def test_work_is_logged_once_per_strip(self):
        prob, _ = build_square_2d(8)
        w = WorkLog()
        strip_sweep(prob, np.zeros(prob.shape), prob.rhs(),
                    get_inner("thomas"), w)
        assert w.solves_by_size == {8: 8}

    def test_wrong_output_shape_raises(self):
        prob, _ = build_square_2d(4)
        with pytest.raises(ValueError, match="expected"):
            strip_sweep(prob, np.zeros(prob.shape), prob.rhs(),
                        lambda A, b: np.zeros(len(b) + 1), WorkLog())

    def test_jacobi_and_gauss_seidel_differ(self):
        """
        Gauss-Seidel uses already-updated strips within the same sweep; Jacobi
        reads a frozen copy. On a coupled problem the two must diverge after a
        single sweep.
        """
        prob, _ = build_square_2d(8)
        inner = get_inner("thomas")

        u_gs = np.zeros(prob.shape)
        strip_sweep(prob, u_gs, prob.rhs(), inner, WorkLog(), jacobi=False)

        u_ja = np.zeros(prob.shape)
        strip_sweep(prob, u_ja, prob.rhs(), inner, WorkLog(), jacobi=True)

        assert not np.allclose(u_gs, u_ja)

    def test_relaxation_factor_scales_the_update(self):
        prob, _ = build_square_2d(8)
        inner = get_inner("thomas")

        full = np.zeros(prob.shape)
        strip_sweep(prob, full, prob.rhs(), inner, WorkLog(), omega=1.0,
                    jacobi=True)

        half = np.zeros(prob.shape)
        strip_sweep(prob, half, prob.rhs(), inner, WorkLog(), omega=0.5,
                    jacobi=True)

        # From a zero start, omega scales the first Jacobi update exactly.
        assert np.allclose(half, 0.5 * full)

    def test_works_unchanged_in_three_dimensions(self):
        prob, _ = build_cube_3d(8)
        w = WorkLog()
        u = np.zeros(prob.shape)
        strip_sweep(prob, u, prob.rhs(), get_inner("thomas"), w)
        assert w.solves_by_size == {8: 64}      # one solve per (j, k) pair
        assert np.all(np.isfinite(u))

    def test_periodic_axis_couples_across_the_wraparound(self):
        prob, _ = build_periodic_3d(8)
        u = np.zeros(prob.shape)
        strip_sweep(prob, u, prob.rhs(), get_inner("thomas"), WorkLog())
        assert np.all(np.isfinite(u))


# -- Stationary schemes --------------------------------------------------------

class TestSolveStationary:

    def test_rejects_unknown_update(self):
        prob, _ = build_square_2d(4)
        with pytest.raises(ValueError, match="update must be"):
            solve_stationary(prob, get_inner("thomas"), update="nonsense")

    def test_rejects_unknown_criterion(self):
        prob, _ = build_square_2d(4)
        with pytest.raises(ValueError, match="criterion must be"):
            solve_stationary(prob, get_inner("thomas"), criterion="nonsense")

    def test_converges_to_the_discrete_solution(self):
        prob, _ = build_square_2d(8)
        res = solve_stationary(prob, get_inner("thomas"), tol=1e-12,
                               max_iter=5000, patience=5001)
        assert res.converged
        assert prob.residual(res.u) < 1e-10

    def test_jacobi_delta_reproduces_the_original_loop_exactly(self):
        """
        The reproducibility guarantee, verified against a loop written here from
        first principles rather than against another implementation.

        The original scheme built every strip of the new iterate from the
        previous one and stopped on max|u^{n+1} − u^n| < tol. Results published
        with that loop must remain reachable, so any divergence — in the field,
        the iteration count or the stopping point — is a regression.
        """
        prob, _ = build_square_2d(8)
        tol, max_iter = 1e-8, 500

        # -- Reference: an explicit line-Jacobi loop over the strip systems. ---
        A = prob.row_matrix()
        rhs = prob.rhs()
        Nx, Ny = prob.shape
        inv_dy2 = 1.0 / prob.dy**2

        u_ref = np.zeros((Nx, Ny))
        ref_iters = 0
        for _ in range(max_iter):
            u_new = np.zeros((Nx, Ny))
            for j in range(Ny):
                b = rhs[:, j].copy()
                if j > 0:
                    b -= u_ref[:, j - 1] * inv_dy2
                if j < Ny - 1:
                    b -= u_ref[:, j + 1] * inv_dy2
                u_new[:, j] = np.linalg.solve(A, b)
            delta = np.max(np.abs(u_new - u_ref))
            u_ref = u_new
            ref_iters += 1
            if delta < tol:
                break

        # -- The package, driven in its reproducibility mode. ------------------
        res = solve(prob, inner="thomas", scheme="jacobi",
                    tol=tol, max_iter=max_iter, patience=max_iter + 1)

        assert res.n_outer == ref_iters
        assert res.converged
        assert np.max(np.abs(res.u - u_ref)) < 1e-12

    def test_jacobi_scheme_defaults_to_the_delta_criterion(self):
        prob, _ = build_square_2d(8)
        res = solve(prob, inner="thomas", scheme="jacobi", tol=1e-8,
                    max_iter=500, patience=501)
        assert res.diagnostics["criterion"] == "delta"
        assert res.diagnostics["update"] == "jacobi"
        assert res.diagnostics["omega"] == 1.0

    def test_delta_criterion_overstates_convergence(self):
        """
        For a stationary scheme the true error exceeds the iterate difference by
        1/(1 − rho) = O(N), so a run reporting delta < tol still carries a
        materially larger residual. This is why "residual" is the honest default
        and "delta" is retained only for backward comparability.
        """
        prob, _ = build_square_2d(16)
        tol = 1e-8
        res = solve(prob, inner="thomas", scheme="jacobi", tol=tol,
                    max_iter=5000, patience=5001)
        assert res.converged
        assert res.residual > tol

    def test_residual_criterion_is_stricter_than_delta(self):
        prob, _ = build_square_2d(8)
        common = dict(inner="thomas", scheme="jacobi", tol=1e-8,
                      max_iter=5000, patience=5001)
        by_delta = solve(prob, criterion="delta", **common)
        by_residual = solve(prob, criterion="residual", **common)
        assert by_residual.n_outer > by_delta.n_outer
        assert by_residual.residual < by_delta.residual

    def test_optimal_sor_beats_gauss_seidel(self):
        prob, _ = build_square_2d(16)
        common = dict(inner="thomas", tol=1e-10, max_iter=5000, patience=5001)
        sor = solve(prob, scheme="sor", **common)
        gs = solve(prob, scheme="gauss-seidel", **common)
        assert sor.converged and gs.converged
        assert sor.n_outer < gs.n_outer

    def test_stagnation_is_reported_as_such_not_as_convergence(self):
        """
        A run halted at its inner solver's error floor has not converged. It
        must say so, so the result is not mistaken for a converged one.
        """
        prob, _ = build_square_2d(16)
        res = solve(prob, inner="perturbed", inner_options={"delta": 0.05},
                    scheme="sor", tol=1e-14, max_iter=3000, patience=20)
        assert res.stop_reason == "stagnated"
        assert not res.converged
        assert np.isfinite(res.diagnostics["residual_floor"])

    def test_max_iter_is_respected(self):
        prob, _ = build_square_2d(16)
        res = solve(prob, inner="thomas", scheme="jacobi", tol=1e-16,
                    max_iter=7, patience=8)
        assert res.n_outer == 7
        assert res.stop_reason == "max_iter"
        assert not res.converged

    def test_initial_guess_reduces_the_iteration_count(self):
        prob, _ = build_square_2d(8)
        common = dict(inner="thomas", scheme="jacobi", tol=1e-10,
                      max_iter=5000, patience=5001)
        cold = solve(prob, **common)
        warm = solve(prob, u0=cold.u, **common)
        assert warm.n_outer < cold.n_outer

    def test_callback_observes_every_iteration(self):
        prob, _ = build_square_2d(8)
        seen = []
        solve(prob, inner="thomas", scheme="jacobi", tol=1e-8, max_iter=20,
              patience=21, callback=lambda it, u, r, d: seen.append(it))
        assert seen == list(range(1, len(seen) + 1))

    def test_diagnostics_are_populated(self):
        prob, _ = build_square_2d(8)
        res = solve(prob, inner="thomas", scheme="sor", tol=1e-10,
                    max_iter=2000, patience=2001)
        assert res.diagnostics["kappa_row"] == pytest.approx(prob.kappa_row())
        assert res.diagnostics["inner_calls"] > 0

    def test_convergence_factor_is_a_geometric_mean(self):
        prob, _ = build_square_2d(8)
        res = solve(prob, inner="thomas", scheme="jacobi", tol=1e-10,
                    max_iter=5000, patience=5001)
        assert 0.0 < res.convergence_factor < 1.0


class TestOptimalOmega:

    def test_lies_within_the_convergent_range(self):
        for N in (4, 8, 16, 32, 64):
            assert 1.0 <= optimal_omega(N, N) < 2.0

    def test_increases_towards_two_with_resolution(self):
        values = [optimal_omega(N, N) for N in (8, 16, 32, 64)]
        assert all(a < b for a, b in zip(values, values[1:]))

    def test_generalises_to_three_axes(self):
        assert 1.0 <= optimal_omega(16, 16, 16) < 2.0


# -- Multigrid -----------------------------------------------------------------

class TestTransferOperators:

    def test_interpolation_shape(self):
        assert interpolation_1d(16, 8, 1.0).shape == (16, 8)

    def test_interpolation_is_non_negative_and_bounded(self):
        P = interpolation_1d(16, 8, 1.0)
        assert np.all(P >= 0.0)
        assert np.all(P.sum(axis=1) <= 1.0 + 1e-12)

    def test_interpolation_reproduces_a_constant_in_the_interior(self):
        """
        Boundary nodes enter the stencil with value zero, which is correct
        because coarse levels always carry homogeneous data. Interior fine
        nodes, away from that influence, must reproduce a constant exactly.
        """
        P = interpolation_1d(32, 16, 1.0)
        interpolated = P @ np.ones(16)
        assert np.allclose(interpolated[8:24], 1.0)

    def test_restriction_rows_are_normalised(self):
        """
        The single easiest thing to get wrong. An unnormalised transpose
        under-weights the coarse residual, and the V-cycle then degrades to a
        convergence factor that grows with N — which looks exactly like
        "multigrid does not work on this problem".
        """
        R = restriction_from(interpolation_1d(16, 8, 1.0))
        assert np.allclose(R.sum(axis=1), 1.0)

    def test_periodic_interpolation_is_nested(self):
        """
        A periodic axis is discretised without boundary nodes, so coarsening is
        exactly nested: coarse point I coincides with fine point 2I.
        """
        P = interpolation_1d_periodic(16, 8)
        assert P.shape == (16, 8)
        for I in range(8):
            assert P[2 * I, I] == pytest.approx(1.0)

    def test_periodic_interpolation_reproduces_a_constant_everywhere(self):
        """With no boundary, every fine node — including the wraparound — sees
        a full stencil, so a constant is reproduced exactly across the axis."""
        P = interpolation_1d_periodic(16, 8)
        assert np.allclose(P @ np.ones(8), 1.0)


class TestHierarchy:

    def test_levels_halve_until_the_minimum_strip(self):
        prob, _ = build_square_2d(32)
        shapes = [lv.problem.shape for lv in build_hierarchy(prob)]
        assert shapes == [(32, 32), (16, 16), (8, 8), (4, 4)]

    def test_max_levels_caps_the_depth(self):
        prob, _ = build_square_2d(32)
        assert len(build_hierarchy(prob, max_levels=2)) == 2

    def test_conditioning_does_not_degrade_with_depth(self):
        """
        Because both directions are coarsened together, dx/dy is preserved and
        κ stays bounded on every level. The QSVT polynomial degree and the HHL
        clock register are therefore constant across the hierarchy.
        """
        prob, _ = build_square_2d(64)
        for level in build_hierarchy(prob):
            assert 1.0 < level.problem.kappa_row() < 3.0

    def test_transfer_operators_are_attached_to_every_level_but_the_last(self):
        prob, _ = build_square_2d(32)
        levels = build_hierarchy(prob)
        assert all(lv.P is not None and lv.R is not None for lv in levels[:-1])
        assert levels[-1].P is None

    def test_restrict_and_prolong_round_trip_shapes(self):
        prob, _ = build_square_2d(16)
        level = build_hierarchy(prob)[0]
        fine = np.ones((16, 16))
        coarse = level.restrict(fine)
        assert coarse.shape == (8, 8)
        assert level.prolong(coarse).shape == (16, 16)

    def test_three_dimensional_hierarchy(self):
        prob, _ = build_cube_3d(16)
        shapes = [lv.problem.shape for lv in build_hierarchy(prob)]
        assert shapes == [(16, 16, 16), (8, 8, 8), (4, 4, 4)]


class TestMultigridSolve:

    def test_refuses_to_run_without_a_coarse_level(self):
        """
        Multigrid never silently degrades to a stationary scheme: a silent
        fallback would quietly restore the O(N) iteration count the user was
        trying to escape.
        """
        prob, _ = build_square_2d(4)
        with pytest.raises(ValueError, match="cannot be coarsened"):
            solve(prob, inner="thomas", scheme="fmg")

    def test_v_cycle_converges(self):
        prob, _ = build_square_2d(16)
        res = solve(prob, inner="thomas", scheme="multigrid", tol=1e-10,
                    max_cycles=100, patience=101)
        assert res.converged
        assert prob.residual(res.u) < 1e-9

    def test_fmg_converges(self):
        prob, _ = build_square_2d(16)
        res = solve(prob, inner="thomas", scheme="fmg", tol=1e-10,
                    max_cycles=100, patience=101)
        assert res.converged
        assert prob.residual(res.u) < 1e-9

    def test_iteration_count_is_grid_independent(self):
        """
        The defining property of multigrid, and the reason it is the default:
        the cycle count must not grow with N, whereas every stationary scheme
        is O(N).
        """
        counts = []
        for N in (16, 32, 64):
            prob, _ = build_square_2d(N)
            res = solve(prob, inner="thomas", scheme="fmg", tol=1e-9,
                        max_cycles=100, patience=101)
            assert res.converged
            counts.append(res.n_outer)
        assert max(counts) - min(counts) <= 3

    def test_multigrid_costs_far_fewer_strip_solves_than_sor(self):
        prob, _ = build_square_2d(64)
        mg = solve(prob, inner="thomas", scheme="fmg", tol=1e-9,
                   max_cycles=100, patience=101)
        sor = solve(prob, inner="thomas", scheme="sor", tol=1e-9,
                    max_iter=5000, patience=5001)
        assert mg.converged and sor.converged
        assert mg.work.total < sor.work.total

    def test_multigrid_tolerates_inner_error_far_better_than_sor(self):
        """
        The error of a converged iterate is amplified by ~1/(1 − rho). Optimal
        SOR is deliberately run with rho close to 1, so it amplifies strip error
        in proportion to N; multigrid has rho ~ 0.13 independently of N. This is
        the central argument for multigrid with a quantum inner solver.
        """
        prob, _ = build_square_2d(32)
        reference = solve(prob, inner="thomas", scheme="fmg", tol=1e-12,
                          max_cycles=200, patience=201).u

        opts = {"delta": 0.002}
        mg = solve(prob, inner="perturbed", inner_options=opts, scheme="fmg",
                   tol=1e-9, max_cycles=60, patience=61)
        sor = solve(prob, inner="perturbed", inner_options=opts, scheme="sor",
                    tol=1e-9, max_iter=3000, patience=3001)

        scale = np.max(np.abs(reference))
        err_mg = np.max(np.abs(mg.u - reference)) / scale
        err_sor = np.max(np.abs(sor.u - reference)) / scale
        assert err_mg < err_sor

    def test_diagnostics_record_the_hierarchy(self):
        prob, _ = build_square_2d(16)
        res = solve(prob, inner="thomas", scheme="fmg", tol=1e-10,
                    max_cycles=50, patience=51)
        assert res.diagnostics["n_levels"] >= 2
        assert len(res.diagnostics["level_shapes"]) == res.diagnostics["n_levels"]
        assert all(k < 3.0 for k in res.diagnostics["level_kappas"])

    def test_coarse_levels_carry_most_of_the_work(self):
        """Multigrid performs most of its solves on cheap, short strips."""
        prob, _ = build_square_2d(32)
        res = solve(prob, inner="thomas", scheme="fmg", tol=1e-9,
                    max_cycles=60, patience=61)
        finest = max(res.work.solves_by_size)
        coarse_solves = sum(k for n, k in res.work.solves_by_size.items()
                            if n < finest)
        assert coarse_solves > res.work.solves_by_size[finest]

    def test_solves_the_three_dimensional_cube(self):
        prob, u_exact = build_cube_3d(16)
        res = solve(prob, inner="thomas", scheme="fmg", tol=1e-9,
                    max_cycles=100, patience=101)
        assert res.converged
        rel = np.max(np.abs(res.u - u_exact)) / np.max(np.abs(u_exact))
        assert rel < 0.05          # limited by O(h²) truncation, not the solve

    def test_wall_time_budget_is_honoured(self):
        """
        Stagnation detection bounds the iteration count, not the cost per
        iteration, so a hard wall-clock budget is the only guard against a
        solver whose per-strip cost is simply large.
        """
        prob, _ = build_square_2d(32)
        res = solve_multigrid(prob, get_inner("thomas"), tol=1e-16,
                              max_cycles=10000, patience=10001, max_wall_s=0.0)
        assert res.stop_reason == "wall_time_exceeded"
        assert not res.converged
        assert res.u.shape == prob.shape


# -- Entry points --------------------------------------------------------------

class TestSolveEntryPoint:

    def test_every_registered_scheme_runs(self):
        prob, _ = build_square_2d(16)
        for scheme in available_schemes():
            kwargs = ({"max_cycles": 50, "patience": 51}
                      if scheme in ("multigrid", "fmg")
                      else {"max_iter": 3000, "patience": 3001})
            res = solve(prob, inner="thomas", scheme=scheme, tol=1e-8, **kwargs)
            assert isinstance(res, OuterResult)
            assert res.converged, f"scheme {scheme} failed to converge"

    def test_unknown_scheme_raises_and_lists_alternatives(self):
        prob, _ = build_square_2d(8)
        with pytest.raises(ValueError, match="Unknown scheme"):
            solve(prob, scheme="quantum_annealing")

    def test_accepts_a_bare_callable_as_the_inner_solver(self):
        prob, _ = build_square_2d(8)
        res = solve(prob, inner=lambda A, b: np.linalg.solve(A, b),
                    scheme="jacobi", tol=1e-10, max_iter=5000, patience=5001)
        assert res.converged

    def test_inner_config_selects_the_relevant_section(self):
        prob, _ = build_square_2d(8)
        cfg = InnerConfig(perturbed={"delta": 0.0}, qsvt={"max_degree": 9999})
        res = solve(prob, inner="perturbed", inner_options=cfg,
                    scheme="jacobi", tol=1e-10, max_iter=5000, patience=5001)
        assert res.converged

    def test_describe_scheme_lists_tunable_parameters(self):
        text = describe_scheme("sor")
        assert "omega" in text and "tol" in text

    def test_result_string_is_informative(self):
        prob, _ = build_square_2d(8)
        res = solve(prob, inner="thomas", scheme="jacobi", tol=1e-8,
                    max_iter=500, patience=501)
        text = str(res)
        assert "thomas" in text and "converged" in text


class TestSolveStaged:

    def test_requires_at_least_one_stage(self):
        prob, _ = build_square_2d(8)
        with pytest.raises(ValueError, match="at least one stage"):
            solve_staged(prob, [], inner="thomas")

    def test_combines_work_and_history_across_stages(self):
        prob, _ = build_square_2d(16)
        res = solve_staged(
            prob,
            [("fmg", {"tol": 1e-6, "max_cycles": 50, "patience": 51}),
             ("jacobi", {"tol": 1e-9, "max_iter": 200, "patience": 201,
                         "criterion": "residual"})],
            inner="thomas",
        )
        assert res.diagnostics["stages"] == ["fmg", "jacobi"]
        assert res.n_outer == len(res.residual_history)
        assert res.work.total > 0
        assert "->" in res.scheme
