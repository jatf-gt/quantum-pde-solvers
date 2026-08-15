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
        """
        Validates that the work accounting mechanism correctly bins and tallies solve counts according to their strip dimensions.
        """
        w = WorkLog()
        w.add(64, 3)
        w.add(32)
        w.add(64)
        assert w.solves_by_size == {64: 4, 32: 1}
        assert w.total == 5

    def test_weighted_cost_is_in_finest_solve_units(self):
        """
        Validates the calculation of the weighted computational cost metric.
        Ensures that coarser operations are appropriately discounted based on the specified algorithmic complexity exponent.
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
        """
        Validates that an empty work log intrinsically yields a zero weighted cost.
        """
        assert WorkLog().weighted_cost(2.35) == 0.0

    def test_merge_accumulates(self):
        """
        Validates that the sequential merging of separate work logs accurately aggregates their internal solve counts across all documented dimensions.
        """
        a, b = WorkLog(), WorkLog()
        a.add(16, 2)
        b.add(16, 3)
        b.add(8, 1)
        a.merge(b)
        assert a.solves_by_size == {16: 5, 8: 1}

    def test_summary_mentions_every_size(self):
        """
        Validates the string summary generation.
        Confirms that all distinct strip sizes and their corresponding solve counts are present in the final output.
        """
        w = WorkLog()
        w.add(16, 2)
        w.add(8, 1)
        s = w.summary()
        assert "3 solves" in s and "n=16:2" in s and "n=8:1" in s


# -- Stagnation detection ------------------------------------------------------

class TestStagnationMonitor:

    def test_silent_before_the_window_fills(self):
        """
        Validates that the stagnation detector avoids premature triggers before its moving history window is completely populated.
        """
        m = StagnationMonitor(window=10)
        assert not any(m.update(1.0) for _ in range(9))

    def test_non_finite_residual_stops_immediately(self):
        """
        Validates that non-finite residual values bypass the standard window logic and immediately trigger a stagnation halt.
        """
        m = StagnationMonitor(window=10)
        assert m.update(float("nan")) is True
        assert StagnationMonitor(window=10).update(float("inf")) is True

    def test_flat_residual_is_detected_as_a_floor(self):
        """
        Validates the detection of stationary residual trajectories.
        Confirms that iteration correctly terminates when improvement drops below the acceptable fractional threshold.
        """
        m = StagnationMonitor(window=10, min_improvement=0.01)
        stopped = [m.update(1e-6) for _ in range(20)]
        assert any(stopped)

    def test_healthy_geometric_convergence_is_not_stopped(self):
        """
        Validates that healthy, albeit slow, geometric convergence avoids false positive stagnation detections over the designated evaluation window.
        """
        m = StagnationMonitor(window=10, min_improvement=0.01)
        r = 1.0
        for _ in range(60):
            r *= 0.995
            assert not m.update(r)

    def test_transient_dip_does_not_trigger_a_stop(self):
        """
        Validates the robustness of the stagnation monitor against transient non-monotone residual dips.
        Ensures that isolated statistical outliers do not artificially truncate execution.
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
        """
        Validates that the monitoring object correctly tracks and retains the minimum residual value encountered during the entire iteration history.
        """
        m = StagnationMonitor(window=5)
        for v in (1.0, 0.1, 0.5):
            m.update(v)
        assert m.best == pytest.approx(0.1)


# -- Inner solver registry -----------------------------------------------------

class TestInnerRegistry:

    def test_expected_solvers_are_registered(self):
        """
        Validates that the essential baseline computational solvers are properly exposed within the available solver registry.
        """
        assert {"thomas", "perturbed", "hhl", "vqls", "qsvt"} <= set(available_inner())

    def test_unknown_solver_raises_and_lists_alternatives(self):
        """
        Validates the error handling mechanism for invalid solver requests.
        Ensures that an appropriate exception is raised, enumerating the available options.
        """
        with pytest.raises(ValueError, match="Unknown inner solver"):
            get_inner("definitely_not_a_solver")

    def test_unknown_option_raises_rather_than_being_ignored(self):
        """
        Validates that unrecognised keyword configuration options strictly trigger a terminal exception rather than being silently absorbed.
        Maintains rigid configuration boundaries to prevent unintended resource expenditures.
        """
        with pytest.raises(ValueError, match="Unknown option"):
            get_inner("qsvt", max_degrees=500)

    def test_error_message_names_the_valid_options(self):
        """
        Validates that the invalid option exception explicitly references the specific unrecognized parameter key and hints at valid alternatives.
        """
        with pytest.raises(ValueError, match="max_degree"):
            get_inner("qsvt", max_degrees=500)

    def test_declared_defaults_are_applied(self):
        """
        Validates that solvers receive their default algorithmic configurations when explicit user overrides are omitted.
        """
        assert resolve_options("perturbed", None) == {"delta": 0.0, "seed": 0}

    def test_unset_options_defer_to_the_underlying_solver(self):
        """
        Validates that the option resolution mechanism correctly defers parameters without strict wrapper defaults directly to the underlying quantum implementation routines.
        """
        assert resolve_options("vqls", None) == {}

    def test_options_are_type_coerced(self):
        """
        Validates the string-to-numeric coercion logic.
        Ensures that configuration inputs, typically ingested from command-line arguments, are appropriately type-cast.
        """
        resolved = resolve_options("perturbed", {"delta": "0.25", "seed": "7"})
        assert resolved == {"delta": 0.25, "seed": 7}
        assert isinstance(resolved["seed"], int)

    def test_uncoercible_option_raises(self):
        """
        Validates that structural type casting failures safely raise clear, descriptive exceptions rather than bubbling up obscure underlying library errors.
        """
        with pytest.raises(ValueError, match="expected float"):
            resolve_options("perturbed", {"delta": "not_a_number"})

    def test_boolean_options_accept_string_spellings(self):
        """
        Validates the normalization of string-based boolean flags into explicit Python boolean primitives.
        """
        assert resolve_options("vqls", {"verbose": "yes"}) == {"verbose": True}
        assert resolve_options("vqls", {"verbose": "off"}) == {"verbose": False}

    def test_available_options_rejects_unknown_solver(self):
        """
        Validates that option enumeration strictly fails when requested against a nonexistent solver identifier.
        """
        with pytest.raises(ValueError, match="Unknown inner solver"):
            available_options("nope")

    def test_describe_is_introspectable(self):
        """
        Validates that the solver introspection mechanism accurately produces detailed text encompassing known parameters and configurations.
        """
        text = describe_inner("qsvt")
        assert "max_degree" in text and "epsilon" in text

    def test_inner_config_routes_per_solver_sections(self):
        """
        Validates that composite configuration objects correctly isolate and extract parameter subsets strictly pertinent to the requested inner solver.
        """
        cfg = InnerConfig(qsvt={"max_degree": 50}, hhl={"epsilon": 0.05})
        assert cfg.for_solver("qsvt") == {"max_degree": 50}
        assert cfg.for_solver("vqls") == {}


class TestThomasInnerSolver:

    def test_solves_a_tridiagonal_system_exactly(self):
        """
        Validates the structural and numerical correctness of the baseline classical Thomas algorithm against a standard tridiagonal matrix formulation.
        """
        A = np.array([[-2.0, 1.0, 0.0],
                      [1.0, -2.0, 1.0],
                      [0.0, 1.0, -2.0]])
        b = np.array([1.0, -2.0, 3.0])
        x = get_inner("thomas")(A, b)
        assert np.allclose(A @ x, b)

    def test_zero_rhs_short_circuits_without_calling_the_solver(self):
        """
        Validates the algorithmic shortcut for zero right-hand side vectors.
        Ensures solver invocation is appropriately bypassed to save overhead on trivial systems.
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
        """
        Validates the solver failover mechanism.
        Confirms that simulated backend errors cleanly fall back to a classical surrogate and that these events are tallied accurately.
        """
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
        """
        Validates that the internal solver wrapper adequately tallies aggregate execution metrics, including invocation frequency and failover incidence.
        """
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
        """
        Validates that a perturbed surrogate solver degenerates perfectly to the exact matrix solution when its synthetic error parameter is strictly zero.
        """
        A = np.array([[-2.0, 1.0], [1.0, -2.0]])
        b = np.array([1.0, 2.0])
        x = get_inner("perturbed", delta=0.0)(A, b)
        assert np.allclose(A @ x, b)

    def test_perturbation_is_deterministic(self):
        """
        Validates the temporal consistency of the perturbed solver surrogate.
        Ensures that fixed random seeds guarantee invariant numerical approximations across multiple invocations.
        """
        A = np.array([[-2.0, 1.0], [1.0, -2.0]])
        b = np.array([1.0, 2.0])
        a = get_inner("perturbed", delta=0.05, seed=3)(A, b)
        c = get_inner("perturbed", delta=0.05, seed=3)(A, b)
        assert np.allclose(a, c)

    def test_larger_delta_increases_the_error(self):
        """
        Validates that the applied perturbation surrogate scales its functional deviation monotonically with the designated error parameter.
        """
        A = np.array([[-2.0, 1.0], [1.0, -2.0]])
        b = np.array([1.0, 2.0])
        exact = np.linalg.solve(A, b)
        small = np.linalg.norm(get_inner("perturbed", delta=0.01)(A, b) - exact)
        large = np.linalg.norm(get_inner("perturbed", delta=0.10)(A, b) - exact)
        assert large > small


# -- Strip sweep ---------------------------------------------------------------

class TestStripSweep:

    def test_single_strip_problem_is_solved_in_one_sweep(self):
        """
        Validates the algebraic baseline that problems devoid of transverse coupling necessarily converge in exactly one continuous strip sweep.
        """
        prob = PoissonLine2D(np.ones((8, 1)))
        u = np.zeros(prob.shape)
        strip_sweep(prob, u, prob.rhs(), get_inner("thomas"), WorkLog())
        assert prob.residual(u) < 1e-12

    def test_work_is_logged_once_per_strip(self):
        """
        Validates the accounting integration within the discrete spatial sweep mechanism.
        Ensures that every individual line inversion is correctly recorded in the overarching work log.
        """
        prob, _ = build_square_2d(8)
        w = WorkLog()
        strip_sweep(prob, np.zeros(prob.shape), prob.rhs(),
                    get_inner("thomas"), w)
        assert w.solves_by_size == {8: 8}

    def test_wrong_output_shape_raises(self):
        """
        Validates the dimensional integrity bounds around the substituted inner solvers.
        Prevents structurally invalid vector solutions from contaminating the execution state.
        """
        prob, _ = build_square_2d(4)
        with pytest.raises(ValueError, match="expected"):
            strip_sweep(prob, np.zeros(prob.shape), prob.rhs(),
                        lambda A, b: np.zeros(len(b) + 1), WorkLog())

    def test_jacobi_and_gauss_seidel_differ(self):
        """
        Validates the computational distinction between the Jacobi and Gauss-Seidel spatial sweep patterns.
        Ensures the latter leverages sequentially updated data within the same pass.
        """
        prob, _ = build_square_2d(8)
        inner = get_inner("thomas")

        u_gs = np.zeros(prob.shape)
        strip_sweep(prob, u_gs, prob.rhs(), inner, WorkLog(), jacobi=False)

        u_ja = np.zeros(prob.shape)
        strip_sweep(prob, u_ja, prob.rhs(), inner, WorkLog(), jacobi=True)

        assert not np.allclose(u_gs, u_ja)

    def test_relaxation_factor_scales_the_update(self):
        """
        Validates the arithmetic scaling of updates under defined relaxation factors.
        Confirms that initial state modifications scale linearly with the assigned coefficient.
        """
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
        """
        Validates the generic applicability of the discrete strip sweep pattern to arbitrary multidimensional formulations.
        """
        prob, _ = build_cube_3d(8)
        w = WorkLog()
        u = np.zeros(prob.shape)
        strip_sweep(prob, u, prob.rhs(), get_inner("thomas"), w)
        assert w.solves_by_size == {8: 64}      # one solve per (j, k) pair
        assert np.all(np.isfinite(u))

    def test_periodic_axis_couples_across_the_wraparound(self):
        """
        Validates that the spatial sweep accurately spans and communicates data across periodic topological boundaries.
        """
        prob, _ = build_periodic_3d(8)
        u = np.zeros(prob.shape)
        strip_sweep(prob, u, prob.rhs(), get_inner("thomas"), WorkLog())
        assert np.all(np.isfinite(u))


# -- Stationary schemes --------------------------------------------------------

class TestSolveStationary:

    def test_rejects_unknown_update(self):
        """
        Validates error boundaries against unrecognized spatial update schemas during stationary solver configuration.
        """
        prob, _ = build_square_2d(4)
        with pytest.raises(ValueError, match="update must be"):
            solve_stationary(prob, get_inner("thomas"), update="nonsense")

    def test_rejects_unknown_criterion(self):
        """
        Validates error boundaries against undefined convergence criteria formulations within the stationary execution logic.
        """
        prob, _ = build_square_2d(4)
        with pytest.raises(ValueError, match="criterion must be"):
            solve_stationary(prob, get_inner("thomas"), criterion="nonsense")

    def test_converges_to_the_discrete_solution(self):
        """
        Validates the macroscopic mathematical convergence of the stationary iterative mechanism toward the correct discrete finite difference solution.
        """
        prob, _ = build_square_2d(8)
        res = solve_stationary(prob, get_inner("thomas"), tol=1e-12,
                               max_iter=5000, patience=5001)
        assert res.converged
        assert prob.residual(res.u) < 1e-10

    def test_jacobi_delta_reproduces_the_original_loop_exactly(self):
        """
        Validates the absolute mathematical reproducibility of the baseline Jacobi relaxation loop.
        Ensures strict parity with historically documented reference implementations.
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
        """
        Validates that the legacy Jacobi execution pathways default correctly to the historical delta-based convergence criterion.
        """
        prob, _ = build_square_2d(8)
        res = solve(prob, inner="thomas", scheme="jacobi", tol=1e-8,
                    max_iter=500, patience=501)
        assert res.diagnostics["criterion"] == "delta"
        assert res.diagnostics["update"] == "jacobi"
        assert res.diagnostics["omega"] == 1.0

    def test_delta_criterion_overstates_convergence(self):
        """
        Validates the intrinsic looseness of the delta convergence metric compared to exact mathematical residual formulations.
        """
        prob, _ = build_square_2d(16)
        tol = 1e-8
        res = solve(prob, inner="thomas", scheme="jacobi", tol=tol,
                    max_iter=5000, patience=5001)
        assert res.converged
        assert res.residual > tol

    def test_residual_criterion_is_stricter_than_delta(self):
        """
        Validates that rigorous residual-based termination rules strictly bound execution further than surrogate coordinate-delta formulations.
        """
        prob, _ = build_square_2d(8)
        common = dict(inner="thomas", scheme="jacobi", tol=1e-8,
                      max_iter=5000, patience=5001)
        by_delta = solve(prob, criterion="delta", **common)
        by_residual = solve(prob, criterion="residual", **common)
        assert by_residual.n_outer > by_delta.n_outer
        assert by_residual.residual < by_delta.residual

    def test_optimal_sor_beats_gauss_seidel(self):
        """
        Validates the expected theoretical acceleration of the Successive Over-Relaxation (SOR) protocol relative to baseline Gauss-Seidel.
        """
        prob, _ = build_square_2d(16)
        common = dict(inner="thomas", tol=1e-10, max_iter=5000, patience=5001)
        sor = solve(prob, scheme="sor", **common)
        gs = solve(prob, scheme="gauss-seidel", **common)
        assert sor.converged and gs.converged
        assert sor.n_outer < gs.n_outer

    def test_stagnation_is_reported_as_such_not_as_convergence(self):
        """
        Validates that numerical stagnation, specifically driven by algorithmic error floors, is cleanly distinguished from genuine mathematical convergence.
        """
        prob, _ = build_square_2d(16)
        res = solve(prob, inner="perturbed", inner_options={"delta": 0.05},
                    scheme="sor", tol=1e-14, max_iter=3000, patience=20)
        assert res.stop_reason == "stagnated"
        assert not res.converged
        assert np.isfinite(res.diagnostics["residual_floor"])

    def test_max_iter_is_respected(self):
        """
        Validates the absolute upper bounds enforced on iterative loops to prevent unbounded computational resource exhaustion.
        """
        prob, _ = build_square_2d(16)
        res = solve(prob, inner="thomas", scheme="jacobi", tol=1e-16,
                    max_iter=7, patience=8)
        assert res.n_outer == 7
        assert res.stop_reason == "max_iter"
        assert not res.converged

    def test_initial_guess_reduces_the_iteration_count(self):
        """
        Validates that high-fidelity initial conditions (warm starts) properly short-circuit the total required volume of iteration.
        """
        prob, _ = build_square_2d(8)
        common = dict(inner="thomas", scheme="jacobi", tol=1e-10,
                      max_iter=5000, patience=5001)
        cold = solve(prob, **common)
        warm = solve(prob, u0=cold.u, **common)
        assert warm.n_outer < cold.n_outer

    def test_callback_observes_every_iteration(self):
        """
        Validates that user-defined diagnostic callback hooks are systematically invoked across every discrete iteration cycle.
        """
        prob, _ = build_square_2d(8)
        seen = []
        solve(prob, inner="thomas", scheme="jacobi", tol=1e-8, max_iter=20,
              patience=21, callback=lambda it, u, r, d: seen.append(it))
        assert seen == list(range(1, len(seen) + 1))

    def test_diagnostics_are_populated(self):
        """
        Validates the structural completeness of the return metadata blocks, including critical diagnostic matrices such as conditioning figures.
        """
        prob, _ = build_square_2d(8)
        res = solve(prob, inner="thomas", scheme="sor", tol=1e-10,
                    max_iter=2000, patience=2001)
        assert res.diagnostics["kappa_row"] == pytest.approx(prob.kappa_row())
        assert res.diagnostics["inner_calls"] > 0

    def test_convergence_factor_is_a_geometric_mean(self):
        """
        Validates that the calculated global convergence factor adheres mathematically to an intermediate geometric progression bounded between zero and unity.
        """
        prob, _ = build_square_2d(8)
        res = solve(prob, inner="thomas", scheme="jacobi", tol=1e-10,
                    max_iter=5000, patience=5001)
        assert 0.0 < res.convergence_factor < 1.0


class TestOptimalOmega:

    def test_lies_within_the_convergent_range(self):
        """
        Validates that derived optimal relaxation coefficients sit safely within the strict theoretical stability bounds for standard Poisson grids.
        """
        for N in (4, 8, 16, 32, 64):
            assert 1.0 <= optimal_omega(N, N) < 2.0

    def test_increases_towards_two_with_resolution(self):
        """
        Validates the theoretical scaling of the optimum relaxation parameter as grid resolution increases, asymptoting cleanly toward the stability ceiling.
        """
        values = [optimal_omega(N, N) for N in (8, 16, 32, 64)]
        assert all(a < b for a, b in zip(values, values[1:]))

    def test_generalises_to_three_axes(self):
        """
        Validates the structural generalization of the optimal coefficient approximation mechanics for arbitrary three-dimensional topologies.
        """
        assert 1.0 <= optimal_omega(16, 16, 16) < 2.0


# -- Multigrid -----------------------------------------------------------------

class TestTransferOperators:

    def test_interpolation_shape(self):
        """
        Validates the precise structural matrix dimensions of the discrete grid interpolation projection operators.
        """
        assert interpolation_1d(16, 8, 1.0).shape == (16, 8)

    def test_interpolation_is_non_negative_and_bounded(self):
        """
        Validates the physical non-negativity constraints and strict upper-bounding behavior embedded within the finite difference interpolation stencils.
        """
        P = interpolation_1d(16, 8, 1.0)
        assert np.all(P >= 0.0)
        assert np.all(P.sum(axis=1) <= 1.0 + 1e-12)

    def test_interpolation_reproduces_a_constant_in_the_interior(self):
        """
        Validates the mathematical reproduction of constant background states within the inner volume during upward grid interpolation.
        """
        P = interpolation_1d(32, 16, 1.0)
        interpolated = P @ np.ones(16)
        assert np.allclose(interpolated[8:24], 1.0)

    def test_restriction_rows_are_normalised(self):
        """
        Validates the strict row normalization applied to spatial restriction operators.
        Prevents systematic weighting divergence within the down-scaling trajectory.
        """
        R = restriction_from(interpolation_1d(16, 8, 1.0))
        assert np.allclose(R.sum(axis=1), 1.0)

    def test_periodic_interpolation_is_nested(self):
        """
        Validates that topological downsampling accurately traces nested coordinate spaces along strictly periodic spatial boundaries.
        """
        P = interpolation_1d_periodic(16, 8)
        assert P.shape == (16, 8)
        for I in range(8):
            assert P[2 * I, I] == pytest.approx(1.0)

    def test_periodic_interpolation_reproduces_a_constant_everywhere(self):
        """
        Validates that complete, boundary-less periodic interpolation pathways globally reproduce constant functions without edge degradation.
        """
        P = interpolation_1d_periodic(16, 8)
        assert np.allclose(P @ np.ones(8), 1.0)


class TestHierarchy:

    def test_levels_halve_until_the_minimum_strip(self):
        """
        Validates the iterative bisection geometry utilized during full-depth multigrid hierarchy construction.
        """
        prob, _ = build_square_2d(32)
        shapes = [lv.problem.shape for lv in build_hierarchy(prob)]
        assert shapes == [(32, 32), (16, 16), (8, 8), (4, 4)]

    def test_max_levels_caps_the_depth(self):
        """
        Validates the explicit structural limit parameters guiding coarse geometric level creation.
        """
        prob, _ = build_square_2d(32)
        assert len(build_hierarchy(prob, max_levels=2)) == 2

    def test_conditioning_does_not_degrade_with_depth(self):
        """
        Validates the preservation of intrinsic condition numbers as grids become progressively coarser.
        """
        prob, _ = build_square_2d(64)
        for level in build_hierarchy(prob):
            assert 1.0 < level.problem.kappa_row() < 3.0

    def test_transfer_operators_are_attached_to_every_level_but_the_last(self):
        """
        Validates the hierarchical linking structure.
        Confirms that all interior domain levels properly mount explicit spatial transfer projection mappings.
        """
        prob, _ = build_square_2d(32)
        levels = build_hierarchy(prob)
        assert all(lv.P is not None and lv.R is not None for lv in levels[:-1])
        assert levels[-1].P is None

    def test_restrict_and_prolong_round_trip_shapes(self):
        """
        Validates the matrix compatibility symmetry between complementary restriction and prolongation mapping algorithms.
        """
        prob, _ = build_square_2d(16)
        level = build_hierarchy(prob)[0]
        fine = np.ones((16, 16))
        coarse = level.restrict(fine)
        assert coarse.shape == (8, 8)
        assert level.prolong(coarse).shape == (16, 16)

    def test_three_dimensional_hierarchy(self):
        """
        Validates the geometric bisection rules and structural scaling laws across three-dimensional computational problem definitions.
        """
        prob, _ = build_cube_3d(16)
        shapes = [lv.problem.shape for lv in build_hierarchy(prob)]
        assert shapes == [(16, 16, 16), (8, 8, 8), (4, 4, 4)]


class TestMultigridSolve:

    def test_refuses_to_run_without_a_coarse_level(self):
        """
        Validates the explicit rejection of multigrid schema configuration on non-coarsenable problem hierarchies.
        """
        prob, _ = build_square_2d(4)
        with pytest.raises(ValueError, match="cannot be coarsened"):
            solve(prob, inner="thomas", scheme="fmg")

    def test_v_cycle_converges(self):
        """
        Validates the end-to-end discrete geometric convergence properties of the standard multigrid V-cycle mechanism.
        """
        prob, _ = build_square_2d(16)
        res = solve(prob, inner="thomas", scheme="multigrid", tol=1e-10,
                    max_cycles=100, patience=101)
        assert res.converged
        assert prob.residual(res.u) < 1e-9

    def test_fmg_converges(self):
        """
        Validates the mathematical convergence limits of the structurally heavier Full Multigrid (FMG) hierarchy solver.
        """
        prob, _ = build_square_2d(16)
        res = solve(prob, inner="thomas", scheme="fmg", tol=1e-10,
                    max_cycles=100, patience=101)
        assert res.converged
        assert prob.residual(res.u) < 1e-9

    def test_iteration_count_is_grid_independent(self):
        """
        Validates the core theoretical value of the multigrid paradigm.
        Confirms that outer cycle counts exhibit strict grid independence across scaling resolutions.
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
        """
        Validates the comparative computational economy of the multigrid pipeline juxtaposed against highly optimized baseline stationary solvers.
        """
        prob, _ = build_square_2d(64)
        mg = solve(prob, inner="thomas", scheme="fmg", tol=1e-9,
                   max_cycles=100, patience=101)
        sor = solve(prob, inner="thomas", scheme="sor", tol=1e-9,
                    max_iter=5000, patience=5001)
        assert mg.converged and sor.converged
        assert mg.work.total < sor.work.total

    def test_multigrid_tolerates_inner_error_far_better_than_sor(self):
        """
        Validates the error-suppression dynamics intrinsic to multigrid schemas.
        Confirms improved resilience against the structural inaccuracies imposed by quantum sub-solvers.
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
        """
        Validates the transparency of the geometric hierarchy properties reported within the finalized algorithmic execution diagnostics.
        """
        prob, _ = build_square_2d(16)
        res = solve(prob, inner="thomas", scheme="fmg", tol=1e-10,
                    max_cycles=50, patience=51)
        assert res.diagnostics["n_levels"] >= 2
        assert len(res.diagnostics["level_shapes"]) == res.diagnostics["n_levels"]
        assert all(k < 3.0 for k in res.diagnostics["level_kappas"])

    def test_coarse_levels_carry_most_of_the_work(self):
        """
        Validates the operational workload shift downward within the hierarchy array.
        Highlights the primary utilization of inexpensive coarse grid segments.
        """
        prob, _ = build_square_2d(32)
        res = solve(prob, inner="thomas", scheme="fmg", tol=1e-9,
                    max_cycles=60, patience=61)
        finest = max(res.work.solves_by_size)
        coarse_solves = sum(k for n, k in res.work.solves_by_size.items()
                            if n < finest)
        assert coarse_solves > res.work.solves_by_size[finest]

    def test_solves_the_three_dimensional_cube(self):
        """
        Validates the comprehensive generalization of the complete multigrid cycle methodology applied to generic spatial volume challenges.
        """
        prob, u_exact = build_cube_3d(16)
        res = solve(prob, inner="thomas", scheme="fmg", tol=1e-9,
                    max_cycles=100, patience=101)
        assert res.converged
        rel = np.max(np.abs(res.u - u_exact)) / np.max(np.abs(u_exact))
        assert rel < 0.05          # limited by O(h²) truncation, not the solve

    def test_wall_time_budget_is_honoured(self):
        """
        Validates the adherence to absolute maximum computational time boundaries as hard stopping conditions.
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
        """
        Validates the end-to-end execution pathways for every registered configuration option within the central solver interface.
        """
        prob, _ = build_square_2d(16)
        for scheme in available_schemes():
            kwargs = ({"max_cycles": 50, "patience": 51}
                      if scheme in ("multigrid", "fmg")
                      else {"max_iter": 3000, "patience": 3001})
            res = solve(prob, inner="thomas", scheme=scheme, tol=1e-8, **kwargs)
            assert isinstance(res, OuterResult)
            assert res.converged, f"scheme {scheme} failed to converge"

    def test_unknown_scheme_raises_and_lists_alternatives(self):
        """
        Validates the strict argument checking and exception clarity for invalid routing commands at the primary solve facade.
        """
        prob, _ = build_square_2d(8)
        with pytest.raises(ValueError, match="Unknown scheme"):
            solve(prob, scheme="quantum_annealing")

    def test_accepts_a_bare_callable_as_the_inner_solver(self):
        """
        Validates the API's adaptability, successfully utilizing raw functional definitions passed explicitly as custom sub-solver components.
        """
        prob, _ = build_square_2d(8)
        res = solve(prob, inner=lambda A, b: np.linalg.solve(A, b),
                    scheme="jacobi", tol=1e-10, max_iter=5000, patience=5001)
        assert res.converged

    def test_inner_config_selects_the_relevant_section(self):
        """
        Validates that the parameter configuration sub-selection pipeline correctly routes dedicated variables from generalized config collections to discrete solvers.
        """
        prob, _ = build_square_2d(8)
        cfg = InnerConfig(perturbed={"delta": 0.0}, qsvt={"max_degree": 9999})
        res = solve(prob, inner="perturbed", inner_options=cfg,
                    scheme="jacobi", tol=1e-10, max_iter=5000, patience=5001)
        assert res.converged

    def test_describe_scheme_lists_tunable_parameters(self):
        """
        Validates that introspection requests cleanly document critical solver-specific adjustment variables.
        """
        text = describe_scheme("sor")
        assert "omega" in text and "tol" in text

    def test_result_string_is_informative(self):
        """
        Validates the formatting and utility of string-based summary definitions associated with complete outer execution trajectories.
        """
        prob, _ = build_square_2d(8)
        res = solve(prob, inner="thomas", scheme="jacobi", tol=1e-8,
                    max_iter=500, patience=501)
        text = str(res)
        assert "thomas" in text and "converged" in text


class TestSolveStaged:

    def test_requires_at_least_one_stage(self):
        """
        Validates the minimum structural length requirement for complex multi-stage solver orchestration routines.
        """
        prob, _ = build_square_2d(8)
        with pytest.raises(ValueError, match="at least one stage"):
            solve_staged(prob, [], inner="thomas")

    def test_combines_work_and_history_across_stages(self):
        """
        Validates the seamless integration of historical computational data, metrics, and metadata across disjoint algorithmic stages.
        """
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
