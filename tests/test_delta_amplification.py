"""
Tests for the classical claims scripts/delta_amplification_hardware.py and
scripts/qsvt_2d_line_degree_sweep.py are built on.

These are the load-bearing, fully-testable parts of Phase 6: the
discretisation-error reference, the amplification measurement, and the
row-operator conditioning claim. The hardware-measurement halves of both
scripts (core.hardware, already tested in tests/test_hardware.py) and the
real QSP angle-finding path (needs pyqsp, present in this development
environment but not asserted as a hard dependency here) are exercised by
running the scripts directly, not duplicated as unit tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pytest

from solvers.outer import solve, PoissonLine2D


# ── Analytic discretisation-error reference ───────────────────────────────────

def analytic_problem(N: int):
    x = np.arange(1, N + 1) / (N + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    f = -2 * np.pi ** 2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    return PoissonLine2D(f), u_exact


class TestDiscretizationErrorReference:
    """
    u = sin(pi x) sin(pi y) is an exact solution of the continuous PDE for
    f = -2 pi^2 sin(pi x) sin(pi y). This pins two things: that the claim is
    true (f really does produce this u under the discrete operator, in the
    N -> infinity limit) and that the measured error follows the expected
    O(h^2) scaling -- both load-bearing for treating the delta=0 solve as a
    clean discretisation-error-only reference.
    """

    @pytest.mark.parametrize("N", [8, 16, 32, 64])
    def test_follows_h_squared_scaling(self, N):
        prob, u_exact = analytic_problem(N)
        res = solve(prob, inner="thomas", scheme="fmg", tol=1e-12)
        err = np.linalg.norm(res.u - u_exact) / np.linalg.norm(u_exact)
        # err * N^2 should be roughly constant across N for true O(h^2)
        # convergence; loose bounds since this is a scaling check, not an
        # exact-constant check.
        assert 0.3 < err * N ** 2 < 1.5

    def test_error_decreases_monotonically_with_N(self):
        errors = []
        for N in (8, 16, 32, 64):
            prob, u_exact = analytic_problem(N)
            res = solve(prob, inner="thomas", scheme="fmg", tol=1e-12)
            errors.append(np.linalg.norm(res.u - u_exact) / np.linalg.norm(u_exact))
        assert errors == sorted(errors, reverse=True)


# ── Amplification measurement ─────────────────────────────────────────────────

class TestAmplificationMeasurement:
    """
    The central empirical claim scripts/delta_amplification_hardware.py
    reports: FMG's amplification of a systematic per-strip error is roughly
    N-independent, SOR's grows with N. Pinned directly, not assumed from
    solvers/outer/multigrid.py's docstring.
    """

    def _measure_vs_analytic(self, N, delta, scheme):
        """Error against the true analytic solution -- includes both
        discretisation error and amplified quantum error, which is exactly
        what scripts/delta_amplification_hardware.py's binding-constraint
        comparison needs (it explicitly wants both error sources on the
        same axis)."""
        prob, u_exact = analytic_problem(N)
        kwargs = {"tol": 1e-10}
        if scheme in ("sor", "gauss-seidel"):
            kwargs["max_iter"] = 800
        else:
            kwargs["max_cycles"] = 50
        res = solve(prob, inner="perturbed", scheme=scheme,
                    inner_options={"delta": delta}, **kwargs)
        return float(np.linalg.norm(res.u - u_exact) / np.linalg.norm(u_exact))

    def _measure_pure_amplification(self, N, delta, scheme):
        """
        Error against a numerical delta=0 reference on the same grid, not
        the analytic solution. This isolates the amplification effect from
        discretisation error, which the vs-analytic comparison does not:
        at small delta, discretisation error is comparable to or larger
        than the delta-induced error, and err_vs_analytic/delta becomes
        dominated by the (delta-independent) discretisation floor rather
        than measuring amplification at all. Confirmed directly: at N=16,
        delta=0.001, discretisation_error/delta = 2.85 while the true
        amplification (measured this way) is ~2.1 -- the vs-analytic ratio
        of 3.99 at that point is contaminated, not a real amplification
        effect. Use this helper, not _measure_vs_analytic, whenever the
        claim under test is about amplification specifically.
        """
        prob, _u_exact = analytic_problem(N)
        kwargs = {"tol": 1e-10}
        if scheme in ("sor", "gauss-seidel"):
            kwargs["max_iter"] = 800
        else:
            kwargs["max_cycles"] = 50
        ref_kwargs = dict(kwargs)
        ref_kwargs["tol"] = 1e-12
        u0 = solve(prob, inner="thomas", scheme=scheme, **ref_kwargs).u
        res = solve(prob, inner="perturbed", scheme=scheme,
                    inner_options={"delta": delta}, **kwargs)
        return float(np.linalg.norm(res.u - u0) / np.linalg.norm(u0))

    def _measure(self, N, delta, scheme):
        # Used by the N-scaling tests below, which compare against the
        # analytic solution deliberately (matching the script's own
        # methodology) at delta values large enough that discretisation
        # contamination is not the dominant effect (checked: at delta=0.005,
        # N=16, discretisation_error/delta ~ 0.57, well below the ~2.1
        # measured amplification -- clean enough for these tests' purpose).
        return self._measure_vs_analytic(N, delta, scheme)

    def test_fmg_amplification_roughly_constant_across_N(self):
        delta = 0.005
        amps = [self._measure(N, delta, "fmg") / delta for N in (8, 16, 32)]
        # "Roughly constant": max/min ratio well under the ~4x SOR shows
        # over the same range (see next test).
        assert max(amps) / min(amps) < 2.0

    def test_sor_amplification_grows_with_N(self):
        delta = 0.005
        amps = [self._measure(N, delta, "sor") / delta for N in (8, 16, 32)]
        assert amps == sorted(amps)  # monotonically increasing
        assert amps[-1] / amps[0] > 2.0  # grows meaningfully, not just noise

    def test_fmg_amplification_roughly_linear_in_delta(self):
        # If amplification is a fixed multiplier (as the 1/(1-rho) model
        # predicts), measured error should scale linearly with delta.
        # Uses _measure_pure_amplification (not vs-analytic) specifically
        # to avoid the discretisation-error contamination documented on
        # that helper -- an earlier version of this test compared against
        # the analytic solution and failed at delta=0.001 for exactly that
        # reason (measured "amplification" of 3.99x vs the true ~2.1x,
        # traced directly to discretisation_error/delta = 2.85 being
        # comparable to the measurement itself at that small delta).
        N = 16
        deltas = [0.001, 0.005, 0.01, 0.02]
        amps = [self._measure_pure_amplification(N, d, "fmg") / d for d in deltas]
        assert max(amps) / min(amps) < 1.5  # roughly constant multiplier

    def test_sor_amplification_less_linear_than_fmg(self):
        # SOR is closer to its stability boundary; its amplification
        # should vary more across a delta sweep than FMG's does.
        N = 16
        deltas = [0.001, 0.01, 0.02]
        fmg_amps = [self._measure_pure_amplification(N, d, "fmg") / d for d in deltas]
        sor_amps = [self._measure_pure_amplification(N, d, "sor") / d for d in deltas]
        fmg_spread = max(fmg_amps) / min(fmg_amps)
        sor_spread = max(sor_amps) / min(sor_amps)
        assert sor_spread > fmg_spread


class TestDivergenceDetection:
    """
    Regression guard for the bug found when a real hardware-measured delta
    (~0.165, an order of magnitude larger than the delta=0.005 this
    module's other tests use) was first tried against
    scripts/delta_amplification_hardware.py: at N=32, delta=0.165, FMG's
    residual grew monotonically over the run (confirmed: 38 -> 65 -> ... ->
    631 over 10 iterations) while solvers.outer.core.StagnationMonitor
    still classified it as "stagnated" -- its median-window test detects a
    *lack* of improvement, not active divergence, and a smooth exponential
    blow-up does not trip it. Reporting err/delta from that run as
    "amplification" presented a snapshot of an undefined process as a
    physical quantity. This class pins the fix: a late/mid residual-history
    ratio > 1.5 is treated as divergence, with amplification omitted.
    """

    def test_small_delta_is_not_flagged_as_diverged(self):
        prob, _u = analytic_problem(16)
        res = solve(prob, inner="perturbed", scheme="fmg",
                    inner_options={"delta": 0.005}, tol=1e-10, max_cycles=50)
        h = res.residual_history
        ratio = h[-1] / h[len(h) // 2]
        assert ratio < 1.5

    def test_large_delta_fmg_diverges_at_N32(self):
        # The exact case that surfaced the bug: confirmed directly that
        # this specific (N, delta, scheme) combination diverges.
        prob, _u = analytic_problem(32)
        res = solve(prob, inner="perturbed", scheme="fmg",
                    inner_options={"delta": 0.165}, tol=1e-10, max_cycles=50)
        h = res.residual_history
        ratio = h[-1] / h[len(h) // 2]
        assert ratio > 1.5, (
            "expected FMG to diverge at N=32, delta=0.165 (the case that "
            "originally surfaced the need for divergence detection); if "
            "this no longer diverges, the fix's premise should be re-checked"
        )

    def test_large_delta_sor_diverges_broadly(self):
        # SOR has a lower stability threshold than FMG (per
        # solvers/outer/multigrid.py's own docstring); confirm it diverges
        # at delta=0.165 even at the smallest N tested here.
        prob, _u = analytic_problem(8)
        res = solve(prob, inner="perturbed", scheme="sor",
                    inner_options={"delta": 0.165}, tol=1e-10, max_iter=800)
        h = res.residual_history
        ratio = h[-1] / h[len(h) // 2]
        assert ratio > 1.5

    def test_diverged_run_residual_is_not_merely_large_but_growing(self):
        # Distinguishes "large but stable" from "diverging": a stable run
        # at a bad delta can have a large absolute residual without its
        # ratio test firing. Confirms the two are genuinely different
        # phenomena, not the same thing at different scales.
        prob, _u = analytic_problem(32)
        stable = solve(prob, inner="perturbed", scheme="fmg",
                       inner_options={"delta": 0.05}, tol=1e-10, max_cycles=50)
        diverging = solve(prob, inner="perturbed", scheme="fmg",
                          inner_options={"delta": 0.165}, tol=1e-10, max_cycles=50)
        h_stable, h_diverging = stable.residual_history, diverging.residual_history
        stable_ratio = h_stable[-1] / h_stable[len(h_stable) // 2]
        diverging_ratio = h_diverging[-1] / h_diverging[len(h_diverging) // 2]
        assert stable_ratio < 1.5
        assert diverging_ratio > 1.5


# ── Row-operator conditioning claim ───────────────────────────────────────────

class TestRowOperatorConditioning:
    """
    The claim scripts/qsvt_2d_line_degree_sweep.py's docstring rests on:
    kappa(A_row) stays low (~2-3) regardless of strip length, because the
    row operator carries an extra diagonal shift from transverse coupling
    that the plain 1-D Poisson operator does not.
    """

    @pytest.mark.parametrize("Nx", [4, 8, 16, 32])
    def test_kappa_row_stays_bounded(self, Nx):
        x = np.arange(1, Nx + 1) / (Nx + 1)
        X, Y = np.meshgrid(x, x, indexing="ij")
        f = np.sin(np.pi * X) * np.sin(np.pi * Y)
        prob = PoissonLine2D(f)
        kappa = prob.kappa_row()
        assert kappa < 4.0  # the paper's "kappa -> 3" claim, with margin

    def test_kappa_row_does_not_grow_with_Nx(self):
        kappas = []
        for Nx in (4, 8, 16, 32, 64):
            x = np.arange(1, Nx + 1) / (Nx + 1)
            X, Y = np.meshgrid(x, x, indexing="ij")
            f = np.sin(np.pi * X) * np.sin(np.pi * Y)
            prob = PoissonLine2D(f)
            kappas.append(prob.kappa_row())
        # Contrast with the plain 1-D Poisson operator, whose kappa grows
        # as O(Nx^2) -- the row operator should show no such growth.
        assert max(kappas) / min(kappas) < 1.5