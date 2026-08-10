"""
test_poisson_1d_4th.py
----------------------
Tests for the fourth-order 1D Poisson discretisation, `problems/poisson_1d_4th.py`.

The subject of these tests is the *boundary closure*, not the interior stencil.
The five-point interior stencil is standard and was never in doubt; the ghost-node
elimination at the two boundary rows was wrong in two independent ways, and the
error escaped detection for as long as it did because the class had only ever been
exercised on `poisson_1d_fS_hom`, whose solution −sin(πx)/π² is odd about *both*
boundaries — precisely the configuration in which the defective closure happens to
be exact.

The two defects, both now fixed and both pinned below:

  * the boundary node contributes +16α and the ghost −2α, which subtract to 14α;
    summing them as 18α leaves an O(1) residual on rows 0 and N−1 and destroys
    convergence outright whenever α ≠ 0;
  * the odd reflection u₋₁ = 2α − u₁ is only O(h²) accurate. The omitted term is
    h²·u″(0), which the governing equation supplies as h²·f(0) at no cost; without
    it the scheme is capped at second order.

The tests are therefore built around solutions that the defective closure could not
have passed:

  * a cubic, on which the stencil is exact, so any residual error is attributable
    to the closure alone;
  * a solution with non-zero Dirichlet data, which the 18α defect breaks completely.

Two further properties are pinned because downstream work depends on them. Both
corrections are right-hand-side only, so `A` — and hence κ(A), the QSVT phase-angle
cache keys and the block encodings built from it — must be bit-identical to what the
defective implementation produced. And `f(0)`/`f(1)` must reach the class: they are
required data for the closure, not a refinement.

All tests are purely classical and run in milliseconds.
"""
from __future__ import annotations

import numpy as np
import pytest

from core import cases
from problems.poisson_1d_4th import PoissonProblem1D4th


RESOLUTIONS = (4, 8, 16, 32, 64)


# ── Manufactured solutions ────────────────────────────────────────────────────
#
# The sign convention is that of the discretisation: A/(12h²) approximates the
# second derivative, so the governing equation is u″ = f, not −u″ = f.

def _u_sin(x):  return np.sin(np.pi * x)
def _f_sin(x):  return -np.pi**2 * np.sin(np.pi * x)

def _u_cubic(x): return x * (1.0 - x) * (2.0 - x)
def _f_cubic(x): return -6.0 + 6.0 * x

def _u_exp(x):  return np.exp(x)
def _f_exp(x):  return np.exp(x)


def _solve(N, f, alpha, beta, f_boundary):
    """Assemble at resolution N and solve densely, returning (u_h, u_exact, h)."""
    x = np.arange(1, N + 1) / (N + 1)
    prob = PoissonProblem1D4th(N=N, f_vals=f(x), alpha=alpha, beta=beta,
                               f_boundary=f_boundary)
    return np.linalg.solve(prob.A, prob.b), prob.x, prob.dx


def _observed_order(f, u, alpha, beta, supply_boundary=True):
    """
    Least-squares order of convergence in the maximum norm over RESOLUTIONS.

    Returns ``float("inf")`` when the scheme is exact to round-off on every
    mesh, which is the expected outcome for a cubic solution.
    """
    fb = (float(f(np.array([0.0]))[0]), float(f(np.array([1.0]))[0])) \
        if supply_boundary else None
    errs, hs = [], []
    for N in RESOLUTIONS:
        u_h, x, h = _solve(N, f, alpha, beta, fb)
        errs.append(float(np.max(np.abs(u_h - u(x)))))
        hs.append(h)
    if max(errs) < 1e-12:
        return float("inf")
    p = np.polyfit(np.log(hs), np.log(errs), 1)
    return float(p[0])


# ── Order of convergence ──────────────────────────────────────────────────────

class TestOrderOfConvergence:

    def test_sinusoid_homogeneous(self):
        """
        The historical regression case: odd about both boundaries.

        This one passed even with the defective closure, so it is pinned to
        guard against a fix that trades the boundary rows for the interior.
        """
        assert _observed_order(_f_sin, _u_sin, 0.0, 0.0) == pytest.approx(4.0, abs=0.15)

    def test_cubic_isolates_the_boundary_closure(self):
        """
        The five-point stencil is exact on cubics, so the *only* possible source
        of error is the boundary closure. The defective implementation returned
        second order here; a correct closure must return the exact solution to
        round-off.
        """
        for N in RESOLUTIONS:
            u_h, x, _ = _solve(N, _f_cubic, 0.0, 0.0,
                               (float(_f_cubic(np.array([0.0]))[0]),
                                float(_f_cubic(np.array([1.0]))[0])))
            assert np.max(np.abs(u_h - _u_cubic(x))) < 1e-12

    def test_non_zero_dirichlet_data(self):
        """
        u = eˣ with u(0) = 1, u(1) = e. The 18α defect made this diverge — the
        measured order was −0.01, i.e. no convergence at all — so this is the
        test that would have caught it.
        """
        order = _observed_order(_f_exp, _u_exp, 1.0, float(np.e))
        assert order == pytest.approx(4.0, abs=0.2)

    def test_extrapolated_boundary_source_preserves_order(self):
        """
        Where a case cannot supply f(0)/f(1) the class extrapolates them. The
        extrapolant is cubic and hence O(h⁴) accurate, so the order must
        survive; a linear or constant fallback would not do, since the term
        carries an O(1) weight in the boundary row.
        """
        order = _observed_order(_f_exp, _u_exp, 1.0, float(np.e),
                                supply_boundary=False)
        assert order == pytest.approx(4.0, abs=0.3)


# ── Consistency of the assembled system ───────────────────────────────────────

class TestBoundaryRowConsistency:

    @pytest.mark.parametrize("N", RESOLUTIONS)
    def test_boundary_rows_are_not_o1_inconsistent(self, N):
        """
        Residual of the exact solution, A·u_exact − b, normalised by ‖b‖∞.

        The signature of the 18α defect is an O(1) residual confined to rows 0
        and N−1 whilst every interior row sits at machine precision — 4α at the
        near boundary and 4β at the far one. Both boundary rows must now sit at
        the truncation level of the scheme.
        """
        x = np.arange(1, N + 1) / (N + 1)
        alpha, beta = 1.0, float(np.e)
        prob = PoissonProblem1D4th(
            N=N, f_vals=_f_exp(x), alpha=alpha, beta=beta,
            f_boundary=(float(_f_exp(np.array([0.0]))[0]),
                        float(_f_exp(np.array([1.0]))[0])))
        r = prob.A @ _u_exp(prob.x) - prob.b
        scale = max(float(np.max(np.abs(prob.b))), 1.0)

        assert abs(r[0]) / scale < 1e-4
        assert abs(r[-1]) / scale < 1e-4

    def test_correction_is_right_hand_side_only(self):
        """
        `A` must not depend on the Dirichlet data or on the boundary source.

        This is what keeps κ(A) — and therefore the QSVT phase-angle cache keys
        and `hpc/jobs/submit_precompute_4th.sh` — valid across the fix. It is
        also what preserves the symmetry that `build_dense_block_encoding` and
        `PentadiagonalToeplitz` both require.
        """
        base = PoissonProblem1D4th(N=16, source_fn="fS")
        moved = PoissonProblem1D4th(N=16, source_fn="fS",
                                    alpha=3.7, beta=-2.1, f_boundary=(5.0, -9.0))

        assert np.array_equal(base.A, moved.A)
        assert base.kappa == moved.kappa
        assert np.array_equal(base.A, base.A.T)
        assert not np.allclose(base.b, moved.b)


# ── Boundary source resolution ────────────────────────────────────────────────

class TestBoundarySourceResolution:

    def test_evaluated_exactly_from_source_fn(self):
        """f_S(x) = sin(πx) vanishes at both boundaries."""
        prob = PoissonProblem1D4th(N=8, source_fn="fS")
        assert prob.f_boundary[0] == pytest.approx(0.0, abs=1e-15)
        assert prob.f_boundary[1] == pytest.approx(0.0, abs=1e-15)

    def test_explicit_values_take_precedence(self):
        prob = PoissonProblem1D4th(N=8, f_vals=np.zeros(8), f_boundary=(2.0, -3.0))
        assert prob.f_boundary == (2.0, -3.0)

    def test_extrapolation_is_exact_on_cubics(self):
        """
        The fallback extrapolant is cubic, so it must reproduce a cubic source
        exactly — the property that makes it O(h⁴) accurate in general.
        """
        N = 16
        x = np.arange(1, N + 1) / (N + 1)
        def f(t): return 1.0 - 2.0 * t + 3.0 * t**2 - 4.0 * t**3
        prob = PoissonProblem1D4th(N=N, f_vals=f(x))

        assert prob.f_boundary[0] == pytest.approx(f(0.0), rel=1e-10)
        assert prob.f_boundary[1] == pytest.approx(f(1.0), rel=1e-10)


# ── Integration with the case registry ────────────────────────────────────────

class TestCaseRegistryBoundarySource:
    """
    `BuiltCase.f_boundary` carries the values the closure needs from the registry,
    where the analytical source is known, to the fourth-order problem class.

    It matters most for `het_1d_3b_gaussian_Vd300`, whose source is a sharply
    peaked Gaussian of magnitude ~10⁹ sited at 0.6 L. Its boundary values are far
    from negligible and are not recoverable by extrapolation at the coarse
    resolutions the fourth-order sweep uses.
    """

    @pytest.mark.parametrize("name", [
        "poisson_1d_fS_hom",
        "poisson_1d_fL_hom",
        "poisson_1d_fH_hom",
        "poisson_1d_fS_nonhom",
        "het_1d_3a_linear",
        "het_1d_3b_gaussian_Vd300",
    ])
    def test_populated_for_the_1d_sweep_cases(self, name):
        built = cases.get(name).build(8)
        assert built.f_boundary is not None
        assert len(built.f_boundary) == 2
        assert all(np.isfinite(v) for v in built.f_boundary)

    def test_matches_the_analytical_source(self):
        """f_L(x) = 10x, so the boundary values are 0 and 10 exactly."""
        built = cases.get("poisson_1d_fL_hom").build(8)
        assert built.f_boundary[0] == pytest.approx(0.0)
        assert built.f_boundary[1] == pytest.approx(10.0)

    def test_gaussian_boundary_values_are_not_negligible(self):
        """
        Guards the reason `f_boundary` was threaded through the registry at all:
        were these values small relative to the peak, extrapolation would have
        sufficed and the added field would be dead weight.
        """
        built = cases.get("het_1d_3b_gaussian_Vd300").build(16)
        peak = float(np.max(np.abs(built.f_values)))
        assert abs(built.f_boundary[1]) > 1e-4 * peak
