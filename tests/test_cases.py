"""
Tests for the canonical case registry, `core/cases.py`.

These assert the structural contract of the registry and the internal
consistency of every registered case: that it builds, that its shapes match its
declared dimension, that a declared closed form actually solves the discrete
system, and that the two deliberately-preserved name collisions really do denote
different mathematics.

What is *not* tested here is equivalence with the original definition sites.
That was established once, at migration time, by a scratchpad harness comparing
the assembled (A, b) against the untouched drivers for every case at N = 4…64;
reproducing it as a permanent test would require the drivers to stay frozen,
which is the opposite of the intent. The residual checks below are the durable
guard: they verify each case against its own mathematics rather than against a
second implementation of it.
"""
import numpy as np
import pytest

from core import cases


# ── Registry Contract ─────────────────────────────────────────────────────────

class TestRegistryContract:
    """The registry's structural invariants."""

    def test_registry_is_not_empty(self):
        assert len(cases.available()) >= 20

    def test_every_dimension_is_populated(self):
        for dim in (1, 2, 3):
            assert cases.available(dim=dim), f"no cases registered for {dim}D"

    def test_every_family_is_populated(self):
        for family in ("poisson", "het"):
            assert cases.available(family=family)

    def test_get_rejects_unknown_names(self):
        # A stale legacy label must fail loudly, not resolve to something
        # plausible: silently accepting "fS" is exactly the ambiguity the
        # registry exists to remove.
        with pytest.raises(KeyError, match="Unknown case"):
            cases.get("fS")

    def test_register_rejects_a_duplicate_name(self):
        existing = cases.available()[0]
        clash = cases.Case(
            name=existing, dim=1, family="poisson",
            summary="duplicate", build=lambda N: None,
        )
        with pytest.raises(ValueError, match="already registered"):
            cases.register(clash)

    def test_register_rejects_a_non_power_of_two_resolution(self):
        bad = cases.Case(
            name="_test_bad_N", dim=1, family="poisson",
            summary="bad", build=lambda N: None, default_N=(6,),
        )
        with pytest.raises(ValueError, match="power of two"):
            cases.register(bad)

    def test_register_rejects_an_unknown_reference_strategy(self):
        bad = cases.Case(
            name="_test_bad_ref", dim=1, family="poisson",
            summary="bad", build=lambda N: None, reference="guesswork",
        )
        with pytest.raises(ValueError, match="unrecognised reference"):
            cases.register(bad)

    def test_describe_renders_every_case(self):
        table = cases.describe()
        for name in cases.available():
            assert name in table
            assert cases.get(name).name in cases.describe(name)


# ── Per-Case Structure ────────────────────────────────────────────────────────

ALL_CASES = cases.available()
CASES_1D = cases.available(dim=1)
CASES_ND = cases.available(dim=2) + cases.available(dim=3)


@pytest.mark.parametrize("name", ALL_CASES)
class TestCaseStructure:
    """Every registered case builds and reports what it declares."""

    def test_builds_at_N4(self, name):
        assert isinstance(cases.get(name).build(4), cases.BuiltCase)

    def test_shapes_match_declared_dimension(self, name):
        case = cases.get(name)
        built = case.build(4)

        assert len(built.coords) == case.dim
        assert len(built.spacings) == case.dim
        assert len(case.lengths) == case.dim
        assert len(case.periodic) == case.dim

        expected = (4,) * case.dim
        assert built.f_values.shape == expected
        for c in built.coords:
            assert c.shape == expected

    def test_condition_number_is_finite_and_at_least_one(self, name):
        built = cases.get(name).build(4)
        assert built.kappa is not None
        assert np.isfinite(built.kappa)
        assert built.kappa >= 1.0 - 1e-12

    def test_exact_solution_shape_when_present(self, name):
        case = cases.get(name)
        built = case.build(4)
        if built.exact is None:
            # Only the strategies that genuinely require a solve may omit it.
            assert case.reference in ("thomas", "fine_mesh")
        else:
            assert built.exact.shape == built.f_values.shape

    def test_declared_default_resolutions_are_powers_of_two(self, name):
        for N in cases.get(name).default_N:
            assert N > 0 and (N & (N - 1)) == 0


@pytest.mark.parametrize("name", CASES_1D)
def test_one_dimensional_cases_carry_a_system(name):
    """1D cases are solved as a matrix system directly, so they carry (A, b)."""
    built = cases.get(name).build(4)
    assert built.A is not None and built.A.shape == (4, 4)
    assert built.b is not None and built.b.shape == (4,)
    assert built.problem is None


@pytest.mark.parametrize("name", CASES_ND)
def test_higher_dimensional_cases_carry_a_line_problem(name):
    """
    2D and 3D cases are decomposed into strips by `solvers.outer`, so they carry
    a line problem rather than an assembled system.
    """
    built = cases.get(name).build(4)
    assert built.problem is not None
    assert built.A is None and built.b is None


# ── Mathematical Consistency ──────────────────────────────────────────────────

def _truncation_error(case, N):
    """
    Local truncation error of a 1D case at resolution N.

    The registry's operator is the *unscaled* TST matrix (−2 on the diagonal,
    +1 off it), with the 1/h² folded into the right-hand side, so the raw
    residual ‖A·u_exact − b‖∞ is h⁴·u⁗/12 rather than h²·u⁗/12. Dividing by h²
    recovers the error in the approximation of u″, which is the quantity that
    is O(h²) and the one worth asserting on.

    Parameters
    ----------
    case : cases.Case
        The case to evaluate.
    N : int
        Resolution.

    Returns
    -------
    float
        ‖A·u_exact − b‖∞ / h².
    """
    built = case.build(N)
    h = built.spacings[0]
    return float(np.max(np.abs(built.A @ built.exact - built.b))) / h**2


class TestOneDimensionalResiduals:
    """
    A declared closed form must actually solve the discrete system.

    For the 1D cases this is checkable directly: substituting the closed form
    into the discrete operator leaves only truncation error. This catches a
    source and a solution that have drifted apart — a defect this repository has
    seen before, in the Neumann sub-case, where the source took a physical σ and
    the reference a normalised one, so the two were not the same problem.

    Note that three of these cases have polynomial closed forms of degree ≤ 3,
    which the three-point stencil reproduces *exactly*. Their truncation error
    is therefore pure floating-point round-off and does not decrease with
    refinement; the test accounts for that rather than asserting a rate that
    does not exist.
    """

    ROUNDOFF_FLOOR = 1e-8   # below this, the stencil is exact and only noise remains

    @pytest.mark.parametrize("name", [
        n for n in ALL_CASES
        if cases.get(n).dim == 1 and cases.get(n).reference == "analytical"
    ])
    def test_truncation_error_does_not_grow_under_refinement(self, name):
        case = cases.get(name)
        coarse = _truncation_error(case, 8)
        fine = _truncation_error(case, 32)

        if max(coarse, fine) < self.ROUNDOFF_FLOOR:
            # Stencil-exact case; nothing to assert beyond smallness.
            return
        assert fine <= coarse * 1.5, (
            f"{name}: truncation error grew from {coarse:.3e} to {fine:.3e} "
            f"under 4x refinement, so source and closed form disagree"
        )

    @pytest.mark.parametrize("name,degree", [
        ("poisson_1d_fL_hom", 3),
        ("het_1d_3a_linear", 3),
        ("het_1d_linear_scaled", 3),
    ])
    def test_cubic_closed_forms_are_reproduced_exactly(self, name, degree):
        """
        The second-difference stencil is exact for polynomials up to degree 3,
        since its error term carries the fourth derivative. These cases must
        therefore sit at round-off, not at O(h²).
        """
        for N in (8, 32):
            assert _truncation_error(cases.get(name), N) < self.ROUNDOFF_FLOOR

    def test_sinusoid_truncation_error_is_second_order(self):
        """
        u = −sin(πx)/π² against u″ = sin(πx) is the cleanest O(h²) check in the
        repository: smooth source, exact closed form, non-zero fourth derivative,
        so the error is entirely the stencil's.

        The mesh is h = 1/(N+1), so refining 8 → 16 shrinks h by 17/9 rather
        than by 2, and the expected ratio is (17/9)² ≈ 3.6, not 4.
        """
        case = cases.get("poisson_1d_fS_hom")
        e = {N: _truncation_error(case, N) for N in (8, 16, 32)}
        assert e[8] / e[16] == pytest.approx((17 / 9) ** 2, rel=0.05)
        assert e[16] / e[32] == pytest.approx((33 / 17) ** 2, rel=0.05)

    def test_neumann_case_operator_is_symmetric(self):
        """
        The Neumann row is halved specifically to keep the operator symmetric.
        Without that, HHL and QSVT are not valid on it and the eigvalsh-based
        condition number would silently describe a different matrix.
        """
        built = cases.get("het_1d_3c_neumann").build(8)
        assert np.allclose(built.A, built.A.T, atol=1e-14)
        assert built.A[0, 0] == pytest.approx(-1.0)
        assert built.A[0, 1] == pytest.approx(+1.0)

    def test_neumann_case_uses_the_origin_including_grid(self):
        """h = 1/N with a node at x = 0, unlike every other 1D case."""
        case = cases.get("het_1d_3c_neumann")
        assert case.grid == "including-origin"
        built = case.build(8)
        assert built.coords[0][0] == pytest.approx(0.0)
        assert built.spacings[0] == pytest.approx(1.0 / 8)

    @pytest.mark.parametrize("name", [
        n for n in ALL_CASES
        if cases.get(n).dim == 1 and cases.get(n).grid == "interior"
    ])
    def test_interior_grid_excludes_the_boundaries(self, name):
        built = cases.get(name).build(8)
        x = built.coords[0]
        assert x[0] == pytest.approx(1.0 / 9)
        assert x[-1] == pytest.approx(8.0 / 9)


class TestStripConditioning:
    """
    The strip operator is far better conditioned than the 1D Poisson matrix,
    which is the whole reason the line decomposition makes quantum solvers
    cheaper in higher dimensions rather than dearer. κ_row → 3⁻ in 2D and → 2⁻
    in 3D as N → ∞, against O(N²) in 1D.
    """

    @pytest.mark.parametrize("name", [n for n in ALL_CASES if cases.get(n).dim == 2])
    def test_two_dimensional_strip_kappa_approaches_three(self, name):
        kappa = cases.get(name).build(16).kappa
        assert 1.0 <= kappa < 3.0

    @pytest.mark.parametrize("name", [n for n in ALL_CASES if cases.get(n).dim == 3])
    def test_three_dimensional_strip_kappa_approaches_two(self, name):
        kappa = cases.get(name).build(16).kappa
        assert 1.0 <= kappa < 2.0

    def test_one_dimensional_kappa_grows_quadratically(self):
        """κ ≈ 4(N+1)²/π² for the 1D TST operator."""
        for N in (8, 16, 32):
            kappa = cases.get("poisson_1d_fS_hom").build(N).kappa
            theory = (4.0 / np.pi**2) * (N + 1) ** 2
            assert kappa == pytest.approx(theory, rel=0.05)


class TestPreservedNameCollisions:
    """
    Two labels denoted different mathematics before consolidation. Both are kept
    under unambiguous names; these tests assert they remain genuinely distinct,
    so that a future tidy-up cannot quietly merge them.
    """

    def test_the_two_two_dimensional_sinusoids_differ(self):
        a = cases.get("poisson_2d_sin_pi").build(8).f_values
        b = cases.get("poisson_2d_fS_10sin2pi").build(8).f_values
        assert np.max(np.abs(a - b)) > 1.0

    def test_only_one_two_dimensional_sinusoid_has_a_closed_form(self):
        assert cases.get("poisson_2d_sin_pi").build(8).exact is not None
        assert cases.get("poisson_2d_fS_10sin2pi").build(8).exact is None

    def test_the_two_het_linear_profiles_differ(self):
        a = cases.get("het_1d_3a_linear").build(8)
        b = cases.get("het_1d_linear_scaled").build(8)
        assert np.max(np.abs(a.f_values - b.f_values)) > 1.0
        assert np.max(np.abs(a.exact - b.exact)) > 1e-6

    def test_the_het_radial_extents_are_recorded_as_they_stand(self):
        """
        2D still carries the legacy 20 mm radial extent whilst 3D uses the
        SPT-100 value of 15 mm derived from the channel radii. This test records
        that divergence deliberately: it must be resolved together with the
        regeneration of the 2D results, not silently.
        """
        from core import het_geometry as geom

        assert geom.L_R == pytest.approx(0.015)
        assert geom.L_R_LEGACY_2D == pytest.approx(0.020)
        assert cases.get("het_2d_mms_spt100").lengths[1] == pytest.approx(0.020)
        assert cases.get("het_3d_mms_spt100").lengths[1] == pytest.approx(0.015)


class TestPeriodicity:
    """Azimuthal periodicity is what makes the 3D HET domain a channel."""

    @pytest.mark.parametrize("name", [
        n for n in ALL_CASES
        if cases.get(n).dim == 3 and cases.get(n).family == "het"
    ])
    def test_het_3d_cases_are_azimuthally_periodic(self, name):
        assert cases.get(name).periodic == (False, False, True)

    @pytest.mark.parametrize("name", [
        n for n in ALL_CASES if cases.get(n).family == "poisson"
    ])
    def test_generic_poisson_cases_are_not_periodic(self, name):
        assert not any(cases.get(name).periodic)

    def test_periodic_axis_grid_has_no_boundary_node(self):
        """
        The azimuthal axis uses ds = L_s/N with a node at s = 0, unlike the
        Dirichlet axes which use L/(N+1) and start at one spacing in.
        """
        from core import het_geometry as geom

        built = cases.get("het_3d_mms_spt100").build(8)
        assert built.spacings[2] == pytest.approx(geom.L_S / 8)
        assert built.spacings[0] == pytest.approx(geom.L_Z / 9)
