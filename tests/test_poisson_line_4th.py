"""
test_poisson_line_4th.py
------------------------
Regression cover for the fourth-order line-decomposed problems,
``problems/poisson_line_2d_4th.py`` and ``problems/poisson_line_3d_4th.py``.

Written to be load-bearing against the two defects that made every previous
fourth-order 2-D/3-D result invalid, and against a third that only appears in
more than one dimension:

1. **An even reflection at the boundary.** Folding the ghost node into A[0,1]
   rather than onto the diagonal imposes a Neumann condition on Dirichlet data.
2. **18α in place of 14α.** The boundary node contributes +16α and the ghost
   −2α; they subtract.
3. **f in place of ∂²u/∂n².** The reflection's second-derivative term is the
   normal second derivative on the face. In 1-D the PDE makes that equal to the
   source, so the 1-D derivation cannot expose the difference; in 2-D and 3-D
   ∂²u/∂n² = f − Σ_t ∂²u/∂t², and using f alone caps the scheme at order 2.

The last of these is why ``test_order_is_four_2d`` and its 3-D counterpart are
parametrised over solutions that are **not odd about the boundaries** and that
carry **non-zero Dirichlet data**. On −sin(πx)·sin(πy) — the historical test
case — all three defects are invisible: the reflection is exact and every
tangential second derivative vanishes on the faces.
"""
from __future__ import annotations

import numpy as np
import pytest

from problems.poisson_line_2d_4th import (PoissonLine2D4th, apply_axis_4th,
                                          extrapolate_face, second_difference)
from problems.poisson_line_3d_4th import PoissonLine3D4th
from solvers.outer import solve


# -- Helpers -------------------------------------------------------------------

def dense_solve(problem):
    """
    Solves the assembled system directly, by columns of ``apply``.

    Exercises ``apply`` and ``rhs`` alone, independently of the strip
    decomposition, so that a discretisation error and an outer-iteration error
    cannot be mistaken for one another.

    Parameters
    ----------
    problem : PoissonLine2D4th or PoissonLine3D4th

    Returns
    -------
    u : np.ndarray
        Solution on the interior nodes, of ``problem.shape``.
    A : np.ndarray
        The (n, n) assembled operator, n = prod(shape).
    """
    shape = tuple(problem.shape)
    n = int(np.prod(shape))
    A = np.zeros((n, n))
    e = np.zeros(shape)
    for k in range(n):
        e.flat[k] = 1.0
        A[:, k] = problem.apply(e).ravel()
        e.flat[k] = 0.0
    return np.linalg.solve(A, problem.rhs().ravel()).reshape(shape), A


def build_2d(N, u_fn, f_fn, exact_faces=True):
    """Assembles a 2-D problem on the unit square from a manufactured solution."""
    h = 1.0 / (N + 1)
    xi = np.arange(1, N + 1) * h
    X, Y = np.meshgrid(xi, xi, indexing="ij")
    ones = np.ones(N)
    faces = {}
    if exact_faces:
        faces = dict(
            f_x0=f_fn(0.0 * ones, xi), f_x1=f_fn(1.0 * ones, xi),
            f_y0=f_fn(xi, 0.0 * ones), f_y1=f_fn(xi, 1.0 * ones),
        )
    problem = PoissonLine2D4th(
        f_fn(X, Y),
        bc_x0=u_fn(0.0 * ones, xi), bc_x1=u_fn(1.0 * ones, xi),
        bc_y0=u_fn(xi, 0.0 * ones), bc_y1=u_fn(xi, 1.0 * ones),
        **faces,
    )
    return problem, u_fn(X, Y)


def build_3d(N, u_fn, f_fn, periodic=(False, False, False)):
    """Assembles a 3-D problem on the unit cube from a manufactured solution."""
    axes = [np.arange(n_) * (1.0 / N) if p else np.arange(1, N + 1) * (1.0 / (N + 1))
            for n_, p in ((N, p) for p in periodic)]
    G = np.meshgrid(*axes, indexing="ij")
    bc_lo, bc_hi, f_lo, f_hi = [], [], [], []
    for d in range(3):
        others = [k for k in range(3) if k != d]
        FG = np.meshgrid(axes[others[0]], axes[others[1]], indexing="ij")
        for end, bc_list, f_list in ((0.0, bc_lo, f_lo), (1.0, bc_hi, f_hi)):
            args = [None] * 3
            args[d] = np.full(FG[0].shape, end)
            args[others[0]], args[others[1]] = FG[0], FG[1]
            bc_list.append(u_fn(*args))
            f_list.append(f_fn(*args))
    problem = PoissonLine3D4th(f_fn(*G), periodic=periodic,
                               bc_lo=bc_lo, bc_hi=bc_hi,
                               f_lo=f_lo, f_hi=f_hi)
    return problem, u_fn(*G)


def observed_order(errors, sizes):
    """Least-restrictive order estimate: the rate over the last refinement."""
    return float(np.log(errors[-2] / errors[-1])
                 / np.log((sizes[-1] + 1) / (sizes[-2] + 1)))


# Manufactured solutions. Every entry other than the sinusoid is deliberately
# neither odd about the boundaries nor homogeneous.
EXP_2D = (lambda x, y: np.exp(x + y),
          lambda x, y: 2.0 * np.exp(x + y))
COS_2D = (lambda x, y: np.cos(2.0 * x) + y ** 2,
          lambda x, y: -4.0 * np.cos(2.0 * x) + 2.0 * np.ones_like(y))
SIN_2D = (lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y),
          lambda x, y: -2.0 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y))
CUBIC_2D = (lambda x, y: x ** 3 + y ** 3,
            lambda x, y: 6.0 * (x + y))

EXP_3D = (lambda x, y, z: np.exp(x + y + z),
          lambda x, y, z: 3.0 * np.exp(x + y + z))
SIN_3D = (lambda x, y, z: np.sin(np.pi * x) * np.sin(np.pi * y) * np.sin(np.pi * z),
          lambda x, y, z: -3.0 * np.pi ** 2 * np.sin(np.pi * x)
          * np.sin(np.pi * y) * np.sin(np.pi * z))
CUBIC_3D = (lambda x, y, z: x ** 3 + y ** 3 + z ** 3,
            lambda x, y, z: 6.0 * (x + y + z))


# -- Order of convergence ------------------------------------------------------

class TestOrderOfConvergence2D:
    """The property the whole exercise exists to obtain."""

    @pytest.mark.parametrize("case,name", [(EXP_2D, "exp(x+y)"),
                                           (COS_2D, "cos(2x)+y^2"),
                                           (SIN_2D, "sin.sin")])
    def test_order_is_four(self, case, name):
        """
        Order ≈ 4 on solutions that are not odd about the boundaries and that
        carry non-zero Dirichlet data, as well as on the sinusoid.

        Against the defective closure this fails for the first two and passes
        for the third, which is precisely why all three are here.
        """
        sizes = [8, 16, 32]
        errors = []
        for N in sizes:
            problem, exact = build_2d(N, *case)
            u, _ = dense_solve(problem)
            errors.append(np.max(np.abs(u - exact)))
        assert observed_order(errors, sizes) > 3.7, (
            f"{name}: order {observed_order(errors, sizes):.2f}, "
            f"errors {errors}")

    def test_cubic_is_machine_exact(self):
        """
        The stencil is exact on cubics and so is the reflection, so any residual
        error is algebraic rather than truncation — the sharpest available test
        of the closure, and the one that first exposed f versus ∂²u/∂n².
        """
        problem, exact = build_2d(8, *CUBIC_2D)
        u, _ = dense_solve(problem)
        assert np.max(np.abs(u - exact)) < 1e-12

    def test_beats_second_order(self):
        """Fourth order must actually be more accurate, not merely convergent."""
        from problems.poisson_line_2d import PoissonLine2D

        N = 16
        u_fn, f_fn = EXP_2D
        h = 1.0 / (N + 1)
        xi = np.arange(1, N + 1) * h
        X, Y = np.meshgrid(xi, xi, indexing="ij")
        ones = np.ones(N)
        bc = dict(bc_x0=u_fn(0.0 * ones, xi), bc_x1=u_fn(1.0 * ones, xi),
                  bc_y0=u_fn(xi, 0.0 * ones), bc_y1=u_fn(xi, 1.0 * ones))
        second = PoissonLine2D(f_fn(X, Y), **bc)
        fourth, exact = build_2d(N, u_fn, f_fn)

        e2 = np.max(np.abs(dense_solve(second)[0] - exact))
        e4 = np.max(np.abs(dense_solve(fourth)[0] - exact))
        assert e4 < e2 / 10.0

    def test_extrapolated_face_source_preserves_order(self):
        """
        The cubic-extrapolation fallback for the face source is O(h⁴) and must
        not degrade the scheme when the caller supplies no face data.
        """
        sizes = [8, 16, 32]
        errors = []
        for N in sizes:
            problem, exact = build_2d(N, *EXP_2D, exact_faces=False)
            u, _ = dense_solve(problem)
            errors.append(np.max(np.abs(u - exact)))
        assert observed_order(errors, sizes) > 3.7


class TestOrderOfConvergence3D:

    @pytest.mark.parametrize("case,name", [(EXP_3D, "exp(x+y+z)"),
                                           (SIN_3D, "triple sin")])
    def test_order_is_four(self, case, name):
        # Kept modest deliberately: the reference here is a *dense* solve of the
        # N³×N³ system, so the cost is O(N⁹). N=24 would be a 13824² factorisation
        # and minutes of suite time for no additional discrimination.
        sizes = [8, 12, 16]
        errors = []
        for N in sizes:
            problem, exact = build_3d(N, *case)
            u, _ = dense_solve(problem)
            errors.append(np.max(np.abs(u - exact)))
        assert observed_order(errors, sizes) > 3.7, (
            f"{name}: order {observed_order(errors, sizes):.2f}, "
            f"errors {errors}")

    def test_cubic_is_machine_exact(self):
        problem, exact = build_3d(8, *CUBIC_3D)
        u, _ = dense_solve(problem)
        assert np.max(np.abs(u - exact)) < 1e-12

    def test_order_is_four_with_a_periodic_axis(self):
        """
        A periodic axis has no boundary and hence no ghost node: all four
        transverse neighbours are genuine and the closure must not fire.
        """
        u_fn = lambda x, y, z: np.exp(x + y) * np.cos(2.0 * np.pi * z)
        f_fn = lambda x, y, z: ((2.0 - 4.0 * np.pi ** 2)
                                * np.exp(x + y) * np.cos(2.0 * np.pi * z))
        # Kept modest deliberately: the reference here is a *dense* solve of the
        # N³×N³ system, so the cost is O(N⁹). N=24 would be a 13824² factorisation
        # and minutes of suite time for no additional discrimination.
        sizes = [8, 12, 16]
        errors = []
        for N in sizes:
            problem, exact = build_3d(N, u_fn, f_fn,
                                      periodic=(False, False, True))
            u, _ = dense_solve(problem)
            errors.append(np.max(np.abs(u - exact)))
        assert observed_order(errors, sizes) > 3.7


# -- The strip decomposition must be the same system ---------------------------

class TestStripDecompositionConsistency:
    """
    ``row_matrix_for`` + ``transverse_terms`` must describe exactly the operator
    ``apply`` implements. If they drift apart the outer iteration converges
    smoothly to the wrong answer, which no residual check would catch: the
    residual is measured with the same drifted operator.
    """

    @pytest.mark.parametrize("scheme", ["sor", "gauss-seidel", "fmg", "multigrid"])
    def test_2d_schemes_reproduce_the_dense_solve(self, scheme):
        problem, _ = build_2d(8, *EXP_2D)
        u_dense, _ = dense_solve(problem)
        kwargs = ({"tol": 1e-12, "max_cycles": 200}
                  if scheme in ("fmg", "multigrid")
                  else {"tol": 1e-12, "max_iter": 20000})
        result = solve(problem, inner="thomas", scheme=scheme, **kwargs)
        assert result.converged
        assert np.max(np.abs(result.u - u_dense)) < 1e-9

    @pytest.mark.parametrize("scheme", ["sor", "fmg"])
    @pytest.mark.parametrize("periodic", [(False, False, False),
                                          (False, False, True)])
    def test_3d_schemes_reproduce_the_dense_solve(self, scheme, periodic):
        problem, _ = build_3d(8, *EXP_3D, periodic=periodic)
        u_dense, _ = dense_solve(problem)
        kwargs = ({"tol": 1e-12, "max_cycles": 200} if scheme == "fmg"
                  else {"tol": 1e-12, "max_iter": 4000})
        result = solve(problem, inner="thomas", scheme=scheme, **kwargs)
        assert result.converged
        assert np.max(np.abs(result.u - u_dense)) < 1e-9


# -- Operator structure --------------------------------------------------------

class TestOperatorStructure:

    def test_2d_assembled_operator_is_symmetric(self):
        """
        Symmetry is required by ``build_dense_block_encoding`` and by
        ``PentadiagonalToeplitz``. Both boundary corrections are right-hand-side
        only for exactly this reason.
        """
        problem, _ = build_2d(8, *EXP_2D)
        _, A = dense_solve(problem)
        assert np.max(np.abs(A - A.T)) == 0.0

    def test_3d_assembled_operator_is_symmetric(self):
        problem, _ = build_3d(8, *EXP_3D)
        _, A = dense_solve(problem)
        assert np.max(np.abs(A - A.T)) == 0.0

    def test_2d_strip_operators_are_symmetric_and_pentadiagonal(self):
        problem, _ = build_2d(8, *EXP_2D)
        for j in range(problem.shape[1]):
            A = problem.row_matrix_for((j,))
            assert np.allclose(A, A.T)
            assert np.all(np.triu(A, 3) == 0.0)
            assert np.all(np.tril(A, -3) == 0.0)
            assert np.any(np.diag(A, 2) != 0.0)

    def test_2d_has_exactly_two_distinct_strip_operators(self):
        """
        The count must not grow with N: one block encoding and one set of QSP
        phase angles is needed per distinct matrix.
        """
        for N in (8, 16, 32):
            problem, _ = build_2d(N, *EXP_2D)
            ids = {id(problem.row_matrix_for((j,)))
                   for j in range(problem.shape[1])}
            assert len(ids) == 2

    def test_3d_has_at_most_four_distinct_strip_operators(self):
        problem, _ = build_3d(8, *EXP_3D)
        ids = {id(problem.row_matrix_for((j, k)))
               for j in range(problem.shape[1])
               for k in range(problem.shape[2])}
        assert len(ids) == 4

    def test_3d_periodic_axis_contributes_no_boundary_operator(self):
        """A periodic axis has no boundary, so it halves the operator count."""
        problem, _ = build_3d(8, *EXP_3D, periodic=(False, False, True))
        ids = {id(problem.row_matrix_for((j, k)))
               for j in range(problem.shape[1])
               for k in range(problem.shape[2])}
        assert len(ids) == 2

    def test_boundary_strip_differs_from_interior_by_a_diagonal_shift(self):
        """
        The transverse ghost fold is +c_y on the diagonal — an *odd* reflection.
        An even one would appear in the off-diagonal instead, which is the
        defect this pins.
        """
        problem, _ = build_2d(8, *EXP_2D)
        interior = problem.row_matrix_for((3,))
        boundary = problem.row_matrix_for((0,))
        delta = boundary - interior
        cy = 1.0 / (12.0 * problem.dy ** 2)
        assert np.allclose(delta, cy * np.eye(problem.shape[0]))

    def test_kappa_is_bounded_and_small(self):
        """
        κ(A_row) stays O(1) rather than O(N²) — the property that makes the
        line decomposition tractable for the quantum inner solvers, and it must
        survive the move to fourth order.
        """
        kappas = [build_2d(N, *EXP_2D)[0].kappa_row() for N in (8, 16, 32)]
        assert all(k < 4.0 for k in kappas)
        assert kappas[-1] > kappas[0]          # increasing towards its bound

    def test_3d_kappa_is_better_than_2d(self):
        """Both transverse directions contribute to the diagonal in 3-D."""
        k2 = build_2d(8, *EXP_2D)[0].kappa_row()
        k3 = build_3d(8, *EXP_3D)[0].kappa_row()
        assert k3 < k2

    def test_transverse_terms_are_the_fourth_order_stencil(self):
        problem, _ = build_2d(8, *EXP_2D)
        c = 1.0 / (12.0 * problem.dy ** 2)
        assert problem.transverse_terms(1, 3, 8) == (
            (-2, -c), (-1, 16.0 * c), (1, 16.0 * c), (2, -c))


# -- Coarsening ----------------------------------------------------------------

class TestCoarsening:

    def test_2d_hierarchy_descends_to_the_floor(self):
        problem, _ = build_2d(32, *EXP_2D)
        shapes = []
        level = problem
        while level is not None:
            shapes.append(level.shape)
            level = level.coarsen()
        assert shapes == [(32, 32), (16, 16), (8, 8), (4, 4)]

    def test_coarse_levels_are_homogeneous(self):
        """Coarse levels carry the error equation: no source, no boundary data."""
        problem, _ = build_2d(16, *EXP_2D)
        coarse = problem.coarsen()
        assert np.all(coarse.rhs() == 0.0)
        assert np.all(coarse.f == 0.0)

    def test_3d_hierarchy_preserves_periodicity(self):
        problem, _ = build_3d(16, *EXP_3D, periodic=(False, False, True))
        coarse = problem.coarsen()
        assert coarse.periodic == (False, False, True)
        assert coarse.shape == (8, 8, 8)


# -- Validation ----------------------------------------------------------------

class TestValidation:

    def test_2d_rejects_a_grid_below_the_stencil_width(self):
        with pytest.raises(ValueError, match="at least 4"):
            PoissonLine2D4th(np.zeros((2, 8)))

    def test_2d_rejects_a_non_2d_source(self):
        with pytest.raises(ValueError, match="2-D"):
            PoissonLine2D4th(np.zeros(8))

    def test_3d_rejects_a_periodic_strip_axis(self):
        with pytest.raises(ValueError, match="non-periodic"):
            PoissonLine3D4th(np.zeros((8, 8, 8)), periodic=(True, False, False))

    def test_3d_rejects_a_grid_below_the_stencil_width(self):
        with pytest.raises(ValueError, match="at least 4"):
            PoissonLine3D4th(np.zeros((8, 2, 8)))


# -- Shared primitives ---------------------------------------------------------

class TestPrimitives:

    def test_extrapolation_is_exact_on_cubics(self):
        """
        The face-source fallback is a cubic extrapolant, so a cubic must come
        back exactly — which is what makes the cubic test above machine-exact.
        """
        h = 0.1
        x = np.arange(1, 9) * h
        g = 2.0 * x ** 3 - x + 5.0
        assert extrapolate_face(g, 0, upper=False) == pytest.approx(
            2.0 * 0.0 ** 3 - 0.0 + 5.0, abs=1e-12)
        assert extrapolate_face(g, 0, upper=True) == pytest.approx(
            2.0 * (9 * h) ** 3 - 9 * h + 5.0, abs=1e-12)

    def test_second_difference_is_exact_on_cubics(self):
        h = 0.1
        x = np.arange(1, 9) * h
        assert second_difference(x ** 3, 0, h) == pytest.approx(6.0 * x, abs=1e-9)

    def test_second_difference_of_a_constant_face_is_zero(self):
        """
        Every constant-potential boundary — which is most of the benchmark set —
        must contribute exactly nothing to the tangential correction.
        """
        g = np.full(8, 3.7)
        assert np.all(second_difference(g, 0, 0.1) == 0.0)

    def test_axis_apply_matches_a_dense_1d_operator(self):
        """
        ``apply_axis_4th`` and the assembled strip operator must implement the
        same stencil, including the ghost fold onto the diagonal.
        """
        n, h = 8, 0.1
        A = np.zeros((n, n))
        c = 1.0 / (12.0 * h ** 2)
        np.fill_diagonal(A, -30.0 * c)
        np.fill_diagonal(A[1:, :], 16.0 * c)
        np.fill_diagonal(A[:, 1:], 16.0 * c)
        np.fill_diagonal(A[2:, :], -c)
        np.fill_diagonal(A[:, 2:], -c)
        A[0, 0] += c
        A[-1, -1] += c

        rng = np.random.default_rng(0)
        u = rng.standard_normal(n)
        assert np.allclose(apply_axis_4th(u, 0, h), A @ u)

    def test_axis_apply_is_periodic_when_asked(self):
        n, h = 8, 0.1
        u = np.arange(n, dtype=float)
        out = apply_axis_4th(u, 0, h, periodic=True)
        shifted = apply_axis_4th(np.roll(u, 3), 0, h, periodic=True)
        assert np.allclose(np.roll(out, 3), shifted)
