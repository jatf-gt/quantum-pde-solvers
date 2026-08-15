"""
test_line_problems.py
---------------------
Tests for the line-decomposed problem classes, `problems/poisson_line_2d.py`
and `problems/poisson_line_3d.py`.

These two classes are the sole 2D and 3D problem types in the repository and
the concrete implementations of the `LineProblem2D` protocol that every outer
scheme is written against. Their correctness underpins every 2D and 3D result,
classical and quantum alike.

Three properties receive particular attention because a silent regression in
any of them would be difficult to attribute:

  * the operator and right-hand side agree with an independently assembled
    dense system, so the Dirichlet absorption is verified rather than assumed;
  * the strip condition number stays bounded (κ → 3⁻ in 2D, → 2⁻ in 3D), which
    is what makes the decomposition tractable for the quantum inner solvers;
  * coarsening preserves both the power-of-two strip lengths and the grid
    aspect ratio, without which the multigrid hierarchy would either break the
    quantum encoding or degrade the smoother.

All tests are purely classical and run in milliseconds.
"""
from __future__ import annotations

import numpy as np
import pytest

from conftest import build_cube_3d, build_periodic_3d, build_square_2d
from problems.poisson_line_2d import PoissonLine2D
from problems.poisson_line_3d import PoissonLine3D


# -- PoissonLine2D: operator and right-hand side -------------------------------

class TestPoissonLine2DOperator:

    def test_rejects_non_2d_source(self):
        """
        Ensures that instantiating a 2D line problem with an array of incorrect 
        dimensionality raises an appropriate ValueError, enforcing strict tensor 
        shapes.
        """
        with pytest.raises(ValueError, match="2-D"):
            PoissonLine2D(np.zeros(8))

    def test_shape_and_spacings(self):
        """
        Validates that the geometric spacings (`dx`, `dy`) and grid shapes are 
        correctly deduced from the continuous domain limits and node counts.
        """
        prob = PoissonLine2D(np.zeros((8, 4)), Lx=2.0, Ly=1.0)
        assert prob.shape == (8, 4)
        assert prob.dx == pytest.approx(2.0 / 9.0)
        assert prob.dy == pytest.approx(1.0 / 5.0)

    def test_strip_operator_is_tst(self):
        """
        Confirms that the 1D operator governing individual strips is Toeplitz, 
        symmetric, and strictly tridiagonal, structurally matching the quantum 
        encoding constraints.
        """
        prob = PoissonLine2D(np.zeros((8, 8)), Lx=1.0, Ly=1.0)
        A = prob.row_matrix()
        a = -2.0 * (1.0 / prob.dx**2 + 1.0 / prob.dy**2)
        b = 1.0 / prob.dx**2

        assert np.allclose(A.diagonal(0), a)
        assert np.allclose(A.diagonal(1), b)
        assert np.allclose(A.diagonal(-1), b)
        # Everything beyond the first off-diagonal must vanish.
        assert np.allclose(A - np.diag(A.diagonal(0))
                           - np.diag(A.diagonal(1), 1)
                           - np.diag(A.diagonal(-1), -1), 0.0)

    def test_dirichlet_data_absorbed_into_rhs(self):
        """
        Validates the correct algebraic absorption of Dirichlet boundaries into 
        the right-hand side source term, matching the −bc/h² penalty logic.
        """
        f = np.zeros((4, 4))
        prob = PoissonLine2D(f, bc_x0=1.0, bc_x1=2.0, bc_y0=3.0, bc_y1=4.0)
        r = prob.rhs()

        # Interior nodes touched by no boundary stay zero.
        assert np.allclose(r[1:-1, 1:-1], 0.0)

        # Corner nodes accumulate both contributions, so probe an edge midpoint.
        assert r[0, 1]  == pytest.approx(-1.0 / prob.dx**2)
        assert r[-1, 1] == pytest.approx(-2.0 / prob.dx**2)
        assert r[1, 0]  == pytest.approx(-3.0 / prob.dy**2)
        assert r[1, -1] == pytest.approx(-4.0 / prob.dy**2)

    def test_corner_accumulates_both_edges(self):
        """
        Verifies that nodes at the geometric corners correctly accumulate boundary 
        penalties from both intersecting exterior edges.
        """
        prob = PoissonLine2D(np.zeros((4, 4)), bc_x0=1.0, bc_y0=3.0)
        expected = -1.0 / prob.dx**2 - 3.0 / prob.dy**2
        assert prob.rhs()[0, 0] == pytest.approx(expected)

    def test_vector_boundary_data(self):
        """
        Ensures that spatially varying (vector) boundary arrays are correctly 
        mapped node-by-node along the relevant edge.
        """
        edge = np.arange(1.0, 5.0)
        prob = PoissonLine2D(np.zeros((4, 4)), bc_x0=edge)
        assert np.allclose(prob.rhs()[0, :], -edge / prob.dx**2)

    def test_apply_matches_dense_five_point_operator(self):
        """
        Confirms that the action of the decomposed `apply` precisely matches 
        an independently assembled dense 5-point Laplacian. Ensures that outer 
        scheme residuals remain consistent with inner scheme operators.
        """
        N = 5
        prob = PoissonLine2D(np.zeros((N, N)), Lx=1.0, Ly=2.0)
        A = _dense_laplacian_2d(N, N, prob.dx, prob.dy)

        rng = np.random.default_rng(0)
        u = rng.standard_normal((N, N))
        assert np.allclose(prob.apply(u), (A @ u.ravel()).reshape(N, N))

    def test_residual_vanishes_at_the_discrete_solution(self):
        """
        Verifies that evaluating the operator residual on the exact discrete 
        solution matrix yields machine zero.
        """
        N = 6
        prob, _ = build_square_2d(N)
        A = _dense_laplacian_2d(N, N, prob.dx, prob.dy)
        u = np.linalg.solve(A, prob.rhs().ravel()).reshape(N, N)
        assert prob.residual(u) < 1e-12

    def test_manufactured_solution_recovered_to_truncation_error(self):
        """
        Validates the O(h²) order of accuracy. Confirms that successive grid 
        refinements reduce the discretisation error against a continuum analytic 
        solution by a factor of roughly four.
        """
        errors = []
        for N in (8, 16, 32):
            prob, u_exact = build_square_2d(N)
            A = _dense_laplacian_2d(N, N, prob.dx, prob.dy)
            u = np.linalg.solve(A, prob.rhs().ravel()).reshape(N, N)
            errors.append(np.max(np.abs(u - u_exact)))

        # Each mesh doubling must cut the error by close to four.
        for coarse, fine in zip(errors, errors[1:]):
            assert 3.0 < coarse / fine < 5.0


# -- PoissonLine2D: conditioning and coarsening --------------------------------

class TestPoissonLine2DHierarchy:

    @pytest.mark.parametrize("N", [4, 8, 16, 32, 64])
    def test_kappa_bounded_by_three(self, N):
        """
        Ensures that the condition number of the 2D strip operator is strictly 
        bounded above by 3, preventing exponential blow-up during HHL phase 
        estimation.
        """
        assert 1.0 < PoissonLine2D(np.zeros((N, N))).kappa_row() < 3.0

    def test_kappa_increases_monotonically(self):
        """
        Verifies that the strip operator condition number monotonically increases 
        with problem size.
        """
        kappas = [PoissonLine2D(np.zeros((N, N))).kappa_row()
                  for N in (4, 8, 16, 32, 64)]
        assert all(a < b for a, b in zip(kappas, kappas[1:]))

    def test_coarsen_halves_both_directions(self):
        """
        Confirms that standard grid coarsening uniformly halves the dimension 
        of both spatial axes, returning a problem instance one hierarchical 
        level deeper.
        """
        coarse = PoissonLine2D(np.zeros((16, 16))).coarsen()
        assert coarse.shape == (8, 8)
        assert coarse.level == 1

    def test_coarsening_preserves_aspect_ratio_and_kappa(self):
        """
        Validates that grid coarsening preserves the geometric aspect ratio (dx/dy) 
        and consequently ensures the condition number limit is maintained, keeping 
        quantum operator costs bounded across the multigrid hierarchy.
        """
        fine = PoissonLine2D(np.zeros((32, 32)))
        coarse = fine.coarsen()
        assert fine.dx / fine.dy == pytest.approx(coarse.dx / coarse.dy)
        assert coarse.kappa_row() < 3.0

    def test_coarsen_stops_at_min_strip(self):
        """
        Ensures that recursive grid coarsening safely halts, returning None, 
        when the strip dimension reaches the minimally resolvable problem size.
        """
        assert PoissonLine2D(np.zeros((4, 4))).coarsen() is None

    def test_coarsen_stops_on_odd_dimension(self):
        """
        Validates that grids with odd dimensions correctly refuse to coarsen, 
        preventing malformed fractional nodal domains.
        """
        assert PoissonLine2D(np.zeros((7, 7))).coarsen() is None

    def test_coarse_levels_carry_homogeneous_data(self):
        """
        Confirms that generated coarse grids carry homogeneous boundary and source 
        data, as required for solving the residual error equation in multigrid 
        cycles.
        """
        coarse = PoissonLine2D(np.ones((16, 16)), bc_x0=5.0).coarsen()
        assert np.allclose(coarse.rhs(), 0.0)

    def test_anisotropic_grid_semi_coarsens(self):
        """
        Verifies that highly anisotropic grids selectively coarsen only along 
        the tightly coupled spatial axes, preserving convergence rates in 
        highly skewed physical domains.
        """
        # dx = 1/17 ≈ 0.059, dy = 16/17 ≈ 0.94 — a ratio far beyond 2.
        prob = PoissonLine2D(np.zeros((16, 16)), Lx=1.0, Ly=16.0)
        coarse = prob.coarsen()
        assert coarse.shape == (8, 16)

    def test_strip_lengths_stay_powers_of_two(self):
        """
        Validates that iterative coarsening perfectly preserves power-of-two 
        strip dimensions, ensuring quantum circuit encodings never encounter 
        unaligned classical sizes.
        """
        prob = PoissonLine2D(np.zeros((64, 64)))
        while prob is not None:
            n = prob.shape[0]
            assert n & (n - 1) == 0, f"strip length {n} is not a power of two"
            prob = prob.coarsen()


# -- PoissonLine3D -------------------------------------------------------------

class TestPoissonLine3D:

    def test_rejects_non_3d_source(self):
        """
        Ensures that 3D problem instantiation rejects input arrays not 
        strictly of rank 3.
        """
        with pytest.raises(ValueError, match="3-D"):
            PoissonLine3D(np.zeros((4, 4)))

    def test_rejects_periodic_strip_axis(self):
        """
        Validates that configuring the primary strip axis (axis 0) as periodic 
        is explicitly rejected, as cyclic-tridiagonal matrices cannot be 
        natively mapped onto the symmetric block encoding framework.
        """
        with pytest.raises(ValueError, match="axis 0"):
            PoissonLine3D(np.zeros((4, 4, 4)), periodic=(True, False, False))

    def test_periodic_axis_spacing_excludes_boundary_node(self):
        """
        Confirms that grid spacing calculations correctly apply L/n logic for 
        periodic dimensions (where the boundary wraps) versus L/(n+1) for 
        Dirichlet clamped boundaries.
        """
        prob = PoissonLine3D(np.zeros((8, 8, 8)), lengths=(1.0, 1.0, 1.0),
                             periodic=(False, False, True))
        assert prob.dx == pytest.approx(1.0 / 9.0)     # Dirichlet: L/(n+1)
        assert prob.dz == pytest.approx(1.0 / 8.0)     # periodic:  L/n

    @pytest.mark.parametrize("N", [4, 8, 16, 32])
    def test_kappa_bounded_by_two(self, N):
        """
        Verifies that the condition number for the 3D strip operator is strictly 
        bounded above by 2, owing to the stabilising presence of two orthogonal 
        transverse coupling dimensions.
        """
        assert 1.0 < PoissonLine3D(np.zeros((N, N, N))).kappa_row() < 2.0

    def test_apply_matches_dense_seven_point_operator(self):
        """
        Validates the action of the 3D decomposed `apply` routine against a 
        reference dense 7-point 3D finite difference matrix.
        """
        N = 4
        prob = PoissonLine3D(np.zeros((N, N, N)), lengths=(1.0, 2.0, 3.0))
        A = _dense_laplacian_3d(prob)

        rng = np.random.default_rng(1)
        u = rng.standard_normal((N, N, N))
        assert np.allclose(prob.apply(u), (A @ u.ravel()).reshape(N, N, N))

    def test_apply_matches_dense_operator_with_periodic_axis(self):
        """
        Ensures that operator application retains accuracy when the wraparound 
        coupling from periodic boundaries is introduced, verifying consistency 
        with the dense reference.
        """
        N = 4
        prob = PoissonLine3D(np.zeros((N, N, N)), lengths=(1.0, 1.0, 1.0),
                             periodic=(False, False, True))
        A = _dense_laplacian_3d(prob)

        rng = np.random.default_rng(2)
        u = rng.standard_normal((N, N, N))
        assert np.allclose(prob.apply(u), (A @ u.ravel()).reshape(N, N, N))

    def test_residual_vanishes_at_the_discrete_solution(self):
        """
        Confirms that the 3D residual precisely vanishes at the exact discrete 
        solution for a representative cube problem.
        """
        prob, _ = build_cube_3d(4)
        A = _dense_laplacian_3d(prob)
        u = np.linalg.solve(A, prob.rhs().ravel()).reshape(prob.shape)
        assert prob.residual(u) < 1e-12

    def test_periodic_manufactured_solution_is_consistent(self):
        """
        Validates the geometric and arithmetic consistency of periodic boundaries. 
        Confirms that the discrete inversion recovers the continuous analytic 
        source solution accurately across periodic slab interfaces.
        """
        prob, phi = build_periodic_3d(8)
        A = _dense_laplacian_3d(prob)
        u = np.linalg.solve(A, prob.rhs().ravel()).reshape(prob.shape)
        rel = np.max(np.abs(u - phi)) / np.max(np.abs(phi))
        assert rel < 0.10

    def test_coarsen_halves_all_isotropic_axes(self):
        """
        Ensures that uniform 3D grid coarsening symmetrically scales down all 
        spatial dimensions for an isotropic problem geometry.
        """
        prob = PoissonLine3D(np.zeros((16, 16, 16)))
        assert prob.coarsen().shape == (8, 8, 8)

    def test_coarsen_stops_at_min_strip(self):
        """
        Validates that 3D grid coarsening prevents degradation past the minimally 
        defined block size, avoiding degenerate quantum registers.
        """
        assert PoissonLine3D(np.zeros((4, 4, 4))).coarsen() is None

    def test_coarsen_preserves_periodicity(self):
        """
        Confirms that instantiated coarse hierarchies inherit and correctly 
        maintain the spatial periodicity flags defined on the finest grid.
        """
        prob = PoissonLine3D(np.zeros((16, 16, 16)),
                             periodic=(False, False, True))
        assert prob.coarsen().periodic == (False, False, True)


# -- Private helpers -----------------------------------------------------------

def _dense_laplacian_2d(Nx: int, Ny: int, dx: float, dy: float) -> np.ndarray:
    """
    Assembles the dense (Nx·Ny)² five-point Laplacian with homogeneous exterior.

    Built by explicit stencil placement rather than by Kronecker products, so it
    shares no code path with `PoissonLine2D` and constitutes a genuinely
    independent check. Unknowns are ordered row-major, matching `ravel()` on an
    (Nx, Ny) field.

    Parameters
    ----------
    Nx, Ny : int
        Interior node counts.
    dx, dy : float
        Mesh spacings.

    Returns
    -------
    A : np.ndarray
        (Nx·Ny, Nx·Ny) dense operator.
    """
    n = Nx * Ny
    A = np.zeros((n, n))
    for i in range(Nx):
        for j in range(Ny):
            k = i * Ny + j
            A[k, k] = -2.0 * (1.0 / dx**2 + 1.0 / dy**2)
            if i > 0:
                A[k, (i - 1) * Ny + j] = 1.0 / dx**2
            if i < Nx - 1:
                A[k, (i + 1) * Ny + j] = 1.0 / dx**2
            if j > 0:
                A[k, i * Ny + (j - 1)] = 1.0 / dy**2
            if j < Ny - 1:
                A[k, i * Ny + (j + 1)] = 1.0 / dy**2
    return A


def _dense_laplacian_3d(problem: PoissonLine3D) -> np.ndarray:
    """
    Assembles the dense seven-point Laplacian for a `PoissonLine3D` instance.

    Honours each axis's periodicity flag, so the same helper validates both the
    fully Dirichlet cube and the azimuthally periodic slab. Unknowns are ordered
    to match `ravel()` on the (Nx, Ny, Nz) field.

    Parameters
    ----------
    problem : PoissonLine3D
        Problem supplying the shape, spacings and periodicity.

    Returns
    -------
    A : np.ndarray
        (Nx·Ny·Nz, Nx·Ny·Nz) dense operator.
    """
    shape = problem.shape
    n = int(np.prod(shape))
    A = np.zeros((n, n))
    inv_h2 = [1.0 / h**2 for h in problem.spacings]

    strides = np.array([shape[1] * shape[2], shape[2], 1])

    for idx in np.ndindex(*shape):
        k = int(np.dot(idx, strides))
        A[k, k] = -2.0 * sum(inv_h2)
        for ax in range(3):
            for step in (-1, 1):
                nb = list(idx)
                nb[ax] += step
                if problem.periodic[ax]:
                    nb[ax] %= shape[ax]
                elif nb[ax] < 0 or nb[ax] >= shape[ax]:
                    continue
                A[k, int(np.dot(nb, strides))] += inv_h2[ax]
    return A
