"""
test_problem_setup.py
---------------------
Tests for problem assembly: grid construction, matrix structure,
RHS assembly, and condition number computation.

These tests are purely classical and run in milliseconds.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.config import SimConfig1D, SimConfig2D
from core.source_functions import SOURCE_FUNCTIONS, SOURCE_FUNCTIONS_2D
from core.exact_solutions import EXACT_SOLUTIONS
from problems.poisson_1d import PoissonProblem1D, build_grid, build_tst_matrix
from problems.poisson_line_2d import PoissonLine2D


# -- SimConfig validation ------------------------------------------------------

class TestSimConfig:

    def test_valid_config_created(self):
        """
        Verifies that a simulation configuration is correctly initialised 
        with valid parameter assignments.
        """
        cfg = SimConfig1D(N=8, epsilon=0.01, source_fn="fS")
        assert cfg.N == 8
        assert cfg.epsilon == 0.01
        assert cfg.source_fn == "fS"

    def test_N_not_power_of_2_raises(self):
        """
        Confirms that initialisation strictly rejects grid sizes that are not powers of two, 
        ensuring compatibility with downstream hierarchical requirements.
        """
        with pytest.raises(ValueError, match="power of 2"):
            SimConfig1D(N=7, epsilon=0.01, source_fn="fS")

    def test_N_zero_raises(self):
        """
        Validates that zero grid dimensions are appropriately rejected during configuration 
        instantiation to prevent degenerate states.
        """
        with pytest.raises(ValueError):
            SimConfig1D(N=0, epsilon=0.01, source_fn="fS")

    def test_unknown_source_fn_raises(self):
        """
        Ensures that invalid source function identifiers raise explicit errors, enforcing 
        strict dependence on registered analytical cases.
        """
        with pytest.raises(ValueError, match="Unrecognised source function"):
            SimConfig1D(N=8, epsilon=0.01, source_fn="fX")

    def test_negative_epsilon_raises(self):
        """
        Verifies that negative tolerance parameters correctly trigger initialisation failures.
        """
        with pytest.raises(ValueError, match="epsilon"):
            SimConfig1D(N=8, epsilon=-0.01, source_fn="fS")

    def test_nonhomogeneous_bcs_stored(self):
        """
        Confirms that explicit non-homogeneous boundary coefficients are correctly preserved 
        within the active configuration state.
        """
        cfg = SimConfig1D(N=4, epsilon=0.01, source_fn="fS", alpha=1.0, beta=-1.0)
        assert cfg.alpha == 1.0
        assert cfg.beta  == -1.0


class TestSimConfig2D:

    def test_valid_config_created(self):
        cfg = SimConfig2D(N=8, epsilon=0.01, source_fn="fS")
        assert cfg.N == 8

    def test_N_not_power_of_2_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            SimConfig2D(N=6, epsilon=0.01, source_fn="fS")

    def test_unknown_source_fn_raises(self):
        """
        Ensures that invalid source function identifiers raise explicit errors, enforcing 
        strict dependence on registered analytical cases.
        """
        with pytest.raises(ValueError):
            SimConfig2D(N=4, epsilon=0.01, source_fn="bad")


# -- Grid construction ---------------------------------------------------------

class TestGrid1D:

    def test_grid_length(self):
        """
        Validates that the constructed spatial grid contains the precisely requested number 
        of internal discrete nodes.
        """
        x, dx = build_grid(8)
        assert len(x) == 8

    def test_grid_spacing(self):
        """
        Verifies that the constant grid spacing matches theoretical calculations based 
        on the uniform interior partition.
        """
        x, dx = build_grid(8)
        assert dx == pytest.approx(1.0 / 9)

    def test_grid_bounds(self):
        """
        Verifies that all evaluated internal spatial coordinates lie strictly within 
        the open interval between domain endpoints.
        """
        x, dx = build_grid(8)
        assert np.all(x > 0) and np.all(x < 1)

    def test_grid_uniform(self):
        """
        Confirms that the finite differences between successive interior nodes are 
        constant, ensuring global mesh uniformity.
        """
        x, dx = build_grid(8)
        diffs = np.diff(x)
        assert np.allclose(diffs, diffs[0])

    @pytest.mark.parametrize("N", [4, 8, 16])
    def test_grid_first_last_node(self, N):
        """
        Validates that the terminal interior nodes are situated exactly one discrete 
        interval away from their corresponding geometric domain boundaries.
        """
        x, dx = build_grid(N)
        assert x[0]  == pytest.approx(dx)
        assert x[-1] == pytest.approx(N * dx)


# -- TST matrix structure ------------------------------------------------------

class TestTSTMatrix1D:

    def test_shape(self):
        """
        Verifies that the generated tridiagonal matrix exhibits the correct square 
        dimensionality relative to the specified grid resolution.
        """
        A = build_tst_matrix(8)
        assert A.shape == (8, 8)

    def test_main_diagonal(self):
        """
        Confirms that the principal diagonal of the discrete Laplacian correctly 
        contains the theoretical finite difference coefficient of -2.0.
        """
        A = build_tst_matrix(8)
        assert np.allclose(np.diag(A), -2.0)

    def test_off_diagonals(self):
        """
        Validates that the immediate superdiagonal and subdiagonal elements possess 
        the theoretically anticipated unitary coefficient.
        """
        A = build_tst_matrix(8)
        assert np.allclose(np.diag(A, 1),  1.0)
        assert np.allclose(np.diag(A, -1), 1.0)

    def test_symmetric(self):
        """
        Verifies the algebraic symmetry of the assembled spatial discretisation matrix, 
        ensuring suitability for symmetric solvers.
        """
        A = build_tst_matrix(8)
        assert np.allclose(A, A.T)

    def test_no_other_nonzero_entries(self):
        """
        Confirms strict tridiagonality by verifying that all matrix entries outside 
        the three principal diagonals evaluate identically to zero.
        """
        A = build_tst_matrix(8)
        mask = np.eye(8, k=0) + np.eye(8, k=1) + np.eye(8, k=-1)
        assert np.allclose(A[mask == 0], 0.0)


# -- PoissonProblem1D ----------------------------------------------------------

class TestPoissonProblem1D:

    def test_matrix_shape(self, problem_1d_N4_fS):
        """
        Validates that the globally assembled linear system operator possesses the 
        expected square geometric structure.
        """
        assert problem_1d_N4_fS.A.shape == (4, 4)

    def test_rhs_length(self, problem_1d_N4_fS):
        """
        Verifies that the compiled right-hand side source vector aligns dimensionally 
        with the underlying finite difference grid.
        """
        assert len(problem_1d_N4_fS.b) == 4

    def test_condition_number_positive(self, problem_1d_N4_fS):
        """
        Confirms that the computationally derived condition number of the system 
        matrix evaluates to a strictly positive scalar magnitude.
        """
        assert problem_1d_N4_fS.kappa > 0

    def test_condition_number_scaling(self):
        """
        Verifies that the analytical condition number of the numerical operator 
        exhibits the theoretically mandated O(N^2) geometric scaling.
        """
        prob4 = PoissonProblem1D(SimConfig1D(N=4, epsilon=0.01, source_fn="fS"))
        prob8 = PoissonProblem1D(SimConfig1D(N=8, epsilon=0.01, source_fn="fS"))
        # κ(N=8) / κ(N=4) should be approximately (8/4)² = 4
        ratio = prob8.kappa / prob4.kappa
        assert 2.0 < ratio < 6.0, f"κ ratio={ratio:.2f}, expected ~4"

    def test_homogeneous_bc_rhs_boundary(self, problem_1d_N4_fS):
        """
        Validates that under strictly homogeneous Dirichlet conditions, the right-hand side 
        source vector receives no residual algebraic boundary shifts.
        """
        # The RHS should equal dx²·f(x) at all interior nodes.
        prob = problem_1d_N4_fS
        x, dx = prob.x, prob.dx
        from core.source_functions import SOURCE_FUNCTIONS
        f_vals = SOURCE_FUNCTIONS["fS"](x)
        expected = dx**2 * f_vals
        # First and last entries have no BC correction for homogeneous case.
        assert np.allclose(prob.b, expected, atol=1e-12)

    def test_nonhomogeneous_bc_rhs_correction(self, problem_1d_N4_nonhom):
        """
        Confirms that explicit non-homogeneous Dirichlet parameters precisely translate 
        into algebraic adjustments at the initial and terminal elements of the source vector.
        """
        prob = problem_1d_N4_nonhom
        # The first entry should have alpha subtracted.
        x, dx = prob.x, prob.dx
        from core.source_functions import SOURCE_FUNCTIONS
        f_vals = SOURCE_FUNCTIONS["fS"](x)
        b_no_bc = dx**2 * f_vals
        assert prob.b[0]  == pytest.approx(b_no_bc[0]  - 0.5,  abs=1e-12)
        assert prob.b[-1] == pytest.approx(b_no_bc[-1] - (-0.5), abs=1e-12)

    def test_summary_string(self, problem_1d_N4_fS):
        """
        Verifies that the procedural descriptor systematically characterises the current 
        operational state, including spatial resolution and forcing regime.
        """
        s = problem_1d_N4_fS.summary()
        assert "N=4" in s
        assert "fS" in s


# -- PoissonLine2D strip operator ----------------------------------------------

class TestPoissonLine2D:
    """
    Minimal retained coverage of the 2D strip operator.

    Comprehensive coverage of the line-decomposed problems and the outer
    schemes that drive them is outstanding work; the conditioning assertion is
    kept here because κ(A_row) → 3⁻ is the property that makes the strip
    decomposition viable for the quantum solvers at all, and it should not go
    untested in the interim.
    """

    def test_strip_operator_shape(self):
        """
        Validates the geometric consistency of the isolated strip decomposition operator 
        and its associated local source term vector.
        """
        prob = PoissonLine2D(np.zeros((4, 4)))
        assert prob.row_matrix().shape == (4, 4)
        assert prob.rhs().shape == (4, 4)

    def test_kappa_row_approaches_3(self):
        """
        Verifies that the condition number of the row matrix monotonically approaches 
        the theoretical limit of 3 from below as grid resolution increases.
        """
        kappa_16 = PoissonLine2D(np.zeros((16, 16))).kappa_row()
        kappa_32 = PoissonLine2D(np.zeros((32, 32))).kappa_row()
        assert 1.0 < kappa_16 < 3.0
        assert 1.0 < kappa_32 < 3.0
        assert kappa_32 > kappa_16

    def test_kappa_invariant_under_h2_rescaling(self):
        """
        Confirms that the row condition number remains strictly invariant beneath 
        an algebraic h^2 rescaling mapping. Ensures consistency between physical 
        and spectral conventions.
        """
        prob = PoissonLine2D(np.zeros((8, 8)))
        scaled = -4.0 * np.eye(8) + np.diag(np.ones(7), 1) + np.diag(np.ones(7), -1)
        eigs = np.abs(np.linalg.eigvalsh(scaled))
        assert prob.kappa_row() == pytest.approx(eigs.max() / eigs.min(), rel=1e-12)


# -- Source functions and exact solutions --------------------------------------

class TestSourceFunctions:

    @pytest.mark.parametrize("fn_key", ["fS", "fL", "fH"])
    def test_1d_source_fn_shape(self, fn_key):
        """
        Verifies that dynamically generated one-dimensional source profiles correctly 
        assume the spatial dimensionality of their input coordinate vector.
        """
        x = np.linspace(0.1, 0.9, 10)
        f = SOURCE_FUNCTIONS[fn_key](x)
        assert f.shape == x.shape

    @pytest.mark.parametrize("fn_key", ["fS", "fL", "fH"])
    def test_2d_source_fn_shape(self, fn_key):
        """
        Validates that mapped two-dimensional forcing functions appropriately respect 
        the dimensional structure of the underlying geometric coordinate grid.
        """
        x = np.linspace(0.1, 0.9, 5)
        y = np.linspace(0.1, 0.9, 5)
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = SOURCE_FUNCTIONS_2D[fn_key](X, Y)
        assert f.shape == X.shape


class TestExactSolutions:

    @pytest.mark.parametrize("fn_key", ["fS", "fL", "fH"])
    def test_exact_solution_satisfies_bc(self, fn_key):
        """
        Confirms that the registered exact analytical solutions natively satisfy 
        the mandated homogeneous Dirichlet boundary condition at both endpoints.
        """
        u_fn = EXACT_SOLUTIONS[fn_key]
        assert u_fn(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-12)
        assert u_fn(np.array([1.0]))[0] == pytest.approx(0.0, abs=1e-12)

    def test_fS_exact_solution_known_value(self):
        """
        Verifies the numerical integrity of the standard sinusoidal analytical solution 
        by querying a documented reference coordinate point.
        """
        u_fn = EXACT_SOLUTIONS["fS"]
        expected = -np.sin(np.pi * 0.5) / np.pi**2
        assert u_fn(np.array([0.5]))[0] == pytest.approx(expected, rel=1e-10)