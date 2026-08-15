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
        cfg = SimConfig1D(N=8, epsilon=0.01, source_fn="fS")
        assert cfg.N == 8
        assert cfg.epsilon == 0.01
        assert cfg.source_fn == "fS"

    def test_N_not_power_of_2_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            SimConfig1D(N=7, epsilon=0.01, source_fn="fS")

    def test_N_zero_raises(self):
        with pytest.raises(ValueError):
            SimConfig1D(N=0, epsilon=0.01, source_fn="fS")

    def test_unknown_source_fn_raises(self):
        with pytest.raises(ValueError, match="Unrecognised source function"):
            SimConfig1D(N=8, epsilon=0.01, source_fn="fX")

    def test_negative_epsilon_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            SimConfig1D(N=8, epsilon=-0.01, source_fn="fS")

    def test_nonhomogeneous_bcs_stored(self):
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
        with pytest.raises(ValueError):
            SimConfig2D(N=4, epsilon=0.01, source_fn="bad")


# -- Grid construction ---------------------------------------------------------

class TestGrid1D:

    def test_grid_length(self):
        x, dx = build_grid(8)
        assert len(x) == 8

    def test_grid_spacing(self):
        x, dx = build_grid(8)
        assert dx == pytest.approx(1.0 / 9)

    def test_grid_bounds(self):
        """Interior nodes must lie strictly inside (0, 1)."""
        x, dx = build_grid(8)
        assert np.all(x > 0) and np.all(x < 1)

    def test_grid_uniform(self):
        """Spacing between consecutive nodes must be uniform."""
        x, dx = build_grid(8)
        diffs = np.diff(x)
        assert np.allclose(diffs, diffs[0])

    @pytest.mark.parametrize("N", [4, 8, 16])
    def test_grid_first_last_node(self, N):
        x, dx = build_grid(N)
        assert x[0]  == pytest.approx(dx)
        assert x[-1] == pytest.approx(N * dx)


# -- TST matrix structure ------------------------------------------------------

class TestTSTMatrix1D:

    def test_shape(self):
        A = build_tst_matrix(8)
        assert A.shape == (8, 8)

    def test_main_diagonal(self):
        A = build_tst_matrix(8)
        assert np.allclose(np.diag(A), -2.0)

    def test_off_diagonals(self):
        A = build_tst_matrix(8)
        assert np.allclose(np.diag(A, 1),  1.0)
        assert np.allclose(np.diag(A, -1), 1.0)

    def test_symmetric(self):
        A = build_tst_matrix(8)
        assert np.allclose(A, A.T)

    def test_no_other_nonzero_entries(self):
        """Only the three diagonals should be non-zero."""
        A = build_tst_matrix(8)
        mask = np.eye(8, k=0) + np.eye(8, k=1) + np.eye(8, k=-1)
        assert np.allclose(A[mask == 0], 0.0)


# -- PoissonProblem1D ----------------------------------------------------------

class TestPoissonProblem1D:

    def test_matrix_shape(self, problem_1d_N4_fS):
        assert problem_1d_N4_fS.A.shape == (4, 4)

    def test_rhs_length(self, problem_1d_N4_fS):
        assert len(problem_1d_N4_fS.b) == 4

    def test_condition_number_positive(self, problem_1d_N4_fS):
        assert problem_1d_N4_fS.kappa > 0

    def test_condition_number_scaling(self):
        """κ(A) should scale as O(N²) — check ratio for N=4 and N=8."""
        prob4 = PoissonProblem1D(SimConfig1D(N=4, epsilon=0.01, source_fn="fS"))
        prob8 = PoissonProblem1D(SimConfig1D(N=8, epsilon=0.01, source_fn="fS"))
        # κ(N=8) / κ(N=4) should be approximately (8/4)² = 4
        ratio = prob8.kappa / prob4.kappa
        assert 2.0 < ratio < 6.0, f"κ ratio={ratio:.2f}, expected ~4"

    def test_homogeneous_bc_rhs_boundary(self, problem_1d_N4_fS):
        """With alpha=beta=0, BC terms should not appear in the RHS."""
        # The RHS should equal dx²·f(x) at all interior nodes.
        prob = problem_1d_N4_fS
        x, dx = prob.x, prob.dx
        from core.source_functions import SOURCE_FUNCTIONS
        f_vals = SOURCE_FUNCTIONS["fS"](x)
        expected = dx**2 * f_vals
        # First and last entries have no BC correction for homogeneous case.
        assert np.allclose(prob.b, expected, atol=1e-12)

    def test_nonhomogeneous_bc_rhs_correction(self, problem_1d_N4_nonhom):
        """With alpha=0.5, beta=-0.5, first and last RHS entries are shifted."""
        prob = problem_1d_N4_nonhom
        # The first entry should have alpha subtracted.
        x, dx = prob.x, prob.dx
        from core.source_functions import SOURCE_FUNCTIONS
        f_vals = SOURCE_FUNCTIONS["fS"](x)
        b_no_bc = dx**2 * f_vals
        assert prob.b[0]  == pytest.approx(b_no_bc[0]  - 0.5,  abs=1e-12)
        assert prob.b[-1] == pytest.approx(b_no_bc[-1] - (-0.5), abs=1e-12)

    def test_summary_string(self, problem_1d_N4_fS):
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
        prob = PoissonLine2D(np.zeros((4, 4)))
        assert prob.row_matrix().shape == (4, 4)
        assert prob.rhs().shape == (4, 4)

    def test_kappa_row_approaches_3(self):
        """κ(A_row) increases towards 3 from below as N → ∞."""
        kappa_16 = PoissonLine2D(np.zeros((16, 16))).kappa_row()
        kappa_32 = PoissonLine2D(np.zeros((32, 32))).kappa_row()
        assert 1.0 < kappa_16 < 3.0
        assert 1.0 < kappa_32 < 3.0
        assert kappa_32 > kappa_16

    def test_kappa_invariant_under_h2_rescaling(self):
        """
        The physical and h²-scaled conventions differ by a uniform factor, so
        the condition number — and hence every quantum resource estimate that
        depends on it — is identical between them.
        """
        prob = PoissonLine2D(np.zeros((8, 8)))
        scaled = -4.0 * np.eye(8) + np.diag(np.ones(7), 1) + np.diag(np.ones(7), -1)
        eigs = np.abs(np.linalg.eigvalsh(scaled))
        assert prob.kappa_row() == pytest.approx(eigs.max() / eigs.min(), rel=1e-12)


# -- Source functions and exact solutions --------------------------------------

class TestSourceFunctions:

    @pytest.mark.parametrize("fn_key", ["fS", "fL", "fH"])
    def test_1d_source_fn_shape(self, fn_key):
        x = np.linspace(0.1, 0.9, 10)
        f = SOURCE_FUNCTIONS[fn_key](x)
        assert f.shape == x.shape

    @pytest.mark.parametrize("fn_key", ["fS", "fL", "fH"])
    def test_2d_source_fn_shape(self, fn_key):
        x = np.linspace(0.1, 0.9, 5)
        y = np.linspace(0.1, 0.9, 5)
        X, Y = np.meshgrid(x, y, indexing="ij")
        f = SOURCE_FUNCTIONS_2D[fn_key](X, Y)
        assert f.shape == X.shape


class TestExactSolutions:

    @pytest.mark.parametrize("fn_key", ["fS", "fL", "fH"])
    def test_exact_solution_satisfies_bc(self, fn_key):
        """Analytical solutions must satisfy u(0)=u(1)=0."""
        u_fn = EXACT_SOLUTIONS[fn_key]
        assert u_fn(np.array([0.0]))[0] == pytest.approx(0.0, abs=1e-12)
        assert u_fn(np.array([1.0]))[0] == pytest.approx(0.0, abs=1e-12)

    def test_fS_exact_solution_known_value(self):
        """u_fS(0.5) = -sin(π/2)/π² = -1/π²."""
        u_fn = EXACT_SOLUTIONS["fS"]
        expected = -np.sin(np.pi * 0.5) / np.pi**2
        assert u_fn(np.array([0.5]))[0] == pytest.approx(expected, rel=1e-10)