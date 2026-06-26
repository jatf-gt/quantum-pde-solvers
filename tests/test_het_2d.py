"""
Verification tests for the 2-D HET plasma Poisson problem assembly
and solver compatibility.

Tests are restricted to N=4 and max_iter=5 to bound runtime. The
purpose is to verify problem assembly correctness — charge density
shape, boundary condition encoding, and solver interface compatibility
— not numerical accuracy of the physical solution.

Expected runtime: under 5 minutes for the full file.
"""
from __future__ import annotations

import numpy as np
import pytest

from problems.het_plasma_2d import HETPoissonProblem2D, HETConfig2D
from solvers.classical.thomas_2d import thomas_solve_2d
from solvers.quantum.hhl_2d import hhl_solve_2d
from solvers.quantum.vqls_2d import vqls_solve_2d, VQLSConfig2D
from solvers.quantum.vqls_1d import VQLSConfig1D


# -- Fixtures -----------------------------------------------------------------

@pytest.fixture(scope="module")
def het_cfg_2d_N4():
    """Minimal HET-2D configuration: N=4, max_iter=5."""
    return HETConfig2D(N=4, epsilon=0.01, max_iter=5)


@pytest.fixture(scope="module")
def het_problem_2d_N4(het_cfg_2d_N4):
    return HETPoissonProblem2D(het_cfg_2d_N4)


@pytest.fixture(scope="module")
def vqls_cfg_het_2d_fast():
    inner = VQLSConfig1D(
        n_layers=2, max_iter=50, tol=1e-1,
        random_seed=0, verbose=False,
    )
    return VQLSConfig2D(inner_config=inner, warm_start=True, verbose=False)


# -- Configuration tests ------------------------------------------------------

class TestHETConfig2D:

    def test_derived_quantities_positive(self, het_cfg_2d_N4):
        assert het_cfg_2d_N4.lambda_D > 0
        assert het_cfg_2d_N4.alpha    > 0
        assert het_cfg_2d_N4.phi_0    > 0

    def test_delta_0_physically_small(self, het_cfg_2d_N4):
        """
        α·δ_0 must be small relative to α_bc to ensure the space charge
        is a perturbation on the applied voltage, not the dominant term.
        """
        cfg = het_cfg_2d_N4
        assert cfg.alpha * cfg.delta_0 < cfg.alpha_bc, (
            f"Space charge contribution α·δ_0={cfg.alpha*cfg.delta_0:.2f} "
            f"exceeds α_bc={cfg.alpha_bc:.2f}."
        )

    def test_invalid_N_raises(self):
        with pytest.raises(ValueError, match="power of 2"):
            HETConfig2D(N=5)


# -- Problem assembly tests ---------------------------------------------------

class TestHETPoissonProblem2D:

    def test_grid_shape(self, het_problem_2d_N4):
        assert het_problem_2d_N4.X.shape == (4, 4)
        assert het_problem_2d_N4.Y.shape == (4, 4)

    def test_charge_density_shape(self, het_problem_2d_N4):
        assert het_problem_2d_N4.delta_n_2d.shape == (4, 4)

    def test_charge_density_finite(self, het_problem_2d_N4):
        assert np.all(np.isfinite(het_problem_2d_N4.delta_n_2d))

    def test_a_row_is_tst_minus_4(self, het_problem_2d_N4):
        """Row matrix must have main diagonal −4 and off-diagonals +1."""
        A = het_problem_2d_N4.A_row
        assert np.allclose(np.diag(A),    -4.0)
        assert np.allclose(np.diag(A, 1),  1.0)
        assert np.allclose(np.diag(A, -1), 1.0)

    def test_get_row_system_shape(self, het_problem_2d_N4):
        u_prev = np.zeros((4, 4))
        A_row, b_row = het_problem_2d_N4.get_row_system(0, u_prev)
        assert A_row.shape == (4, 4)
        assert len(b_row)  == 4

    def test_anode_bc_in_first_row_rhs(self, het_problem_2d_N4):
        """
        For row j=0, the inner wall BC (α_bc) must be subtracted from
        the RHS, in addition to the anode x-direction BC on b_row[0].
        """
        cfg    = het_problem_2d_N4.het_config
        u_prev = np.zeros((4, 4))
        _, b_row = het_problem_2d_N4.get_row_system(0, u_prev)
        # b_row[0] should contain both the anode x-BC and the inner wall y-BC.
        # With zero u_prev and zero source, b_row[0] = -alpha_bc (x) - alpha_bc (y).
        # We check the sign is negative (potential is subtracted).
        assert b_row[0] < 0, (
            f"b_row[0]={b_row[0]:.4f} should be negative due to BC subtractions."
        )

    def test_build_full_rhs_length(self, het_problem_2d_N4):
        rhs = het_problem_2d_N4.build_full_rhs()
        assert len(rhs) == 4 * 4

    def test_summary_string(self, het_problem_2d_N4):
        s = het_problem_2d_N4.summary()
        assert "HET-2D" in s
        assert "N=4" in s


# -- Solver compatibility tests -----------------------------------------------

class TestHET2DSolverCompatibility:

    def test_thomas_runs_on_het_2d(self, het_problem_2d_N4):
        r = thomas_solve_2d(het_problem_2d_N4)
        assert r.u.shape == (4, 4)
        assert np.all(np.isfinite(r.u))

    def test_thomas_solution_sign(self, het_problem_2d_N4):
        """
        With a positive anode BC (α_bc > 0) and grounded cathode and
        walls, the potential must be positive throughout the domain.
        """
        r = thomas_solve_2d(het_problem_2d_N4)
        assert np.all(r.u >= -0.1), (
            "Potential should be non-negative for positive anode BC."
        )

    def test_hhl_runs_on_het_2d(self, het_problem_2d_N4):
        r = hhl_solve_2d(het_problem_2d_N4)
        assert r.u.shape == (4, 4)
        assert np.all(np.isfinite(r.u))

    def test_vqls_runs_on_het_2d(
        self, het_problem_2d_N4, vqls_cfg_het_2d_fast
    ):
        r = vqls_solve_2d(het_problem_2d_N4, config=vqls_cfg_het_2d_fast)
        assert r.u.shape == (4, 4)
        assert np.all(np.isfinite(r.u))

    def test_all_solvers_agree_in_sign(self, het_problem_2d_N4):
        """
        Thomas, HHL, and VQLS must agree on the sign of the dominant
        solution component. A sign disagreement indicates a
        proportionality recovery failure in one of the quantum solvers.
        """
        inner = VQLSConfig1D(
            n_layers=2, max_iter=50, tol=1e-1,
            random_seed=0, verbose=False,
        )
        vqls_cfg = VQLSConfig2D(
            inner_config=inner, warm_start=False, verbose=False
        )

        r_thomas = thomas_solve_2d(het_problem_2d_N4)
        r_hhl    = hhl_solve_2d(het_problem_2d_N4)
        r_vqls   = vqls_solve_2d(het_problem_2d_N4, config=vqls_cfg)

        idx = np.unravel_index(
            np.argmax(np.abs(r_thomas.u)), r_thomas.u.shape
        )
        sign_t = np.sign(r_thomas.u[idx])
        sign_h = np.sign(r_hhl.u[idx])
        sign_v = np.sign(r_vqls.u[idx])

        assert sign_t == sign_h, (
            f"HHL sign mismatch at {idx}: "
            f"Thomas={r_thomas.u[idx]:.3f}, HHL={r_hhl.u[idx]:.3f}"
        )
        assert sign_t == sign_v, (
            f"VQLS sign mismatch at {idx}: "
            f"Thomas={r_thomas.u[idx]:.3f}, VQLS={r_vqls.u[idx]:.3f}"
        )