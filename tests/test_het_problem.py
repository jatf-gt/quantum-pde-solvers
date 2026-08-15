"""
test_het_problem.py
-------------------
Tests for the HET plasma problem assembly and solver compatibility.

These tests verify:
  1. HETConfig computes derived quantities correctly
  2. HETPoissonProblem1D assembles the correct matrix and RHS
  3. The analytical solution (linear profile) is correct
  4. HHL and VQLS can be called on the HET problem without error
  5. The electric field recovery produces physically sensible values
"""
from __future__ import annotations

import numpy as np
import pytest

from core.het_config import HETConfig, HETPhysicalConfig, EPS_0, E_CHARGE, EV_TO_J
from problems.het_plasma_1d import HETPoissonProblem1D, HETPhysicalProblem1D
from solvers.classical.thomas import thomas_solve_system
from solvers.quantum.hhl_1d import hhl_solve_system
from solvers.quantum.vqls_1d import vqls_solve_system
from conftest import HHL_REL_ERROR_TOL, VQLS_REL_ERROR_TOL, VQLS_COST_TOL


class TestHETConfig:

    def test_debye_length_formula(self, het_cfg_N4_linear_hom):
        """
        Validates the analytical derivation of the Debye length against physical parameters.
        
        Confirms λ_D = sqrt(ε_0 k_B T_e / (e² n_0)).
        """
        cfg   = het_cfg_N4_linear_hom
        T_e_J = cfg.T_e_eV * EV_TO_J
        expected_lambda_D = np.sqrt(EPS_0 * T_e_J / (E_CHARGE**2 * cfg.n_0))
        assert cfg.lambda_D == pytest.approx(expected_lambda_D, rel=1e-8)

    def test_alpha_formula(self, het_cfg_N4_linear_hom):
        """
        Ensures the non-dimensional scaling parameter α is computed accurately.
        
        Confirms α = (L/λ_D)².
        """
        cfg = het_cfg_N4_linear_hom
        expected_alpha = (cfg.L / cfg.lambda_D)**2
        assert cfg.alpha == pytest.approx(expected_alpha, rel=1e-8)

    def test_phi_0_equals_T_e_eV(self, het_cfg_N4_linear_hom):
        """
        Confirms that the base potential normalization correctly aligns with the electron temperature.
        
        Evaluates φ_0 = k_B T_e / e = T_e [eV] in volts.
        """
        cfg = het_cfg_N4_linear_hom
        assert cfg.phi_0 == pytest.approx(cfg.T_e_eV, rel=1e-8)

    def test_alpha_bc_formula(self):
        """
        Validates the normalization of the boundary condition scalar against the characteristic potential.
        
        Checks α_bc = V_discharge / φ_0.
        """
        cfg = HETConfig(N=4, epsilon=0.01, rho_profile="linear",
                        V_discharge=300.0, T_e_eV=20.0)
        assert cfg.alpha_bc == pytest.approx(300.0 / 20.0, rel=1e-8)

    def test_homogeneous_bc_gives_zero_alpha_bc(self, het_cfg_N4_linear_hom):
        """Ensures that homogeneous boundary conditions translate correctly to a null boundary forcing term."""
        assert het_cfg_N4_linear_hom.alpha_bc == pytest.approx(0.0, abs=1e-10)

    def test_invalid_N_raises(self):
        """Confirms that spatial resolutions not conforming to power-of-two requirements induce failure."""
        with pytest.raises(ValueError, match="power of 2"):
            HETConfig(N=5, epsilon=0.01, rho_profile="linear")

    def test_invalid_T_e_raises(self):
        """Validates error emission upon instantiation with physically invalid electron temperature inputs."""
        with pytest.raises(ValueError, match="T_e_eV"):
            HETConfig(N=4, epsilon=0.01, rho_profile="linear", T_e_eV=-1.0)


class TestHETPoissonProblem1D:

    def test_matrix_is_tst(self, het_problem_N4_linear):
        """
        Validates that the discrete operator correctly assembles as a tridiagonal symmetric Toeplitz matrix.
        
        Confirms parity with the generic Poisson differential operator.
        """
        A = het_problem_N4_linear.A
        assert np.allclose(np.diag(A),    -2.0)
        assert np.allclose(np.diag(A, 1),  1.0)
        assert np.allclose(np.diag(A, -1), 1.0)

    def test_rhs_length(self, het_problem_N4_linear):
        """Ensures that the assembled source vector aligns with the interior domain degrees of freedom."""
        assert len(het_problem_N4_linear.b) == 4

    def test_kappa_matches_generic_poisson(self, het_problem_N4_linear):
        """
        Confirms that the condition number scales identically to the generic Poisson baseline.
        """
        from core.config import SimConfig1D
        from problems.poisson_1d import PoissonProblem1D
        prob_generic = PoissonProblem1D(SimConfig1D(N=4, epsilon=0.01, source_fn="fS"))
        assert het_problem_N4_linear.kappa == pytest.approx(
            prob_generic.kappa, rel=1e-6
        )

    def test_analytical_solution_satisfies_bcs(self, het_problem_N4_linear):
        """
        Validates the formal compliance of the analytical reference solution against imposed boundary domains.
        """
        u_exact = het_problem_N4_linear.analytical_solution()
        assert u_exact is not None
        # The analytical solution is evaluated at interior nodes.
        # Extrapolate to boundaries: u(0) and u(1) should both be 0.
        cfg = het_problem_N4_linear.config
        x0  = np.array([0.0])
        x1  = np.array([1.0])
        from core.exact_solutions import HET_EXACT_SOLUTIONS
        assert HET_EXACT_SOLUTIONS["linear"](x0, cfg.rho_0, cfg.alpha)[0] \
               == pytest.approx(0.0, abs=1e-10)
        assert HET_EXACT_SOLUTIONS["linear"](x1, cfg.rho_0, cfg.alpha)[0] \
               == pytest.approx(0.0, abs=1e-10)

    def test_analytical_solution_satisfies_pde(self, het_problem_N4_linear):
        """
        Ensures the discrete numerical solution converges asymptotically towards the analytical truth.
        
        Checks that second-order finite difference formulations match continuous results to O(h²).
        """
        prob    = het_problem_N4_linear
        u_exact = prob.analytical_solution()
        u_thomas = thomas_solve_system(prob.A, prob.b)
        max_err  = np.max(np.abs(u_thomas - u_exact))
        # Second-order FD error: O(h²) = O(1/(N+1)²) ~ 0.04 for N=4.
        assert max_err < 0.1, f"Thomas vs analytical: max_err={max_err:.4f}"

    def test_no_analytical_solution_for_gaussian(self, het_problem_N4_gaussian):
        """Confirms the correct absence of closed-form representations for complex density profiles."""
        assert het_problem_N4_gaussian.analytical_solution() is None

    def test_no_analytical_solution_for_nonzero_bc(self):
        """Validates that non-homogeneous boundary instances correctly suppress analytical baseline expectations."""
        cfg  = HETConfig(N=4, epsilon=0.01, rho_profile="linear",
                         V_discharge=300.0)
        prob = HETPoissonProblem1D(cfg)
        assert prob.analytical_solution() is None


class TestHETPhysicalProblem:

    def test_density_profile_bounds(self, het_physical_problem_N4):
        """
        Ensures that generated physical density profiles remain strictly bounded within expected normalized limits.
        """
        cfg = het_physical_problem_N4.config
        n   = het_physical_problem_N4.n_profile
        assert np.all(n >= cfg.n_min - 1e-10)
        assert np.all(n <= 1.0 + 1e-10)

    def test_delta_0_physically_small(self, het_physical_cfg_N4):
        """
        Validates the perturbative scale of space charge relative to the primary potential discharge.
        
        Confirms that the background plasma contributes modestly compared to the imposed boundary potential.
        """
        cfg = het_physical_cfg_N4
        space_charge_contribution = cfg.alpha * cfg.delta_0
        assert space_charge_contribution < cfg.alpha_bc, (
            f"Space charge contribution α·δ_0={space_charge_contribution:.2f} "
            f"is not small compared to α_bc={cfg.alpha_bc:.2f}."
        )

    def test_electric_field_shape(self, het_physical_problem_N4):
        """Confirms that electric field extraction maps robustly to the spatial domain configuration."""
        prob    = het_physical_problem_N4
        u_dummy = np.zeros(prob.config.N)
        x_full, E = prob.electric_field(u_dummy)
        assert len(x_full) == prob.config.N + 2
        assert len(E)      == prob.config.N + 2

    def test_electric_field_order_of_magnitude(self, het_physical_problem_N4):
        """
        Ensures that computed electric field gradients align reasonably with anticipated macro-scale bounds.
        """
        prob     = het_physical_problem_N4
        cfg      = prob.config
        u_thomas = thomas_solve_system(prob.A, prob.b)
        _, E     = prob.electric_field(u_thomas)
        E_peak   = np.max(np.abs(E))
        E_ref    = cfg.V_discharge / cfg.L   # V_d/L
        assert E_ref / 5 < E_peak < E_ref * 5, (
            f"Peak E={E_peak:.2e} V/m is not within factor 5 of "
            f"V_d/L={E_ref:.2e} V/m."
        )


@pytest.mark.quantum
class TestHETSolverCompatibility:
    """
    Verify that HHL and VQLS can be called on HET problems and return
    physically sensible results.  Uses N=4 for speed.
    """

    def test_hhl_runs_on_het_linear(self, het_problem_N4_linear):
        """Validates that the HHL formulation reliably produces finite results on HET structures."""
        prob = het_problem_N4_linear
        u, _, c = hhl_solve_system(prob.A, prob.b, prob.config.epsilon)
        assert u.shape == (4,)
        assert np.isfinite(c)
        assert np.all(np.isfinite(u))

    def test_hhl_agrees_with_thomas_on_het(self, het_problem_N4_linear):
        """Confirms relative congruity between quantum linear solving and classical reference targets."""
        prob     = het_problem_N4_linear
        u_thomas = thomas_solve_system(prob.A, prob.b)
        u_hhl, _, _ = hhl_solve_system(prob.A, prob.b, prob.config.epsilon)
        err = np.max(np.abs(u_hhl - u_thomas)) / np.max(np.abs(u_thomas))
        assert err < HHL_REL_ERROR_TOL, (
            f"HHL vs Thomas rel err={err*100:.2f}% on HET linear problem."
        )

    def test_vqls_runs_on_het_linear(self, het_problem_N4_linear, vqls_cfg_fast):
        """Validates continuous operation and structural soundness of the VQLS workflow on HET matrices."""
        prob = het_problem_N4_linear
        r    = vqls_solve_system(prob.A, prob.b, vqls_cfg_fast)
        assert r.u.shape == (4,)
        assert np.isfinite(r.final_cost)
        assert np.all(np.isfinite(r.u))

    def test_vqls_cost_converges_on_het(self, het_problem_N4_linear, vqls_cfg_fast):
        """Ensures that variational cost topologies accurately decay within prescribed structural parameters."""
        prob = het_problem_N4_linear
        r    = vqls_solve_system(prob.A, prob.b, vqls_cfg_fast)
        assert r.final_cost < VQLS_COST_TOL, (
            f"VQLS cost={r.final_cost:.4f} on HET linear problem."
        )