"""
conftest.py
-----------
Shared pytest fixtures for the entire test suite.

All quantum solver fixtures use N=4 (2 qubits) to keep individual
test runtime under ~10 seconds.  The N=8 cases are reserved for the
benchmark scripts which are run separately.

Fixtures are scoped at 'module' level where the setup cost is non-trivial
(e.g. building a PoissonProblem2D with a refined reference) so they are
only computed once per test file rather than once per test function.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.config import SimConfig1D, SimConfig2D
from core.het_config import HETConfig, HETPhysicalConfig
from problems.poisson_1d import PoissonProblem1D
from problems.poisson_2d import PoissonProblem2D
from problems.het_plasma_1d import HETPoissonProblem1D, HETPhysicalProblem1D
from solvers.quantum.vqls_1d import VQLSConfig1D


# ── 1D Poisson fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg_1d_N4_fS():
    """Smallest valid 1D config: N=4, fS, homogeneous BCs."""
    return SimConfig1D(N=4, epsilon=0.01, source_fn="fS")


@pytest.fixture(scope="module")
def cfg_1d_N4_fL():
    return SimConfig1D(N=4, epsilon=0.01, source_fn="fL")


@pytest.fixture(scope="module")
def cfg_1d_N4_fH():
    return SimConfig1D(N=4, epsilon=0.01, source_fn="fH")


@pytest.fixture(scope="module")
def cfg_1d_N4_nonhom():
    """N=4, non-homogeneous BCs — tests BC encoding."""
    return SimConfig1D(N=4, epsilon=0.01, source_fn="fS", alpha=0.5, beta=-0.5)


@pytest.fixture(scope="module")
def problem_1d_N4_fS(cfg_1d_N4_fS):
    return PoissonProblem1D(cfg_1d_N4_fS)


@pytest.fixture(scope="module")
def problem_1d_N4_fL(cfg_1d_N4_fL):
    return PoissonProblem1D(cfg_1d_N4_fL)


@pytest.fixture(scope="module")
def problem_1d_N4_fH(cfg_1d_N4_fH):
    return PoissonProblem1D(cfg_1d_N4_fH)


@pytest.fixture(scope="module")
def problem_1d_N4_nonhom(cfg_1d_N4_nonhom):
    return PoissonProblem1D(cfg_1d_N4_nonhom)


# ── 2D Poisson fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg_2d_N4_fS():
    """N=4 2D config — smallest valid 2D system."""
    return SimConfig2D(N=4, epsilon=0.01, source_fn="fS", max_iter=200)


@pytest.fixture(scope="module")
def cfg_2d_N4_fL():
    return SimConfig2D(N=4, epsilon=0.01, source_fn="fL", max_iter=200)


@pytest.fixture(scope="module")
def problem_2d_N4_fS(cfg_2d_N4_fS):
    return PoissonProblem2D(cfg_2d_N4_fS)


@pytest.fixture(scope="module")
def problem_2d_N4_fL(cfg_2d_N4_fL):
    return PoissonProblem2D(cfg_2d_N4_fL)


# ── VQLS config fixture ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def vqls_cfg_fast():
    """
    Minimal VQLS config for fast testing.

    n_layers=3, max_iter=150 per restart, 3 restarts = 450 total.
    For N=4 (2 qubits) this converges in under 10 seconds.
    Tolerance is loose (1e-3) — we are checking the solver runs and
    produces a reasonable answer, not publication-level accuracy.
    """
    return VQLSConfig1D(
        n_layers    = 3,
        optimiser   = "COBYLA",
        max_iter    = 150,
        tol         = 1e-3,
        random_seed = 0,
        verbose     = False,
    )


# ── HET fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def het_cfg_N4_linear_hom():
    """HET linear profile, homogeneous BCs, N=4 — has analytical solution."""
    return HETConfig(
        N=4, epsilon=0.01,
        rho_profile="linear",
        V_discharge=0.0,
    )


@pytest.fixture(scope="module")
def het_cfg_N4_gaussian_phys():
    """HET Gaussian profile, physical BCs (V_d=300V), N=4."""
    return HETConfig(
        N=4, epsilon=0.01,
        rho_profile="gaussian",
        V_discharge=300.0,
    )


@pytest.fixture(scope="module")
def het_problem_N4_linear(het_cfg_N4_linear_hom):
    return HETPoissonProblem1D(het_cfg_N4_linear_hom)


@pytest.fixture(scope="module")
def het_problem_N4_gaussian(het_cfg_N4_gaussian_phys):
    return HETPoissonProblem1D(het_cfg_N4_gaussian_phys)


@pytest.fixture(scope="module")
def het_physical_cfg_N4():
    """Boeuf-Garrigues physical config, N=4."""
    return HETPhysicalConfig(N=4, epsilon=0.01)


@pytest.fixture(scope="module")
def het_physical_problem_N4(het_physical_cfg_N4):
    return HETPhysicalProblem1D(het_physical_cfg_N4)


# ── Shared tolerance constants ────────────────────────────────────────────────

# These are loose — tests check correctness, not publication accuracy.
THOMAS_RESIDUAL_TOL  = 1e-10   # Thomas should be near machine precision
HHL_REL_ERROR_TOL    = 0.20    # 20% — Trotter error at N=4, epsilon=0.01
VQLS_REL_ERROR_TOL   = 0.15    # 15% — variational error with fast config
VQLS_COST_TOL        = 0.05    # cost < 0.05 means the optimiser converged