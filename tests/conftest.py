"""
conftest.py
-----------
Shared pytest fixtures for the entire test suite.

All quantum solver fixtures use N=4 (2 qubits) to keep the suite fast: the
slowest single test is ~6 s and the whole suite ~26 s.  The N=8 cases are
reserved for the benchmark scripts, which are run separately.

Fixtures are scoped at 'module' level where the setup cost is non-trivial
so they are only computed once per test file rather than once per test function.

Manufactured solutions
----------------------
The 2D and 3D fixtures return a ``(problem, u_exact)`` pair built by the method
of manufactured solutions: a solution is chosen first, the corresponding source
term is derived analytically, and the solver is then required to recover the
original. This gives an exact target that is independent of any other solver,
so a failure implicates the code under test rather than a reference
implementation. The builders mirror those in ``scripts/debug_2d.py`` and
``scripts/debug_3d.py``, which have been exercised extensively.

Note that a discrete solve recovers the chosen solution only to within the
truncation error of the 5-point (or 7-point) stencil, O(h²). Tests that assert
against ``u_exact`` must therefore use a tolerance scaled to the mesh; tests
that need an exact target should compare against the discrete solution obtained
with the direct Thomas strip solver instead.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.config import SimConfig1D
from core.het_config import HETConfig, HETPhysicalConfig
from problems.poisson_1d import PoissonProblem1D
from problems.het_plasma_1d import HETPoissonProblem1D, HETPhysicalProblem1D
from problems.poisson_line_2d import PoissonLine2D
from problems.poisson_line_3d import PoissonLine3D
from solvers.quantum.vqls_1d import VQLSConfig1D


# -- 1D Poisson fixtures -------------------------------------------------------

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


# -- Line-decomposed 2D/3D manufactured solutions ------------------------------

def build_square_2d(N: int) -> tuple[PoissonLine2D, np.ndarray]:
    """
    Constructs ∇²u = sin(πx)·sin(πy) on the unit square, homogeneous Dirichlet.

    The continuum solution is u = −sin(πx)·sin(πy) / (2π²), since the chosen
    source is an eigenfunction of the Laplacian with eigenvalue −2π².

    Parameters
    ----------
    N : int
        Interior nodes per direction.

    Returns
    -------
    problem : PoissonLine2D
        (N, N) line-decomposed problem.
    u_exact : np.ndarray
        (N, N) continuum solution sampled at the interior nodes.
    """
    h = 1.0 / (N + 1)
    p = np.arange(1, N + 1) * h
    X, Y = np.meshgrid(p, p, indexing="ij")
    f = np.sin(np.pi * X) * np.sin(np.pi * Y)
    return PoissonLine2D(f), -f / (2.0 * np.pi**2)


def build_cube_3d(N: int) -> tuple[PoissonLine3D, np.ndarray]:
    """
    Constructs the triple-sine manufactured solution on the unit cube.

    φ = sin(πx)·sin(πy)·sin(πz) satisfies ∇²φ = −3π²·φ under homogeneous
    Dirichlet data on all six faces.

    Parameters
    ----------
    N : int
        Interior nodes per direction.

    Returns
    -------
    problem : PoissonLine3D
        (N, N, N) line-decomposed problem.
    u_exact : np.ndarray
        (N, N, N) continuum solution at the interior nodes.
    """
    h = 1.0 / (N + 1)
    p = np.arange(1, N + 1) * h
    X, Y, Z = np.meshgrid(p, p, p, indexing="ij")
    phi = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    return PoissonLine3D(-3.0 * np.pi**2 * phi, lengths=(1.0, 1.0, 1.0)), phi


def build_periodic_3d(N: int, m: int = 1) -> tuple[PoissonLine3D, np.ndarray]:
    """
    Constructs a slab with one periodic axis, as in the unwrapped HET channel.

    φ = sin(πx/Lx)·sin(πy/Ly)·cos(2πm·z/Lz) is periodic in z with mode number m
    and vanishes on the four Dirichlet faces, giving

        ∇²φ = −π²·(1/Lx² + 1/Ly² + 4m²/Lz²)·φ

    The periodic axis is discretised without boundary nodes, so its spacing is
    Lz/N rather than Lz/(N+1) — the distinction that the transfer operators and
    the coarsening logic must respect.

    Parameters
    ----------
    N : int
        Interior nodes per direction.
    m : int, default=1
        Azimuthal mode number.

    Returns
    -------
    problem : PoissonLine3D
        (N, N, N) problem, periodic in the third axis.
    u_exact : np.ndarray
        (N, N, N) continuum solution at the interior nodes.
    """
    Lx, Ly, Lz = 0.025, 0.020, 0.080
    dx, dy, dz = Lx / (N + 1), Ly / (N + 1), Lz / N
    X, Y, Z = np.meshgrid(np.arange(1, N + 1) * dx,
                          np.arange(1, N + 1) * dy,
                          np.arange(N) * dz, indexing="ij")
    phi = (np.sin(np.pi * X / Lx) * np.sin(np.pi * Y / Ly)
           * np.cos(2.0 * np.pi * m * Z / Lz))
    lap = -np.pi**2 * (1.0 / Lx**2 + 1.0 / Ly**2 + 4.0 * m**2 / Lz**2)
    problem = PoissonLine3D(lap * phi, lengths=(Lx, Ly, Lz),
                            periodic=(False, False, True))
    return problem, phi


@pytest.fixture(scope="module")
def square_2d_N8():
    """Unit-square manufactured solution at N=8 — the default 2D test case."""
    return build_square_2d(8)


@pytest.fixture(scope="module")
def square_2d_N16():
    """Unit-square manufactured solution at N=16, for mesh-refinement checks."""
    return build_square_2d(16)


@pytest.fixture(scope="module")
def cube_3d_N8():
    """Unit-cube manufactured solution at N=8 — the default 3D test case."""
    return build_cube_3d(8)


@pytest.fixture(scope="module")
def periodic_3d_N8():
    """Slab with a periodic third axis at N=8."""
    return build_periodic_3d(8)


# -- VQLS config fixture -------------------------------------------------------

@pytest.fixture(scope="module")
def vqls_cfg_fast():
    """
    Minimal VQLS config for fast testing.

    n_layers=3, max_iter=150 per restart, 3 restarts = 450 total.
    For N=4 (2 qubits) this converges in under 10 seconds.
    Tolerance is loose (1e-3) — this validates that the solver executes and
    produces a reasonable outcome, rather than publication-level accuracy.
    """
    return VQLSConfig1D(
        n_layers    = 3,
        optimiser   = "COBYLA",
        max_iter    = 150,
        tol         = 1e-3,
        random_seed = 0,
        verbose     = False,
    )


# -- HET fixtures --------------------------------------------------------------

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


# -- Shared error metric -------------------------------------------------------

def rel_err(u: np.ndarray, ref: np.ndarray) -> float:
    """
    Computes the maximum relative error, excluding near-zero reference nodes.

    Nodes satisfying |ref| < 1e-6·max|ref| are masked out. Without the mask the
    metric diverges at the interior zeros of the oscillatory source functions,
    where the reference passes through zero but the solver error does not — an
    artefact of the metric rather than of the solver.

    Parameters
    ----------
    u : np.ndarray
        Solver solution, of any shape matching `ref`.
    ref : np.ndarray
        Reference solution.

    Returns
    -------
    float
        Maximum relative error as a fraction (not a percentage), or 0.0 if
        every node is masked.
    """
    scale = np.max(np.abs(ref))
    mask  = np.abs(ref) > 1e-6 * scale
    if not mask.any():
        return 0.0
    return float(np.max(np.abs((u - ref)[mask]) / np.abs(ref[mask])))


# -- Shared tolerance constants ------------------------------------------------

# These are loose — tests check correctness, not publication accuracy.
THOMAS_RESIDUAL_TOL  = 1e-10   # Thomas should be near machine precision
HHL_REL_ERROR_TOL    = 0.20    # 20% — Trotter error at N=4, epsilon=0.01
VQLS_REL_ERROR_TOL   = 0.15    # 15% — variational error with fast config
VQLS_COST_TOL        = 0.05    # cost < 0.05 means the optimiser converged


# -------- Addition for Quantum Hardware Test (QHT) ---------------------

import json
from pathlib import Path


def pytest_addoption(parser):
    parser.addoption(
        "--update-baseline",
        action="store_true",
        default=False,
        help=(
            "Regenerate tests/baselines/baseline_v1.json from the current "
            "code. Run this once, on the baseline commit, in the msc_qiskit "
            "environment, and commit the result."
        ),
    )


def pytest_sessionfinish(session, exitstatus):
    """Write the golden baseline file, if this session was generating one."""
    if not session.config.getoption("--update-baseline", default=False):
        return

    try:
        from tests.test_regression_baseline import (
            BASELINE_DIR, BASELINE_FILE, _STASH, _provenance,
        )
    except Exception:                                    # pragma: no cover
        return

    if not _STASH:
        print("\n--update-baseline was given but no cases ran; nothing written.")
        return

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema":     1,
        "provenance": _provenance(),
        "cases":      dict(sorted(_STASH.items())),
    }
    BASELINE_FILE.write_text(json.dumps(payload, indent=2) + "\n")

    print(f"\nBaseline written: {BASELINE_FILE} ({len(_STASH)} cases)")
    print(f"  git sha : {payload['provenance']['git_sha']}")
    print(f"  versions: {payload['provenance']['versions']}")
    print("  Commit this file. It is the thesis replication contract.")