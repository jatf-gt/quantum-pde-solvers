"""
Baseline regression lock: enforces the replication guarantee.

Purpose
-------
Thesis results derive from exact statevector evolution at tag ``v1.0-thesis-baseline``. Subsequent hardware modifications (execution modes, transpilation accounting, noise models, and device backends) must not perturb baseline metrics.

This module guarantees that property. It records canonical solver outputs in a committed baseline file and asserts their invariance. As long as this suite passes, thesis figures remain reproducible from ``main``, obviating the need for a separate frozen fork.

Design
------
Baseline values are generated, not hand-written, to prevent transcription errors. Workflow:

    # once, on the baseline commit
    pytest tests/test_regression_baseline.py --update-baseline

    # thereafter, on every change
    pytest tests/test_regression_baseline.py

The generated file ``tests/baselines/baseline_v1.json`` carries the git SHA, platform, and pinned library versions. A baseline without provenance is an assertion rather than a record.

Tolerances
----------
Deterministic paths (Thomas, QSVT, outer-scheme iteration counts) are compared at ``rtol=0`` where arithmetic permits, and otherwise at ``1e-12`` to accommodate floating-point reassociation.

VQLS is stochastic; its optimiser exhibits varying convergence depending on BLAS reduction order. Its final cost is bounded rather than pinned, preventing false failures across different hardware. 

Anchors
-------
Two values in ``EXPECTED_OUTER`` are transcribed from historical logs rather than generated. These 2-D legacy Jacobi iteration counts (N=4 and N=8) predate this file, verifying that the outer-solver rewrite of 2026-08-07 preserved the loop's structure exactly.
"""
from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest


BASELINE_DIR  = Path(__file__).parent / "baselines"
BASELINE_FILE = BASELINE_DIR / "baseline_v1.json"

# Exact-comparison tolerance for deterministic solver paths.
RTOL_EXACT = 1e-12
ATOL_EXACT = 1e-14

# VQLS is stochastic; bound it rather than pin it.
VQLS_COST_CEILING = 1e-3


# -- pytest wiring -------------------------------------------------------------
#
# The ``--update-baseline`` flag and the session-finish writer are declared in
# tests/conftest.py, because pytest only honours ``pytest_addoption`` and
# ``pytest_sessionfinish`` from conftest files and plugins - never from a test
# module. Placing them here would have failed silently: the flag would be
# rejected as unknown and the baseline would never be written.


@pytest.fixture(scope="session")
def updating(request) -> bool:
    return bool(request.config.getoption("--update-baseline", default=False))


@pytest.fixture(scope="session")
def baseline(updating) -> Dict[str, Any]:
    if updating:
        return {}
    if not BASELINE_FILE.exists():
        pytest.skip(
            f"No baseline at {BASELINE_FILE}. Generate it on the baseline "
            f"commit with: pytest {Path(__file__).name} --update-baseline"
        )
    return json.loads(BASELINE_FILE.read_text())


# -- Provenance ----------------------------------------------------------------

def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent.parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _provenance() -> Dict[str, Any]:
    versions = {}
    for mod in ("numpy", "scipy", "qiskit", "qiskit_aer", "pennylane", "pyqsp"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return {
        "git_sha":  _git_sha(),
        "python":   platform.python_version(),
        "platform": platform.platform(),
        "versions": versions,
    }


# -- Canonical cases -----------------------------------------------------------
#
# Deliberately restricted in scale. This suite is calibrated to execute within
# a latency bound acceptable for pre-commit verification; if it exceeds this,
# it loses utility as an active safeguard. Comprehensive coverage belongs in
# the benchmark scripts; this serves exclusively as an integration tripwire.

CASES_1D = [
    # (case_id, N, source_fn, solver)
    ("thomas_N4_fS", 4, "fS", "thomas"),
    ("thomas_N8_fS", 8, "fS", "thomas"),
    ("qsvt_N4_fS",   4, "fS", "qsvt"),
    ("hhl_N4_fS",    4, "fS", "hhl"),
    ("vqls_N4_fS",   4, "fS", "vqls"),
]


def _solve_1d(N: int, source_fn: str, solver: str) -> Dict[str, Any]:
    """Run one canonical 1-D solve and reduce it to comparable scalars."""
    from core.config import SimConfig1D
    from problems.poisson_1d import PoissonProblem1D

    problem = PoissonProblem1D(SimConfig1D(N=N, epsilon=0.01, source_fn=source_fn))

    if solver == "thomas":
        from solvers.classical.thomas import thomas_solve
        result = thomas_solve(problem)
    elif solver == "hhl":
        from solvers.quantum.hhl_1d import hhl_solve
        result = hhl_solve(problem)
    elif solver == "vqls":
        from solvers.quantum.vqls_1d import vqls_solve
        result = vqls_solve(problem)
    elif solver == "qsvt":
        from solvers.quantum.qsvt_1d import qsvt_solve
        result = qsvt_solve(problem)
    else:  # pragma: no cover
        raise ValueError(solver)

    record: Dict[str, Any] = {
        "u":        np.asarray(result.u, dtype=float).tolist(),
        "residual": _maybe_float(result.euclidean_residual),
    }
    for extra in ("polynomial_degree", "circuit_depth", "n_qubits",
                  "alpha", "kappa_effective", "final_cost"):
        if hasattr(result, extra):
            record[extra] = _maybe_float(getattr(result, extra))
    return record


def _maybe_float(v):
    if v is None:
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    return float(v)


# -- 1-D solver lock -----------------------------------------------------------

@pytest.mark.quantum
@pytest.mark.parametrize("case_id,N,source_fn,solver", CASES_1D,
                         ids=[c[0] for c in CASES_1D])
def test_1d_solver_output_unchanged(case_id, N, source_fn, solver,
                                    baseline, updating, record_property):
    """
    Validates that the 1D solver output remains consistent with the established baseline.
    Ensures structural quantities remain exact while permitting bounded stochastic variation for VQLS algorithms.
    """
    actual = _solve_1d(N, source_fn, solver)

    if updating:
        _stash(case_id, actual)
        pytest.skip("baseline updated")

    expected = baseline.get("cases", {}).get(case_id)
    assert expected is not None, (
        f"case {case_id!r} missing from the baseline file — regenerate it "
        f"on the baseline commit rather than adding the value by hand"
    )

    record_property("baseline_sha", baseline.get("provenance", {}).get("git_sha"))

    if solver == "vqls":
        # Stochastic optimiser: bound the quality, do not pin the vector.
        assert actual["final_cost"] <= VQLS_COST_CEILING, (
            f"VQLS cost {actual['final_cost']:.3e} exceeds the ceiling "
            f"{VQLS_COST_CEILING:.0e}; the optimiser has regressed"
        )
        np.testing.assert_allclose(
            actual["u"], expected["u"], rtol=5e-2,
            err_msg="VQLS solution moved by more than 5% — not attributable "
                    "to optimiser stochasticity alone",
        )
        return

    np.testing.assert_allclose(
        actual["u"], expected["u"], rtol=RTOL_EXACT, atol=ATOL_EXACT,
        err_msg=f"{case_id}: solution vector changed",
    )
    if expected.get("residual") is not None:
        assert actual["residual"] == pytest.approx(
            expected["residual"], rel=RTOL_EXACT, abs=ATOL_EXACT
        ), f"{case_id}: residual changed"

    # Circuit-shape quantities are integers and must match exactly: a change
    # here means the circuit itself changed, which is never incidental.
    for key in ("polynomial_degree", "circuit_depth", "n_qubits"):
        if expected.get(key) is not None:
            assert actual[key] == expected[key], (
                f"{case_id}: {key} changed from {expected[key]} to "
                f"{actual[key]} — the circuit construction has been altered"
            )


# -- Outer-scheme lock ---------------------------------------------------------
#
# Transcribed from logged runs predating this file. They pin the legacy Jacobi
# path, which the 2026-08-07 outer-solver rewrite was required to reproduce
# identically, confirming the structural safety of that refactoring.

EXPECTED_OUTER = {
    #  N : (iterations, final_delta, tol)
    4: (26, 9.617e-07, 5e-3),
    8: (73, 9.450e-07, 5e-3),
}


@pytest.mark.quantum
@pytest.mark.parametrize("N", sorted(EXPECTED_OUTER))
def test_legacy_jacobi_iteration_count_unchanged(N):
    """
    Validates that the legacy Jacobi outer loop terminates at the historical iteration count.
    Iteration count provides a discrete measure that is highly sensitive to changes in the stopping criterion, strip ordering, or update rule.
    """
    from solvers.outer import solve
    from problems.poisson_line_2d import PoissonLine2D

    # Inlined rather than imported from tests.conftest.build_square_2d: this
    # project's local tests/ has no __init__.py, so it is an implicit
    # namespace package, and `from tests.conftest import ...` is resolved
    # against *every* directory named `tests` on sys.path -- including any
    # accidentally-installed site-packages `tests` package, which can
    # shadow the local one entirely. Six lines duplicated here is cheaper
    # and more robust than depending on cross-test-module imports, which
    # no other file in this suite relies on either.
    h = 1.0 / (N + 1)
    p = np.arange(1, N + 1) * h
    X, Y = np.meshgrid(p, p, indexing="ij")
    f = np.sin(np.pi * X) * np.sin(np.pi * Y)
    problem = PoissonLine2D(f)

    result = solve(problem, inner="thomas", scheme="jacobi", tol=1e-6)

    expected_iters, expected_delta, delta_rtol = EXPECTED_OUTER[N]
    assert result.n_outer == expected_iters, (
        f"legacy Jacobi at N={N} took {result.n_outer} outer iterations, "
        f"expected {expected_iters}"
    )
    # The legacy loop's stopping test is on the update delta, which the scheme
    # records in diagnostics; OuterResult.residual is the true relative
    # residual and is a different quantity.
    final_delta = result.diagnostics.get("final_delta", result.residual)
    assert final_delta == pytest.approx(expected_delta, rel=delta_rtol)


# -- Baseline generation -------------------------------------------------------

_STASH: Dict[str, Any] = {}


def _stash(case_id: str, record: Dict[str, Any]) -> None:
    _STASH[case_id] = record