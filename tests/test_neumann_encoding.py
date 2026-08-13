"""
Regression cover for the block-encoding selection on non-Toeplitz operators.

Both quantum 1-D entry points historically reconstructed their operator from
``A[0,0]`` and ``A[0,1]`` alone. That reconstruction is exact for the Toeplitz
Symmetric Tridiagonal matrix every generic Poisson case assembles, and it is
silently wrong for any tridiagonal matrix whose diagonals are not constant.

Sub-case 3c is the instance in this repository: a Neumann condition at x = 0,
halved to keep the operator symmetric, gives ``A[0,0] = -1`` against ``-2``
everywhere else. The reconstruction therefore built ``tridiag(1, -1, 1)`` — a
uniformly shifted operator, not the Neumann one — and HHL and QSVT solved that
instead, at every N and every polynomial degree. The recorded solutions matched
the surrogate's solution to machine precision while sitting at ~100 % relative
error against 3c's own, and the pre-existing band check passed the matrix
cleanly because its bandwidth was never the problem.

These tests pin the three properties that together close that failure mode:
the structural predicate distinguishes the two cases, the solvers now agree with
the classical reference on 3c, and the Toeplitz path is left bit-for-bit
unchanged so the published 2nd-order figures still reproduce.
"""
from __future__ import annotations

import numpy as np
import pytest

from core import cases
from solvers.quantum.block_encoding import (
    assert_tridiagonal,
    is_toeplitz_tridiagonal,
)

# Degree cap for the QSVT solves below. 3c's Neumann row raises κ well above the
# Dirichlet cases at the same N (29.3 at N=4 against 9.5), and the angle solve
# cost grows as ~O(d^2.5), so the cap is what keeps this test in seconds. It is
# chosen against the degree/κ ratio that governs accuracy, measured here:
#
#     N    κ        cap    degree/κ    error vs classical    angle solve
#     4    29.3     400       13.7          1.7e-06              8.9 s
#     4    29.3     600       20.5          2.8e-09             20.4 s
#     8   113.5     800        7.0          4.2e-04             34.6 s
#     8   113.5    1200       10.6          9.8e-06             78.6 s
#
# This reproduces the threshold near 11 seen across the order-2 sweep; 400 at N=4
# sits comfortably above it. N=8 is not exercised for QSVT: it would need ~1200
# to clear the threshold, at 79 s, and it verifies nothing beyond what the N=4 case already establishes.
_MAX_DEGREE = 400

_TOEPLITZ_CASES = (
    "poisson_1d_fS_hom",
    "poisson_1d_fH_hom",
    "poisson_1d_fL_hom",
    "poisson_1d_fS_nonhom",
    "het_1d_3a_linear",
    "het_1d_3b_gaussian_Vd300",
)


def _built(case_key: str, N: int):
    """Assemble one case as (A, b, u_exact, u_classical)."""
    bc = cases.get(case_key).build(N)
    A = np.asarray(bc.A, dtype=float)
    b = np.asarray(bc.b, dtype=float)
    exact = None if bc.exact is None else np.asarray(bc.exact, dtype=float)
    return A, b, exact, np.linalg.solve(A, b)


def _rel_l2(u: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(u - ref) / np.linalg.norm(ref))


# ── Structural predicate ──────────────────────────────────────────────────────

@pytest.mark.parametrize("case_key", _TOEPLITZ_CASES)
def test_generic_cases_are_toeplitz(case_key):
    """Every case validated against the two-scalar path must continue to satisfy the Toeplitz predicate."""
    A, _, _, _ = _built(case_key, 8)
    assert is_toeplitz_tridiagonal(A)


def test_neumann_case_is_not_toeplitz():
    """3c must be recognised as outside the two-scalar path's domain."""
    A, _, _, _ = _built("het_1d_3c_neumann", 8)
    assert not is_toeplitz_tridiagonal(A)


def test_band_check_alone_does_not_catch_the_neumann_row():
    """
    The failure's defining property: 3c is tridiagonal, so a bandwidth test
    cannot distinguish it. Pinning this keeps the Toeplitz check from being
    mistaken for a duplicate of the band check and removed.
    """
    A, _, _, _ = _built("het_1d_3c_neumann", 8)
    idx = np.arange(A.shape[0])
    off_band = np.abs(idx[:, None] - idx[None, :]) > 1
    assert np.max(np.abs(A[off_band])) == 0.0


def test_assert_tridiagonal_rejects_the_neumann_operator():
    """The guard must name the Toeplitz violation, not merely refuse."""
    A, _, _, _ = _built("het_1d_3c_neumann", 8)
    with pytest.raises(ValueError, match="not Toeplitz"):
        assert_tridiagonal(A, "QSVT")


# ── Solver behaviour ──────────────────────────────────────────────────────────

@pytest.mark.quantum
def test_qsvt_solves_the_neumann_case():
    """
    QSVT must now agree with the classical solution of 3c's OWN operator.

    Before the encoding selection was added this stood at ~100 % error, and the
    returned vector reproduced the solution of the Toeplitz surrogate instead.
    """
    from solvers.quantum.qsvt_1d import QSVTConfig1D, qsvt_solve_system

    N = 4
    A, b, exact, u_classical = _built("het_1d_3c_neumann", N)
    res = qsvt_solve_system(
        A, b, config=QSVTConfig1D(epsilon=0.01, max_degree=_MAX_DEGREE))
    u = np.asarray(res.u, dtype=float)

    assert _rel_l2(u, u_classical) < 1e-4
    # With the correct operator encoded, the residual error relative to the analytical solution reflects only the discretisation error, not the solver.
    assert _rel_l2(u, exact) == pytest.approx(_rel_l2(u_classical, exact),
                                              rel=1e-2)


@pytest.mark.quantum
def test_hhl_solves_the_neumann_case():
    """
    HHL must now agree with the classical solution of 3c's own operator.

    The tolerance is looser than QSVT's because HHL carries Trotter error at
    ε = 0.01 on top of the encoding; the point is the two orders of magnitude
    between this and the ~1.0 that the Toeplitz surrogate produced.

    N=4 only. The general `NumPyMatrix` evolution is dense, so N=8 costs 29 s
    against 1.1 s here and demonstrates nothing further — the surrogate was
    wrong at every N, and one resolution is sufficient to establish that the defect is resolved.
    """
    from solvers.quantum.hhl_1d import hhl_solve_system

    N = 4
    A, b, _, u_classical = _built("het_1d_3c_neumann", N)
    u, _raw, _c = hhl_solve_system(A, b, 0.01)

    assert _rel_l2(np.asarray(u, dtype=float), u_classical) < 0.05


@pytest.mark.quantum
def test_toeplitz_path_is_unchanged_by_the_selection():
    """
    `encoding="auto"` must be bit-for-bit identical to `encoding="tst"` wherever
    the latter was valid, so the published 2nd-order figures still reproduce.
    """
    from solvers.quantum.qsvt_1d import QSVTConfig1D, qsvt_solve_system

    A, b, _, _ = _built("poisson_1d_fS_hom", 4)
    cfg = dict(config=QSVTConfig1D(epsilon=0.01, max_degree=_MAX_DEGREE))
    u_auto = np.asarray(qsvt_solve_system(A, b, **cfg).u, dtype=float)
    u_tst = np.asarray(
        qsvt_solve_system(A, b, encoding="tst", **cfg).u, dtype=float)

    np.testing.assert_array_equal(u_auto, u_tst)
