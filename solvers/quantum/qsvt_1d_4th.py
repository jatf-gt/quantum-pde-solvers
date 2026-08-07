"""
-------------------------------
QSVT solver for the fourth-order 1D Poisson system.

The pentadiagonal matrix A_pent produced by PoissonProblem1D4th is
symmetric, Hermitian, and negative definite — the same structural
properties as the TST matrix.  The Sz.-Nagy block encoding in
block_encoding.py therefore applies without modification: it only
requires ||A / alpha||_2 <= 1, satisfied by alpha = ||A||_2.

The main difference from the second-order QSVT case is the larger
subnormalisation factor alpha:

    alpha_tri  = ||A_tri||_2  ≈ 4 / pi^2 * (N+1)^2  * (2/h^2)
    alpha_pent = ||A_pent||_2 ≈ (30/12) * alpha_tri  ≈ 2.5 * alpha_tri

This increases the effective condition number seen by the QSP phase
angle computation:

    kappa_eff = kappa(A_pent)

and therefore the polynomial degree required for the 1/x approximation.
For small N (4, 8) this is still tractable on a laptop.

Architecture
------------
This module is intentionally thin.  It:
  1. Builds the block encoding from block_encoding.py (unchanged).
  2. Computes QSP phase angles from qsp_angles.py (unchanged).
  3. Assembles and runs the QSVT circuit from qsvt_1d.py (unchanged).

No new circuit primitives are needed.  The entire QSVT machinery
generalises to the pentadiagonal case for free.

Usage
-----
    from problems.poisson_1d_4th import PoissonProblem1D4th
    from solvers.quantum.qsvt_1d_4th import qsvt_solve_4th, QSVTConfig1D4th

    prob   = PoissonProblem1D4th(N=4, source_fn='fS')
    result = qsvt_solve_4th(prob)
    print(result.u)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from problems.poisson_1d_4th import PoissonProblem1D4th
from solvers.quantum.qsvt_1d import qsvt_solve_system, QSVTConfig1D
from solvers.quantum.result import QSVTSolverResult


@dataclass
class QSVTConfig1D4th:
    """
    Configuration for the fourth-order QSVT solver.

    Mirrors QSVTConfig1D but with defaults tuned for the pentadiagonal
    matrix's higher condition number.

    Parameters
    ----------
    epsilon : float
        Approximation tolerance for the 1/x polynomial.  The polynomial
        degree scales as O(kappa / epsilon), so tighter epsilon means
        more phase angles and a deeper circuit.  Default 0.05 is looser
        than the second-order default (0.01) to keep N=4/8 tractable
        on a laptop.
    max_degree : int or None
        Hard cap on the polynomial degree passed to the phase angle
        computation.  None means uncapped (use pyqsp's own degree
        selection).  For the pentadiagonal case at N=8 (kappa~42),
        the uncapped degree is ~1500-2000; a cap of 500 gives a
        fast proof-of-concept at the cost of some accuracy.
    angle_method : str
        Phase angle computation method passed to qsp_angles.py.
        'auto' selects the best available method (pyqsp if installed,
        Chebyshev fallback otherwise).
    """
    epsilon: float = 0.05
    max_degree: Optional[int] = None
    angle_method: str = "auto"

    def to_qsvt_config(self) -> QSVTConfig1D:
        """Convert to the standard QSVTConfig1D used by qsvt_solve_system."""
        return QSVTConfig1D(
            epsilon=self.epsilon,
            max_degree=self.max_degree,
            angle_method=self.angle_method,
        )


def qsvt_solve_4th(
    problem: PoissonProblem1D4th,
    config: QSVTConfig1D4th | None = None,
) -> QSVTSolverResult:
    """
    Solve the fourth-order 1D Poisson system using QSVT.

    Delegates entirely to qsvt_solve_system() from qsvt_1d.py, which
    accepts any (A, b) pair and handles block encoding, phase angle
    computation, circuit assembly, and solution extraction internally.

    Parameters
    ----------
    problem : PoissonProblem1D4th
        The fourth-order discretised problem.
    config : QSVTConfig1D4th, optional
        QSVT configuration.  Defaults to QSVTConfig1D4th() with
        epsilon=0.05 and no degree cap.

    Returns
    -------
    QSVTSolverResult
        Same result type as the second-order QSVT solver.
        Key fields (from result.py): u, residual, wall_time,
        polynomial_degree, circuit_depth, n_circuit_evals.

    Notes
    -----
    For a proof-of-concept run at N=4 on a laptop, the default config
    (epsilon=0.05, uncapped) typically produces degree ~200-400 and
    completes in under a minute.  At N=8 (kappa~42), uncapped degree
    is ~1500-2000 and may take 5-20 minutes; pass
    QSVTConfig1D4th(max_degree=500) for a fast approximate result.
    """
    if config is None:
        config = QSVTConfig1D4th()

    return qsvt_solve_system(
        problem.A,
        problem.b,
        config=config.to_qsvt_config(),
    )