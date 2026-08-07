"""
solvers/quantum/vqls_1d_4th.py
-------------------------------
VQLS solver for the fourth-order 1D Poisson system.

This is an intentionally thin wrapper.  It extracts (A, b) from a
PoissonProblem1D4th instance and delegates to the existing
vqls_solve_system() in vqls_1d.py.  All circuit construction, Pauli
decomposition, cost function evaluation, and optimisation logic is
inherited unchanged.

The pentadiagonal matrix A_pent is symmetric and Hermitian — the same
structural properties as the TST matrix — so vqls_utils.py's Pauli
decomposition works without modification.  The condition number is
~2.5× higher at the same N, which may require a slightly deeper ansatz
(n_layers=3 rather than the default 2) for comparable accuracy.

Usage
-----
    from problems.poisson_1d_4th import PoissonProblem1D4th
    from solvers.quantum.vqls_1d_4th import vqls_solve_4th

    prob   = PoissonProblem1D4th(N=4, source_fn='fS')
    result = vqls_solve_4th(prob)
    print(result.u)
"""

from __future__ import annotations

import numpy as np

from problems.poisson_1d_4th import PoissonProblem1D4th
from solvers.quantum.vqls_1d import vqls_solve_system, VQLSConfig1D
from solvers.quantum.result import VQLSSolverResult


def vqls_solve_4th(
    problem: PoissonProblem1D4th,
    config: VQLSConfig1D | None = None,
) -> VQLSSolverResult:
    """
    Solve the fourth-order 1D Poisson system using VQLS.

    Parameters
    ----------
    problem : PoissonProblem1D4th
        The fourth-order discretised problem.
    config : VQLSConfig1D, optional
        VQLS configuration.  Defaults to VQLSConfig1D(n_layers=3) —
        one layer deeper than the second-order default to accommodate
        the higher condition number of the pentadiagonal matrix.

    Returns
    -------
    VQLSSolverResult
        Identical result type to the second-order VQLS solver, with
        fields: u, cost_history, n_circuit_evals, converged, wall_time.
    """
    if config is None:
        config = VQLSConfig1D(n_layers=4)

    return vqls_solve_system(problem.A, problem.b, config=config)