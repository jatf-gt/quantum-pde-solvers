"""
benchmark.py
------------
Error metrics and result containers for the 1D Poisson benchmark.

This module is deliberately free of any solver or Qiskit imports — it only
does post-processing arithmetic and printing.  That keeps it fast and easy
to test independently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from config import SimConfig, EXACT_SOLUTIONS
from problem_setup import PoissonProblem1D
from solvers import SolverResult


# ── Threshold below which an analytical value is treated as "near zero" ───────
# Nodes where |u_exact| < this are excluded from relative-error calculations.
# The paper does this implicitly for the fH central nodes (Section IV A).
_NEAR_ZERO_TOL = 1e-10


# ── Per-run result container ──────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """
    All error metrics for one solver on one problem configuration.

    Attributes
    ----------
    config          : the SimConfig that generated this result
    solver          : 'Thomas' or 'HHL'
    x               : interior node coordinates
    u_solver        : solution vector from the solver
    u_exact         : analytical solution at interior nodes (None if
                      no closed form is available, e.g. non-homogeneous BCs
                      with a source function that has no known antiderivative)
    u_thomas        : Thomas solution, stored alongside HHL results so the
                      two can be compared node-by-node
    rel_error       : pointwise relative errors (NaN where |u_exact| < tol)
    abs_error       : pointwise absolute errors
    max_rel_error   : maximum relative error (excluding near-zero nodes)
    avg_rel_error   : mean relative error (excluding near-zero nodes)
    max_abs_error   : maximum absolute error
    avg_abs_error   : mean absolute error
    euclidean_residual : ||Au - b|| / ||b|| from the solver
    prop_const      : HHL proportionality constant c (None for Thomas)
    """
    config:           SimConfig
    solver:           str
    x:                np.ndarray
    u_solver:         np.ndarray
    u_exact:          Optional[np.ndarray]
    u_thomas:         Optional[np.ndarray]
    rel_error:        Optional[np.ndarray]
    abs_error:        np.ndarray
    max_rel_error:    Optional[float]
    avg_rel_error:    Optional[float]
    max_abs_error:    float
    avg_abs_error:    float
    euclidean_residual: Optional[float]
    prop_const:       Optional[float] = None


# ── Error computation ─────────────────────────────────────────────────────────

def compute_errors(
    problem:       PoissonProblem1D,
    result:        SolverResult,
    u_thomas:      Optional[np.ndarray] = None,
) -> BenchmarkResult:
    """
    Compute all error metrics for one solver result.

    For homogeneous BCs (alpha = beta = 0) the analytical solution is
    available from EXACT_SOLUTIONS and relative errors are computed.
    For non-homogeneous BCs we fall back to absolute errors only, since
    the paper does not provide closed-form solutions for those cases and
    the analytical solution would need to be computed numerically.

    Parameters
    ----------
    problem  : the PoissonProblem1D that was solved
    result   : the SolverResult from thomas_solve or hhl_solve
    u_thomas : the Thomas solution vector, passed in when processing the
               HHL result so both can be stored together for comparison
    """
    cfg = problem.config
    x   = problem.x
    u   = result.u

    # ── Analytical solution (homogeneous BCs only) ────────────────────────────
    # We only have closed-form solutions for the homogeneous case.
    # For non-homogeneous runs the paper reports absolute errors against
    # the Thomas solution, so u_exact is left as None there.
    has_exact = (cfg.alpha == 0.0 and cfg.beta == 0.0 and
                 cfg.source_fn in EXACT_SOLUTIONS)

    if has_exact:
        u_exact = EXACT_SOLUTIONS[cfg.source_fn](x)
    else:
        u_exact = None

    # ── Absolute errors ───────────────────────────────────────────────────────
    if u_exact is not None:
        abs_error = np.abs(u - u_exact)
    elif u_thomas is not None:
        # For non-homogeneous runs, report absolute error against Thomas.
        abs_error = np.abs(u - u_thomas)
    else:
        # Thomas solver itself — absolute error against exact if available,
        # otherwise against NumPy (not computed here; left as zeros).
        abs_error = np.zeros_like(u)

    max_abs = float(np.max(abs_error))
    avg_abs = float(np.mean(abs_error))

    # ── Relative errors ───────────────────────────────────────────────────────
    # Only meaningful when we have an analytical reference.
    # Nodes where |u_exact| < _NEAR_ZERO_TOL are masked out (set to NaN)
    # to avoid spurious blow-up, matching the paper's treatment of the
    # fH central nodes.
    if u_exact is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_error = np.where(
                np.abs(u_exact) > _NEAR_ZERO_TOL,
                np.abs(u - u_exact) / np.abs(u_exact) * 100.0,  # as %
                np.nan,
            )
        valid = rel_error[~np.isnan(rel_error)]
        max_rel = float(np.max(valid))   if valid.size > 0 else None
        avg_rel = float(np.mean(valid))  if valid.size > 0 else None
    else:
        rel_error = None
        max_rel   = None
        avg_rel   = None

    return BenchmarkResult(
        config=cfg,
        solver=result.solver,
        x=x,
        u_solver=u,
        u_exact=u_exact,
        u_thomas=u_thomas,
        rel_error=rel_error,
        abs_error=abs_error,
        max_rel_error=max_rel,
        avg_rel_error=avg_rel,
        max_abs_error=max_abs,
        avg_abs_error=avg_abs,
        euclidean_residual=result.euclidean_residual,
        prop_const=result.prop_const,
    )


# ── Console reporting ─────────────────────────────────────────────────────────

def print_result_table(results: list[BenchmarkResult]) -> None:
    """
    Print a formatted comparison table to stdout.

    The layout mirrors the paper's figures: each row is one solver/config
    combination, columns show the key error metrics.
    """
    header = (
        f"{'Solver':<8} {'N':>4} {'f':>3} {'α':>5} {'β':>5} "
        f"{'ε':>8} {'MaxRel%':>9} {'AvgRel%':>9} "
        f"{'MaxAbs':>10} {'AvgAbs':>10} {'Residual':>10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        cfg = r.config
        max_rel = f"{r.max_rel_error:9.3f}" if r.max_rel_error is not None else f"{'N/A':>9}"
        avg_rel = f"{r.avg_rel_error:9.3f}" if r.avg_rel_error is not None else f"{'N/A':>9}"
        residual = f"{r.euclidean_residual:.3e}" if r.euclidean_residual is not None else "N/A"
        print(
            f"{r.solver:<8} {cfg.N:>4} {cfg.source_fn:>3} "
            f"{cfg.alpha:>5.2f} {cfg.beta:>5.2f} "
            f"{cfg.epsilon:>8.4f} "
            f"{max_rel} {avg_rel} "
            f"{r.max_abs_error:>10.6f} {r.avg_abs_error:>10.6f} "
            f"{residual:>10}"
        )

    print(sep)


def print_hhl_summary(hhl_result: BenchmarkResult) -> None:
    """
    Print a concise per-node breakdown for an HHL result, useful for
    debugging individual runs and matching against the paper's figures.
    """
    cfg = hhl_result.config
    print(
        f"\nHHL node-by-node: N={cfg.N}, f={cfg.source_fn}, "
        f"ε={cfg.epsilon}, α={cfg.alpha}, β={cfg.beta}"
    )
    print(f"  Proportionality constant c = {hhl_result.prop_const:.6f}")
    print(f"  Euclidean residual ||Au-b||/||b|| = {hhl_result.euclidean_residual:.4e}")
    print(f"\n  {'x_i':>8}  {'u_HHL':>12}  {'u_exact':>12}  {'RelErr%':>10}")
    for i, xi in enumerate(hhl_result.x):
        u_h = hhl_result.u_solver[i]
        u_e = hhl_result.u_exact[i] if hhl_result.u_exact is not None else float("nan")
        re  = hhl_result.rel_error[i] if hhl_result.rel_error is not None else float("nan")
        print(f"  {xi:8.4f}  {u_h:12.6f}  {u_e:12.6f}  {re:10.3f}")