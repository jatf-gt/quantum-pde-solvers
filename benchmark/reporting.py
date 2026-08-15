"""
Console reporting utilities for the benchmarking framework.

Provides structured, aligned tabular output for interactive inspection of
benchmark results during HPC runs and post-processing sessions. All output
is formatted for 120-character terminal width.

This module is a thin layer over benchmark/tables.py: it calls the console
table builders and adds run-level summary statistics and diagnostic flags.
"""

from __future__ import annotations

import sys
from typing import Optional

from benchmark.metrics import BenchmarkResult
from benchmark.equal_accuracy import EqualAccuracyResult
from benchmark.sensitivity import SensitivitySweepResult
from benchmark import tables


# -- Diagnostic thresholds -----------------------------------------------------

_WARN_RESIDUAL:    float = 1.0e-1   # residual above this triggers a warning
_WARN_REL_ERR:     float = 20.0     # max relative error [%] above this triggers a warning
_WARN_PROP_RESID:  float = 0.5      # proportionality recovery residual threshold


def print_primary_comparison(
    results: list[BenchmarkResult],
    N_values: Optional[list[int]] = None,
    solvers: Optional[list[str]] = None,
    show_circuit: bool = True,
    file=sys.stdout,
) -> None:
    """
    Print the primary comparison table to the console.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Benchmark results.
    N_values : list[int], optional
        Problem sizes to include.
    solvers : list[str], optional
        Solvers to include.
    show_circuit : bool
        If True, include circuit depth and qubit count columns.
    file : file-like object
        Output stream. Defaults to stdout.
    """
    table_str = tables.console_primary_comparison(
        results, N_values=N_values, solvers=solvers, show_circuit=show_circuit
    )
    print(table_str, file=file)
    _print_diagnostics(results, file=file)


def print_equal_accuracy(
    ea_results: list[EqualAccuracyResult],
    r_target: float,
    file=sys.stdout,
) -> None:
    """
    Print the equal-accuracy comparison table to the console.

    Parameters
    ----------
    ea_results : list[EqualAccuracyResult]
        One result per solver.
    r_target : float
        Target residual.
    file : file-like object
        Output stream.
    """
    print(tables.console_equal_accuracy(ea_results, r_target), file=file)

    # Flag out-of-band results
    for ear in ea_results:
        if not ear.in_band:
            print(
                f"  [WARN] {ear.solver.upper()} did not achieve "
                f"r_target={r_target:.2e} within the parameter grid. "
                f"Best residual: {ear.best_result.residual:.4e}. "
                f"Notes: {ear.notes}",
                file=file,
            )


def print_sensitivity(
    sweep_results: list[SensitivitySweepResult],
    solver: str,
    N: int,
    file=sys.stdout,
) -> None:
    """
    Print the sensitivity analysis table for one solver.

    Parameters
    ----------
    sweep_results : list[SensitivitySweepResult]
        All OAT sweeps for the solver.
    solver : str
        Solver name.
    N : int
        Problem size.
    file : file-like object
        Output stream.
    """
    print(tables.console_sensitivity(sweep_results, solver, N), file=file)


def print_run_summary(
    results: list[BenchmarkResult],
    ea_results: Optional[list[EqualAccuracyResult]] = None,
    file=sys.stdout,
) -> None:
    """
    Print a high-level run summary: counts, timing, and diagnostic flags.

    Parameters
    ----------
    results : list[BenchmarkResult]
        All primary benchmark results.
    ea_results : list[EqualAccuracyResult], optional
        Equal-accuracy results, if available.
    file : file-like object
        Output stream.
    """
    sep = "─" * 72
    print(f"\n  {sep}", file=file)
    print(f"  RUN SUMMARY", file=file)
    print(f"  {sep}", file=file)

    n_total = len(results)
    n_failed = sum(
        1 for r in results
        if r.residual is None or r.residual > _WARN_RESIDUAL
    )
    total_time = sum(r.wall_time_s for r in results if r.wall_time_s)

    print(f"  Total results:    {n_total}", file=file)
    print(f"  High-residual:    {n_failed}  (r > {_WARN_RESIDUAL:.0e})", file=file)
    print(f"  Total wall time:  {total_time:.1f} s  ({total_time/60:.1f} min)", file=file)

    # Per-solver summary
    solvers = sorted({r.solver for r in results})
    print(f"\n  Per-solver:", file=file)
    for solver in solvers:
        rows = [r for r in results if r.solver == solver]
        t_total = sum(r.wall_time_s for r in rows)
        residuals = [r.residual for r in rows if r.residual is not None]
        r_mean = sum(residuals) / len(residuals) if residuals else float("nan")
        print(
            f"    {solver.upper():<8}  n={len(rows):3d}  "
            f"mean_r={r_mean:.3e}  total_t={t_total:.1f}s",
            file=file,
        )

    if ea_results:
        print(f"\n  Equal-accuracy:", file=file)
        for ear in ea_results:
            status = "IN BAND" if ear.in_band else "OUT OF BAND"
            print(
                f"    {ear.solver.upper():<8}  r_target={ear.r_target:.2e}  "
                f"best_r={ear.best_result.residual:.4e}  [{status}]",
                file=file,
            )

    print(f"  {sep}\n", file=file)


def _print_diagnostics(
    results: list[BenchmarkResult],
    file=sys.stdout,
) -> None:
    """
    Print diagnostic warnings for results that exceed alert thresholds.
    """
    warnings = []

    for r in results:
        if r.residual is not None and r.residual > _WARN_RESIDUAL:
            warnings.append(
                f"  [WARN] {r.solver.upper()} N={r.N} case={r.case_id}: "
                f"residual={r.residual:.3e} > threshold {_WARN_RESIDUAL:.0e}"
            )
        if (r.max_rel_err_vs_exact is not None
                and r.max_rel_err_vs_exact > _WARN_REL_ERR):
            warnings.append(
                f"  [WARN] {r.solver.upper()} N={r.N} case={r.case_id}: "
                f"max_rel_err={r.max_rel_err_vs_exact:.2f}% > {_WARN_REL_ERR:.0f}%"
            )
        if (r.proportionality_residual is not None
                and r.proportionality_residual > _WARN_PROP_RESID):
            warnings.append(
                f"  [WARN] {r.solver.upper()} N={r.N} case={r.case_id}: "
                f"proportionality_residual={r.proportionality_residual:.3e} "
                f"> {_WARN_PROP_RESID:.1f} — check recovery step"
            )

    if warnings:
        print(f"\n  Diagnostic alerts:", file=file)
        for w in warnings:
            print(w, file=file)
        print(file=file)