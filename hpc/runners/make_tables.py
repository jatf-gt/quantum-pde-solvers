#!/usr/bin/env python3
"""
Publication tables and hardware-feasibility reporting from recorded sweeps.

Purpose
-------
`benchmark/tables.py` renders LaTeX and console tables from typed
`BenchmarkResult` objects, and `benchmark/hardware.py` estimates whether a
circuit could run on real hardware. Neither was reachable from the recorded
sweeps: those write the runners' own row schema to `results_full.json`, not
`BenchmarkResult`. This module is the driver that closes that gap, adapting the
archives through `benchmark.hpc_archive.rows_to_benchmark_results` and emitting
every table the thesis needs.

It is pure post-processing — it reads archives and writes text — so it runs on a
login node in seconds and needs no PBS job.

Hardware feasibility
--------------------
`estimate_hardware_feasibility` is an order-of-magnitude heuristic in circuit
depth and qubit count. Where the sweep recorded a *measured* depth and two-qubit
gate count, that measurement is used instead and the estimate is reported only as
a fallback, with the source stated per row. A benchmark that has measured its own
circuits should not report a heuristic as though it were a result.

The feasibility criterion follows the constraints in the framework design notes:
two-qubit gate count dominates the error budget on real hardware, so a circuit is
judged against a two-qubit gate budget rather than total depth. With a per-gate
error rate ε₂, the probability that a circuit of n₂ two-qubit gates completes
without a fault is ≈ (1 − ε₂)^n₂, so the budget for a target success probability
p is n₂ ≤ ln(p) / ln(1 − ε₂). At ε₂ = 1e-3 and p = 0.5 this gives ≈ 693 gates,
which is the default below.

Outputs
-------
Written under ``<results-dir>/tables/``:

  tab_primary_comparison.tex   Errors, residual, depth, qubits, wall time.
  tab_circuit_resources.tex    Depth and gate counts alone.
  tab_equal_accuracy.tex       Cost at matched residual, where studies exist.
  tab_sensitivity_<solver>.tex One per solver swept.
  tab_order_comparison.tex     2nd against 4th order, where both are present.
  tab_hardware_feasibility.tex Two-qubit gate count against the budget.
  tables_console.txt           The same content as aligned plain text.

References
----------
Preskill, J. (2018). Quantum Computing in the NISQ era and beyond.
    Quantum, 2, 79.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Optional

# `pytest.ini` sets `pythonpath = .`, but a bare `python3 hpc/runners/make_tables.py`
# puts `hpc/runners/` on sys.path[0] rather than the repository root. Resolving the
# root from `__file__` decouples the import path from the invocation directory.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark import tables as T                                   # noqa: E402
from benchmark.hardware import estimate_hardware_feasibility        # noqa: E402
from benchmark.hpc_archive import SweepArchive as LegacyArchive     # noqa: E402
from benchmark.hpc_archive import rows_to_benchmark_results         # noqa: E402
from benchmark.results_io import SweepArchive as StudyArchive       # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s",
                    stream=sys.stdout)
log = logging.getLogger("tables")

# Default results directory per (dimension, order), matching the sweep layout.
RESULTS_DIR: dict[tuple[int, int], str] = {
    (1, 2): "results/1Dhpc_run",   (1, 4): "results/1Dhpc_run_4th",
    (2, 2): "results/2Dhpc_run",   (2, 4): "results/2Dhpc_run_4th",
    (3, 2): "results/3Dhpc_run",   (3, 4): "results/3Dhpc_run_4th",
}
STUDY_DIR: dict[int, str] = {
    1: "results/1Dstudies", 2: "results/2Dstudies", 3: "results/3Dstudies",
}

# -- Hardware feasibility ------------------------------------------------------

# Two-qubit gate error rate representative of current superconducting hardware.
# Superseded by a device's own calibration where one is available; utilised here
# exclusively to derive the budget below, which serves as the reported metric.
DEFAULT_TWO_QUBIT_ERROR: float = 1.0e-3

# Target probability that a circuit completes without a two-qubit fault.
DEFAULT_SUCCESS_TARGET: float = 0.5


def two_qubit_budget(error_rate: float = DEFAULT_TWO_QUBIT_ERROR,
                     success_target: float = DEFAULT_SUCCESS_TARGET) -> int:
    """
    Largest two-qubit gate count meeting a target fault-free probability.

    Treating two-qubit faults as independent with per-gate error ε₂, a circuit of
    n₂ such gates completes without a fault with probability (1 − ε₂)^n₂. Setting
    that equal to the target p and solving gives n₂ = ln(p) / ln(1 − ε₂).

    This is deliberately a bound on TWO-QUBIT gates rather than on total depth.
    Two-qubit gates dominate the error budget on superconducting hardware by
    roughly an order of magnitude, so a depth bound that counts single-qubit
    rotations equally misstates the constraint.

    Parameters
    ----------
    error_rate : float
        Per-gate two-qubit error rate ε₂, in (0, 1).
    success_target : float
        Target fault-free probability p, in (0, 1).

    Returns
    -------
    int
        Gate budget, floored to an integer.

    Raises
    ------
    ValueError
        If either argument lies outside (0, 1).
    """
    if not 0.0 < error_rate < 1.0:
        raise ValueError(f"error_rate must lie in (0, 1), got {error_rate}.")
    if not 0.0 < success_target < 1.0:
        raise ValueError(f"success_target must lie in (0, 1), got {success_target}.")
    return int(math.log(success_target) / math.log(1.0 - error_rate))


def hardware_feasibility_rows(results: list, budget: int) -> list[dict]:
    """
    Judge each recorded solve against the two-qubit gate budget.

    Measured circuit metrics are preferred over the heuristic estimate wherever
    the sweep recorded them, and the source is stated per row so that a
    heuristic figure is never mistaken for a measurement. Where a two-qubit count
    was not recorded but a depth was, the depth is used as a bound: it can only
    overestimate the two-qubit count, so a circuit judged infeasible on depth is
    infeasible on gate count too, while the converse is not claimed.

    Parameters
    ----------
    results : list of BenchmarkResult
        Adapted sweep rows.
    budget : int
        Two-qubit gate budget from `two_qubit_budget`.

    Returns
    -------
    list of dict
        One row per (case, solver, N) carrying the count, its provenance, and
        the verdict, sorted by solver then N.
    """
    rows: list[dict] = []
    for r in results:
        if r.solver == "thomas":
            continue                      # The Thomas algorithm is a classical solver; it generates no quantum circuit.

        n_two_qubit: Optional[int] = None
        source = "estimated"
        cm = r.circuit_metrics
        if cm is not None and cm.n_cx_gates is not None:
            n_two_qubit, source = cm.n_cx_gates, "measured (2q count)"
        elif cm is not None and cm.depth_opt1 is not None:
            n_two_qubit, source = cm.depth_opt1, "measured (depth, upper bound)"
        else:
            # The estimate is supplied with whatever circuit data the row recorded. Its VQLS branch
            # otherwise assumes a 2-layer ansatz against the 6-14 the sweeps run,
            # understating the circuit by up to 7x, and its QSVT branch infers a
            # degree from kappa that ignores any cap the run applied.
            try:
                est = estimate_hardware_feasibility(
                    r.N, r.solver, r.kappa or 1.0,
                    n_layers=r.vqls_n_layers,
                    polynomial_degree=r.qsvt_polynomial_degree,
                )
                n_two_qubit = est.get("estimated_two_qubit")
            except Exception:
                n_two_qubit = None

        if n_two_qubit is None:
            continue

        rows.append({
            "case": r.case_id, "solver": r.solver, "N": r.N,
            "n_qubits": cm.n_qubits if cm is not None else None,
            "n_two_qubit": int(n_two_qubit),
            "source": source,
            "budget": budget,
            "feasible": int(n_two_qubit) <= budget,
            "overshoot": round(int(n_two_qubit) / budget, 3),
        })

    rows.sort(key=lambda d: (d["solver"], d["N"], d["case"]))
    return rows


def latex_hardware_feasibility(rows: list[dict], budget: int) -> str:
    """
    LaTeX table of two-qubit gate count against the hardware budget.

    Written here rather than in `benchmark/tables.py` because it consumes the
    feasibility dicts above rather than `BenchmarkResult`, and because the budget
    is a property of an assumed device rather than of the benchmark.

    Parameters
    ----------
    rows : list of dict
        As returned by `hardware_feasibility_rows`.
    budget : int
        Two-qubit gate budget, quoted in the caption.

    Returns
    -------
    str
        A complete booktabs table.
    """
    out = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Two-qubit gate count against a hardware budget of "
        rf"{budget:,} gates, the largest count retaining a 50\% probability of "
        r"completing without a two-qubit fault at a per-gate error rate of "
        r"$10^{-3}$. Counts marked \emph{measured} are taken from the recorded "
        r"circuits; where only a depth was recorded it is used as an upper "
        r"bound on the two-qubit count.}",
        r"  \label{tab:hardware_feasibility}",
        r"  \begin{tabular}{llrrrrl}",
        r"    \toprule",
        r"    Case & Solver & $N$ & $n_q$ & $n_{2Q}$ & $n_{2Q}/\mathrm{budget}$"
        r" & Feasible \\",
        r"    \midrule",
    ]
    for d in rows:
        nq = "---" if d["n_qubits"] is None else f"{d['n_qubits']}"
        mark = r"\checkmark" if d["feasible"] else r"$\times$"
        case = str(d["case"]).replace("_", r"\_")
        out.append(
            rf"    {case} & {d['solver'].upper()} & {d['N']} & {nq} & "
            rf"{d['n_two_qubit']:,} & {d['overshoot']:.3g} & {mark} \\"
        )
    out += [r"    \bottomrule", r"  \end{tabular}", r"\end{table}"]
    return "\n".join(out) + "\n"


def console_hardware_feasibility(rows: list[dict], budget: int) -> str:
    """
    Generate an aligned plain-text rendering of the hardware feasibility assessment.

    Parameters
    ----------
    rows : list of dict
        As returned by `hardware_feasibility_rows`.
    budget : int
        Two-qubit gate budget, reported in the table header.

    Returns
    -------
    str
        Formatted ASCII table.
    """
    sep = "  " + "-" * 90
    out = [sep,
           f"  HARDWARE FEASIBILITY  -  two-qubit gate budget = {budget:,}",
           sep,
           f"  {'Case':32s}{'Solver':8s}{'N':>5s}{'nq':>5s}"
           f"{'n_2Q':>12s}{'ratio':>9s}  {'source':28s}",
           sep]
    for d in rows:
        nq = "---" if d["n_qubits"] is None else str(d["n_qubits"])
        verdict = "OK " if d["feasible"] else "NO "
        out.append(
            f"  {str(d['case'])[:31]:32s}{d['solver'].upper():8s}{d['N']:>5d}"
            f"{nq:>5s}{d['n_two_qubit']:>12,}{d['overshoot']:>9.3g}  "
            f"{verdict}{d['source']}"
        )
    out.append(sep)
    return "\n".join(out) + "\n"


# -- Driver --------------------------------------------------------------------

def load_sweep(results_dir: Path, dim: int, order: int) -> list:
    """
    Read one sweep archive and adapt it to typed BenchmarkResult objects.

    Returns an empty list rather than raising when the directory holds no summary,
    because several (dimension, order) combinations are legitimately absent at any
    given time; a missing sweep should skip its tables rather than abort the run.

    Parameters
    ----------
    results_dir : Path
        Sweep output directory.
    dim : int
        Spatial dimension, selecting the error-field convention in the adapter.
    order : int
        Discretisation order stamped on rows that predate the column.

    Returns
    -------
    list of BenchmarkResult
        Adapted rows, or an empty list if the directory holds no summary.
    """
    if not (results_dir / "results_full.json").exists():
        log.info("  %-28s no results_full.json; skipped", str(results_dir))
        return []
    archive = LegacyArchive(results_dir, dim=dim, skip_scheme_comparison=True)
    rows = list(archive.rows())
    adapted = rows_to_benchmark_results(rows, dim=dim, order=order)
    log.info("  %-28s %d rows", str(results_dir), len(adapted))
    return adapted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate publication tables from recorded HPC sweeps.")
    parser.add_argument("--dim", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--order", type=int, choices=(2, 4), default=2)
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Sweep directory; defaults to the standard layout "
                             "for --dim/--order.")
    parser.add_argument("--study-dir", type=Path, default=None,
                        help="Parameter-study directory; defaults per dimension.")
    parser.add_argument("--compare-order", action="store_true",
                        help="Also emit the 2nd-against-4th order table, reading "
                             "both sweeps for this dimension.")
    parser.add_argument("--two-qubit-error", type=float,
                        default=DEFAULT_TWO_QUBIT_ERROR,
                        help="Per-gate two-qubit error rate setting the budget.")
    parser.add_argument("--success-target", type=float,
                        default=DEFAULT_SUCCESS_TARGET,
                        help="Target fault-free probability setting the budget.")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="Where to write; defaults to <results-dir>/tables.")
    args = parser.parse_args()

    results_dir = args.results_dir or Path(RESULTS_DIR[(args.dim, args.order)])
    study_dir = args.study_dir or Path(STUDY_DIR[args.dim])
    output_dir = args.output_dir or (results_dir / "tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 78)
    log.info("  PUBLICATION TABLES  -  %d-D  order %d", args.dim, args.order)
    log.info("=" * 78)

    primary = load_sweep(results_dir, args.dim, args.order)
    if not primary:
        log.error("  No rows to tabulate; nothing written.")
        return 1

    results_2nd = results_4th = None
    if args.compare_order:
        results_2nd = load_sweep(Path(RESULTS_DIR[(args.dim, 2)]), args.dim, 2)
        results_4th = load_sweep(Path(RESULTS_DIR[(args.dim, 4)]), args.dim, 4)
        if not (results_2nd and results_4th):
            log.warning("  Order comparison needs both sweeps; skipping it.")
            results_2nd = results_4th = None

    # Parameter studies are optional: the primary tables stand without them.
    ea_results: list = []
    sens_results: dict[str, list] = {}
    r_target = 1.0e-3
    if study_dir.exists():
        study = StudyArchive(study_dir)
        try:
            ea_results = study.read_equal_accuracy()
        except Exception as exc:
            log.warning("  equal-accuracy study unreadable (%s); skipped", exc)
        for solver in ("hhl", "vqls", "qsvt"):
            try:
                sweeps = study.read_sensitivity(solver)
            except Exception:
                sweeps = []
            if sweeps:
                sens_results[solver] = sweeps
        if ea_results:
            r_target = ea_results[0].r_target
        log.info("  %-28s %d equal-accuracy, %d sensitivity sweep(s)",
                 str(study_dir), len(ea_results),
                 sum(len(v) for v in sens_results.values()))
    else:
        log.info("  %-28s absent; study tables skipped", str(study_dir))

    written = T.save_latex_tables(
        output_dir=output_dir,
        primary_results=primary,
        ea_results=ea_results or None,
        sensitivity_results=sens_results or None,
        results_2nd=results_2nd,
        results_4th=results_4th,
        r_target=r_target,
    )

    # -- Hardware feasibility ------------------------------------------------------
    budget = two_qubit_budget(args.two_qubit_error, args.success_target)
    hw_rows = hardware_feasibility_rows(primary, budget)

    hw_tex = output_dir / "tab_hardware_feasibility.tex"
    hw_tex.write_text(latex_hardware_feasibility(hw_rows, budget),
                      encoding="utf-8")
    written.append(hw_tex)

    hw_json = output_dir / "hardware_feasibility.json"
    hw_json.write_text(
        json.dumps({"budget": budget,
                    "two_qubit_error": args.two_qubit_error,
                    "success_target": args.success_target,
                    "rows": hw_rows}, indent=2),
        encoding="utf-8")
    written.append(hw_json)

    # -- Console rendering ---------------------------------------------------------
    console = [T.console_primary_comparison(primary)]
    if ea_results:
        console.append(T.console_equal_accuracy(ea_results, r_target))
    # console_sensitivity and latex_sensitivity both take the LIST of sweeps for
    # one solver, rendering one block per swept parameter, not a single sweep.
    for solver, sweeps in sens_results.items():
        N = next((s.results[0].N for s in sweeps if s.results), 0)
        console.append(T.console_sensitivity(sweeps, solver, N))
    console.append(console_hardware_feasibility(hw_rows, budget))

    console_path = output_dir / "tables_console.txt"
    console_path.write_text("\n".join(console), encoding="utf-8")
    written.append(console_path)

    log.info("-" * 78)
    n_feasible = sum(1 for d in hw_rows if d["feasible"])
    log.info("  Hardware: %d of %d circuits within the %d-gate budget",
             n_feasible, len(hw_rows), budget)
    for path in written:
        log.info("    wrote %s", path)
    log.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
