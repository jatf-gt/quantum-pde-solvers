"""
Post-processing layer for the equal-accuracy and parameter-sensitivity studies.

Purpose
-------
`hpc/runners/run_studies.py` writes `equal_accuracy.json` and
`sensitivity_<solver>.json` into `results/<dim>Dstudies/`, and
`benchmark/tables.py` renders those archives as LaTeX. Nothing rendered them as
figures. This module closes that gap, and is the direct counterpart of
`benchmark/hpc_plotting.py` for the primary sweeps: orchestration only, reading
archives and writing figures, so it is cheap and safe to re-run at any time.

Two study designs, two readings
-------------------------------
The **equal-accuracy** protocol fixes a target relative residual r_target and
searches each solver's parameter grid for the cheapest configuration reaching it.
Holding accuracy fixed and comparing cost is the only comparison in which a
difference in cost is attributable to the algorithm rather than to one solver
having terminated with a worse answer. A configuration that never reaches the
target is reported as such rather than dropped: an algorithm that cannot be made
accurate enough at any setting is a result, not a gap in the data.

The **sensitivity** study varies one parameter at a time about a fixed baseline.
Its informative ordinate differs by dimension, and conflating the two misreads
the data:

  - In 1-D the residual is the residual of the linear system the solver actually
    solved, so it responds directly to the parameter and is the natural ordinate.
  - In 2-D and 3-D the reported residual belongs to the *outer* iteration, which
    terminates on its own tolerance. Every configuration precise enough to let
    the outer loop close therefore reports approximately the same residual, and a
    residual-against-parameter curve is flat by construction. The comparison
    there is cost at matched accuracy, and the finding is the cost, not the
    residual.

Both readings are served by plotting the *algorithmic* error alongside cost.
Every study record separates

    err_alg  = max_rel_err_vs_thomas   the solver's own error, in %
    err_disc = max_rel_err(Thomas)     the discretisation error, in %

so the figures carry the discretisation error as a horizontal reference. It is
the floor: driving err_alg below err_disc buys no accuracy in the solution of the
PDE, only in the solution of the linear system, and every parameter increase past
that crossing is spent for nothing. That crossing is the point of these studies.

References
----------
  Ghafourpour & Laizet (2025) Phys. Rev. Applied 24, 024032.
  Bravo-Prieto et al. (2023) Quantum 7, 1188.
  Saad, Y. (2003). Iterative Methods for Sparse Linear Systems, 2nd ed. SIAM.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

from benchmark.equal_accuracy import EqualAccuracyResult
from benchmark.results_io import SweepArchive
from benchmark.sensitivity import SensitivitySweepResult

log = logging.getLogger("study_plotting")

# ── Presentation Constants ─────────────────────────────────────────────────────

# Shared with benchmark/hpc_plotting.py so that a solver keeps one colour across
# every figure in the thesis. Restated rather than imported: importing that module
# binds its matplotlib globals as a side effect, which is precisely the backend
# coupling its own `_matplotlib` helper exists to avoid.
SOLVER_COLOUR: dict[str, str] = {
    "thomas": "#000000",
    "hhl":    "#1f77b4",
    "vqls":   "#2ca02c",
    "qsvt":   "#d62728",
}

SOLVER_ORDER: tuple[str, ...] = ("hhl", "vqls", "qsvt")

# Marker per case, so a single panel can carry several cases legibly in print.
CASE_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X")

# Axis labels for every parameter the sweeps vary. A parameter absent from this
# table falls back to its raw identifier rather than raising, so that adding a
# sweep does not require editing the plotting layer before it can be inspected.
PARAM_LABELS: dict[str, str] = {
    "epsilon":     r"HHL precision parameter $\epsilon$  ($n_T = \lceil 1/\epsilon \rceil$)",
    "n_layers":    "VQLS ansatz layers",
    "n_restarts":  "VQLS optimiser restarts",
    "cobyla_tol":  "COBYLA termination tolerance",
    "max_degree":  "QSVT polynomial degree cap",
    "trotter_steps": "HHL Trotter steps",
}

# Parameters swept over several decades, which read correctly only on a log axis.
LOG_PARAMS: frozenset[str] = frozenset({"epsilon", "cobyla_tol"})

# Sentinel recorded for an uncapped QSVT degree. The sweep stores the absence of a
# cap as a negative value so that the column stays numeric; it is placed at the
# right-hand end of the axis, where an uncapped run belongs.
UNCAPPED_SENTINEL: float = -1.0


# ── Private Utility Methods ────────────────────────────────────────────────────

def _matplotlib():
    """
    Import pyplot, selecting a headless backend only where none is yet fixed.

    Mirrors `benchmark.hpc_plotting._matplotlib`. Forcing Agg unconditionally
    would break `benchmark/plotting.py`'s interactive `plt.show()` for any caller
    that had already imported pyplot in this interpreter.

    Returns
    -------
    module
        The `matplotlib.pyplot` module.
    """
    import sys

    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi":      140,
        "savefig.dpi":     300,
        "font.size":       9,
        "axes.grid":       True,
        "grid.alpha":      0.3,
        "legend.frameon":  False,
        "axes.titlesize":  10,
    })
    return plt


def _param_axis(values: list[float]) -> tuple[list[float], list[str], bool]:
    """
    Map recorded parameter values onto plotting positions and tick labels.

    The uncapped QSVT sentinel has no place on a numeric axis, so every sweep
    containing it is plotted against an ordinal index with explicit tick labels.
    Sweeps without it keep their true numeric abscissa, which preserves the
    spacing that makes a diminishing-returns curve readable.

    Parameters
    ----------
    values : list of float
        Recorded `sensitivity_value` entries, in sweep order.

    Returns
    -------
    positions : list of float
        Abscissa for each value.
    labels : list of str
        Tick label for each position, or an empty list where the numeric axis is
        retained.
    ordinal : bool
        Whether an ordinal axis was substituted.
    """
    # An uncapped run is recorded two ways across the sweeps: as the negative
    # sentinel in 1-D, where the column had to stay numeric, and as a plain None
    # in 2-D/3-D. Both mean the same thing and both are unplottable on a numeric
    # abscissa, so either forces the ordinal axis.
    if any(v is None or v <= UNCAPPED_SENTINEL for v in values):
        positions = list(range(len(values)))
        labels = [
            "uncapped" if (v is None or v <= UNCAPPED_SENTINEL) else f"{v:g}"
            for v in values
        ]
        return positions, labels, True
    return [float(v) for v in values], [], False


def _err_alg(result) -> Optional[float]:
    """
    Extract the algorithmic error [%] from one study record.

    Prefers the error measured against the Thomas solution of the *same*
    discretisation, which isolates the solver's own error from the truncation
    error of the stencil. Falls back to the error against the analytical solution
    only where no Thomas reference was recorded, in which case the figure reports
    the two errors summed and the caption must say so.

    Parameters
    ----------
    result : BenchmarkResult
        One record from a sensitivity sweep or an equal-accuracy grid.

    Returns
    -------
    float or None
        Algorithmic error in per cent, or None where neither field is populated.
    """
    if result.max_rel_err_vs_thomas is not None:
        return result.max_rel_err_vs_thomas
    return result.max_rel_err_vs_exact


def _err_disc(result) -> Optional[float]:
    """
    Extract the discretisation error [%] recorded alongside one study record.

    Parameters
    ----------
    result : BenchmarkResult
        One record from a sensitivity sweep or an equal-accuracy grid.

    Returns
    -------
    float or None
        Truncation error of the stencil in per cent, or None where the record
        predates the column.
    """
    return getattr(result, "err_disc", None)


def _write_csv(path: Path, header: list[str], rows: list[list]) -> Path:
    """
    Write one tidy data table beside its figure.

    Every figure this module produces is accompanied by the series it draws, so
    that a final rendering in another tool plots identical numbers rather than
    values re-derived from the archives by a second, independently written path.

    Parameters
    ----------
    path : Path
        Destination file.
    header : list of str
        Column names.
    rows : list of list
        Data rows, one per plotted point.

    Returns
    -------
    Path
        The path written.
    """
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


# ── Equal-Accuracy Figures ─────────────────────────────────────────────────────

def plot_equal_accuracy(
    ea_results: list[EqualAccuracyResult],
    out_dir: Path,
    dim: int,
) -> list[Path]:
    """
    Render the equal-accuracy comparison: cost at a matched residual target.

    Two panels. The left panel is the result: the wall time of the cheapest
    configuration reaching r_target, one bar per (case, solver). The right panel
    is the methodological warrant for reading the left one — the residual each
    solver actually achieved, against the target and its acceptance band. A bar
    in the left panel is meaningful only if its counterpart in the right panel
    lies inside the band; those that do not are hatched and marked, never hidden.

    Parameters
    ----------
    ea_results : list of EqualAccuracyResult
        One record per (case, solver) from the equal-accuracy sweep.
    out_dir : Path
        Destination directory for the figure and its data table.
    dim : int
        Spatial dimension, used in the title and the file stem.

    Returns
    -------
    list of Path
        Files written: the PNG, the vector PDF and the CSV.
    """
    if not ea_results:
        return []

    plt = _matplotlib()

    cases = sorted({r.best_result.case_id for r in ea_results})
    solvers = [s for s in SOLVER_ORDER
               if any(r.solver.lower() == s for r in ea_results)]
    r_target = ea_results[0].r_target
    band = ea_results[0].band_factor

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    width = 0.8 / max(len(solvers), 1)

    csv_rows: list[list] = []
    for s_idx, solver in enumerate(solvers):
        offs = (s_idx - (len(solvers) - 1) / 2.0) * width
        xs, costs, hatches = [], [], []
        for c_idx, case in enumerate(cases):
            rec = next((r for r in ea_results
                        if r.solver.lower() == solver
                        and r.best_result.case_id == case), None)
            if rec is None:
                continue
            cost = rec.best_result.wall_time_s
            if cost is None or cost <= 0.0:
                continue
            xs.append(c_idx + offs)
            costs.append(cost)
            hatches.append("" if rec.in_band else "///")
            csv_rows.append([
                case, solver, rec.best_result.N, rec.r_target,
                rec.best_result.residual, cost,
                rec.best_result.sensitivity_param,
                rec.best_result.sensitivity_value,
                rec.in_band, rec.n_solver_calls,
            ])
            axes[1].semilogy(
                c_idx + offs, rec.best_result.residual,
                marker="o" if rec.in_band else "x",
                ms=8, color=SOLVER_COLOUR[solver], ls="none",
            )
        if not xs:
            continue
        bars = axes[0].bar(xs, costs, width=width * 0.9,
                           color=SOLVER_COLOUR[solver],
                           label=solver.upper(), edgecolor="black", lw=0.5)
        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)

    axes[0].set_yscale("log")
    axes[0].set_ylabel("wall time of cheapest configuration reaching "
                       r"$r_\mathrm{target}$  [s]")
    axes[0].set_title("(a)  Cost at matched accuracy")

    axes[1].axhline(r_target, color="black", ls="--", lw=1.2,
                    label=r"$r_\mathrm{target}$")
    axes[1].axhspan(r_target / band, r_target * band, color="grey", alpha=0.18,
                    label="acceptance band")
    axes[1].set_ylabel(r"achieved residual  $\|Au-b\|_2 / \|b\|_2$")
    axes[1].set_title("(b)  Residual actually achieved")
    axes[1].legend(fontsize=8, loc="best")

    for ax in axes:
        ax.set_xticks(range(len(cases)))
        ax.set_xticklabels([c.replace("_", "\n") for c in cases], fontsize=7)
        ax.grid(alpha=0.3, which="both", axis="y")
    axes[0].legend(fontsize=8, title="Solver", title_fontsize=8)

    fig.suptitle(
        f"Equal-accuracy protocol, {dim}-D  "
        rf"($r_\mathrm{{target}} = {r_target:.0e}$; hatched bars did not reach "
        "the band)",
        fontweight="bold", fontsize=10,
    )
    fig.tight_layout()

    written: list[Path] = []
    stem = out_dir / f"fig_equal_accuracy_{dim}D"
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight")
        written.append(stem.with_suffix(suffix))
    plt.close(fig)

    written.append(_write_csv(
        out_dir / f"data_equal_accuracy_{dim}D.csv",
        ["case", "solver", "N", "r_target", "residual_achieved",
         "wall_time_s", "param", "param_value", "in_band", "n_solver_calls"],
        csv_rows,
    ))
    return written


# ── Sensitivity Figures ────────────────────────────────────────────────────────

def plot_sensitivity(
    sweeps: list[SensitivitySweepResult],
    solver: str,
    out_dir: Path,
    dim: int,
) -> list[Path]:
    """
    Render the one-at-a-time sensitivity sweeps for a single solver.

    One row of panels per swept parameter. The left panel carries the algorithmic
    error against the parameter, with the discretisation error of the same case
    drawn as a horizontal reference; the right panel carries the wall time. Read
    together they locate the point past which additional resource buys precision
    in the linear solve that the discretisation discards.

    In 2-D and 3-D the recorded residual is the outer iteration's and is floored
    by its tolerance, so the error panel there reports the algorithmic error
    against Thomas rather than the residual, which would be flat by construction.

    Parameters
    ----------
    sweeps : list of SensitivitySweepResult
        Every sweep recorded for this solver, across parameters and cases.
    solver : str
        Solver name, lower case.
    out_dir : Path
        Destination directory for the figure and its data table.
    dim : int
        Spatial dimension, used in the title and the file stem.

    Returns
    -------
    list of Path
        Files written: the PNG, the vector PDF and the CSV.
    """
    if not sweeps:
        return []

    plt = _matplotlib()

    params = sorted({s.param_name for s in sweeps})
    cases = sorted({r.case_id for s in sweeps for r in s.results})

    fig, axes = plt.subplots(
        len(params), 2,
        figsize=(10.5, 3.3 * len(params)),
        squeeze=False,
    )

    csv_rows: list[list] = []
    for p_idx, param in enumerate(params):
        ax_err, ax_cost = axes[p_idx][0], axes[p_idx][1]
        ordinal_labels: list[str] = []
        ordinal = False

        for c_idx, case in enumerate(cases):
            sweep = next((s for s in sweeps
                          if s.param_name == param
                          and any(r.case_id == case for r in s.results)), None)
            if sweep is None:
                continue
            records = [r for r in sweep.results if r.case_id == case]
            if not records:
                continue

            values = [r.sensitivity_value for r in records]
            positions, labels, ordinal = _param_axis(values)
            if labels:
                ordinal_labels = labels

            marker = CASE_MARKERS[c_idx % len(CASE_MARKERS)]

            errs = [_err_alg(r) for r in records]
            pairs = [(p, e) for p, e in zip(positions, errs)
                     if e is not None and e > 0.0]
            if pairs:
                ax_err.semilogy(*zip(*pairs), marker=marker, lw=1.6,
                                color=SOLVER_COLOUR.get(solver, "grey"),
                                mfc="none", label=case)

            # The discretisation error is a property of the case and resolution,
            # not of the swept parameter, so it is constant across the panel and
            # is drawn as the floor rather than as a series.
            #
            # Drawn only where it lies within two decades of the algorithmic
            # errors it is meant to bound. A case whose analytical solution the
            # stencil represents exactly -- the linear HET profile 3a, whose
            # truncation error is at machine precision -- otherwise stretches a
            # logarithmic ordinate across fifteen decades and compresses every
            # curve in the panel into a single line.
            disc = next((_err_disc(r) for r in records
                         if _err_disc(r) is not None), None)
            plotted = [e for _, e in pairs]
            if (disc is not None and disc > 0.0 and plotted
                    and disc > min(plotted) / 100.0):
                ax_err.axhline(disc, ls=":", lw=1.2, color="black", alpha=0.7,
                               label=f"{case}: discretisation floor")

            times = [r.wall_time_s for r in records]
            tpairs = [(p, t) for p, t in zip(positions, times)
                      if t is not None and t > 0.0]
            if tpairs:
                ax_cost.semilogy(*zip(*tpairs), marker=marker, lw=1.6,
                                 color=SOLVER_COLOUR.get(solver, "grey"),
                                 mfc="none", label=case)

            for pos, val, err, t, rec in zip(positions, values, errs, times,
                                             records):
                csv_rows.append([
                    case, solver, param,
                    "uncapped" if (val is None or val <= UNCAPPED_SENTINEL)
                    else val,
                    pos, rec.N, err, _err_disc(rec), rec.residual, t,
                ])

        label = PARAM_LABELS.get(param, param)
        for ax in (ax_err, ax_cost):
            ax.set_xlabel(label)
            ax.grid(alpha=0.3, which="both")
            if ordinal and ordinal_labels:
                ax.set_xticks(range(len(ordinal_labels)))
                ax.set_xticklabels(ordinal_labels, fontsize=7)
            elif param in LOG_PARAMS:
                ax.set_xscale("log")
        ax_err.set_ylabel(r"algorithmic error $e_\infty$ against Thomas  [%]")
        ax_cost.set_ylabel("wall time  [s]")
        ax_err.set_title(f"Accuracy — varying {param}")
        ax_cost.set_title(f"Cost — varying {param}")
        ax_err.legend(fontsize=6.5, loc="best")

    fig.suptitle(
        f"Parameter sensitivity, {solver.upper()}, {dim}-D  "
        "(one parameter at a time about the baseline configuration)",
        fontweight="bold", fontsize=10,
    )
    fig.tight_layout()

    written: list[Path] = []
    stem = out_dir / f"fig_sensitivity_{solver}_{dim}D"
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight")
        written.append(stem.with_suffix(suffix))
    plt.close(fig)

    written.append(_write_csv(
        out_dir / f"data_sensitivity_{solver}_{dim}D.csv",
        ["case", "solver", "param", "param_value", "plot_position", "N",
         "err_alg_pct", "err_disc_pct", "residual", "wall_time_s"],
        csv_rows,
    ))
    return written


# ── Driver ─────────────────────────────────────────────────────────────────────

def run_studies(study_dir: Path, dim: int) -> list[Path]:
    """
    Render every figure the study archives in one directory support.

    Absent archives are skipped rather than treated as an error: the 2-D and 3-D
    studies deliberately record QSVT alone, because HHL and VQLS cost hours per
    grid point at those dimensions, and a missing `sensitivity_hhl.json` there is
    the documented scope rather than a failed run.

    Parameters
    ----------
    study_dir : Path
        A `results/<dim>Dstudies/` directory.
    dim : int
        Spatial dimension of the studies held there.

    Returns
    -------
    list of Path
        Every file written.

    Raises
    ------
    SystemExit
        If the directory does not exist, which is a mistyped path rather than an
        incomplete run and is worth reporting plainly.
    """
    study_dir = Path(study_dir)
    if not study_dir.is_dir():
        raise SystemExit(f"No study directory at {study_dir}.")

    out_dir = study_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    archive = SweepArchive(study_dir)
    written: list[Path] = []

    ea_results = archive.read_equal_accuracy()
    if ea_results:
        written += plot_equal_accuracy(ea_results, out_dir, dim)
        log.info("  equal-accuracy      %d record(s)", len(ea_results))
    else:
        log.info("  equal-accuracy      absent; skipped")

    for solver in SOLVER_ORDER:
        sweeps = archive.read_sensitivity(solver)
        if not sweeps:
            log.info("  sensitivity %-7s absent; skipped", solver)
            continue
        written += plot_sensitivity(sweeps, solver, out_dir, dim)
        log.info("  sensitivity %-7s %d sweep(s), %d record(s)",
                 solver, len(sweeps), sum(len(s.results) for s in sweeps))

    return written
