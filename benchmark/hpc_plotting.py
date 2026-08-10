"""
Post-processing and visualisation of HPC sweep output.

Consolidates the plotting layer for all three sweep drivers — `hpc/runners/run_1d.py`,
`run_2d.py` and `run_3d.py`. Nothing here re-runs a solve: it
reads only what a runner already wrote (`results_full.json` plus the archived
per-solution `.npz` files), so it is cheap and safe to re-run at any time, and
in particular after a walltime-killed job that lost its summary but not its
per-solution data.

These are diagnostic figures first and thesis figures second. The priority is
catching a wrong field or a suspicious trend quickly — a sign error, a misplaced
boundary condition, a solver that converged to the wrong fixed point — which a
table of norms hides and a picture does not.

Structure
---------
This module orchestrates: it loads through `benchmark/results_io.py`, reshapes,
draws and writes. It does not define the on-disk schema, and it no longer spells
archive field names at the point of use.

    results_io.SweepArchive   reading, grouping and figure destination for one
                              sweep directory, in any dimension
    Metric plots              convergence, accuracy vs N, cost vs N, quantum
                              overhead, error decomposition
    Dimension-specific plots  1-D profiles and summary tables; 2-D fields;
                              3-D orthogonal slices, polar unwrapping, cutaways
                              and azimuthal fidelity

`hpc/runners/plot_results.py --dim {1,2,3}` is the thin command-line wrapper
over the `run_1d`, `run_2d` and `run_3d` entry points at the end of this module.

What the three sweeps do and do not share
-----------------------------------------
2-D and 3-D share a result schema and therefore share their scalar-metric plots.
**1-D does not.** Its summary rows carry no `scheme`, `stop_reason`, `linf_err`,
`weighted_cost` or `err_vs_thomas`, which are exactly the fields those plots
read, so `run_1d` calls none of them and has its own parallel implementations in
a different visual style. Pointing a 2-D/3-D metric plot at a 1-D sweep raises
`KeyError`. This is a real schema divergence, not an oversight in the plotting
layer; earlier documentation claimed all three shared these plots, which was
true only of two.

A note on the two colour schemes
--------------------------------
`SOLVER_STYLE` (1-D) and `SOLVER_COLOUR` (2-D/3-D) assign different colours to
the same solvers. They are deliberately not unified: figures generated from both
already appear in the project record, and silently recolouring one family would
make previously produced figures disagree with newly produced ones for no
analytical gain. Unify them only as a deliberate, one-off restyling.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from benchmark.hpc_archive import SOLVER_ORDER, SweepArchive, field

# Bound by `_matplotlib()` on first use rather than imported here. Two reasons,
# both of which have bitten this code before: the Agg backend must be selected
# before `pyplot` is first imported, and forcing Agg at import time would break
# `benchmark/plotting.py`, whose laptop-scale figures call `plt.show()` and need
# an interactive backend. Importing this module to reach a sweep's contents — to
# list them, say — therefore requires no plotting stack at all, and
# `benchmark/results_io.py` imports none either.
plt = None
ticker = None


# ── Solver Presentation ───────────────────────────────────────────────────────

# SOLVER_ORDER is imported from benchmark.results_io: the canonical ordering is
# a property of the schema, not of the presentation, and both the grouping and
# the figure column order must agree on it.

# 2-D and 3-D palette.
SOLVER_COLOUR = {"Thomas": "#444444", "HHL": "#d62728",
                 "VQLS": "#2ca02c", "QSVT": "#1f77b4"}

# 1-D palette, with markers and line styles so the curves remain separable in
# greyscale print.
SOLVER_STYLE = {
    "Thomas": {"color": "#1f77b4", "marker": "o",  "ls": "-",  "label": "Thomas (classical)"},
    "HHL":    {"color": "#ff7f0e", "marker": "s",  "ls": "--", "label": "HHL"},
    "VQLS":   {"color": "#2ca02c", "marker": "^",  "ls": "-.", "label": "VQLS"},
    "QSVT":   {"color": "#d62728", "marker": "D",  "ls": ":",  "label": "QSVT"},
}

CASE_LABELS = {
    "1D_Poisson_fS_hom":              r"1D Poisson, $f_S$, hom. BCs",
    "1D_Poisson_fL_hom":              r"1D Poisson, $f_L$, hom. BCs",
    "1D_Poisson_fH_hom":              r"1D Poisson, $f_H$, hom. BCs",
    "1D_Poisson_fS_nonhom":           r"1D Poisson, $f_S$, non-hom. BCs",
    "HET_1D_3a_linear_hom":           r"HET 1D, linear profile, hom. BCs",
    "HET_1D_3b_gaussian_Vd300":       r"HET 1D, Gaussian, $V_d=300$ V",
    "HET_1D_3c_gaussian_NeumannDirichlet": r"HET 1D, Gaussian, Neumann–Dirichlet BCs",
}

# Publication-oriented style, applied by the 1-D entry point only. It is applied
# explicitly rather than at import so that importing this module never mutates
# a caller's global Matplotlib state — the 2-D and 3-D figures are produced
# under Matplotlib's defaults and must stay that way.
RC_PARAMS_1D = {
    "font.family":       "serif",
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    12,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "lines.linewidth":   1.8,
    "lines.markersize":  4,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
}

# Axis labels per 3-D case family. Anything not matching a HET prefix falls back
# to generic Cartesian labels.
AXIS_LABELS = {
    "het": ("axial z (mm)", "radial r (mm)", "azimuthal s (mm)"),
    "generic": ("x", "y", "z"),
}

# The three orthogonal cutplanes through a 3-D field, each fixing one axis at
# its midpoint. Together they are the primary diagnostic: a field that looks
# right in all three is very unlikely to be wrong.
#   (name, fixed axis, the two varying axes, whether this is the HET spoke view)
_PLANES = [
    ("axial-radial",     2, (0, 1), False),   # fix s (or z2), show (axis0, axis1)
    ("axial-azimuthal",  1, (0, 2), True),    # fix r,          show (axis0, axis2)
    ("radial-azimuthal", 0, (1, 2), False),   # fix z,          show (axis1, axis2)
]


# ── Sweep Reader ──────────────────────────────────────────────────────────────
#
# `SweepArchive` lives in benchmark/results_io.py. It is imported above and
# used unchanged: loading, grouping and the figure destination are properties
# of the sweep on disk, not of the plotting layer, and the runners are meant
# to share that declaration once they adopt it.


def save_fig(fig, sweep: SweepArchive, stem: str, save_pdf: bool = False) -> Path:
    """
    Writes a figure as PNG, and additionally as PDF when requested.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to write.
    sweep : SweepArchive
        Supplies the destination directory.
    stem : str
        Filename without extension.
    save_pdf : bool, default=False
        Also emit a vector PDF, for figures destined for the thesis.

    Returns
    -------
    Path
        Path of the written PNG.
    """
    sweep.plots_dir.mkdir(parents=True, exist_ok=True)
    png_path = sweep.plots_dir / f"{stem}.png"
    fig.savefig(png_path)
    print(f"  Saved: {png_path}")
    if save_pdf:
        pdf_path = sweep.plots_dir / f"{stem}.pdf"
        fig.savefig(pdf_path)
        print(f"  Saved: {pdf_path}")
    return png_path


def _warn_missing(sweep: SweepArchive, rows: list[dict]) -> None:
    """
    Reports summary rows whose solution archive is absent.

    Every plotting function treats a missing archive as "not run" and quietly
    omits it, which is correct for a partial sweep and badly wrong for a broken
    filename convention — the two are indistinguishable in the output. Stating
    the count up front separates them: a handful of gaps matching failed solves
    is expected, whereas every row missing means the reader is looking in the
    wrong place or for the wrong stem.

    Parameters
    ----------
    sweep : SweepArchive
        Sweep to inspect.
    rows : list of dict
        Summary records.
    """
    gaps = sweep.missing(rows)
    if not gaps:
        return
    print(f"  {len(gaps)} of {len(rows)} result rows have no solution archive.")
    if len(gaps) == len(rows):
        print(f"  ALL archives are missing. Expected stem "
              f"'{sweep.solution_prefix}_<case>_<solver>_N<N>.npz' in "
              f"{sweep.results_dir}; check the directory and the dimension.")
        return
    shown = ", ".join(f"{c}/{s}/N{n}" for c, s, n in gaps[:4])
    print(f"  e.g. {shown}{' …' if len(gaps) > 4 else ''}")
    print("  These are omitted from the figures; check the summary for "
          "'converged' and 'notes' on those rows.")


def _save_and_close(fig, sweep: SweepArchive, filename: str) -> Path:
    """
    Writes a 2-D or 3-D figure and releases it.

    The 2-D and 3-D families previously repeated this three-line block at ten
    separate call sites, with `dpi=130` spelled out each time. The values are
    kept exactly as they were: these figures are checked by byte comparison
    against a control run, so a changed `savefig` keyword would register as a
    regression.

    The 1-D family deliberately does not use this. It goes through `save_fig`,
    which honours `RC_PARAMS_1D` (300 dpi, tight bounding box, serif) and can
    additionally emit PDF, and it announces each file as it is written.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to write.
    sweep : SweepArchive
        Supplies the destination directory.
    filename : str
        Filename including extension.

    Returns
    -------
    Path
        Path of the written figure.
    """
    out = sweep.plots_dir / filename
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def load_solution_1d(sweep: SweepArchive, case: str, solver: str, N: int) -> dict | None:
    """
    Loads a 1-D solution, projected onto the keys the 1-D plots expect.

    Field names are resolved through `results_io.field` rather than spelled
    here, so this reads a 1-D archive (`u_solver`), a 2-D one (`phi_solver`) or
    a 3-D one (`phi`) without knowing which it holds.

    Parameters
    ----------
    sweep : SweepArchive
        Sweep to read from.
    case, solver : str
        Case and solver identifiers.
    N : int
        Resolution.

    Returns
    -------
    dict or None
        {'x', 'u'} and, when the case admits a closed form, 'u_exact';
        None if the combination was not run.
    """
    data = sweep.solution(case, solver, N)
    if data is None:
        return None
    out = {"x": data["x"], "u": field(data, "solution")}
    exact = field(data, "exact")
    if exact is not None:
        out["u_exact"] = exact
    return out


# ── Metric Plots (2-D and 3-D) ────────────────────────────────────────────────
#
# These read `scheme`, `linf_err`, `weighted_cost` and `err_vs_thomas`, which
# 1-D result rows do not carry. `run_1d` therefore calls none of them; see the
# module docstring on the schema divergence.

def plot_convergence(sweep, case: str, N: int, rows: list[dict], plt) -> Path | None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    any_curve = False
    for r in rows:
        sol = sweep.solution(case, r["solver"], N)
        if sol is None or "residual_history" not in sol:
            continue
        h = sol["residual_history"]
        if len(h) == 0:
            continue
        ax.semilogy(range(1, len(h) + 1), h, lw=1.8,
                    color=SOLVER_COLOUR.get(r["solver"]),
                    label=f"{r['solver']} ({r['scheme']})")
        any_curve = True
    if not any_curve:
        plt.close(fig)
        return None
    ax.set_xlabel("outer iteration (sweep or cycle)")
    ax.set_ylabel(r"$\|b - Au\|_2 / \|b\|_2$")
    ax.set_title(f"Convergence - {case}, N={N}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = _save_and_close(fig, sweep, f"convergence_{case}_N{N}.png")
    return out

def plot_accuracy_vs_n(sweep, case: str, by_solver: dict, plt) -> Path | None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    any_line = False
    N_all = []
    for solver, rs in sweep.series(by_solver, case):
        Ns = [r["N"] for r in rs if r.get("linf_err") is not None]
        errs = [r["linf_err"] for r in rs if r.get("linf_err") is not None]
        if not Ns:
            continue
        ax.loglog(Ns, errs, "o-", lw=1.8, color=SOLVER_COLOUR.get(solver),
                  label=solver)
        N_all.extend(Ns)
        any_line = True
    if not any_line:
        plt.close(fig)
        return None
    N_all = sorted(set(N_all))
    if len(N_all) >= 2:
        ref = [(N_all[0] / n) ** 2 for n in N_all]
        ref_all = [r * (max(ax.get_ylim()) * 0.5 / max(ref)) for r in ref]
        ax.loglog(N_all, ref_all, "k--", lw=1, alpha=0.5, label=r"$O(h^2)$ ref.")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$L_\infty$ error vs. reference (%)")
    ax.set_title(f"Accuracy vs N - {case}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = _save_and_close(fig, sweep, f"accuracy_vs_N_{case}.png")
    return out

def plot_cost_vs_n(sweep, case: str, by_solver: dict, plt) -> Path | None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    any_line = False
    for solver, rs in sweep.series(by_solver, case):
        Ns = [r["N"] for r in rs]
        wc = [r["weighted_cost"] for r in rs if r.get("weighted_cost") is not None]
        wt = [r["wall_time_s"] for r in rs]
        if wc and len(wc) == len(Ns):
            axes[0].loglog(Ns, wc, "o-", color=SOLVER_COLOUR.get(solver), label=solver)
        axes[1].loglog(Ns, wt, "o-", color=SOLVER_COLOUR.get(solver), label=solver)
        any_line = True
    if not any_line:
        plt.close(fig)
        return None
    axes[0].set_xlabel("N"); axes[0].set_ylabel("weighted strip-solve cost")
    axes[0].set_title("Cost (finest-solve units)")
    axes[1].set_xlabel("N"); axes[1].set_ylabel("wall time (s)")
    axes[1].set_title("Wall time")
    for a in axes:
        a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
    fig.suptitle(f"Cost vs N - {case}", fontweight="bold")
    plt.tight_layout()
    out = _save_and_close(fig, sweep, f"cost_vs_N_{case}.png")
    return out

def plot_overhead(sweep, case: str, by_N: dict, plt) -> Path | None:
    xs, ys, cs, labels = [], [], [], []
    for (c, N), rs in sorted(by_N.items()):
        if c != case:
            continue
        base = next((r["wall_time_s"] for r in rs if r["solver"] == "Thomas"), None)
        if not base:
            continue
        for r in rs:
            if r["solver"] == "Thomas" or not r["wall_time_s"]:
                continue
            xs.append(N); ys.append(r["wall_time_s"] / base)
            cs.append(SOLVER_COLOUR.get(r["solver"], "gray")); labels.append(r["solver"])
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    seen = set()
    for x, y, c, l in zip(xs, ys, cs, labels):
        ax.semilogy(x, y, "o", color=c, label=l if l not in seen else None,
                   markersize=8)
        seen.add(l)
    ax.set_xlabel("N")
    ax.set_ylabel("wall time / Thomas wall time")
    ax.set_title(f"Quantum overhead vs Thomas - {case}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = _save_and_close(fig, sweep, f"overhead_vs_thomas_{case}.png")
    return out

def plot_error_decomposition(sweep, case: str, by_solver: dict, plt) -> Path | None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    any_line = False
    disc_plotted = False
    for solver, rs in sweep.series(by_solver, case, exclude=("Thomas",)):
        Ns = [r["N"] for r in rs if r.get("err_vs_thomas") is not None]
        alg = [r["err_vs_thomas"] for r in rs if r.get("err_vs_thomas") is not None]
        if Ns:
            ax.semilogy(Ns, [max(v, 1e-6) for v in alg], "o-",
                       color=SOLVER_COLOUR.get(solver), label=f"{solver} (vs Thomas)")
            any_line = True
        if not disc_plotted:
            Nd = [r["N"] for r in rs if r.get("err_thomas_vs_exact") is not None]
            disc = [r["err_thomas_vs_exact"] for r in rs
                   if r.get("err_thomas_vs_exact") is not None]
            if Nd:
                ax.semilogy(Nd, [max(v, 1e-6) for v in disc], "k--", lw=1.6,
                           label="discretisation (Thomas vs exact)")
                disc_plotted = True
    if not any_line:
        plt.close(fig)
        return None
    ax.set_xlabel("N")
    ax.set_ylabel("error (%)")
    ax.set_title(f"Error decomposition - {case}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = _save_and_close(fig, sweep, f"error_decomposition_{case}.png")
    return out


# ── One-Dimensional Plots ─────────────────────────────────────────────────────

def plot_solution_profiles(
    sweep: SweepArchive,
    grouped: dict,
    save_pdf: bool,
    N_plot: int = 8,
) -> None:
    """
    For each case, plot the solution profiles at N=N_plot for all solvers.
    One figure per case, two panels: solution + pointwise error.
    """
    for case, solver_data in grouped.items():
        # Collect available solvers for this case at N_plot.
        available = {}
        for solver in ["Thomas", "HHL", "VQLS", "QSVT"]:
            if solver not in solver_data:
                continue
            sol = load_solution_1d(sweep, case, solver, N_plot)
            if sol is None:
                continue
            available[solver] = sol

        if not available:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax_sol, ax_err = axes

        # Reference exact solution (from Thomas or NPZ).
        u_exact = None
        if "Thomas" in available and "u_exact" in available["Thomas"]:
            u_exact = available["Thomas"]["u_exact"]
            x_ref   = available["Thomas"]["x"]

        for solver, sol in available.items():
            st = SOLVER_STYLE[solver]
            ax_sol.plot(sol["x"], sol["u"],
                        color=st["color"], marker=st["marker"],
                        ls=st["ls"], label=st["label"],
                        markevery=max(1, len(sol["x"]) // 32))

        if u_exact is not None:
            ax_sol.plot(x_ref, u_exact, "k--", lw=1.2, label="Exact", zorder=0)

        ax_sol.set_xlabel(r"$x$")
        ax_sol.set_ylabel(r"$u(x)$")
        ax_sol.set_title(f"{CASE_LABELS.get(case, case)}\n$N={N_plot}$")
        ax_sol.legend(fontsize=9)

        # Pointwise absolute error vs Thomas reference.
        u_thomas = available.get("Thomas", {}).get("u")
        if u_thomas is not None:
            for solver, sol in available.items():
                if solver == "Thomas":
                    continue
                st = SOLVER_STYLE[solver]
                err = np.abs(sol["u"] - u_thomas)
                ax_err.semilogy(sol["x"], err + 1e-16,
                                color=st["color"], marker=st["marker"],
                                ls=st["ls"], label=st["label"],
                                markevery=max(1, len(sol["x"]) // 8))

        ax_err.set_xlabel(r"$x$")
        ax_err.set_ylabel(r"$|u_\mathrm{solver} - u_\mathrm{Thomas}|$")
        ax_err.set_title("Pointwise absolute error vs Thomas")
        ax_err.legend(fontsize=9)

        fig.tight_layout()
        stem = f"fig_profiles_{case}_N{N_plot}"
        save_fig(fig, sweep, stem, save_pdf)
        plt.close(fig)

def _loglog_solver_series(ax, solver_data: dict, metric: str, keep=None) -> bool:
    """
    Draws one log-log curve per solver, in canonical order and 1-D styling.

    The four 1-D vs-N figures differ only in which metric they read and which
    rows they consider valid; the series loop itself was written out four times.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    solver_data : dict
        {solver: [row, …]} for one case, sorted by N.
    metric : str
        Summary field to plot against N.
    keep : callable, optional
        Predicate deciding whether a row is usable. Defaults to "the metric is
        not None". Passed explicitly where a metric needs more than that — a NaN
        guard for residuals, a positivity guard for wall times, both of which
        would otherwise break the logarithmic axis.

    Returns
    -------
    bool
        Whether any curve was drawn, so the caller can discard an empty figure
        rather than writing a blank one.
    """
    if keep is None:
        def keep(r):
            return r[metric] is not None

    drew = False
    for solver in SOLVER_ORDER:
        if solver not in solver_data:
            continue
        rows = [r for r in solver_data[solver] if keep(r)]
        if not rows:
            continue
        st = SOLVER_STYLE[solver]
        ax.loglog([r["N"] for r in rows], [r[metric] for r in rows],
                  color=st["color"], marker=st["marker"],
                  ls=st["ls"], label=st["label"])
        drew = True
    return drew


def _reference_n_squared(ax) -> None:
    """
    Adds the O(N⁻²) guide line, the second-order convergence the five-point
    stencil should achieve. Its offset is fixed rather than fitted, so the line
    indicates a slope to compare against and not a fit to the data.
    """
    Ns_ref = np.array([4, 8, 16, 32, 64])
    ax.loglog(Ns_ref, 10.0 / Ns_ref**2, "k:", lw=1.0, label=r"$\mathcal{O}(N^{-2})$")


def _format_n_axis(ax, ticks=(4, 8, 16, 32, 64)) -> None:
    """
    Labels the resolution axis with the actual N values rather than powers of
    ten, which is what a logarithmic axis defaults to and what makes these
    figures hard to read at a glance.
    """
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(list(ticks))


def plot_error_vs_N(
    grouped: dict,
    sweep: SweepArchive,
    save_pdf: bool,
    cases_to_plot: list[str] | None = None,
) -> None:
    """
    Log-log plot of max relative error vs N for all solvers.
    One figure per case (or a combined figure for the generic Poisson cases).
    """
    if cases_to_plot is None:
        cases_to_plot = list(grouped.keys())

    for case in cases_to_plot:
        if case not in grouped:
            continue

        fig, ax = plt.subplots(figsize=(7, 5))
        _loglog_solver_series(ax, grouped[case], "max_rel_err")
        _reference_n_squared(ax)

        ax.set_xlabel(r"$N$ (system size)")
        ax.set_ylabel(r"Max relative error (\%)")
        ax.set_title(f"Convergence: {CASE_LABELS.get(case, case)}")
        _format_n_axis(ax)
        ax.legend()
        fig.tight_layout()
        save_fig(fig, sweep, f"fig_error_vs_N_{case}", save_pdf)
        plt.close(fig)

def plot_residual_vs_N(
    grouped: dict,
    sweep: SweepArchive,
    save_pdf: bool,
) -> None:
    """Log-log plot of ||Au-b||/||b|| vs N for all solvers and cases."""
    def usable(r):
        # A failed solve records NaN, which would silently break a log axis.
        return r["residual"] is not None and not np.isnan(float(r["residual"]))

    for case, solver_data in grouped.items():
        fig, ax = plt.subplots(figsize=(7, 5))

        if not _loglog_solver_series(ax, solver_data, "residual", keep=usable):
            plt.close(fig)
            continue

        ax.set_xlabel(r"$N$ (system size)")
        ax.set_ylabel(r"Relative residual $\|Au - b\| / \|b\|$")
        ax.set_title(f"Residual: {CASE_LABELS.get(case, case)}")
        _format_n_axis(ax)
        ax.legend()
        fig.tight_layout()
        save_fig(fig, sweep, f"fig_residual_vs_N_{case}", save_pdf)
        plt.close(fig)

def plot_time_vs_N(
    grouped: dict,
    sweep: SweepArchive,
    save_pdf: bool,
) -> None:
    """Log-log plot of wall time vs N for all solvers."""
    # Aggregate across all cases for a single summary plot.
    fig, ax = plt.subplots(figsize=(7, 5))

    # Use the generic Poisson fS case as representative.
    case = "1D_Poisson_fS_hom"
    if case not in grouped:
        plt.close(fig)
        return

    # A zero wall time means the solve never ran; it is also inadmissible on a
    # logarithmic axis.
    _loglog_solver_series(ax, grouped[case], "wall_time_s",
                          keep=lambda r: r["wall_time_s"] > 0)

    ax.set_xlabel(r"$N$ (system size)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(r"Computational cost: 1D Poisson, $f_S$, homogeneous BCs")
    _format_n_axis(ax)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, sweep, "fig_time_vs_N", save_pdf)
    plt.close(fig)

def plot_het_1d(
    sweep: SweepArchive,
    grouped: dict,
    save_pdf: bool,
    N_plot: int = 8,
) -> None:
    """
    Three-panel figure for the HET 1D cases:
    Left: sub-case 3a potential; Centre: sub-case 3b potential + E-field;
    Right: sub-case 3c potential (new Neumann-Dirichlet benchmark).
    """
    het_cases = [
        "HET_1D_3a_linear_hom",
        "HET_1D_3b_gaussian_Vd300",
        "HET_1D_3c_gaussian_NeumannDirichlet",
    ]
    panel_titles = [
        r"Sub-case 3a: linear profile, hom. BCs",
        r"Sub-case 3b: Gaussian, $V_d=300$ V",
        r"Sub-case 3c: Gaussian, Neumann–Dirichlet (new)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, case, title in zip(axes, het_cases, panel_titles):
        if case not in grouped:
            ax.set_title(title + "\n(no data)")
            continue

        solver_data = grouped[case]
        for solver in ["Thomas", "HHL", "VQLS"]:
            if solver not in solver_data:
                continue
            sol = load_solution_1d(sweep, case, solver, N_plot)
            if sol is None:
                continue
            st = SOLVER_STYLE[solver]
            ax.plot(sol["x"], sol["u"],
                    color=st["color"], marker=st["marker"],
                    ls=st["ls"], label=st["label"],
                    markevery=max(1, len(sol["x"]) // 6))

        # Exact solution overlay if available.
        thomas_sol = load_solution_1d(sweep, case, "Thomas", N_plot)
        if thomas_sol is not None and "u_exact" in thomas_sol:
            x_fine = np.linspace(thomas_sol["x"][0], thomas_sol["x"][-1], 300)
            # Interpolate exact from the saved NPZ.
            u_ex_interp = np.interp(x_fine, thomas_sol["x"], thomas_sol["u_exact"])
            ax.plot(x_fine, u_ex_interp, "k--", lw=1.2, label="Exact", zorder=0)

        ax.set_xlabel(r"$x / L$")
        ax.set_ylabel(r"$\phi$ (normalised)")
        ax.set_title(title + f"\n$N={N_plot}$")
        ax.legend(fontsize=8)

    fig.suptitle("HET 1D Axial Poisson — Potential Profiles", fontsize=13)
    fig.tight_layout()
    save_fig(fig, sweep, f"fig_het_1d_profiles_N{N_plot}", save_pdf)
    plt.close(fig)

    # Separate electric field plot for sub-case 3b.
    case = "HET_1D_3b_gaussian_Vd300"
    if case not in grouped:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for solver in ["Thomas", "HHL", "VQLS"]:
        sol = load_solution_1d(sweep, case, solver, N_plot)
        if sol is None:
            continue
        E = -np.gradient(sol["u"], sol["x"])
        st = SOLVER_STYLE[solver]
        ax.plot(sol["x"], np.abs(E),
                color=st["color"], marker=st["marker"],
                ls=st["ls"], label=st["label"],
                markevery=max(1, len(sol["x"]) // 6))

    ax.set_xlabel(r"$x / L$")
    ax.set_ylabel(r"$|E|$ (V/m)")
    ax.set_title(r"HET 1D: Electric field magnitude, $V_d=300$ V, $N=" + str(N_plot) + r"$")
    ax.legend()
    fig.tight_layout()
    save_fig(fig, sweep, f"fig_het_1d_Efield_N{N_plot}", save_pdf)
    plt.close(fig)

def plot_summary_table(
    grouped: dict,
    sweep: SweepArchive,
    save_pdf: bool,
) -> None:
    """
    A 2×2 grid of error-vs-N plots for the four main generic Poisson cases,
    suitable for a single thesis figure.
    """
    cases = [
        "1D_Poisson_fS_hom",
        "1D_Poisson_fL_hom",
        "1D_Poisson_fH_hom",
        "1D_Poisson_fS_nonhom",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    for ax, case in zip(axes_flat, cases):
        if case not in grouped:
            ax.set_title(CASE_LABELS.get(case, case) + "\n(no data)")
            continue

        _loglog_solver_series(ax, grouped[case], "max_rel_err")
        _reference_n_squared(ax)

        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"Max rel. error (\%)")
        ax.set_title(CASE_LABELS.get(case, case))
        # One tick fewer than the single-case figures: the 2x2 tiling leaves too
        # little width for five labels without them colliding.
        _format_n_axis(ax, ticks=(4, 8, 16, 32))
        ax.legend(fontsize=8)

    fig.suptitle("1D Poisson: Algorithm Comparison — All Cases", fontsize=13)
    fig.tight_layout()
    save_fig(fig, sweep, "fig_summary_generic_poisson", save_pdf)
    plt.close(fig)


# ── Two-Dimensional Plots ─────────────────────────────────────────────────────

def plot_fields(sweep, case: str, N: int, rows: list[dict], plt, TwoSlopeNorm) -> Path | None:
    """
    Exact (if available) plus one column per solver: field on top, signed
    error underneath.  The single most important plot in this script.
    """
    sols = {}
    exact = None
    for r in rows:
        sol = sweep.solution(case, r["solver"], N)
        if sol is None:
            continue
        sols[r["solver"]] = sol
        if exact is None and "phi_exact" in sol:
            exact = sol["phi_exact"]
    if not sols:
        return None

    any_sol = next(iter(sols.values()))
    x, y = any_sol["x"], any_sol["y"]
    ref = exact if exact is not None else sols.get("Thomas", any_sol)["phi_solver"]
    ref_is_exact = exact is not None

    names = list(sols)
    n_cols = 1 + len(names)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7), squeeze=False)

    vmin, vmax = float(ref.min()), float(ref.max())
    im = axes[0, 0].pcolormesh(x, y, ref, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                               shading="auto")
    axes[0, 0].set_title("Exact" if ref_is_exact else "Thomas (reference)",
                         fontweight="bold")
    axes[0, 0].set_aspect("equal")
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
    axes[1, 0].axis("off")

    for ci, name in enumerate(names, start=1):
        phi = sols[name]["phi_solver"]
        im = axes[0, ci].pcolormesh(x, y, phi, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                                    shading="auto")
        axes[0, ci].set_title(name, fontweight="bold",
                              color=SOLVER_COLOUR.get(name, "black"))
        axes[0, ci].set_aspect("equal")
        plt.colorbar(im, ax=axes[0, ci], shrink=0.8)

        err = phi - ref
        abs_max = max(float(np.abs(err).max()), 1e-12)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
        im2 = axes[1, ci].pcolormesh(x, y, err, cmap="seismic", norm=norm,
                                     shading="auto")
        pct = (np.max(np.abs(err)) / (np.max(np.abs(ref)) + 1e-300)) * 100.0
        label = "vs exact" if ref_is_exact else "vs Thomas"
        axes[1, ci].set_title(f"Error {label} ({pct:.3f}%)")
        axes[1, ci].set_aspect("equal")
        plt.colorbar(im2, ax=axes[1, ci], shrink=0.8)

    fig.suptitle(f"{case}  N={N}", fontweight="bold")
    plt.tight_layout()
    out = _save_and_close(fig, sweep, f"fields_{case}_N{N}.png")
    return out


# ── Three-Dimensional Plots ───────────────────────────────────────────────────

def axis_labels_for(case: str, sol: dict) -> tuple[str, str, str]:
    if "HET" in case or bool(sol.get("periodic", (False,) * 3)[2]):
        z, r, s = AXIS_LABELS["het"]
        return z, r, s
    return AXIS_LABELS["generic"]

def _slice_at(arr: np.ndarray, fixed_axis: int, idx: int) -> np.ndarray:
    sl = [slice(None)] * 3
    sl[fixed_axis] = idx
    return arr[tuple(sl)]

def plot_slices(sweep, case: str, N: int, rows: list[dict], plt, TwoSlopeNorm) -> list[Path]:
    """One PNG per cutplane, each laid out like the 2-D fields plot."""
    sols, exact = {}, None
    for r in rows:
        sol = sweep.solution(case, r["solver"], N)
        if sol is None:
            continue
        sols[r["solver"]] = sol
        if exact is None and "phi_exact" in sol:
            exact = sol["phi_exact"]
    if not sols:
        return []

    any_sol = next(iter(sols.values()))
    coords = [any_sol["x0"], any_sol["x1"], any_sol["x2"]]
    labels = axis_labels_for(case, any_sol)
    shape = any_sol["phi"].shape
    ref = exact if exact is not None else sols.get("Thomas", any_sol)["phi"]
    ref_is_exact = exact is not None
    names = list(sols)

    out_paths = []
    for plane_name, fixed_ax, (a0, a1), is_spoke in _PLANES:
        idx = shape[fixed_ax] // 2
        # Coordinate grid for this plane: take the two varying axes' meshgrid
        # at the fixed slice (coords[a] is already a full 3-D meshgrid, so
        # slicing it the same way as the field gives the right 2-D grid).
        cx = _slice_at(coords[a0], fixed_ax, idx)
        cy = _slice_at(coords[a1], fixed_ax, idx)
        ref_slice = _slice_at(ref, fixed_ax, idx)

        n_cols = 1 + len(names)
        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7), squeeze=False)
        vmin, vmax = float(ref.min()), float(ref.max())

        im = axes[0, 0].pcolormesh(cx, cy, ref_slice, cmap="RdBu_r",
                                   vmin=vmin, vmax=vmax, shading="auto")
        axes[0, 0].set_title("Exact" if ref_is_exact else "Thomas (reference)",
                             fontweight="bold")
        axes[0, 0].set_xlabel(labels[a0]); axes[0, 0].set_ylabel(labels[a1])
        plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
        axes[1, 0].axis("off")

        for ci, name in enumerate(names, start=1):
            phi_slice = _slice_at(sols[name]["phi"], fixed_ax, idx)
            im = axes[0, ci].pcolormesh(cx, cy, phi_slice, cmap="RdBu_r",
                                        vmin=vmin, vmax=vmax, shading="auto")
            axes[0, ci].set_title(name, fontweight="bold",
                                  color=SOLVER_COLOUR.get(name, "black"))
            axes[0, ci].set_xlabel(labels[a0])
            plt.colorbar(im, ax=axes[0, ci], shrink=0.8)

            err = phi_slice - ref_slice
            abs_max = max(float(np.abs(err).max()), 1e-12)
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
            im2 = axes[1, ci].pcolormesh(cx, cy, err, cmap="seismic", norm=norm,
                                         shading="auto")
            pct = float(np.max(np.abs(phi_slice - ref_slice))
                       / (np.max(np.abs(ref)) + 1e-300) * 100.0)
            lab = "vs exact" if ref_is_exact else "vs Thomas"
            axes[1, ci].set_title(f"Error {lab} ({pct:.3f}%)")
            axes[1, ci].set_xlabel(labels[a0])
            plt.colorbar(im2, ax=axes[1, ci], shrink=0.8)

        tag = "  <- rotating-spoke view (z vs theta, unrolled)" if is_spoke else ""
        fig.suptitle(f"{case}  N={N}  |  {plane_name} slice{tag}", fontweight="bold")
        plt.tight_layout()
        safe_plane = plane_name.replace("-", "_")
        out = _save_and_close(fig, sweep, f"slice_{safe_plane}_{case}_N{N}.png")
        out_paths.append(out)
    return out_paths

def plot_polar_unwrap(sweep, case: str, N: int, rows: list[dict], plt) -> Path | None:
    """
    Axial-azimuthal slice recast onto an annulus: the classic HET "plan
    view" used to show a rotating spoke as a coloured ring.

    The radial coordinate is RELATIVE, not the true channel radius: the
    archived data carries the channel width (Lr) but not the inner radius
    r_in, so the annulus inner/outer radii here are an arbitrary offset
    chosen only to make the ring legible.  Treat this as a schematic
    orientation aid; the rectangular unrolled view (plot_slices,
    axial-azimuthal plane) is the quantitatively meaningful one.
    """
    any_sol = None
    for r in rows:
        sol = sweep.solution(case, r["solver"], N)
        if sol is not None:
            any_sol = sol
            break
    if any_sol is None or not bool(any_sol.get("periodic", (False,) * 3)[2]):
        return None

    solver_name = next(r["solver"] for r in rows
                       if sweep.solution(case, r["solver"], N) is not None)
    sol = sweep.solution(case, solver_name, N)
    phi, r_coord, s_coord = sol["phi"], sol["x1"], sol["x2"]
    Lr = float(sol["lengths"][1])
    r_idx = phi.shape[0] // 2                     # mid-axial slice

    r_rel = r_coord[0, :, 0]                       # 1-D radial coordinate
    s_full = s_coord[0, 0, :]
    r_annulus = 3.0 * Lr + r_rel                    # arbitrary inner offset
    theta = 2.0 * np.pi * s_full / float(sol["lengths"][2])

    R, TH = np.meshgrid(r_annulus, theta, indexing="ij")
    X, Y = R * np.cos(TH), R * np.sin(TH)
    field = phi[r_idx, :, :]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    im = ax.pcolormesh(X, Y, field, cmap="RdBu_r", shading="auto")
    ax.set_aspect("equal")
    ax.set_title(f"{case}  N={N}  {solver_name}\n"
                 f"plan view (schematic radius) - mid-axial slice")
    ax.set_xlabel("(schematic x)"); ax.set_ylabel("(schematic y)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="phi")
    plt.tight_layout()
    out = _save_and_close(fig, sweep, f"polar_spoke_{case}_N{N}_{solver_name}.png")
    return out

def plot_3d_cutaway(sweep, case: str, N: int, rows: list[dict], plt) -> Path | None:
    """
    Single 3-D view with the three orthogonal slice planes embedded at their
    true offsets, using matplotlib's contourf(..., zdir=..., offset=...).

    This is an orientation aid, not a quantitative tool: matplotlib's 3-D
    backend does not depth-sort filled patches reliably, so overlapping
    planes can occlude each other in a way that is not physically
    meaningful.  Use the orthogonal slice PNGs for anything that needs to be
    read precisely.
    """
    sol = None
    for r in rows:
        if r["solver"] == "Thomas":
            sol = sweep.solution(case, r["solver"], N)
            if sol is not None:
                break
    if sol is None:
        for r in rows:
            sol = sweep.solution(case, r["solver"], N)
            if sol is not None:
                break
    if sol is None:
        return None

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D proj)

    phi = sol["phi"]
    x0, x1, x2 = sol["x0"][:, 0, 0], sol["x1"][0, :, 0], sol["x2"][0, 0, :]
    nx, ny, nz = phi.shape
    ix, iy, iz = nx // 2, ny // 2, nz // 2

    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    vmin, vmax = float(phi.min()), float(phi.max())
    cmap = plt.get_cmap("RdBu_r")

    Y, Z = np.meshgrid(x1, x2, indexing="ij")
    ax.contourf(Y, Z, phi[ix, :, :], zdir="x", offset=float(x0[ix]),
               levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    X, Z = np.meshgrid(x0, x2, indexing="ij")
    ax.contourf(X, phi[:, iy, :], Z, zdir="y", offset=float(x1[iy]),
               levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    X, Y = np.meshgrid(x0, x1, indexing="ij")
    ax.contourf(X, Y, phi[:, :, iz], zdir="z", offset=float(x2[iz]),
               levels=20, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlim(x0.min(), x0.max()); ax.set_ylim(x1.min(), x1.max())
    ax.set_zlim(x2.min(), x2.max())
    labels = axis_labels_for(case, sol)
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_zlabel(labels[2])
    ax.set_title(f"{case}  N={N}  ({sol.get('_solver_name', 'field')})\n"
                 "3-D cutaway (orientation only - see slice PNGs for values)")
    plt.tight_layout()
    out = _save_and_close(fig, sweep, f"cutaway3d_{case}_N{N}.png")
    return out

def plot_azimuthal_fidelity(sweep, case: str, by_solver: dict, plt) -> Path | None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    any_line = False
    for solver, rs in sweep.series(by_solver, case, exclude=("Thomas",)):
        Ns = [r["N"] for r in rs if r.get("azimuthal_mode_rel_err") is not None]
        errs = [r["azimuthal_mode_rel_err"] for r in rs
               if r.get("azimuthal_mode_rel_err") is not None]
        if not Ns:
            continue
        ax.semilogy(Ns, [max(v, 1e-6) for v in errs], "o-",
                   color=SOLVER_COLOUR.get(solver), label=solver)
        any_line = True
    if not any_line:
        plt.close(fig)
        return None
    ax.set_xlabel("N"); ax.set_ylabel("azimuthal mode amplitude error (%)")
    ax.set_title(f"Spoke-mode fidelity vs N - {case}\n"
                "(does the solver reproduce the mode, not just the field norm)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    plt.tight_layout()
    out = _save_and_close(fig, sweep, f"azimuthal_fidelity_{case}.png")
    return out


# ── Entry Points ──────────────────────────────────────────────────────────────

def _matplotlib():
    """
    Imports Matplotlib with a headless backend and binds the module globals.

    See the note beside the `plt = None` declaration at the top of this module
    for why the import is deferred. The 1-D plotting functions reference `plt`
    and `ticker` as module globals rather than taking them as arguments, so
    those names are bound here on first use.

    The headless backend is selected only if nothing has imported `pyplot`
    yet. Selecting it unconditionally is process-global and one-way: a caller
    that already has an interactive session — a notebook, or anything that
    imported `benchmark/plotting.py` first — would find every subsequent
    `plt.show()` silently doing nothing. Deferring the import protects against
    Agg being forced at *import* time; this check protects against it being
    forced at *call* time, which is the same hazard one step later.

    Returns
    -------
    tuple
        (pyplot module, TwoSlopeNorm class).

    Raises
    ------
    SystemExit
        If Matplotlib is unavailable or its backend cannot be initialised.
    """
    global plt, ticker
    try:
        import matplotlib
        if "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import matplotlib.ticker as _ticker
        from matplotlib.colors import TwoSlopeNorm
    except (ImportError, ValueError) as exc:
        # `matplotlib.use` raises ValueError, not ImportError, when a backend
        # exists but cannot be initialised, so catching ImportError alone lets
        # a broken Agg build escape as a raw traceback.
        raise SystemExit(f"matplotlib is required to produce these figures: {exc}")
    plt, ticker = _plt, _ticker
    return plt, TwoSlopeNorm


def filter_rows(rows: list[dict], case: str | None, N: int | None) -> list[dict]:
    """
    Restricts a sweep's rows to one case and/or one resolution.

    Parameters
    ----------
    rows : list[dict]
        Summary rows.
    case : str or None
        Case identifier to keep, or None for all.
    N : int or None
        Resolution to keep, or None for all.

    Returns
    -------
    list[dict]
        The surviving rows.

    Raises
    ------
    SystemExit
        If the filter selects nothing, which is almost always a mistyped case
        name rather than a genuinely empty sweep.
    """
    if case:
        rows = [r for r in rows if r["case"] == case]
    if N:
        rows = [r for r in rows if r["N"] == N]
    if not rows:
        raise SystemExit("No matching rows. Use --list to see what is available.")
    return rows


def print_listing(rows: list[dict], case_width: int = 38) -> None:
    """Prints the available (case, N, solver) combinations and their outcomes."""
    for r in sorted(rows, key=lambda r: (r["case"], r["N"], r["solver"])):
        print(f"  {r['case']:<{case_width}} N={r['N']:<4} {r['solver']:<8} "
              f"scheme={r['scheme']:<10} {r['stop_reason']}")


def _report(made: list, sweep: SweepArchive) -> None:
    """Summarises what was written."""
    print(f"Wrote {len(made)} plot(s) to {sweep.plots_dir}")
    for p in made:
        print(f"  {p.name}")


def run_1d(
    results_dir: Path,
    save_pdf:    bool = False,
    N_profile:   int = 32,
) -> None:
    """
    Produces the full 1-D figure set for a completed sweep.

    Parameters
    ----------
    results_dir : Path
        Sweep output directory.
    save_pdf : bool, default=False
        Also emit vector PDFs alongside the PNGs.
    N_profile : int, default=32
        Resolution at which to draw the solution-profile figures. Profiles are
        drawn at a single resolution because they are a qualitative check on
        the shape of the solution, not a convergence study — the vs-N figures
        cover the latter.
    """
    plt, _ = _matplotlib()
    sweep = SweepArchive(results_dir, dim=1, plots_subdir=None)

    print(f"Loading results from: {sweep.results_dir}")
    rows = sweep.rows()
    grouped = sweep.group_nested(rows)
    print(f"Found {len(rows)} result rows across {len(grouped)} cases.")
    _warn_missing(sweep, rows)
    print("Generating plots...")

    # Scoped rather than applied globally. `plt.rcParams.update` leaves the
    # publication style in force for the rest of the process, so a caller that
    # ran run_1d and then run_2d would get serif, 300-dpi 2-D figures, which the
    # 2-D family is explicitly documented not to use.
    with plt.rc_context(RC_PARAMS_1D):
        plot_solution_profiles(sweep, grouped, save_pdf, N_plot=N_profile)
        plot_error_vs_N(grouped, sweep, save_pdf)
        plot_residual_vs_N(grouped, sweep, save_pdf)
        plot_time_vs_N(grouped, sweep, save_pdf)
        plot_het_1d(sweep, grouped, save_pdf, N_plot=N_profile)
        plot_summary_table(grouped, sweep, save_pdf)

    print(f"\nAll figures saved to: {sweep.plots_dir.resolve()}")


def run_2d(
    results_dir: Path,
    case:        str | None = None,
    N:           int | None = None,
    listing:     bool = False,
) -> None:
    """
    Produces the full 2-D figure set for a completed sweep.

    Parameters
    ----------
    results_dir : Path
        Sweep output directory.
    case : str or None
        Restrict to one case.
    N : int or None
        Restrict to one resolution.
    listing : bool, default=False
        Print the available combinations and return without plotting.
    """
    sweep = SweepArchive(results_dir, dim=2, skip_scheme_comparison=True)
    rows = filter_rows(sweep.rows(), case, N)
    _warn_missing(sweep, rows)

    if listing:
        print_listing(rows)
        return

    plt, TwoSlopeNorm = _matplotlib()
    sweep.plots_dir.mkdir(parents=True, exist_ok=True)

    by_case_N = sweep.group_by_case_N(rows)
    by_case_solver = sweep.group_by_case_solver(rows)
    cases = sorted({c for c, _ in by_case_N})

    made = []
    for (c, n), case_rows in sorted(by_case_N.items()):
        for p in (plot_fields(sweep, c, n, case_rows, plt, TwoSlopeNorm),
                  plot_convergence(sweep, c, n, case_rows, plt)):
            if p:
                made.append(p)

    for c in cases:
        for fn in (plot_accuracy_vs_n, plot_cost_vs_n, plot_error_decomposition):
            p = fn(sweep, c, by_case_solver, plt)
            if p:
                made.append(p)
        p = plot_overhead(sweep, c, by_case_N, plt)
        if p:
            made.append(p)

    _report(made, sweep)


def run_3d(
    results_dir: Path,
    case:        str | None = None,
    N:           int | None = None,
    listing:     bool = False,
    cutaway:     bool = True,
) -> None:
    """
    Produces the full 3-D figure set for a completed sweep.

    Parameters
    ----------
    results_dir : Path
        Sweep output directory.
    case : str or None
        Restrict to one case.
    N : int or None
        Restrict to one resolution.
    listing : bool, default=False
        Print the available combinations and return without plotting.
    cutaway : bool, default=True
        Include the mplot3d cutaway orientation figure, the slowest and least
        essential of the set.
    """
    sweep = SweepArchive(results_dir, dim=3)
    rows = filter_rows(sweep.rows(), case, N)
    _warn_missing(sweep, rows)

    if listing:
        print_listing(rows, case_width=34)
        return

    plt, TwoSlopeNorm = _matplotlib()
    sweep.plots_dir.mkdir(parents=True, exist_ok=True)

    by_case_N = sweep.group_by_case_N(rows)
    by_case_solver = sweep.group_by_case_solver(rows)
    cases = sorted({c for c, _ in by_case_N})

    made = []
    for (c, n), case_rows in sorted(by_case_N.items()):
        made += plot_slices(sweep, c, n, case_rows, plt, TwoSlopeNorm)
        p = plot_polar_unwrap(sweep, c, n, case_rows, plt)
        if p:
            made.append(p)
        if cutaway:
            p = plot_3d_cutaway(sweep, c, n, case_rows, plt)
            if p:
                made.append(p)
        p = plot_convergence(sweep, c, n, case_rows, plt)
        if p:
            made.append(p)

    for c in cases:
        for fn in (plot_accuracy_vs_n, plot_cost_vs_n, plot_error_decomposition,
                   plot_azimuthal_fidelity):
            p = fn(sweep, c, by_case_solver, plt)
            if p:
                made.append(p)
        p = plot_overhead(sweep, c, by_case_N, plt)
        if p:
            made.append(p)

    _report(made, sweep)
