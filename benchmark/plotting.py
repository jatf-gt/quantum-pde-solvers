"""
Publication-standard figure generation for the quantum PDE solver benchmark.

All functions follow the contract: data in → matplotlib Figure out.
No file I/O is performed inside figure functions; callers are responsible
for saving. This separation ensures figures can be previewed interactively
or saved to any format without modifying the plotting code.

Figure catalogue
----------------
  solution_profiles_1d       Solution and pointwise error for one (N, case).
  convergence_loglog         Max relative error vs N (log-log).
  residual_vs_N              Relative residual vs N (log-log).
  walltime_vs_N              Wall time vs N (log-log).
  circuit_depth_vs_N         Circuit depth vs N (log-log) with theory lines.
  sensitivity_curves         OAT sensitivity: metric vs parameter value.
  equal_accuracy_bar         Resource cost at matched residual (bar chart).
  error_decomposition        Algorithmic vs discretisation error breakdown.
  order_comparison           2nd vs 4th order accuracy at matched N.
  hardware_vs_simulation     Real hardware vs statevector simulation comparison.

Style conventions
-----------------
  Solver colours:  Thomas = black, HHL = royalblue,
                   VQLS = darkorange, QSVT = crimson
  Line styles:     2nd order = solid, 4th order = dashed
  Markers:         Thomas = s, HHL = o, VQLS = ^, QSVT = D
  Figure size:     single column = (5.5, 4.0), double column = (11.0, 4.0)
  Font size:       axis labels = 11, tick labels = 9, legend = 9

References
----------
  Ghafourpour & Laizet (2025) Phys. Rev. Applied 24, 024032.
  Bravo-Prieto et al. (2023) Quantum 7, 1188.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

# Matplotlib is imported lazily inside each function to allow the module to be
# imported in headless environments without a display backend.

# -- Style constants -----------------------------------------------------------

SOLVER_COLOURS: dict[str, str] = {
    "thomas": "black",
    "hhl":    "royalblue",
    "vqls":   "darkorange",
    "qsvt":   "crimson",
}

SOLVER_MARKERS: dict[str, str] = {
    "thomas": "s",
    "hhl":    "o",
    "vqls":   "^",
    "qsvt":   "D",
}

SOLVER_LABELS: dict[str, str] = {
    "thomas": "Thomas",
    "hhl":    "HHL",
    "vqls":   "VQLS",
    "qsvt":   "QSVT",
}

_FIG_SINGLE = (5.5, 4.0)
_FIG_DOUBLE = (11.0, 4.0)
_FIG_SQUARE = (5.0, 5.0)

_LABEL_FS  = 11
_TICK_FS   = 9
_LEGEND_FS = 9


def _apply_style(ax, xlabel: str, ylabel: str, title: str = "") -> None:
    """Apply consistent axis styling."""
    ax.set_xlabel(xlabel, fontsize=_LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=_TICK_FS)
    if title:
        ax.set_title(title, fontsize=_LABEL_FS)
    ax.tick_params(labelsize=_TICK_FS)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.legend(fontsize=_LEGEND_FS, framealpha=0.9)


# -- Figure 1: Solution profiles -----------------------------------------------

def solution_profiles_1d(
    x: np.ndarray,
    solutions: dict[str, np.ndarray],
    u_exact: Optional[np.ndarray],
    case_id: str,
    N: int,
    discretisation_order: int = 2,
):
    """
    Plot solution profiles and pointwise relative error for a 1D case.

    Layout: left panel = solution profiles; right panel = pointwise
    relative error |û_i - u*_i| / |u*_i| [%] against the analytical
    solution (or Thomas if analytical is unavailable).

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Interior node coordinates.
    solutions : dict[str, np.ndarray]
        Mapping solver_name -> solution vector, shape (N,).
    u_exact : np.ndarray or None, shape (N,)
        Analytical solution. If None, Thomas is used as reference.
    case_id : str
        Case identifier for the figure title.
    N : int
        Problem size.
    discretisation_order : int
        Spatial discretisation order (2 or 4).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    u_ref = u_exact if u_exact is not None else solutions.get("thomas")
    ref_label = "Exact" if u_exact is not None else "Thomas"

    fig, axes = plt.subplots(1, 2, figsize=_FIG_DOUBLE)

    # Left: solution profiles
    ax = axes[0]
    if u_exact is not None:
        ax.plot(x, u_exact, "k--", lw=1.5, label="Exact", zorder=5)

    for solver, u in solutions.items():
        if u is None:
            continue
        c = SOLVER_COLOURS.get(solver, "grey")
        m = SOLVER_MARKERS.get(solver, "x")
        lbl = SOLVER_LABELS.get(solver, solver.upper())
        ax.plot(x, u, color=c, marker=m, markersize=4,
                lw=1.2, label=lbl, alpha=0.85)

    order_str = f"{discretisation_order}\\textsuperscript{{nd}}" if discretisation_order == 2 else "4\\textsuperscript{th}"
    _apply_style(
        ax,
        xlabel=r"$x$",
        ylabel=r"$\phi(x)$",
        title=f"{case_id}  ($N={N}$, {discretisation_order}nd-order)",
    )

    # Right: pointwise relative error
    ax2 = axes[1]
    if u_ref is not None:
        mask = np.abs(u_ref) > 1.0e-10
        for solver, u in solutions.items():
            if u is None or solver == "thomas" and u_exact is not None:
                continue
            c = SOLVER_COLOURS.get(solver, "grey")
            m = SOLVER_MARKERS.get(solver, "x")
            lbl = SOLVER_LABELS.get(solver, solver.upper())
            err = np.zeros_like(u)
            err[mask] = np.abs(u[mask] - u_ref[mask]) / np.abs(u_ref[mask]) * 100.0
            ax2.semilogy(x, np.maximum(err, 1.0e-14),
                         color=c, marker=m, markersize=4,
                         lw=1.2, label=lbl, alpha=0.85)

    _apply_style(
        ax2,
        xlabel=r"$x$",
        ylabel=rf"$|û_i - u^*_i| / |u^*_i|$ [\%]",
        title=f"Pointwise error vs {ref_label}",
    )

    fig.tight_layout()
    return fig


# -- Figure 2: Convergence log-log ---------------------------------------------

def convergence_loglog(
    results,
    metric: str = "max_rel_err_vs_exact",
    solvers: Optional[list[str]] = None,
    title: str = "",
    show_reference_line: bool = True,
):
    """
    Log-log plot of a chosen accuracy metric vs problem size N.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Benchmark results to plot.
    metric : str
        Attribute name on BenchmarkResult to plot on the y-axis.
        Common choices: 'max_rel_err_vs_exact', 'max_rel_err_vs_thomas',
        'residual'.
    solvers : list[str], optional
        Solvers to include. Defaults to all present in results.
    title : str
        Figure title.
    show_reference_line : bool
        If True, overlay an O(N^{-2}) reference line.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if solvers is None:
        solvers = sorted({r.solver for r in results})

    fig, ax = plt.subplots(figsize=_FIG_SINGLE)

    for solver in solvers:
        rows = sorted(
            [r for r in results if r.solver.lower() == solver],
            key=lambda r: r.N,
        )
        if not rows:
            continue
        N_vals = [r.N for r in rows]
        y_vals = [getattr(r, metric) for r in rows]
        valid = [(n, y) for n, y in zip(N_vals, y_vals) if y is not None]
        if not valid:
            continue
        N_plot, y_plot = zip(*valid)

        ax.loglog(
            N_plot, y_plot,
            color=SOLVER_COLOURS.get(solver, "grey"),
            marker=SOLVER_MARKERS.get(solver, "x"),
            markersize=6, lw=1.8,
            label=SOLVER_LABELS.get(solver, solver.upper()),
        )

    if show_reference_line and results:
        N_all = sorted({r.N for r in results})
        N_arr = np.array(N_all, dtype=float)
        # Anchor the reference line at the largest N of the Thomas solver
        thomas_rows = [r for r in results if r.solver == "thomas"]
        if thomas_rows:
            anchor = max(thomas_rows, key=lambda r: r.N)
            y_anchor = getattr(anchor, metric)
            if y_anchor is not None and y_anchor > 0:
                ref = y_anchor * (N_arr / anchor.N) ** (-2)
                ax.loglog(
                    N_arr, ref, "k:", lw=1.0,
                    label=r"$\mathcal{O}(N^{-2})$",
                )

    ylabel_map = {
        "max_rel_err_vs_exact":  r"$e_\infty$ [\%] vs exact",
        "max_rel_err_vs_thomas": r"$e_\infty$ [\%] vs Thomas",
        "residual":              r"$r = \|Au-b\|_2 / \|b\|_2$",
    }
    _apply_style(
        ax,
        xlabel=r"$N$ (interior nodes)",
        ylabel=ylabel_map.get(metric, metric),
        title=title,
    )
    fig.tight_layout()
    return fig


# -- Figure 3: Residual vs N ---------------------------------------------------

def residual_vs_N(
    results,
    solvers: Optional[list[str]] = None,
    title: str = "",
):
    """
    Log-log plot of relative residual r = ‖Au-b‖₂/‖b‖₂ vs N.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Benchmark results.
    solvers : list[str], optional
        Solvers to include.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    return convergence_loglog(
        results,
        metric="residual",
        solvers=solvers,
        title=title or "Relative residual vs problem size",
        show_reference_line=False,
    )


# -- Figure 4: Wall time vs N --------------------------------------------------

def walltime_vs_N(
    results,
    solvers: Optional[list[str]] = None,
    title: str = "",
):
    """
    Log-log plot of wall time [s] vs N.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Benchmark results.
    solvers : list[str], optional
        Solvers to include.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if solvers is None:
        solvers = sorted({r.solver for r in results})

    fig, ax = plt.subplots(figsize=_FIG_SINGLE)

    for solver in solvers:
        rows = sorted(
            [r for r in results if r.solver.lower() == solver],
            key=lambda r: r.N,
        )
        if not rows:
            continue
        N_vals = [r.N for r in rows]
        t_vals = [r.wall_time_s for r in rows]

        ax.loglog(
            N_vals, t_vals,
            color=SOLVER_COLOURS.get(solver, "grey"),
            marker=SOLVER_MARKERS.get(solver, "x"),
            markersize=6, lw=1.8,
            label=SOLVER_LABELS.get(solver, solver.upper()),
        )

    _apply_style(
        ax,
        xlabel=r"$N$ (interior nodes)",
        ylabel=r"Wall time [s]",
        title=title or "Wall time vs problem size",
    )
    fig.tight_layout()
    return fig


# -- Figure 5: Circuit depth vs N ---------------------------------------------

def circuit_depth_vs_N(
    results,
    solvers: Optional[list[str]] = None,
    show_theory_lines: bool = True,
    epsilon_hhl: float = 0.01,
    epsilon_qsvt: float = 0.01,
    title: str = "",
):
    """
    Log-log plot of circuit depth vs N, with theoretical scaling overlaid.

    Theoretical scaling lines:
      HHL:  O(κ² / ε) = O(N⁴ / ε)   (κ ~ N² for 1D Poisson)
      QSVT: O(κ / ε)  = O(N² / ε)
      VQLS: O(n_layers · n_q) = O(n_layers · log₂ N)

    Parameters
    ----------
    results : list[BenchmarkResult]
        Benchmark results (quantum solvers only; Thomas is excluded).
    solvers : list[str], optional
        Solvers to include.
    show_theory_lines : bool
        If True, overlay theoretical scaling lines.
    epsilon_hhl : float
        HHL precision parameter for the theory line.
    epsilon_qsvt : float
        QSVT precision parameter for the theory line.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if solvers is None:
        solvers = [s for s in ["hhl", "vqls", "qsvt"]
                   if any(r.solver == s for r in results)]

    fig, ax = plt.subplots(figsize=_FIG_SINGLE)

    for solver in solvers:
        rows = sorted(
            [r for r in results
             if r.solver.lower() == solver and r.circuit_metrics is not None],
            key=lambda r: r.N,
        )
        if not rows:
            continue
        N_vals = [r.N for r in rows]
        d_vals = [r.circuit_metrics.depth_opt1 for r in rows]

        ax.loglog(
            N_vals, d_vals,
            color=SOLVER_COLOURS.get(solver, "grey"),
            marker=SOLVER_MARKERS.get(solver, "x"),
            markersize=6, lw=1.8,
            label=SOLVER_LABELS.get(solver, solver.upper()),
        )

    if show_theory_lines:
        N_all = sorted({r.N for r in results})
        if len(N_all) >= 2:
            N_arr = np.array(N_all, dtype=float)

            # Anchor theory lines at the smallest N with data
            def _theory_line(ax_obj, N_arr, exponent, anchor_N, anchor_d,
                             label, colour):
                theory = anchor_d * (N_arr / anchor_N) ** exponent
                ax_obj.loglog(N_arr, theory, linestyle=":", lw=1.0,
                              color=colour, label=label, alpha=0.6)

            hhl_rows = sorted(
                [r for r in results
                 if r.solver == "hhl" and r.circuit_metrics is not None],
                key=lambda r: r.N,
            )
            if hhl_rows:
                _theory_line(
                    ax, N_arr, exponent=4,
                    anchor_N=hhl_rows[0].N,
                    anchor_d=hhl_rows[0].circuit_metrics.depth_opt1,
                    label=r"$\mathcal{O}(N^4/\varepsilon)$ HHL",
                    colour=SOLVER_COLOURS["hhl"],
                )

            qsvt_rows = sorted(
                [r for r in results
                 if r.solver == "qsvt" and r.circuit_metrics is not None],
                key=lambda r: r.N,
            )
            if qsvt_rows:
                _theory_line(
                    ax, N_arr, exponent=2,
                    anchor_N=qsvt_rows[0].N,
                    anchor_d=qsvt_rows[0].circuit_metrics.depth_opt1,
                    label=r"$\mathcal{O}(N^2/\varepsilon)$ QSVT",
                    colour=SOLVER_COLOURS["qsvt"],
                )

    _apply_style(
        ax,
        xlabel=r"$N$ (interior nodes)",
        ylabel=r"Circuit depth (gates, opt.\ level 1)",
        title=title or "Circuit depth vs problem size",
    )
    fig.tight_layout()
    return fig


# -- Figure 6: Sensitivity curves ---------------------------------------------

def sensitivity_curves(
    sweep_results,
    solver: str,
    N: int,
    metrics: Optional[list[str]] = None,
    title: str = "",
):
    """
    Multi-panel sensitivity curves: one panel per parameter.

    Each panel shows the chosen metrics as a function of the swept
    parameter value. Designed for N ∈ {4, 8} where the parameter
    range is tractable.

    Parameters
    ----------
    sweep_results : list[SensitivitySweepResult]
        All OAT sweep results for the solver.
    solver : str
        Solver name (for title).
    N : int
        Problem size (for title).
    metrics : list[str], optional
        Metrics to plot in each panel. Defaults to
        ['max_rel_err_vs_exact', 'residual', 'wall_time_s'].
    title : str
        Figure suptitle.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = ["max_rel_err_vs_exact", "residual", "wall_time_s"]

    n_params = len(sweep_results)
    n_metrics = len(metrics)
    if n_params == 0:
        raise ValueError("sweep_results is empty.")

    fig, axes = plt.subplots(
        n_metrics, n_params,
        figsize=(4.0 * n_params, 3.5 * n_metrics),
        squeeze=False,
    )

    metric_labels = {
        "max_rel_err_vs_exact":  r"$e_\infty$ [\%] vs exact",
        "max_rel_err_vs_thomas": r"$e_\infty$ [\%] vs Thomas",
        "residual":              r"$r$",
        "wall_time_s":           r"$t$ [s]",
    }

    for col, sweep in enumerate(sweep_results):
        vals = [r.sensitivity_value for r in sweep.results]
        # Replace -1.0 sentinel (uncapped) with a large number for plotting
        vals_plot = [v if v >= 0 else max(v for v in vals if v >= 0) * 2
                     for v in vals]

        for row, metric in enumerate(metrics):
            ax = axes[row][col]
            y_vals = [getattr(r, metric) for r in sweep.results]
            valid = [(v, y) for v, y in zip(vals_plot, y_vals) if y is not None]
            if not valid:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            v_plot, y_plot = zip(*valid)

            use_log = metric in ("residual", "wall_time_s")
            if use_log:
                ax.semilogy(v_plot, y_plot,
                            color=SOLVER_COLOURS.get(solver, "grey"),
                            marker="o", markersize=5, lw=1.5)
            else:
                ax.plot(v_plot, y_plot,
                        color=SOLVER_COLOURS.get(solver, "grey"),
                        marker="o", markersize=5, lw=1.5)

            ax.set_xlabel(sweep.param_name, fontsize=_TICK_FS)
            ax.set_ylabel(metric_labels.get(metric, metric), fontsize=_TICK_FS)
            ax.tick_params(labelsize=_TICK_FS)
            ax.grid(True, alpha=0.25, linestyle="--")
            if col == 0:
                ax.set_ylabel(metric_labels.get(metric, metric), fontsize=_TICK_FS)

    suptitle = title or (
        f"Sensitivity analysis — {SOLVER_LABELS.get(solver, solver.upper())}  "
        f"$N={N}$"
    )
    fig.suptitle(suptitle, fontsize=_LABEL_FS + 1)
    fig.tight_layout()
    return fig


# -- Figure 7: Equal-accuracy bar chart ---------------------------------------

def equal_accuracy_bar(
    ea_results,
    metric: str = "depth_opt1",
    r_target: float = 1.0e-3,
    title: str = "",
):
    """
    Horizontal bar chart comparing resource cost at matched residual.

    Parameters
    ----------
    ea_results : list[EqualAccuracyResult]
        One result per solver from the equal-accuracy sweep.
    metric : str
        Resource metric to plot: 'depth_opt1' | 'n_qubits' |
        'wall_time_s' | 'n_cx_gates'.
    r_target : float
        Target residual (for title annotation).
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    labels, values, colours = [], [], []
    for ear in ea_results:
        br = ear.best_result
        lbl = SOLVER_LABELS.get(ear.solver, ear.solver.upper())
        if not ear.in_band:
            lbl += " *"

        if metric == "wall_time_s":
            val = br.wall_time_s
        elif metric in ("depth_opt1", "depth_opt0", "depth_raw",
                        "n_cx_gates", "n_qubits"):
            val = getattr(br.circuit_metrics, metric, None) if br.circuit_metrics else None
        else:
            val = getattr(br, metric, None)

        if val is None:
            continue
        labels.append(lbl)
        values.append(val)
        colours.append(SOLVER_COLOURS.get(ear.solver, "grey"))

    bars = ax.barh(labels, values, color=colours, alpha=0.85, edgecolor="black", lw=0.5)
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=_TICK_FS)

    metric_labels = {
        "depth_opt1":  "Circuit depth (gates, opt. level 1)",
        "depth_opt0":  "Circuit depth (gates, opt. level 0)",
        "n_cx_gates":  "Two-qubit gate count",
        "n_qubits":    "Qubit count",
        "wall_time_s": "Wall time [s]",
    }
    ax.set_xlabel(metric_labels.get(metric, metric), fontsize=_LABEL_FS)
    ax.set_title(
        title or rf"Resource cost at $r \approx {r_target:.0e}$",
        fontsize=_LABEL_FS,
    )
    ax.tick_params(labelsize=_TICK_FS)
    ax.grid(True, alpha=0.25, axis="x", linestyle="--")
    ax.text(
        0.99, 0.02, "* target not achieved within parameter grid",
        ha="right", va="bottom", transform=ax.transAxes,
        fontsize=_TICK_FS - 1, style="italic",
    )
    fig.tight_layout()
    return fig


# -- Figure 8: Error decomposition --------------------------------------------

def error_decomposition(
    results,
    N_values: Optional[list[int]] = None,
    solvers: Optional[list[str]] = None,
    title: str = "",
):
    """
    Stacked bar chart decomposing total error into discretisation and
    algorithmic components.

    Discretisation error: e_disc = max_rel_err(Thomas vs exact).
    Algorithmic error:    e_alg  = max(0, e∞ - e_disc).

    Parameters
    ----------
    results : list[BenchmarkResult]
        Benchmark results. Only results with max_rel_err_vs_exact available
        are included.
    N_values : list[int], optional
        Problem sizes to include.
    solvers : list[str], optional
        Solvers to include (Thomas excluded automatically).
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    if N_values is None:
        N_values = sorted({r.N for r in results})
    if solvers is None:
        solvers = [s for s in ["hhl", "vqls", "qsvt"]
                   if any(r.solver == s for r in results)]

    fig, axes = plt.subplots(
        1, len(N_values),
        figsize=(4.5 * len(N_values), 4.5),
        squeeze=False,
    )

    for col, N in enumerate(N_values):
        ax = axes[0][col]
        n_rows = [r for r in results if r.N == N]

        solver_labels_plot, disc_vals, alg_vals = [], [], []
        for solver in solvers:
            row_list = [r for r in n_rows if r.solver.lower() == solver
                        and r.err_disc is not None]
            if not row_list:
                continue
            row = row_list[0]
            solver_labels_plot.append(SOLVER_LABELS.get(solver, solver.upper()))
            disc_vals.append(row.err_disc or 0.0)
            alg_vals.append(row.err_alg or 0.0)

        x = range(len(solver_labels_plot))
        ax.bar(x, disc_vals, label="Discretisation", color="steelblue",
               alpha=0.8, edgecolor="black", lw=0.5)
        ax.bar(x, alg_vals, bottom=disc_vals, label="Algorithmic",
               color="tomato", alpha=0.8, edgecolor="black", lw=0.5)

        ax.set_xticks(list(x))
        ax.set_xticklabels(solver_labels_plot, fontsize=_TICK_FS)
        ax.set_ylabel(r"$e_\infty$ [\%]", fontsize=_LABEL_FS)
        ax.set_title(f"$N={N}$", fontsize=_LABEL_FS)
        ax.tick_params(labelsize=_TICK_FS)
        ax.grid(True, alpha=0.25, axis="y", linestyle="--")
        if col == 0:
            ax.legend(fontsize=_LEGEND_FS)

    fig.suptitle(title or "Error decomposition: discretisation vs algorithmic",
                 fontsize=_LABEL_FS + 1)
    fig.tight_layout()
    return fig


# -- Figure 9: Order comparison ------------------------------------------------

def order_comparison(
    results_2nd,
    results_4th,
    metric: str = "max_rel_err_vs_exact",
    title: str = "",
):
    """
    Log-log comparison of 2nd-order and 4th-order discretisation accuracy.

    Parameters
    ----------
    results_2nd : list[BenchmarkResult]
        Second-order (tridiagonal) benchmark results.
    results_4th : list[BenchmarkResult]
        Fourth-order (pentadiagonal) benchmark results.
    metric : str
        Accuracy metric to plot.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=_FIG_SINGLE)

    for results, order, ls in [
        (results_2nd, 2, "-"),
        (results_4th, 4, "--"),
    ]:
        for solver in ["thomas", "hhl", "vqls", "qsvt"]:
            rows = sorted(
                [r for r in results if r.solver.lower() == solver],
                key=lambda r: r.N,
            )
            if not rows:
                continue
            N_vals = [r.N for r in rows]
            y_vals = [getattr(r, metric) for r in rows]
            valid = [(n, y) for n, y in zip(N_vals, y_vals) if y is not None]
            if not valid:
                continue
            N_plot, y_plot = zip(*valid)
            lbl = (
                f"{SOLVER_LABELS.get(solver, solver.upper())} "
                f"({order}{'nd' if order == 2 else 'th'}-order)"
            )
            ax.loglog(
                N_plot, y_plot,
                color=SOLVER_COLOURS.get(solver, "grey"),
                linestyle=ls,
                marker=SOLVER_MARKERS.get(solver, "x"),
                markersize=5, lw=1.5, label=lbl,
            )

    # Reference lines
    N_all = sorted({r.N for r in results_2nd + results_4th})
    N_arr = np.array(N_all, dtype=float)
    thomas_2nd = sorted(
        [r for r in results_2nd if r.solver == "thomas" and getattr(r, metric) is not None],
        key=lambda r: r.N,
    )
    if thomas_2nd:
        anchor = thomas_2nd[-1]
        y_a = getattr(anchor, metric)
        ax.loglog(N_arr, y_a * (N_arr / anchor.N) ** (-2),
                  "k:", lw=0.8, label=r"$\mathcal{O}(N^{-2})$", alpha=0.5)
        ax.loglog(N_arr, y_a * (N_arr / anchor.N) ** (-4),
                  "k-.", lw=0.8, label=r"$\mathcal{O}(N^{-4})$", alpha=0.5)

    metric_labels = {
        "max_rel_err_vs_exact":  r"$e_\infty$ [\%] vs exact",
        "residual":              r"$r$",
    }
    _apply_style(
        ax,
        xlabel=r"$N$ (interior nodes)",
        ylabel=metric_labels.get(metric, metric),
        title=title or "2nd-order vs 4th-order discretisation accuracy",
    )
    fig.tight_layout()
    return fig


# -- Figure 10: Hardware vs simulation ----------------------------------------

def hardware_vs_simulation(
    sim_results,
    hw_results,
    metric: str = "max_rel_err_vs_exact",
    title: str = "",
):
    """
    Comparison of statevector simulation vs real hardware results.

    Plots both result sets on the same axes, with hardware results shown
    as filled markers and simulation results as open markers.

    Parameters
    ----------
    sim_results : list[BenchmarkResult]
        Statevector simulation results (hardware_run=False).
    hw_results : list[BenchmarkResult]
        Real hardware results (hardware_run=True).
    metric : str
        Accuracy metric to compare.
    title : str
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=_FIG_SINGLE)

    for results, label_suffix, fill in [
        (sim_results, " (sim.)", False),
        (hw_results,  " (HW)",  True),
    ]:
        for solver in ["hhl", "vqls", "qsvt"]:
            rows = sorted(
                [r for r in results if r.solver.lower() == solver],
                key=lambda r: r.N,
            )
            if not rows:
                continue
            N_vals = [r.N for r in rows]
            y_vals = [getattr(r, metric) for r in rows]
            valid = [(n, y) for n, y in zip(N_vals, y_vals) if y is not None]
            if not valid:
                continue
            N_plot, y_plot = zip(*valid)
            lbl = SOLVER_LABELS.get(solver, solver.upper()) + label_suffix
            mfc = SOLVER_COLOURS.get(solver, "grey") if fill else "white"
            ax.semilogy(
                N_plot, y_plot,
                color=SOLVER_COLOURS.get(solver, "grey"),
                marker=SOLVER_MARKERS.get(solver, "x"),
                markerfacecolor=mfc,
                markersize=7, lw=1.5, label=lbl,
            )

    metric_labels = {
        "max_rel_err_vs_exact":  r"$e_\infty$ [\%] vs exact",
        "residual":              r"$r$",
    }
    _apply_style(
        ax,
        xlabel=r"$N$ (interior nodes)",
        ylabel=metric_labels.get(metric, metric),
        title=title or "Statevector simulation vs real hardware",
    )
    fig.tight_layout()
    return fig


# -- Batch save utility --------------------------------------------------------

def save_figure(fig, path, dpi: int = 300, formats: tuple = ("pdf", "png")) -> None:
    """
    Save a matplotlib Figure to one or more formats.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or Path
        Output path without extension. The extension is added per format.
    dpi : int
        Resolution for raster formats (PNG). Default 300 for publication.
    formats : tuple[str]
        Output formats. Default ('pdf', 'png').
    """
    from pathlib import Path as _Path
    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(p.with_suffix(f".{fmt}"), dpi=dpi, bbox_inches="tight")