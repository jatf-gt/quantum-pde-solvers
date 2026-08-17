"""
benchmark/tables.py
--------------------------------------------------------------------------------
Publication-standard table generation for the quantum PDE solver benchmark.

Produces two output formats from BenchmarkResult collections:

  1. LaTeX tables — directly insertable into the thesis via \\input{}.
     Formatted to match the booktabs style used in the thesis template
     (usepackages.tex: \\usepackage{booktabs}, \\usepackage{siunitx}).

  2. Console tables — aligned ASCII for HPC log inspection and
     interactive debugging sessions.

Table catalogue
---------------
  primary_comparison   Primary benchmark: all solvers × all N, fixed parameters.
  equal_accuracy       Equal-accuracy protocol: resource cost at matched residual.
  sensitivity          OAT sensitivity: one parameter swept, all metrics shown.
  circuit_resources    Circuit depth, qubit count, two-qubit gate count.
  het_application      HET plasma application results.
  order_comparison     2nd-order vs 4th-order discretisation at matched accuracy.

Mathematical notation
---------------------
  κ     condition number of A
  r     relative residual ‖Au - b‖₂ / ‖b‖₂
  e∞    maximum relative error [%]
  d     circuit depth (gates, optimisation level 1)
  nq    qubit count
  t     wall time [s]

References
----------
  Ghafourpour & Laizet (2025) Phys. Rev. Applied 24, 024032.
  Bravo-Prieto et al. (2023) Quantum 7, 1188.
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Optional, Sequence

from benchmark.metrics import BenchmarkResult
from benchmark.equal_accuracy import EqualAccuracyResult
from benchmark.sensitivity import SensitivitySweepResult


# -- Formatting helpers --------------------------------------------------------

def _fmt_sci(val: Optional[float], decimals: int = 2) -> str:
    """Format a float in scientific notation, or '---' if None."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "---"
    if val == 0.0:
        return "0"
    exp = int(math.floor(math.log10(abs(val))))
    mantissa = val / 10 ** exp
    return f"{mantissa:.{decimals}f}e{exp:+03d}"


def _fmt_pct(val: Optional[float], decimals: int = 3) -> str:
    """Format a percentage value, or '---' if None."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "---"
    return f"{val:.{decimals}f}"


def _fmt_int(val: Optional[int]) -> str:
    """Format an integer with thousands separator, or '---' if None."""
    if val is None:
        return "---"
    return f"{val:,}"


def _fmt_float(val: Optional[float], decimals: int = 2) -> str:
    """
    Format a fixed-point quantity for console output, or '---' if unrecorded.

    The console counterpart of `_latex_num`. Recovered rows carry None for every
    quantity that existed only in a killed process's memory — the row condition
    number among them — and an unguarded format specification raises on None,
    which takes an entire table down for one absent entry.

    Parameters
    ----------
    val : float or None
        Quantity to render.
    decimals : int
        Digits after the decimal point.

    Returns
    -------
    str
        Fixed-point rendering, or '---' where the value is absent or not a number.
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "---"
    return f"{val:.{decimals}f}"


def _fmt_time(val: Optional[float]) -> str:
    """Format wall time in seconds with adaptive precision."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "---"
    if val < 0.001:
        return f"{val*1000:.2f} ms"
    if val < 60.0:
        return f"{val:.2f} s"
    return f"{val/60.0:.1f} min"


def _solver_label(solver: str) -> str:
    """Return a display-friendly solver label."""
    return {
        "thomas": "Thomas",
        "hhl":    "HHL",
        "vqls":   "VQLS",
        "qsvt":   "QSVT",
    }.get(solver.lower(), solver.upper())


def _latex_sci(val: Optional[float], decimals: int = 2) -> str:
    """Format a float in LaTeX scientific notation for siunitx."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return r"{\text{---}}"
    if val == 0.0:
        return "0"
    exp = int(math.floor(math.log10(abs(val))))
    mantissa = val / 10 ** exp
    return rf"{mantissa:.{decimals}f} \times 10^{{{exp}}}"


def _latex_pct(val: Optional[float], decimals: int = 3) -> str:
    """Format a percentage for LaTeX."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return r"{\text{---}}"
    return f"{val:.{decimals}f}"


def _latex_num(val: Optional[float], decimals: int = 1) -> str:
    """
    Format a fixed-point quantity for LaTeX, or an em-dash rule if unrecorded.

    Required wherever a column may be populated from a *recovered* row. A sweep
    killed mid-work-unit loses the instrumented record but not the per-solution
    archive, and `scripts/utils/recover_orphan_rows.py` deliberately leaves the
    fields that existed only in the killed process's memory — wall time,
    strip-solve counts, the row condition number — as None rather than zero. A
    zero would read as "this solve was free", the opposite of the truth. This
    formatter renders that absence explicitly instead of raising, which is what
    an unguarded format specification does on None.

    Parameters
    ----------
    val : float or None
        Quantity to render.
    decimals : int
        Digits after the decimal point.

    Returns
    -------
    str
        Fixed-point rendering, or ``\\text{---}`` where the value is absent or
        not a number.
    """
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return r"{\text{---}}"
    return f"{val:.{decimals}f}"


def _latex_int(val: Optional[int]) -> str:
    """
    Format an integer for a siunitx `S` column, or an em-dash rule if unrecorded.

    The LaTeX counterpart of `_fmt_int`. The thousands separator is omitted:
    siunitx applies its own digit grouping, and a literal comma inside an `S`
    column is parsed as a decimal marker under several locale settings.

    Parameters
    ----------
    val : int or None
        Quantity to render.

    Returns
    -------
    str
        The integer, or a brace-wrapped em-dash where the value is absent.
    """
    if val is None:
        return r"{\text{---}}"
    return f"{int(val)}"


def _latex_case(case_id: str) -> str:
    """
    Render a case identifier as LaTeX text, escaping the underscores.

    Case identifiers are snake_case (`2D_HET_MMS_SPT100`), and an unescaped
    underscore is a subscript operator in LaTeX maths mode; inside an `S` column
    from siunitx it aborts the compile outright.

    Parameters
    ----------
    case_id : str
        Case identifier as recorded by the sweep.

    Returns
    -------
    str
        The identifier wrapped in `\\text{}` with every underscore escaped.
    """
    return r"\text{" + str(case_id).replace("_", r"\_") + "}"


# -- LaTeX table builders ------------------------------------------------------

def _latex_header(caption: str, label: str, col_spec: str) -> str:
    return (
        "\\begin{table}[htbp]\n"
        "  \\centering\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{{label}}}\n"
        f"  \\begin{{tabular}}{{{col_spec}}}\n"
        "    \\toprule\n"
    )


def _latex_footer() -> str:
    return (
        "    \\bottomrule\n"
        "  \\end{tabular}\n"
        "\\end{table}\n"
    )


def latex_primary_comparison(
    results: list[BenchmarkResult],
    caption: str = (
        "Primary benchmark: maximum relative error, residual, circuit depth, "
        "qubit count, and wall time for every solver, grouped by case and "
        "resolution. "
        "Thomas serves as the classical reference; errors are reported "
        "relative to the analytical solution where available, otherwise "
        "relative to the Thomas solution. "
        r"An em-dash denotes a quantity the sweep did not record: a solve "
        r"reconstructed from its solution archive retains every field derived "
        r"from the field itself and none that existed only in the process."
    ),
    label: str = "tab:primary_comparison",
    N_values: Optional[list[int]] = None,
    solvers: Optional[list[str]] = None,
) -> str:
    """
    Generate the primary comparison LaTeX table.

    Columns: N | κ | Solver | e∞ [%] | r | d | nq | t [s]

    Parameters
    ----------
    results : list[BenchmarkResult]
        All benchmark results to include. Filtered by N_values and solvers.
    caption : str
        LaTeX table caption.
    label : str
        LaTeX \\label key for cross-referencing.
    N_values : list[int], optional
        Problem sizes to include. If None, all N in results are used.
    solvers : list[str], optional
        Solvers to include. If None, all solvers in results are used.

    Returns
    -------
    str
        Complete LaTeX table as a string.
    """
    if N_values is None:
        N_values = sorted({r.N for r in results})
    if solvers is None:
        solvers = ["thomas", "hhl", "vqls", "qsvt"]

    col_spec = "l S[table-format=3.0] S[table-format=4.1] l S[table-format=3.3] S[table-format=1.2e2] S[table-format=6.0] S[table-format=2.0] S[table-format=4.2]"

    buf = io.StringIO()
    buf.write(_latex_header(caption, label, col_spec))
    buf.write(
        "    {Case} & {$N$} & {$\\kappa(A)$} & {Solver} & "
        "{$e_\\infty$ [\\%]} & {$r$} & "
        "{Depth} & {$n_q$} & {$t$ [s]} \\\\\n"
        "    \\midrule\n"
    )

    # Grouped by case first. The sweeps carry between one and seven cases at each
    # resolution, and selecting a single row per (N, solver) — as this builder did
    # while it served the one-case laptop runner — silently discarded every case
    # but the first, with nothing in the rendered table to indicate which had
    # survived.
    cases = sorted({r.case_id for r in results})
    for case_idx, case_id in enumerate(cases):
        case_rows = [r for r in results if r.case_id == case_id]
        first_case = True
        for N in N_values:
            n_rows = [r for r in case_rows if r.N == N]
            first_N = True
            for solver in solvers:
                row_list = [r for r in n_rows if r.solver.lower() == solver]
                if not row_list:
                    continue
                row = row_list[0]

                case_str = _latex_case(case_id) if first_case else ""
                first_case = False
                kappa_str = _latex_num(row.kappa) if first_N else ""
                N_str = str(N) if first_N else ""
                first_N = False

                err = row.max_rel_err_vs_exact
                if err is None:
                    err = row.max_rel_err_vs_thomas
                err_str = _latex_pct(err)

                res_str = _latex_sci(row.residual)
                depth_str = (
                    _latex_int(row.circuit_metrics.depth_opt1)
                    if row.circuit_metrics else r"{\text{---}}"
                )
                nq_str = (
                    _latex_int(row.circuit_metrics.n_qubits)
                    if row.circuit_metrics else r"{\text{---}}"
                )
                time_str = _latex_num(row.wall_time_s, 2)

                buf.write(
                    f"    {case_str} & {N_str} & {kappa_str} & "
                    f"\\text{{{_solver_label(solver)}}} & "
                    f"{err_str} & ${res_str}$ & "
                    f"{depth_str} & {nq_str} & {time_str} \\\\\n"
                )
        if case_idx != len(cases) - 1:
            buf.write("    \\midrule\n")

    buf.write(_latex_footer())
    return buf.getvalue()


def latex_equal_accuracy(
    ea_results: list[EqualAccuracyResult],
    r_target: float,
    caption: str = (
        "Equal-accuracy comparison: circuit depth, qubit count, and wall time "
        "for each quantum solver at a matched relative residual target "
        r"$r_\mathrm{target}$. "
        "The parameter value required to achieve the target is shown for each "
        "algorithm. Entries marked $\\dagger$ did not achieve the target within "
        "the parameter grid; the closest result is reported."
    ),
    label: str = "tab:equal_accuracy",
) -> str:
    """
    Generate the equal-accuracy comparison LaTeX table.

    Columns: Solver | Parameter | Value | r achieved | d | nq | t [s]

    Parameters
    ----------
    ea_results : list[EqualAccuracyResult]
        One result per solver from the equal-accuracy sweep.
    r_target : float
        Target residual used in the sweep (for caption).
    """
    col_spec = "l l S[table-format=1.3] S[table-format=1.2e2] S[table-format=6.0] S[table-format=2.0] S[table-format=4.2]"

    buf = io.StringIO()
    buf.write(_latex_header(caption, label, col_spec))
    buf.write(
        "    {Solver} & {Parameter} & {Value} & "
        "{$r$ achieved} & {Depth} & {$n_q$} & {$t$ [s]} \\\\\n"
        "    \\midrule\n"
    )

    param_labels = {
        "epsilon":    r"$\varepsilon$",
        "n_layers":   r"$n_\mathrm{layers}$",
        "n_restarts": r"$n_\mathrm{restarts}$",
        "max_degree": r"$d_\mathrm{max}$",
    }

    for ear in ea_results:
        br = ear.best_result
        dagger = "" if ear.in_band else r"$^\dagger$"

        param = br.sensitivity_param or "---"
        param_label = param_labels.get(param, param)

        val = br.sensitivity_value
        if val is not None and val < 0:
            val_str = "uncapped"
        elif val is not None:
            val_str = f"{val:.3g}"
        else:
            val_str = "---"

        depth_str = (
            _latex_int(br.circuit_metrics.depth_opt1)
            if br.circuit_metrics else r"{\text{---}}"
        )
        nq_str = (
            str(br.circuit_metrics.n_qubits)
            if br.circuit_metrics else r"{\text{---}}"
        )

        buf.write(
            f"    \\text{{{_solver_label(ear.solver)}}}{dagger} & "
            f"{param_label} & "
            f"\\text{{{val_str}}} & "
            f"${_latex_sci(br.residual)}$ & "
            f"{depth_str} & {nq_str} & {_latex_num(br.wall_time_s, 2)} \\\\\n"
        )

    buf.write(
        f"    \\multicolumn{{7}}{{l}}"
        f"{{\\footnotesize $r_\\mathrm{{target}} = {_latex_sci(r_target)}$; "
        f"$\\dagger$ target not achieved within parameter grid.}}\\\\\n"
    )
    buf.write(_latex_footer())
    return buf.getvalue()


def latex_sensitivity(
    sweep_results: list[SensitivitySweepResult],
    solver: str,
    N: int,
    caption: Optional[str] = None,
    label: str = "tab:sensitivity",
) -> str:
    """
    Generate a sensitivity analysis LaTeX table for one solver.

    One sub-table per parameter, separated by \\midrule.
    Columns: Parameter value | e∞ [%] | r | d | t [s]

    Parameters
    ----------
    sweep_results : list[SensitivitySweepResult]
        All OAT sweep results for the specified solver.
    solver : str
        Solver name (for caption and label).
    N : int
        Problem size (for caption).
    """
    if caption is None:
        caption = (
            f"Sensitivity analysis for the {_solver_label(solver)} algorithm "
            f"at $N={N}$. Each block varies one parameter while holding all "
            "others at the baseline value. "
            r"$e_\infty$ is the maximum relative error against the analytical "
            "solution; $r$ is the relative residual."
        )

    col_spec = "l S[table-format=3.3] S[table-format=1.2e2] S[table-format=6.0] S[table-format=4.2]"

    param_labels = {
        "epsilon":    r"$\varepsilon$",
        "n_layers":   r"$n_\mathrm{layers}$",
        "n_restarts": r"$n_\mathrm{restarts}$",
        "cobyla_tol": r"$\tau_\mathrm{COBYLA}$",
        "max_degree": r"$d_\mathrm{max}$",
    }

    buf = io.StringIO()
    buf.write(_latex_header(caption, label, col_spec))
    buf.write(
        "    {Parameter value} & {$e_\\infty$ [\\%]} & "
        "{$r$} & {Depth} & {$t$ [s]} \\\\\n"
    )

    for idx, sweep in enumerate(sweep_results):
        if idx > 0:
            buf.write("    \\midrule\n")
        plabel = param_labels.get(sweep.param_name, sweep.param_name)
        buf.write(f"    \\multicolumn{{5}}{{l}}{{\\textit{{Varying {plabel}}}}}\\\\\n")
        buf.write("    \\midrule\n")

        for res in sweep.results:
            val = res.sensitivity_value
            if val is not None and val < 0:
                val_str = "uncapped"
            elif val is not None:
                val_str = f"{val:.3g}"
            else:
                val_str = "---"

            err = res.max_rel_err_vs_exact
            if err is None:
                err = res.max_rel_err_vs_thomas
            err_str = _latex_pct(err)

            depth_str = (
                _latex_int(res.circuit_metrics.depth_opt1)
                if res.circuit_metrics else r"{\text{---}}"
            )

            buf.write(
                f"    \\text{{{val_str}}} & "
                f"{err_str} & "
                f"${_latex_sci(res.residual)}$ & "
                f"{depth_str} & "
                f"{_latex_num(res.wall_time_s, 2)} \\\\\n"
            )

    buf.write(_latex_footer())
    return buf.getvalue()


def latex_circuit_resources(
    results: list[BenchmarkResult],
    caption: str = (
        "Circuit resource metrics for quantum solvers across problem sizes. "
        r"$d_0$: circuit depth at Qiskit optimisation level 0 (logical). "
        r"$d_1$: circuit depth at optimisation level 1 (light compilation). "
        r"$n_\mathrm{CX}$: two-qubit gate count at level 1. "
        r"$n_q$: total qubit count."
    ),
    label: str = "tab:circuit_resources",
    N_values: Optional[list[int]] = None,
) -> str:
    """
    Generate a circuit resource metrics LaTeX table.

    Columns: N | κ | Solver | d_raw | d_opt0 | d_opt1 | n_CX | n_q
    """
    if N_values is None:
        N_values = sorted({r.N for r in results if r.solver != "thomas"})

    col_spec = "l S[table-format=3.0] S[table-format=4.1] l S[table-format=6.0] S[table-format=6.0] S[table-format=6.0] S[table-format=5.0] S[table-format=2.0]"

    buf = io.StringIO()
    buf.write(_latex_header(caption, label, col_spec))
    buf.write(
        "    {Case} & {$N$} & {$\\kappa$} & {Solver} & "
        "{$d_\\mathrm{raw}$} & {$d_0$} & {$d_1$} & "
        "{$n_\\mathrm{CX}$} & {$n_q$} \\\\\n"
        "    \\midrule\n"
    )

    # Grouped by case for the same reason as the primary comparison: without the
    # column, a multi-case sweep emits several identically labelled rows per
    # (N, solver) whose provenance cannot be recovered from the rendered table.
    cases = sorted({r.case_id for r in results if r.solver != "thomas"})
    for case_idx, case_id in enumerate(cases):
        case_rows = [r for r in results
                     if r.case_id == case_id and r.solver != "thomas"]
        first_case = True
        for N in N_values:
            n_rows = [r for r in case_rows if r.N == N]
            first_N = True
            for row in n_rows:
                if row.circuit_metrics is None:
                    continue
                cm = row.circuit_metrics
                case_str = _latex_case(case_id) if first_case else ""
                first_case = False
                N_str = str(N) if first_N else ""
                kappa_str = _latex_num(row.kappa) if first_N else ""
                first_N = False

                buf.write(
                    f"    {case_str} & {N_str} & {kappa_str} & "
                    f"\\text{{{_solver_label(row.solver)}}} & "
                    f"{_latex_int(cm.depth_raw)} & "
                    f"{_latex_int(cm.depth_opt0)} & "
                    f"{_latex_int(cm.depth_opt1)} & "
                    f"{_latex_int(cm.n_cx_gates)} & "
                    f"{_latex_int(cm.n_qubits)} \\\\\n"
                )
        if case_idx != len(cases) - 1:
            buf.write("    \\midrule\n")

    buf.write(_latex_footer())
    return buf.getvalue()


def latex_order_comparison(
    results_2nd: list[BenchmarkResult],
    results_4th: list[BenchmarkResult],
    caption: str = (
        "Comparison of second-order (tridiagonal TST) and fourth-order "
        "(pentadiagonal) discretisations at matched solution accuracy. "
        r"$\kappa_\mathrm{ratio} = \kappa_\mathrm{4th} / \kappa_\mathrm{2nd}$ "
        "quantifies the condition number penalty of the higher-order stencil."
    ),
    label: str = "tab:order_comparison",
) -> str:
    """
    Generate a 2nd-order vs 4th-order discretisation comparison table.

    Columns: N | κ_2nd | κ_4th | κ_ratio | e∞_2nd [%] | e∞_4th [%]
    """
    N_values = sorted({r.N for r in results_2nd})
    col_spec = "S[table-format=2.0] S[table-format=4.1] S[table-format=4.1] S[table-format=1.3] S[table-format=3.3] S[table-format=3.3]"

    buf = io.StringIO()
    buf.write(_latex_header(caption, label, col_spec))
    buf.write(
        "    {$N$} & {$\\kappa_\\mathrm{2nd}$} & "
        "{$\\kappa_\\mathrm{4th}$} & "
        "{$\\kappa_\\mathrm{ratio}$} & "
        "{$e_\\infty^\\mathrm{2nd}$ [\\%]} & "
        "{$e_\\infty^\\mathrm{4th}$ [\\%]} \\\\\n"
        "    \\midrule\n"
    )

    for N in N_values:
        r2 = next((r for r in results_2nd if r.N == N and r.solver == "thomas"), None)
        r4 = next((r for r in results_4th if r.N == N and r.solver == "thomas"), None)
        if r2 is None or r4 is None:
            continue

        kappa_ratio = (
            r4.kappa / r2.kappa
            if (r2.kappa and r4.kappa is not None) else float("nan")
        )
        buf.write(
            f"    {N} & {_latex_num(r2.kappa)} & {_latex_num(r4.kappa)} & "
            f"{_latex_num(kappa_ratio, 3)} & "
            f"{_latex_pct(r2.max_rel_err_vs_exact)} & "
            f"{_latex_pct(r4.max_rel_err_vs_exact)} \\\\\n"
        )

    buf.write(_latex_footer())
    return buf.getvalue()


# -- Console table builders ----------------------------------------------------

def console_primary_comparison(
    results: list[BenchmarkResult],
    N_values: Optional[list[int]] = None,
    solvers: Optional[list[str]] = None,
    show_circuit: bool = True,
) -> str:
    """
    Generate an aligned ASCII primary comparison table for console output.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Benchmark results to tabulate.
    N_values : list[int], optional
        Problem sizes to include.
    solvers : list[str], optional
        Solvers to include.
    show_circuit : bool
        If True, include circuit depth and qubit count columns.

    Returns
    -------
    str
        Formatted ASCII table.
    """
    if N_values is None:
        N_values = sorted({r.N for r in results})
    if solvers is None:
        solvers = ["thomas", "hhl", "vqls", "qsvt"]

    if show_circuit:
        header = (
            f"  {'Case':<34}{'N':>4}  {'κ':>8}  {'Solver':<8}  "
            f"{'e∞ [%]':>10}  {'r':>10}  "
            f"{'Depth':>8}  {'nq':>4}  {'t':>10}"
        )
        sep = "  " + "─" * (len(header) - 2)
    else:
        header = (
            f"  {'Case':<34}{'N':>4}  {'κ':>8}  {'Solver':<8}  "
            f"{'e∞ [%]':>10}  {'r':>10}  {'t':>10}"
        )
        sep = "  " + "─" * (len(header) - 2)

    buf = io.StringIO()
    buf.write(sep + "\n")
    buf.write(header + "\n")
    buf.write(sep + "\n")

    # Grouped by case, matching `latex_primary_comparison`: a multi-case sweep
    # otherwise reports only whichever case happened to be listed first.
    for case_id in sorted({r.case_id for r in results}):
        case_rows = [r for r in results if r.case_id == case_id]
        first_case = True
        for N in N_values:
            n_rows = [r for r in case_rows if r.N == N]
            for solver in solvers:
                row_list = [r for r in n_rows if r.solver.lower() == solver]
                if not row_list:
                    continue
                row = row_list[0]

                case_str = str(case_id)[:33] if first_case else ""
                first_case = False

                err = row.max_rel_err_vs_exact
                if err is None:
                    err = row.max_rel_err_vs_thomas
                err_str = _fmt_pct(err) if err is not None else "---"

                if show_circuit and row.circuit_metrics:
                    cm = row.circuit_metrics
                    # Formatted through _fmt_int rather than directly: a solver
                    # that timed out or was skipped records no circuit at all, and
                    # a partially populated CircuitMetrics is the normal case when
                    # rows are read back from an archive whose sweep predates a
                    # given column. Direct formatting raises on None instead of
                    # printing a dash, which took the whole table down for one
                    # absent entry.
                    buf.write(
                        f"  {case_str:<34}{N:>4}  {_fmt_float(row.kappa):>8}  "
                        f"{_solver_label(solver):<8}  "
                        f"{err_str:>10}  "
                        f"{_fmt_sci(row.residual):>10}  "
                        f"{_fmt_int(cm.depth_opt1):>8}  "
                        f"{_fmt_int(cm.n_qubits):>4}  "
                        f"{_fmt_time(row.wall_time_s):>10}\n"
                    )
                else:
                    buf.write(
                        f"  {case_str:<34}{N:>4}  {_fmt_float(row.kappa):>8}  "
                        f"{_solver_label(solver):<8}  "
                        f"{err_str:>10}  "
                        f"{_fmt_sci(row.residual):>10}  "
                        f"{_fmt_time(row.wall_time_s):>10}\n"
                    )
        buf.write(sep + "\n")

    return buf.getvalue()


def console_equal_accuracy(
    ea_results: list[EqualAccuracyResult],
    r_target: float,
) -> str:
    """
    Generate an aligned ASCII equal-accuracy comparison table.

    Parameters
    ----------
    ea_results : list[EqualAccuracyResult]
        One result per solver.
    r_target : float
        Target residual for the sweep.
    """
    header = (
        f"  {'Solver':<8}  {'Param':<12}  {'Value':>10}  "
        f"{'r achieved':>12}  {'Depth':>8}  {'nq':>4}  {'t':>10}  {'In band':>8}"
    )
    sep = "  " + "─" * (len(header) - 2)

    buf = io.StringIO()
    buf.write(f"\n  Equal-accuracy comparison  (r_target = {_fmt_sci(r_target)})\n")
    buf.write(sep + "\n")
    buf.write(header + "\n")
    buf.write(sep + "\n")

    for ear in ea_results:
        br = ear.best_result
        val = br.sensitivity_value
        val_str = "uncapped" if (val is not None and val < 0) else (
            f"{val:.3g}" if val is not None else "---"
        )
        depth_str = (
            _fmt_int(br.circuit_metrics.depth_opt1)
            if br.circuit_metrics else "---"
        )
        nq_str = (
            str(br.circuit_metrics.n_qubits)
            if br.circuit_metrics else "---"
        )
        band_str = "YES" if ear.in_band else "NO *"

        buf.write(
            f"  {_solver_label(ear.solver):<8}  "
            f"{(br.sensitivity_param or '---'):<12}  "
            f"{val_str:>10}  "
            f"{_fmt_sci(br.residual):>12}  "
            f"{depth_str:>8}  {nq_str:>4}  "
            f"{_fmt_time(br.wall_time_s):>10}  {band_str:>8}\n"
        )

    buf.write(sep + "\n")
    buf.write("  * Target not achieved within parameter grid.\n")
    return buf.getvalue()


def console_sensitivity(
    sweep_results: list[SensitivitySweepResult],
    solver: str,
    N: int,
) -> str:
    """
    Generate an aligned ASCII sensitivity table for one solver.

    Parameters
    ----------
    sweep_results : list[SensitivitySweepResult]
        All OAT sweeps for the solver.
    solver : str
        Solver name.
    N : int
        Problem size.
    """
    buf = io.StringIO()
    buf.write(
        f"\n  Sensitivity analysis — {_solver_label(solver)}  N={N}\n"
    )

    for sweep in sweep_results:
        header = (
            f"  {'Value':>12}  {'e∞ [%]':>10}  "
            f"{'r':>10}  {'Depth':>8}  {'t':>10}"
        )
        sep = "  " + "─" * (len(header) - 2)
        buf.write(f"\n  Varying: {sweep.param_name}\n")
        buf.write(sep + "\n")
        buf.write(header + "\n")
        buf.write(sep + "\n")

        for res in sweep.results:
            val = res.sensitivity_value
            val_str = "uncapped" if (val is not None and val < 0) else (
                f"{val:.3g}" if val is not None else "---"
            )
            err = res.max_rel_err_vs_exact
            if err is None:
                err = res.max_rel_err_vs_thomas
            depth_str = (
                f"{res.circuit_metrics.depth_opt1:,}"
                if res.circuit_metrics else "---"
            )
            buf.write(
                f"  {val_str:>12}  "
                f"{_fmt_pct(err):>10}  "
                f"{_fmt_sci(res.residual):>10}  "
                f"{depth_str:>8}  "
                f"{_fmt_time(res.wall_time_s):>10}\n"
            )
        buf.write(sep + "\n")

    return buf.getvalue()


# -- File output ---------------------------------------------------------------

def save_latex_tables(
    output_dir: Path,
    primary_results: Optional[list[BenchmarkResult]] = None,
    ea_results: Optional[list[EqualAccuracyResult]] = None,
    sensitivity_results: Optional[dict[str, list[SensitivitySweepResult]]] = None,
    results_2nd: Optional[list[BenchmarkResult]] = None,
    results_4th: Optional[list[BenchmarkResult]] = None,
    r_target: float = 1.0e-3,
) -> list[Path]:
    """
    Generate and save all applicable LaTeX tables to output_dir.

    Each table is saved as a separate .tex file for \\input{} inclusion.

    Parameters
    ----------
    output_dir : Path
        Directory in which to write .tex files.
    primary_results : list[BenchmarkResult], optional
        Results for the primary comparison table.
    ea_results : list[EqualAccuracyResult], optional
        Results for the equal-accuracy table.
    sensitivity_results : dict, optional
        Mapping solver -> list[SensitivitySweepResult] for sensitivity tables.
    results_2nd : list[BenchmarkResult], optional
        Second-order results for the order comparison table.
    results_4th : list[BenchmarkResult], optional
        Fourth-order results for the order comparison table.
    r_target : float
        Target residual for the equal-accuracy table caption.

    Returns
    -------
    list[Path]
        Paths of all files written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    if primary_results:
        path = output_dir / "tab_primary_comparison.tex"
        path.write_text(latex_primary_comparison(primary_results), encoding="utf-8")
        written.append(path)

        path = output_dir / "tab_circuit_resources.tex"
        path.write_text(
            latex_circuit_resources(primary_results), encoding="utf-8"
        )
        written.append(path)

    if ea_results:
        path = output_dir / "tab_equal_accuracy.tex"
        path.write_text(
            latex_equal_accuracy(ea_results, r_target=r_target), encoding="utf-8"
        )
        written.append(path)

    if sensitivity_results:
        for solver, sweeps in sensitivity_results.items():
            if not sweeps:
                continue
            N = sweeps[0].results[0].N if sweeps[0].results else 0
            path = output_dir / f"tab_sensitivity_{solver}.tex"
            path.write_text(
                latex_sensitivity(sweeps, solver=solver, N=N), encoding="utf-8"
            )
            written.append(path)

    if results_2nd and results_4th:
        path = output_dir / "tab_order_comparison.tex"
        path.write_text(
            latex_order_comparison(results_2nd, results_4th), encoding="utf-8"
        )
        written.append(path)

    return written