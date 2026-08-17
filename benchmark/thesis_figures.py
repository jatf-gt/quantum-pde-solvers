"""
Assembly of the main-body thesis figures from the recorded sweeps.

Purpose
-------
`benchmark/hpc_plotting.py` renders every figure a sweep supports — several
hundred across the six sweeps — because its job is diagnosis. The thesis body is
limited to forty pages including the abstract and the references, which admits
of order ten figures in total. This module is the selection layer: it assembles
exactly the series each planned main-body figure draws, writes them as tidy CSV,
and renders a reference plot of each.

The CSV is the deliverable and the reference plot is the check. A figure typeset
in another tool must plot the same numbers on the same axes in the same units,
and re-deriving those numbers by a second, independently written path is how two
figures of the same quantity come to disagree. Here there is one path.

Unit conventions, which are not uniform on disk
-----------------------------------------------
Three error columns coexist in the archives and two of them do not share a scale.
This module normalises all of them to **per cent** on the way out, and every CSV
it writes states the unit in the column name.

  `max_rel_err`   Per cent in the 1-D and 2-D/3-D primary sweeps.
                  Exception: rows carrying `notes="recovered_from_npz"` in the
                  2-D order-2 sweep record it as a *fraction*. Those rows are
                  excluded from any series drawn from this column; `linf_err`
                  and `rel_l2_err` are consistent across every row and are
                  preferred wherever a recovered row must be included.
  `err_alg`       Fraction in the primary sweeps; per cent in the study
  `err_disc`      archives written by `hpc/runners/run_studies.py`.
  `linf_err`      Per cent, consistently, in 2-D and 3-D.

Error taxonomy
--------------
The decomposition the figures rest on separates two independent error sources:

    e_total ≈ e_disc + e_alg

  e_disc  Truncation error of the stencil: ‖u_Thomas − u_exact‖ / ‖u_exact‖.
          Falls as O(h^p) with the discretisation order p and is a property of
          the mesh alone, identical for every solver on that mesh.
  e_alg   The solver's own error: ‖u_solver − u_Thomas‖ / ‖u_Thomas‖. Zero by
          construction for Thomas; for a quantum solver it carries the
          Hamiltonian-simulation, phase-estimation, variational or polynomial
          approximation error of that algorithm.

The comparison that matters for a PDE solver is not whether e_alg is small in
absolute terms but whether it is small against e_disc: precision in the linear
solve beyond the truncation error of the stencil is discarded by the
discretisation and is bought for nothing.

References
----------
  Ghafourpour & Laizet (2025) Phys. Rev. Applied 24, 024032.
  Gilyén, A., Su, Y., Low, G. H. & Wiebe, N. (2019). Quantum singular value
      transformation and beyond. STOC 2019, 193–204.
  Harrow, A. W., Hassidim, A. & Lloyd, S. (2009). Quantum algorithm for linear
      systems of equations. Phys. Rev. Lett. 103, 150502.
  Preskill, J. (2018). Quantum Computing in the NISQ era and beyond.
      Quantum, 2, 79.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("thesis_figures")

# ── Layout of the Recorded Sweeps ──────────────────────────────────────────────

SWEEP_DIR: dict[tuple[int, int], str] = {
    (1, 2): "results/1Dhpc_run",   (1, 4): "results/1Dhpc_run_4th",
    (2, 2): "results/2Dhpc_run",   (2, 4): "results/2Dhpc_run_4th",
    (3, 2): "results/3Dhpc_run",   (3, 4): "results/3Dhpc_run_4th",
}

# The case carried through the main body for each dimension. Chosen as a
# manufactured solution with an exact analytical form, so the discretisation
# error is a measurement rather than an estimate against a fine mesh. Every other
# case is reported in the appendix.
#
# In one dimension the *non-homogeneous* sinusoid is carried rather than the
# homogeneous one, deliberately. The homogeneous case has a right-hand side that
# is a single eigenvector of the discrete operator, which makes the inversion a
# scalar division that any polynomial accurate at one eigenvalue performs
# exactly. QSVT reports machine precision there at every degree, and presenting
# that as the headline result would credit the algorithm with an accuracy the
# problem supplied. The non-homogeneous case attains its formal order at both
# stencils and excites the whole spectrum.
PRIMARY_CASE: dict[int, str] = {
    1: "1D_Poisson_fS_nonhom",
    2: "2D_Poisson_sin_hom",
    3: "3D_Poisson_TripleSin_cube",
}

# The HET case carried through the application chapter, per dimension.
HET_CASE: dict[int, str] = {
    1: "HET_1D_3b_gaussian_Vd300",
    2: "2D_HET_MMS_SPT100",
    3: "3D_HET_MMS_SPT100",
}

SOLVERS: tuple[str, ...] = ("Thomas", "HHL", "VQLS", "QSVT")
QUANTUM_SOLVERS: tuple[str, ...] = ("HHL", "VQLS", "QSVT")

SOLVER_COLOUR: dict[str, str] = {
    "Thomas": "#000000",
    "HHL":    "#1f77b4",
    "VQLS":   "#2ca02c",
    "QSVT":   "#d62728",
}
SOLVER_MARKER: dict[str, str] = {
    "Thomas": "s", "HHL": "o", "VQLS": "^", "QSVT": "D",
}

# Degree-to-condition-number ratio below which the QSP polynomial ceases to
# approximate 1/x over the spectrum of A. Established empirically at ~11 from the
# 1-D sweeps; quoted here so the threshold drawn on the figure and the threshold
# discussed in the text cannot drift apart.
DEGREE_KAPPA_THRESHOLD: float = 11.0

# Per-gate two-qubit error rate measured on ibm_kingston, 2026-08-13 calibration
# (median over 352 pairs). Supersedes the 1e-3 placeholder used by
# `hpc/runners/make_tables.py`, which is a round number rather than a measurement
# and understates the true budget by a factor of about two.
IBM_KINGSTON_TWO_QUBIT_ERROR: float = 1.956349128978227e-3


# ── Private Utility Methods ────────────────────────────────────────────────────

def _matplotlib():
    """
    Import pyplot, selecting a headless backend only where none is yet fixed.

    Returns
    -------
    module
        The `matplotlib.pyplot` module, with the thesis rcParams applied.
    """
    import sys

    import matplotlib

    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi":     140,
        "savefig.dpi":    300,
        "font.size":      9,
        "axes.grid":      True,
        "grid.alpha":     0.3,
        "legend.frameon": False,
        "axes.titlesize": 10,
    })
    return plt


def load_rows(repo_root: Path, dim: int, order: int) -> list[dict]:
    """
    Read one sweep's summary rows.

    Parameters
    ----------
    repo_root : Path
        Repository root, against which `SWEEP_DIR` is resolved.
    dim : int
        Spatial dimension, 1, 2 or 3.
    order : int
        Discretisation order, 2 or 4.

    Returns
    -------
    list of dict
        Summary rows, or an empty list where the sweep has written no summary.
        A sweep still running on the cluster is a routine state, not an error:
        the caller marks the series as pending and continues.
    """
    path = repo_root / SWEEP_DIR[(dim, order)] / "results_full.json"
    if not path.exists():
        log.warning("  %-30s absent", str(path))
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _series(rows: list[dict], case: str, solver: str) -> list[dict]:
    """
    Select one solver's rows for one case, ordered by resolution.

    Parameters
    ----------
    rows : list of dict
        Summary rows from one sweep.
    case : str
        Case identifier.
    solver : str
        Solver label as recorded, e.g. 'QSVT'.

    Returns
    -------
    list of dict
        Matching rows sorted by N.
    """
    return sorted(
        (r for r in rows
         if r.get("case") == case and str(r.get("solver")) == solver),
        key=lambda r: r.get("N") or 0,
    )


def _is_recovered(row: dict) -> bool:
    """
    Whether a row was reconstructed from its solution archive.

    A recovered row carries every accuracy metric that is a function of the
    stored field and none of the instrumentation that lived only in the killed
    process, and it records `max_rel_err` as a fraction where an instrumented row
    records per cent. Series drawn from that column must exclude it or the curve
    acquires a hundred-fold artificial dip at exactly the resolutions the
    recovery covered.

    Parameters
    ----------
    row : dict
        One summary row.

    Returns
    -------
    bool
        True where the row was reconstructed rather than instrumented.
    """
    return "recovered" in str(row.get("notes") or "")


def _pct(value: Optional[float], already_pct: bool) -> Optional[float]:
    """
    Normalise an error to per cent.

    Parameters
    ----------
    value : float or None
        Recorded error.
    already_pct : bool
        Whether the source column is already expressed in per cent.

    Returns
    -------
    float or None
        The error in per cent, or None where unrecorded or not a number.
    """
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v if already_pct else v * 100.0


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> Path:
    """
    Write one tidy data table for a figure.

    Parameters
    ----------
    path : Path
        Destination file.
    header : list of str
        Column names, each stating its unit where the quantity is dimensional.
    rows : list of list
        One row per plotted point.

    Returns
    -------
    Path
        The path written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _label_n_axis(ax, ticks: tuple[int, ...] = (4, 8, 16, 32, 64)) -> None:
    """
    Label a logarithmic resolution axis with the resolutions actually run.

    A logarithmic axis defaults to powers of ten, which for a sweep over
    N = 4…64 places no tick on any point that was measured.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    ticks : tuple of int
        Resolutions to label.
    """
    from matplotlib import ticker

    lo, hi = ax.get_xlim()
    shown = [n for n in ticks if lo <= n <= hi] or list(ticks)
    ax.set_xticks(shown)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())


def _reference_slope(ax, Ns: list[int], errs: list[float], p: int,
                     label: str) -> None:
    """
    Draw an O(N^-p) guide anchored to the coarsest plotted point.

    Anchored rather than fitted, so that the line indicates the slope a scheme of
    order p should achieve and is never mistaken for a fit to the data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    Ns : list of int
        Resolutions plotted.
    errs : list of float
        Errors plotted, used only to set the anchor.
    p : int
        Formal order of the scheme.
    label : str
        Legend entry.
    """
    if len(Ns) < 2 or not errs:
        return
    N0, e0 = Ns[0], max(errs)
    ax.loglog(Ns, [e0 * (N0 / n) ** p for n in Ns],
              "k--" if p == 2 else "k:", lw=1.0, alpha=0.7, label=label)


# ── Figure 1: Accuracy Against Resolution ──────────────────────────────────────

def figure_accuracy_vs_N(repo_root: Path, out_dir: Path,
                         dim: int = 1) -> list[Path]:
    """
    Total relative error against resolution, both discretisation orders.

    One panel per order, four curves per panel. Thomas carries the discretisation
    error alone and is therefore the floor every quantum solver is judged
    against; the O(h²) and O(h⁴) guides confirm that the stencil achieves its
    formal order on this case, which is the precondition for reading the quantum
    curves as algorithmic error.

    Parameters
    ----------
    repo_root : Path
        Repository root.
    out_dir : Path
        Destination for the CSV and the reference plot.
    dim : int
        Spatial dimension.

    Returns
    -------
    list of Path
        Files written.
    """
    plt = _matplotlib()
    case = PRIMARY_CASE[dim]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)

    csv_rows: list[list[Any]] = []
    for ax, order in zip(axes, (2, 4)):
        rows = load_rows(repo_root, dim, order)
        if not rows:
            ax.text(0.5, 0.5, f"order {order}: sweep pending",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color="grey")
            continue
        anchor_N: list[int] = []
        anchor_e: list[float] = []
        for solver in SOLVERS:
            recs = [r for r in _series(rows, case, solver)
                    if not _is_recovered(r)]
            pts = [(r["N"], _pct(r.get("max_rel_err"), already_pct=True))
                   for r in recs]
            pts = [(n, e) for n, e in pts if e is not None and e > 0.0]
            if not pts:
                continue
            ax.loglog(*zip(*pts), marker=SOLVER_MARKER[solver], lw=1.7,
                      color=SOLVER_COLOUR[solver], mfc="none", label=solver)
            if solver == "Thomas":
                anchor_N = [n for n, _ in pts]
                anchor_e = [e for _, e in pts]
            for (n, e), r in zip(pts, recs):
                csv_rows.append([case, order, solver, n, r.get("kappa")
                                 or r.get("kappa_row"), e,
                                 _pct(r.get("err_alg"), already_pct=False),
                                 _pct(r.get("err_disc"), already_pct=False)])
        _reference_slope(ax, anchor_N, anchor_e, order,
                         rf"$\mathcal{{O}}(h^{order})$")
        ax.set_title(f"Order-{order} stencil")
        ax.set_xlabel("$N$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        _label_n_axis(ax)

    axes[0].set_ylabel(r"total relative error $e_\infty$  [%]")
    fig.suptitle(f"Accuracy against resolution — {case}  ({dim}-D)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()

    written = _save(fig, out_dir, f"F1_accuracy_vs_N_{dim}D", plt)
    written.append(write_csv(
        out_dir / f"F1_accuracy_vs_N_{dim}D.csv",
        ["case", "order", "solver", "N", "kappa",
         "err_total_pct", "err_alg_pct", "err_disc_pct"],
        csv_rows,
    ))
    return written


# ── Figure 2: Error Decomposition ──────────────────────────────────────────────

def figure_error_decomposition(repo_root: Path, out_dir: Path,
                               dim: int = 1) -> list[Path]:
    """
    Algorithmic error against discretisation error, both orders.

    The crossing is the result. Left of it the total error of a quantum solve is
    set by the stencil and the algorithm is, for the purposes of solving the PDE,
    exact; right of it the algorithm dominates and refining the mesh no longer
    improves the answer. Raising the discretisation order moves the floor down
    and therefore moves the crossing to a coarser mesh, which is why the
    fourth-order panel is not merely a more accurate copy of the second-order
    one.

    Parameters
    ----------
    repo_root : Path
        Repository root.
    out_dir : Path
        Destination for the CSV and the reference plot.
    dim : int
        Spatial dimension.

    Returns
    -------
    list of Path
        Files written.
    """
    plt = _matplotlib()
    case = PRIMARY_CASE[dim]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)

    csv_rows: list[list[Any]] = []
    for ax, order in zip(axes, (2, 4)):
        rows = load_rows(repo_root, dim, order)
        if not rows:
            ax.text(0.5, 0.5, f"order {order}: sweep pending",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color="grey")
            continue

        disc = [(r["N"], _pct(r.get("err_disc"), already_pct=False))
                for r in _series(rows, case, "Thomas")]
        disc = [(n, e) for n, e in disc if e is not None and e > 0.0]
        if disc:
            ax.loglog(*zip(*disc), "k--s", lw=1.8, mfc="none",
                      label=r"$e_\mathrm{disc}$  (Thomas vs. exact)")
            for n, e in disc:
                csv_rows.append([case, order, "Thomas", n, None, e])

        for solver in QUANTUM_SOLVERS:
            recs = [r for r in _series(rows, case, solver)
                    if not _is_recovered(r)]
            pts = [(r["N"], _pct(r.get("err_alg"), already_pct=False))
                   for r in recs]
            pts = [(n, e) for n, e in pts if e is not None and e > 0.0]
            if not pts:
                continue
            ax.loglog(*zip(*pts), marker=SOLVER_MARKER[solver], lw=1.7,
                      color=SOLVER_COLOUR[solver], mfc="none",
                      label=rf"$e_\mathrm{{alg}}$  {solver}")
            for n, e in pts:
                csv_rows.append([case, order, solver, n, e, None])

        ax.set_title(f"Order-{order} stencil")
        ax.set_xlabel("$N$")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        _label_n_axis(ax)

    axes[0].set_ylabel("relative error  [%]")
    fig.suptitle(
        f"Algorithmic against discretisation error — {case}  ({dim}-D)",
        fontweight="bold", fontsize=10)
    fig.tight_layout()

    written = _save(fig, out_dir, f"F2_error_decomposition_{dim}D", plt)
    written.append(write_csv(
        out_dir / f"F2_error_decomposition_{dim}D.csv",
        ["case", "order", "solver", "N", "err_alg_pct", "err_disc_pct"],
        csv_rows,
    ))
    return written


# ── Figure 3: The QSVT Degree Threshold ────────────────────────────────────────

def figure_qsvt_degree_threshold(repo_root: Path, out_dir: Path) -> list[Path]:
    """
    QSVT algorithmic error against the degree-to-condition-number ratio.

    Collapses every recorded 1-D QSVT solve — seven cases, two discretisation
    orders, five resolutions, seventy solves — onto a single abscissa d/κ. The
    QSP polynomial approximates 1/x uniformly over the spectral interval
    [1/κ, 1], and the degree required for a given uniform error depends on the
    spectrum only through κ. If that account is right then d/κ, and not N, κ or
    the case separately, determines whether a QSVT solve succeeds.

    The ordinate is the relative residual rather than the error against Thomas,
    because the residual measures the one thing the polynomial is responsible
    for: whether p(A)b inverts A. An error measured against Thomas can be small
    for a reason that has nothing to do with the polynomial, and one case here is
    exactly that — the single-mode source `1D_Poisson_fS_hom` has a right-hand
    side that is an eigenvector of the operator, so the inversion is exact at any
    degree and the case reports machine precision at every ratio. It is drawn
    with a distinct marker rather than excluded, since a reader is entitled to
    see the one point that does not collapse and why.

    The practical consequence is the reason the figure is in the main body: a
    degree cap set without reference to κ silently converts a converged solver
    into a divergent one, and the ratio says where that happens.

    Parameters
    ----------
    repo_root : Path
        Repository root.
    out_dir : Path
        Destination for the CSV and the reference plot.

    Returns
    -------
    list of Path
        Files written.
    """
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    # Right-hand sides that are eigenvectors of the discrete operator. The
    # inversion is then a single scalar division, exact for any polynomial that
    # is accurate at that one eigenvalue, so these solves carry no information
    # about the uniform quality of the approximation over the spectrum.
    eigenvector_rhs = {"1D_Poisson_fS_hom"}

    csv_rows: list[list[Any]] = []
    for order, colour in ((2, SOLVER_COLOUR["QSVT"]), (4, "#7f2020")):
        rows = load_rows(repo_root, 1, order)
        generic_x, generic_y, eig_x, eig_y = [], [], [], []
        for r in rows:
            if str(r.get("solver")) != "QSVT":
                continue
            kappa, degree = r.get("kappa"), r.get("qsvt_degree")
            residual = r.get("residual")
            if not kappa or not degree or residual is None:
                continue
            ratio = degree / kappa
            value = max(float(residual), 1e-16)
            if r.get("case") in eigenvector_rhs:
                eig_x.append(ratio)
                eig_y.append(value)
            else:
                generic_x.append(ratio)
                generic_y.append(value)
            csv_rows.append([order, r.get("case"), r["N"], kappa, degree,
                             r.get("qsvt_max_degree"), ratio, float(residual),
                             _pct(r.get("err_alg"), already_pct=False),
                             r.get("case") in eigenvector_rhs])
        if generic_x:
            ax.loglog(generic_x, generic_y, "o" if order == 2 else "^",
                      ms=6.5, mfc="none", alpha=0.9, color=colour, ls="none",
                      label=f"order {order}")
        if eig_x:
            ax.loglog(eig_x, eig_y, "*", ms=9, alpha=0.9, color=colour,
                      ls="none",
                      label=f"order {order}, eigenvector RHS")

    ax.axvline(DEGREE_KAPPA_THRESHOLD, color="black", ls="--", lw=1.3,
               label=rf"$d/\kappa = {DEGREE_KAPPA_THRESHOLD:g}$")
    ax.axvspan(1e-2, DEGREE_KAPPA_THRESHOLD, color="grey", alpha=0.13)
    ax.annotate("under-resolved polynomial", xy=(0.04, 0.94),
                xycoords="axes fraction", fontsize=8.5, color="dimgrey")
    ax.set_xlabel(r"degree-to-condition-number ratio  $d / \kappa(A)$")
    ax.set_ylabel(r"relative residual  $\|Au - b\|_2 / \|b\|_2$")
    ax.set_title(r"QSVT accuracy is set by $d/\kappa$, not by $N$ or the case",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()

    written = _save(fig, out_dir, "F3_qsvt_degree_threshold", plt)
    written.append(write_csv(
        out_dir / "F3_qsvt_degree_threshold.csv",
        ["order", "case", "N", "kappa", "degree", "max_degree",
         "degree_over_kappa", "residual", "err_alg_pct", "eigenvector_rhs"],
        csv_rows,
    ))
    return written


# ── Figure 4: Condition-Number Scaling ─────────────────────────────────────────

def figure_kappa_scaling(repo_root: Path, out_dir: Path) -> list[Path]:
    """
    Condition number against resolution for the 1-D, 2-D and 3-D operators.

    The architectural claim of this work in one figure. The 1-D Poisson operator
    has κ = O(N²), which is what makes a direct quantum solve of the whole system
    intractable: every quantum linear-system algorithm carries κ in its query
    complexity, and QSVT carries it in the polynomial degree specifically.
    Decomposing the 2-D and 3-D problems into one-dimensional strips replaces
    that operator with a strip operator whose transverse coupling contributes a
    diagonal shift, bounding κ_row by 3 in two dimensions and by 2 in three,
    independently of N.

    A bounded κ is what makes the higher-dimensional solves cheaper per strip at
    N = 256 than the one-dimensional solve is at N = 64, and it is the reason
    the cost of the outer iteration, rather than the conditioning of the inner
    solve, is the binding constraint at scale.

    Parameters
    ----------
    repo_root : Path
        Repository root.
    out_dir : Path
        Destination for the CSV and the reference plot.

    Returns
    -------
    list of Path
        Files written.
    """
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    styles = {1: ("#1f77b4", "o"), 2: ("#2ca02c", "s"), 3: ("#d62728", "^")}
    csv_rows: list[list[Any]] = []
    for dim in (1, 2, 3):
        for order, ls in ((2, "-"), (4, "--")):
            rows = load_rows(repo_root, dim, order)
            case = PRIMARY_CASE[dim]
            pts = []
            for r in _series(rows, case, "Thomas"):
                kappa = r.get("kappa") if dim == 1 else r.get("kappa_row")
                if kappa:
                    pts.append((r["N"], float(kappa)))
                    csv_rows.append([dim, order, case, r["N"], float(kappa)])
            if not pts:
                continue
            colour, marker = styles[dim]
            ax.loglog(*zip(*pts), ls, marker=marker, lw=1.7, color=colour,
                      mfc="none",
                      label=f"{dim}-D, order {order}"
                            + (r"  ($\kappa$)" if dim == 1
                               else r"  ($\kappa_\mathrm{row}$)"))

    Ns = [4, 8, 16, 32, 64]
    ax.loglog(Ns, [0.6 * n ** 2 for n in Ns], "k:", lw=1.0,
              label=r"$\mathcal{O}(N^2)$")
    ax.axhline(3.0, color="grey", ls="-.", lw=1.0)
    ax.annotate(r"$\kappa_\mathrm{row} \to 3$ (2-D)", xy=(0.55, 0.16),
                xycoords="axes fraction", fontsize=8, color="dimgrey")
    ax.axhline(2.0, color="grey", ls="-.", lw=1.0)
    ax.annotate(r"$\kappa_\mathrm{row} \to 2$ (3-D)", xy=(0.55, 0.07),
                xycoords="axes fraction", fontsize=8, color="dimgrey")

    ax.set_xlabel("$N$")
    ax.set_ylabel(r"condition number")
    ax.set_title("Strip decomposition bounds the condition number", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()

    written = _save(fig, out_dir, "F4_kappa_scaling", plt)
    written.append(write_csv(
        out_dir / "F4_kappa_scaling.csv",
        ["dim", "order", "case", "N", "kappa"],
        csv_rows,
    ))
    return written


# ── Figure 5: Cost Against Resolution ──────────────────────────────────────────

def figure_cost_vs_N(repo_root: Path, out_dir: Path,
                     dim: int = 1) -> list[Path]:
    """
    Wall time against resolution, with terminated solves marked.

    A per-solve timeout is a measurement, not a missing datum: it states that the
    solve did not complete within the budget, which is precisely the quantity a
    feasibility argument needs. Those points are drawn as open markers with an
    upward arrow and are labelled in the legend, never dropped, since dropping
    them would make the surviving curve look better the more often the solver
    failed.

    Parameters
    ----------
    repo_root : Path
        Repository root.
    out_dir : Path
        Destination for the CSV and the reference plot.
    dim : int
        Spatial dimension.

    Returns
    -------
    list of Path
        Files written.
    """
    plt = _matplotlib()
    case = PRIMARY_CASE[dim]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)

    csv_rows: list[list[Any]] = []
    for ax, order in zip(axes, (2, 4)):
        rows = load_rows(repo_root, dim, order)
        if not rows:
            ax.text(0.5, 0.5, f"order {order}: sweep pending",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=11, color="grey")
            continue
        for solver in SOLVERS:
            done_n, done_t, cut_n, cut_t = [], [], [], []
            for r in _series(rows, case, solver):
                t = r.get("wall_time_s")
                if not t or t <= 0.0:
                    continue
                terminated = "timeout" in str(r.get("notes") or "")
                (cut_n if terminated else done_n).append(r["N"])
                (cut_t if terminated else done_t).append(float(t))
                csv_rows.append([case, order, solver, r["N"], float(t),
                                 terminated, r.get("notes") or ""])
            if done_n:
                ax.loglog(done_n, done_t, marker=SOLVER_MARKER[solver], lw=1.7,
                          color=SOLVER_COLOUR[solver], mfc="none", label=solver)
            if cut_n:
                ax.loglog(cut_n, cut_t, marker="^", ms=9, ls="none",
                          color=SOLVER_COLOUR[solver], mfc="none", mew=1.6,
                          label=f"{solver} (terminated at the cap)")
        ax.set_title(f"Order-{order} stencil")
        ax.set_xlabel("$N$")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3, which="both")
        _label_n_axis(ax)

    axes[0].set_ylabel("wall time  [s]")
    fig.suptitle(f"Computational cost — {case}  ({dim}-D)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()

    written = _save(fig, out_dir, f"F5_cost_vs_N_{dim}D", plt)
    written.append(write_csv(
        out_dir / f"F5_cost_vs_N_{dim}D.csv",
        ["case", "order", "solver", "N", "wall_time_s", "terminated", "notes"],
        csv_rows,
    ))
    return written


# ── Figure 6: Hardware Verification ────────────────────────────────────────────

def figure_hardware(repo_root: Path, out_dir: Path) -> list[Path]:
    """
    Measured circuit fidelity on ibm_kingston, against degree and against N.

    Left panel: block-encoding fidelity against QSVT polynomial degree, measured
    by direct fidelity estimation on the 2-D strip operator. The prediction being
    tested is that error composes multiplicatively across degree, F_d ≈ F₁^d,
    which is the premise every depth-based feasibility estimate in this work
    rests on. The measurement is bounded below by 2⁻ⁿ, the fidelity of the
    maximally mixed state against the target: a circuit whose output has
    decohered completely does not report zero fidelity but that floor, and a
    reading at the floor carries no information beyond "the circuit failed".

    Right panel: transpiled two-qubit gate count for the 1-D QSVT circuit against
    resolution, judged against the gate budget implied by the device's own
    calibration rather than by a round figure. With per-gate error ε₂ the
    probability that a circuit of n₂ two-qubit gates completes without a fault is
    (1 − ε₂)^n₂, so a target fault-free probability p admits n₂ ≤ ln p / ln(1−ε₂).

    Parameters
    ----------
    repo_root : Path
        Repository root.
    out_dir : Path
        Destination for the CSV and the reference plot.

    Returns
    -------
    list of Path
        Files written.
    """
    plt = _matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    inv = repo_root / "results" / "investigations"
    csv_rows: list[list[Any]] = []

    # -- Left: fidelity against degree -----------------------------------------
    files = sorted((inv / "degree_composition").glob("hardware_*.json"))
    n_qubits = None
    f1_by_level: dict[int, float] = {}
    for path in files:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        level = payload.get("resilience_level")
        pts = [(r["degree"], r["fidelity"], r["fidelity_std_err"],
                r["two_qubit_gates"], r["transpiled_depth"])
               for r in payload.get("rows", [])]
        if not pts:
            continue
        n_qubits = n_qubits or int(math.log2(payload.get("Nx", 8))) + 1
        degs = [p[0] for p in pts]
        fids = [p[1] for p in pts]
        errs = [p[2] for p in pts]
        # Several runs share a resilience level but cover disjoint degree
        # ranges; the range is stated so that two curves of the same colour
        # cannot be mistaken for a repeated measurement of the same points.
        axes[0].errorbar(degs, fids, yerr=errs, marker="o" if level else "s",
                         ms=5, lw=1.4, capsize=2, mfc="none",
                         label=f"measured, level {level}, "
                               f"$d \\in [{min(degs)}, {max(degs)}]$")
        for d, f, e, g, dep in pts:
            csv_rows.append(["fidelity_vs_degree", payload.get("backend"),
                             level, d, g, dep, f, e])
            if d == 1:
                f1_by_level[level] = f

    for level, f1 in sorted(f1_by_level.items()):
        degs = list(range(0, 64))
        axes[0].plot(degs, [f1 ** d for d in degs], ls="--", lw=1.1,
                     color="grey" if level else "black", alpha=0.8,
                     label=rf"$F_1^{{\,d}}$ prediction, level {level}")

    if n_qubits:
        floor = 2.0 ** (-n_qubits)
        axes[0].axhline(floor, color="crimson", ls=":", lw=1.3)
        axes[0].annotate(rf"maximally mixed floor, $2^{{-{n_qubits}}}$",
                         xy=(0.35, floor * 1.12), xycoords=("axes fraction",
                                                            "data"),
                         fontsize=7.5, color="crimson")

    axes[0].set_yscale("log")
    axes[0].set_xlim(-1.5, 65)
    axes[0].set_xlabel("QSVT polynomial degree $d$")
    axes[0].set_ylabel("measured state fidelity $F_d$")
    axes[0].set_title("(a)  Error composition on ibm_kingston")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, which="both")

    # -- Right: transpiled gate count against the calibrated budget ------------
    feas = inv / "hardware_feasibility_1d" / "results_full.json"
    budget = int(math.log(0.5) / math.log(1.0 - IBM_KINGSTON_TWO_QUBIT_ERROR))
    if feas.exists():
        with open(feas, encoding="utf-8") as fh:
            rows = json.load(fh)
        Ns = [r["N"] for r in rows]
        counts = [r["total_two_qubit_count"] for r in rows]
        axes[1].loglog(Ns, counts, "D-", lw=1.7, mfc="none",
                       color=SOLVER_COLOUR["QSVT"],
                       label="transpiled two-qubit count, 1-D QSVT")
        for r in rows:
            csv_rows.append(["two_qubit_vs_N", "transpiled", None, r["N"],
                             r["total_two_qubit_count"], r["degree"],
                             r["kappa"], None])
        axes[1].axhline(budget, color="black", ls="--", lw=1.3)
        axes[1].annotate(
            f"budget {budget} gates  "
            rf"($\varepsilon_2 = {IBM_KINGSTON_TWO_QUBIT_ERROR:.2e}$, $p = 0.5$)",
            xy=(0.03, budget * 1.2), xycoords=("axes fraction", "data"),
            fontsize=7.5)
    else:
        axes[1].text(0.5, 0.5, "transpiled feasibility sweep pending",
                     transform=axes[1].transAxes, ha="center", va="center",
                     fontsize=11, color="grey")

    axes[1].set_xlabel("$N$")
    axes[1].set_ylabel("two-qubit gate count")
    axes[1].set_title("(b)  Circuit size against the device budget")
    axes[1].legend(fontsize=7.5)
    axes[1].grid(alpha=0.3, which="both")

    fig.suptitle("Hardware verification, IBM Kingston (156-qubit Heron)",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()

    written = _save(fig, out_dir, "F6_hardware_verification", plt)
    written.append(write_csv(
        out_dir / "F6_hardware_verification.csv",
        ["panel", "backend", "resilience_level", "degree_or_N",
         "two_qubit_gates", "depth_or_degree", "fidelity_or_kappa",
         "fidelity_std_err"],
        csv_rows,
    ))
    return written


# ── Figure 7: Solution Fields ──────────────────────────────────────────────────

def export_fields(repo_root: Path, out_dir: Path, dim: int,
                  N: Optional[int] = None) -> list[Path]:
    """
    Export the solution field and its signed error for the HET case.

    Written as CSV rather than plotted here: a field figure is a presentation
    decision — colour map, slice plane, aspect ratio — that belongs with the
    typesetting, and `benchmark/hpc_plotting.py` already renders a diagnostic
    version of every one of them. What is exported is the array itself, on the
    physical coordinates, so the published figure is drawn from the archive and
    not from a screenshot of the diagnostic.

    In three dimensions the mid-plane slice normal to the azimuthal direction is
    exported, which is the plane the axial–radial physics of a Hall thruster
    channel lives in.

    Parameters
    ----------
    repo_root : Path
        Repository root.
    out_dir : Path
        Destination for the CSV files.
    dim : int
        Spatial dimension, 2 or 3.
    N : int, optional
        Resolution. Defaults to the finest for which every solver has an archive.

    Returns
    -------
    list of Path
        Files written, one per solver.

    Raises
    ------
    ValueError
        If `dim` is not 2 or 3; the one-dimensional profiles are exported by
        `export_profiles_1d`.
    """
    import numpy as np

    if dim not in (2, 3):
        raise ValueError(f"export_fields serves 2-D and 3-D only, got {dim}.")

    sweep = repo_root / SWEEP_DIR[(dim, 2)]
    case = HET_CASE[dim]
    prefix = "solutions" if dim == 2 else "solution3d"

    written: list[Path] = []
    candidates = sorted(
        {int(p.stem.rsplit("_N", 1)[1])
         for p in sweep.glob(f"{prefix}_{case}_*_N*.npz")},
        reverse=True,
    )
    if not candidates:
        log.warning("  no field archives for %s in %s", case, sweep)
        return written
    target = N or next(
        (n for n in candidates
         if all((sweep / f"{prefix}_{case}_{s}_N{n}.npz").exists()
                for s in SOLVERS)),
        candidates[0],
    )

    for solver in SOLVERS:
        path = sweep / f"{prefix}_{case}_{solver}_N{target}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=False)
        if dim == 2:
            phi = data["phi_solver"] if "phi_solver" in data else data["u_solver"]
            exact = data.get("phi_exact") if hasattr(data, "get") else None
            exact = data["phi_exact"] if "phi_exact" in data.files else (
                data["u_exact"] if "u_exact" in data.files else None)
            xx, yy = data["x"], data["y"]
        else:
            # Mid-plane slice normal to the azimuthal (third) axis.
            mid = data["phi"].shape[2] // 2
            phi = data["phi"][:, :, mid]
            exact = (data["phi_exact"][:, :, mid]
                     if "phi_exact" in data.files else None)
            xx, yy = data["x0"][:, :, mid], data["x1"][:, :, mid]

        rows: list[list[Any]] = []
        for i in range(phi.shape[0]):
            for j in range(phi.shape[1]):
                err = (float(phi[i, j] - exact[i, j])
                       if exact is not None else None)
                rows.append([float(xx[i, j]), float(yy[i, j]),
                             float(phi[i, j]),
                             float(exact[i, j]) if exact is not None else None,
                             err])
        written.append(write_csv(
            out_dir / f"F7_field_{dim}D_{case}_{solver}_N{target}.csv",
            ["x_m", "y_m", "phi_V", "phi_exact_V", "signed_error_V"],
            rows,
        ))
    return written


def export_profiles_1d(repo_root: Path, out_dir: Path,
                       N: int = 32) -> list[Path]:
    """
    Export the 1-D HET potential profile and the axial electric field.

    The electric field is obtained from the potential by central differences,
    E = −dφ/dz [V/m], one-sided at the two end nodes. It is the quantity the
    thruster physics is judged on: the potential itself is monotone and
    forgiving, whereas differentiating it amplifies exactly the high-wavenumber
    error a quantum solver introduces, so a solver that looks acceptable in φ can
    be plainly wrong in E.

    Parameters
    ----------
    repo_root : Path
        Repository root.
    out_dir : Path
        Destination for the CSV.
    N : int
        Resolution to export.

    Returns
    -------
    list of Path
        Files written.
    """
    import numpy as np

    sweep = repo_root / SWEEP_DIR[(1, 2)]
    case = HET_CASE[1]
    rows: list[list[Any]] = []
    for solver in SOLVERS:
        path = sweep / f"solutions_{case}_{solver}_N{N}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=False)
        z = data["x"]
        phi = data["u_solver"]
        E = -np.gradient(phi, z)
        exact = data["u_exact"] if "u_exact" in data.files else None
        for k in range(len(z)):
            rows.append([solver, float(z[k]), float(phi[k]), float(E[k]),
                         float(exact[k]) if exact is not None else None])

    if not rows:
        log.warning("  no 1-D HET profile archives at N=%d in %s", N, sweep)
        return []
    return [write_csv(
        out_dir / f"F8_het_1d_profile_N{N}.csv",
        ["solver", "z_m", "phi_V", "E_axial_V_per_m", "phi_exact_V"],
        rows,
    )]


# ── Table Data ─────────────────────────────────────────────────────────────────

def table_observed_order(repo_root: Path, out_dir: Path) -> list[Path]:
    """
    Observed order of accuracy of the stencil, per case, dimension and order.

    Computed from the Thomas rows alone, since the discretisation error is a
    property of the mesh and not of the solver. The observed order between two
    consecutive resolutions is

        p_obs = log(e_N / e_2N) / log(2)

    A scheme that does not attain its formal order has a defect in its boundary
    closure or in its treatment of the source, and reporting p_obs is how that is
    detected rather than assumed. Cases whose analytical solution the stencil
    represents exactly report an error at machine precision and no meaningful
    order; those are marked rather than given a spurious value.

    Parameters
    ----------
    repo_root : Path
        Repository root.
    out_dir : Path
        Destination for the CSV.

    Returns
    -------
    list of Path
        Files written.
    """
    csv_rows: list[list[Any]] = []
    for dim in (1, 2, 3):
        for order in (2, 4):
            rows = load_rows(repo_root, dim, order)
            cases = sorted({r.get("case") for r in rows if r.get("case")})
            for case in cases:
                recs = _series(rows, case, "Thomas")
                pts = []
                for r in recs:
                    e = (_pct(r.get("err_disc"), already_pct=False)
                         if dim == 1 else
                         _pct(r.get("err_thomas_vs_exact"), already_pct=True))
                    if e is not None and e > 0.0:
                        pts.append((r["N"], e))
                if len(pts) < 2:
                    continue
                exact_to_roundoff = max(e for _, e in pts) < 1e-10
                for (n1, e1), (n2, e2) in zip(pts, pts[1:]):
                    p_obs = (None if exact_to_roundoff else
                             math.log(e1 / e2) / math.log(n2 / n1))
                    csv_rows.append([dim, order, case, n1, n2, e1, e2, p_obs,
                                     "exact to round-off" if exact_to_roundoff
                                     else ""])
    return [write_csv(
        out_dir / "T3_observed_order.csv",
        ["dim", "order", "case", "N_coarse", "N_fine",
         "err_disc_coarse_pct", "err_disc_fine_pct", "p_observed", "note"],
        csv_rows,
    )]


def table_primary_condensed(repo_root: Path, out_dir: Path) -> list[Path]:
    """
    The condensed primary comparison carried in the main body.

    One case per dimension against the full solver set at every resolution, with
    the total error, the algorithmic error, the residual and the wall time. The
    complete tables over all twenty-seven cases belong in the appendix; this is
    what a reader needs in order to follow the argument.

    Parameters
    ----------
    repo_root : Path
        Repository root.
    out_dir : Path
        Destination for the CSV.

    Returns
    -------
    list of Path
        Files written.
    """
    csv_rows: list[list[Any]] = []
    for dim in (1, 2, 3):
        for order in (2, 4):
            rows = load_rows(repo_root, dim, order)
            for case_kind, case in (("poisson", PRIMARY_CASE[dim]),
                                    ("het", HET_CASE[dim])):
                for solver in SOLVERS:
                    for r in _series(rows, case, solver):
                        err = (_pct(r.get("max_rel_err"), already_pct=True)
                               if not _is_recovered(r) else
                               _pct(r.get("linf_err"), already_pct=True))
                        csv_rows.append([
                            dim, order, case_kind, case, solver, r["N"],
                            r.get("kappa") or r.get("kappa_row"),
                            err,
                            _pct(r.get("err_alg"), already_pct=(dim != 1)),
                            r.get("residual"),
                            r.get("wall_time_s"),
                            r.get("n_qubits"),
                            r.get("circuit_depth") or r.get("qsvt_depth"),
                            r.get("notes") or "",
                        ])
    return [write_csv(
        out_dir / "T2_primary_condensed.csv",
        ["dim", "order", "case_kind", "case", "solver", "N", "kappa",
         "err_total_pct", "err_alg_pct", "residual", "wall_time_s",
         "n_qubits", "circuit_depth", "notes"],
        csv_rows,
    )]


# ── Driver ─────────────────────────────────────────────────────────────────────

def _save(fig, out_dir: Path, stem: str, plt) -> list[Path]:
    """
    Write one reference plot as PNG and as vector PDF.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to write.
    out_dir : Path
        Destination directory.
    stem : str
        File stem, without a suffix.
    plt : module
        The pyplot module, closed against here rather than re-imported.

    Returns
    -------
    list of Path
        The two files written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".png", ".pdf"):
        path = (out_dir / stem).with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight")
        written.append(path)
    plt.close(fig)
    return written


def build_all(repo_root: Path, out_dir: Path) -> list[Path]:
    """
    Assemble every main-body figure and table dataset.

    Parameters
    ----------
    repo_root : Path
        Repository root.
    out_dir : Path
        Destination directory for the CSV files and reference plots.

    Returns
    -------
    list of Path
        Every file written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    written += figure_accuracy_vs_N(repo_root, out_dir, dim=1)
    written += figure_error_decomposition(repo_root, out_dir, dim=1)
    written += figure_qsvt_degree_threshold(repo_root, out_dir)
    written += figure_kappa_scaling(repo_root, out_dir)
    written += figure_cost_vs_N(repo_root, out_dir, dim=1)
    written += figure_hardware(repo_root, out_dir)
    written += export_fields(repo_root, out_dir, dim=2)
    written += export_fields(repo_root, out_dir, dim=3)
    written += export_profiles_1d(repo_root, out_dir, N=32)
    written += table_observed_order(repo_root, out_dir)
    written += table_primary_condensed(repo_root, out_dir)

    return written
