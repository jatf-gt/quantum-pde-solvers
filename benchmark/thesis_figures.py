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

# Solvers drawn in the one-dimensional thruster profile, F8. VQLS is excluded
# there and only there. At N = 32 its solution has already collapsed, so it
# contributes a flat line near zero in both panels — a reader spends longer
# working out why a curve is horizontal than the omission would have cost, and
# the finding it would carry is made far better by F9, which shows the same
# collapse developing across four resolutions. The exported CSV retains all four
# solvers; this governs the rendering only.
PROFILE_SOLVERS_1D: tuple[str, ...] = ("Thomas", "HHL", "QSVT")
QUANTUM_SOLVERS: tuple[str, ...] = ("HHL", "VQLS", "QSVT")

# Okabe--Ito, the qualitative palette designed to stay separable under the three
# common forms of colour blindness. It replaces matplotlib's default cycle, whose
# green (#2ca02c) and red (#d62728) are the pair deuteranopes cannot separate --
# and those were VQLS and QSVT, the two curves every accuracy figure asks the
# reader to tell apart. The hue families are unchanged, so the text's "the red
# QSVT curve" still reads correctly. Marker shape carries the same distinction
# independently, so the figures also survive being printed in greyscale.
SOLVER_COLOUR: dict[str, str] = {
    "Thomas": "#000000",   # black
    "HHL":    "#0072B2",   # blue
    "VQLS":   "#009E73",   # bluish green
    "QSVT":   "#D55E00",   # vermillion
}

# Second series colour where a figure separates by discretisation order rather
# than by solver. Reddish purple against vermillion is separable for every form
# of colour blindness the palette covers, which two shades of red would not be.
ORDER_COLOUR: dict[int, str] = {2: SOLVER_COLOUR["QSVT"], 4: "#CC79A7"}

# The full palette, installed as matplotlib's default property cycle so that a
# series drawn without an explicit colour -- the per-resilience-level hardware
# measurements, for one -- comes out of the same system as the named ones rather
# than out of the tab10 default the named colours were chosen to replace.
OKABE_ITO: tuple[str, ...] = (
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#000000",
)

# Condition-number series are separated by spatial dimension rather than by
# solver, so they take their own three hues from the same palette.
DIMENSION_COLOUR: dict[int, str] = {1: "#0072B2", 2: "#009E73", 3: "#D55E00"}

# Sequential map for a scalar field, diverging map for a signed error. Both are
# deliberate: viridis is perceptually uniform, so equal steps in potential are
# equal steps in apparent brightness and it survives greyscale conversion; RdBu_r
# is symmetric about its midpoint, so zero error is white and the sign of the
# departure is read from the hue rather than inferred from a key.
# Reference lines that mark a physical limit rather than a series: the
# maximally mixed floor in the hardware figure. Reddish purple sits outside
# the solver hues, so it cannot be mistaken for a measurement.
FLOOR_COLOUR: str = "#CC79A7"

FIELD_CMAP: str = "viridis"
SIGNED_ERROR_CMAP: str = "RdBu_r"
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

# ── Typographic Calibration ────────────────────────────────────────────────────
#
# Every figure is rendered at the exact width of the dissertation's text block
# and placed with `\includegraphics[width=\textwidth]`, so LaTeX applies a scale
# factor of unity and a point in the figure is a point on the page. Rendering
# wider and letting LaTeX shrink — the arrangement these figures previously used,
# 10.5 in reduced to 0.92 \textwidth — multiplies every glyph by 0.55, which put
# 9 pt axis text on the page at 5 pt, half the size of the caption beneath it.
#
# Geometry, from `usepackages.tex`: A4 with 2.5 cm margins on all four sides, so
# the text block is 16.0 cm wide and 24.7 cm tall.
TEXT_WIDTH_IN: float = 6.2992      # 16.0 cm, to the fourth decimal
TEXT_HEIGHT_IN: float = 9.7244     # 24.7 cm

# The document sets Helvetica (`helvet`) at `scaled=0.95` as the default family,
# the departmental Arial requirement; Arial and Helvetica share metrics, so Arial
# is the exact match on Windows and Liberation Sans on Linux. Ordered by
# preference and resolved once against the installed set, because naming a face
# matplotlib cannot find silently substitutes DejaVu Sans and the figure ships in
# a different typeface from the body text.
BODY_FONT_STACK: tuple[str, ...] = (
    "Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans",
)

# Point sizes, all measured on the page because the scale factor is unity.
#
# The body and caption are set at 11 pt scaled by 0.95, so caption glyphs are
# 10.45 pt. Axis labels and tick labels are therefore set at or above that: the
# reader must not have to magnify the page to read an axis. Legends, in-panel
# annotations and small-multiple titles are subordinate labelling and are allowed
# below it, but not far below — 8 pt is the floor adopted here.
AXIS_PT:   float = 11.0     # axis labels, panel titles
TICK_PT:   float = 10.5     # tick labels; equals the caption glyph size exactly
LEGEND_PT: float =  9.0     # legends
ANNOT_PT:  float =  8.5     # in-panel annotations and small-multiple titles
SMALL_PT:  float =  8.0     # densest small-multiple labelling only


def _body_face() -> str:
    """
    Resolve the document's text face against the installed fonts.

    Returns
    -------
    str
        The first family in `BODY_FONT_STACK` matplotlib can actually load,
        falling back on DejaVu Sans, which ships with matplotlib and is always
        present.
    """
    from matplotlib import font_manager as fm

    installed = {f.name for f in fm.fontManager.ttflist}
    for name in BODY_FONT_STACK:
        if name in installed:
            return name
    return "DejaVu Sans"


# Whether a figure carries its own headline. A reference plot read on its own
# needs one; the same plot set beside a LaTeX caption repeats it, which reads as
# a duplicated title in the typeset document. Panel labels ("Order-2 stencil",
# "(a)", "(b)") are not affected — they identify a panel rather than restate the
# figure, and the caption refers to them.
DRAW_TITLES: bool = True


def set_draw_titles(draw: bool) -> None:
    """
    Enable or suppress the figure-level headline on every rendered figure.

    Parameters
    ----------
    draw : bool
        False for figures destined for a LaTeX float, whose caption states what
        the headline would.
    """
    global DRAW_TITLES
    DRAW_TITLES = bool(draw)


def _headline(target, text: str, **kwargs) -> None:
    """
    Apply a figure- or axes-level headline unless headlines are suppressed.

    Parameters
    ----------
    target : matplotlib.figure.Figure or matplotlib.axes.Axes
        Receiver of the headline; a Figure takes `suptitle`, an Axes `set_title`.
    text : str
        The headline.
    **kwargs
        Forwarded to the underlying matplotlib call.
    """
    if not DRAW_TITLES:
        return
    if hasattr(target, "suptitle"):
        target.suptitle(text, **kwargs)
    else:
        target.set_title(text, **kwargs)


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

    face = _body_face()
    plt.rcParams.update({
        "figure.dpi":     140,
        "savefig.dpi":    400,

        # Match the document's text face, and set mathtext in the same family so
        # that "$e_\infty$" beside "algorithmic error" is one typeface rather
        # than two. `mathtext.fontset = "custom"` is what allows the four math
        # styles to be pointed at a named family at all.
        "font.family":       "sans-serif",
        "font.sans-serif":   list(BODY_FONT_STACK),
        "mathtext.fontset":  "custom",
        "mathtext.rm":       face,
        "mathtext.it":       f"{face}:italic",
        "mathtext.bf":       f"{face}:bold",
        "mathtext.sf":       face,
        "mathtext.default":  "it",

        "font.size":        AXIS_PT,
        "axes.labelsize":   AXIS_PT,
        "axes.titlesize":   AXIS_PT,
        "figure.titlesize": AXIS_PT,
        "xtick.labelsize":  TICK_PT,
        "ytick.labelsize":  TICK_PT,
        "legend.fontsize":  LEGEND_PT,

        "axes.prop_cycle":  __import__("cycler").cycler(color=list(OKABE_ITO)),

        "axes.grid":        True,
        "grid.alpha":       0.3,
        "grid.linewidth":   0.6,
        "legend.frameon":   False,
        "legend.handlelength":  1.9,
        "legend.borderaxespad": 0.4,
        "legend.labelspacing":  0.35,
        "axes.linewidth":   0.8,
        "lines.linewidth":  1.7,
        "lines.markersize": 5.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # Constrained layout packs the axes inside a canvas of fixed size, where
        # `tight_layout` resizes the canvas to suit the axes. Only the former
        # preserves the exact figure width the scale factor of unity depends on.
        "figure.constrained_layout.use":    True,
        "figure.constrained_layout.h_pad":  0.02,
        "figure.constrained_layout.w_pad":  0.02,
        "figure.constrained_layout.hspace": 0.03,
        "figure.constrained_layout.wspace": 0.03,

        # Embed TrueType outlines rather than Type 3 bitmapped subsets: Type 3 is
        # what makes text in a submitted PDF unsearchable and renders poorly at
        # print resolution.
        "pdf.fonttype": 42,
        "ps.fonttype":  42,
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


def _qsvt_construction(row: dict) -> str:
    """
    Which QSP polynomial a recorded QSVT row was built from.

    Two constructions are in use and they are not interchangeable. An *uncapped*
    solve calls `pyqsp`'s 1/x generator, which targets a prescribed uniform error
    ε and returns whatever degree that demands. A *capped* solve fits the
    truncated Chebyshev expansion of 1/x directly at the degree given, padding up
    to the cap where the natural degree is lower. At equal degree the capped
    construction is the more accurate by seven to nine orders of magnitude on any
    right-hand side with broadband spectral content, so a curve that mixes the
    two is not a curve in one variable and must not be drawn as one.

    Which construction a row used is not recorded directly; it follows from
    whether a cap was set, since the cap is what selects the Chebyshev path.

    Parameters
    ----------
    row : dict
        One summary row.

    Returns
    -------
    str
        'capped' or 'uncapped'.
    """
    return "capped" if row.get("qsvt_max_degree") is not None else "uncapped"


def _degree_over_kappa(row: dict) -> Optional[float]:
    """
    Ratio of QSP polynomial degree to operator condition number for one row.

    Parameters
    ----------
    row : dict
        One summary row.

    Returns
    -------
    float or None
        d/κ, or None where either quantity is unrecorded.
    """
    d = row.get("qsvt_degree")
    k = row.get("kappa") or row.get("kappa_row")
    if d is None or not k:
        return None
    try:
        return float(d) / float(k)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


# Drawing style for the two QSP constructions, shared by F1 and F2 so the two
# figures cannot come to disagree about which series is which.
QSVT_CONSTRUCTION_STYLE: dict[str, dict] = {
    "capped":   {"linestyle": "-",  "mfc": "none",
                 "label": "QSVT (capped)"},
    "uncapped": {"linestyle": ":",  "mfc": SOLVER_COLOUR["QSVT"],
                 "label": "QSVT (uncapped)"},
}


def _plot_qsvt_by_construction(ax, recs: list[dict], value_of,
                               annotate: bool = True) -> list[tuple]:
    """
    Draw the QSVT series as one line per QSP construction.

    Parameters
    ----------
    ax : matplotlib axis
        Target axis.
    recs : list of dict
        QSVT rows for one case and order, ordered by resolution.
    value_of : callable
        Maps a row to the ordinate in per cent, or to None to drop it.
    annotate : bool
        Whether to label each point with its d/κ ratio. The ratio is what
        governs whether the polynomial inverts the operator at all, and it is
        not recoverable from the abscissa.

    Returns
    -------
    list of tuple
        (construction, N, value, degree, d_over_kappa) per drawn point.
    """
    drawn: list[tuple] = []
    first = True
    for construction, style in QSVT_CONSTRUCTION_STYLE.items():
        sub = [r for r in recs if _qsvt_construction(r) == construction]
        pts = [(r["N"], value_of(r)) for r in sub]
        keep = [(n, e) for n, e in pts if e is not None and e > 0.0]
        if not keep:
            continue
        ax.loglog(*zip(*keep), marker=SOLVER_MARKER["QSVT"], lw=1.7,
                  color=SOLVER_COLOUR["QSVT"],
                  linestyle=style["linestyle"], mfc=style["mfc"],
                  label=style["label"])
        for (n, e), r in zip(keep, sub):
            ratio = _degree_over_kappa(r)
            drawn.append((construction, n, e, r.get("qsvt_degree"), ratio))
            if annotate and ratio is not None:
                # Offset away from the line on the side the construction sits,
                # and set on an opaque patch: these labels cross the Thomas
                # curve at several resolutions.
                dy = 9 if construction == "uncapped" else -13
                # The first label on the panel names the quantity, the rest
                # give the value alone. A column of bare integers beside a curve
                # is unreadable without the caption; repeating the symbol on
                # every point would bury the curve it annotates.
                text = (rf"$d/\kappa$ = {ratio:.0f}" if first
                        else f"{ratio:.0f}")
                first = False
                ax.annotate(
                    text, (n, e), textcoords="offset points",
                    xytext=(4, dy), fontsize=SMALL_PT,
                    color=SOLVER_COLOUR["QSVT"],
                    bbox=dict(boxstyle="round,pad=0.12", fc="white",
                              ec="none", alpha=0.75))
    return drawn


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


def _sci(value: float, sig: int = 2) -> str:
    """
    Render a number as a mathtext power of ten.

    `f"{x:.2e}"` produces "1.96e-03", which typesets as an italic *e* beside a
    minus sign and reads as a variable rather than as an exponent. This returns
    the form a reader of a thesis expects.

    Parameters
    ----------
    value : float
        Quantity to render.
    sig : int
        Significant figures in the mantissa.

    Returns
    -------
    str
        A mathtext fragment, without the enclosing dollar signs.
    """
    mantissa, exponent = f"{value:.{sig}e}".split("e")
    return rf"{mantissa} \times 10^{{{int(exponent)}}}"


def _pct_label(value, sig: int = 2) -> str:
    """
    Render a per-cent value as mathtext, switching to a power of ten when the
    decimal form would be unreadable.

    QSVT's algorithmic error reaches 10 to the minus twelve, where `"%.2g"`
    yields "1.1e-12" — an italic *e* against a minus sign, which reads as a
    variable. Values within three decades of unity keep the plain decimal form,
    which is shorter and needs no interpretation.

    Parameters
    ----------
    value : float or None
        Error in per cent.
    sig : int
        Significant figures.

    Returns
    -------
    str
        A mathtext fragment including the per-cent sign, or "n/a".
    """
    if value is None:
        return "n/a"
    plain = f"{value:.{sig}g}"
    if "e" not in plain:
        return f"{plain}%"
    return rf"${_sci(float(value), sig - 1)}$%"


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
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.65), sharey=True)

    csv_rows: list[list[Any]] = []
    for ax, order in zip(axes, (2, 4)):
        rows = load_rows(repo_root, dim, order)
        if not rows:
            ax.text(0.5, 0.5, f"order {order}: sweep pending",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=AXIS_PT, color="grey")
            continue
        anchor_N: list[int] = []
        anchor_e: list[float] = []
        for solver in SOLVERS:
            recs = [r for r in _series(rows, case, solver)
                    if not _is_recovered(r)]
            if solver == "QSVT":
                # Drawn as one line per QSP construction rather than one line
                # per solver: see `_qsvt_construction`. The two differ by seven
                # to nine orders at equal degree, so a single curve through both
                # reports a change of algorithm as though it were a change of
                # resolution.
                for construction, n, e, degree, ratio in (
                        _plot_qsvt_by_construction(
                            ax, recs,
                            lambda r: _pct(r.get("max_rel_err"),
                                           already_pct=True))):
                    src = next(r for r in recs if r["N"] == n)
                    csv_rows.append([
                        case, order, f"QSVT ({construction})", n,
                        src.get("kappa") or src.get("kappa_row"), e,
                        _pct(src.get("err_alg"), already_pct=False),
                        _pct(src.get("err_disc"), already_pct=False),
                        degree, ratio])
                continue
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
                                 _pct(r.get("err_disc"), already_pct=False),
                                 None, None])
        _reference_slope(ax, anchor_N, anchor_e, order,
                         rf"$\mathcal{{O}}(h^{order})$")
        ax.set_title(f"Order-{order} stencil")
        ax.set_xlabel("$N$")
        ax.legend(fontsize=SMALL_PT)
        ax.grid(alpha=0.3, which="both")
        # Open the data limits before the ticks are chosen. The QSVT degree
        # annotation sits a few points to the right of its marker, so the point
        # at the largest N writes past the right spine otherwise, and the
        # terminated-solve markers at N = 64 are clipped by the same edge.
        # Applied before `_label_n_axis`, which reads the limits to decide which
        # resolutions to tick.
        ax.margins(x=0.13, y=0.09)
        _label_n_axis(ax)

    axes[0].set_ylabel(r"total relative error $e_\infty$  [%]")
    _headline(fig, f"Accuracy against resolution — {case}  ({dim}-D)",
                 fontweight="bold", fontsize=10)

    written = _save(fig, out_dir, f"F1_accuracy_vs_N_{dim}D", plt)
    written.append(write_csv(
        out_dir / f"F1_accuracy_vs_N_{dim}D.csv",
        ["case", "order", "solver", "N", "kappa",
         "err_total_pct", "err_alg_pct", "err_disc_pct",
         "qsvt_degree", "d_over_kappa"],
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
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.65), sharey=True)

    csv_rows: list[list[Any]] = []
    for ax, order in zip(axes, (2, 4)):
        rows = load_rows(repo_root, dim, order)
        if not rows:
            ax.text(0.5, 0.5, f"order {order}: sweep pending",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=AXIS_PT, color="grey")
            continue

        disc = [(r["N"], _pct(r.get("err_disc"), already_pct=False))
                for r in _series(rows, case, "Thomas")]
        disc = [(n, e) for n, e in disc if e is not None and e > 0.0]
        if disc:
            ax.loglog(*zip(*disc), "k--s", lw=1.8, mfc="none",
                      label=r"$e_\mathrm{disc}$  (Thomas vs. exact)")
            for n, e in disc:
                csv_rows.append([case, order, "Thomas", n, None, e,
                                 None, None])

        for solver in QUANTUM_SOLVERS:
            recs = [r for r in _series(rows, case, solver)
                    if not _is_recovered(r)]
            if solver == "QSVT":
                for construction, n, e, degree, ratio in (
                        _plot_qsvt_by_construction(
                            ax, recs,
                            lambda r: _pct(r.get("err_alg"),
                                           already_pct=False))):
                    csv_rows.append([case, order, f"QSVT ({construction})",
                                     n, e, None, degree, ratio])
                continue
            pts = [(r["N"], _pct(r.get("err_alg"), already_pct=False))
                   for r in recs]
            pts = [(n, e) for n, e in pts if e is not None and e > 0.0]
            if not pts:
                continue
            ax.loglog(*zip(*pts), marker=SOLVER_MARKER[solver], lw=1.7,
                      color=SOLVER_COLOUR[solver], mfc="none",
                      label=rf"$e_\mathrm{{alg}}$  {solver}")
            for n, e in pts:
                csv_rows.append([case, order, solver, n, e, None, None, None])

        ax.set_title(f"Order-{order} stencil")
        ax.set_xlabel("$N$")
        ax.legend(fontsize=SMALL_PT)
        ax.grid(alpha=0.3, which="both")
        # Open the data limits before the ticks are chosen. The QSVT degree
        # annotation sits a few points to the right of its marker, so the point
        # at the largest N writes past the right spine otherwise, and the
        # terminated-solve markers at N = 64 are clipped by the same edge.
        # Applied before `_label_n_axis`, which reads the limits to decide which
        # resolutions to tick.
        ax.margins(x=0.13, y=0.09)
        _label_n_axis(ax)

    axes[0].set_ylabel("relative error  [%]")
    _headline(
        fig,
        f"Algorithmic against discretisation error — {case}  ({dim}-D)",
        fontweight="bold", fontsize=10)

    written = _save(fig, out_dir, f"F2_error_decomposition_{dim}D", plt)
    written.append(write_csv(
        out_dir / f"F2_error_decomposition_{dim}D.csv",
        ["case", "order", "solver", "N", "err_alg_pct", "err_disc_pct",
         "qsvt_degree", "d_over_kappa"],
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

    Capped and uncapped solves are drawn distinctly, because they are not the
    same algorithm. `max_degree` selects the construction rather than bounding
    it: uncapped calls `pyqsp.PolyOneOverX.generate`, which targets a prescribed
    uniform error, whereas a cap fits the truncated Chebyshev expansion of 1/x
    directly at that degree. The fitted polynomial is the more accurate of the
    two at equal degree, by nine to ten orders of magnitude in the measurements
    of `benchmark/sensitivity.py`. Superimposing the two without marking them
    would show a collapse that fails above d/κ ≈ 26 for a reason the abscissa
    does not carry.

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
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.30))

    # Right-hand sides that are eigenvectors of the discrete operator. The
    # inversion is then a single scalar division, exact for any polynomial that
    # is accurate at that one eigenvalue, so these solves carry no information
    # about the uniform quality of the approximation over the spectrum.
    eigenvector_rhs = {"1D_Poisson_fS_hom"}

    csv_rows: list[list[Any]] = []
    for order, colour in sorted(ORDER_COLOUR.items()):
        rows = load_rows(repo_root, 1, order)
        capped: list[tuple[float, float]] = []
        uncapped: list[tuple[float, float]] = []
        eig: list[tuple[float, float]] = []
        for r in rows:
            if str(r.get("solver")) != "QSVT":
                continue
            kappa, degree = r.get("kappa"), r.get("qsvt_degree")
            residual = r.get("residual")
            if not kappa or not degree or residual is None:
                continue
            ratio = degree / kappa
            value = max(float(residual), 1e-16)
            cap = r.get("qsvt_max_degree")
            if r.get("case") in eigenvector_rhs:
                eig.append((ratio, value))
            elif cap:
                capped.append((ratio, value))
            else:
                uncapped.append((ratio, value))
            csv_rows.append([order, r.get("case"), r["N"], kappa, degree,
                             cap, ratio, float(residual),
                             _pct(r.get("err_alg"), already_pct=False),
                             r.get("case") in eigenvector_rhs])
        marker = "o" if order == 2 else "^"
        if capped:
            ax.loglog(*zip(*capped), marker, ms=6.5, mfc="none", alpha=0.9,
                      color=colour, ls="none",
                      label=f"order {order}, capped (Chebyshev fit)")
        if uncapped:
            ax.loglog(*zip(*uncapped), marker, ms=6.5, mfc=colour, alpha=0.55,
                      mec=colour, color=colour, ls="none",
                      label=f"order {order}, uncapped (PolyOneOverX)")
        if eig:
            ax.loglog(*zip(*eig), "*", ms=9, alpha=0.9, color=colour,
                      ls="none",
                      label=f"order {order}, eigenvector RHS")

    ax.axvline(DEGREE_KAPPA_THRESHOLD, color="black", ls="--", lw=1.3,
               label=rf"$d/\kappa = {DEGREE_KAPPA_THRESHOLD:g}$")
    ax.axvspan(1e-2, DEGREE_KAPPA_THRESHOLD, color="grey", alpha=0.13)
    ax.annotate("under-resolved polynomial", xy=(0.04, 0.94),
                xycoords="axes fraction", fontsize=ANNOT_PT, color="dimgrey")
    ax.set_xlabel(r"degree-to-condition-number ratio  $d / \kappa(A)$")
    ax.set_ylabel(r"relative residual  $\|Au - b\|_2 / \|b\|_2$")
    _headline(ax, r"QSVT accuracy is set by $d/\kappa$, not by $N$ or the case",
                 fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=SMALL_PT, loc="lower left", ncol=2)

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
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 4.05))

    styles = {dim: (DIMENSION_COLOUR[dim], marker)
              for dim, marker in ((1, "o"), (2, "s"), (3, "^"))}
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
                xycoords="axes fraction", fontsize=ANNOT_PT, color="dimgrey")
    ax.axhline(2.0, color="grey", ls="-.", lw=1.0)
    ax.annotate(r"$\kappa_\mathrm{row} \to 2$ (3-D)", xy=(0.55, 0.07),
                xycoords="axes fraction", fontsize=ANNOT_PT, color="dimgrey")

    ax.set_xlabel("$N$")
    ax.set_ylabel(r"condition number")
    _headline(ax, "Strip decomposition bounds the condition number", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=SMALL_PT, ncol=2)
    # The 2-D strip operator is measured out to N = 256, so this axis carries two
    # resolutions beyond the 1-D sweep's range; the default decade ticks label
    # none of the six actually run.
    ax.margins(x=0.06)
    _label_n_axis(ax, ticks=(4, 8, 16, 32, 64, 128, 256))

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
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.65), sharey=True)

    csv_rows: list[list[Any]] = []
    for ax, order in zip(axes, (2, 4)):
        rows = load_rows(repo_root, dim, order)
        if not rows:
            ax.text(0.5, 0.5, f"order {order}: sweep pending",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=AXIS_PT, color="grey")
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
        ax.legend(fontsize=SMALL_PT)
        ax.grid(alpha=0.3, which="both")
        # Open the data limits before the ticks are chosen. The QSVT degree
        # annotation sits a few points to the right of its marker, so the point
        # at the largest N writes past the right spine otherwise, and the
        # terminated-solve markers at N = 64 are clipped by the same edge.
        # Applied before `_label_n_axis`, which reads the limits to decide which
        # resolutions to tick.
        ax.margins(x=0.13, y=0.09)
        _label_n_axis(ax)

    axes[0].set_ylabel("wall time  [s]")
    _headline(fig, f"Computational cost — {case}  ({dim}-D)",
                 fontweight="bold", fontsize=10)

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
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.85))

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
        axes[0].axhline(floor, color=FLOOR_COLOUR, ls=":", lw=1.3)
        axes[0].annotate(rf"maximally mixed floor, $2^{{-{n_qubits}}}$",
                         xy=(0.32, floor * 1.35), xycoords=("axes fraction",
                                                            "data"),
                         fontsize=ANNOT_PT, color=FLOOR_COLOUR)

    axes[0].set_yscale("log")
    axes[0].set_xlim(-1.5, 65)
    axes[0].set_xlabel("QSVT polynomial degree $d$")
    axes[0].set_ylabel("measured state fidelity $F_d$")
    axes[0].set_title("(a)  Error composition")
    axes[0].legend(fontsize=SMALL_PT)
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
            f"budget {budget} gates   "
            rf"($\varepsilon_2 = {_sci(IBM_KINGSTON_TWO_QUBIT_ERROR)}$, "
            rf"$p = 0.5$)",
            xy=(0.03, budget * 1.2), xycoords=("axes fraction", "data"),
            fontsize=SMALL_PT)
    else:
        axes[1].text(0.5, 0.5, "transpiled feasibility sweep pending",
                     transform=axes[1].transAxes, ha="center", va="center",
                     fontsize=AXIS_PT, color="grey")

    axes[1].set_xlabel("$N$")
    axes[1].set_ylabel("two-qubit gate count")
    # Without this the logarithmic axis labels the decade minor ticks, which over
    # N = 4..32 is five overlapping "4 x 10^0"-style strings and no tick on any
    # resolution that was actually run.
    _label_n_axis(axes[1], ticks=(4, 8, 16, 32))
    axes[1].set_title("(b)  Circuit size vs. budget")
    axes[1].legend(fontsize=SMALL_PT)
    axes[1].grid(alpha=0.3, which="both")

    _headline(fig, "Hardware verification, IBM Kingston (156-qubit Heron)",
                 fontweight="bold", fontsize=10)

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
    E = −dφ/dz, one-sided at the two end nodes. It is the quantity the thruster
    physics is judged on: the potential itself is monotone and forgiving,
    whereas differentiating it amplifies exactly the high-wavenumber error a
    quantum solver introduces, so a solver that looks acceptable in φ can be
    plainly wrong in E.

    Units. The axial coordinate recorded in the archive is non-dimensional,
    ξ = z/L_z on (0, 1); it is converted here to millimetres against the channel
    length L_z = 40 mm of `core.het_geometry`, which is unambiguous. The
    potential is **not** converted. Sub-case 3b assembles b = h²f with the
    source f in physical [V/m²] but h non-dimensional, and absorbs the anode
    constraint as b[0] −= V_d with V_d in volts, so the recorded φ is not in
    volts under either reading and the two boundary treatments are not on one
    scale. The columns are therefore named for what they hold, and
    `figure_het_profile_1d` normalises every series against the classical
    solution rather than asserting a unit. Column `phi_V` of revisions before
    2026-08-20 carried these same numbers under a volt label, and the axial
    column carried ξ under a metre label; neither was ever quoted in the text.

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

    from core import het_geometry as geom

    sweep = repo_root / SWEEP_DIR[(1, 2)]
    case = HET_CASE[1]
    rows: list[list[Any]] = []
    for solver in SOLVERS:
        path = sweep / f"solutions_{case}_{solver}_N{N}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=False)
        xi = data["x"]                       # non-dimensional, z/L_z on (0, 1)
        z_mm = xi * geom.L_Z * 1.0e3
        phi = data["u_solver"]
        E = -np.gradient(phi, z_mm)
        exact = data["u_exact"] if "u_exact" in data.files else None
        for k in range(len(xi)):
            rows.append([solver, float(xi[k]), float(z_mm[k]), float(phi[k]),
                         float(E[k]),
                         float(exact[k]) if exact is not None else None])

    if not rows:
        log.warning("  no 1-D HET profile archives at N=%d in %s", N, sweep)
        return []
    return [write_csv(
        out_dir / f"F8_het_1d_profile_N{N}.csv",
        ["solver", "xi", "z_mm", "phi_code", "E_axial_code_per_mm",
         "phi_exact_code"],
        rows,
    )]


# ── Figures 7 and 8: The Thruster Solution ─────────────────────────────────────

def _read_field_csv(path: Path):
    """
    Read one exported field CSV back onto its structured grid.

    Reading the CSV rather than the archive is deliberate: the published figure
    and the tidy data a reader is given must be the same numbers, and a second
    extraction path is how the two come to disagree.

    Parameters
    ----------
    path : Path
        A `F7_field_*.csv` written by `export_fields`.

    Returns
    -------
    tuple of np.ndarray
        (X, Y, PHI, ERR), each an (n, n) array on the physical coordinates [m],
        where ERR is the signed error against the manufactured solution. ERR is
        all-NaN where the archive carried no exact field.
    """
    import numpy as np

    with open(path, encoding="utf-8") as fh:
        recs = list(csv.DictReader(fh))
    n = int(round(math.sqrt(len(recs))))

    def grid(key):
        return np.array(
            [float(r[key]) if r[key] not in ("", "None") else math.nan
             for r in recs]).reshape(n, n)

    return grid("x_m"), grid("y_m"), grid("phi_V"), grid("signed_error_V")


# Dimensions drawn in F7. Two dimensions was dropped from the main body once
# `figure_resolution_grid_2d` covered the same case at four resolutions with its
# error and cost attached; the CSVs for both are still exported, so widening this
# back to (2, 3) restores the earlier two-row figure unchanged.
DIMENSIONS_F7: tuple[int, ...] = (3,)


def _attach_colourbar(fig, mappable, ax, label_size: float):
    """
    Attach a colour bar whose height tracks the axes box exactly.

    `fig.colorbar(..., ax=ax)` sizes the bar from the *grid cell*, which is the
    right answer until `set_box_aspect` shrinks the axes inside that cell to make
    it square: the bar then overhangs the panel it belongs to. An inset placed in
    axes coordinates follows the box instead, so bar and panel stay the same
    height whatever the aspect.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure owning the axes.
    mappable : matplotlib.cm.ScalarMappable
        Artist whose colour scale is being drawn.
    ax : matplotlib.axes.Axes
        Panel the bar belongs to.
    label_size : float
        Point size for the bar's tick labels and its offset text.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The bar, so the caller can set a formatter on it.
    """
    cax = ax.inset_axes([1.05, 0.0, 0.055, 1.0])
    bar = fig.colorbar(mappable, cax=cax)
    bar.ax.tick_params(labelsize=label_size)
    bar.ax.yaxis.get_offset_text().set_fontsize(label_size)
    return bar


def figure_het_fields(repo_root: Path, out_dir: Path) -> list[Path]:
    """
    Thruster potential and each solver's signed error, three dimensions.

    A norm collapses a field to one number and hides where the error sits. A
    signed map does not: a sign error, a boundary condition imposed on the wrong
    face, and convergence to the wrong fixed point each have a distinct
    signature in the map and none in the norm. The plane drawn is the mid-plane
    slice of the three-dimensional case, normal to the azimuthal direction,
    which is the plane the channel physics lives in.

    The two-dimensional row this figure used to carry has been removed, not
    lost: `figure_resolution_grid_2d` shows the same case across four
    resolutions with per-panel error and cost, which subsumes a single-resolution
    view of it. Restricting F7 to three dimensions leaves each figure with one
    job — where the error sits, against how the solvers separate as the mesh
    refines — and recovers most of a page in a chapter that has none to spare.
    `DIMENSIONS_F7` governs this; the exported CSVs are still written for both
    dimensions, so restoring the row is a one-line change.

    The leftmost column carries the manufactured solution itself, so the error
    maps beside it are read against the field they belong to. Each error map
    carries its own symmetric scale: the three solvers differ by orders of
    magnitude, and one shared scale would render two of the three uniformly
    flat.

    Parameters
    ----------
    repo_root : Path
        Repository root. Unused beyond signature symmetry with the other figure
        builders; the exported CSVs are read from `out_dir`.
    out_dir : Path
        Directory holding the `F7_field_*.csv` files and receiving the figure.

    Returns
    -------
    list of Path
        Files written, empty where the exported fields are absent.
    """
    import numpy as np

    plt = _matplotlib()

    panels = []
    for dim in DIMENSIONS_F7:
        case = HET_CASE[dim]
        found = sorted(out_dir.glob(f"F7_field_{dim}D_{case}_*.csv"))
        by_solver = {p.stem.rsplit("_", 2)[-2]: p for p in found}
        if "Thomas" not in by_solver:
            log.warning("  no exported %d-D field for %s", dim, case)
            continue
        panels.append((dim, case, by_solver))
    if not panels:
        return []

    # Four panels per dimension -- the reference field and one signed error per
    # quantum solver -- tiled two by two rather than in a single row of four.
    # Across the full text width a row of four leaves each panel 1.5 in wide,
    # which cannot carry an axis label at the body point size; two by two leaves
    # 2.5 in, which can.
    ncol = 2
    fig, axes = plt.subplots(2 * len(panels), ncol,
                             figsize=(TEXT_WIDTH_IN, 2.85 * 2 * len(panels)),
                             squeeze=False)

    for block, (dim, case, by_solver) in enumerate(panels):
        cells = [axes[2 * block + k // ncol][k % ncol] for k in range(4)]

        X, Y, PHI, _ = _read_field_csv(by_solver["Thomas"])
        ax = cells[0]
        im = ax.pcolormesh(X * 1e3, Y * 1e3, PHI, shading="auto",
                           cmap=FIELD_CMAP, rasterized=True)
        _attach_colourbar(fig, im, ax, SMALL_PT)
        ax.set_title(f"{dim}-D manufactured " + r"$\phi$  [V]")

        for k, solver in enumerate(QUANTUM_SOLVERS, start=1):
            ax = cells[k]
            if solver not in by_solver:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"{solver}\nnot recorded", ha="center",
                        va="center", fontsize=ANNOT_PT, color="dimgrey")
                continue
            Xs, Ys, _, ERR = _read_field_csv(by_solver[solver])
            finite = ERR[np.isfinite(ERR)]
            scale = float(np.max(np.abs(finite))) if finite.size else 0.0
            scale = scale or 1.0
            im = ax.pcolormesh(Xs * 1e3, Ys * 1e3, ERR, shading="auto",
                               cmap=SIGNED_ERROR_CMAP, vmin=-scale, vmax=scale,
                               rasterized=True)
            cb = _attach_colourbar(fig, im, ax, SMALL_PT)
            cb.formatter.set_powerlimits((-2, 2))
            ax.set_title(f"{solver}  signed error  [V]",
                         color=SOLVER_COLOUR[solver])

        radial = "radial  [mm]" if dim == 2 else "radial, mid-plane  [mm]"
        for k, ax in enumerate(cells):
            ax.grid(False)
            # The channel is twice as long axially as it is deep radially, but
            # the mesh is N x N, so one cell of the computation is one square of
            # the panel only on a square axes box. Drawn that way the radial
            # coordinate is exaggerated twofold against the axial one, which is
            # what makes the interior structure of the error legible at this
            # size; the axis values state the true extent of each direction.
            ax.set_box_aspect(1.0)
            if k >= 2:
                ax.set_xlabel("axial  [mm]")
            else:
                ax.tick_params(labelbottom=False)
            if k % ncol == 0:
                ax.set_ylabel(radial)
            else:
                ax.tick_params(labelleft=False)

    _headline(fig, "Thruster potential and where each solver puts its error",
              fontweight="bold", fontsize=10)
    return _save(fig, out_dir, "F7_het_fields", plt)


def figure_het_profile_1d(repo_root: Path, out_dir: Path,
                          N: int = 32) -> list[Path]:
    """
    Axial potential and the electric field recovered from it, one dimension.

    The point of the figure is the contrast between its two panels. The
    potential is monotone and forgiving: a solver reproduces it to a fraction of
    a per cent and looks acceptable. The field E = −dφ/dz is not, because
    differentiation multiplies each Fourier component of the error by its
    wavenumber, and the error a truncated Trotter product or a truncated QSP
    polynomial introduces is concentrated at high wavenumber. A solver can
    therefore be right in φ and plainly wrong in E, which is the quantity a
    thruster designer uses.

    Both quantities are drawn normalised against the classical solution's
    extreme rather than in volts. Sub-case 3b does not carry a consistent
    physical scale for φ — see `export_profiles_1d` — and a normalised ordinate
    states exactly what the figure is for without asserting a unit the archive
    does not support. The axial coordinate is physical.

    Parameters
    ----------
    repo_root : Path
        Repository root. Unused beyond signature symmetry with the other figure
        builders; the exported CSV is read from `out_dir`.
    out_dir : Path
        Directory holding `F8_het_1d_profile_N<N>.csv` and receiving the figure.
    N : int
        Resolution whose exported profile is drawn.

    Returns
    -------
    list of Path
        Files written, empty where the exported profile is absent.
    """
    import numpy as np

    path = out_dir / f"F8_het_1d_profile_N{N}.csv"
    if not path.exists():
        log.warning("  %s absent; F8 not rendered", path)
        return []

    with open(path, encoding="utf-8") as fh:
        recs = list(csv.DictReader(fh))
    series: dict[str, dict[str, list]] = {}
    for r in recs:
        s = series.setdefault(r["solver"], {"z": [], "phi": [], "E": []})
        s["z"].append(float(r["z_mm"]))
        s["phi"].append(float(r["phi_code"]))
        s["E"].append(float(r["E_axial_code_per_mm"]))
    if "Thomas" not in series:
        log.warning("  no classical reference in %s; F8 not rendered", path)
        return []

    phi_ref = max(abs(v) for v in series["Thomas"]["phi"]) or 1.0
    E_ref = max(abs(v) for v in series["Thomas"]["E"]) or 1.0

    def _rel_l2(values: list[float], reference: list[float]) -> float:
        """Relative L² error against the classical reference, in per cent."""
        a = np.asarray(values, dtype=float)
        b = np.asarray(reference, dtype=float)
        denom = float(np.linalg.norm(b))
        return 100.0 * float(np.linalg.norm(a - b)) / denom if denom else np.nan

    plt = _matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 3.40))
    for solver in PROFILE_SOLVERS_1D:
        if solver not in series:
            continue
        s = series[solver]
        style = dict(color=SOLVER_COLOUR[solver], marker=SOLVER_MARKER[solver],
                     mfc="none", ms=3.5, lw=1.5)
        # The error is carried in the legend of each panel separately, which is
        # the whole argument of the figure: the same solve is accurate in the
        # potential and inaccurate in the field derived from it.
        if solver == "Thomas":
            lab_phi = lab_E = "Thomas  (reference)"
        else:
            lab_phi = (rf"{solver}   $e_\phi$ = "
                       f"{_rel_l2(s['phi'], series['Thomas']['phi']):.2g}%")
            lab_E = (f"{solver}   $e_E$ = "
                     f"{_rel_l2(s['E'], series['Thomas']['E']):.2g}%")
        axes[0].plot(s["z"], np.array(s["phi"]) / phi_ref,
                     label=lab_phi, **style)
        axes[1].plot(s["z"], np.array(s["E"]) / E_ref, label=lab_E, **style)

    axes[0].set_title("(a)  potential")
    axes[0].set_ylabel(r"$\phi\,/\,|\phi|^{\mathrm{Thomas}}_{\max}$")
    axes[1].set_title("(b)  axial electric field")
    axes[1].set_ylabel(r"$E\,/\,|E|^{\mathrm{Thomas}}_{\max}$")
    for ax in axes:
        ax.set_xlabel("axial position  [mm]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=SMALL_PT)

    _headline(fig, f"HET axial profile, $N={N}$", fontweight="bold",
              fontsize=10)
    return _save(fig, out_dir, f"F8_het_1d_profile_N{N}", plt)


# ── Figure 9: Resolution Against Solver Quality ────────────────────────────────

# Resolutions carried in the F9 panel grid. Chosen as the four at which every
# solver has an archive, and as the range over which the VQLS collapse develops:
# it is invisible at N = 8, first legible at N = 32 and complete by N = 64.
PANEL_RESOLUTIONS_2D: tuple[int, ...] = (8, 16, 32, 64)


def _format_seconds(wall: Optional[float]) -> str:
    """
    Render a wall time for a figure annotation, at two significant figures.

    A fixed number of decimal places cannot serve a range that runs from tens of
    milliseconds for the classical solve to sixteen hours for VQLS at N = 64;
    rounding the classical row to whole seconds prints "0 s", which reads as an
    unmeasured entry rather than as a fast one.

    Parameters
    ----------
    wall : float or None
        Recorded simulation wall time in seconds, or None where not preserved.

    Returns
    -------
    str
        A mathtext label such as ``$t$ = 0.037 s`` or ``$t$ = n/a``.
    """
    if wall is None:
        return "$t$ = n/a"
    if wall >= 100.0:
        return f"$t$ = {wall:.0f} s"
    if wall >= 1.0:
        return f"$t$ = {wall:.1f} s"
    return f"$t$ = {wall:.2g} s"


def figure_resolution_grid_2d(repo_root: Path, out_dir: Path) -> list[Path]:
    """
    Solver against resolution for the two-dimensional thruster potential.

    A grid of solved fields, one row per solver and one column per resolution, on
    a single shared colour scale. It carries two things at once that the scalar
    figures separate. Down a column it shows what each algorithm does to the same
    problem; across a row it shows the mesh resolving the potential. The two
    together make the failure of a variational solver legible as a picture —
    VQLS passes from indistinguishable at N = 8 to visibly granular at N = 32 and
    to noise at N = 64 — where a table of norms records it only as a number
    growing.

    Every panel is annotated with the quantity diagnostic for its row. The
    classical row carries the discretisation error e_disc against the analytical
    solution, which is the floor the mesh alone achieves and the reason that row
    sharpens from left to right. The quantum rows carry the algorithmic error
    e_alg against the classical solution on the same mesh, which isolates the
    solver from the stencil. The distinction is not cosmetic: at N = 8 the total
    error of all four solvers agrees to two significant figures because the mesh
    dominates it, so a total-error annotation would print four near-identical
    numbers across a column whose algorithms differ by ten orders of magnitude.

    Wall time is reported per panel as state-vector simulation time. Two panels
    carry none: their summary rows were reconstructed from the solution archive
    after the recording process was killed, and the instrumentation did not
    survive the recovery.

    The colour scale is fixed once from the classical field and shared by every
    panel. A per-panel scale would renormalise a diverged solution back into the
    same colours as a converged one and conceal precisely what the figure is for.

    Parameters
    ----------
    repo_root : Path
        Repository root.
    out_dir : Path
        Destination for the CSV files and the rendered figure.

    Returns
    -------
    list of Path
        Files written; empty where no field archive is present.
    """
    import numpy as np

    sweep = repo_root / SWEEP_DIR[(2, 2)]
    case = HET_CASE[2]
    rows = load_rows(repo_root, 2, 2)

    fields: dict[tuple[str, int], dict] = {}
    for solver in SOLVERS:
        for N in PANEL_RESOLUTIONS_2D:
            path = sweep / f"solutions_{case}_{solver}_N{N}.npz"
            if not path.exists():
                continue
            data = np.load(path, allow_pickle=False)
            fields[(solver, N)] = {
                "x":     data["x"],
                "y":     data["y"],
                "phi":   (data["phi_solver"] if "phi_solver" in data.files
                          else data["u_solver"]),
                "exact": (data["phi_exact"] if "phi_exact" in data.files
                          else None),
            }
    if not fields:
        log.warning("  no 2-D field archives for %s in %s; F9 not rendered",
                    case, sweep)
        return []

    def _rel_l2(a, b) -> Optional[float]:
        """Relative L2 error of `a` against `b`, in per cent."""
        if a is None or b is None:
            return None
        denom = float(np.linalg.norm(b))
        if denom == 0.0:
            return None
        return 100.0 * float(np.linalg.norm(a - b)) / denom

    def _wall(solver: str, N: int) -> Optional[float]:
        """Recorded simulation wall time, or None where it was not preserved."""
        match = [r for r in rows if r.get("case") == case
                 and r.get("solver") == solver and r.get("N") == N]
        if not match:
            return None
        value = match[0].get("wall_time_s")
        return None if value is None else float(value)

    ref_key = max((k for k in fields if k[0] == "Thomas"),
                  key=lambda k: k[1], default=None)
    if ref_key is None:
        log.warning("  no classical reference field for F9")
        return []
    ref_phi = fields[ref_key]["phi"]
    vmin, vmax = float(np.min(ref_phi)), float(np.max(ref_phi))

    plt = _matplotlib()
    ncol = len(PANEL_RESOLUTIONS_2D)
    nrow = len(SOLVERS)
    # The channel cross-section is roughly twice as long axially as it is deep
    # radially. Panels are drawn at equal aspect, so the canvas has to be sized
    # from the domain or the figure is mostly margin.
    ref_x, ref_y = fields[ref_key]["x"], fields[ref_key]["y"]
    span_x = float(ref_x.max() - ref_x.min()) or 1.0
    span_y = float(ref_y.max() - ref_y.min()) or 1.0
    # Panels keep the true 2:1 aspect of the channel here, where F7 squares
    # them. The two figures ask different questions: F7 is read for the interior
    # structure of one error field, which needs the radial direction stretched,
    # where this one is read across sixteen panels for the pattern of colour and
    # the annotations above each, for which geometry is not the variable. Sixteen
    # square panels would also add two and a half inches of page for nothing.
    panel_w = (TEXT_WIDTH_IN - 1.30) / ncol
    panel_h = panel_w * (span_y / span_x)
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(TEXT_WIDTH_IN, panel_h * nrow + 2.45))

    scalar_rows: list[list[Any]] = []
    field_rows: list[list[Any]] = []
    mesh = None
    for i, solver in enumerate(SOLVERS):
        for j, N in enumerate(PANEL_RESOLUTIONS_2D):
            ax = axes[i][j]
            entry = fields.get((solver, N))
            if entry is None:
                ax.text(0.5, 0.5, "no archive", transform=ax.transAxes,
                        ha="center", va="center", fontsize=ANNOT_PT, color="grey")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
                continue

            mesh = ax.pcolormesh(entry["x"] * 1e3, entry["y"] * 1e3,
                                 entry["phi"], shading="auto",
                                 vmin=vmin, vmax=vmax, rasterized=True)
            ax.grid(False)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=SMALL_PT)
            if i != nrow - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])

            classical = fields.get(("Thomas", N))
            if solver == "Thomas":
                err = _rel_l2(entry["phi"], entry["exact"])
                err_label, err_kind = r"$e_\mathrm{disc}$", "disc"
            else:
                err = _rel_l2(entry["phi"],
                              classical["phi"] if classical else None)
                err_label, err_kind = r"$e_\mathrm{alg}$", "alg"
            wall = _wall(solver, N)

            # Stacked rather than joined by spaces: at N = 8 the joined
            # string is wider than the panel, so adjacent columns' titles run
            # into one another and read as one number.
            parts = [f"{err_label} = {_pct_label(err)}"]
            parts.append(_format_seconds(wall))
            ax.set_title("\n".join(parts), fontsize=SMALL_PT,
                         color=SOLVER_COLOUR[solver], pad=2,
                         linespacing=1.25)

            scalar_rows.append([case, solver, N, err_kind, err, wall])
            xx, yy, pp = entry["x"], entry["y"], entry["phi"]
            for a in range(pp.shape[0]):
                for b in range(pp.shape[1]):
                    field_rows.append([solver, N, float(xx[a, b] * 1e3),
                                       float(yy[a, b] * 1e3), float(pp[a, b])])

    for j, N in enumerate(PANEL_RESOLUTIONS_2D):
        axes[0][j].text(0.5, 1.62, f"$N = {N}$",
                        transform=axes[0][j].transAxes,
                        ha="center", va="bottom", fontsize=AXIS_PT)
    for i, solver in enumerate(SOLVERS):
        axes[i][0].set_ylabel(f"{solver}\nradial  [mm]", fontsize=ANNOT_PT,
                              color=SOLVER_COLOUR[solver])
    for j in range(ncol):
        axes[-1][j].set_xlabel("axial  [mm]", fontsize=ANNOT_PT)

    if mesh is not None:
        # `aspect` is length over width: the default of 20 gives a stub beside a
        # grid four panels tall, so it is raised until the bar spans them.
        cb = fig.colorbar(mesh, ax=axes, fraction=0.020, pad=0.015,
                          aspect=55)
        cb.set_label(r"$\phi$  [V]", fontsize=ANNOT_PT)
        cb.ax.tick_params(labelsize=SMALL_PT)

    _headline(fig, "Resolution against solver quality - two-dimensional "
                   "thruster channel", fontweight="bold", fontsize=10)

    written = _save(fig, out_dir, "F9_resolution_grid_2D", plt)
    written.append(write_csv(
        out_dir / "F9_resolution_grid_2D_metrics.csv",
        ["case", "solver", "N", "error_kind", "error_pct", "wall_time_s"],
        scalar_rows,
    ))
    written.append(write_csv(
        out_dir / "F9_resolution_grid_2D_fields.csv",
        ["solver", "N", "z_mm", "r_mm", "phi"],
        field_rows,
    ))
    return written


# ── Table Data ─────────────────────────────────────────────────────────────────

# Parameter-study archives holding the equal-accuracy sweeps, per dimension, at
# second order. One directory per dimension; the fourth-order studies live in the
# `_4th` siblings and are not carried in the main body.
STUDY_DIR_2ND: dict[int, str] = {
    2: "results/2Dstudies",
    3: "results/3Dstudies",
}

# The knob each solver is swept over by the equal-accuracy protocol, and how it
# is written in a table. The protocol varies one parameter per solver and reports
# the setting whose achieved residual first meets the target.
EA_PARAMETER: dict[str, str] = {
    "hhl":  r"$\varepsilon$",
    "vqls": r"$n_\mathrm{layers}$",
    "qsvt": r"$d_\mathrm{max}$",
}


def table_equal_accuracy_multi_d(repo_root: Path, out_dir: Path) -> list[Path]:
    """
    Cost at a matched residual target in two and three dimensions.

    The one-dimensional equal-accuracy comparison is reported from the primary
    sweep. This is its higher-dimensional counterpart, assembled from the
    parameter studies, and it is the comparison the architecture of this work
    actually delivers: every two- and three-dimensional solve is an outer
    iteration over strips, so the quantity being priced is a whole coupled solve
    rather than one inversion.

    Reported per (dimension, case, solver): the swept parameter and the setting
    the protocol selected, the residual achieved there, whether that residual
    fell inside the acceptance band, and the wall time. A solver whose best
    setting never reaches the band is recorded with `in_band` false rather than
    omitted — an algorithm that cannot be made accurate enough at any available
    setting is a result, and dropping it would flatter the comparison.

    Two caveats travel with these numbers and are carried in the CSV so a reader
    cannot separate them from it. `wall_clamped` marks a row whose outer
    iteration was stopped at the study's wall-clock budget rather than
    converging, so its wall time is the budget and its residual an upper bound.
    `error_measure` names the error column deliberately: `max_rel_err_vs_thomas`
    is a pointwise maximum of a relative error and is unbounded near a node of
    the reference field, which on the three-dimensional thruster case makes it
    read in the millions of per cent while the L² measure `err_alg` reads
    10⁻³ %. Both are written; quote `err_alg` on that case.

    Parameters
    ----------
    repo_root : Path
        Repository root.
    out_dir : Path
        Destination for the CSV.

    Returns
    -------
    list of Path
        Files written; empty where no study archive is present.
    """
    rows: list[list[Any]] = []
    for dim, rel in STUDY_DIR_2ND.items():
        path = repo_root / rel / "equal_accuracy.json"
        if not path.exists():
            log.warning("  %s absent; skipped in the equal-accuracy table", path)
            continue
        meta = repo_root / rel / "run_metadata.json"
        budgets: dict[str, float] = {}
        for candidate in sorted((repo_root / rel).glob("run_metadata*.json")):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            config = payload.get("config", {})
            budget = config.get("max_wall_s")
            for solver in config.get("solvers", []):
                if budget is not None:
                    budgets[str(solver).lower()] = float(budget)
        del meta

        with open(path, encoding="utf-8") as fh:
            records = json.load(fh)
        for rec in records:
            solver = str(rec.get("solver", "")).lower()
            best = rec.get("best_result") or {}
            budget = budgets.get(solver)
            wall = best.get("wall_time_s")
            clamped = (budget is not None and wall is not None
                       and float(wall) >= budget - 5.0)
            rows.append([
                dim, best.get("case_id"), solver.upper(),
                EA_PARAMETER.get(solver, ""), best.get("sensitivity_value"),
                best.get("residual"), rec.get("r_target"), rec.get("in_band"),
                wall, clamped,
                best.get("err_alg"), best.get("max_rel_err_vs_thomas"),
                rec.get("n_solver_calls"), rec.get("notes"),
            ])

    if not rows:
        return []
    rows.sort(key=lambda r: (r[0], str(r[1]), str(r[2])))
    return [write_csv(
        out_dir / "T4_equal_accuracy_2D3D.csv",
        ["dim", "case", "solver", "parameter", "setting", "residual",
         "r_target", "in_band", "wall_time_s", "wall_clamped",
         "err_alg_pct", "max_rel_err_vs_thomas_pct", "n_solver_calls", "notes"],
        rows,
    )]


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
        # Deliberately *not* `bbox_inches="tight"`. That option crops the canvas
        # to the drawn content, so the written width is whatever the content
        # happened to need rather than `TEXT_WIDTH_IN`; `width=\textwidth` then
        # rescales by an unknown factor and the point sizes calibrated above no
        # longer hold on the page. Constrained layout keeps the margins tight
        # instead, at a canvas size that is fixed.
        fig.savefig(path)
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
    written += figure_het_fields(repo_root, out_dir)
    written += figure_het_profile_1d(repo_root, out_dir, N=32)
    written += figure_resolution_grid_2d(repo_root, out_dir)
    written += table_observed_order(repo_root, out_dir)
    written += table_primary_condensed(repo_root, out_dir)
    written += table_equal_accuracy_multi_d(repo_root, out_dir)

    return written
