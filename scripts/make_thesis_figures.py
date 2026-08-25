#!/usr/bin/env python3
"""
make_thesis_figures.py

Assembles the main-body thesis figures and their underlying data from the
recorded sweeps.

Where this sits
---------------
`hpc/runners/plot_results.py` renders every figure a sweep supports, for
diagnosis. `hpc/runners/plot_studies.py` does the same for the parameter
studies. Neither selects. This script is the selection layer for the forty-page
main body: it writes one tidy CSV per planned figure, alongside a reference plot
rendered from that same CSV's series.

The CSV is the deliverable. A figure finally typeset in another tool — MATLAB,
TikZ — must plot the same numbers, on the same axes, in the same units as the
reference plot; re-deriving them by a second, independently written path is how
two renderings of one quantity come to disagree in a thesis. Reading the CSV
guarantees they cannot.

Output
------
Written to `results/thesis/` unless redirected:

  F1_accuracy_vs_N_1D          Total error against resolution, orders 2 and 4.
  F2_error_decomposition_1D    Algorithmic against discretisation error.
  F3_qsvt_degree_threshold     QSVT error against the ratio d/kappa.
  F4_kappa_scaling             Condition number, 1-D against 2-D and 3-D strips.
  F5_cost_vs_N_1D              Wall time, with terminated solves marked.
  F6_hardware_verification     Measured fidelity and gate count on ibm_kingston.
  F7_field_*                   2-D and 3-D solution fields and signed error.
  F8_het_1d_profile_*          HET potential and axial electric field.
  T2_primary_condensed         The condensed comparison table's data.
  T3_observed_order            Observed order of accuracy per case.

Each figure that has a rendered form is written as both PNG and vector PDF. A
series whose sweep has not yet landed is drawn as a labelled placeholder rather
than omitted, so the figure keeps its shape and its axes while the run completes.

Usage
-----
    python scripts/make_thesis_figures.py
    python scripts/make_thesis_figures.py --out-dir results/thesis
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.thesis_figures import build_all, set_draw_titles      # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("-o", "--out-dir", type=Path,
                    default=REPO_ROOT / "results" / "thesis",
                    help="Destination for the data tables and reference plots.")
    ap.add_argument("--no-titles", action="store_true",
                    help="Suppress the figure-level headline on every plot. Set "
                         "when the figures are destined for a LaTeX float, whose "
                         "caption states what the headline would and beside "
                         "which a headline reads as a duplicated title.")
    args = ap.parse_args()

    set_draw_titles(not args.no_titles)

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)
    log = logging.getLogger("thesis_figures")

    log.info("=" * 78)
    log.info("  MAIN-BODY THESIS FIGURES")
    log.info("=" * 78)

    written = build_all(REPO_ROOT, args.out_dir)

    log.info("-" * 78)
    for path in written:
        # `--out-dir` accepts any path, including one outside the repository —
        # the untitled render destined for the dissertation tree is written that
        # way. `relative_to` raises rather than falling back on such a path, so
        # the report is shortened only where shortening is defined.
        try:
            shown = path.relative_to(REPO_ROOT)
        except ValueError:
            shown = path
        log.info("    wrote %s", shown)
    log.info("  %d file(s) under %s", len(written), args.out_dir)
    log.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
