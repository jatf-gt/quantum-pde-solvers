#!/usr/bin/env python3
"""
plot_studies.py

Post-processing figures for the equal-accuracy and parameter-sensitivity studies
written by `hpc/runners/run_studies.py`, dispatched by --dim.

The counterpart of `hpc/runners/plot_results.py`, which serves the primary
sweeps. The plotting logic itself lives in `benchmark/study_plotting.py`, shared
across all three dimensions; this file is the command-line surface alone.

Reads `results/<dim>Dstudies/` — `equal_accuracy.json` and
`sensitivity_<solver>.json` — and writes into its `figures/` subdirectory.
Nothing here re-runs a solve, so it is cheap and safe to re-run at any time.

Figures produced
----------------
  fig_equal_accuracy_<dim>D        Cost at a matched residual target, with the
                                   residual each solver actually achieved shown
                                   alongside so that a bar cannot be read
                                   without its warrant.
  fig_sensitivity_<solver>_<dim>D  Algorithmic error and wall time against each
                                   swept parameter, one row of panels per
                                   parameter, with the discretisation error of
                                   the case drawn as the accuracy floor.

Each figure is written as PNG and as vector PDF, and is accompanied by a tidy
`data_*.csv` carrying exactly the series drawn, so that a final rendering
elsewhere plots identical numbers rather than values re-derived by a second,
independently written path.

Usage
-----
    python hpc/runners/plot_studies.py --dim 1
    python hpc/runners/plot_studies.py --dim 2
    python hpc/runners/plot_studies.py --dim 3 --study-dir results/3Dstudies
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# `pytest.ini` sets `pythonpath = .`, but a bare invocation of this file places
# `hpc/runners/` on sys.path[0] rather than the repository root. Resolving the
# root from `__file__` decouples the import path from the invocation directory.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.study_plotting import run_studies                     # noqa: E402

DEFAULT_STUDY_DIR: dict[int, Path] = {
    1: REPO_ROOT / "results" / "1Dstudies",
    2: REPO_ROOT / "results" / "2Dstudies",
    3: REPO_ROOT / "results" / "3Dstudies",
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dim", type=int, choices=(1, 2, 3), required=True,
                    help="Which dimension's studies to plot.")
    ap.add_argument("--study-dir", type=Path, default=None,
                    help="Directory holding equal_accuracy.json and "
                         "sensitivity_<solver>.json (default: "
                         "results/{1,2,3}Dstudies to match --dim).")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s",
                        stream=sys.stdout)
    log = logging.getLogger("study_plotting")

    study_dir = args.study_dir or DEFAULT_STUDY_DIR[args.dim]

    log.info("=" * 78)
    log.info("  PARAMETER STUDIES  -  %d-D", args.dim)
    log.info("=" * 78)
    log.info("  source              %s", study_dir)

    written = run_studies(study_dir, args.dim)

    if not written:
        log.error("  No study archives found; nothing written.")
        return 1

    log.info("-" * 78)
    for path in written:
        log.info("    wrote %s", path)
    log.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
