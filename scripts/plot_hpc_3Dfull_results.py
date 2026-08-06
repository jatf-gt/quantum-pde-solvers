#!/usr/bin/env python3
"""
Post-processing plots for a run_hpc_3Dfull.py sweep.

Command-line wrapper over `benchmark.hpc_plotting.run_3d`; the plotting logic
itself lives there, shared with the other two dimensions.

Reads a sweep directory (`results_full.json` plus the archived per-solution `.npz` files) and writes figures into its `plots/` subdirectory. Nothing here
re-runs a solve — it only reads what the runner already wrote — so it is cheap
and safe to re-run at any time, including after a walltime-killed job that lost
its summary but not its per-solution data.

Figures produced
----------------
  1. Orthogonal slices — axial-radial, axial-azimuthal and radial-azimuthal
     planes through the field, with signed error maps. The primary check.
  2. Polar unwrapping of the azimuthal direction, for the HET channel.
  3. A 3-D cutaway for orientation (skip with --no-cutaway; slowest of the set).
  4. Convergence history.
  5. Accuracy, cost, overhead and error decomposition vs N.
  6. Azimuthal fidelity — how well the periodic direction is resolved.

Usage
-----
    python scripts/plot_hpc_3Dfull_results.py
    python scripts/plot_hpc_3Dfull_results.py --case 3D_HET_MMS --no-cutaway
    python scripts/plot_hpc_3Dfull_results.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.hpc_plotting import run_3d


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results-dir", type=Path,
                    default=REPO_ROOT / "results" / "3Dhpc_run",
                    help="Sweep output directory.")
    ap.add_argument("--case", default=None, help="Restrict to one case.")
    ap.add_argument("--N", type=int, default=None, help="Restrict to one N.")
    ap.add_argument("--no-cutaway", action="store_true",
                    help="Skip the mplot3d cutaway orientation figure.")
    ap.add_argument("--list", action="store_true",
                    help="List the available combinations and exit.")
    a = ap.parse_args()

    run_3d(a.results_dir, case=a.case, N=a.N, listing=a.list,
           cutaway=not a.no_cutaway)


if __name__ == "__main__":
    main()
