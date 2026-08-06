#!/usr/bin/env python3
"""
Post-processing plots for a run_hpc_2Dfull.py sweep.

Command-line wrapper over `benchmark.hpc_plotting.run_2d`; the plotting logic
itself lives there, shared with the other two dimensions.

Reads a sweep directory (`results_full.json` plus the archived per-solution `.npz` files) and writes figures into its `plots/` subdirectory. Nothing here
re-runs a solve — it only reads what the runner already wrote — so it is cheap
and safe to re-run at any time, including after a walltime-killed job that lost
its summary but not its per-solution data.

Figures produced
----------------
  1. Solution fields — the most important one. For every (case, N): exact or
     Thomas reference plus each solver, with a signed error map underneath.
     This catches a sign error, a misplaced boundary condition, or a solver
     that converged to the wrong fixed point — things a table of norms hides.
  2. Convergence history — residual vs outer iteration, log scale. Shows
     stagnation directly.
  3. Accuracy vs N — log-log, with an O(h²) reference slope.
  4. Cost vs N — weighted strip-solve cost and wall time.
  5. Quantum overhead relative to Thomas.
  6. Error decomposition — algorithmic error against discretisation error, so
     it is visible which one dominates at each N.

Usage
-----
    python scripts/plot_hpc_2Dfull_results.py
    python scripts/plot_hpc_2Dfull_results.py --case 2D_HET_MMS_SPT100
    python scripts/plot_hpc_2Dfull_results.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.hpc_plotting import run_2d


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results-dir", type=Path,
                    default=REPO_ROOT / "results" / "2Dhpc_run",
                    help="Sweep output directory.")
    ap.add_argument("--case", default=None, help="Restrict to one case.")
    ap.add_argument("--N", type=int, default=None, help="Restrict to one N.")
    ap.add_argument("--list", action="store_true",
                    help="List the available combinations and exit.")
    a = ap.parse_args()

    run_2d(a.results_dir, case=a.case, N=a.N, listing=a.list)


if __name__ == "__main__":
    main()
