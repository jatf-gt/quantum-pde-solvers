#!/usr/bin/env python3
"""
plot_results.py

Post-processing plots for a run_{1,2,3}d.py sweep, dispatched by --dim.

Consolidates the three former plot_hpc_{1,2,3}Dfull_results.py wrappers, which
were ~90% identical (argument parsing, REPO_ROOT resolution, a one-line call
into benchmark.hpc_plotting) behind disjoint CLI surfaces that were disjoint
by omission rather than by genuine dimensional difference: --N-profile/
--save-pdf existed only for 1D, --case/--N/--list only for 2D/3D, though
filter_rows (benchmark/hpc_plotting.py) is itself dimension-agnostic. This
file keeps every flag; --dim selects which are meaningful.

Command-line wrapper over `benchmark.hpc_plotting.run_{1,2,3}d`; the plotting
logic itself lives there, shared across all three dimensions.

Reads a sweep directory (`results_full.json` plus the archived per-solution
`.npz` files) and writes figures alongside it (1D) or into its `plots/`
subdirectory (2D/3D). Nothing here re-runs a solve — it only reads what the
runner already wrote — so it is cheap and safe to re-run at any time,
including after a walltime-killed job that lost its summary but not its
per-solution data.

Figures produced
-----------------
1D (results/1Dhpc_run/, PNG always, PDF with --save-pdf):
  1. Solution profiles at a chosen N — Thomas vs HHL vs VQLS vs QSVT, with the
     pointwise error underneath, one figure per case.
  2. Maximum relative error vs N — all solvers, log-log.
  3. Residual vs N — all solvers, log-log.
  4. Wall time vs N — all solvers, log-log.
  5. HET 1-D potential profiles and electric field (sub-cases 3a, 3b, 3c).
  6. Summary table of the generic Poisson cases.

2D (results/2Dhpc_run/plots/):
  1. Solution fields — the most important one. For every (case, N): exact or
     Thomas reference plus each solver, with a signed error map underneath.
     This catches a sign error, a misplaced boundary condition, or a solver
     that converged to the wrong fixed point — things a table of norms hides.
  2. Convergence history — residual vs outer iteration, log scale.
  3. Accuracy vs N — log-log, with an O(h²) reference slope.
  4. Cost vs N — weighted strip-solve cost and wall time.
  5. Quantum overhead relative to Thomas.
  6. Error decomposition — algorithmic error against discretisation error.

3D (results/3Dhpc_run/plots/):
  1. Orthogonal slices — axial-radial, axial-azimuthal and radial-azimuthal
     planes through the field, with signed error maps. The primary check.
  2. Polar unwrapping of the azimuthal direction, for the HET channel.
  3. A 3-D cutaway for orientation (skip with --no-cutaway; slowest of the set).
  4. Convergence history.
  5. Accuracy, cost, overhead and error decomposition vs N.
  6. Azimuthal fidelity — how well the periodic direction is resolved.

Usage
-----
    python hpc/runners/plot_results.py --dim 1
    python hpc/runners/plot_results.py --dim 1 --save-pdf --N-profile 64
    python hpc/runners/plot_results.py --dim 2
    python hpc/runners/plot_results.py --dim 2 --case 2D_HET_MMS_SPT100
    python hpc/runners/plot_results.py --dim 3 --case 3D_HET_MMS --no-cutaway
    python hpc/runners/plot_results.py --dim 2 --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RESULTS_DIR = {
    1: REPO_ROOT / "results" / "1Dhpc_run",
    2: REPO_ROOT / "results" / "2Dhpc_run",
    3: REPO_ROOT / "results" / "3Dhpc_run",
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dim", type=int, choices=(1, 2, 3), required=True,
                    help="Which sweep's results to plot.")
    ap.add_argument("--results-dir", type=Path, default=None,
                    help="Directory holding results_full.json and the NPZ "
                         "files (default: results/{1,2,3}Dhpc_run to match "
                         "--dim).")
    ap.add_argument("--case", default=None,
                    help="Restrict to one case. 2D/3D only.")
    ap.add_argument("--N", type=int, default=None,
                    help="Restrict to one N. 2D/3D only.")
    ap.add_argument("--list", action="store_true",
                    help="List the available combinations and exit. 2D/3D only.")
    ap.add_argument("--no-cutaway", action="store_true",
                    help="Skip the mplot3d cutaway orientation figure. 3D only.")
    ap.add_argument("--save-pdf", action="store_true",
                    help="Also save vector PDFs, for inclusion in the thesis. "
                         "1D only.")
    ap.add_argument("--N-profile", type=int, default=32,
                    help="Resolution for the solution profiles (default: 32). "
                         "1D only.")
    a = ap.parse_args()

    results_dir = a.results_dir or DEFAULT_RESULTS_DIR[a.dim]

    if a.dim == 1:
        from benchmark.hpc_plotting import run_1d
        run_1d(results_dir, save_pdf=a.save_pdf, N_profile=a.N_profile)
    elif a.dim == 2:
        from benchmark.hpc_plotting import run_2d
        run_2d(results_dir, case=a.case, N=a.N, listing=a.list)
    else:
        from benchmark.hpc_plotting import run_3d
        run_3d(results_dir, case=a.case, N=a.N, listing=a.list,
              cutaway=not a.no_cutaway)


if __name__ == "__main__":
    main()
