#!/usr/bin/env python3
"""
Post-processing plots for a run_hpc_1Dfull.py sweep.

Command-line wrapper over `benchmark.hpc_plotting.run_1d`; the plotting logic
itself lives there, shared with the other two dimensions.

Reads a sweep directory (`results_full.json` plus the archived per-solution `.npz` files) and writes figures alongside it. Nothing here
re-runs a solve — it only reads what the runner already wrote — so it is cheap
and safe to re-run at any time, including after a walltime-killed job that lost
its summary but not its per-solution data.

Figures produced
----------------
  1. Solution profiles at a chosen N — Thomas vs HHL vs VQLS vs QSVT, with the
     pointwise error underneath, one figure per case.
  2. Maximum relative error vs N — all solvers, log-log.
  3. Residual vs N — all solvers, log-log.
  4. Wall time vs N — all solvers, log-log.
  5. HET 1-D potential profiles and electric field (sub-cases 3a, 3b, 3c).
  6. Summary table of the generic Poisson cases.

Usage
-----
    python scripts/plot_hpc_1Dfull_results.py --results-dir results/1Dhpc_run
    python scripts/plot_hpc_1Dfull_results.py --results-dir results/1Dhpc_run --save-pdf
    python scripts/plot_hpc_1Dfull_results.py --N-profile 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.hpc_plotting import run_1d


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--results-dir", type=Path,
                    default=Path("results_hpc/results/1Dhpc_run"),
                    help="Directory holding results_full.json and the NPZ files.")
    ap.add_argument("--save-pdf", action="store_true",
                    help="Also save vector PDFs, for inclusion in the thesis.")
    ap.add_argument("--N-profile", type=int, default=32,
                    help="Resolution for the solution profiles (default: 32).")
    a = ap.parse_args()

    run_1d(a.results_dir, save_pdf=a.save_pdf, N_profile=a.N_profile)


if __name__ == "__main__":
    main()
