"""
Generate the IBM Heron r2 hardware feasibility table for 1-D QSVT.

Answers, with real transpiled numbers rather than pre-transpilation depth,
the question this project's hardware-scoping discussion posed: at which N
does the 1-D QSVT circuit fit inside what IBM Heron r2 can actually execute?

Usage
-----
    python scripts/resource_feasibility_1d.py
    python scripts/resource_feasibility_1d.py --out results/hardware/

Output
------
results_full.json / results_summary.csv in the target directory, via the
same benchmark.results_io.save_summary used by the HPC sweep drivers, so
this composes with the existing plotting pipeline
(benchmark/hpc_plotting.py) without a separate ingestion path.

Sizes swept
-----------
The (N, kappa, degree) triples are the QSVTConfig1D docstring's own
recommended max_degree operating points -- the degree actually used in
production via the 'reduced_degree' method, not the theoretical
polynomial_degree_estimate() worst-case guide, which the codebase's own
comments note overshoots by an O(1) factor and is not what any real solve
uses. Using the production degree is the honest choice: it is what a real
hardware run would actually submit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.resources import (
    HERON_R2_TWO_QUBIT_GATE_BUDGET,
    feasibility_table,
    validate_composability,
)

# (N, kappa, degree) -- degree from QSVTConfig1D's recommended max_degree
# table (docstring in solvers/quantum/qsvt_1d.py), not the rough guide.
PRODUCTION_SIZES = [
    (4,  9.0,   63),
    (8,  32.0,  127),
    (16, 117.0, 255),
    (32, 441.0, 511),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("results/hardware_feasibility_1d"),
        help="Output directory (default: results/hardware_feasibility_1d)",
    )
    parser.add_argument(
        "--budget", type=int, default=HERON_R2_TWO_QUBIT_GATE_BUDGET,
        help=f"Two-qubit gate budget (default: {HERON_R2_TWO_QUBIT_GATE_BUDGET}, "
             f"IBM's reported Heron r2 circuit capacity)",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip the composability safety check (faster, less trustworthy)",
    )
    args = parser.parse_args()

    if not args.skip_validation:
        print("Validating the composability safe-upper-bound property "
              "before trusting the extrapolated estimates...")
        for N, _kappa, _degree in PRODUCTION_SIZES:
            # Cheap degree, just to re-confirm the bound holds for this N
            # before trusting the large-degree extrapolation below.
            check = validate_composability(N, degree=11)
            status = "OK (safe upper bound)" if check["is_safe_upper_bound"] else "BROKEN"
            print(f"  N={N:3d}: composed={check['composed_two_qubit_count']:5d}  "
                  f"direct={check['direct_two_qubit_count']:5d}  "
                  f"overshoot={check['overshoot_fraction']:+.1%}  [{status}]")
            if not check["is_safe_upper_bound"]:
                raise RuntimeError(
                    f"Composability safe-upper-bound check failed at N={N}. "
                    f"Do not trust the feasibility table below without "
                    f"investigating this first (see core.resources module "
                    f"docstring)."
                )
        print()

    print(f"Estimating post-transpilation resource cost against IBM Heron r2 "
          f"(budget = {args.budget} two-qubit gates)...\n")
    rows = feasibility_table(PRODUCTION_SIZES, budget=args.budget)

    header = (f"{'N':>4} {'kappa':>8} {'degree':>7} {'unit_2Q':>8} "
              f"{'total_2Q':>10} {'feasible':>9} {'ratio':>8}")
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['N']:4d} {row['kappa']:8.1f} {row['degree']:7d} "
              f"{row['unit_two_qubit_count']:8d} {row['total_two_qubit_count']:10d} "
              f"{str(row['feasible']):>9} {row['overshoot_factor']:7.2f}x")

    from benchmark.results_io import save_summary
    save_summary(args.out, rows)
    print(f"\nWritten: {args.out}/results_full.json, "
          f"{args.out}/results_summary.csv")


if __name__ == "__main__":
    main()