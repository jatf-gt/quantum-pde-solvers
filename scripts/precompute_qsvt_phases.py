#!/usr/bin/env python3
"""
precompute_qsvt_phases.py
=========================
Precompute and cache QSVT phase angles for all (kappa, epsilon) pairs
used in the benchmark sweep (N = 4, 8, 16, 32, 64).

Run this script ONCE on the HPC before launching the main benchmark:

    python scripts/precompute_qsvt_phases.py [--include-n64] [--epsilon 0.01]

The computed phases are saved to results/qsvt_phase_cache/ as individual
.npz files. All subsequent calls to compute_inversion_angles() with the
same parameters will load from disk instantly, with zero recomputation.

This script can also be run in parallel for different N values using
multiple PBS jobs, since each (kappa, epsilon) pair is independent.

Usage examples
--------------
    # Standard benchmark (N=4,8,16,32), epsilon=0.01
    python scripts/precompute_qsvt_phases.py

    # Include N=64 (large kappa, may take several minutes)
    python scripts/precompute_qsvt_phases.py --include-n64

    # Use reduced degree for fast approximate computation
    python scripts/precompute_qsvt_phases.py --include-n64 --reduced-degree 255

    # Custom epsilon
    python scripts/precompute_qsvt_phases.py --epsilon 0.5

PBS job example (compute all N including N=64)
-----------------------------------------------
    #!/bin/bash
    #PBS -N qsvt_precompute
    #PBS -l select=1:ncpus=1:mem=8gb
    #PBS -l walltime=04:00:00
    #PBS -q cpu72
    cd $PBS_O_WORKDIR
    source activate quantum-pde-solvers
    python scripts/precompute_qsvt_phases.py --include-n64

Author : Juan Antonio Trobajo Flecha
Date   : July 2026
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root is on path.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from solvers.quantum.qsp_angles import compute_inversion_angles, _DISK_CACHE_DIR


def _kappa_for_N(N: int) -> float:
    """
    Compute the condition number of the N×N TST Poisson matrix.

    The exact spectral norm is lambda_max = 2 - 2*cos(pi/(N+1)) * ... 
    but for the subnormalisation we use the spectral norm of the
    negated positive-definite matrix, which equals the largest eigenvalue:
        lambda_max = 2 - 2*cos(pi/(N+1))  ... no, that's lambda_min.
    
    For the TST with a=-2, b=1 (negated: a=+2, b=-1):
        lambda_k = 2 - 2*cos(k*pi/(N+1))  for k=1,...,N
        lambda_max = 2 - 2*cos(N*pi/(N+1)) ≈ 4 for large N
        lambda_min = 2 - 2*cos(pi/(N+1))  ≈ pi^2/(N+1)^2 for large N
        kappa = lambda_max / lambda_min
    """
    A = -2.0 * np.eye(N) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)
    # Negate to get positive definite matrix.
    A_pos = -A
    eigs  = np.linalg.eigvalsh(A_pos)
    return float(eigs.max() / eigs.min())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute QSVT phase angles for the benchmark sweep.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--include-n64", action="store_true",
        help="Include N=64 (kappa~1700). May take several minutes.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.01,
        help="Approximation error epsilon (default: 0.01).",
    )
    parser.add_argument(
        "--extra-epsilons", type=str, default="0.5,0.1",
        help="Comma-separated additional epsilon values to precompute "
             "(default: '0.5,0.1'). These are used by the runner script "
             "for fast approximate QSVT.",
    )
    parser.add_argument(
        "--reduced-degree", type=int, default=None,
        help="If set, also precompute reduced-degree phases with this cap. "
             "Recommended: 255 for N=32, 511 for N=64.",
    )
    parser.add_argument(
        "--method", type=str, default="sym_qsp_direct",
        choices=["sym_qsp_direct", "sym_qsp_wrapper"],
        help="Phase computation method (default: sym_qsp_direct).",
    )
    args = parser.parse_args()

    # N values to precompute.
    N_values = [4, 8, 16, 32]
    if args.include_n64:
        N_values.append(64)

    # Epsilon values.
    extra_eps = [float(e) for e in args.extra_epsilons.split(",") if e.strip()]
    epsilon_values = sorted(set([args.epsilon] + extra_eps), reverse=True)

    print("=" * 68)
    print("  QSVT Phase Precomputation")
    print("  Imperial College London, Department of Aeronautics")
    print("=" * 68)
    print(f"  N values:      {N_values}")
    print(f"  Epsilon values: {epsilon_values}")
    print(f"  Method:        {args.method}")
    print(f"  Reduced degree: {args.reduced_degree}")
    print(f"  Cache dir:     {_DISK_CACHE_DIR.resolve()}")
    print()

    t_total = time.perf_counter()
    n_computed = 0
    n_cached   = 0

    for N in N_values:
        kappa = _kappa_for_N(N)
        print(f"  N={N:3d}  kappa={kappa:.2f}")

        for epsilon in epsilon_values:
            # Standard method.
            key = (round(kappa, 4), round(epsilon, 8), args.method, None)
            from solvers.quantum.qsp_angles import _load_disk
            if _load_disk(key) is not None:
                print(f"    epsilon={epsilon:.3f}  [{args.method}]  "
                      f"→ already cached, skipping.")
                n_cached += 1
                continue

            t0 = time.perf_counter()
            print(f"    epsilon={epsilon:.3f}  [{args.method}]  computing...",
                  end="", flush=True)
            try:
                angles, degree = compute_inversion_angles(
                    kappa, epsilon, method=args.method
                )
                elapsed = time.perf_counter() - t0
                print(f" done. degree={degree}, time={elapsed:.1f}s")
                n_computed += 1
            except Exception as exc:
                print(f" FAILED: {exc}")

            # Reduced degree variant.
            if args.reduced_degree is not None:
                cap = args.reduced_degree
                key_rd = (round(kappa, 4), round(epsilon, 8),
                          "reduced_degree", cap)
                if _load_disk(key_rd) is not None:
                    print(f"    epsilon={epsilon:.3f}  [reduced_degree={cap}]  "
                          f"→ already cached, skipping.")
                    n_cached += 1
                    continue

                t0 = time.perf_counter()
                print(f"    epsilon={epsilon:.3f}  [reduced_degree={cap}]  "
                      f"computing...", end="", flush=True)
                try:
                    angles_rd, degree_rd = compute_inversion_angles(
                        kappa, epsilon,
                        method     = "reduced_degree",
                        max_degree = cap,
                    )
                    elapsed = time.perf_counter() - t0
                    print(f" done. degree={degree_rd}, time={elapsed:.1f}s")
                    n_computed += 1
                except Exception as exc:
                    print(f" FAILED: {exc}")

        print()

    elapsed_total = time.perf_counter() - t_total
    print("=" * 68)
    print(f"  Precomputation complete.")
    print(f"  Computed: {n_computed} phase sets.")
    print(f"  Skipped (already cached): {n_cached} phase sets.")
    print(f"  Total time: {elapsed_total:.1f}s")
    print(f"  Cache location: {_DISK_CACHE_DIR.resolve()}")
    print("=" * 68)


if __name__ == "__main__":
    main()