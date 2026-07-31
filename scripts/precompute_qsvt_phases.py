#!/usr/bin/env python3
"""
Offline precompute of QSVT phase angles into the disk cache used by
solvers/quantum/qsp_angles.py. Run this on the HPC as a batch job (see
submit_precompute_hpc.sh) -- not interactively, since a login-node/
interactive session will get killed on disconnect or idle timeout long
before the larger N values finish.

N values are always processed in ASCENDING order, regardless of the
order given on the command line. Each individual (N, epsilon) result
is written to disk the moment it's computed (via qsp_angles's own
_save_disk, inside compute_inversion_angles) -- nothing is batched or
held in memory until the end. So if the job is killed partway through
a large N (walltime, OOM), everything already computed for smaller N
is already safe on disk; nothing is lost, and re-running with the same
--n-values simply skips what's already cached.

kappa is computed from problems.poisson_1d.build_tst_matrix -- the SAME
matrix construction the live 1D/2D solvers use -- rather than a
separately maintained formula, so a precomputed cache entry is
guaranteed to match what qsvt_1d.py/qsvt_2d.py will look up at runtime.
(qsvt_1d.py's kappa_eff = alpha * kappa_A / ||A||_2 reduces to plain
kappa_A = lambda_max/lambda_min, because alpha is set equal to
||A||_2 = lambda_max there -- see the comment in qsvt_1d.py Stage 1.)
Generic Poisson and both current HET problem classes build the
identical -2/+1/+1 TST matrix for a given N, so one set of phases per
(N, epsilon) serves all of them; there is no separate HET kappa.

Usage
-----
    python scripts/precompute_qsvt_phases.py --n-values 4,8,16
    python scripts/precompute_qsvt_phases.py --n-values 32 --max-degree 2000
    python scripts/precompute_qsvt_phases.py --n-values 64 --max-degree 2000

--max-degree caps the degree solved for (trades approximation error for
tractable Newton-solve runtime/memory). NOTE: per the analysis behind
this script, PolyOneOverX.generate() -- which builds the target
polynomial *before* any cap is applied -- has its own cost that grows
steeply with kappa and is NOT avoided by --max-degree. Treat N=32/64
as exploratory: they may still be impractical even capped. N=4,8,16
are expected to be safe.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import solvers.quantum.qsp_angles as qsp_angles
from problems.poisson_1d import build_tst_matrix


def _kappa_for_N(N: int) -> float:
    """kappa_A = lambda_max/lambda_min of the same TST matrix qsvt_1d.py builds."""
    A = build_tst_matrix(N)
    eigs = np.abs(np.linalg.eigvalsh(A))
    return float(eigs.max() / eigs.min())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-values", type=str, default="4,8,16",
        help="Comma-separated N values, e.g. '4,8,16'. Always processed "
             "in ascending order regardless of the order given. (default: 4,8,16)",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.01,
        help="Primary target epsilon (default: 0.01).",
    )
    parser.add_argument(
        "--extra-epsilons", type=str, default="0.5,0.1",
        help="Additional epsilon values to also precompute (default: 0.5,0.1). "
             "Within each N, epsilons are processed largest-first (cheapest "
             "first), same ascending-safety idea as the N ordering.",
    )
    parser.add_argument(
        "--max-degree", type=int, default=None,
        help="Cap solved degree (see module docstring caveat). Applies to "
             "every (N, epsilon) pair in this invocation -- run separate "
             "invocations if you want different caps for different N.",
    )
    args = parser.parse_args()

    N_values = sorted({int(n) for n in args.n_values.split(",") if n.strip()})
    eps_values = sorted(
        {round(args.epsilon, 8), *[round(float(e), 8) for e in args.extra_epsilons.split(",") if e.strip()]},
        reverse=True,  # largest (cheapest) epsilon first within each N
    )

    # Only this script writes to the disk cache; live solver runs only read.
    qsp_angles._ENABLE_DISK_WRITE = True

    print("=" * 68)
    print("  QSVT Phase Precomputation")
    print("=" * 68)
    print(f"  N values (ascending) : {N_values}")
    print(f"  epsilon values        : {eps_values}")
    print(f"  max_degree cap        : {args.max_degree}")
    print(f"  cache dir              : {qsp_angles._DISK_CACHE_DIR.resolve()}")
    print(flush=True)

    n_ok = n_skip = n_fail = 0
    t_total = time.perf_counter()

    for N in N_values:
        kappa = _kappa_for_N(N)
        print(f"N={N:3d}  kappa={kappa:.4f}", flush=True)

        for eps in eps_values:
            max_deg_key = args.max_degree if args.max_degree is not None else -1
            cache_key = (round(kappa, 4), round(eps, 8), "auto", max_deg_key)

            if qsp_angles._load_disk(cache_key) is not None:
                print(f"    epsilon={eps:<8.4g} -> already cached, skipping.", flush=True)
                n_skip += 1
                continue

            t0 = time.perf_counter()
            print(f"    epsilon={eps:<8.4g} computing...", end="", flush=True)
            try:
                angles, degree = qsp_angles.compute_inversion_angles(
                    kappa, eps, method="auto", max_degree=args.max_degree,
                )
                print(
                    f" done. degree={degree}, n_angles={len(angles)}, "
                    f"time={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )
                n_ok += 1
            except Exception as exc:
                # Deliberately not fatal: must not discard results already
                # written for smaller N, or block remaining (N, epsilon)
                # pairs in this same run.
                print(f" FAILED: {exc}", flush=True)
                n_fail += 1
                continue

        print(flush=True)  # blank line between N blocks

    print("=" * 68)
    print(
        f"  Done in {time.perf_counter() - t_total:.1f}s. "
        f"{n_ok} computed, {n_skip} already cached, {n_fail} failed."
    )
    print(f"  Cache dir: {qsp_angles._DISK_CACHE_DIR.resolve()}")
    print("=" * 68)


if __name__ == "__main__":
    main()