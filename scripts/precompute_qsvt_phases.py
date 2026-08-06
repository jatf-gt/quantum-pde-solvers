#!/usr/bin/env python3
"""
Offline precompute of QSVT phase angles into the disk cache read by
solvers/quantum/qsp_angles.py.

Run this on the HPC as a batch job (see hpc/submit_precompute_hpc.sh
and hpc/submit_precompute_2D.sh) rather than interactively: a login-node or
interactive session is killed on disconnect or idle timeout long before the
larger 1-D cases finish.

Why phases are precomputed at all
---------------------------------
Phase-angle computation is the expensive, non-parallelisable stage of QSVT and
depends only on (kappa, epsilon) — not on the right-hand side. Every strip solve
at a given resolution therefore reuses one set of angles, and computing them
once offline removes them from the critical path of the sweep entirely.

Cost is governed by kappa, and the two dimensions sit at opposite extremes:

    1-D   kappa = O(N²)   — degree ~939 at N=4, rising steeply. Large N may be
                            impractical even capped; treat N >= 32 as exploratory.
    2-D   kappa -> 3⁻     — degree ~30-60 irrespective of N. The whole 2-D set
                            precomputes in under two minutes.

This is the same asymptotic contrast that motivates the line decomposition in
the first place.

Incremental safety
------------------
N values are processed in ascending order and epsilons largest-first (cheapest
first) regardless of the order given. Each (kappa, epsilon) result is written to
disk the moment it is computed, inside `compute_inversion_angles` — nothing is
batched or held until the end. A job killed partway through a large N therefore
loses nothing already computed, and re-running the same invocation skips what is
already cached.

Sourcing of kappa
-----------------
Kappa is always computed from the same problem classes the live solvers use —
`problems.poisson_1d.build_tst_matrix` in 1-D and `problems.poisson_line_2d.
PoissonLine2D` in 2-D — never from a maintained table. This matters more than it
appears: the cache key is `(round(kappa, 4), round(epsilon, 8), method,
max_degree)`, so a kappa that differs from the solver's in the fourth decimal is
a guaranteed cache miss, and the expensive computation silently happens live
during the sweep. A hardcoded table cannot be kept in step with the code by
inspection, and an earlier version of this script demonstrated exactly that
failure — its 2-D HET entries had drifted by up to 0.28, so every 2-D HET
precompute above N=4 was being written under a key no solver would ever request.

In 1-D, `qsvt_1d.py`'s kappa_eff = alpha·kappa_A/‖A‖₂ reduces to plain
kappa_A = lambda_max/lambda_min because alpha is set equal to ‖A‖₂ = lambda_max
(see qsvt_1d.py, Stage 1). The generic Poisson and HET 1-D problems build the
identical −2/+1/+1 TST matrix for a given N, so one set of phases per
(N, epsilon) serves both and there is no separate HET kappa.

In 2-D the strip operator depends on the grid aspect ratio, so the unit square
and the HET channel (Lz/Lr = 25/20) give genuinely different kappas and both are
precomputed by default.

Usage
-----
    # 1-D, small N (expected safe)
    python scripts/precompute_qsvt_phases.py --dim 1 --n-values 4,8,16

    # 1-D, larger N with a degree cap
    python scripts/precompute_qsvt_phases.py --dim 1 --n-values 32 --max-degree 2000

    # 2-D, both domains, all N (under two minutes)
    python scripts/precompute_qsvt_phases.py --dim 2

    # 2-D, one domain only
    python scripts/precompute_qsvt_phases.py --dim 2 --domain het --n-values 8,16

    # Show the kappa each case will use, without computing anything
    python scripts/precompute_qsvt_phases.py --dim 2 --list-kappas

`--max-degree` caps the degree solved for, trading approximation error for a
tractable Newton solve. Note that it does not bound the whole cost:
`PolyOneOverX.generate()` builds the target polynomial *before* any cap applies,
and its own cost grows steeply with kappa. The cap is not needed in 2-D.
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
from problems.poisson_line_2d import PoissonLine2D


# ── Domain Definitions ────────────────────────────────────────────────────────

# Physical extents of the HET discharge channel [m]. The strip operator depends
# on the grid through the aspect ratio Lz/Lr alone, so these fix the 2-D HET
# kappa sequence.
HET_LZ, HET_LR = 0.025, 0.020

# Default resolutions per dimension. The 1-D default stops at 16 because kappa
# grows as O(N²) and larger values are not guaranteed to complete; the 2-D
# default spans the full sweep because kappa is bounded by 3 throughout.
DEFAULT_N_1D = "4,8,16"
DEFAULT_N_2D = "4,8,16,32,64,128,256"


def kappa_1d(N: int) -> float:
    """
    Computes κ(A) for the 1-D Poisson operator at resolution N.

    Uses the identical matrix construction as the live solver, so the resulting
    cache key is guaranteed to match the runtime lookup.

    Parameters
    ----------
    N : int
        Number of interior nodes.

    Returns
    -------
    float
        κ = |λ|_max / |λ|_min, growing as O(N²).
    """
    eigs = np.abs(np.linalg.eigvalsh(build_tst_matrix(N)))
    return float(eigs.max() / eigs.min())


def kappa_2d(N: int, domain: str) -> float:
    """
    Computes κ(A_row) for the 2-D line-decomposed strip operator.

    Parameters
    ----------
    N : int
        Number of interior nodes per direction.
    domain : {'square', 'het'}
        'square' is the unit square (dx = dy); 'het' is the axial-radial HET
        channel, whose aspect ratio Lz/Lr = 1.25 gives a distinct κ sequence.

    Returns
    -------
    float
        κ(A_row), bounded above by 3 for the square and approaching ≈2.28 for
        the HET aspect ratio.

    Raises
    ------
    ValueError
        If `domain` is unrecognised.
    """
    if domain == "square":
        problem = PoissonLine2D(np.zeros((N, N)))
    elif domain == "het":
        problem = PoissonLine2D(np.zeros((N, N)), Lx=HET_LZ, Ly=HET_LR)
    else:
        raise ValueError(f"Unknown 2-D domain {domain!r}. Valid: 'square', 'het'.")
    return problem.kappa_row()


def build_targets(dim: int, n_values: list[int], domain: str) -> list[tuple]:
    """
    Enumerates the (label, N, kappa) triples to precompute.

    Deduplicated on the rounded kappa that forms the cache key, so a resolution
    whose κ coincides with one already scheduled is not computed twice.

    Parameters
    ----------
    dim : {1, 2}
        Problem dimension.
    n_values : list[int]
        Resolutions, processed in ascending order.
    domain : {'square', 'het', 'all'}
        2-D domain selection; ignored when dim == 1.

    Returns
    -------
    targets : list[tuple[str, int, float]]
        Ordered (label, N, kappa) triples.
    """
    targets: list[tuple[str, int, float]] = []
    seen: set[float] = set()

    if dim == 1:
        pairs = [("1D", N) for N in n_values]
    else:
        domains = ("square", "het") if domain == "all" else (domain,)
        pairs = [(d, N) for N in n_values for d in domains]

    for label, N in pairs:
        kappa = kappa_1d(N) if dim == 1 else kappa_2d(N, label)
        key = round(kappa, 4)
        if key in seen:
            continue
        seen.add(key)
        targets.append((label, N, kappa))

    return targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dim", type=int, choices=(1, 2), required=True,
        help="Problem dimension. 1-D has kappa = O(N²) and is expensive; 2-D "
             "has kappa -> 3 and is cheap at every N.",
    )
    parser.add_argument(
        "--n-values", type=str, default=None,
        help=f"Comma-separated resolutions, always processed in ascending "
             f"order. Default: {DEFAULT_N_1D} for --dim 1, "
             f"{DEFAULT_N_2D} for --dim 2.",
    )
    parser.add_argument(
        "--domain", type=str, default="all", choices=("square", "het", "all"),
        help="2-D domain (ignored for --dim 1). The strip operator depends on "
             "the grid aspect ratio, so the unit square and the HET channel "
             "have different kappa sequences. (default: all)",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.01,
        help="Primary target epsilon (default: 0.01).",
    )
    parser.add_argument(
        "--extra-epsilons", type=str, default="0.5,0.1",
        help="Additional epsilons to precompute (default: 0.5,0.1). Within "
             "each resolution these are processed largest-first, i.e. cheapest "
             "first, mirroring the ascending-N ordering.",
    )
    parser.add_argument(
        "--max-degree", type=int, default=None,
        help="Cap the solved degree (see the module docstring caveat). Applies "
             "to every pair in this invocation; use separate invocations for "
             "different caps.",
    )
    parser.add_argument(
        "--list-kappas", action="store_true",
        help="Print the kappa each case will use and exit, computing nothing. "
             "Use this to confirm a cache key before committing to a long job.",
    )
    args = parser.parse_args()

    default_n = DEFAULT_N_1D if args.dim == 1 else DEFAULT_N_2D
    n_values = sorted({int(n) for n in (args.n_values or default_n).split(",")
                       if n.strip()})

    eps_values = sorted(
        {round(args.epsilon, 8),
         *[round(float(e), 8) for e in args.extra_epsilons.split(",") if e.strip()]},
        reverse=True,      # largest (cheapest) epsilon first
    )

    targets = build_targets(args.dim, n_values, args.domain)

    if args.list_kappas:
        print(f"\n  {args.dim}-D kappa values (as used for the cache key)\n")
        print(f"  {'domain':<8} {'N':>5} {'kappa':>12}")
        print("  " + "-" * 27)
        for label, N, kappa in targets:
            print(f"  {label:<8} {N:>5} {kappa:>12.4f}")
        print()
        return

    # Only this script writes to the disk cache; live solver runs only read.
    qsp_angles._ENABLE_DISK_WRITE = True

    print("=" * 68)
    print(f"  QSVT Phase Precomputation — {args.dim}-D")
    print("=" * 68)
    print(f"  N values (ascending)  : {n_values}")
    if args.dim == 2:
        print(f"  domain(s)             : {args.domain}")
    print(f"  epsilon values         : {eps_values}")
    print(f"  max_degree cap         : {args.max_degree}")
    print(f"  distinct kappa targets : {len(targets)}")
    print(f"  cache dir              : {qsp_angles._DISK_CACHE_DIR.resolve()}")
    print(flush=True)

    n_ok = n_skip = n_fail = 0
    t_total = time.perf_counter()

    for label, N, kappa in targets:
        print(f"{label:<8} N={N:<4} kappa={kappa:.4f}", flush=True)

        for eps in eps_values:
            max_deg_key = args.max_degree if args.max_degree is not None else -1
            cache_key = (round(kappa, 4), round(eps, 8), "auto", max_deg_key)

            if qsp_angles._load_disk(cache_key) is not None:
                print(f"    epsilon={eps:<8.4g} -> already cached, skipping.",
                      flush=True)
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
                # Deliberately non-fatal: a failure here must not discard
                # results already written for smaller N, nor block the
                # remaining pairs in this same invocation.
                print(f" FAILED: {exc}", flush=True)
                n_fail += 1
                continue

        print(flush=True)

    print("=" * 68)
    print(
        f"  Done in {time.perf_counter() - t_total:.1f}s. "
        f"{n_ok} computed, {n_skip} already cached, {n_fail} failed."
    )
    print(f"  Cache dir: {qsp_angles._DISK_CACHE_DIR.resolve()}")
    print("=" * 68)


if __name__ == "__main__":
    main()
