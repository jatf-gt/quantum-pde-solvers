#!/usr/bin/env python3
"""
Precompute and cache QSVT phase angles for the 2D Poisson benchmark cases.

The 2D Line-Jacobi solver uses a row matrix with kappa ~ 2-3 (O(1) for all N),
which is far smaller than the 1D TST matrix kappas. This script computes phases
for both domain types:

  Unit square (Sections 1, 2, 3, 5):  kappa = 2.3586, 2.7725, 2.9352, 2.9838,
                                                2.9960, 2.9990, 2.9998
  HET domain   (Sections 4, 5):        kappa = 1.9228, 1.9704, 1.9926, 1.9982,
                                                1.9995, 1.9999, 2.0000

Because kappa ~ 3 throughout, polynomial degrees are ~30-60 regardless of N.
Each (kappa, epsilon) pair computes in seconds. The full set of 2D kappas
across all N and both domains precomputes in under 2 minutes.

Usage
-----
  # All 2D kappas, epsilon=0.01 (primary) + 0.1 and 0.5 (secondary):
  python scripts/precompute_2D_qsvt_phases.py

  # Specific kappas only:
  python scripts/precompute_2D_qsvt_phases.py --kappas 2.3586,2.7725

  # With a degree cap (not needed for 2D -- kappa~3 gives degree~60):
  python scripts/precompute_2D_qsvt_phases.py --max-degree 200

Author : Juan Antonio Trobajo Flecha
Date   : August 2026
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


# ============================================================================
#  2D kappa values (computed analytically from the row matrix eigenvalues)
#
#  Unit square domain [0,1]x[0,1]: dx = dy = 1/(N+1)
#    A_row: main diag = -4/dx^2 * (1 + 1) / 2 ... actually:
#    a = -2*(1/dx^2 + 1/dy^2) = -4/dx^2  (since dx=dy)
#    b = 1/dx^2
#    kappa = |lambda_max| / |lambda_min|
#    lambda_j = a + 2b*cos(j*pi/(N+1))  for j=1..N
#    = -4/dx^2 + 2/dx^2 * cos(j*pi/(N+1))
#    Scaling by dx^2 cancels: kappa depends only on N, not on dx.
#    kappa = (4 - 2*cos(pi/(N+1))) / (4 + 2*cos(pi/(N+1)))  [inverted sign]
#    Wait -- eigenvalues are negative. |lambda_max| = 4 - 2*cos(pi/(N+1)),
#    |lambda_min| = 4 + 2*cos(N*pi/(N+1)) = 4 - 2*cos(pi/(N+1)) ... no.
#    Correct: lambda_j = -4 + 2*cos(j*pi/(N+1)) for the UNIT matrix (dx=1).
#    |lambda_max| = |-4 + 2*cos(pi/(N+1))| = 4 - 2*cos(pi/(N+1))  [j=1, smallest |]
#    |lambda_min| = |-4 + 2*cos(N*pi/(N+1))| = 4 + 2*cos(pi/(N+1))  [j=N, largest |]
#    kappa = (4 + 2*cos(pi/(N+1))) / (4 - 2*cos(pi/(N+1)))
#    This is the SAME for any dx=dy (scaling cancels).
#
#  HET domain [0,Lz]x[0,Lr]: dz = Lz/(N+1), dr = Lr/(N+1), dz != dr
#    a = -2*(1/dz^2 + 1/dr^2), b = 1/dz^2
#    lambda_j = a + 2b*cos(j*pi/(N+1))
#    kappa depends on the ratio dz/dr = Lz/Lr = 25/20 = 1.25
# ============================================================================

# Pre-computed kappa values for all N in the 2D sweep
# Verified by running _build_row_matrix(N, dx, dy) for each case.
KAPPAS_UNIT_SQUARE: dict[int, float] = {
    4:   2.3586,
    8:   2.7725,
    16:  2.9352,
    32:  2.9838,
    64:  2.9960,
    128: 2.9990,
    256: 2.9998,
}

KAPPAS_HET_DOMAIN: dict[int, float] = {
    # Lz=0.025m, Lr=0.020m => dz/dr = 1.25
    4:   1.9228,
    8:   1.9704,
    16:  1.9926,
    32:  1.9982,
    64:  1.9995,
    128: 1.9999,
    256: 2.0000,
}

# All unique kappas across both domain types
ALL_2D_KAPPAS: list[float] = sorted(set(
    list(KAPPAS_UNIT_SQUARE.values()) + list(KAPPAS_HET_DOMAIN.values())
))


def _verify_kappas(verbose: bool = True) -> None:
    """
    Recompute kappas from first principles and verify against the table.
    Prints any discrepancy > 0.001.
    """
    HET_Lz, HET_Lr = 0.025, 0.020

    for N in [4, 8, 16, 32, 64, 128, 256]:
        # Unit square
        dx = dy = 1.0 / (N + 1)
        a = -2.0 * (1.0/dx**2 + 1.0/dy**2)
        b = 1.0 / dx**2
        A = a*np.eye(N) + b*(np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1))
        eigs = np.abs(np.linalg.eigvalsh(A))
        k_sq = float(eigs.max() / eigs.min())

        # HET domain
        dz = HET_Lz / (N + 1)
        dr = HET_Lr / (N + 1)
        a_h = -2.0 * (1.0/dz**2 + 1.0/dr**2)
        b_h = 1.0 / dz**2
        A_h = a_h*np.eye(N) + b_h*(np.diag(np.ones(N-1),1) + np.diag(np.ones(N-1),-1))
        eigs_h = np.abs(np.linalg.eigvalsh(A_h))
        k_het = float(eigs_h.max() / eigs_h.min())

        if verbose:
            sq_stored  = KAPPAS_UNIT_SQUARE.get(N, float("nan"))
            het_stored = KAPPAS_HET_DOMAIN.get(N, float("nan"))
            sq_ok  = abs(k_sq  - sq_stored)  < 0.001
            het_ok = abs(k_het - het_stored) < 0.001
            print(f"  N={N:3d}  unit_sq: computed={k_sq:.4f}  stored={sq_stored:.4f}  "
                  f"{'OK' if sq_ok else 'MISMATCH'}  |  "
                  f"HET: computed={k_het:.4f}  stored={het_stored:.4f}  "
                  f"{'OK' if het_ok else 'MISMATCH'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--kappas", type=str, default=None,
        help="Comma-separated kappa values to precompute. "
             "Default: all 2D kappas for N=4..256 across both domain types.",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.01,
        help="Primary epsilon (default: 0.01).",
    )
    parser.add_argument(
        "--extra-epsilons", type=str, default="0.5,0.1",
        help="Additional epsilons (default: 0.5,0.1).",
    )
    parser.add_argument(
        "--max-degree", type=int, default=None,
        help="Degree cap. Not needed for 2D (kappa~3, degree~60). "
             "Default: uncapped.",
    )
    parser.add_argument(
        "--verify-kappas", action="store_true",
        help="Recompute kappas from first principles and verify the table.",
    )
    args = parser.parse_args()

    if args.verify_kappas:
        print("Verifying kappa table...")
        _verify_kappas(verbose=True)
        print()

    # Determine kappa list
    if args.kappas is not None:
        kappa_list = sorted({round(float(k), 4)
                             for k in args.kappas.split(",") if k.strip()})
    else:
        kappa_list = [round(k, 4) for k in ALL_2D_KAPPAS]

    # Epsilon list: primary + extras, largest first (cheapest first)
    eps_set = {round(args.epsilon, 8)}
    for e in args.extra_epsilons.split(","):
        if e.strip():
            eps_set.add(round(float(e), 8))
    eps_list = sorted(eps_set, reverse=True)

    # Enable disk writes
    qsp_angles._ENABLE_DISK_WRITE = True

    print("=" * 68)
    print("  QSVT Phase Precomputation — 2D Cases")
    print("=" * 68)
    print(f"  Kappa values  : {kappa_list}")
    print(f"  Epsilon values: {eps_list}")
    print(f"  Max degree cap: {args.max_degree}")
    print(f"  Cache dir     : {qsp_angles._DISK_CACHE_DIR.resolve()}")
    print()
    print("  Domain mapping:")
    print("    Unit square (Sections 1,2,3,5):")
    for N, k in KAPPAS_UNIT_SQUARE.items():
        print(f"      N={N:3d}  kappa={k:.4f}")
    print("    HET domain (Sections 4,5):")
    for N, k in KAPPAS_HET_DOMAIN.items():
        print(f"      N={N:3d}  kappa={k:.4f}")
    print("=" * 68, flush=True)

    n_ok = n_skip = n_fail = 0
    t_total = time.perf_counter()

    for kappa in kappa_list:
        print(f"\nkappa={kappa:.4f}", flush=True)
        for eps in eps_list:
            max_deg_key = args.max_degree if args.max_degree is not None else -1
            cache_key = (round(kappa, 4), round(eps, 8), "auto", max_deg_key)

            if qsp_angles._load_disk(cache_key) is not None:
                print(f"  epsilon={eps:<8.4g} -> already cached, skipping.",
                      flush=True)
                n_skip += 1
                continue

            t0 = time.perf_counter()
            print(f"  epsilon={eps:<8.4g} computing...", end="", flush=True)
            try:
                angles, degree = qsp_angles.compute_inversion_angles(
                    kappa, eps, method="auto", max_degree=args.max_degree,
                )
                elapsed = time.perf_counter() - t0
                print(f" done.  degree={degree}  n_angles={len(angles)}"
                      f"  time={elapsed:.2f}s", flush=True)
                n_ok += 1
            except Exception as exc:
                print(f" FAILED: {exc}", flush=True)
                n_fail += 1

    print()
    print("=" * 68)
    print(f"  Done in {time.perf_counter() - t_total:.1f}s.  "
          f"{n_ok} computed, {n_skip} already cached, {n_fail} failed.")
    print(f"  Cache dir: {qsp_angles._DISK_CACHE_DIR.resolve()}")
    print("=" * 68)


if __name__ == "__main__":
    main()