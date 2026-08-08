#!/usr/bin/env python3
"""
estimate_hhl_n64.py
====================
Measures ONE real HHL strip solve at n=64, directly - not extrapolated from
a cost model fit to N=4/N=8 data. Answers exactly the question "is this job
about to produce a result, or is it going to run for a day": if one solve
takes T seconds, a full sweep at the finest level is 64*T seconds, and the
job's 8h cap is exceeded by at most one more solve beyond that (with the
finer-grained wall-time fix now in place - see the accompanying core.py).

Run this WHILE the job is still running, on the SAME login/environment
(it does not touch results/ or interfere with the running job in any way -
it just times one independent HHL call). Takes a few minutes at most,
regardless of how long the actual job's per-solve cost turns out to be,
since it is exactly one solve.

Usage:
    python3 estimate_hhl_n64.py
    python3 estimate_hhl_n64.py --n 32     # cross-check against the N=32
                                           # data you already have
"""
import argparse
import time

import numpy as np

from solvers.outer import PoissonLine2D, get_inner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--epsilon", type=float, default=0.01)
    args = ap.parse_args()

    N = args.n
    h = 1.0 / (N + 1)
    p = np.arange(1, N + 1) * h
    X, Y = np.meshgrid(p, p, indexing="ij")
    prob = PoissonLine2D(np.sin(np.pi * X) * np.sin(np.pi * Y))
    A = prob.row_matrix()
    b = prob.rhs()[:, N // 2]   # one representative strip's right-hand side

    print(f"Timing ONE real HHL solve at n={N} (epsilon={args.epsilon})...")
    print("This calls the actual backend - it will take as long as it takes.")

    hhl = get_inner("hhl", epsilon=args.epsilon)
    t0 = time.perf_counter()
    x = hhl(A, b)
    dt = time.perf_counter() - t0

    print()
    print(f"ONE strip solve at n={N}: {dt:.1f}s")
    print(f"One full sweep at this level ({N} such solves): "
         f"{dt*N:.0f}s = {dt*N/3600:.2f}h")
    print()
    print("With the finer-grained wall-time fix (checks before each strip")
    print("solve, not each sweep), the job's cap is exceeded by AT MOST one")
    print("more solve beyond the cap itself - i.e. current elapsed time")
    print(f"should already be close to the cap + {dt:.0f}s, not cap + one sweep.")
    print()
    print("If the running job is already well past (cap + a few multiples of")
    print(f"{dt:.0f}s) with no result, that is now genuinely anomalous given")
    print("this measurement, and is a real reason to kill and investigate -")
    print("not just 'still capped, be patient'.")


if __name__ == "__main__":
    main()