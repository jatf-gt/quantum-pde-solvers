#!/usr/bin/env python3
"""
debug_1d.py
===========
Debug and benchmark tool for the 1-D solvers (solvers/classical, solvers/quantum).

New in this consolidation: no 1-D counterpart of debug_2d.py / debug_3d.py
existed before - the capability was scattered across run_qsvt_debug.py,
run_hpc_1Dfull.py and benchmark/runner.py's sweeps D/H4. Cases come from the
canonical registry in core/cases.py; sub-cases 3b and 3c are solved as raw
``(A, b)`` systems rather than through ``PoissonProblem1D``, because 3b has no
closed form and 3c uses the "including-origin" grid for its Neumann
condition, which ``PoissonProblem1D`` does not support. Dispatch goes through
``solvers.outer.get_inner`` - the same ``(A, b) -> x`` registry the outer
iteration uses per strip in debug_2d.py/debug_3d.py - rather than the
per-algorithm ``*_solve`` wrappers, so the same ``-I`` option syntax, the same
validated registry, and the same timing/diagnostics collection apply here too.

Usage
-----
    python scripts/debug_1d.py --case poisson_1d_fS_hom --N 8
    python scripts/debug_1d.py --case het_1d_3c_neumann --N 16 --inner qsvt
    python scripts/debug_1d.py --case all --N 8 --inner all
    python scripts/debug_1d.py --N 8 --inner qsvt -I max_degree=300
    python scripts/debug_1d.py --N 32 --inner all \
           -I qsvt.max_degree=300 -I hhl.epsilon=0.05 -I vqls.n_restarts=2

    # node-by-node solution and pointwise error
    python scripts/debug_1d.py --dump --case het_1d_3a_linear --N 8 --inner qsvt

    # kappa(N) against the plain-TST theoretical (4/pi^2)(N+1)^2, every case
    python scripts/debug_1d.py --kappa-table

    python scripts/debug_1d.py --list-cases
    python scripts/debug_1d.py --list-options

QSVT phase cache
----------------
QSVT phase angles are looked up from a disk cache keyed on
``(kappa, epsilon, angle_method, max_degree)`` - see
hpc/runners/precompute_phases.py. Unlike hpc/runners/run_1d.py, this
tool does not proactively cap the degree on a cache miss; it warns and lets
you supply ``-I max_degree=<n>`` yourself. An unattended HPC sweep needs a
silent, proactive cap; an interactive debug session run at the keyboard is
better served by a warning you can act on, including Ctrl-C.

kappa-table scope
------------------
The legacy table in scripts/run_het_benchmark.py additionally reported
alpha, lambda_D and alpha_bc per HET case; those come from the underlying
``HETConfig``, which ``core.cases.BuiltCase`` does not expose (only ``kappa``
is carried through). Reproducing them here would mean re-deriving the
HETConfig each case was built from, duplicating knowledge the registry
already owns. --kappa-table therefore reports kappa against the plain-TST
theoretical scaling for every case uniformly; for the four cases built from
HETConfig (het_1d_linear_scaled, het_1d_gaussian_hom,
het_1d_gaussian_Vd300_scaled, het_1d_step_scaled) the ratio column itself is
the diagnostic - it is not 1, and that is the point.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import cases
from benchmark.diagnostics import BOLD, RED, RESET, YELLOW, colour, rel_err
from solvers.outer import available_inner, describe_inner, get_inner, parse_kv

QUANTUM = ("hhl", "vqls", "qsvt")


def _fmt_pct(e: float, good: float = 1.0, ok: float = 5.0) -> str:
    """Coloured percentage, or 'n/a' when no reference is available."""
    if not np.isfinite(e):
        return f"{'n/a':>10}"
    return f"{colour(e, good, ok)}{e:>9.3f}%{RESET}"


def _notes(summary: dict) -> str:
    """Renders the solver-specific diagnostics from InnerSolverWrapper.summary()."""
    parts = []
    if "polynomial_degree_mean" in summary:
        parts.append(f"degree={int(summary['polynomial_degree_mean'])}")
    if "final_cost_mean" in summary:
        parts.append(f"cost={summary['final_cost_mean']:.2e}")
    if "prop_const_mean" in summary:
        parts.append(f"c={summary['prop_const_mean']:.3e}")
    if summary.get("inner_failures"):
        parts.append(f"{summary['inner_failures']} fallback(s)")
    return ", ".join(parts)


def _warn_if_qsvt_cache_miss(kappa: float, opts: dict) -> None:
    """Warns when the disk phase cache has no entry for this exact key."""
    import solvers.quantum.qsp_angles as qsp_angles

    epsilon = float(opts.get("epsilon", 0.01))
    angle_method = opts.get("angle_method", "auto")
    max_degree = opts.get("max_degree")
    max_degree = int(max_degree) if max_degree is not None else None
    key = (round(kappa, 4), round(epsilon, 8), angle_method,
           max_degree if max_degree is not None else -1)
    if qsp_angles._load_disk(key) is None:
        print(f"  {YELLOW}QSVT: no cached phases for key={key} "
              f"(kappa={kappa:.4f}); live angle-finding may take minutes to "
              f"hours at this kappa. Pass -I max_degree=1000 for a cheap "
              f"capped run, or precompute via "
              f"hpc/runners/precompute_phases.py.{RESET}")


def _rel_err_pct_pointwise(u: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Pointwise relative error in per cent, NaN where |ref| < 1% of its max."""
    scale = np.max(np.abs(ref))
    mask = np.abs(ref) > 0.01 * scale
    return np.where(mask, np.abs(u - ref) / np.abs(ref) * 100.0, np.nan)


def _dump_solution(case_name: str, N: int, solver: str, x: np.ndarray,
                   u: np.ndarray, u_exact: np.ndarray | None) -> None:
    """Node-by-node solution and pointwise error, for one solver."""
    print(f"\n  {BOLD}{solver} solution, case={case_name} N={N}{RESET}")
    print(f"    x = {np.round(x, 4).tolist()}")
    print(f"    u = {np.round(u, 6).tolist()}")
    if u_exact is not None:
        err = _rel_err_pct_pointwise(u, u_exact)
        print(f"    pointwise error vs exact (%) = {np.round(err, 3).tolist()}")


# -- Case comparison ------------------------------------------------------------

def run_case(name: str, N: int, inners: list[str], inner_opts: dict,
            dump: bool = False, verbose: bool = False) -> list[tuple]:
    case = cases.get(name)
    if case.dim != 1:
        raise SystemExit(f"{name!r} is {case.dim}D, not 1D.")
    built = case.build(N)
    A, b, u_exact, kappa = built.A, built.b, built.exact, built.kappa

    print(f"\n{BOLD}{'=' * 88}{RESET}")
    print(f"{BOLD}  CASE {case.name}   N={N}   grid={case.grid}   "
          f"kappa(A)={kappa:.4f}{RESET}")
    print(f"{BOLD}  {case.summary}{RESET}")
    if u_exact is None:
        print(f"{BOLD}  no closed-form/quadrature reference for this case - "
              f"'vs exact%' reports n/a{RESET}")
    print(f"{BOLD}{'=' * 88}{RESET}")

    ref = None
    rows = []
    for inner in inners:
        opts = dict(inner_opts.get(inner, {}))
        if inner == "qsvt":
            opts.setdefault("label", f"{case.name}-N{N}")
            _warn_if_qsvt_cache_miss(kappa, opts)

        t0 = time.perf_counter()
        try:
            solver = get_inner(inner, fallback_to_thomas=False, **opts)
            u = solver(A, b)
        except Exception as exc:
            print(f"  {RED}[FAIL] {inner}: {exc}{RESET}")
            continue
        wall = time.perf_counter() - t0

        if ref is None and inner == "thomas":
            ref = u
        e_exact = rel_err(u, u_exact) if u_exact is not None else float("nan")
        e_ref = rel_err(u, ref) if ref is not None else float("nan")
        residual = float(np.linalg.norm(A @ u - b) / (np.linalg.norm(b) + 1e-300))
        rows.append((inner, u, e_exact, e_ref, residual, wall, solver.summary()))

        if dump:
            _dump_solution(case.name, N, inner, built.coords[0], u, u_exact)
        if verbose:
            print(f"    {inner}: residual={residual:.3e} summary={solver.summary()}")

    print(f"\n  {'solver':<9} {'time s':>9} {'residual':>12} "
          f"{'vs exact%':>10} {'vs Thomas%':>11}  {'notes'}")
    print(f"  {'-' * 78}")
    for inner, u, e_exact, e_ref, residual, wall, summary in rows:
        print(f"  {inner:<9} {wall:>9.3f} {residual:>12.4e} "
              f"{_fmt_pct(e_exact)} {_fmt_pct(e_ref, 0.5, 2.0)}  {_notes(summary)}")

    return rows


# -- Kappa scaling table --------------------------------------------------------

def kappa_table(case_names: list[str] | None = None) -> None:
    """
    kappa(N) for every requested case against the plain-TST theoretical
    scaling (4/pi^2)(N+1)^2 - the 1-D counterpart of benchmark/runner.py's
    sweep D, extended to every registered 1D case rather than just the
    generic Poisson matrix.
    """
    names = case_names or cases.available(dim=1)
    print(f"\n{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  1-D CONDITION NUMBER SCALING{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
    for name in names:
        case = cases.get(name)
        print(f"\n  {BOLD}{case.name}{RESET}")
        print(f"  {'N':>5} {'kappa':>12} {'(4/pi^2)(N+1)^2':>18} {'ratio':>8}")
        print(f"  {'-' * 48}")
        for N in case.default_N:
            built = case.build(N)
            theo = (4.0 / np.pi**2) * (N + 1)**2
            print(f"  {N:>5} {built.kappa:>12.4f} {theo:>18.4f} "
                  f"{built.kappa / theo:>8.4f}")
    print(f"\n  ratio != 1 is expected and diagnostic, not an error: cases "
          f"built on the plain\n  N x N TST matrix (every poisson_1d_* case "
          f"and het_1d_3a/3b) match the theoretical\n  O(N^2) scaling exactly; "
          f"het_1d_3c_neumann's Neumann row and the four\n  HETConfig-scaled "
          f"HET cases do not, because they are not that matrix.")


# -- Main ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="poisson_1d_fS_hom",
                    help="Case name from core.cases.available(dim=1), or "
                         "'all' (default: poisson_1d_fS_hom).")
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--inner", default="thomas",
                    help=f"one of {available_inner()} or 'all'")
    ap.add_argument("-I", "--inner-opt", action="append", metavar="[SOLVER.]KEY=VAL",
                    help="inner solver option, e.g. -I max_degree=300 or "
                         "-I qsvt.max_degree=300. Validated against the "
                         "registry; unknown keys are an error, not ignored.")
    ap.add_argument("--list-options", action="store_true",
                    help="print every tunable inner solver parameter")
    ap.add_argument("--list-cases", action="store_true",
                    help="print every registered 1D case")
    ap.add_argument("--kappa-table", action="store_true",
                    help="print kappa(N) vs the plain-TST theoretical scaling")
    ap.add_argument("--dump", action="store_true",
                    help="print node-by-node solution and pointwise error")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.list_options:
        print("\n=== INNER SOLVER OPTIONS ===\n")
        print(describe_inner())
        return

    if args.list_cases:
        for name in cases.available(dim=1):
            print(cases.describe(name))
            print()
        return

    if args.kappa_table:
        kappa_table()
        return

    if args.N < 2 or (args.N & (args.N - 1)):
        raise SystemExit(f"N must be a power of two, got {args.N}")

    inners = list(QUANTUM) if args.inner == "all" else [args.inner]
    requested = list(inners)
    if "thomas" not in inners:
        inners = ["thomas"] + inners          # always keep the reference

    raw = parse_kv(args.inner_opt, "inner-opt")
    inner_opts = {k: v for k, v in raw.items() if isinstance(v, dict)}
    flat = {k: v for k, v in raw.items() if not isinstance(v, dict)}
    if flat:
        if len(requested) != 1:
            raise SystemExit(
                f"-I {list(flat)[0]}=... is ambiguous with --inner {args.inner}. "
                f"Use the namespaced form, e.g. -I qsvt.{list(flat)[0]}=...")
        inner_opts.setdefault(requested[0], {}).update(flat)
    unknown = set(inner_opts) - set(available_inner())
    if unknown:
        raise SystemExit(f"-I refers to unknown solver(s) {sorted(unknown)}. "
                         f"Available: {', '.join(available_inner())}")

    case_names = cases.available(dim=1) if args.case == "all" else [args.case]

    print(f"\n{BOLD}{'=' * 88}{RESET}")
    print(f"{BOLD}  1D SOLVER DEBUG TOOL{RESET}")
    print(f"{BOLD}  cases={case_names}  N={args.N}  inner={inners}{RESET}")
    if inner_opts:
        print(f"{BOLD}  inner options: {inner_opts}{RESET}")
    print(f"{BOLD}{'=' * 88}{RESET}")

    for name in case_names:
        run_case(name, args.N, inners, inner_opts, dump=args.dump, verbose=args.verbose)
    print()


if __name__ == "__main__":
    main()
