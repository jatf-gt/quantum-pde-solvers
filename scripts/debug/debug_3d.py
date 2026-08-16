#!/usr/bin/env python3
"""
debug_3d.py
===========
Debug and benchmark tool for the 3-D solver (solvers/outer/multigrid.py).

Renamed from debug_outer_3d.py and brought level with debug_2d.py: cases now
come from the canonical registry in core/cases.py, the noise/polish studies
and the "saving versus SOR" summary are shared via benchmark/diagnostics.py
rather than existing only on the 2-D side, and --plot and --criterion are now
accepted here too.

No new quantum solver was needed for 3-D. A 3-D problem decomposes into the
same tridiagonal strips as a 2-D one, so hhl_1d / vqls_1d / qsvt_1d are used
unmodified, on log2(N) qubits, with the existing TST block encoding.

Cases
-----
`--case` accepts the short aliases `cube` (poisson_3d_triple_sin_cube), `het`
(het_3d_mms_spt100) and `slab` (het_3d_slab_m4) used by the original tool, or
any full name from `core.cases.available(dim=3)` - the registry adds a
BC-driven Laplace check, a two-Gaussian cube, a high-wavenumber eigenmode, the
m=2 rotating spoke, and the SPT-100 300 V discharge case beyond the three this
tool originally shipped. The last of those has no closed form (its Thomas
solve is the reference), so it reports "n/a" in the "vs exact%" column.

Usage
-----
    python scripts/debug_3d.py --case cube --N 16
    python scripts/debug_3d.py --case het  --N 16 --inner qsvt
    python scripts/debug_3d.py --case all  --N 8  --inner all
    python scripts/debug_3d.py --hierarchy --case het --N 32
    python scripts/debug_3d.py --convergence-study --case cube
    python scripts/debug_3d.py --N 16 --inner qsvt -I max_degree=300
    python scripts/debug_3d.py --noise-study --N 16
    python scripts/debug_3d.py --polish-study --N 16
    python scripts/debug_3d.py --list-cases

Deliberate remaining asymmetry with debug_2d.py: field plotting. --plot here
writes only the dimension-agnostic convergence/cost figure
(benchmark/diagnostics.py::plot_convergence_and_cost) - a 3-D solution field
has no single natural 2-D rendering the way a 2-D field does, and a
slice-based renderer is future work, not part of this consolidation.
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
from benchmark.diagnostics import (ALPHA, BOLD, RED, RESET, YELLOW, colour,
                                   noise_study, plot_convergence_and_cost,
                                   polish_study, rel_err, savings_summary)
from solvers.outer import (available_inner, available_schemes, build_hierarchy,
                           coerce_scheme_opts, describe_inner, describe_scheme,
                           parse_kv, solve)

OUT_DIR = REPO_ROOT / "results" / "debugging"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Short aliases for the three cases this tool originally shipped with, kept
# so existing invocations keep working.
CASE_ALIASES = {
    "cube": "poisson_3d_triple_sin_cube",
    "het": "het_3d_mms_spt100",
    "slab": "het_3d_slab_m4",
}


def _resolve_case(name: str) -> str:
    return CASE_ALIASES.get(name, name)


def build_case(name: str, N: int):
    """Resolves an alias or registry name and builds it at resolution N."""
    full = _resolve_case(name)
    case = cases.get(full)
    if case.dim != 3:
        raise SystemExit(f"--case {name!r} resolves to {full!r}, which is "
                         f"{case.dim}D, not 3D.")
    built = case.build(N)
    return built.problem, built.exact, full


def _kw_for(scheme: str, tol: float, criterion: str | None = None) -> dict:
    kw = {"tol": tol}
    if scheme in ("sor", "gauss-seidel", "jacobi"):
        kw["max_iter"] = 5000
        kw["criterion"] = criterion or "residual"
    else:
        kw["max_cycles"] = 200
    return kw


def _fmt_pct(e: float, good: float = 1.0, ok: float = 5.0) -> str:
    """Coloured percentage, or 'n/a' when no reference is available."""
    if not np.isfinite(e):
        return f"{'n/a':>10}"
    return f"{colour(e, good, ok)}{e:>9.3f}%{RESET}"


# -- Hierarchy inspection ------------------------------------------------------

def show_hierarchy(case: str, N: int) -> None:
    prob, _, tag = build_case(case, N)
    levels = build_hierarchy(prob)
    print(f"\n{BOLD}  3-D MULTIGRID HIERARCHY   case={tag}  N={N}{RESET}")
    print(f"  {'level':>5} {'shape':>16} {'h (mm)':>26} {'h_max/h_min':>12} "
          f"{'kappa':>7} {'qubits':>7}")
    print(f"  {'-' * 82}")
    for i, lv in enumerate(levels):
        p = lv.problem
        hs = [h * 1e3 for h in p.spacings]
        aniso = max(p.spacings) / min(p.spacings)
        nq = int(np.ceil(np.log2(p.shape[0])))
        shape = "x".join(str(n) for n in p.shape)
        print(f"  {i:>5} {shape:>16} "
              f"({hs[0]:7.3f},{hs[1]:7.3f},{hs[2]:7.3f}) {aniso:>12.2f} "
              f"{p.kappa_row():>7.4f} {nq:>7}")

    print(f"\n  Axis 0 is the strip direction: each strip is an N0 x N0 TST")
    print(f"  system on log2(N0) qubits, identical in form to the 1-D case, so")
    print(f"  the existing block encoding and quantum solvers apply unchanged.")
    print(f"  kappa stays near 2 (the 3-D asymptote) rather than the 2-D value")
    print(f"  of 3, because both transverse directions add to the diagonal.")
    fixed = [i for i in range(3)
             if len({lv.problem.shape[i] for lv in levels}) == 1 and len(levels) > 1]
    if fixed:
        print(f"\n  {YELLOW}Axes {fixed} are never coarsened: their spacing exceeds")
        print(f"  {levels[0].problem.COARSEN_RATIO}x the finest axis, so they are "
              f"weakly coupled and")
        print(f"  coarsening them would degrade the coarse-grid correction.{RESET}")


# -- Scheme / solver comparison ------------------------------------------------

def run_comparison(case: str, N: int, inners: list[str], schemes: list[str],
                   tol: float, inner_opts: dict, scheme_opts: dict,
                   verbose: bool, criterion: str | None = None):
    prob, phi_exact, tag = build_case(case, N)
    shape = "x".join(str(n) for n in prob.shape)
    total = int(np.prod(prob.shape))

    print(f"\n{BOLD}{'=' * 92}{RESET}")
    print(f"{BOLD}  CASE {tag.upper()}   grid {shape} = {total:,} unknowns   "
          f"kappa(A_line)={prob.kappa_row():.4f}{RESET}")
    print(f"{BOLD}  h = ({', '.join(f'{h*1e3:.3f}' for h in prob.spacings)}) mm   "
          f"periodic={prob.periodic}   tol={tol:.0e}{RESET}")
    print(f"{BOLD}{'=' * 92}{RESET}")

    levels = build_hierarchy(prob)
    print("  hierarchy: " + " -> ".join(
        "x".join(str(n) for n in lv.problem.shape) for lv in levels))

    ref, rows = None, []
    for inner in inners:
        for scheme in schemes:
            kw = _kw_for(scheme, tol, criterion)
            kw.update(scheme_opts or {})
            io = (inner_opts or {}).get(inner, {})
            t0 = time.perf_counter()
            try:
                res = solve(prob, inner=inner, scheme=scheme,
                            inner_options=io, **kw)
            except Exception as exc:
                print(f"  {RED}[FAIL] {inner}/{scheme}: {exc}{RESET}")
                continue
            wall = time.perf_counter() - t0
            if ref is None and inner == "thomas":
                ref = res.u
            e_ex = rel_err(res.u, phi_exact) if phi_exact is not None else float("nan")
            rows.append((inner, scheme, res, e_ex,
                         rel_err(res.u, ref) if ref is not None else float("nan"),
                         wall))
            if verbose:
                print(f"    {inner}/{scheme}: {res}")

    print(f"\n  {'inner':<9} {'scheme':<13} {'outer':>6} {'solves':>9} "
          f"{'w.cost':>9} {'rho':>6} {'vs exact%':>10} {'vs Thomas%':>11} "
          f"{'time s':>8} {'stop':>12}")
    print(f"  {'-' * 90}")
    for inner, scheme, res, e_ex, e_ref, wall in rows:
        wc = res.work.weighted_cost(ALPHA.get(inner, 1.0))
        print(f"  {inner:<9} {scheme:<13} {res.n_outer:>6} {res.work.total:>9} "
              f"{wc:>9.0f} {res.convergence_factor:>6.3f} "
              f"{_fmt_pct(e_ex)} {_fmt_pct(e_ref, 0.5, 2.0)} "
              f"{wall:>8.2f} {res.stop_reason:>12}")

    savings_summary(rows, header="Saving versus line-SOR")
    return rows


# -- Grid-independence / discretisation study ----------------------------------

def run_convergence_study(case: str, tol: float) -> None:
    """
    Two things at once: that the discretisation is second order (the MMS is
    correct), and that the multigrid cycle count does not grow with N.
    """
    full = _resolve_case(case)
    reg_case = cases.get(full)
    if reg_case.reference not in ("analytical", "manufactured"):
        raise SystemExit(
            f"--convergence-study needs a case with a known exact solution; "
            f"{full!r} uses reference={reg_case.reference!r}. Try cube, het, "
            f"slab, poisson_3d_two_gaussian_cube or het_3d_rotating_spoke.")

    print(f"\n{BOLD}{'=' * 84}{RESET}")
    print(f"{BOLD}  3-D CONVERGENCE STUDY   case={case}{RESET}")
    print(f"{BOLD}{'=' * 84}{RESET}")
    print(f"  {'N':>5} {'unknowns':>11} {'err%':>9} {'order':>7} "
          f"{'SOR its':>8} {'FMG cyc':>8} {'FMG rho':>8} {'solves gain':>12}")
    print(f"  {'-' * 76}")
    prev_err = prev_N = None
    for N in (8, 16, 32):
        prob, phi, _ = build_case(case, N)
        a = solve(prob, inner="thomas", scheme="sor", tol=tol, max_iter=5000)
        b = solve(prob, inner="thomas", scheme="fmg", tol=tol, max_cycles=200)
        err = rel_err(b.u, phi)
        order = ""
        if prev_err is not None and err > 0:
            order = f"{np.log(prev_err / err) / np.log(N / prev_N):7.2f}"
        print(f"  {N:>5} {int(np.prod(prob.shape)):>11,} {err:>9.4f} {order:>7} "
              f"{a.n_outer:>8} {b.n_outer:>8} {b.convergence_factor:>8.3f} "
              f"{a.work.total / max(b.work.total, 1):>11.1f}x")
        prev_err, prev_N = err, N
    print(f"\n  'order' should approach 2 for a correct second-order scheme.")
    print(f"  'FMG cyc' should stay flat while 'SOR its' roughly doubles per")
    print(f"  refinement - that flatness is the entire point of multigrid.")
    print(f"\n  Note the crossover: in 3-D each sweep costs N^2 strip solves, so")
    print(f"  multigrid only pays off above N~12 (isotropic) or N~24 (HET, whose")
    print(f"  anisotropy forces a shallower hierarchy).")


# -- Main ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="cube",
                    help="Case name: 'cube', 'het', 'slab', 'all', or any name "
                         "from core.cases.available(dim=3) (default: cube).")
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--inner", default="thomas",
                    help=f"one of {available_inner()} or 'all'")
    ap.add_argument("--scheme", default="all",
                    help=f"one of {available_schemes()} or 'all'")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--criterion", default=None, choices=["residual", "delta"],
                    help="stopping test for stationary schemes")
    ap.add_argument("-I", "--inner-opt", action="append",
                    metavar="[SOLVER.]KEY=VAL")
    ap.add_argument("-S", "--scheme-opt", action="append", metavar="KEY=VAL")
    ap.add_argument("--list-options", action="store_true")
    ap.add_argument("--list-cases", action="store_true",
                    help="print every registered 3D case")
    ap.add_argument("--hierarchy", action="store_true")
    ap.add_argument("--convergence-study", action="store_true")
    ap.add_argument("--noise-study", action="store_true")
    ap.add_argument("--polish-study", action="store_true")
    ap.add_argument("--plot", action="store_true",
                    help="write the convergence/cost figure (no field plot; "
                         "see the module docstring)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.list_options:
        print("\n=== INNER SOLVER OPTIONS ===\n"); print(describe_inner())
        print("\n=== OUTER SCHEME OPTIONS ===\n"); print(describe_scheme())
        return

    if args.list_cases:
        for name in cases.available(dim=3):
            print(cases.describe(name))
            print()
        return

    if args.N < 4 or (args.N & (args.N - 1)):
        raise SystemExit(f"N must be a power of two and at least 4, got {args.N}")

    all_cases = list(CASE_ALIASES) if args.case == "all" else [args.case]

    if args.hierarchy:
        for c in all_cases:
            show_hierarchy(c, args.N)
        return

    if args.convergence_study:
        for c in all_cases:
            run_convergence_study(c, args.tol)
        return

    if args.noise_study:
        for c in all_cases:
            prob, _, tag = build_case(c, args.N)
            noise_study(prob, tag, args.N, args.tol)
        return

    if args.polish_study:
        for c in all_cases:
            prob, _, tag = build_case(c, args.N)
            polish_study(prob, tag, args.N)
        return

    inners = (["hhl", "vqls", "qsvt"] if args.inner == "all" else [args.inner])
    requested = list(inners)
    if "thomas" not in inners:
        inners = ["thomas"] + inners

    raw = parse_kv(args.inner_opt, "--inner-opt")
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
        raise SystemExit(f"-I refers to unknown solver(s) {sorted(unknown)}")

    scheme_opts = coerce_scheme_opts(parse_kv(args.scheme_opt, "--scheme-opt"))
    schemes = available_schemes() if args.scheme == "all" else [args.scheme]

    print(f"\n{BOLD}{'=' * 92}{RESET}")
    print(f"{BOLD}  3D OUTER-SCHEME DEBUG TOOL{RESET}")
    print(f"{BOLD}  cases={all_cases}  N={args.N}  inner={inners}  schemes={schemes}{RESET}")
    if inner_opts:
        print(f"{BOLD}  inner options : {inner_opts}{RESET}")
    if scheme_opts:
        print(f"{BOLD}  scheme options: {scheme_opts}{RESET}")
    print(f"{BOLD}{'=' * 92}{RESET}")

    for c in all_cases:
        rows = run_comparison(c, args.N, inners, schemes, args.tol,
                              inner_opts, scheme_opts, args.verbose, args.criterion)
        if args.plot and rows:
            _, _, tag = build_case(c, args.N)
            plot_convergence_and_cost(rows, tag, args.N, OUT_DIR)
    print()


if __name__ == "__main__":
    main()
