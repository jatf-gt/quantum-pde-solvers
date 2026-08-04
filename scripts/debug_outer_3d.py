#!/usr/bin/env python3
"""
debug_outer_3d.py
=================
Debug and benchmark tool for the 3-D solver (solvers/outer/poisson3d.py).

The 3-D counterpart of debug_outer_2d.py, and deliberately its near-twin:
the schemes, the inner-solver registry and the work accounting are shared,
so this file only declares cases and prints tables.

No new quantum solver was needed for 3-D.  A 3-D problem decomposes into the
same tridiagonal strips as a 2-D one, so hhl_1d / vqls_1d / qsvt_1d are used
unmodified, on log2(N) qubits, with the existing TST block encoding.

Cases
-----
    cube      Triple-sin MMS on the unit cube.
              phi = sin(pi x) sin(pi y) sin(pi z),  f = -3 pi^2 phi
              The standard 3-D Poisson verification case.

    het       SPT-100 channel unwrapped to a Cartesian slab, azimuthally
              periodic - the geometry used for axial-azimuthal HET studies,
              where the annulus is thin enough (dr = 15 mm against a mean
              circumference of 267 mm) to be treated as a slab.
                  axis 0 : axial      z in [0, 25 mm],  Dirichlet (anode/cathode)
                  axis 1 : radial     r in [0, 15 mm],  Dirichlet (walls)
                  axis 2 : azimuthal  s in [0, 2 pi r_mean], PERIODIC
              MMS: phi = phi0 sin(pi z/Lz) sin(pi r/Lr) cos(2 pi m s/Ls),
                   f   = -phi0 pi^2 (1/Lz^2 + 1/Lr^2 + 4 m^2/Ls^2) * profile
              Exactly periodic in s, so the periodic transfer operators and
              the wrapped stencil are both genuinely exercised.

    slab      Same channel with an azimuthal mode number m=4, to check that
              convergence does not depend on the azimuthal wavenumber.

Usage
-----
    python scripts/debug_outer_3d.py --case cube --N 16
    python scripts/debug_outer_3d.py --case het  --N 16 --inner qsvt
    python scripts/debug_outer_3d.py --case all  --N 8  --inner all
    python scripts/debug_outer_3d.py --hierarchy --case het --N 32
    python scripts/debug_outer_3d.py --convergence-study --case cube
    python scripts/debug_outer_3d.py --N 16 --inner qsvt -I max_degree=300

Author : Juan Antonio Trobajo Flecha
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from problems.poisson_line_3d import PoissonLine3D
from solvers.outer import (available_inner, available_schemes, build_hierarchy, 
                           describe_inner, describe_scheme, solve)

OUT_DIR = REPO_ROOT / "results" / "debugging"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_G, _Y, _R = "\033[92m", "\033[93m", "\033[91m"
_C, _B, _X = "\033[96m", "\033[1m", "\033[0m"

# Per-strip-solve cost exponents, as in the 2-D tool.
ALPHA = {"hhl": 2.35, "vqls": 1.29, "qsvt": 0.60,
         "thomas": 1.0, "perturbed": 1.0}

# SPT-100 channel geometry.
HET_Lz = 0.025                       # axial length, m
HET_R_IN, HET_R_OUT = 0.035, 0.050   # channel inner / outer radius, m
HET_LR = HET_R_OUT - HET_R_IN        # channel width, m
HET_R_MEAN = 0.5 * (HET_R_IN + HET_R_OUT)
HET_LS = 2.0 * np.pi * HET_R_MEAN    # mean circumference, m
HET_PHI0 = 300.0                     # discharge voltage scale, V


# =============================================================================
#  Cases
# =============================================================================

def case_cube(N: int):
    """Triple-sin MMS on the unit cube, all Dirichlet."""
    h = 1.0 / (N + 1)
    p = np.arange(1, N + 1) * h
    X, Y, Z = np.meshgrid(p, p, p, indexing="ij")
    phi = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    prob = PoissonLine3D(-3.0 * np.pi**2 * phi, lengths=(1.0, 1.0, 1.0))
    return prob, phi, "cube"


def _case_het(N: int, m: int, tag: str):
    """HET channel unwrapped to a slab; azimuthally periodic MMS."""
    dz, dr = HET_Lz / (N + 1), HET_LR / (N + 1)
    ds = HET_LS / N                       # periodic axis has no boundary node
    Zg, Rg, Sg = np.meshgrid(np.arange(1, N + 1) * dz,
                             np.arange(1, N + 1) * dr,
                             np.arange(N) * ds, indexing="ij")
    profile = (np.sin(np.pi * Zg / HET_Lz)
               * np.sin(np.pi * Rg / HET_LR)
               * np.cos(2.0 * np.pi * m * Sg / HET_LS))
    phi = HET_PHI0 * profile
    lap = -HET_PHI0 * np.pi**2 * (1.0 / HET_Lz**2 + 1.0 / HET_LR**2
                                  + 4.0 * m**2 / HET_LS**2)
    prob = PoissonLine3D(lap * profile,
                         lengths=(HET_Lz, HET_LR, HET_LS),
                         periodic=(False, False, True))
    return prob, phi, tag


def case_het(N: int):
    return _case_het(N, m=1, tag="het")


def case_slab(N: int):
    return _case_het(N, m=4, tag="slab_m4")


CASES = {"cube": case_cube, "het": case_het, "slab": case_slab}


# =============================================================================
#  Helpers
# =============================================================================

def rel_err(u, ref) -> float:
    """Max absolute error normalised by the reference amplitude, per cent."""
    return float(np.max(np.abs(u - ref)) / (np.max(np.abs(ref)) + 1e-300) * 100.0)


def colour(pct, good=1.0, ok=5.0):
    return _G if pct < good else (_Y if pct < ok else _R)


def _kw_for(scheme: str, tol: float) -> dict:
    kw = {"tol": tol}
    if scheme in ("sor", "gauss-seidel", "jacobi"):
        kw["max_iter"] = 5000
        kw["criterion"] = "residual"
    else:
        kw["max_cycles"] = 200
    return kw


def parse_kv(items, flag: str) -> dict:
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"{flag} expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        if "." in key:
            solver, k = key.split(".", 1)
            out.setdefault(solver, {})[k] = value
        else:
            out[key] = value
    return out


def coerce_scheme_opts(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if k == "omega" and v == "optimal":
            out[k] = v
        elif k == "criterion":
            out[k] = v
        elif k in ("symmetric", "fmg"):
            out[k] = str(v).lower() in ("true", "1", "yes", "on")
        elif k in ("tol", "omega"):
            out[k] = float(v)
        else:
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = float(v)
    return out


# =============================================================================
#  Hierarchy inspection
# =============================================================================

def show_hierarchy(case: str, N: int) -> None:
    prob, _, tag = CASES[case](N)
    levels = build_hierarchy(prob)
    print(f"\n{_B}  3-D MULTIGRID HIERARCHY   case={tag}  N={N}{_X}")
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
        print(f"\n  {_Y}Axes {fixed} are never coarsened: their spacing exceeds")
        print(f"  {prob.COARSEN_RATIO}x the finest axis, so they are weakly coupled and")
        print(f"  coarsening them would degrade the coarse-grid correction.{_X}")


# =============================================================================
#  Scheme / solver comparison
# =============================================================================

def run_comparison(case: str, N: int, inners: list[str], schemes: list[str],
                   tol: float, inner_opts: dict, scheme_opts: dict,
                   verbose: bool):
    prob, phi_exact, tag = CASES[case](N)
    shape = "x".join(str(n) for n in prob.shape)
    total = int(np.prod(prob.shape))

    print(f"\n{_B}{'=' * 92}{_X}")
    print(f"{_B}  CASE {tag.upper()}   grid {shape} = {total:,} unknowns   "
          f"kappa(A_line)={prob.kappa_row():.4f}{_X}")
    print(f"{_B}  h = ({', '.join(f'{h*1e3:.3f}' for h in prob.spacings)}) mm   "
          f"periodic={prob.periodic}   tol={tol:.0e}{_X}")
    print(f"{_B}{'=' * 92}{_X}")

    levels = build_hierarchy(prob)
    print("  hierarchy: " + " -> ".join(
        "x".join(str(n) for n in lv.problem.shape) for lv in levels))

    ref, rows = None, []
    for inner in inners:
        for scheme in schemes:
            kw = _kw_for(scheme, tol)
            kw.update(scheme_opts or {})
            io = (inner_opts or {}).get(inner, {})
            t0 = time.perf_counter()
            try:
                res = solve(prob, inner=inner, scheme=scheme,
                            inner_options=io, **kw)
            except Exception as exc:
                import traceback
                print(f"  {_R}[FAIL] {inner}/{scheme}: {exc}{_X}")
                # traceback.print_exc()
                continue
            wall = time.perf_counter() - t0
            if ref is None and inner == "thomas":
                ref = res.u
            rows.append((inner, scheme, res,
                         rel_err(res.u, phi_exact),
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
              f"{colour(e_ex)}{e_ex:>9.3f}%{_X} "
              f"{colour(e_ref, 0.5, 2.0)}{e_ref:>10.3f}%{_X} "
              f"{wall:>8.2f} {res.stop_reason:>12}")

    by_inner: dict = {}
    for inner, scheme, res, *_ in rows:
        by_inner.setdefault(inner, {})[scheme] = res
    print(f"\n  {_B}Saving versus line-SOR{_X}")
    for inner, d in by_inner.items():
        if "sor" not in d:
            continue
        a = ALPHA.get(inner, 1.0)
        s = d["sor"]
        for scheme, r in d.items():
            if scheme == "sor":
                continue
            print(f"    {inner:<9} {scheme:<12} "
                  f"{s.work.total / max(r.work.total, 1):5.1f}x fewer strip solves, "
                  f"{s.work.weighted_cost(a) / max(r.work.weighted_cost(a), 1e-9):5.1f}x "
                  f"lower weighted cost")
    return rows


# =============================================================================
#  Grid-independence / discretisation study
# =============================================================================

def run_convergence_study(case: str, tol: float) -> None:
    """
    Two things at once: that the discretisation is second order (the MMS is
    correct), and that the multigrid cycle count does not grow with N.
    """
    print(f"\n{_B}{'=' * 84}{_X}")
    print(f"{_B}  3-D CONVERGENCE STUDY   case={case}{_X}")
    print(f"{_B}{'=' * 84}{_X}")
    print(f"  {'N':>5} {'unknowns':>11} {'err%':>9} {'order':>7} "
          f"{'SOR its':>8} {'FMG cyc':>8} {'FMG rho':>8} {'solves gain':>12}")
    print(f"  {'-' * 76}")
    prev_err = prev_N = None
    for N in (8, 16, 32):
        prob, phi, _ = CASES[case](N)
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


# =============================================================================
#  Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="cube",
                    choices=list(CASES) + ["all"])
    ap.add_argument("--N", type=int, default=16)
    ap.add_argument("--inner", default="thomas",
                    help=f"one of {available_inner()} or 'all'")
    ap.add_argument("--scheme", default="all",
                    help=f"one of {available_schemes()} or 'all'")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("-I", "--inner-opt", action="append",
                    metavar="[SOLVER.]KEY=VAL")
    ap.add_argument("-S", "--scheme-opt", action="append", metavar="KEY=VAL")
    ap.add_argument("--list-options", action="store_true")
    ap.add_argument("--hierarchy", action="store_true")
    ap.add_argument("--convergence-study", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.list_options:
        print("\n=== INNER SOLVER OPTIONS ===\n"); print(describe_inner())
        print("\n=== OUTER SCHEME OPTIONS ===\n"); print(describe_scheme())
        return

    if args.N < 4 or (args.N & (args.N - 1)):
        raise SystemExit(f"N must be a power of two and at least 4, got {args.N}")

    cases = list(CASES) if args.case == "all" else [args.case]

    if args.hierarchy:
        for c in cases:
            show_hierarchy(c, args.N)
        return

    if args.convergence_study:
        for c in cases:
            run_convergence_study(c, args.tol)
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

    print(f"\n{_B}{'=' * 92}{_X}")
    print(f"{_B}  3D OUTER-SCHEME DEBUG TOOL{_X}")
    print(f"{_B}  cases={cases}  N={args.N}  inner={inners}  schemes={schemes}{_X}")
    if inner_opts:
        print(f"{_B}  inner options : {inner_opts}{_X}")
    if scheme_opts:
        print(f"{_B}  scheme options: {scheme_opts}{_X}")
    print(f"{_B}{'=' * 92}{_X}")

    for c in cases:
        run_comparison(c, args.N, inners, schemes, args.tol,
                       inner_opts, scheme_opts, args.verbose)
    print()


if __name__ == "__main__":
    main()