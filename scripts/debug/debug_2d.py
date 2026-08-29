#!/usr/bin/env python3
"""
debug_2d.py
===========
Debug and benchmark tool for the 2-D outer-iteration layer (solvers/outer).

Renamed from debug_outer_2d.py. Cases now come from the canonical registry in
core/cases.py rather than being declared locally, and the machinery shared
with debug_3d.py - the comparison metric, the cost-weighting table, the
"saving versus SOR" summary, the noise and polish studies - lives in
benchmark/diagnostics.py. This file only builds cases, prints the
dimension-specific comparison table, and wires the CLI together.

Usage
-----
    # scheme comparison with the classical inner solver (fast, no quantum)
    python scripts/debug/debug_2d.py --case square --N 64

    # reproduce the original validated line-Jacobi behaviour
    python scripts/debug/debug_2d.py --case square --N 8 --scheme jacobi \
           --criterion delta --tol 1e-6

    # one quantum solver, all schemes
    python scripts/debug/debug_2d.py --case het --N 8 --inner hhl

    # everything
    python scripts/debug/debug_2d.py --case all --N 8 --inner all --plot

    # tune the inner solvers: -I applies to the selected solver,
    # -I solver.key=value targets one solver in a multi-solver sweep
    python scripts/debug/debug_2d.py --N 32 --inner qsvt -I max_degree=300
    python scripts/debug/debug_2d.py --N 32 --inner all \
           -I qsvt.max_degree=300 -I hhl.epsilon=0.05 -I vqls.n_restarts=2

    # tune the outer scheme
    python scripts/debug/debug_2d.py --N 64 --scheme fmg -S nu1=2 -S n_coarse=8

    # list every tunable parameter, or every registered 2-D case
    python scripts/debug/debug_2d.py --list-options
    python scripts/debug/debug_2d.py --list-cases

    # how much inner-solver error each scheme tolerates (no quantum needed)
    python scripts/debug/debug_2d.py --noise-study --N 32

    # does finishing a multigrid solve with SOR/Jacobi help?  (it does not)
    python scripts/debug/debug_2d.py --polish-study --N 64

    # check the hierarchy that would be built
    python scripts/debug/debug_2d.py --hierarchy --N 64

    # confirm second-order discretisation and grid-independent FMG cycles
    python scripts/debug/debug_2d.py --convergence-study --case square

`--case` accepts the short aliases `square` (poisson_2d_sin_pi) and `het`
(het_2d_mms_spt100) used by the original tool, or any full name from
`core.cases.available(dim=2)` - the registry adds two more generic Poisson
cases (a fine-mesh-referenced sinusoid, a two-Gaussian PlasmaNet benchmark, a
single Fourier eigenmode) and two more HET cases (a 20 V sinusoid, the
Boeuf-Garrigues charge density) beyond the two this tool originally shipped.
Cases without a closed-form or manufactured solution report "n/a" in the
"vs exact%" column; the "vs Thomas%" column still applies to every case.
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
from benchmark.diagnostics import (ALPHA, BOLD, GREEN, RED, RESET, YELLOW,
                                   colour, noise_study, plot_convergence_and_cost,
                                   polish_study, rel_err, savings_summary)
from solvers.outer import (available_inner, available_schemes, build_hierarchy,
                           coerce_scheme_opts, describe_inner, describe_scheme,
                           parse_kv, solve)

OUT_DIR = REPO_ROOT / "results" / "debugging"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Short aliases for the two cases this tool originally shipped with, kept so
# existing invocations and scripts/tutorial.py's examples keep working.
CASE_ALIASES = {"square": "poisson_2d_sin_pi", "het": "het_2d_mms_spt100"}


def _resolve_case(name: str) -> str:
    return CASE_ALIASES.get(name, name)


def build_case(name: str, N: int):
    """Resolves an alias or registry name and builds it at resolution N."""
    full = _resolve_case(name)
    case = cases.get(full)
    if case.dim != 2:
        raise SystemExit(f"--case {name!r} resolves to {full!r}, which is "
                         f"{case.dim}D, not 2D.")
    built = case.build(N)
    return built.problem, built.exact, full


def _kw_for(scheme: str, tol: float, criterion: str | None) -> dict:
    kw = {"tol": tol}
    if scheme in ("sor", "gauss-seidel", "jacobi"):
        kw["max_iter"] = 20000
        # Force the residual criterion in comparison tables to maintain
        # consistency across schemes. The "jacobi" scheme
        # otherwise defaults to the original delta test, under which the same
        # numeric tol is far looser and the row is not comparable.
        kw["criterion"] = criterion or "residual"
    else:
        kw["max_cycles"] = 200
    return kw


def _fmt_pct(e: float, good: float = 1.0, ok: float = 5.0) -> str:
    """Coloured percentage, or 'n/a' when no reference is available."""
    if not np.isfinite(e):
        return f"{'n/a':>10}"
    return f"{colour(e, good, ok)}{e:>9.3f}%{RESET}"


# -- Scheme comparison ---------------------------------------------------------

def run_comparison(case: str, N: int, inners: list[str], schemes: list[str],
                   tol: float, verbose: bool, criterion: str | None = None,
                   inner_opts: dict | None = None,
                   scheme_opts: dict | None = None):
    prob, u_exact, tag = build_case(case, N)

    print(f"\n{BOLD}{'=' * 86}{RESET}")
    print(f"{BOLD}  CASE {tag.upper()}   grid {prob.shape[0]}x{prob.shape[1]}   "
          f"dx={prob.dx:.4e}  dy={prob.dy:.4e}  kappa(A_row)={prob.kappa_row():.4f}{RESET}")
    if u_exact is not None:
        print(f"{BOLD}  max|u_exact| = {np.max(np.abs(u_exact)):.6g}   "
              f"algebraic tol = {tol:.0e}{RESET}")
    else:
        print(f"{BOLD}  no closed-form/manufactured reference for this case - "
              f"'vs exact%' reports n/a   algebraic tol = {tol:.0e}{RESET}")
    print(f"{BOLD}{'=' * 86}{RESET}")

    levels = build_hierarchy(prob)
    print(f"  hierarchy: " + " -> ".join(
        f"{lv.problem.shape[0]}x{lv.problem.shape[1]}(k={lv.problem.kappa_row():.2f})"
        for lv in levels))

    ref = None
    rows = []
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
            e_exact = rel_err(res.u, u_exact) if u_exact is not None else float("nan")
            e_ref = rel_err(res.u, ref) if ref is not None else float("nan")
            rows.append((inner, scheme, res, e_exact, e_ref, wall))

            if verbose:
                print(f"    {inner}/{scheme}: {res}")

    # -- table -----------------------------------------------------------------
    print(f"\n  {'inner':<9} {'scheme':<13} {'outer':>6} {'solves':>8} "
          f"{'w.cost':>8} {'rho':>6} {'vs exact%':>10} {'vs Thomas%':>11} "
          f"{'time s':>8} {'stop':>16}")
    print(f"  {'-' * 84}")
    for inner, scheme, res, e_exact, e_ref, wall in rows:
        a = ALPHA.get(inner, 1.0)
        wc = res.work.weighted_cost(a)
        print(f"  {inner:<9} {scheme:<13} {res.n_outer:>6} {res.work.total:>8} "
              f"{wc:>8.0f} {res.convergence_factor:>6.3f} "
              f"{_fmt_pct(e_exact)} {_fmt_pct(e_ref, 0.5, 2.0)} "
              f"{wall:>8.2f} {res.stop_reason:>16}")

    savings_summary(rows)
    return rows


# -- Hierarchy inspection ------------------------------------------------------

def show_hierarchy(case: str, N: int):
    prob, _, tag = build_case(case, N)
    levels = build_hierarchy(prob)
    print(f"\n{BOLD}  MULTIGRID HIERARCHY  case={tag}  N={N}{RESET}")
    print(f"  {'level':>5} {'shape':>12} {'dx':>11} {'dy':>11} {'dx/dy':>7} "
          f"{'kappa':>7} {'qubits':>7}")
    print(f"  {'-' * 66}")
    for i, lv in enumerate(levels):
        p = lv.problem
        nq = int(np.ceil(np.log2(p.shape[0])))
        print(f"  {i:>5} {p.shape[0]:>5}x{p.shape[1]:<6} {p.dx:>11.4e} "
              f"{p.dy:>11.4e} {p.dx / p.dy:>7.3f} {p.kappa_row():>7.3f} {nq:>7}")
    print(f"\n  kappa should stay ~2-3 on every level: this is what keeps the")
    print(f"  QSVT polynomial degree and the HHL clock register constant with depth.")
    print(f"  A level whose dx/dy departs far from 1 weakens the line smoother;")
    print(f"  keep Nx/Ny close to Lx/Ly.")


# -- Order-of-accuracy study ----------------------------------------------------

def run_convergence_study(case: str, tol: float) -> None:
    """
    Confirms the discretisation is second order and that FMG's cycle count
    does not grow with N.

    The 2-D counterpart of debug_3d.py's --convergence-study; added here to
    close that asymmetry now that both drivers share build_hierarchy and the
    same solve() interface. Needs a case with a known exact or manufactured
    solution, since the order estimate compares against it directly.
    """
    full = _resolve_case(case)
    reg_case = cases.get(full)
    if reg_case.reference not in ("analytical", "manufactured"):
        raise SystemExit(
            f"--convergence-study needs a case with a known exact solution; "
            f"{full!r} uses reference={reg_case.reference!r}. Try square, het, "
            f"poisson_2d_single_mode_n1m1 or het_2d_sin_meeting_report.")

    print(f"\n{BOLD}{'=' * 84}{RESET}")
    print(f"{BOLD}  2-D CONVERGENCE STUDY   case={full}{RESET}")
    print(f"{BOLD}{'=' * 84}{RESET}")
    print(f"  {'N':>5} {'unknowns':>11} {'err%':>9} {'order':>7} "
          f"{'SOR its':>8} {'FMG cyc':>8} {'FMG rho':>8} {'solves gain':>12}")
    print(f"  {'-' * 76}")
    prev_err = prev_N = None
    for N in (8, 16, 32, 64):
        prob, u_exact, _ = build_case(case, N)
        a = solve(prob, inner="thomas", scheme="sor", tol=tol, max_iter=20000,
                  criterion="residual")
        b = solve(prob, inner="thomas", scheme="fmg", tol=tol, max_cycles=200)
        err = rel_err(b.u, u_exact)
        order = ""
        if prev_err is not None and err > 0:
            order = f"{np.log(prev_err / err) / np.log(N / prev_N):7.2f}"
        print(f"  {N:>5} {N * N:>11,} {err:>9.4f} {order:>7} "
              f"{a.n_outer:>8} {b.n_outer:>8} {b.convergence_factor:>8.3f} "
              f"{a.work.total / max(b.work.total, 1):>11.1f}x")
        prev_err, prev_N = err, N
    print(f"\n  'order' should approach 2 for a correct second-order scheme.")
    print(f"  'FMG cyc' should stay flat while 'SOR its' roughly doubles per")
    print(f"  refinement - that flatness is the entire point of multigrid.")


# -- Field plotting (2-D specific) ---------------------------------------------

def plot_fields(rows, prob, u_exact, case: str, N: int) -> None:
    """Solution and pointwise-error fields, one column per (inner, scheme)."""
    if u_exact is None or not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print(f"  {YELLOW}matplotlib unavailable - skipping field plot{RESET}")
        return

    x, y = prob.grid()                      # (Nx, Ny) meshgrid, 'ij' indexing
    labels = [f"{inner}/{scheme}" for inner, scheme, *_ in rows]
    fields = [res.u for _, _, res, *_ in rows]

    n_cols = 1 + len(fields)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7), squeeze=False)
    vmin, vmax = float(u_exact.min()), float(u_exact.max())

    im = axes[0, 0].pcolormesh(x, y, u_exact, cmap="RdBu_r",
                               vmin=vmin, vmax=vmax, shading="auto")
    axes[0, 0].set_title("Exact", fontweight="bold")
    axes[0, 0].set_aspect("equal")
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
    axes[1, 0].axis("off")

    for ci, (label, phi) in enumerate(zip(labels, fields), start=1):
        im = axes[0, ci].pcolormesh(x, y, phi, cmap="RdBu_r",
                                    vmin=vmin, vmax=vmax, shading="auto")
        axes[0, ci].set_title(label, fontweight="bold")
        axes[0, ci].set_aspect("equal")
        plt.colorbar(im, ax=axes[0, ci], shrink=0.8)

        err = phi - u_exact
        abs_max = max(float(np.abs(err).max()), 1e-12)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
        im2 = axes[1, ci].pcolormesh(x, y, err, cmap="seismic",
                                     norm=norm, shading="auto")
        axes[1, ci].set_title(f"Error ({rel_err(phi, u_exact):.3f}%)")
        axes[1, ci].set_aspect("equal")
        plt.colorbar(im2, ax=axes[1, ci], shrink=0.8)

    fig.suptitle(f"2D solution fields - {case}, N={N}", fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / f"debug_{case}_N{N}_fields.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {GREEN}saved {out}{RESET}")


# -- Main ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="square",
                    help="Case name: 'square', 'het', 'all', or any name from "
                         "core.cases.available(dim=2) (default: square).")
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--inner", default="thomas",
                    help=f"one of {available_inner()} or 'all'")
    ap.add_argument("--scheme", default="all",
                    help=f"one of {available_schemes()} or 'all'")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="algebraic residual tolerance (default 1e-4: one order "
                         "below typical discretisation error)")
    ap.add_argument("--criterion", default=None, choices=["residual", "delta"],
                    help="stopping test for stationary schemes; 'delta' "
                         "reproduces the original convergence check")
    ap.add_argument("--noise-study", action="store_true")
    ap.add_argument("--polish-study", action="store_true")
    ap.add_argument("--hierarchy", action="store_true")
    ap.add_argument("--convergence-study", action="store_true",
                    help="order-of-accuracy and FMG grid-independence check")
    ap.add_argument("-I", "--inner-opt", action="append", metavar="[SOLVER.]KEY=VAL",
                    help="inner solver option, e.g. -I max_degree=300 or "
                         "-I qsvt.max_degree=300. Validated against the "
                         "registry; unknown keys are an error, not ignored.")
    ap.add_argument("-S", "--scheme-opt", action="append", metavar="KEY=VAL",
                    help="outer scheme option, e.g. -S nu1=2 -S n_coarse=8")
    ap.add_argument("--list-options", action="store_true",
                    help="print every tunable inner and scheme parameter")
    ap.add_argument("--list-cases", action="store_true",
                    help="print every registered 2D case")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.list_options:
        print("\n=== INNER SOLVER OPTIONS ===\n")
        print(describe_inner())
        print("\n=== OUTER SCHEME OPTIONS ===\n")
        print(describe_scheme())
        return

    if args.list_cases:
        for name in cases.available(dim=2):
            print(cases.describe(name))
            print()
        return

    if args.N < 4 or (args.N & (args.N - 1)):
        raise SystemExit(f"N must be a power of two and at least 4, got {args.N}")

    raw_inner = parse_kv(args.inner_opt, "inner-opt")
    scheme_opts = coerce_scheme_opts(parse_kv(args.scheme_opt, "scheme-opt"))

    all_cases = ["square", "het"] if args.case == "all" else [args.case]

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
        inners = ["thomas"] + inners          # always keep the reference

    # Normalise inner options into {solver: {key: value}}.  Bare "key=value"
    # targets the solver named by --inner; it must never leak onto the
    # automatically added Thomas reference, which accepts no options.
    inner_opts: dict = {}
    flat = {k: v for k, v in raw_inner.items() if not isinstance(v, dict)}
    for k, v in raw_inner.items():
        if isinstance(v, dict):
            inner_opts.setdefault(k, {}).update(v)
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
    schemes = available_schemes() if args.scheme == "all" else [args.scheme]

    print(f"\n{BOLD}{'=' * 86}{RESET}")
    print(f"{BOLD}  2D OUTER-SCHEME DEBUG TOOL{RESET}")
    print(f"{BOLD}  cases={all_cases}  N={args.N}  inner={inners}  schemes={schemes}{RESET}")
    if inner_opts:
        print(f"{BOLD}  inner options : {inner_opts}{RESET}")
    if scheme_opts:
        print(f"{BOLD}  scheme options: {scheme_opts}{RESET}")
    print(f"{BOLD}{'=' * 86}{RESET}")

    for c in all_cases:
        rows = run_comparison(c, args.N, inners, schemes, args.tol,
                              args.verbose, args.criterion,
                              inner_opts, scheme_opts)
        if args.plot and rows:
            prob, u_exact, tag = build_case(c, args.N)
            plot_convergence_and_cost(rows, tag, args.N, OUT_DIR)
            plot_fields(rows, prob, u_exact, tag, args.N)
    print()


if __name__ == "__main__":
    main()
