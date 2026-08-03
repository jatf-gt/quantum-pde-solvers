#!/usr/bin/env python3
"""
debug_outer_2d.py
=================
Debug and benchmark tool for the outer-iteration layer (solvers/outer).

Replaces debug_2d_solvers.py.  It is much shorter because the schemes, the
problems and the inner solvers now live in the package: this file only
declares cases and prints tables.

Usage
-----
    # scheme comparison with the classical inner solver (fast, no quantum)
    python scripts/debug_outer_2d.py --case square --N 64

    # reproduce the original validated line-Jacobi behaviour
    python scripts/debug_outer_2d.py --case square --N 8 --scheme jacobi \
           --criterion delta --tol 1e-6

    # one quantum solver, all schemes
    python scripts/debug_outer_2d.py --case het --N 8 --inner hhl

    # everything
    python scripts/debug_outer_2d.py --case all --N 8 --inner all --plot

    # tune the inner solvers: -I applies to the selected solver,
    # -I solver.key=value targets one solver in a multi-solver sweep
    python scripts/debug_outer_2d.py --N 32 --inner qsvt -I max_degree=300
    python scripts/debug_outer_2d.py --N 32 --inner all \
           -I qsvt.max_degree=300 -I hhl.epsilon=0.05 -I vqls.n_restarts=2

    # tune the outer scheme
    python scripts/debug_outer_2d.py --N 64 --scheme fmg -S nu1=2 -S n_coarse=8

    # list every tunable parameter
    python scripts/debug_outer_2d.py --list-options

    # how much inner-solver error each scheme tolerates (no quantum needed)
    python scripts/debug_outer_2d.py --noise-study --N 32

    # does finishing a multigrid solve with SOR/Jacobi help?  (it does not)
    python scripts/debug_outer_2d.py --polish-study --N 64

    # check the hierarchy that would be built
    python scripts/debug_outer_2d.py --hierarchy --N 64

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

from solvers.outer import (PoissonLine2D, available_inner, available_options,
                           available_schemes, build_hierarchy, describe_inner,
                           describe_scheme, solve, solve_staged)

OUT_DIR = REPO_ROOT / "results" / "debugging"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_G, _Y, _R = "\033[92m", "\033[93m", "\033[91m"
_C, _B, _X = "\033[96m", "\033[1m", "\033[0m"

# Empirical per-strip-solve cost exponents t(n) ~ n^alpha, fitted from the
# N=4 / N=8 statevector timings (HHL 0.267 s -> 1.36 s, VQLS 0.806 -> 1.965,
# QSVT 0.0259 -> 0.0393).  Used to weight coarse-level work correctly.
ALPHA = {"hhl": 2.35, "vqls": 1.29, "qsvt": 0.60, "thomas": 1.0, "perturbed": 1.0}

HET_Lz, HET_Lr, HET_phi0 = 0.025, 0.020, 300.0


# =============================================================================
#  Cases
# =============================================================================

def case_square(N: int):
    """nabla^2 u = sin(pi x) sin(pi y) on [0,1]^2, u = 0 on the boundary."""
    h = 1.0 / (N + 1)
    p = np.arange(1, N + 1) * h
    X, Y = np.meshgrid(p, p, indexing="ij")
    f = np.sin(np.pi * X) * np.sin(np.pi * Y)
    u_exact = -f / (2.0 * np.pi**2)
    return PoissonLine2D(f), u_exact, "square"


def case_het(Nz: int, Nr: int | None = None):
    """
    HET axial-radial channel, manufactured solution
        phi = phi0 sin(pi z / Lz) cos(pi r / 2 Lr)
    zero at anode, cathode and outer wall; sinusoidal on the inner wall.
    """
    Nr = Nr or Nz
    dz, dr = HET_Lz / (Nz + 1), HET_Lr / (Nr + 1)
    z, r = np.arange(1, Nz + 1) * dz, np.arange(1, Nr + 1) * dr
    Z, R = np.meshgrid(z, r, indexing="ij")
    prof = np.sin(np.pi * Z / HET_Lz) * np.cos(np.pi * R / (2 * HET_Lr))
    phi_exact = HET_phi0 * prof
    f = -HET_phi0 * np.pi**2 * (1 / HET_Lz**2 + 1 / (4 * HET_Lr**2)) * prof
    bc_inner = HET_phi0 * np.sin(np.pi * z / HET_Lz)
    prob = PoissonLine2D(f, Lx=HET_Lz, Ly=HET_Lr, bc_y0=bc_inner)
    return prob, phi_exact, "het"


CASES = {"square": lambda N: case_square(N), "het": lambda N: case_het(N)}


# =============================================================================
#  Helpers
# =============================================================================

def rel_err(u, ref):
    return float(np.max(np.abs(u - ref)) / (np.max(np.abs(ref)) + 1e-300) * 100.0)


def colour(pct, good=1.0, ok=5.0):
    return _G if pct < good else (_Y if pct < ok else _R)


def parse_kv(items, what: str) -> dict:
    """
    Parse ``key=value`` and ``solver.key=value`` CLI pairs.

    Values stay as strings; the option registry does the type coercion and
    the validation, so this parser never has to know what a solver accepts
    and cannot drift out of step with it.
    """
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--{what} expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        if "." in key:
            solver, k = key.split(".", 1)
            out.setdefault(solver, {})[k] = value
        else:
            out[key] = value
    return out


def coerce_scheme_opts(d: dict) -> dict:
    """Scheme options are plain function kwargs, so coerce them here."""
    out = {}
    for k, v in d.items():
        if k == "omega" and v == "optimal":
            out[k] = v
        elif k in ("criterion",):
            out[k] = v
        elif k in ("symmetric", "fmg"):
            out[k] = v.lower() in ("true", "1", "yes", "on")
        elif k in ("tol",):
            out[k] = float(v)
        elif k in ("omega",):
            out[k] = float(v)
        else:
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = float(v)
    return out


def _kw_for(scheme: str, tol: float, criterion: str | None) -> dict:
    kw = {"tol": tol}
    if scheme in ("sor", "gauss-seidel", "jacobi"):
        kw["max_iter"] = 20000
        # Force the residual criterion in comparison tables so that `tol`
        # means the same thing for every scheme.  The "jacobi" scheme
        # otherwise defaults to the original delta test, under which the same
        # numeric tol is far looser and the row is not comparable.
        kw["criterion"] = criterion or "residual"
    else:
        kw["max_cycles"] = 200
    return kw


# =============================================================================
#  Polish study - does a stationary finish improve a multigrid solution?
# =============================================================================

def run_polish_study(case: str, N: int):
    """
    Tests the intuition that multigrid should be used to get close and a
    stationary scheme to finish the job.

    It does not work, and the reason is worth stating: a convergent
    stationary iteration has a *unique* fixed point.  With an inexact inner
    solver that fixed point sits roughly 1/(1 - rho) times the per-strip
    error away from the true solution, and where the iteration starts has no
    bearing on where it ends up.  Polishing therefore does not refine the
    multigrid answer - it walks away from it towards the stationary one,
    which for optimal SOR is much worse and grows worse with N.
    """
    prob, u_exact, tag = CASES[case](N)
    ref = solve(prob, inner="thomas", scheme="fmg", tol=1e-12).u

    print(f"\n{_B}{'=' * 76}{_X}")
    print(f"{_B}  MULTIGRID-THEN-POLISH STUDY   case={tag}  N={N}{_X}")
    print(f"{_B}{'=' * 76}{_X}")
    print("  Final error against the exact discrete solution.\n")
    print(f"  {'delta':>7} | {'FMG only':>10} {'SOR only':>10} {'Jacobi only':>12} "
          f"| {'FMG+SOR':>10} {'FMG+Jacobi':>11}")
    print(f"  {'-' * 70}")

    for d in (0.002, 0.005, 0.01, 0.02):
        io = {"delta": d}
        common = dict(inner="perturbed", inner_options=io)
        f = solve(prob, scheme="fmg", tol=1e-9, max_cycles=60, **common)
        s_ = solve(prob, scheme="sor", tol=1e-9, max_iter=3000, **common)
        j_ = solve(prob, scheme="jacobi", tol=1e-9, max_iter=3000,
                   criterion="residual", **common)
        ps = solve_staged(prob, [("fmg", {"tol": 1e-9, "max_cycles": 60}),
                                 ("sor", {"tol": 1e-12, "max_iter": 400})],
                          **common)
        pj = solve_staged(prob, [("fmg", {"tol": 1e-9, "max_cycles": 60}),
                                 ("jacobi", {"tol": 1e-12, "max_iter": 400,
                                             "criterion": "residual"})],
                          **common)

        def fmt(r):
            e = rel_err(r.u, ref)
            return "diverged" if (not np.isfinite(e) or e > 1e4) else f"{e:.3f}"

        print(f"  {d * 100:>6.1f}% | {fmt(f):>10} {fmt(s_):>10} {fmt(j_):>12} "
              f"| {fmt(ps):>10} {fmt(pj):>11}")

    print(f"\n  A stationary finish reproduces the stationary error, not the")
    print(f"  multigrid one.  If you need more accuracy than multigrid delivers,")
    print(f"  the lever is the inner solver's per-strip error, not the outer loop.")


# =============================================================================
#  Scheme comparison
# =============================================================================

def run_comparison(case: str, N: int, inners: list[str], schemes: list[str],
                   tol: float, verbose: bool, criterion: str | None = None,
                   inner_opts: dict | None = None,
                   scheme_opts: dict | None = None):
    prob, u_exact, tag = CASES[case](N)

    print(f"\n{_B}{'=' * 86}{_X}")
    print(f"{_B}  CASE {tag.upper()}   grid {prob.shape[0]}x{prob.shape[1]}   "
          f"dx={prob.dx:.4e}  dy={prob.dy:.4e}  kappa(A_row)={prob.kappa_row():.4f}{_X}")
    print(f"{_B}  max|u_exact| = {np.max(np.abs(u_exact)):.6g}   "
          f"algebraic tol = {tol:.0e}{_X}")
    print(f"{_B}{'=' * 86}{_X}")

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
                print(f"  {_R}[FAIL] {inner}/{scheme}: {exc}{_X}")
                continue
            wall = time.perf_counter() - t0

            if ref is None and inner == "thomas":
                ref = res.u
            e_exact = rel_err(res.u, u_exact)
            e_ref = rel_err(res.u, ref) if ref is not None else float("nan")
            rows.append((inner, scheme, res, e_exact, e_ref, wall))

            if verbose:
                print(f"    {inner}/{scheme}: {res}")

    # ---- table ---------------------------------------------------------------
    print(f"\n  {'inner':<9} {'scheme':<13} {'outer':>6} {'solves':>8} "
          f"{'w.cost':>8} {'rho':>6} {'vs exact%':>10} {'vs Thomas%':>11} "
          f"{'time s':>8} {'stop':>16}")
    print(f"  {'-' * 84}")
    base = None
    for inner, scheme, res, e_exact, e_ref, wall in rows:
        a = ALPHA.get(inner, 1.0)
        wc = res.work.weighted_cost(a)
        if scheme == "sor" and inner == rows[0][0]:
            base = wc
        print(f"  {inner:<9} {scheme:<13} {res.n_outer:>6} {res.work.total:>8} "
              f"{wc:>8.0f} {res.convergence_factor:>6.3f} "
              f"{colour(e_exact)}{e_exact:>9.3f}%{_X} "
              f"{colour(e_ref, 0.5, 2.0)}{e_ref:>10.3f}%{_X} "
              f"{wall:>8.2f} {res.stop_reason:>16}")

    # ---- speed-up summary ----------------------------------------------------
    by_inner: dict[str, dict] = {}
    for inner, scheme, res, *_ in rows:
        by_inner.setdefault(inner, {})[scheme] = res
    print(f"\n  {_B}Saving versus line-SOR (the current architecture){_X}")
    for inner, d in by_inner.items():
        if "sor" not in d:
            continue
        a = ALPHA.get(inner, 1.0)
        s = d["sor"]
        for scheme, r in d.items():
            if scheme == "sor":
                continue
            f_solves = s.work.total / max(r.work.total, 1)
            f_cost = s.work.weighted_cost(a) / max(r.work.weighted_cost(a), 1e-9)
            print(f"    {inner:<8} {scheme:<12} {f_solves:5.1f}x fewer strip solves, "
                  f"{f_cost:5.1f}x lower weighted cost")
    return rows


# =============================================================================
#  Inner-solver error tolerance study
# =============================================================================

def run_noise_study(case: str, N: int, tol: float):
    """
    Characterise how much systematic inner-solver error each outer scheme
    tolerates, using an exactly-perturbed direct solve as a surrogate for the
    quantum solver.  Runs in seconds and needs no quantum backend.

    The quantity of interest is the *amplification factor*: the ratio of the
    error in the converged field to the error in a single strip solve.  It is
    1/(1 - rho) to leading order, so a scheme with rho -> 1 (optimal SOR)
    amplifies inner-solver error in proportion to N, whereas multigrid, with
    rho ~ 0.13 independently of N, does not.
    """
    prob, u_exact, tag = CASES[case](N)
    ref = solve(prob, inner="thomas", scheme="fmg", tol=1e-12).u

    print(f"\n{_B}{'=' * 78}{_X}")
    print(f"{_B}  INNER-SOLVER ERROR TOLERANCE   case={tag}  N={N}{_X}")
    print(f"{_B}{'=' * 78}{_X}")
    print("  delta = relative size of a systematic perturbation to each strip solve.")
    print("  Reported error is of the converged field, against the exact discrete solution.\n")
    print(f"  {'delta':>7} | {'SOR err%':>11} {'its':>6} | {'FMG err%':>10} {'cyc':>5} "
          f"| {'amplification':>14}")
    print(f"  {'-' * 68}")

    for d in (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10):
        io = {"delta": d}
        a = solve(prob, inner="perturbed", scheme="sor", tol=tol,
                  max_iter=3000, inner_options=io)
        b = solve(prob, inner="perturbed", scheme="fmg", tol=tol,
                  max_cycles=60, inner_options=io)
        ea, eb = rel_err(a.u, ref), rel_err(b.u, ref)
        bad = (not np.isfinite(ea)) or ea > 1e4
        ea_s = "diverged" if bad else f"{ea:.3f}"
        amp = "-" if (bad or eb < 1e-12) else f"{ea / eb:.1f}x worse"
        print(f"  {d * 100:>6.1f}% | {ea_s:>11} {a.n_outer:>6} | {eb:>10.3f} "
              f"{b.n_outer:>5} | {amp:>14}")

    print(f"\n  Read this as the quantum error budget: an inner solver whose")
    print(f"  per-strip error exceeds the value at which SOR diverges cannot be")
    print(f"  used in the current architecture at this N, at any iteration count.")


# =============================================================================
#  Hierarchy inspection
# =============================================================================

def show_hierarchy(case: str, N: int):
    prob, _, tag = CASES[case](N)
    levels = build_hierarchy(prob)
    print(f"\n{_B}  MULTIGRID HIERARCHY  case={tag}  N={N}{_X}")
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


# =============================================================================
#  Plotting
# =============================================================================

def plot_results(rows, prob, u_exact, case, N):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print(f"  {_Y}matplotlib unavailable - skipping plots{_X}")
        return

    # ---- convergence + quantum cost (unchanged) ------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for inner, scheme, res, *_ in rows:
        h = res.residual_history
        if h:
            ax[0].semilogy(range(len(h)), h, lw=1.8, label=f"{inner}/{scheme}")
    ax[0].set_xlabel("outer iteration (sweep or cycle)")
    ax[0].set_ylabel(r"$\|b - Au\|_2 / \|b\|_2$")
    ax[0].set_title(f"Outer convergence - {case}, N={N}")
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)

    names = [f"{i}/{s}" for i, s, *_ in rows]
    costs = [r.work.weighted_cost(ALPHA.get(i, 1.0)) for i, s, r, *_ in rows]
    ax[1].barh(names, costs, color="steelblue")
    ax[1].set_xlabel("weighted strip-solve cost (finest-solve units)")
    ax[1].set_title("Quantum cost")
    ax[1].grid(alpha=0.3, axis="x")
    plt.tight_layout()
    out = OUT_DIR / f"debug_outer_{case}_N{N}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {_G}saved {out}{_X}")

    # ---- solution fields -------------------------------------------------------
    if u_exact is None or not rows:
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
    out = OUT_DIR / f"debug_outer_{case}_N{N}_fields.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {_G}saved {out}{_X}")


# =============================================================================
#  Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="square", choices=["square", "het", "all"])
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
    ap.add_argument("-I", "--inner-opt", action="append", metavar="[SOLVER.]KEY=VAL",
                    help="inner solver option, e.g. -I max_degree=300 or "
                         "-I qsvt.max_degree=300. Validated against the "
                         "registry; unknown keys are an error, not ignored.")
    ap.add_argument("-S", "--scheme-opt", action="append", metavar="KEY=VAL",
                    help="outer scheme option, e.g. -S nu1=2 -S n_coarse=8")
    ap.add_argument("--list-options", action="store_true",
                    help="print every tunable inner and scheme parameter")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.list_options:
        print("\n=== INNER SOLVER OPTIONS ===\n")
        print(describe_inner())
        print("\n=== OUTER SCHEME OPTIONS ===\n")
        print(describe_scheme())
        return

    raw_inner = parse_kv(args.inner_opt, "inner-opt")
    scheme_opts = coerce_scheme_opts(parse_kv(args.scheme_opt, "scheme-opt"))

    cases = ["square", "het"] if args.case == "all" else [args.case]

    if args.hierarchy:
        for c in cases:
            show_hierarchy(c, args.N)
        return

    if args.noise_study:
        for c in cases:
            run_noise_study(c, args.N, args.tol)
        return

    if args.polish_study:
        for c in cases:
            run_polish_study(c, args.N)
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

    print(f"\n{_B}{'=' * 86}{_X}")
    print(f"{_B}  2D OUTER-SCHEME DEBUG TOOL{_X}")
    print(f"{_B}  cases={cases}  N={args.N}  inner={inners}  schemes={schemes}{_X}")
    if inner_opts:
        print(f"{_B}  inner options : {inner_opts}{_X}")
    if scheme_opts:
        print(f"{_B}  scheme options: {scheme_opts}{_X}")
    print(f"{_B}{'=' * 86}{_X}")

    for c in cases:
        rows = run_comparison(c, args.N, inners, schemes, args.tol,
                              args.verbose, args.criterion,
                              inner_opts, scheme_opts)
        if args.plot and rows:
            prob, u_exact, _ = CASES[c](args.N)
            plot_results(rows, prob, u_exact, c, args.N)
    print()


if __name__ == "__main__":
    main()