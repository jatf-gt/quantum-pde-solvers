#!/usr/bin/env python3
"""
tutorial.py — the single entry point for trying this repository out.

Renamed from explore.py. Solves the Poisson equation in one, two or three
dimensions with any combination of the classical and quantum solvers, and
prints a comparison table. If you are new to this codebase, start here.

    python scripts/tutorial.py --dim 1 --N 8
    python scripts/tutorial.py --dim 2 --N 16 --inner all
    python scripts/tutorial.py --dim 3 --N 8 --scheme fmg

What the dimensions actually do
-------------------------------
In 1-D the solvers are used directly: a single N x N tridiagonal system is
handed to Thomas, HHL, VQLS or QSVT.

In 2-D and 3-D there is no separate solver. The domain is decomposed into 1-D
strips and an *outer iteration* sweeps over them, handing each strip to exactly
the same 1-D solver. Two things follow, and they are the central design idea of
this project:

  * every quantum solver works in any dimension with no modification, because
    it only ever sees a 1-D strip; and
  * the strip operator is far better conditioned than the 1-D Poisson operator
    (kappa -> 3 in 2-D and -> 2 in 3-D, against O(N^2) in 1-D), so the quantum
    solvers are *cheaper* per strip in higher dimensions, not dearer.

The outer iteration is a free choice — `--scheme` — and it matters as much as
the solver. "jacobi" reproduces the originally published scheme; "fmg" (full
multigrid, the default) reaches the same answer in a grid-independent number of
iterations and is dramatically cheaper at large N.

Suggested first commands
------------------------
    # Classical, instant. Confirms the installation works.
    python scripts/tutorial.py --dim 2 --N 32

    # A quantum solver on a small 2-D problem (~1 min).
    python scripts/tutorial.py --dim 2 --N 8 --inner qsvt

    # Why the outer scheme matters: compare iteration counts and cost.
    python scripts/tutorial.py --dim 2 --N 64 --scheme all

    # Every case in the canonical registry, and every tunable parameter.
    python scripts/tutorial.py --list-cases
    python scripts/tutorial.py --list-options

Relationship to the other scripts
---------------------------------
For 1-D this is a front end onto `debug_1d.py`, which carries the raw-matrix
sub-cases (3b, 3c), QSVT degree/cache diagnostics and the kappa-scaling
tables. For 2-D and 3-D it is a front end onto `debug_2d.py` and
`debug_3d.py`, which carry the full diagnostic surface — noise studies,
multigrid hierarchy inspection, polish studies. Anything not exposed here is
available there. For systematic sweeps at many resolutions, use the
`hpc/` drivers instead; this script is for single configurations.

Note on the import below: unlike explore.py, which imported debug_outer_2d /
debug_outer_3d by bare module name and relied on Python placing a script's own
directory on sys.path, this module is imported as `scripts.debug_2d` /
`scripts.debug_3d` via importlib with an explicit path, so it does not depend
on which directory the interpreter was launched from.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import cases
from core.config import SimConfig1D
from core.exact_solutions import EXACT_SOLUTIONS
from problems.poisson_1d import PoissonProblem1D
from solvers.classical.thomas import thomas_solve
from solvers.outer import available_inner, available_schemes, describe_inner, describe_scheme

# Empirical per-strip-solve cost exponents, t(n) ~ n^alpha, fitted from N=4/N=8
# statevector timings. Used to weight coarse-level work correctly when
# comparing outer schemes: multigrid does most of its solves on short strips,
# so a raw solve count understates its advantage.
ALPHA = {"hhl": 2.35, "vqls": 1.29, "qsvt": 0.60, "thomas": 1.0, "perturbed": 1.0}

_G, _Y, _R = "\033[92m", "\033[93m", "\033[91m"
_B, _X = "\033[1m", "\033[0m"

QUANTUM = ("hhl", "vqls", "qsvt")


def _colour(pct: float, good: float = 1.0, ok: float = 5.0) -> str:
    """Green below `good` per cent, amber below `ok`, red above."""
    return _G if pct < good else (_Y if pct < ok else _R)


def _rel_err(u: np.ndarray, ref: np.ndarray) -> float:
    """Maximum absolute error normalised by the reference amplitude, per cent."""
    return float(np.max(np.abs(u - ref)) / (np.max(np.abs(ref)) + 1e-300) * 100.0)


def _load_driver(dim: int):
    """
    Imports scripts/debug_2d.py or scripts/debug_3d.py by explicit file path.

    Deliberately not a bare `import debug_2d`: that form only resolves because
    Python puts a running script's own directory on sys.path first, which is
    fragile the moment this module is imported rather than run directly (e.g.
    from a test). Loading by path has no such dependency.
    """
    name = "debug_2d" if dim == 2 else "debug_3d"
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# -- One Dimension -------------------------------------------------------------

def run_1d(
    N:          int,
    source_fn:  str,
    solvers:    list[str],
    epsilon:    float,
    plot:       bool,
) -> None:
    """
    Solves the 1-D Poisson equation directly with each requested solver.

    No outer iteration is involved: the whole problem is one tridiagonal system.
    This is the configuration in which the quantum solvers are most expensive,
    because kappa grows as O(N²) — which is precisely the motivation for the
    strip decomposition used in higher dimensions.

    Parameters
    ----------
    N : int
        Interior nodes. Must be a power of two.
    source_fn : str
        Source function identifier ('fS', 'fL', 'fH').
    solvers : list[str]
        Solvers to run, from 'thomas', 'hhl', 'vqls', 'qsvt'.
    epsilon : float
        Precision parameter for the quantum solvers.
    plot : bool
        Write a solution/error figure to results/tutorial/.
    """
    cfg = SimConfig1D(N=N, epsilon=epsilon, source_fn=source_fn)
    problem = PoissonProblem1D(cfg)

    print(f"\n{_B}{'=' * 78}{_X}")
    print(f"{_B}  1-D POISSON   N={N}  f={source_fn}  eps={epsilon}  "
          f"kappa(A)={problem.kappa:.2f}{_X}")
    print(f"{_B}{'=' * 78}{_X}")
    print(f"  kappa grows as O(N^2) here — this is the hard case for a quantum "
          f"solver.\n")

    reference = thomas_solve(problem).u

    # A closed form exists only for homogeneous boundary data; otherwise the
    # Thomas solution is the only available reference.
    exact = (EXACT_SOLUTIONS[source_fn](problem.x)
             if cfg.alpha == 0.0 and cfg.beta == 0.0
             and source_fn in EXACT_SOLUTIONS else None)

    print(f"  {'solver':<10} {'time s':>9} {'vs Thomas %':>13} "
          f"{'vs exact %':>12}  {'notes'}")
    print(f"  {'-' * 70}")

    fields = {}
    for name in solvers:
        try:
            t0 = time.perf_counter()
            result = _solve_1d(problem, name)
            wall = time.perf_counter() - t0
        except Exception as exc:
            print(f"  {_R}{name:<10} FAILED: {exc}{_X}")
            continue

        fields[name] = result.u
        e_ref = _rel_err(result.u, reference)
        e_exact = _rel_err(result.u, exact) if exact is not None else float("nan")
        notes = []
        if getattr(result, "final_cost", None) is not None:
            notes.append(f"cost={result.final_cost:.2e}")
        if getattr(result, "polynomial_degree", None) is not None:
            notes.append(f"degree={result.polynomial_degree}")

        print(f"  {name:<10} {wall:>9.2f} "
              f"{_colour(e_ref, 0.5, 2.0)}{e_ref:>12.3f}%{_X} "
              f"{_colour(e_exact)}{e_exact:>11.3f}%{_X}  {', '.join(notes)}")

    if plot and fields:
        _plot_1d(problem, fields, exact, N, source_fn)


def _solve_1d(problem, name: str):
    """
    Dispatches to one 1-D solver.

    Imports are deferred per solver so that a classical-only run never loads
    Qiskit or PennyLane, which dominate start-up time.
    """
    if name == "thomas":
        return thomas_solve(problem)
    if name == "hhl":
        from solvers.quantum.hhl_1d import hhl_solve
        return hhl_solve(problem)
    if name == "vqls":
        from solvers.quantum.vqls_1d import vqls_solve
        return vqls_solve(problem)
    if name == "qsvt":
        from solvers.quantum.qsvt_1d import qsvt_solve
        return qsvt_solve(problem)
    raise ValueError(f"Unknown 1-D solver {name!r}. "
                     f"Valid: thomas, hhl, vqls, qsvt.")


def _plot_1d(problem, fields: dict, exact, N: int, source_fn: str) -> None:
    """Writes a two-panel solution and error figure."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  {_Y}matplotlib unavailable — skipping the plot{_X}")
        return

    out_dir = REPO_ROOT / "results" / "tutorial"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    x = problem.x
    if exact is not None:
        ax1.plot(x, exact, "k-", lw=2, label="exact")
    for name, u in fields.items():
        ax1.plot(x, u, "o--", ms=4, label=name)
        reference = exact if exact is not None else fields.get("thomas")
        if reference is not None:
            ax2.semilogy(x, np.abs(u - reference) + 1e-18, "o-", ms=3, label=name)

    ax1.set_xlabel("x"); ax1.set_ylabel("u"); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_title(f"1-D Poisson, N={N}, f={source_fn}")
    ax2.set_xlabel("x"); ax2.set_ylabel("|error|"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_title("Pointwise error")

    plt.tight_layout()
    out = out_dir / f"tutorial_1d_N{N}_{source_fn}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  {_G}saved {out}{_X}")


# -- Two and Three Dimensions --------------------------------------------------

def run_nd(dim: int, args: argparse.Namespace) -> None:
    """
    Delegates a 2-D or 3-D run to the corresponding debug driver.

    Both drivers already implement exactly the comparison this script wants —
    build the case, run every (inner, scheme) pair, print the table and the
    saving-versus-SOR summary — so they are called rather than reimplemented.
    Anything not exposed here is available by invoking them directly.

    Parameters
    ----------
    dim : {2, 3}
        Spatial dimension.
    args : argparse.Namespace
        Parsed command line.
    """
    driver = _load_driver(dim)
    default_case = "square" if dim == 2 else "cube"

    case = args.case or default_case
    valid = set(driver.CASE_ALIASES) | set(cases.available(dim=dim))
    if case not in valid and case != "all":
        raise SystemExit(f"Unknown --case {case!r} for --dim {dim}. "
                         f"Valid: {', '.join(sorted(valid))}, or 'all'.")

    inners = list(QUANTUM) if args.inner == "all" else [args.inner]
    if "thomas" not in inners:
        inners = ["thomas"] + inners          # always keep the classical reference

    schemes = available_schemes() if args.scheme == "all" else [args.scheme]

    from solvers.outer import coerce_scheme_opts, parse_kv

    inner_opts = parse_kv(args.inner_opt, "inner-opt")
    # A bare key=value targets the single requested solver; it must never leak
    # onto the automatically added Thomas reference, which accepts no options.
    flat = {k: v for k, v in inner_opts.items() if not isinstance(v, dict)}
    nested = {k: v for k, v in inner_opts.items() if isinstance(v, dict)}
    if flat:
        if args.inner == "all":
            raise SystemExit(
                f"-I {list(flat)[0]}=... is ambiguous with --inner all. "
                f"Use the namespaced form, e.g. -I qsvt.{list(flat)[0]}=...")
        nested.setdefault(args.inner, {}).update(flat)

    scheme_opts = coerce_scheme_opts(parse_kv(args.scheme_opt, "scheme-opt"))

    print(f"\n{_B}{'=' * 78}{_X}")
    print(f"{_B}  {dim}-D POISSON via strip decomposition   case={case}  "
          f"N={args.N}{_X}")
    print(f"{_B}  inner solvers={inners}   outer schemes={schemes}{_X}")
    print(f"{_B}{'=' * 78}{_X}")

    # Both drivers accept the same keyword names for run_comparison, even
    # though their positional order differs; calling by keyword makes that
    # irrelevant rather than something the caller has to track.
    rows = driver.run_comparison(case=case, N=args.N, inners=inners, schemes=schemes,
                                 tol=args.tol, verbose=args.verbose,
                                 criterion=args.criterion,
                                 inner_opts=nested, scheme_opts=scheme_opts)

    if args.plot and rows:
        from benchmark.diagnostics import plot_convergence_and_cost
        prob, u_exact, tag = driver.build_case(case, args.N)
        plot_convergence_and_cost(rows, tag, args.N, driver.OUT_DIR)
        if dim == 2:
            driver.plot_fields(rows, prob, u_exact, tag, args.N)


# -- Main ----------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dim", type=int, choices=(1, 2, 3), default=2,
                    help="Spatial dimension (default: 2).")
    ap.add_argument("--N", type=int, default=16,
                    help="Interior nodes per direction; a power of two "
                         "(default: 16).")
    ap.add_argument("--inner", default="thomas",
                    help=f"Solver for each 1-D system: {available_inner()}, "
                         f"or 'all' for every quantum solver. In 1-D this is "
                         f"the solver itself. (default: thomas)")
    ap.add_argument("--scheme", default="fmg",
                    help=f"Outer iteration for 2-D/3-D: {available_schemes()}, "
                         f"or 'all'. Ignored for --dim 1. (default: fmg)")
    ap.add_argument("--case", default=None,
                    help="Test case. 2-D: square, het, or any name from "
                         "core.cases.available(dim=2). 3-D: cube, het, slab, "
                         "or any name from core.cases.available(dim=3). "
                         "Ignored for --dim 1, which uses --source.")
    ap.add_argument("--source", default="fS", choices=("fS", "fL", "fH"),
                    help="1-D source function (default: fS).")
    ap.add_argument("--epsilon", type=float, default=0.01,
                    help="1-D quantum precision parameter (default: 0.01).")
    ap.add_argument("--tol", type=float, default=1e-4,
                    help="2-D/3-D outer residual tolerance (default: 1e-4, one "
                         "order below typical discretisation error).")
    ap.add_argument("--criterion", default=None, choices=("residual", "delta"),
                    help="Stopping test for stationary schemes; 'delta' "
                         "reproduces the originally published check. 2-D/3-D only.")
    ap.add_argument("-I", "--inner-opt", action="append",
                    metavar="[SOLVER.]KEY=VAL",
                    help="Inner solver option, e.g. -I max_degree=300 or "
                         "-I qsvt.max_degree=300. Validated against the "
                         "registry: an unknown key is an error, not ignored.")
    ap.add_argument("-S", "--scheme-opt", action="append", metavar="KEY=VAL",
                    help="Outer scheme option, e.g. -S nu1=2 -S n_coarse=8.")
    ap.add_argument("--list-options", action="store_true",
                    help="Print every tunable inner and scheme parameter, "
                         "then exit.")
    ap.add_argument("--list-cases", action="store_true",
                    help="Print every registered case for --dim, then exit.")
    ap.add_argument("--plot", action="store_true",
                    help="Write figures to results/tutorial/ (1-D) or "
                         "results/debugging/ (2-D/3-D).")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.list_options:
        print("\n=== INNER SOLVER OPTIONS ===\n")
        print(describe_inner())
        print("\n=== OUTER SCHEME OPTIONS ===\n")
        print(describe_scheme())
        return

    if args.list_cases:
        if args.dim == 1:
            for name in cases.available(dim=1):
                print(cases.describe(name))
                print()
        else:
            for name in cases.available(dim=args.dim):
                print(cases.describe(name))
                print()
        return

    if args.dim == 1:
        solvers = list(QUANTUM) if args.inner == "all" else [args.inner]
        if "thomas" not in solvers:
            solvers = ["thomas"] + solvers
        run_1d(args.N, args.source, solvers, args.epsilon, args.plot)
    else:
        run_nd(args.dim, args)

    print()


if __name__ == "__main__":
    main()
