"""
Outer-iteration layer for line-decomposed 2-D (and later 3-D) BVPs.

The point of this package is separation of concerns.  A user declares a
problem and picks an inner solver; the choice of outer iteration becomes a
tuning parameter rather than something they have to implement:

    from solvers.outer import solve, PoissonLine2D

    problem = PoissonLine2D(f_values, Lx=0.025, Ly=0.020, bc_y0=phi_wall)
    result  = solve(problem, inner="hhl", scheme="fmg", tol=1e-6)

    result.u                          # (Nx, Ny) solution field
    result.n_outer                    # cycles or sweeps
    result.work.total                 # strip solves performed
    result.work.weighted_cost(2.35)   # cost in finest-strip-solve units

Adding a scheme, an inner solver or a problem each touch exactly one file:

    scheme       -> a function (problem, inner, **opts) -> OuterResult,
                    registered in SCHEMES below
    inner solver -> a factory registered in inner.py
    problem      -> a class satisfying the LineProblem2D protocol in core.py

Schemes
-------
    "jacobi"        line Jacobi, delta criterion - reproduces the original
                    validated scheme exactly.  Use this to reproduce or
                    fall back to previously published small-N results.
    "sor"           line SOR with optimal omega - the current production
                    scheme.
    "gauss-seidel"  omega = 1; poor standalone, correct as an MG smoother.
    "multigrid"     V-cycles.
    "fmg"           full multigrid (default): grid-independent iteration
                    count and the lowest quantum cost.

Nothing here imports Qiskit unless a quantum inner solver is requested, so
the classical path stays fast to import and the tests run without a backend.
"""
from __future__ import annotations

from typing import Callable, Sequence, Union

import numpy as np

from solvers.outer.core import (InnerSolver, LineProblem2D, OuterResult,
                                StagnationMonitor, WorkLog, strip_sweep)
from solvers.outer.inner import (InnerConfig, Option, available as available_inner,
                                 available_options, describe as describe_inner,
                                 get_inner, resolve_options)
from solvers.outer.multigrid import (build_hierarchy, interpolation_1d,
                                     interpolation_1d_periodic, solve_multigrid)
from problems.poisson_line_2d import PoissonLine2D
from problems.poisson_line_3d import PoissonLine3D
from solvers.outer.stationary import optimal_omega, solve_stationary

__all__ = [
    "solve", "solve_staged", "SCHEMES", "available_schemes", "available_inner",
    "InnerConfig", "Option", "available_options", "describe_inner",
    "resolve_options", "SCHEME_OPTIONS", "describe_scheme",
    "PoissonLine2D", "PoissonLine3D", "OuterResult", "WorkLog",
    "StagnationMonitor", "interpolation_1d_periodic",
    "LineProblem2D", "InnerSolver",
    "get_inner", "solve_stationary", "solve_multigrid",
    "optimal_omega", "build_hierarchy", "interpolation_1d", "strip_sweep",
    "parse_kv", "coerce_scheme_opts",
]


# ── Scheme registry ───────────────────────────────────────────────────────────

def _jacobi(problem, inner, **kw):
    """The original scheme: simultaneous strip update, delta stopping test."""
    kw.setdefault("update", "jacobi")
    kw.setdefault("criterion", "delta")
    kw.setdefault("omega", 1.0)
    return solve_stationary(problem, inner, **kw)


def _sor(problem, inner, **kw):
    kw.setdefault("omega", "optimal")
    kw.setdefault("update", "gauss-seidel")
    return solve_stationary(problem, inner, **kw)


def _gauss_seidel(problem, inner, **kw):
    kw["omega"] = 1.0
    kw.setdefault("update", "gauss-seidel")
    return solve_stationary(problem, inner, **kw)


def _multigrid(problem, inner, **kw):
    kw.setdefault("fmg", False)
    return solve_multigrid(problem, inner, **kw)


def _fmg(problem, inner, **kw):
    kw.setdefault("fmg", True)
    return solve_multigrid(problem, inner, **kw)


SCHEMES: dict[str, Callable[..., OuterResult]] = {
    "jacobi":       _jacobi,
    "sor":          _sor,
    "gauss-seidel": _gauss_seidel,
    "multigrid":    _multigrid,
    "fmg":          _fmg,
}

_STATIONARY = {"jacobi", "sor", "gauss-seidel"}

# Tunable parameters of each outer scheme, for runner scripts and --help.
# Kept as plain metadata: schemes are ordinary functions and their signatures
# are the authority, but a runner needs something it can enumerate.
SCHEME_OPTIONS: dict[str, dict[str, str]] = {
    "jacobi": {
        "tol": "stopping tolerance (delta by default for this scheme)",
        "max_iter": "iteration cap",
        "criterion": "'delta' (original) or 'residual'",
        "omega": "relaxation factor",
        "patience": "stagnation window",
    },
    "sor": {
        "tol": "relative residual tolerance",
        "max_iter": "iteration cap",
        "omega": "'optimal' or a float in (0,2)",
        "criterion": "'residual' or 'delta'",
        "symmetric": "alternate sweep direction (SSOR)",
        "patience": "stagnation window",
    },
    "multigrid": {
        "tol": "relative residual tolerance",
        "max_cycles": "V-cycle cap",
        "nu1": "pre-smoothing sweeps per level",
        "nu2": "post-smoothing sweeps per level",
        "n_coarse": "relaxation sweeps on the coarsest grid",
        "max_levels": "hierarchy depth cap",
        "patience": "stagnation window",
    },
}
SCHEME_OPTIONS["gauss-seidel"] = SCHEME_OPTIONS["sor"]
SCHEME_OPTIONS["fmg"] = SCHEME_OPTIONS["multigrid"]


def available_schemes() -> list[str]:
    return sorted(SCHEMES)


def describe_scheme(name: str | None = None) -> str:
    """Human-readable table of scheme options, for --list-options."""
    names = available_schemes() if name is None else [name]
    out = []
    for n in names:
        out.append(f"{n}:")
        for k, h in sorted(SCHEME_OPTIONS.get(n, {}).items()):
            out.append(f"    {k:<14} {h}")
    return "\n".join(out)


# ── CLI option parsing ────────────────────────────────────────────────────────
#
# Shared by every runner script that exposes -I/-S flags for inner-solver and
# outer-scheme options. Kept beside the registries they are validated against
# (available_inner(), SCHEME_OPTIONS above) rather than duplicated per script.

def parse_kv(items: list[str] | None, what: str) -> dict:
    """
    Parses ``key=value`` and ``solver.key=value`` CLI pairs into a dict.

    Values are left as strings; the inner-solver option registry
    (``resolve_options``) performs type coercion and validation, so this
    parser never has to know what a given solver accepts and cannot drift
    out of step with it.

    Parameters
    ----------
    items : list of str, optional
        Raw ``--flag`` values, e.g. from ``argparse``'s ``action="append"``.
    what : str
        Flag name, used only to phrase the error message.

    Returns
    -------
    dict
        Bare keys map to their string value; namespaced keys
        (``solver.key``) nest as ``{solver: {key: value}}``.

    Raises
    ------
    SystemExit
        If an item is not of the form ``key=value``.
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
    """
    Type-coerces a parsed outer-scheme option dict.

    Outer-scheme options are plain function keyword arguments (unlike inner
    solver options, which are validated and coerced by
    ``resolve_options``), so this coercion happens here instead.

    Parameters
    ----------
    d : dict
        String-valued options, as returned by ``parse_kv``.

    Returns
    -------
    dict
        Same keys, values coerced to ``str`` (``criterion``, ``omega ==
        "optimal"``), ``bool`` (``symmetric``, ``fmg``), or numeric
        (``int`` where possible, else ``float``).
    """
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


# ── Public entry points ───────────────────────────────────────────────────────

def solve(
    problem: LineProblem2D,
    inner:   Union[str, InnerSolver] = "thomas",
    scheme:  str = "fmg",
    inner_options: dict | None = None,
    u0:      np.ndarray | None = None,
    **scheme_options,
) -> OuterResult:
    """
    Solve a line-decomposed 2-D problem.

    Parameters
    ----------
    problem : anything satisfying ``LineProblem2D`` (e.g. ``PoissonLine2D``).
    inner : name of a strip solver - "thomas", "hhl", "vqls", "qsvt",
        "perturbed" - or any callable ``(A, b) -> x``.
    scheme : see the module docstring.  Default "fmg".
    inner_options : options for the inner solver, validated against its
        declared set - an unknown key raises rather than being ignored.
        Either a flat dict for this solver, e.g. ``{"max_degree": 500}``,
        or an ``InnerConfig`` holding one section per solver, in which case
        only the relevant section is used.  See ``describe_inner()``.
    u0 : optional initial guess.
    **scheme_options : forwarded to the scheme - ``tol``, ``max_iter`` or
        ``max_cycles``, ``nu1``, ``nu2``, ``omega``, ``criterion``,
        ``patience``, ``callback``.

    Raises
    ------
    ValueError : unknown scheme or inner solver, or a problem that cannot be
        coarsened when a multigrid scheme was requested.  Multigrid never
        silently degrades to a stationary scheme: if the hierarchy cannot be
        built you are told, because a silent fallback would quietly restore
        the O(N) iteration count you were trying to escape.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown scheme {scheme!r}. "
                         f"Available: {', '.join(available_schemes())}")

    if isinstance(inner, str):
        opts = (inner_options.for_solver(inner)
                if isinstance(inner_options, InnerConfig)
                else dict(inner_options or {}))
        inner_solver = get_inner(inner, **opts)
    else:
        inner_solver = inner

    if u0 is not None:
        scheme_options["u0"] = u0

    result = SCHEMES[scheme](problem, inner_solver, **scheme_options)

    if hasattr(inner_solver, "summary"):
        result.diagnostics.update(inner_solver.summary())
    return result


def solve_staged(
    problem: LineProblem2D,
    stages:  Sequence[tuple[str, dict]],
    inner:   Union[str, InnerSolver] = "thomas",
    inner_options: dict | None = None,
) -> OuterResult:
    """
    Run several schemes in sequence, each starting from the previous result.

        solve_staged(prob, [("fmg", {"tol": 1e-6}),
                            ("jacobi", {"tol": 1e-9, "max_iter": 200})],
                     inner="hhl")

    Provided because it is the obvious thing to reach for, and because it
    makes the composition explicit and measurable.  Be aware of what it
    does and does not buy you.

    A convergent stationary iteration has a *unique* fixed point, and with an
    inexact inner solver that fixed point is displaced from the true solution
    by roughly 1/(1 - rho) times the per-strip error.  Where the iteration
    starts is irrelevant to where it ends up.  So finishing a multigrid
    solution with Jacobi or SOR sweeps does not refine it - it walks away
    from the multigrid answer towards the (much worse) stationary one.
    Measured on the unit square with a 0.2 % systematic strip error at N=64:

        FMG alone          0.94 %
        SOR alone         20.77 %
        FMG then SOR      20.66 %   <- the polish undoes the multigrid result
        FMG then Jacobi   15.67 %

    The staging is genuinely useful in the other direction and for other
    purposes: a cheap classical FMG solve to generate a starting guess for a
    quantum run, switching inner solvers between stages, or reproducing a
    legacy result from a modern starting point.  It is not a way to squeeze
    extra accuracy out of a noisy inner solver.
    """
    if isinstance(inner, str):
        opts = (inner_options.for_solver(inner)
                if isinstance(inner_options, InnerConfig)
                else dict(inner_options or {}))
        inner_solver = get_inner(inner, **opts)
    else:
        inner_solver = inner

    u = None
    combined = WorkLog()
    history: list[float] = []
    names, last = [], None
    total_time = 0.0

    for scheme_name, opts in stages:
        res = solve(problem, inner=inner_solver, scheme=scheme_name, u0=u, **opts)
        u = res.u
        combined.merge(res.work)
        history.extend(res.residual_history)
        total_time += res.wall_time_s
        names.append(f"{scheme_name}({res.n_outer})")
        last = res

    if last is None:
        raise ValueError("solve_staged requires at least one stage")

    return OuterResult(
        u=u,
        scheme=" -> ".join(names),
        inner=getattr(inner_solver, "name", "?"),
        converged=last.converged,
        n_outer=len(history),
        residual=history[-1] if history else float("nan"),
        residual_history=history,
        work=combined,
        wall_time_s=total_time,
        stop_reason=last.stop_reason,
        diagnostics={"stages": [s for s, _ in stages], **last.diagnostics},
    )