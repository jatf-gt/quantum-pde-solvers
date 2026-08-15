"""
Shared comparison-table and study primitives for the interactive debug drivers.

Purpose
-------
``scripts/debug_2d.py`` and ``scripts/debug_3d.py`` both compare inner solvers
and outer schemes on a line-decomposed problem, print a results table, and
characterise how the choice of outer scheme affects sensitivity to inner-solver
error and to a post-hoc stationary "polish". That machinery does not depend on
spatial dimension - ``solvers.outer.solve`` and ``solvers.outer.solve_staged``
take the same ``LineProblem2D``/``LineProblem3D`` protocol either way - so it
lives here once rather than twice.

What stays out of this module, deliberately: the result table itself
(``run_comparison`` in each debug script) is not consolidated, because its
column set genuinely differs by dimension (2-D reports ``dx``/``dy``
separately; 3-D reports an anisotropy ratio and physical spacings in mm).
Consolidating these signatures would introduce architectural complexity
without computational benefit. Only the components that are byte-for-byte
identical - the error metric, the colour threshold, the cost-weighting
table and the "saving versus SOR" summary - are centralised here.

References
----------
Ghafourpour & Laizet, Phys. Rev. Applied 24, 024032 (2025) - the outer-scheme
    cost comparison this module's tables support.
"""
from __future__ import annotations

import numpy as np

from solvers.outer import solve, solve_staged

# -- ANSI console colours -----------------------------------------------------

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

# -- Empirical per-strip-solve cost exponents ---------------------------------
#
# t(n) ~ n^alpha, fitted from the N=4 / N=8 statevector timings (HHL 0.267 s ->
# 1.36 s, VQLS 0.806 -> 1.965, QSVT 0.0259 -> 0.0393). Used to weight coarse-
# level work correctly: multigrid does most of its solves on short strips, so
# a raw solve count understates its advantage over a stationary scheme.
ALPHA: dict[str, float] = {
    "hhl": 2.35, "vqls": 1.29, "qsvt": 0.60, "thomas": 1.0, "perturbed": 1.0,
}


# -- Error metric and colour threshold ----------------------------------------

def rel_err(u: np.ndarray, ref: np.ndarray) -> float:
    """Maximum absolute error normalised by the reference amplitude, per cent."""
    return float(np.max(np.abs(u - ref)) / (np.max(np.abs(ref)) + 1e-300) * 100.0)


def colour(pct: float, good: float = 1.0, ok: float = 5.0) -> str:
    """Green below `good` per cent, amber below `ok`, red above."""
    return GREEN if pct < good else (YELLOW if pct < ok else RED)


# -- Saving versus line-SOR ----------------------------------------------------

def savings_summary(rows: list[tuple],
                    header: str = "Saving versus line-SOR "
                                  "(the current architecture)") -> None:
    """
    Prints, for each inner solver, how many fewer strip solves and how much
    lower weighted cost every non-SOR scheme achieves relative to line-SOR.

    Parameters
    ----------
    rows : list of tuple
        ``(inner, scheme, res, ...)``, one per (inner, scheme) pair already
        run; trailing tuple elements are ignored. ``res`` must expose
        ``.work.total`` and ``.work.weighted_cost(alpha)``.
    header : str
        Section title. The two debug drivers differ by a parenthetical, kept
        as a parameter rather than forced identical.
    """
    by_inner: dict[str, dict] = {}
    for inner, scheme, res, *_ in rows:
        by_inner.setdefault(inner, {})[scheme] = res

    print(f"\n  {BOLD}{header}{RESET}")
    for inner, per_scheme in by_inner.items():
        if "sor" not in per_scheme:
            continue
        a = ALPHA.get(inner, 1.0)
        sor_res = per_scheme["sor"]
        for scheme, res in per_scheme.items():
            if scheme == "sor":
                continue
            f_solves = sor_res.work.total / max(res.work.total, 1)
            f_cost = (sor_res.work.weighted_cost(a)
                     / max(res.work.weighted_cost(a), 1e-9))
            print(f"    {inner:<9} {scheme:<12} {f_solves:5.1f}x fewer strip "
                  f"solves, {f_cost:5.1f}x lower weighted cost")


# -- Inner-solver error tolerance study ----------------------------------------

def noise_study(prob, tag: str, N: int, tol: float) -> None:
    """
    Characterises how much systematic inner-solver error each outer scheme
    tolerates, using an exactly-perturbed direct solve (the ``"perturbed"``
    inner solver) as a surrogate for the quantum solver. Runs in seconds and
    needs no quantum backend.

    The quantity of interest is the *amplification factor*: the ratio of the
    error in the converged field to the error in a single strip solve. It is
    1/(1 - rho) to leading order, so a scheme with rho -> 1 (optimal SOR)
    amplifies inner-solver error in proportion to N, whereas multigrid, with
    rho ~ 0.13 independently of N, does not.

    Parameters
    ----------
    prob : LineProblem2D or LineProblem3D
        The problem to study.
    tag : str
        Case label, for the printed header only.
    N : int
        Resolution, for the printed header only.
    tol : float
        Algebraic residual tolerance passed to SOR and FMG.
    """
    ref = solve(prob, inner="thomas", scheme="fmg", tol=1e-12).u

    print(f"\n{BOLD}{'=' * 78}{RESET}")
    print(f"{BOLD}  INNER-SOLVER ERROR TOLERANCE   case={tag}  N={N}{RESET}")
    print(f"{BOLD}{'=' * 78}{RESET}")
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

    print(f"\n  Quantum error budget interpretation: an inner solver exhibiting a")
    print(f"  per-strip error exceeding the SOR divergence threshold is inviable")
    print(f"  for the current architecture at this N, regardless of iteration count.")


# -- Multigrid-then-polish study -----------------------------------------------

def polish_study(prob, tag: str, N: int) -> None:
    """
    Tests the intuition that multigrid should be used to get close and a
    stationary scheme to finish the job.

    It does not work, and the reason is worth stating: a convergent
    stationary iteration has a *unique* fixed point. With an inexact inner
    solver that fixed point sits roughly 1/(1 - rho) times the per-strip
    error away from the true solution, and where the iteration starts has no
    bearing on where it ends up. Polishing therefore does not refine the
    multigrid answer - it walks away from it towards the stationary one,
    which for optimal SOR is much worse and grows worse with N.

    Parameters
    ----------
    prob : LineProblem2D or LineProblem3D
        The problem to study.
    tag : str
        Case label, for the printed header only.
    N : int
        Resolution, for the printed header only.
    """
    ref = solve(prob, inner="thomas", scheme="fmg", tol=1e-12).u

    print(f"\n{BOLD}{'=' * 76}{RESET}")
    print(f"{BOLD}  MULTIGRID-THEN-POLISH STUDY   case={tag}  N={N}{RESET}")
    print(f"{BOLD}{'=' * 76}{RESET}")
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
    print(f"  multigrid one. If you need more accuracy than multigrid delivers,")
    print(f"  the lever is the inner solver's per-strip error, not the outer loop.")


# -- Convergence and cost plot -------------------------------------------------

def plot_convergence_and_cost(rows: list[tuple], case: str, N: int,
                              out_dir) -> None:
    """
    Writes a two-panel figure: outer-iteration residual history (left) and
    weighted strip-solve cost per (inner, scheme) pair (right).

    Dimension-agnostic - unlike a solution-field plot, this only touches
    ``res.residual_history`` and ``res.work``, so both the 2-D and 3-D debug
    drivers share it. Field plotting stays 2-D only: a 3-D field has no single
    natural 2-D rendering, and building a slice-based one is future work, not
    part of this consolidation.

    Parameters
    ----------
    rows : list of tuple
        ``(inner, scheme, res, ...)``, as returned by each driver's
        ``run_comparison``.
    case : str
        Case name, for the title and output filename.
    N : int
        Resolution, for the title and output filename.
    out_dir : Path
        Directory to write the figure into; created if absent.

    Raises
    ------
    None. If matplotlib is unavailable, prints a warning and returns.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  {YELLOW}matplotlib unavailable - skipping plots{RESET}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

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
    out = out_dir / f"debug_{case}_N{N}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {GREEN}saved {out}{RESET}")
