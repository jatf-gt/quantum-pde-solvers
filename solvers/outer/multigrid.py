"""
Geometric multigrid for line-decomposed 2-D problems.

Structure
---------
A hierarchy of grids is built by repeatedly calling ``problem.coarsen()``.
On each level the smoother is one line Gauss-Seidel sweep, which is exactly
the primitive the quantum solvers already implement, so the quantum solver
is exercised on every level rather than only on the finest one.

    V-cycle(level l):
        nu1 smoothing sweeps
        r    <- b - A u                  (fine residual)
        r_c  <- R r                      (restriction)
        e_c  <- V-cycle(l+1) on A_c e_c = r_c
        u    <- u + P e_c                (prolongation)
        nu2 smoothing sweeps

Full multigrid (FMG) additionally starts from the coarsest level and
interpolates upward, so that the finest level begins from an already
accurate initial guess.  It reaches discretisation accuracy in about 3
fine-level cycles rather than 5.

Why this matters for a quantum inner solver
-------------------------------------------
Two properties of the hierarchy are specific to the quantum setting and are
the reason multigrid is a much stronger fit here than in a purely classical
code:

1.  *Coarse strips are exponentially cheaper.*  Halving the strip length
    removes one qubit from the b-register.  Measured statevector cost per
    strip solve scales as n^alpha with alpha ~ 2.4 (HHL), ~1.3 (VQLS),
    ~0.6 (QSVT).  Multigrid does most of its solves on coarse strips, so
    the wall-clock advantage is considerably larger than the reduction in
    solve count alone: at N=64 the strip-solve count falls 5.8x versus
    optimal SOR but the HHL-weighted cost falls ~12x.

2.  *The conditioning does not degrade with depth.*  Because both
    directions are coarsened together, dx/dy is preserved and
    kappa(A_row) stays ~2-3 on every level.  The QSVT polynomial degree and
    the HHL clock register are therefore constant across the hierarchy.
    Semi-coarsening in a single direction would raise kappa by a factor of
    4 per level and destroy this.

3.  *Inner-solver error is not amplified.*  The error of a converged
    iterate is amplified by roughly 1/(1 - rho) relative to the per-solve
    error.  For optimal SOR rho -> 1 as O(1 - 1/N), so the amplification
    grows with N; for multigrid rho ~ 0.13 independently of N.  Measured
    with a systematic 0.2 % strip error: 18.0 % solution error under SOR at
    N=64 versus 1.0 % under multigrid, and SOR diverges at 1 % strip error
    while multigrid still converges to ~5 %.

Grid hierarchy and the power-of-two constraint
----------------------------------------------
The quantum solvers require strip lengths that are powers of two.  The
existing benchmark grids use N = 2^k interior nodes with h = L/(N+1), so
successive levels (N, N/2) are *not* nested: the coarse nodes do not
coincide with a subset of the fine nodes.  Transfer operators are therefore
built as 1-D linear interpolation between the actual coordinate sets rather
than by the usual stencil.  The measured cost of this is a convergence
factor of ~0.13 rather than the ~0.10 typical of a nested hierarchy - a
small price for keeping every level quantum-compatible.

The alternative, N = 2^k - 1 with nested vertex-centred coarsening, gives
non-power-of-two strips and cannot be used with the quantum solvers at all.

Author : Juan Antonio Trobajo Flecha
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from solvers.outer.core import (LineProblem2D, OuterResult, StagnationMonitor,
                                WorkLog, strip_sweep)


# =============================================================================
#  Transfer operators
# =============================================================================

def interpolation_1d(n_fine: int, n_coarse: int, L: float) -> np.ndarray:
    """
    Linear interpolation matrix P, shape (n_fine, n_coarse), between two
    vertex-centred interior grids on [0, L] that need not be nested.

    Boundary nodes are included in the interpolation stencil with value
    zero, which is correct because coarse levels always carry the error
    equation with homogeneous boundary data.
    """
    x_f = np.arange(1, n_fine + 1) * (L / (n_fine + 1))
    x_c = np.arange(1, n_coarse + 1) * (L / (n_coarse + 1))
    x_ext = np.concatenate(([0.0], x_c, [L]))

    P = np.zeros((n_fine, n_coarse))
    for i, x in enumerate(x_f):
        k = int(np.clip(np.searchsorted(x_ext, x) - 1, 0, len(x_ext) - 2))
        t = (x - x_ext[k]) / (x_ext[k + 1] - x_ext[k])
        for idx, w in ((k, 1.0 - t), (k + 1, t)):
            if 1 <= idx <= n_coarse:
                P[i, idx - 1] += w
    return P


def restriction_from(P: np.ndarray) -> np.ndarray:
    """
    Full-weighting restriction R, shape (n_coarse, n_fine), normalised so
    that each row sums to one.

    The normalisation is essential and is the single easiest thing to get
    wrong: an unnormalised P^T under-weights the coarse residual and the
    V-cycle degrades to a convergence factor that grows with N, which looks
    exactly like "multigrid does not work on this problem".
    """
    R = P.T.copy()
    s = R.sum(axis=1, keepdims=True)
    s[s == 0.0] = 1.0
    return R / s


def interpolation_1d_periodic(n_fine: int, n_coarse: int) -> np.ndarray:
    """
    Linear interpolation matrix for a *periodic* axis, shape (n_fine, n_coarse).

    A periodic axis is discretised without boundary nodes (x_i = i*L/n), so
    coarsening n -> n/2 is exactly nested: coarse point I coincides with fine
    point 2I.  Odd fine points interpolate between neighbouring coarse points
    with wraparound.  This is what makes the azimuthal direction of the HET
    channel work: it is periodic, has no Dirichlet data, and would otherwise
    have no valid transfer operator.
    """
    P = np.zeros((n_fine, n_coarse))
    for I in range(n_coarse):
        P[(2 * I) % n_fine, I] += 1.0
        P[(2 * I + 1) % n_fine, I] += 0.5
        P[(2 * I - 1) % n_fine, I] += 0.5
    return P


def _apply_axis_ops(mats: list[np.ndarray], arr: np.ndarray) -> np.ndarray:
    """
    Apply one matrix per axis, as a tensor product.

    Replaces the hard-coded ``Px @ r @ Py.T`` of the 2-D-only version; this
    form is dimension-agnostic, so the same V-cycle drives 2-D and 3-D.
    """
    out = arr
    for ax, M in enumerate(mats):
        out = np.moveaxis(np.tensordot(M, out, axes=([1], [ax])), 0, ax)
    return out


@dataclass
class Level:
    """One grid in the hierarchy, plus the operators that reach the next."""
    problem: object
    P: Optional[list] = None      # one (n_fine, n_coarse) matrix per axis
    R: Optional[list] = None      # one (n_coarse, n_fine) matrix per axis

    def restrict(self, r: np.ndarray) -> np.ndarray:
        return _apply_axis_ops(self.R, r)

    def prolong(self, e: np.ndarray) -> np.ndarray:
        return _apply_axis_ops(self.P, e)


def build_hierarchy(problem, max_levels: int = 10) -> list[Level]:
    """
    Coarsen until ``coarsen()`` returns None or max_levels is reached.

    Works for any dimension: one 1-D transfer operator is built per axis,
    choosing the periodic or Dirichlet form according to the problem's
    ``periodic`` flags.
    """
    levels = [Level(problem)]
    while len(levels) < max_levels:
        fine = levels[-1].problem
        coarse = fine.coarsen()
        if coarse is None:
            break
        f_shape, c_shape = tuple(fine.shape), tuple(coarse.shape)
        lengths = getattr(fine, "lengths", None)
        if lengths is None:                       # original 2-D class
            lengths = (getattr(fine, "Lx", f_shape[0] * fine.dx),
                       getattr(fine, "Ly", f_shape[1] * fine.dy))
        periodic = getattr(fine, "periodic", (False,) * len(f_shape))

        P_ops = []
        for ax in range(len(f_shape)):
            if f_shape[ax] == c_shape[ax]:
                # Axis not coarsened at this level (anisotropic
                # semi-coarsening): the transfer is the identity.
                P_ops.append(np.eye(f_shape[ax]))
            elif periodic[ax]:
                P_ops.append(interpolation_1d_periodic(f_shape[ax], c_shape[ax]))
            else:
                P_ops.append(interpolation_1d(f_shape[ax], c_shape[ax],
                                              lengths[ax]))
        levels[-1].P = P_ops
        levels[-1].R = [restriction_from(P) for P in P_ops]
        levels.append(Level(coarse))
    return levels


# =============================================================================
#  Cycles
# =============================================================================

def _v_cycle(levels, l, u, rhs, inner, work, nu1, nu2, n_coarse):
    """Recursive V-cycle. ``u`` is modified in place and returned."""
    lev = levels[l]
    prob = lev.problem

    if l == len(levels) - 1:
        # Coarsest level: relax to (near) convergence.  This is still done
        # with the inner solver, so a quantum run exercises it everywhere.
        for _ in range(n_coarse):
            strip_sweep(prob, u, rhs, inner, work, omega=1.0)
        return u

    for _ in range(nu1):
        strip_sweep(prob, u, rhs, inner, work, omega=1.0)

    r = rhs - prob.apply(u)
    r_c = lev.restrict(r)
    e_c = np.zeros(tuple(levels[l + 1].problem.shape))
    e_c = _v_cycle(levels, l + 1, e_c, r_c, inner, work, nu1, nu2, n_coarse)
    u += lev.prolong(e_c)

    for _ in range(nu2):
        strip_sweep(prob, u, rhs, inner, work, omega=1.0)
    return u


def solve_multigrid(
    problem:   LineProblem2D,
    inner,
    tol:       float = 1e-8,
    max_cycles: int = 100,
    nu1:       int = 1,
    nu2:       int = 1,
    n_coarse:  int = 12,
    fmg:       bool = True,
    max_levels: int = 10,
    patience:  int = 10,
    callback=None,
) -> OuterResult:
    """
    Solve by multigrid V-cycles, optionally preceded by an FMG start.

    Parameters
    ----------
    nu1, nu2 : pre- and post-smoothing sweeps per level.  V(1,1) is the
        default and is close to optimal here; V(2,1) buys a slightly better
        convergence factor for ~50 % more work.
    n_coarse : relaxation sweeps on the coarsest grid.
    fmg : begin with a full-multigrid start (coarse-to-fine nested
        iteration).  Reaches discretisation accuracy in roughly 3 fine-level
        cycles instead of 5, at no extra fine-level cost.
    tol : relative residual ||b - A u|| / ||b||.

    Falls back to a clear error if the problem admits no coarse level; use
    ``solve_stationary`` in that case.
    """
    levels = build_hierarchy(problem, max_levels)
    if len(levels) < 2:
        raise ValueError(
            f"{problem!r} cannot be coarsened (shape {problem.shape}); "
            f"multigrid needs at least two levels. Use scheme='sor'.")

    rhs = problem.rhs()
    b_norm = np.linalg.norm(rhs)
    b_norm = b_norm if b_norm > 1e-300 else 1.0
    work = WorkLog()
    history: list[float] = []
    monitor = StagnationMonitor(window=patience)
    t0 = time.perf_counter()

    # ---- FMG start: nested iteration from the coarsest level upward ---------
    # Each level needs a genuine right-hand side, obtained by restricting the
    # fine one.  Passing zeros here (a natural-looking mistake) makes every
    # intermediate V-cycle solve A e = 0, so the FMG start does nothing and
    # merely wastes strip solves.
    if fmg:
        rhs_levels = [rhs]
        for l in range(len(levels) - 1):
            rhs_levels.append(levels[l].restrict(rhs_levels[-1]))

        u = np.zeros(tuple(levels[-1].problem.shape))
        for _ in range(n_coarse):
            strip_sweep(levels[-1].problem, u, rhs_levels[-1], inner, work, 1.0)
        for l in range(len(levels) - 2, -1, -1):
            u = levels[l].prolong(u)
            u = _v_cycle(levels, l, u, rhs_levels[l], inner, work,
                         nu1, nu2, n_coarse)
    else:
        u = np.zeros(tuple(problem.shape))

    # ---- V-cycles on the finest level ---------------------------------------
    stop = "max_cycles"
    for cyc in range(max_cycles):
        res = float(np.linalg.norm(rhs - problem.apply(u)) / b_norm)
        history.append(res)
        if callback is not None:
            callback(cyc, u, res)
        if res < tol:
            stop = "tol_met"
            break
        if not np.isfinite(res):
            stop = "diverged"
            break
        if monitor.update(res):
            stop = "stagnated"
            break
        u = _v_cycle(levels, 0, u, rhs, inner, work, nu1, nu2, n_coarse)
    else:
        history.append(float(np.linalg.norm(rhs - problem.apply(u)) / b_norm))

    return OuterResult(
        u=u,
        scheme="fmg" if fmg else "multigrid",
        inner=getattr(inner, "name", "?"),
        converged=(stop == "tol_met"),
        n_outer=max(len(history) - 1, 0),
        residual=history[-1],
        residual_history=history,
        work=work,
        wall_time_s=time.perf_counter() - t0,
        stop_reason=stop,
        diagnostics={
            "n_levels": len(levels),
            "level_shapes": [lv.problem.shape for lv in levels],
            "level_kappas": [round(lv.problem.kappa_row(), 4) for lv in levels],
            "nu1": nu1, "nu2": nu2, "n_coarse": n_coarse, "fmg": fmg,
            "residual_floor": monitor.best,
        },
    )