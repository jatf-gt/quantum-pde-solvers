"""
Core types for the outer-iteration layer.

This module defines the two abstractions that the whole package rests on:

    LineProblem2D  — a 2-D BVP that has been decomposed into independent
                     1-D tridiagonal strip problems.
    InnerSolver    — anything that can solve one such strip:  (A, b) -> x

Every outer scheme (SOR, Krylov, multigrid) is written against these two
protocols only.  It therefore knows nothing about Poisson, about Hall
thrusters, or about whether the strip solve is classical or quantum, and a
new problem or a new inner solver can be added without touching any scheme.

This module is deliberately problem-agnostic: it contains no PDE, no
physics, and no Poisson-specific code.  Concrete problems that satisfy
``LineProblem2D`` / ``LineProblem3D`` live in ``problems/`` -
``problems/poisson_line_2d.py`` (PoissonLine2D) and
``problems/poisson_line_3d.py`` (PoissonLine3D) - which is also where the
sign and scaling convention used throughout this package is documented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np


# ── Work accounting ───────────────────────────────────────────────────────────

@dataclass
class WorkLog:
    """
    Counts inner (strip) solves, bucketed by strip size.

    The strip size matters: a quantum solve on a strip of size n costs
    roughly t(n) ~ n^alpha, with alpha measured empirically as ~2.4 for HHL,
    ~1.3 for VQLS and ~0.6 for QSVT under statevector simulation.  Multigrid
    performs most of its solves on coarse (small-n) strips, so a plain count
    of solves understates its advantage; ``weighted_cost`` corrects for this.
    """
    solves_by_size: dict[int, int] = field(default_factory=dict)

    def add(self, n: int, count: int = 1) -> None:
        self.solves_by_size[n] = self.solves_by_size.get(n, 0) + count

    @property
    def total(self) -> int:
        return sum(self.solves_by_size.values())

    def weighted_cost(self, alpha: float) -> float:
        """
        Total cost in units of one *finest-level* strip solve, assuming the
        per-solve cost scales as n^alpha.
        """
        if not self.solves_by_size:
            return 0.0
        n_max = max(self.solves_by_size)
        return sum(k * (n / n_max) ** alpha
                   for n, k in self.solves_by_size.items())

    def merge(self, other: "WorkLog") -> None:
        for n, k in other.solves_by_size.items():
            self.add(n, k)

    def summary(self) -> str:
        parts = [f"n={n}:{k}" for n, k in sorted(self.solves_by_size.items(),
                                                 reverse=True)]
        return f"{self.total} solves ({', '.join(parts)})"


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class OuterResult:
    """
    Outcome of an outer solve.  Deliberately scheme-agnostic: SOR reports
    sweeps, multigrid reports cycles, Krylov reports Krylov steps, but all
    of them populate the same fields.
    """
    u:              np.ndarray               # (Nx, Ny) solution field
    scheme:         str                      # "sor" | "multigrid" | ...
    inner:          str                      # "thomas" | "hhl" | "vqls" | "qsvt"
    converged:      bool
    n_outer:        int                      # sweeps / cycles / Krylov steps
    residual:       float                    # final ||r||_2 / ||b||_2
    residual_history: list[float] = field(default_factory=list)
    work:           WorkLog       = field(default_factory=WorkLog)
    wall_time_s:    float         = 0.0
    stop_reason:    str           = ""
    diagnostics:    dict          = field(default_factory=dict)

    @property
    def convergence_factor(self) -> float:
        """Geometric mean residual reduction per outer iteration."""
        h = self.residual_history
        if len(h) < 2 or h[0] <= 0.0:
            return float("nan")
        return float((h[-1] / h[0]) ** (1.0 / (len(h) - 1)))

    def __str__(self) -> str:
        flag = "converged" if self.converged else "NOT converged"
        return (f"{self.scheme}/{self.inner}: {flag} in {self.n_outer} outer "
                f"iters, res={self.residual:.2e}, rho={self.convergence_factor:.3f}, "
                f"{self.work.summary()}, {self.wall_time_s:.2f}s")


# ── Inner solver protocol ─────────────────────────────────────────────────────

@runtime_checkable
class InnerSolver(Protocol):
    """
    Solves one tridiagonal strip system A x = b.

    Implementations must be *pure* in the sense that the same (A, b) yields
    the same x; stochastic solvers (VQLS) are allowed but their variance is
    then part of the smoother error budget (see multigrid.py).
    """
    name: str

    def __call__(self, A: np.ndarray, b: np.ndarray) -> np.ndarray: ...


# ── Problem protocol ──────────────────────────────────────────────────────────

@runtime_checkable
class LineProblem2D(Protocol):
    """
    A 2-D boundary value problem decomposed into 1-D strips.

    The outer schemes require exactly four things.  Any problem providing
    them - Poisson on a square, the HET axial-radial channel, a variable
    permittivity sheath model, a 3-D slab reduced plane-by-plane - can be
    driven by every scheme in this package without modification.
    """

    shape:       tuple[int, int]    # (Nx, Ny): strip length, number of strips
    dx:          float
    dy:          float

    def row_matrix(self) -> np.ndarray:
        """The Nx x Nx tridiagonal strip operator (identical for all strips)."""
        ...

    def rhs(self) -> np.ndarray:
        """(Nx, Ny) right-hand side with Dirichlet data already absorbed."""
        ...

    def apply(self, u: np.ndarray) -> np.ndarray:
        """
        Full 2-D operator applied to u, with *homogeneous* boundary data.
        Used for residual evaluation:  r = rhs() - apply(u).
        """
        ...

    def coarsen(self) -> Optional["LineProblem2D"]:
        """
        Return the same problem on a grid with half as many points in each
        direction, or None if it cannot be coarsened further.

        The coarse problem is only ever used to solve the *error* equation
        A e = r, so it always carries homogeneous boundary data and a zero
        right-hand side.  Implementations therefore need to reproduce the
        operator, not the boundary conditions.
        """
        ...


# ── Stagnation detection ──────────────────────────────────────────────────────

class StagnationMonitor:
    """
    Detects when an outer iteration has hit the inner solver's error floor.

    Every quantum inner solver has a floor below which the strip solution
    cannot be improved: Trotter truncation for HHL, the cost-function minimum
    for VQLS, polynomial truncation for QSVT.  Once the outer residual reaches
    that floor it stops decreasing, and every further iteration spends its full
    quota of circuit simulations for nothing.  On an HPC sweep that is the
    difference between a run finishing and a run being killed by the walltime.

    A run that stagnates is not necessarily a failure - it has converged to the
    accuracy the inner solver can deliver.  The result records the floor so it
    can be reported honestly rather than being mistaken for convergence.

    Detection compares the current residual with the residual ``window``
    iterations ago, rather than with the immediately preceding one.  A
    per-iteration test is unusable for stationary schemes: line-SOR has
    rho -> 1 as N grows, so its per-iteration improvement tends to zero and
    any fixed per-iteration threshold eventually mistakes healthy - if slow -
    convergence for stagnation.  Over a window of 10 iterations even
    rho = 0.995 still yields ~5 % improvement, comfortably above the
    threshold, while a true noise floor yields essentially zero.
    """

    def __init__(self, window: int = 20, min_improvement: float = 0.01):
        self.window = window
        self.min_improvement = min_improvement
        self.history: list[float] = []
        self.best = float("inf")

    def update(self, residual: float) -> bool:
        """Return True if the iteration should stop."""
        if not np.isfinite(residual):
            return True
        self.history.append(residual)
        self.best = min(self.best, residual)
        if len(self.history) < self.window:
            return False

        # Compare the MEDIAN residual of the two halves of the window.
        # The median is essential, not cosmetic.  Line-SOR residuals are not
        # monotone and show a sharp transient dip - a factor of ~50 at N=64,
        # around iteration 64 - where truncation and iteration error briefly
        # cancel.  A test built on the raw residual, or on the running
        # minimum, treats the recovery from that dip as a stall and stops the
        # run roughly 40 % short of convergence.  A median over a half-window
        # ignores the outlier while still detecting a genuine floor, where
        # both medians coincide.
        half = self.window // 2
        prior = float(np.median(self.history[-self.window:-half]))
        recent = float(np.median(self.history[-half:]))
        if prior <= 0.0:
            return False
        return (prior - recent) / prior < self.min_improvement


# ── Strip sweep - the primitive every scheme is built from ────────────────────

def strip_sweep(
    problem,
    u:        np.ndarray,
    rhs:      np.ndarray,
    inner:    Callable[[np.ndarray, np.ndarray], np.ndarray],
    work:     WorkLog,
    omega:    float = 1.0,
    reverse:  bool = False,
    jacobi:   bool = False,
) -> np.ndarray:
    """
    One line-relaxation sweep, updating u in place.  Works in any dimension.

    Axis 0 is always the strip direction: the sweep visits every transverse
    index tuple, gathers that strip's transverse neighbours into the
    right-hand side, and hands the resulting tridiagonal system to ``inner``.
    In 2-D the transverse index is a single j; in 3-D it is a pair (j, k).
    The 2-D behaviour is bit-for-bit unchanged.

    Strips are traversed in Gauss-Seidel order (each strip sees the already
    updated values of its predecessors), then relaxed by omega:

        u[:, idx] <- omega * inner(A, b_idx) + (1 - omega) * u[:, idx]

    omega = 1 gives line Gauss-Seidel, the correct choice as a *multigrid
    smoother*.  Over-relaxation (omega ~ 1.9) accelerates a standalone
    stationary iteration but destroys the smoothing property and makes the
    iteration fragile to inner-solver error - see stationary.py.

    Why the strip direction, and not a plane, generalises to 3-D
    ------------------------------------------------------------
    The natural-looking 3-D analogue of a 2-D line smoother is a *plane*
    smoother: fix z, solve the whole (x, y) plane.  It is the wrong choice
    here, though not for the obvious reason.  The plane operator carries a
    -2/dz^2 diagonal shift from the z-coupling, which bounds its condition
    number at 5 - it is not the O(N^2) system it first appears to be.  The
    decisive objection is structural: a plane solve is an N^2 x N^2 system
    with a 5-point stencil, needing 2*log2(N) qubits and a block encoding
    that does not exist in this repository.  A strip solve stays an N x N
    TST system on log2(N) qubits, identical to the 1-D case, so every
    existing quantum solver works in 3-D with no modification whatsoever.
    Line relaxation also gives a *better* conditioned inner system in 3-D
    than in 2-D: kappa -> 2 rather than 3, because the two transverse
    directions both contribute to the diagonal.
    """
    shape = tuple(problem.shape)
    Nx = shape[0]
    transverse = shape[1:]
    A = problem.row_matrix()

    # Spacings and periodicity, with a fallback for the original 2-D class.
    spacings = getattr(problem, "spacings", None)
    if spacings is None:
        spacings = (problem.dx, problem.dy)
    periodic = getattr(problem, "periodic", None)
    if periodic is None:
        periodic = (False,) * len(shape)
    inv_h2 = [1.0 / spacings[d + 1] ** 2 for d in range(len(transverse))]

    order = list(np.ndindex(*transverse)) if transverse else [()]
    if reverse:
        order = order[::-1]
    src = u.copy() if jacobi else u

    for idx in order:
        key = (slice(None),) + idx
        b = rhs[key].copy()
        for d, n in enumerate(transverse):
            for step in (-1, 1):
                j = idx[d] + step
                if periodic[d + 1]:
                    j %= n
                elif j < 0 or j >= n:
                    continue
                nb = (slice(None),) + idx[:d] + (j,) + idx[d + 1:]
                b -= src[nb] * inv_h2[d]
        x = np.asarray(inner(A, b), dtype=float)
        work.add(Nx)
        if x.shape != (Nx,):
            raise ValueError(f"inner solver returned shape {x.shape}, "
                             f"expected ({Nx},)")
        u[key] = omega * x + (1.0 - omega) * u[key]
    return u