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

Sign and scaling convention
---------------------------
Throughout this package the *physical* (unscaled) convention is used:

    A_row = tridiag( 1/dx^2,  -2*(1/dx^2 + 1/dy^2),  1/dx^2 )
    rhs   = f  (with Dirichlet contributions absorbed, see PoissonLine2D)

so that A_row . u[:,j] + u[:,j-1]/dy^2 + u[:,j+1]/dy^2 = rhs[:,j].

Note this differs from ``problems/poisson_2d.py``, which uses the h^2-scaled
form (A_row diagonal = -4, rhs = h^2 f).  The two are equivalent when
dx = dy; the physical form is used here because it extends unchanged to
non-square cells (dz != dr in the HET geometry) and to variable coefficients.

Author : Juan Antonio Trobajo Flecha
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, runtime_checkable

import numpy as np


# =============================================================================
#  Work accounting
# =============================================================================

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


# =============================================================================
#  Result container
# =============================================================================

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


# =============================================================================
#  Inner solver protocol
# =============================================================================

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


# =============================================================================
#  Problem protocol
# =============================================================================

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


# =============================================================================
#  Concrete implementation: constant-coefficient Poisson on a rectangle
# =============================================================================

class PoissonLine2D:
    """
    nabla^2 u = f on [0,Lx] x [0,Ly] with Dirichlet boundaries, discretised
    by the standard 5-point stencil on a vertex-centred interior grid
    (Nx x Ny interior nodes, dx = Lx/(Nx+1), dy = Ly/(Ny+1)) and decomposed
    into Ny strips of length Nx along x.

    This single class covers both the unit-square benchmarks and the HET
    axial-radial channel; the HET case is just Lx=Lz, Ly=Lr with a
    non-zero ``bc_y0``.

    Boundary data
    -------------
    bc_x0, bc_x1 : arrays of length Ny (or scalars) - values at x=0, x=Lx
    bc_y0, bc_y1 : arrays of length Nx (or scalars) - values at y=0, y=Ly

    For the HET channel with the current benchmark convention:
        bc_x0 = anode, bc_x1 = cathode, bc_y0 = inner wall, bc_y1 = outer wall
    """

    def __init__(
        self,
        f_values: np.ndarray,
        Lx: float = 1.0,
        Ly: float = 1.0,
        bc_x0=0.0, bc_x1=0.0, bc_y0=0.0, bc_y1=0.0,
        _level: int = 0,
    ) -> None:
        f_values = np.asarray(f_values, dtype=float)
        if f_values.ndim != 2:
            raise ValueError(f"f_values must be 2-D, got shape {f_values.shape}")

        self.shape = (int(f_values.shape[0]), int(f_values.shape[1]))
        Nx, Ny = self.shape
        self.Lx, self.Ly = float(Lx), float(Ly)
        self.dx = self.Lx / (Nx + 1)
        self.dy = self.Ly / (Ny + 1)
        self.level = _level

        self.f = f_values
        self.bc_x0 = np.broadcast_to(np.asarray(bc_x0, dtype=float), (Ny,)).copy()
        self.bc_x1 = np.broadcast_to(np.asarray(bc_x1, dtype=float), (Ny,)).copy()
        self.bc_y0 = np.broadcast_to(np.asarray(bc_y0, dtype=float), (Nx,)).copy()
        self.bc_y1 = np.broadcast_to(np.asarray(bc_y1, dtype=float), (Nx,)).copy()

        self._A = self._build_row_matrix()
        self._rhs = self._build_rhs()

    # ---------------------------------------------------------------- operator

    def _build_row_matrix(self) -> np.ndarray:
        Nx = self.shape[0]
        a = -2.0 * (1.0 / self.dx**2 + 1.0 / self.dy**2)
        b = 1.0 / self.dx**2
        return (a * np.eye(Nx)
                + b * (np.diag(np.ones(Nx - 1), 1) + np.diag(np.ones(Nx - 1), -1)))

    def _build_rhs(self) -> np.ndarray:
        r = self.f.copy()
        r[0, :]  -= self.bc_x0 / self.dx**2
        r[-1, :] -= self.bc_x1 / self.dx**2
        r[:, 0]  -= self.bc_y0 / self.dy**2
        r[:, -1] -= self.bc_y1 / self.dy**2
        return r

    def row_matrix(self) -> np.ndarray:
        return self._A

    def rhs(self) -> np.ndarray:
        return self._rhs

    def apply(self, u: np.ndarray) -> np.ndarray:
        """5-point Laplacian with homogeneous exterior."""
        r = np.zeros_like(u)
        r[1:, :]  += u[:-1, :] / self.dx**2
        r[:-1, :] += u[1:, :]  / self.dx**2
        r[:, 1:]  += u[:, :-1] / self.dy**2
        r[:, :-1] += u[:, 1:]  / self.dy**2
        r += -2.0 * (1.0 / self.dx**2 + 1.0 / self.dy**2) * u
        return r

    # ---------------------------------------------------------------- coarsening

    MIN_STRIP = 4          # quantum solvers need >= 2 qubits, i.e. n >= 4

    def coarsen(self) -> Optional["PoissonLine2D"]:
        """
        Halve both directions.  Both dimensions stay powers of two, so the
        strip operator remains a Toeplitz symmetric tridiagonal matrix of
        power-of-two size at every level and the quantum inner solvers
        require no modification.

        Halving both directions (rather than semi-coarsening in y only) also
        keeps dx/dy fixed, which keeps kappa(A_row) ~ 3 on every level.
        Semi-coarsening would make kappa grow by 4x per level, driving up
        the QSVT polynomial degree and the HHL clock register.

        Returns None once either dimension reaches MIN_STRIP.
        """
        Nx, Ny = self.shape
        if Nx <= self.MIN_STRIP or Ny <= self.MIN_STRIP:
            return None
        if Nx % 2 or Ny % 2:
            return None
        # Coarse levels carry the error equation: zero source, zero boundaries.
        return PoissonLine2D(
            np.zeros((Nx // 2, Ny // 2)),
            Lx=self.Lx, Ly=self.Ly,
            _level=self.level + 1,
        )

    # ---------------------------------------------------------------- utilities

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        Nx, Ny = self.shape
        x = np.arange(1, Nx + 1) * self.dx
        y = np.arange(1, Ny + 1) * self.dy
        return np.meshgrid(x, y, indexing="ij")

    def kappa_row(self) -> float:
        e = np.abs(np.linalg.eigvalsh(self._A))
        return float(e.max() / e.min())

    def residual(self, u: np.ndarray) -> float:
        """Relative 2-norm residual of the full coupled system."""
        b = self.rhs()
        bn = np.linalg.norm(b)
        r = np.linalg.norm(b - self.apply(u))
        return float(r / bn) if bn > 1e-300 else float(r)

    def __repr__(self) -> str:
        Nx, Ny = self.shape
        return (f"PoissonLine2D({Nx}x{Ny}, dx={self.dx:.3e}, dy={self.dy:.3e}, "
                f"kappa={self.kappa_row():.3f}, level={self.level})")


# =============================================================================
#  Stagnation detection
# =============================================================================

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


# =============================================================================
#  Strip sweep - the primitive every scheme is built from
# =============================================================================

def strip_sweep(
    problem:  LineProblem2D,
    u:        np.ndarray,
    rhs:      np.ndarray,
    inner:    Callable[[np.ndarray, np.ndarray], np.ndarray],
    work:     WorkLog,
    omega:    float = 1.0,
    reverse:  bool = False,
    jacobi:   bool = False,
) -> np.ndarray:
    """
    One line-relaxation sweep, updating u in place.

    Strips are traversed in Gauss-Seidel order (each strip sees the already
    updated values of its predecessor), then relaxed by omega:

        u[:,j] <- omega * inner(A, b_j) + (1 - omega) * u[:,j]

    omega = 1 gives line Gauss-Seidel, which is the correct choice as a
    *multigrid smoother*.  Over-relaxation (omega ~ 1.9) accelerates a
    standalone stationary iteration but destroys the smoothing property and
    makes the iteration fragile to inner-solver error - see the omega
    discussion in stationary.py.
    """
    Nx, Ny = problem.shape
    A = problem.row_matrix()
    dy2 = problem.dy ** 2
    order = range(Ny - 1, -1, -1) if reverse else range(Ny)
    src = u.copy() if jacobi else u

    for j in order:
        b = rhs[:, j].copy()
        if j > 0:      b -= src[:, j - 1] / dy2
        if j < Ny - 1: b -= src[:, j + 1] / dy2
        x = np.asarray(inner(A, b), dtype=float)
        work.add(Nx)
        if x.shape != (Nx,):
            raise ValueError(f"inner solver returned shape {x.shape}, expected ({Nx},)")
        u[:, j] = omega * x + (1.0 - omega) * u[:, j]
    return u