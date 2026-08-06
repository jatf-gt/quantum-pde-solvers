"""
2D Poisson problem, decomposed into line-relaxation strips.

This is the *problem* half of the outer/inner architecture in ``solvers/outer``:
a concrete implementation of the ``LineProblem2D`` protocol (defined in
``solvers/outer/core.py``), which is what every outer scheme (SOR, multigrid,
…) and every inner solver (Thomas, HHL, VQLS, QSVT) is written against.

This is the sole 2D Poisson problem type in the repository. It superseded an
earlier ``PoissonProblem2D``, which assembled the full N²×N² system for a
parallel set of 2D solvers; that stack was retired once ``solvers/outer`` was
shown to reproduce its results exactly, and no trace of it remains.

Sign and scaling convention
---------------------------
The *physical* (unscaled) convention is used throughout:

    A_row = tridiag( 1/dx²,  -2·(1/dx² + 1/dy²),  1/dx² )
    rhs   = f  (with Dirichlet contributions absorbed, see _build_rhs)

so that A_row · u[:,j] + u[:,j-1]/dy² + u[:,j+1]/dy² = rhs[:,j].

The reference literature instead uses the h²-scaled form (diagonal = -4,
rhs = h²f). The two are algebraically identical when dx = dy, and yield
bit-identical solutions; the physical form is used here because it extends
unchanged to non-square cells (dz ≠ dr in the HET geometry) and to the 3D case
in ``poisson_line_3d.py``. Quantities invariant under the uniform h² rescaling
— the condition number κ(A_row) above all — therefore agree exactly between the
two conventions.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class PoissonLine2D:
    """
    ∇²u = f on [0,Lx] × [0,Ly] with Dirichlet boundaries, discretised by the
    standard 5-point stencil on a vertex-centred interior grid (Nx × Ny
    interior nodes, dx = Lx/(Nx+1), dy = Ly/(Ny+1)) and decomposed into Ny
    strips of length Nx along x.

    This single class covers both the unit-square benchmarks and the HET
    axial-radial channel; the HET case is simply Lx=Lz, Ly=Lr with a non-zero
    ``bc_y0``.

    Boundary data
    -------------
    bc_x0, bc_x1 : arrays of length Ny (or scalars) — values at x=0, x=Lx
    bc_y0, bc_y1 : arrays of length Nx (or scalars) — values at y=0, y=Ly

    For the HET channel with the current benchmark convention:
        bc_x0 = anode, bc_x1 = cathode, bc_y0 = inner wall, bc_y1 = outer wall

    Attributes
    ----------
    shape : tuple[int, int]
        (Nx, Ny) interior node counts along x and y.
    Lx, Ly : float
        Physical domain extents [m], or unity for non-dimensional benchmarks.
    dx, dy : float
        Mesh spacings, dx = Lx/(Nx+1) and dy = Ly/(Ny+1).
    level : int
        Multigrid level index; 0 is the finest grid. Coarse levels carry the
        error equation and therefore homogeneous source and boundary data.
    f : np.ndarray
        (Nx, Ny) source field at the interior nodes.
    bc_x0, bc_x1 : np.ndarray
        Length-Ny Dirichlet data on the x=0 and x=Lx faces.
    bc_y0, bc_y1 : np.ndarray
        Length-Nx Dirichlet data on the y=0 and y=Ly faces.
    """

    def __init__(
        self,
        f_values: np.ndarray,
        Lx: float = 1.0,
        Ly: float = 1.0,
        bc_x0=0.0, bc_x1=0.0, bc_y0=0.0, bc_y1=0.0,
        _level: int = 0,
    ) -> None:
        """
        Assembles the strip operator and the boundary-absorbed right-hand side.

        Parameters
        ----------
        f_values : np.ndarray
            (Nx, Ny) source field sampled at the interior nodes.
        Lx, Ly : float
            Physical domain extents. Default to unity for the non-dimensional
            benchmarks.
        bc_x0, bc_x1 : float or np.ndarray
            Dirichlet data on x=0 and x=Lx; scalar or length-Ny.
        bc_y0, bc_y1 : float or np.ndarray
            Dirichlet data on y=0 and y=Ly; scalar or length-Nx.
        _level : int
            Multigrid level index, set internally by ``coarsen``.

        Raises
        ------
        ValueError
            If ``f_values`` is not 2-D.
        """
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

    # ── Operator ──────────────────────────────────────────────────────────────

    def _build_row_matrix(self) -> np.ndarray:
        """Assembles the (Nx, Nx) Toeplitz symmetric tridiagonal strip operator."""
        Nx = self.shape[0]
        a = -2.0 * (1.0 / self.dx**2 + 1.0 / self.dy**2)
        b = 1.0 / self.dx**2
        return (a * np.eye(Nx)
                + b * (np.diag(np.ones(Nx - 1), 1) + np.diag(np.ones(Nx - 1), -1)))

    def _build_rhs(self) -> np.ndarray:
        """Absorbs the Dirichlet data into a copy of the (Nx, Ny) source field."""
        r = self.f.copy()
        r[0, :]  -= self.bc_x0 / self.dx**2
        r[-1, :] -= self.bc_x1 / self.dx**2
        r[:, 0]  -= self.bc_y0 / self.dy**2
        r[:, -1] -= self.bc_y1 / self.dy**2
        return r

    def row_matrix(self) -> np.ndarray:
        """The Nx × Nx tridiagonal strip operator, satisfying LineProblem2D."""
        return self._A

    def rhs(self) -> np.ndarray:
        """(Nx, Ny) right-hand side with Dirichlet data already absorbed."""
        return self._rhs

    def apply(self, u: np.ndarray) -> np.ndarray:
        """
        5-point Laplacian with homogeneous exterior.

        Used for residual evaluation (r = rhs() - apply(u)) and on coarse
        multigrid levels, which always carry homogeneous boundary data — see
        coarsen() below.

        Parameters
        ----------
        u : np.ndarray
            (Nx, Ny) field at the interior nodes.

        Returns
        -------
        np.ndarray
            (Nx, Ny) result of applying the discrete Laplacian to u.
        """
        r = np.zeros_like(u)
        r[1:, :]  += u[:-1, :] / self.dx**2
        r[:-1, :] += u[1:, :]  / self.dx**2
        r[:, 1:]  += u[:, :-1] / self.dy**2
        r[:, :-1] += u[:, 1:]  / self.dy**2
        r += -2.0 * (1.0 / self.dx**2 + 1.0 / self.dy**2) * u
        return r

    # ── Coarsening ────────────────────────────────────────────────────────────

    MIN_STRIP = 4          # quantum solvers need >= 2 qubits, i.e. n >= 4

    # An axis is coarsened only if its spacing is within this factor of the
    # finer one.  On the benchmark grids (unit square, and HET with
    # dz/dr = 1.25) both axes always qualify, so this changes nothing there;
    # it guards against strongly anisotropic grids, where coarsening a
    # weakly-coupled direction degrades the V-cycle below plain SOR.  See the
    # extended discussion in poisson_line_3d.py::coarsen, where the effect
    # is large enough to flip the sign of the multigrid speedup.
    COARSEN_RATIO = 2.0

    def coarsen(self) -> Optional["PoissonLine2D"]:
        """
        Halve each direction whose spacing is within COARSEN_RATIO of the
        finer one (anisotropic semi-coarsening).

        On an isotropic grid both directions always qualify, so this reduces
        to standard full coarsening: both dimensions stay powers of two, the
        strip operator remains a Toeplitz symmetric tridiagonal matrix of
        power-of-two size at every level, and the quantum inner solvers
        require no modification.  Halving both directions together also
        keeps dx/dy fixed, which keeps κ(A_row) ~ 3 on every level;
        coarsening only one axis would make κ grow by 4× per level,
        driving up the QSVT polynomial degree and the HHL clock register.

        Returns
        -------
        PoissonLine2D or None
            The next coarser level, or None once neither remaining direction
            can be halved (either it is already at MIN_STRIP, or it is odd,
            or — the anisotropic case — its spacing already exceeds
            COARSEN_RATIO times the finest axis).
        """
        Nx, Ny = self.shape
        h_min = min(self.dx, self.dy)
        do = [h <= self.COARSEN_RATIO * h_min for h in (self.dx, self.dy)]
        do = [d and n > self.MIN_STRIP and n % 2 == 0
              for d, n in zip(do, (Nx, Ny))]
        if not any(do):
            return None
        # Coarse levels carry the error equation: zero source, zero boundaries.
        return PoissonLine2D(
            np.zeros((Nx // 2 if do[0] else Nx, Ny // 2 if do[1] else Ny)),
            Lx=self.Lx, Ly=self.Ly,
            _level=self.level + 1,
        )

    # ── Utilities ─────────────────────────────────────────────────────────────

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the (Nx, Ny) interior coordinate matrices in 'ij' index order.
        """
        Nx, Ny = self.shape
        x = np.arange(1, Nx + 1) * self.dx
        y = np.arange(1, Ny + 1) * self.dy
        return np.meshgrid(x, y, indexing="ij")

    def kappa_row(self) -> float:
        """
        Spectral condition number κ(A_row) of the strip operator.

        For the unit square this tends to 3⁻ as N → ∞, far better conditioned
        than the O(N²) growth of the full 1D Poisson operator — the property
        that makes the line-decomposed formulation tractable for the quantum
        inner solvers.
        """
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
