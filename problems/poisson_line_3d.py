"""
3D line-decomposed problems.

Adding 3D required *no* changes to any quantum solver, and no new ``hhl_3d.py``
/ ``vqls_3d.py`` / ``qsvt_3d.py`` modules. That is a direct consequence of the
outer/inner split: the quantum solvers were never 2D to begin with, they solve
one tridiagonal strip, and a 3D problem decomposes into exactly the same strips.
What is new here is one problem class and one grid-transfer form; the schemes,
the work accounting and the inner-solver registry are reused verbatim.

Discretisation
--------------
Standard 7-point stencil on a structured grid. Axis 0 is the strip direction and
must be non-periodic; axes 1 and 2 may be periodic.

    Dirichlet axis d:  N_d interior nodes, h_d = L_d/(N_d+1),
                       x_i = (i+1) h_d          (boundary nodes excluded)
    Periodic axis d:   N_d nodes,          h_d = L_d/N_d,
                       x_i = i h_d              (no boundary nodes)

The two conventions differ because a periodic axis has no boundary to exclude.
This also makes a periodic axis *exactly* nested under coarsening (coarse node I
coincides with fine node 2I), which a Dirichlet axis is not — see the
transfer-operator discussion in multigrid.py.

Why line and not plane relaxation
---------------------------------
The strip operator is

    A_line = tridiag( 1/h0²,  -2(1/h0² + 1/h1² + 1/h2²),  1/h0² )

which is TST, of size N0, and needs log₂(N0) qubits — identical in form to the
1D and 2D cases, so ``build_tst_block_encoding`` applies unchanged. Its
condition number tends to 2 on an isotropic grid, *better* than the 2D value of
3, because both transverse directions add to the diagonal.

A plane smoother would instead hand the inner solver an N²×N² system with a
5-point stencil on 2·log₂(N) qubits, requiring a block encoding this repository
does not have. Its conditioning would be fine (κ → 5, bounded by the same
diagonal shift), so the objection is structural rather than numerical — but it
is decisive.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


class PoissonLine3D:
    """
    ∇²u = f on a box, decomposed into strips along axis 0.

    Attributes
    ----------
    shape : tuple[int, int, int]
        (N0, N1, N2) node counts. Dirichlet axes count interior nodes only.
    lengths : tuple[float, float, float]
        Physical extent of each axis.
    periodic : tuple[bool, bool, bool]
        Per-axis periodicity. Axis 0 is always False.
    spacings : tuple[float, float, float]
        Mesh spacing per axis: L/N on a periodic axis, L/(N+1) on a Dirichlet
        axis, reflecting whether boundary nodes are excluded.
    level : int
        Multigrid level index; 0 is the finest grid.
    f : np.ndarray
        (N0, N1, N2) source field.
    bc_lo, bc_hi : list
        Per-axis boundary data. Unused entries correspond to periodic axes.

    Examples
    --------
    Unit cube, all Dirichlet::

        PoissonLine3D(f, lengths=(1.0, 1.0, 1.0))

    HET channel unwrapped to a slab, azimuthally periodic::

        PoissonLine3D(f, lengths=(L_axial, L_radial, L_azimuthal),
                      periodic=(False, False, True))
    """

    MIN_STRIP = 4          # quantum register needs >= 2 qubits

    # An axis is coarsened only if its spacing is within this factor of the
    # finest axis.  See coarsen() for why this is not optional.
    COARSEN_RATIO = 2.0

    def __init__(
        self,
        f_values: np.ndarray,
        lengths: Sequence[float] = (1.0, 1.0, 1.0),
        periodic: Sequence[bool] = (False, False, False),
        bc_lo: Sequence = (0.0, 0.0, 0.0),
        bc_hi: Sequence = (0.0, 0.0, 0.0),
        _level: int = 0,
    ) -> None:
        """
        Assembles the strip operator and the boundary-absorbed right-hand side.

        Parameters
        ----------
        f_values : np.ndarray
            (N0, N1, N2) source term.
        lengths : Sequence[float]
            Physical extent of each axis.
        periodic : Sequence[bool]
            Per-axis periodicity. Axis 0 must be False — it is the strip
            direction, and a periodic strip operator is not tridiagonal (it
            carries corner entries), which would break the TST block encoding.
        bc_lo, bc_hi : Sequence
            Per-axis boundary data, each entry a scalar or an array over the
            remaining two axes. Ignored for periodic axes.
        _level : int
            Multigrid level index, set internally by ``coarsen``.

        Raises
        ------
        ValueError
            If ``f_values`` is not 3-D, or if axis 0 is marked periodic.
        """
        f_values = np.asarray(f_values, dtype=float)
        if f_values.ndim != 3:
            raise ValueError(f"f_values must be 3-D, got shape {f_values.shape}")

        self.shape = tuple(int(n) for n in f_values.shape)
        self.lengths = tuple(float(L) for L in lengths)
        self.periodic = tuple(bool(p) for p in periodic)
        if self.periodic[0]:
            raise ValueError(
                "axis 0 is the strip direction and must be non-periodic: a "
                "periodic strip operator is cyclic-tridiagonal, not TST, and "
                "the quantum block encoding assumes TST.")
        self.level = _level

        # Dirichlet axes exclude their boundary nodes; periodic axes do not.
        self.spacings = tuple(
            L / n if per else L / (n + 1)
            for L, n, per in zip(self.lengths, self.shape, self.periodic))

        self.f = f_values
        self.bc_lo = list(bc_lo)
        self.bc_hi = list(bc_hi)

        self._A = self._build_row_matrix()
        self._rhs = self._build_rhs()

    # ── Convenience Aliases Matching the 2D Class ─────────────────────────────

    @property
    def dx(self) -> float:
        """Spacing along axis 0, the strip direction."""
        return self.spacings[0]

    @property
    def dy(self) -> float:
        """Spacing along axis 1."""
        return self.spacings[1]

    @property
    def dz(self) -> float:
        """Spacing along axis 2."""
        return self.spacings[2]

    # ── Operator ──────────────────────────────────────────────────────────────

    def _diag(self) -> float:
        """Diagonal entry of the 7-point stencil, -2·Σ_d 1/h_d²."""
        return -2.0 * sum(1.0 / h**2 for h in self.spacings)

    def _build_row_matrix(self) -> np.ndarray:
        """Assembles the (N0, N0) TST strip operator along axis 0."""
        n = self.shape[0]
        a, b = self._diag(), 1.0 / self.spacings[0] ** 2
        return (a * np.eye(n)
                + b * (np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)))

    def _build_rhs(self) -> np.ndarray:
        """Source term with Dirichlet contributions folded in."""
        r = self.f.copy()
        for ax in range(3):
            if self.periodic[ax]:
                continue
            inv_h2 = 1.0 / self.spacings[ax] ** 2
            lo = np.asarray(self.bc_lo[ax], dtype=float)
            hi = np.asarray(self.bc_hi[ax], dtype=float)
            face = tuple(n for d, n in enumerate(self.shape) if d != ax)
            lo_idx = [slice(None)] * 3; lo_idx[ax] = 0
            hi_idx = [slice(None)] * 3; hi_idx[ax] = self.shape[ax] - 1
            r[tuple(lo_idx)] -= np.broadcast_to(lo, face) * inv_h2
            r[tuple(hi_idx)] -= np.broadcast_to(hi, face) * inv_h2
        return r

    def row_matrix(self) -> np.ndarray:
        """The N0 × N0 tridiagonal strip operator, satisfying the protocol."""
        return self._A

    def rhs(self) -> np.ndarray:
        """(N0, N1, N2) right-hand side with Dirichlet data already absorbed."""
        return self._rhs

    def apply(self, u: np.ndarray) -> np.ndarray:
        """
        7-point Laplacian with homogeneous exterior / periodic wrap.

        Parameters
        ----------
        u : np.ndarray
            (N0, N1, N2) field.

        Returns
        -------
        np.ndarray
            (N0, N1, N2) result of applying the discrete Laplacian to u.
        """
        out = self._diag() * u
        for ax in range(3):
            inv_h2 = 1.0 / self.spacings[ax] ** 2
            if self.periodic[ax]:
                out += (np.roll(u, 1, axis=ax) + np.roll(u, -1, axis=ax)) * inv_h2
            else:
                lo = [slice(None)] * 3; hi = [slice(None)] * 3
                lo[ax] = slice(1, None); hi[ax] = slice(None, -1)
                out[tuple(lo)] += u[tuple(hi)] * inv_h2
                out[tuple(hi)] += u[tuple(lo)] * inv_h2
        return out

    # ── Coarsening ────────────────────────────────────────────────────────────

    def coarsen(self) -> Optional["PoissonLine3D"]:
        """
        Coarsen, halving only those axes whose spacing is within
        COARSEN_RATIO of the finest axis (anisotropic semi-coarsening).

        On an isotropic grid every axis qualifies and this reduces to
        standard full coarsening: all axes stay powers of two, the strip
        operator stays TST, and κ(A_line) stays ~2 at every level
        because the spacing ratios are preserved.

        On an anisotropic grid, coarsening every axis regardless is not a
        mild inefficiency, it breaks the method.  The HET channel unwrapped
        to a slab has h = (2.78, 1.67, 33.4) mm at N=8: the azimuthal
        spacing is twenty times the radial, so azimuthal coupling (~1/h²)
        is some four hundred times weaker and the problem is very nearly a
        stack of decoupled (axial, radial) planes.  Coarsening that
        near-decoupled direction produces a coarse operator that does not
        represent the fine one, and the V-cycle degrades from ρ ~ 0.17 to
        ρ ~ 0.94 — measurably *worse* than plain SOR.  Restricting
        coarsening to the strongly-coupled axes restores ρ ~ 0.2 and
        additionally equalises the spacings as the hierarchy descends, so
        the weak axis becomes eligible once the others have caught up.

        Coarse levels carry the error equation, hence zero source and
        homogeneous boundaries.

        Returns
        -------
        PoissonLine3D or None
            The next coarser level, or None once no axis can be halved.
        """
        h_min = min(self.spacings)
        do = [h <= self.COARSEN_RATIO * h_min for h in self.spacings]
        # An axis that is already at the floor, or odd, cannot be halved.
        do = [d and n > self.MIN_STRIP and n % 2 == 0
              for d, n in zip(do, self.shape)]
        if not any(do):
            return None
        return PoissonLine3D(
            np.zeros(tuple(n // 2 if d else n
                           for n, d in zip(self.shape, do))),
            lengths=self.lengths, periodic=self.periodic,
            _level=self.level + 1)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the (N0, N1, N2) coordinate matrices in 'ij' index order.

        Periodic axes start at 0; Dirichlet axes start at one spacing in,
        their boundary nodes being excluded from the unknowns.
        """
        axes = []
        for n, h, per in zip(self.shape, self.spacings, self.periodic):
            axes.append(np.arange(n) * h if per else np.arange(1, n + 1) * h)
        return np.meshgrid(*axes, indexing="ij")

    def kappa_row(self) -> float:
        """
        Spectral condition number κ(A_line) of the strip operator.

        Tends to 2⁻ as N → ∞ on an isotropic grid — better conditioned than
        the 2D strip operator, since both transverse directions contribute to
        the diagonal.
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
        per = "".join("P" if p else "D" for p in self.periodic)
        return (f"PoissonLine3D({self.shape[0]}x{self.shape[1]}x{self.shape[2]}, "
                f"bc={per}, h={tuple(round(h, 6) for h in self.spacings)}, "
                f"kappa={self.kappa_row():.4f}, level={self.level})")
