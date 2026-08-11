"""
3-D Poisson problem at fourth order, decomposed into line-relaxation strips.

The fourth-order counterpart of ``problems/poisson_line_3d.py``, and the 3-D
counterpart of ``problems/poisson_line_2d_4th.py``, whose module docstring
carries the derivation of the scheme and of the boundary closure in full. Only
what is specific to three dimensions is repeated here.

As at second order, moving to 3-D requires no change to any quantum solver and
no new solver module: the strip is still a one-dimensional system of size N₀ on
log₂(N₀) qubits, and only the problem class and the grid transfers are new. What
*is* new relative to the second-order 3-D class is that the strip operator is
pentadiagonal rather than tridiagonal, so the strip solves go through the dense
block encoding (``hhl_4th``/``qsvt_4th``) rather than the TST one.

Discretisation
--------------
Standard fourth-order five-point stencil applied along each of the three axes
and summed — a wide 19-point stencil overall. Axis 0 is the strip direction and
must be non-periodic; axes 1 and 2 may be periodic. Writing c_d = 1/(12·h_d²),
the strip operator for a strip touching no transverse boundary is

    A_row = pentadiag( −c₀, 16c₀, −30(c₀ + c₁ + c₂), 16c₀, −c₀ )

with the transverse coupling entering only through the diagonal shift, and the
off-diagonal transverse terms carried to the right-hand side by
``transverse_terms``.

Distinct strip operators
------------------------
The odd reflection at a transverse boundary folds the ghost node onto the
strip's own diagonal, so a strip adjacent to a transverse boundary carries
A_row + c_d·I for each such adjacency. With two Dirichlet transverse axes there
are therefore **four** distinct operators — none, axis 1, axis 2, both — and
with one periodic transverse axis only two. The count is independent of N, which
is what keeps the quantum cost bounded: one block encoding and one set of QSP
phase angles per distinct matrix, not per strip. ``kappa_rows`` enumerates them,
and every key it returns must be present in the phase cache before a QSVT sweep
is submitted.

Conditioning
------------
κ(A_row) is bounded as N → ∞ and is *better* in 3-D than in 2-D, both
transverse directions contributing to the diagonal — the same mechanism that
gives κ → 2⁻ against 3⁻ at second order. The fourth-order operator is somewhat
worse conditioned than the second-order one at equal N, in the 4/3 proportion
measured in 1-D, and remains O(1) rather than O(N²).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from problems.poisson_line_2d_4th import (apply_axis_4th, extrapolate_face,
                                          normal_second_derivative)


class PoissonLine3D4th:
    """
    ∇²u = f on a box, discretised to O(h⁴) and decomposed into strips along
    axis 0.

    Attributes
    ----------
    shape : tuple[int, int, int]
        (N0, N1, N2) node counts. Dirichlet axes count interior nodes only.
        Every axis must carry at least ``MIN_STRIP`` nodes.
    lengths : tuple[float, float, float]
        Physical extent of each axis [m].
    periodic : tuple[bool, bool, bool]
        Per-axis periodicity. Axis 0 is always False.
    spacings : tuple[float, float, float]
        Mesh spacing per axis: L/N on a periodic axis, L/(N+1) on a Dirichlet
        axis, reflecting whether boundary nodes are excluded.
    level : int
        Multigrid level index; 0 is the finest grid.
    f : np.ndarray
        (N0, N1, N2) source field at the interior nodes.
    bc_lo, bc_hi : list of np.ndarray
        Per-axis Dirichlet data, each of the shape of the face normal to that
        axis. Entries for periodic axes are unused.
    f_lo, f_hi : list of np.ndarray
        Per-axis source evaluated *on* the faces, resolved from the constructor
        arguments where given and by cubic extrapolation otherwise.
    unn_lo, unn_hi : list of np.ndarray
        Per-axis ∂²u/∂n² on the faces — the quantity the closure actually
        requires, being f on the face less the tangential second derivatives of
        the Dirichlet data. Retained for diagnostics: an inaccurate value here
        degrades the order of the scheme with no other visible symptom.

    Examples
    --------
    Unit cube, all Dirichlet::

        PoissonLine3D4th(f, lengths=(1.0, 1.0, 1.0))

    HET channel unwrapped to a slab, azimuthally periodic::

        PoissonLine3D4th(f, lengths=(L_axial, L_radial, L_azimuthal),
                         periodic=(False, False, True))
    """

    #: Quantum register needs >= 2 qubits; the five-point stencil independently
    #: needs n >= 4. The two floors coincide.
    MIN_STRIP = 4

    #: An axis is coarsened only if its spacing is within this factor of the
    #: finest axis. See ``coarsen`` — on the HET slab this is not optional.
    COARSEN_RATIO = 2.0

    def __init__(
        self,
        f_values: np.ndarray,
        lengths: Sequence[float] = (1.0, 1.0, 1.0),
        periodic: Sequence[bool] = (False, False, False),
        bc_lo: Sequence = (0.0, 0.0, 0.0),
        bc_hi: Sequence = (0.0, 0.0, 0.0),
        f_lo: Optional[Sequence] = None,
        f_hi: Optional[Sequence] = None,
        _level: int = 0,
    ) -> None:
        """
        Assembles the strip operators and the boundary-absorbed right-hand side.

        Parameters
        ----------
        f_values : np.ndarray
            (N0, N1, N2) source term at the interior nodes.
        lengths : Sequence[float]
            Physical extent of each axis [m].
        periodic : Sequence[bool]
            Per-axis periodicity. Axis 0 must be False — it is the strip
            direction, and a periodic strip operator carries corner entries,
            which the block encodings do not represent.
        bc_lo, bc_hi : Sequence
            Per-axis Dirichlet data, each entry a scalar or an array over the
            remaining two axes. Ignored for periodic axes.
        f_lo, f_hi : Sequence, optional
            Per-axis source evaluated on the faces, each entry a scalar or an
            array over the remaining two axes. Required data for the
            fourth-order closure; omitted, they are extrapolated from the
            interior samples.
        _level : int
            Multigrid level index, set internally by ``coarsen``.

        Raises
        ------
        ValueError
            If ``f_values`` is not 3-D, if axis 0 is marked periodic, or if any
            axis carries fewer than ``MIN_STRIP`` nodes.
        """
        f_values = np.asarray(f_values, dtype=float)
        if f_values.ndim != 3:
            raise ValueError(f"f_values must be 3-D, got shape {f_values.shape}")

        self.shape = tuple(int(n) for n in f_values.shape)
        if min(self.shape) < self.MIN_STRIP:
            raise ValueError(
                f"the fourth-order stencil spans two nodes either side of its "
                f"centre, so every axis needs at least {self.MIN_STRIP} nodes; "
                f"got {self.shape}.")

        self.lengths = tuple(float(L) for L in lengths)
        self.periodic = tuple(bool(p) for p in periodic)
        if self.periodic[0]:
            raise ValueError(
                "axis 0 is the strip direction and must be non-periodic: a "
                "periodic strip operator is cyclic-pentadiagonal, not banded, "
                "and the dense block encoding assumes the banded form.")
        self.level = _level

        # Dirichlet axes exclude their boundary nodes; periodic axes do not.
        self.spacings = tuple(
            L / n if per else L / (n + 1)
            for L, n, per in zip(self.lengths, self.shape, self.periodic))

        self.f = f_values
        #: 1/(12h²) per axis: the prefactor of the five-point stencil.
        self._c = tuple(1.0 / (12.0 * h ** 2) for h in self.spacings)

        f_lo = list(f_lo) if f_lo is not None else [None, None, None]
        f_hi = list(f_hi) if f_hi is not None else [None, None, None]

        self.bc_lo, self.bc_hi = [], []
        self.f_lo, self.f_hi = [], []
        self.unn_lo, self.unn_hi = [], []
        for ax in range(3):
            face = self._face_shape(ax)
            lo = np.broadcast_to(np.asarray(bc_lo[ax], dtype=float), face).copy()
            hi = np.broadcast_to(np.asarray(bc_hi[ax], dtype=float), face).copy()
            self.bc_lo.append(lo)
            self.bc_hi.append(hi)

            f0 = self._resolve_face(f_lo[ax], ax, upper=False)
            f1 = self._resolve_face(f_hi[ax], ax, upper=True)
            self.f_lo.append(f0)
            self.f_hi.append(f1)

            # The closure needs ∂²u/∂n², which the PDE gives only after the two
            # tangential second derivatives are removed. Those are derivatives
            # of the Dirichlet data alone; see poisson_line_2d_4th for why the
            # natural generalisation - using f itself - is second order.
            tang = self._tangential(ax)
            self.unn_lo.append(normal_second_derivative(f0, lo, tang))
            self.unn_hi.append(normal_second_derivative(f1, hi, tang))

        self._A_cache: dict[tuple[int, int], np.ndarray] = {}
        self._A_int = self._build_row_matrix((0, 0))
        self._A_cache[(0, 0)] = self._A_int
        self._rhs = self._build_rhs()

    # ── Face geometry ─────────────────────────────────────────────────────────

    def _face_shape(self, axis: int) -> tuple[int, ...]:
        """Shape of the face normal to ``axis``: the other two extents, in order."""
        return tuple(n for d, n in enumerate(self.shape) if d != axis)

    def _tangential(self, axis: int) -> tuple[tuple[int, float, bool], ...]:
        """
        The (face axis, spacing, periodicity) triples of a face's own axes.

        The face normal to ``axis`` retains the other two volume axes in their
        original relative order, so volume axis d maps to face axis 0 or 1
        according to whether it precedes or follows ``axis``.
        """
        others = [d for d in range(3) if d != axis]
        return tuple((k, self.spacings[d], self.periodic[d])
                     for k, d in enumerate(others))

    def _resolve_face(self, supplied, axis: int, upper: bool) -> np.ndarray:
        """
        Resolves the source on one face, by supply or by cubic extrapolation.

        Parameters
        ----------
        supplied : float, np.ndarray or None
            The caller's face values, if any.
        axis : int
            Axis normal to the face.
        upper : bool
            Which end of ``axis`` the face lies beyond.

        Returns
        -------
        np.ndarray
            Source on the face, of the shape ``_face_shape(axis)`` returns.
        """
        if supplied is None:
            return np.asarray(extrapolate_face(self.f, axis, upper), dtype=float)
        return np.broadcast_to(np.asarray(supplied, dtype=float),
                               self._face_shape(axis)).copy()

    # ── Operator ──────────────────────────────────────────────────────────────

    def _build_row_matrix(self, adjacency: tuple[int, int]) -> np.ndarray:
        """
        Assembles the (N0, N0) pentadiagonal strip operator for one adjacency.

        Parameters
        ----------
        adjacency : tuple[int, int]
            Number of transverse boundaries the strip touches along axes 1 and
            2 respectively — 0 or 1 in every realistic case, 2 only on an axis
            of a single node, which ``MIN_STRIP`` precludes.

        Returns
        -------
        np.ndarray
            (N0, N0) symmetric pentadiagonal strip operator.
        """
        n = self.shape[0]
        c0, c1, c2 = self._c

        A = np.zeros((n, n))
        np.fill_diagonal(A, -30.0 * (c0 + c1 + c2))
        np.fill_diagonal(A[1:, :], 16.0 * c0)
        np.fill_diagonal(A[:, 1:], 16.0 * c0)
        np.fill_diagonal(A[2:, :], -c0)
        np.fill_diagonal(A[:, 2:], -c0)

        # Ghost fold in the strip direction, at each end of every strip. The
        # odd reflection puts this on the DIAGONAL; folding it into A[0,1]
        # instead is an even reflection, i.e. a Neumann condition.
        A[0, 0] += c0
        A[-1, -1] += c0

        # Ghost fold from each adjacent transverse boundary, onto the whole
        # diagonal - the reflected node is the strip's own value.
        shift = adjacency[0] * c1 + adjacency[1] * c2
        if shift:
            A += shift * np.eye(n)
        return A

    def _adjacency(self, idx: tuple[int, ...]) -> tuple[int, int]:
        """
        How many transverse boundaries the strip at ``idx`` touches, per axis.

        A periodic axis has no boundary and therefore never contributes.
        """
        out = []
        for k, ax in enumerate((1, 2)):
            if self.periodic[ax]:
                out.append(0)
                continue
            n = self.shape[ax]
            out.append(int(idx[k] == 0) + int(idx[k] == n - 1))
        return (out[0], out[1])

    def _build_rhs(self) -> np.ndarray:
        """
        Absorbs the Dirichlet data and the face closures into the source field.

        Per non-periodic axis, with c = 1/(12h²), boundary value g and ∂²u/∂n²
        the normal second derivative on that face:

            first interior slab:   −14·c·g   and   + (∂²u/∂n²)/12
            second interior slab:  + c·g

        symmetrically at the far end. The −14 is −16 from the known boundary
        node and +2 from the ghost, which subtract rather than accumulate; the
        second term is what lifts the closure from O(h²) to O(h⁴).

        Returns
        -------
        np.ndarray
            (N0, N1, N2) right-hand side.
        """
        r = self.f.copy()
        for ax in range(3):
            if self.periodic[ax]:
                continue
            c = self._c[ax]
            n = self.shape[ax]

            def sl(i):
                idx = [slice(None)] * 3
                idx[ax] = i
                return tuple(idx)

            r[sl(0)] += -14.0 * c * self.bc_lo[ax] + self.unn_lo[ax] / 12.0
            r[sl(1)] += c * self.bc_lo[ax]
            r[sl(n - 1)] += -14.0 * c * self.bc_hi[ax] + self.unn_hi[ax] / 12.0
            r[sl(n - 2)] += c * self.bc_hi[ax]
        return r

    def row_matrix(self) -> np.ndarray:
        """
        The N0 × N0 strip operator for a strip touching no transverse boundary.

        Satisfies the ``LineProblem2D`` protocol, which every scheme is written
        against. The schemes reach the individual strips through
        ``row_matrix_for``, which distinguishes the boundary-adjacent ones.
        """
        return self._A_int

    def row_matrix_for(self, idx: tuple[int, ...]) -> np.ndarray:
        """
        The strip operator for the strip at transverse index ``idx``.

        Returns the *same array object* for every strip sharing an operator, so
        a caller may cache a block encoding or a set of QSP phase angles keyed
        on identity. There are at most four such objects, independent of N.

        Parameters
        ----------
        idx : tuple of int
            Transverse index ``(j, k)``.

        Returns
        -------
        np.ndarray
            (N0, N0) strip operator for that strip's boundary adjacency.
        """
        key = self._adjacency(idx)
        A = self._A_cache.get(key)
        if A is None:
            A = self._build_row_matrix(key)
            self._A_cache[key] = A
        return A

    def transverse_terms(self, axis: int, index: int,
                         n: int) -> tuple[tuple[int, float], ...]:
        """
        The transverse neighbours to gather into a strip's right-hand side.

        The four coefficients of the fourth-order stencil, in ascending offset
        order. ``strip_sweep`` wraps them on a periodic axis and discards those
        falling outside a Dirichlet one, where the contribution is boundary data
        already carried by ``rhs()`` and the node beyond it is the ghost, folded
        onto the diagonal by ``row_matrix_for``.

        Parameters
        ----------
        axis : int
            Index into ``shape``; 1 or 2 in 3-D.
        index : int
            Position of the strip along ``axis``. Unused — the coefficients are
            uniform, the boundary strips being distinguished by the operator and
            the right-hand side instead.
        n : int
            Extent of ``axis``. Unused, for the same reason.

        Returns
        -------
        tuple of (int, float)
            ((−2, −c), (−1, 16c), (1, 16c), (2, −c)) with c = 1/(12·h_axis²).
        """
        c = self._c[axis]
        return ((-2, -c), (-1, 16.0 * c), (1, 16.0 * c), (2, -c))

    def apply(self, u: np.ndarray) -> np.ndarray:
        """
        The full fourth-order Laplacian with homogeneous exterior / periodic wrap.

        Parameters
        ----------
        u : np.ndarray
            (N0, N1, N2) field at the interior nodes.

        Returns
        -------
        np.ndarray
            (N0, N1, N2) result of applying the discrete Laplacian to u.
        """
        out = apply_axis_4th(u, 0, self.spacings[0], self.periodic[0])
        for ax in (1, 2):
            out = out + apply_axis_4th(u, ax, self.spacings[ax],
                                       self.periodic[ax])
        return out

    def rhs(self) -> np.ndarray:
        """(N0, N1, N2) right-hand side with Dirichlet data already absorbed."""
        return self._rhs

    # ── Convenience aliases matching the 2-D class ────────────────────────────

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

    # ── Coarsening ────────────────────────────────────────────────────────────

    def coarsen(self) -> Optional["PoissonLine3D4th"]:
        """
        Coarsen, halving only those axes whose spacing is within
        COARSEN_RATIO of the finest axis (anisotropic semi-coarsening).

        Identical in policy to ``PoissonLine3D.coarsen``: coarsening is a
        property of the grid, not of the stencil. The policy is not optional on
        an anisotropic grid — the HET slab has an azimuthal spacing twenty times
        the radial one, and coarsening that near-decoupled direction degrades
        the V-cycle from ρ ≈ 0.17 to ρ ≈ 0.94, measurably worse than plain SOR.
        Restricting coarsening to the strongly coupled axes restores ρ ≈ 0.2 and
        equalises the spacings as the hierarchy descends, so the weak axis
        becomes eligible once the others have caught up.

        Coarse levels carry the error equation, hence zero source, homogeneous
        boundaries and — since the closure's ∂²u/∂n² term vanishes with both —
        no face data to propagate.

        Returns
        -------
        PoissonLine3D4th or None
            The next coarser level, or None once no axis can be halved.
        """
        h_min = min(self.spacings)
        do = [h <= self.COARSEN_RATIO * h_min for h in self.spacings]
        do = [d and n > self.MIN_STRIP and n % 2 == 0
              for d, n in zip(do, self.shape)]
        if not any(do):
            return None
        return PoissonLine3D4th(
            np.zeros(tuple(n // 2 if d else n
                           for n, d in zip(self.shape, do))),
            lengths=self.lengths, periodic=self.periodic,
            _level=self.level + 1)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns the (N0, N1, N2) coordinate matrices in 'ij' index order.

        Periodic axes start at 0; Dirichlet axes start at one spacing in, their
        boundary nodes being excluded from the unknowns.
        """
        axes = []
        for n, h, per in zip(self.shape, self.spacings, self.periodic):
            axes.append(np.arange(n) * h if per else np.arange(1, n + 1) * h)
        return np.meshgrid(*axes, indexing="ij")

    def kappa_row(self) -> float:
        """
        Spectral condition number κ(A_row) of the interior strip operator.

        Bounded as N → ∞, and better than the 2-D value: both transverse
        directions contribute to the diagonal.
        """
        e = np.abs(np.linalg.eigvalsh(self._A_int))
        return float(e.max() / e.min())

    def kappa_rows(self) -> dict[tuple[int, int], float]:
        """
        κ of every strip operator the grid actually calls for.

        Enumerated over the adjacencies that occur, rather than over the four
        that could in principle: a periodic transverse axis contributes none.

        Returns
        -------
        dict
            {(adjacency along axis 1, along axis 2): κ}. Every key needs its own
            block encoding and its own set of QSP phase angles, so all of them
            must be in the phase cache before a QSVT sweep is submitted.
        """
        n1, n2 = self.shape[1], self.shape[2]
        js = (0,) if self.periodic[1] else (0, 1, n1 - 1)
        ks = (0,) if self.periodic[2] else (0, 1, n2 - 1)
        keys = {self._adjacency((j, k)) for j in js for k in ks}

        out = {}
        for key in sorted(keys):
            A = self._A_cache.get(key)
            if A is None:
                A = self._build_row_matrix(key)
                self._A_cache[key] = A
            e = np.abs(np.linalg.eigvalsh(A))
            out[key] = float(e.max() / e.min())
        return out

    def residual(self, u: np.ndarray) -> float:
        """Relative 2-norm residual of the full coupled system."""
        b = self.rhs()
        bn = np.linalg.norm(b)
        r = np.linalg.norm(b - self.apply(u))
        return float(r / bn) if bn > 1e-300 else float(r)

    def __repr__(self) -> str:
        per = "".join("P" if p else "D" for p in self.periodic)
        return (f"PoissonLine3D4th({self.shape[0]}x{self.shape[1]}x"
                f"{self.shape[2]}, bc={per}, "
                f"h={tuple(round(h, 6) for h in self.spacings)}, "
                f"kappa={self.kappa_row():.4f}, level={self.level})")
