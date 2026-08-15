"""
2-D Poisson problem at fourth order, decomposed into line-relaxation strips.

The fourth-order counterpart of ``problems/poisson_line_2d.py``, and the *problem*
half of the outer/inner architecture in ``solvers/outer``: a concrete
implementation of the ``LineProblem2D`` protocol together with the optional
``HigherOrderTransverse`` extension (both defined in ``solvers/outer/core.py``),
so that every outer scheme (SOR, multigrid, …) and every inner solver (dense
direct, HHL, VQLS, QSVT) drives it without modification.

Discretisation
--------------
The Laplacian is the sum of two one-dimensional operators, each discretised by
the five-point centred difference that ``problems/poisson_1d_4th.py`` uses:

    ∂²u/∂x²|ᵢⱼ ≈ (−uᵢ₋₂,ⱼ + 16uᵢ₋₁,ⱼ − 30uᵢⱼ + 16uᵢ₊₁,ⱼ − uᵢ₊₂,ⱼ) / (12 dx²)

and likewise along y, giving a wide 9-point stencil of formal order O(h⁴) in
each direction. This is the tensor sum of two fourth-order operators, not the
compact Mehrstellen nine-point scheme; the two are both fourth order, but only
the tensor sum leaves the strip operator banded in one direction alone, which is
what makes the line decomposition — and hence the quantum inner solvers — apply
unchanged.

True fourth order in *both* directions is deliberate. The mixed alternative —
fourth order along the strip, second order transverse — is capped at order 2 by
construction, measured 1.95 even with an exact strip operator, because the
transverse truncation error is O(h²) irrespective of how the strip is solved.
Reaching order 4 therefore requires the transverse stencil to reach j±2, which
is what ``transverse_terms`` supplies, and requires the strips adjacent to a
transverse boundary to carry a different operator, which is what
``row_matrix_for`` supplies.

Sign and scaling convention
---------------------------
The *physical* (unscaled) convention of ``PoissonLine2D`` is retained: the
1/(12h²) prefactors are carried in the operator rather than folded into the
right-hand side. The 1-D class folds 12h² into b to keep A integral, which is
not available here — dx and dy are independent, so no single scalar clears both
prefactors — and the physical form is in any case what extends to the
non-square cells of the HET geometry and to 3-D.

Writing cₓ = 1/(12 dx²) and c_y = 1/(12 dy²), the strip operator for an interior
strip is the symmetric pentadiagonal matrix

    A_row = pentadiag( −cₓ, 16cₓ, −30(cₓ + c_y), 16cₓ, −cₓ )

and the coupled system solved on strip j is

    A_row · u[:,j] + Σ_{s ∈ {−2,−1,1,2}} γ_s · u[:,j+s] = rhs[:,j],
    γ_{±1} = 16c_y,   γ_{±2} = −c_y

which is precisely the (offset, coefficient) list ``transverse_terms`` returns.

Boundary closure
----------------
Each direction carries the closure derived and corrected in
``problems/poisson_1d_4th.py``. At the first interior node the stencil reaches a
ghost node one spacing outside the domain, eliminated by the odd reflection
carried to fourth order:

    u₋₁ = 2α − u₁ + h²·u″(0) + O(h⁴) = 2α − u₁ + h²·f(0) + O(h⁴)

the last equality being the governing equation evaluated *on* the boundary,
which supplies the second-derivative term at no cost. Substituting into the
first interior row, the ghost contributes +u₁ to the operator and −2α + h²·f(0)
to the data, whilst the known boundary node contributes a further +16α:

    operator:   A[0,0] += c        (c = 1/(12h²) for the direction concerned)
    data:       b[0]   -= 14·c·α   and   b[0] += f(0)/12

Two errors are possible here and both were present in the retired
``solvers/outer/multigrid_4th.py``, which this module replaces:

* **The reflection must be odd, not even.** The +u₁ term belongs on the
  *diagonal*, A[0,0] += c. Folding it into the first off-diagonal instead
  (A[0,1] += −c) imposes a Neumann condition on Dirichlet data.
* **The coefficient is 14α, not 18α.** The boundary node contributes +16α and
  the ghost −2α; they subtract. Summing them with a common sign destroys
  convergence outright whenever α ≠ 0.

Neither defect is visible on a solution odd about both boundaries — where the
plain reflection happens to be exact — and −sin(πx)/π² is exactly such a
solution. Any verification of this module must therefore include at least one
solution that is *not* odd about the boundaries and one with non-zero Dirichlet
data; see ``tests/test_poisson_line_4th.py``.

In the transverse direction the same closure applies, but the folded ghost term
lands on the diagonal of the *strip* operator rather than on a single entry. The
strips adjacent to a transverse boundary consequently carry

    A_row^bnd = A_row^int + c_y · I

which is why ``row_matrix_for`` exists. The number of *distinct* strip operators
is two in 2-D — interior and boundary-adjacent — independent of N, so the
quantum cost is bounded: one block encoding and one set of QSP phase angles per
distinct matrix, not per strip.

Boundary source data
--------------------
f evaluated *on* the four faces is required data for the closure, not a
refinement. Where the caller does not supply it, it is recovered by cubic
Lagrange extrapolation from the four nearest interior samples: O(h⁴) accurate
and therefore order-preserving, since the term enters the boundary row divided
by 12. A linear or constant extrapolation is not sufficient — this term carries
an O(1) weight in that row.

Condition number
----------------
κ(A_row) is bounded as N → ∞ exactly as at second order, the transverse
coupling supplying a diagonal shift that does not vanish with h. On the unit
square the fourth-order strip operator is somewhat worse conditioned than the
second-order one at equal N, in the same 4/3 asymptotic proportion measured in
1-D, and remains O(1) rather than O(N²) — the property that makes the line
decomposition tractable for the quantum inner solvers in the first place.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# -- Shared N-dimensional primitives -------------------------------------------
# Used by this module and by problems/poisson_line_3d_4th.py. They are written
# once, against an arbitrary number of axes, because the alternative is ~150
# lines of index arithmetic duplicated between the two classes, where a
# divergence would not be visible as a test failure in either one alone.

#: Newton-Gregory backward-extrapolation weights over four equispaced samples,
#: extrapolating one interval beyond the block. Error O(h⁴).
_EXTRAP_WEIGHTS = np.array([4.0, -6.0, 4.0, -1.0])


def _axis_slice(ndim: int, axis: int, s) -> tuple:
    """
    Builds an index tuple selecting ``s`` along ``axis`` and everything else.

    Parameters
    ----------
    ndim : int
        Rank of the array being indexed.
    axis : int
        Axis to which ``s`` applies.
    s : slice or int
        Selection along ``axis``.

    Returns
    -------
    tuple
        Index tuple suitable for ``ndarray.__getitem__``.
    """
    idx = [slice(None)] * ndim
    idx[axis] = s
    return tuple(idx)


def apply_axis_4th(u: np.ndarray, axis: int, h: float,
                   periodic: bool = False) -> np.ndarray:
    """
    Applies the fourth-order second-derivative operator along one axis.

    Homogeneous exterior data throughout: the boundary node is zero and the
    ghost node is its odd reflection, u₋₁ = −u₁, the h²·f(0) term vanishing with
    the source. This is the form required for residual evaluation
    (r = rhs() − apply(u)) and on coarse multigrid levels, which carry the error
    equation and therefore homogeneous source and boundary data alike.

    The ghost fold is what makes this consistent with the assembled operator:
    the +u₁ it contributes is the same term ``row_matrix_for`` places on the
    diagonal, so the residual measured here is the residual of the system the
    strips actually solve.

    Parameters
    ----------
    u : np.ndarray
        Field of any rank; the operator acts along ``axis`` alone.
    axis : int
        Axis along which to differentiate.
    h : float
        Mesh spacing along ``axis`` [m].
    periodic : bool
        If True, wrap instead of reflecting. A periodic axis has no boundary
        node and hence no ghost node, so all four neighbours are genuine.

    Returns
    -------
    np.ndarray
        Array of the same shape as ``u``, holding ∂²u/∂x_axis² to O(h⁴).
    """
    c = 1.0 / (12.0 * h * h)
    nd = u.ndim
    out = (-30.0 * c) * u

    if periodic:
        out += (16.0 * c) * (np.roll(u, 1, axis=axis) + np.roll(u, -1, axis=axis))
        out += (-c) * (np.roll(u, 2, axis=axis) + np.roll(u, -2, axis=axis))
        return out

    # Neighbours that lie inside the domain. Those that do not are either a
    # boundary node (zero here) or the ghost node, handled by the fold below.
    out[_axis_slice(nd, axis, slice(1, None))] += \
        (16.0 * c) * u[_axis_slice(nd, axis, slice(None, -1))]
    out[_axis_slice(nd, axis, slice(None, -1))] += \
        (16.0 * c) * u[_axis_slice(nd, axis, slice(1, None))]
    out[_axis_slice(nd, axis, slice(2, None))] += \
        (-c) * u[_axis_slice(nd, axis, slice(None, -2))]
    out[_axis_slice(nd, axis, slice(None, -2))] += \
        (-c) * u[_axis_slice(nd, axis, slice(2, None))]

    # Ghost fold: −u₋₁ = +u₁ under the odd reflection about a homogeneous
    # boundary. Symmetrically at the far end.
    out[_axis_slice(nd, axis, 0)] += c * u[_axis_slice(nd, axis, 0)]
    out[_axis_slice(nd, axis, -1)] += c * u[_axis_slice(nd, axis, -1)]
    return out


def second_difference(g: np.ndarray, axis: int, h: float,
                      periodic: bool = False) -> np.ndarray:
    """
    Second derivative along one axis of data sampled at the interior nodes.

    Used on the *boundary data* to form the tangential part of the closure (see
    ``normal_second_derivative``), never on the solution. The three-point
    difference is deliberate rather than economical: its error is h²·g⁗/12,
    which enters the boundary row divided by a further 12, giving h²·g⁗/144 —
    identical in order and in constant to the error the ghost reflection itself
    carries. A wider stencil would refine one of two terms of the same size and
    buy nothing.

    The array spans interior nodes only, so one node beyond each end is
    recovered by the same cubic extrapolation used for the face sources. On a
    constant face — every benchmark whose boundary is held at a fixed potential
    — the extrapolation is exact and the result is identically zero.

    Parameters
    ----------
    g : np.ndarray
        Data on the face, sampled at the interior nodes of ``axis``.
    axis : int
        Axis along which to differentiate.
    h : float
        Mesh spacing along ``axis`` [m].
    periodic : bool
        If True, wrap instead of extrapolating.

    Returns
    -------
    np.ndarray
        Array of the same shape as ``g``, holding ∂²g/∂x_axis².
    """
    if periodic:
        return (np.roll(g, 1, axis=axis) - 2.0 * g
                + np.roll(g, -1, axis=axis)) / (h * h)

    lo = np.expand_dims(extrapolate_face(g, axis, upper=False), axis)
    hi = np.expand_dims(extrapolate_face(g, axis, upper=True), axis)
    ge = np.concatenate([lo, g, hi], axis=axis)
    nd = ge.ndim
    return (ge[_axis_slice(nd, axis, slice(None, -2))]
            - 2.0 * ge[_axis_slice(nd, axis, slice(1, -1))]
            + ge[_axis_slice(nd, axis, slice(2, None))]) / (h * h)


def normal_second_derivative(f_face: np.ndarray, bc_face: np.ndarray,
                             tangential: tuple) -> np.ndarray:
    """
    The second derivative of u *normal* to a face, from the PDE.

    This is the quantity the fourth-order ghost reflection requires, and the
    one place where the multidimensional closure genuinely differs from the
    1-D one rather than merely repeating it per direction.

    In 1-D the governing equation is u″ = f, so the reflection's
    second-derivative term is the source evaluated on the boundary and nothing
    further is needed. In 2-D and 3-D the equation is ∇²u = f, so along the
    normal n

        ∂²u/∂n²|_face = f|_face − Σ_t ∂²u/∂t²|_face

    the sum running over the tangential directions of that face. The tangential
    terms are *known*: u on the face is the Dirichlet data, so ∂²u/∂t² is the
    second derivative of ``bc_face`` and requires no solution values at all.

    Using f alone is the natural but wrong generalisation, and it is not a
    small error: it leaves the whole scheme second-order accurate, with the
    boundary rows carrying a residual of exactly −f|_face/12. It is invisible
    on any solution whose tangential second derivative vanishes on the faces —
    which includes every constant-boundary case and, in particular,
    sin(πx)·sin(πy), the standard test.

    Parameters
    ----------
    f_face : np.ndarray
        Source evaluated on the face.
    bc_face : np.ndarray
        Dirichlet data on the face, same shape as ``f_face``.
    tangential : tuple of (int, float, bool)
        One (axis, spacing, periodic) triple per tangential direction, the axis
        indexing into ``bc_face``'s own axes.

    Returns
    -------
    np.ndarray
        ∂²u/∂n² on the face, same shape as ``f_face``.
    """
    out = np.asarray(f_face, dtype=float).copy()
    for axis, h, periodic in tangential:
        out -= second_difference(np.asarray(bc_face, dtype=float),
                                 axis, h, periodic)
    return out


def extrapolate_face(f: np.ndarray, axis: int, upper: bool) -> np.ndarray:
    """
    Estimates the source on one boundary face from the interior samples.

    Cubic Lagrange extrapolation one interval beyond the four nearest interior
    nodes along ``axis``. The error is O(h⁴), and the value enters the boundary
    row divided by 12, so the fourth order of the scheme is preserved.

    This is the fallback route only. Where the analytical source is known, the
    caller should pass the face values explicitly: extrapolation cannot recover
    a face value that is small against a large interior peak, which is exactly
    the situation of the HET Gaussian sources.

    Parameters
    ----------
    f : np.ndarray
        Interior source field, of any rank.
    axis : int
        Axis normal to the face.
    upper : bool
        False for the face at index −1 (the lower end), True for the face
        beyond index n−1.

    Returns
    -------
    np.ndarray
        Estimated source on the face, of rank ``f.ndim - 1``.

    Raises
    ------
    ValueError
        If ``axis`` carries fewer than four samples, which is below the minimum
        strip length the schemes coarsen to.
    """
    n = f.shape[axis]
    if n < 4:
        raise ValueError(
            f"cubic extrapolation of the face source needs at least 4 samples "
            f"along axis {axis}; got {n}.")
    take = [0, 1, 2, 3] if not upper else [n - 1, n - 2, n - 3, n - 4]
    block = np.take(f, take, axis=axis)
    return np.tensordot(_EXTRAP_WEIGHTS, block, axes=([0], [axis]))


# -- The 2-D problem -----------------------------------------------------------

class PoissonLine2D4th:
    """
    ∇²u = f on [0,Lx] × [0,Ly] with Dirichlet boundaries, discretised to O(h⁴)
    in both directions and decomposed into Ny strips of length Nx along x.

    The fourth-order counterpart of ``PoissonLine2D``, with the same interface
    plus the two optional hooks of ``HigherOrderTransverse``. As there, one
    class covers both the unit-square benchmarks and the HET axial-radial
    channel; the HET case is Lx=Lz, Ly=Lr with a non-zero ``bc_y0``.

    Attributes
    ----------
    shape : tuple[int, int]
        (Nx, Ny) interior node counts along x and y. Both must be at least 4:
        the five-point stencil spans two nodes either side of its centre.
    Lx, Ly : float
        Physical domain extents [m], or unity for non-dimensional benchmarks.
    dx, dy : float
        Mesh spacings, dx = Lx/(Nx+1) and dy = Ly/(Ny+1).
    spacings : tuple[float, float]
        (dx, dy), the form ``strip_sweep`` reads.
    level : int
        Multigrid level index; 0 is the finest grid. Coarse levels carry the
        error equation and therefore homogeneous source and boundary data.
    f : np.ndarray
        (Nx, Ny) source field at the interior nodes.
    bc_x0, bc_x1 : np.ndarray
        Length-Ny Dirichlet data on the x=0 and x=Lx faces.
    bc_y0, bc_y1 : np.ndarray
        Length-Nx Dirichlet data on the y=0 and y=Ly faces.
    f_x0, f_x1 : np.ndarray
        Length-Ny source values *on* the x=0 and x=Lx faces, required by the
        boundary closure. Resolved from the constructor argument where given
        and by cubic extrapolation otherwise; retained because an inaccurate
        value degrades the order of the scheme with no other visible symptom.
    f_y0, f_y1 : np.ndarray
        Length-Nx source values on the y=0 and y=Ly faces, likewise.
    """

    #: Quantum solvers need >= 2 qubits, i.e. n >= 4; the five-point stencil
    #: independently requires n >= 4. The two floors coincide.
    MIN_STRIP = 4

    #: An axis is coarsened only if its spacing is within this factor of the
    #: finer one. See ``PoissonLine2D.coarsen`` and the extended discussion in
    #: ``PoissonLine3D.coarsen``, where anisotropy flips the sign of the
    #: multigrid speedup.
    COARSEN_RATIO = 2.0

    def __init__(
        self,
        f_values: np.ndarray,
        Lx: float = 1.0,
        Ly: float = 1.0,
        bc_x0=0.0, bc_x1=0.0, bc_y0=0.0, bc_y1=0.0,
        f_x0=None, f_x1=None, f_y0=None, f_y1=None,
        _level: int = 0,
    ) -> None:
        """
        Assembles the strip operators and the boundary-absorbed right-hand side.

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
        f_x0, f_x1 : float or np.ndarray, optional
            Source evaluated on the x=0 and x=Lx faces; scalar or length-Ny.
            Required data for the fourth-order closure — see the module
            docstring. Omitted, they are extrapolated from the interior.
        f_y0, f_y1 : float or np.ndarray, optional
            Source on the y=0 and y=Ly faces; scalar or length-Nx.
        _level : int
            Multigrid level index, set internally by ``coarsen``.

        Raises
        ------
        ValueError
            If ``f_values`` is not 2-D, or if either axis carries fewer than
            ``MIN_STRIP`` nodes.
        """
        f_values = np.asarray(f_values, dtype=float)
        if f_values.ndim != 2:
            raise ValueError(f"f_values must be 2-D, got shape {f_values.shape}")

        self.shape = (int(f_values.shape[0]), int(f_values.shape[1]))
        Nx, Ny = self.shape
        if Nx < self.MIN_STRIP or Ny < self.MIN_STRIP:
            raise ValueError(
                f"the fourth-order stencil spans two nodes either side of its "
                f"centre, so every axis needs at least {self.MIN_STRIP} interior "
                f"nodes; got {self.shape}.")

        self.Lx, self.Ly = float(Lx), float(Ly)
        self.dx = self.Lx / (Nx + 1)
        self.dy = self.Ly / (Ny + 1)
        self.spacings = (self.dx, self.dy)
        self.periodic = (False, False)
        self.level = _level

        self.f = f_values
        self.bc_x0 = np.broadcast_to(np.asarray(bc_x0, dtype=float), (Ny,)).copy()
        self.bc_x1 = np.broadcast_to(np.asarray(bc_x1, dtype=float), (Ny,)).copy()
        self.bc_y0 = np.broadcast_to(np.asarray(bc_y0, dtype=float), (Nx,)).copy()
        self.bc_y1 = np.broadcast_to(np.asarray(bc_y1, dtype=float), (Nx,)).copy()

        # 1/(12h²) per direction: the prefactor of the five-point stencil.
        self._cx = 1.0 / (12.0 * self.dx ** 2)
        self._cy = 1.0 / (12.0 * self.dy ** 2)

        self.f_x0 = self._resolve_face(f_x0, axis=0, upper=False, size=Ny)
        self.f_x1 = self._resolve_face(f_x1, axis=0, upper=True,  size=Ny)
        self.f_y0 = self._resolve_face(f_y0, axis=1, upper=False, size=Nx)
        self.f_y1 = self._resolve_face(f_y1, axis=1, upper=True,  size=Nx)

        # The closure needs ∂²u/∂n², not f: in 2-D the PDE gives the normal
        # second derivative only after the tangential one is subtracted, and
        # the tangential one is a derivative of the Dirichlet data alone. See
        # ``normal_second_derivative``.
        tang_x = ((0, self.dy, False),)     # the x-faces run along y
        tang_y = ((0, self.dx, False),)     # the y-faces run along x
        self.unn_x0 = normal_second_derivative(self.f_x0, self.bc_x0, tang_x)
        self.unn_x1 = normal_second_derivative(self.f_x1, self.bc_x1, tang_x)
        self.unn_y0 = normal_second_derivative(self.f_y0, self.bc_y0, tang_y)
        self.unn_y1 = normal_second_derivative(self.f_y1, self.bc_y1, tang_y)

        self._A_int, self._A_bnd = self._build_row_matrices()
        self._rhs = self._build_rhs()

    # -- Boundary source data --------------------------------------------------

    def _resolve_face(self, supplied, axis: int, upper: bool,
                      size: int) -> np.ndarray:
        """
        Resolves the source on one face, by supply or by extrapolation.

        Parameters
        ----------
        supplied : float, np.ndarray or None
            The caller's face values, if any.
        axis : int
            Axis normal to the face.
        upper : bool
            Which end of ``axis`` the face lies beyond.
        size : int
            Length of the face, i.e. the extent of the *other* axis.

        Returns
        -------
        np.ndarray
            Length-``size`` source values on the face.
        """
        if supplied is None:
            return np.asarray(extrapolate_face(self.f, axis, upper), dtype=float)
        return np.broadcast_to(np.asarray(supplied, dtype=float), (size,)).copy()

    # -- Operator --------------------------------------------------------------

    def _build_row_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Assembles the two distinct (Nx, Nx) pentadiagonal strip operators.

        Both carry the x-direction stencil and the diagonal shift −30·c_y from
        the transverse coupling. They differ only in the ghost fold: a strip
        adjacent to a transverse boundary gains +c_y on every diagonal entry,
        from the odd reflection of the y-ghost onto the strip's own value.

        Returns
        -------
        A_int : np.ndarray
            (Nx, Nx) operator for a strip with no transverse boundary adjacency.
        A_bnd : np.ndarray
            (Nx, Nx) operator for a strip adjacent to one transverse boundary.

        Notes
        -----
        A strip adjacent to *both* transverse boundaries — possible only when
        Ny = 4 does not hold, i.e. never on the grids used here, since MIN_STRIP
        is 4 — would need a third matrix. ``row_matrix_for`` constructs it on
        demand rather than pretending it cannot arise.
        """
        Nx = self.shape[0]
        cx, cy = self._cx, self._cy

        A = np.zeros((Nx, Nx))
        np.fill_diagonal(A, -30.0 * (cx + cy))
        np.fill_diagonal(A[1:, :], 16.0 * cx)
        np.fill_diagonal(A[:, 1:], 16.0 * cx)
        np.fill_diagonal(A[2:, :], -cx)
        np.fill_diagonal(A[:, 2:], -cx)

        # Ghost fold in the strip direction: −u₋₁ contributes +u₁, i.e. +cx on
        # the diagonal, at each end of every strip. This is the correction that
        # must NOT be written into A[0,1]: an even reflection there imposes a
        # Neumann condition on Dirichlet data.
        A[0, 0] += cx
        A[-1, -1] += cx

        A_int = A
        A_bnd = A + cy * np.eye(Nx)
        return A_int, A_bnd

    def _build_rhs(self) -> np.ndarray:
        """
        Absorbs the Dirichlet data and the face sources into the source field.

        Per direction, with c = 1/(12h²), boundary value g and ∂²u/∂n² the
        normal second derivative on that face:

            first interior row:   −14·c·g   and   + (∂²u/∂n²)/12
            second interior row:  + c·g

        The −14 is the sum of −16 from the known boundary node and +2 from the
        ghost; they subtract. The second term is the second-derivative part of
        the reflection, without which the closure is only O(h²) accurate and
        caps the whole scheme at second order — and it is ∂²u/∂n², *not* f. The
        two coincide in 1-D only; see ``normal_second_derivative``.

        Returns
        -------
        np.ndarray
            (Nx, Ny) right-hand side.
        """
        cx, cy = self._cx, self._cy
        r = self.f.copy()

        # x faces: the strip direction.
        r[0, :] += -14.0 * cx * self.bc_x0 + self.unn_x0 / 12.0
        r[1, :] += cx * self.bc_x0
        r[-1, :] += -14.0 * cx * self.bc_x1 + self.unn_x1 / 12.0
        r[-2, :] += cx * self.bc_x1

        # y faces: the transverse direction.
        r[:, 0] += -14.0 * cy * self.bc_y0 + self.unn_y0 / 12.0
        r[:, 1] += cy * self.bc_y0
        r[:, -1] += -14.0 * cy * self.bc_y1 + self.unn_y1 / 12.0
        r[:, -2] += cy * self.bc_y1

        return r

    def row_matrix(self) -> np.ndarray:
        """
        The Nx × Nx strip operator for an interior strip.

        Satisfies ``LineProblem2D``. Every scheme in ``solvers/outer`` reaches
        the strips through ``row_matrix_for`` instead, which is what
        distinguishes the boundary-adjacent strips; this accessor is the one the
        protocol requires and the one ``kappa_row`` reports on.
        """
        return self._A_int

    def row_matrix_for(self, idx: tuple[int, ...]) -> np.ndarray:
        """
        The strip operator for the strip at transverse index ``idx``.

        Returns the *same array object* for every strip sharing an operator, so
        that a caller may cache a block encoding or a set of QSP phase angles
        keyed on identity. There are two such objects in 2-D whatever N is.

        Parameters
        ----------
        idx : tuple of int
            Transverse index ``(j,)``.

        Returns
        -------
        np.ndarray
            (Nx, Nx) strip operator: the interior matrix, or the
            boundary-adjacent one when j is 0 or Ny−1.
        """
        j = idx[0]
        Ny = self.shape[1]
        adjacent = (j == 0) + (j == Ny - 1)
        if adjacent == 0:
            return self._A_int
        if adjacent == 1:
            return self._A_bnd
        # Ny = 1, unreachable while MIN_STRIP is 4. Constructed rather than
        # asserted away, so the method stays total.
        return self._A_int + 2.0 * self._cy * np.eye(self.shape[0])

    def transverse_terms(self, axis: int, index: int,
                         n: int) -> tuple[tuple[int, float], ...]:
        """
        The transverse neighbours to gather into a strip's right-hand side.

        The four coefficients of the fourth-order stencil, in ascending offset
        order. ``strip_sweep`` discards the offsets that fall outside the
        domain: their contribution is Dirichlet data and is already in
        ``rhs()``, and the ghost node beyond them is not a neighbour at all but
        a multiple of the strip's own value, carried by ``row_matrix_for``.

        Parameters
        ----------
        axis : int
            Index into ``shape``. Only axis 1 is transverse in 2-D.
        index : int
            Position of the strip along ``axis``. Unused: the coefficients are
            uniform, the boundary rows being handled by the operator and the
            right-hand side rather than by omitting a term here.
        n : int
            Extent of ``axis``. Unused, for the same reason.

        Returns
        -------
        tuple of (int, float)
            ((−2, −c), (−1, 16c), (1, 16c), (2, −c)) with c = 1/(12·h_axis²).
        """
        c = 1.0 / (12.0 * self.spacings[axis] ** 2)
        return ((-2, -c), (-1, 16.0 * c), (1, 16.0 * c), (2, -c))

    def apply(self, u: np.ndarray) -> np.ndarray:
        """
        The full fourth-order Laplacian with homogeneous exterior.

        Used for residual evaluation (r = rhs() − apply(u)) and on coarse
        multigrid levels, which always carry homogeneous boundary data — see
        ``coarsen`` below.

        Parameters
        ----------
        u : np.ndarray
            (Nx, Ny) field at the interior nodes.

        Returns
        -------
        np.ndarray
            (Nx, Ny) result of applying the discrete Laplacian to u.
        """
        return (apply_axis_4th(u, 0, self.dx)
                + apply_axis_4th(u, 1, self.dy))

    # -- Coarsening ------------------------------------------------------------

    def coarsen(self) -> Optional["PoissonLine2D4th"]:
        """
        Halve each direction whose spacing is within COARSEN_RATIO of the
        finer one (anisotropic semi-coarsening).

        Identical in policy to ``PoissonLine2D.coarsen``: coarsening is a
        property of the grid, not of the stencil, so the fourth-order hierarchy
        descends exactly as the second-order one does. Both dimensions stay
        powers of two, the strip operator stays pentadiagonal of power-of-two
        size at every level, and halving both directions together keeps dx/dy
        fixed and hence κ(A_row) bounded on every level.

        Coarse levels carry the error equation: zero source, zero boundaries,
        and therefore zero face sources as well — the h²·f(0) term of the
        closure vanishes with the source, which is why no face data need be
        propagated here.

        Returns
        -------
        PoissonLine2D4th or None
            The next coarser level, or None once neither direction can be
            halved.
        """
        Nx, Ny = self.shape
        h_min = min(self.dx, self.dy)
        do = [h <= self.COARSEN_RATIO * h_min for h in (self.dx, self.dy)]
        do = [d and n > self.MIN_STRIP and n % 2 == 0
              for d, n in zip(do, (Nx, Ny))]
        if not any(do):
            return None
        return PoissonLine2D4th(
            np.zeros((Nx // 2 if do[0] else Nx, Ny // 2 if do[1] else Ny)),
            Lx=self.Lx, Ly=self.Ly,
            _level=self.level + 1,
        )

    # -- Utilities -------------------------------------------------------------

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the (Nx, Ny) interior coordinate matrices in 'ij' index order.
        """
        Nx, Ny = self.shape
        x = np.arange(1, Nx + 1) * self.dx
        y = np.arange(1, Ny + 1) * self.dy
        return np.meshgrid(x, y, indexing="ij")

    def face_coords(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the coordinates along each face, for evaluating the face sources.

        Returns
        -------
        x : np.ndarray
            Length-Nx interior x coordinates, at which the y-faces are sampled.
        y : np.ndarray
            Length-Ny interior y coordinates, at which the x-faces are sampled.
        """
        Nx, Ny = self.shape
        return (np.arange(1, Nx + 1) * self.dx,
                np.arange(1, Ny + 1) * self.dy)

    def kappa_row(self) -> float:
        """
        Spectral condition number κ(A_row) of the interior strip operator.

        Bounded as N → ∞, as at second order: the transverse coupling supplies
        a diagonal shift that does not vanish with h. This is the property that
        makes the line-decomposed formulation tractable for the quantum inner
        solvers, and it survives the move to fourth order.
        """
        e = np.abs(np.linalg.eigvalsh(self._A_int))
        return float(e.max() / e.min())

    def kappa_rows(self) -> dict[str, float]:
        """
        κ of every distinct strip operator, for cache and cost accounting.

        Returns
        -------
        dict
            {'interior': κ(A_int), 'boundary': κ(A_bnd)}. Each distinct matrix
            needs its own block encoding and its own set of QSP phase angles,
            so both keys must be present in the phase cache before a QSVT sweep
            is submitted.
        """
        out = {}
        for name, A in (("interior", self._A_int), ("boundary", self._A_bnd)):
            e = np.abs(np.linalg.eigvalsh(A))
            out[name] = float(e.max() / e.min())
        return out

    def residual(self, u: np.ndarray) -> float:
        """Relative 2-norm residual of the full coupled system."""
        b = self.rhs()
        bn = np.linalg.norm(b)
        r = np.linalg.norm(b - self.apply(u))
        return float(r / bn) if bn > 1e-300 else float(r)

    def rhs(self) -> np.ndarray:
        """(Nx, Ny) right-hand side with Dirichlet data already absorbed."""
        return self._rhs

    def __repr__(self) -> str:
        Nx, Ny = self.shape
        return (f"PoissonLine2D4th({Nx}x{Ny}, dx={self.dx:.3e}, "
                f"dy={self.dy:.3e}, kappa={self.kappa_row():.3f}, "
                f"level={self.level})")
