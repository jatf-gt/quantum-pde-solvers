"""
High-fidelity reference solutions for the two-dimensional benchmark cases.

Every 2D benchmark metric in `benchmark/metrics.py` is measured against an
externally supplied reference field; this module is the sole producer of that
field. Two regimes are covered:

Analytical reference
    Where a closed-form solution exists (manufactured solutions such as
    φ(x,y) = sin(πx)·sin(πy)), it is evaluated directly at the coarse interior
    nodes. This is exact and carries no discretisation error whatsoever.

Fine-mesh numerical reference
    Where no closed form exists, the discrete solution is computed on a mesh
    refined by an integer factor and then sampled back onto the coarse nodes.
    This is the protocol of Ghafourpour & Laizet (2025), Section IV E, in which
    the coarse-mesh benchmark result is judged against a refined classical
    solve rather than against the coarse discrete solution — the latter would
    measure only the iterative solver's algebraic error, concealing the
    truncation error of the discretisation itself.

Discretisation error scales as O(h²) = O((N+1)⁻²), so refining by a factor r
suppresses the reference's own truncation error by r² relative to the coarse
field under evaluation. The default r = 19 yields a suppression factor of 361,
placing the reference roughly two and a half decades below the coarse solution
it certifies.

The fine solve is performed by `solvers.outer.solve` with the direct Thomas
strip solver under a full-multigrid (FMG) outer scheme. FMG is essential here
rather than merely convenient: the refined meshes reach N_fine ~ 300 in the
default configuration, where the stationary line-Jacobi iteration formerly used
requires O(N_fine²) sweeps — thousands of sweeps, several minutes — whereas FMG
attains discretisation accuracy in a grid-independent handful of cycles.

References
----------
Ghafourpour & Laizet, Phys. Rev. Applied 24, 024032 (2025), Section IV E.
Briggs, Henson & McCormick, *A Multigrid Tutorial*, 2nd ed. (SIAM, 2000).
"""
from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np

from problems.poisson_line_2d import PoissonLine2D
from solvers.outer import solve


# ── Default Refinement Parameters ─────────────────────────────────────────────

# Integer mesh refinement multiplier, N_fine = N · REFINE_FACTOR. The value 19
# reproduces the reference literature's h = 1/153 fine mesh at the N = 8
# benchmark resolution: (8·19 + 1) = 153.
REFINE_FACTOR = 19

# Algebraic tolerance imposed on the fine solve, expressed as the relative
# Euclidean residual ‖b − A·u‖₂ / ‖b‖₂. Set two decades below the O(h_fine²)
# truncation error so the reference is limited by discretisation, not by
# incomplete convergence of the outer iteration.
REFERENCE_TOL = 1e-10

# V-cycle ceiling for the fine solve. FMG reaches the tolerance above in
# typically fewer than ten cycles at every resolution considered; the ceiling
# exists solely to bound pathological cases.
REFERENCE_MAX_CYCLES = 200


# ── Reference Solution Construction ───────────────────────────────────────────

def fine_mesh_reference(
    f_fn:          Callable[[np.ndarray, np.ndarray], np.ndarray],
    N:             int,
    Lx:            float = 1.0,
    Ly:            float = 1.0,
    bc_x0:         "float | Callable" = 0.0,
    bc_x1:         "float | Callable" = 0.0,
    bc_y0:         "float | Callable" = 0.0,
    bc_y1:         "float | Callable" = 0.0,
    refine_factor: int   = REFINE_FACTOR,
    target_h:      Optional[float] = None,
    analytical_fn: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    tol:           float = REFERENCE_TOL,
    max_cycles:    int   = REFERENCE_MAX_CYCLES,
) -> np.ndarray:
    """
    Computes the reference solution field sampled at the coarse interior nodes.

    Execution follows a strict hierarchical priority. Only one mode is ever
    active for a given call.

    Mode 1 — Analytical solution (highest priority)
        Triggered when `analytical_fn` is supplied. The closed-form solution is
        evaluated directly at the coarse nodes and returned; no fine solve is
        performed. Exact to machine precision.

    Mode 2 — Targeted mesh spacing
        Triggered when `target_h` is supplied and `analytical_fn` is not. The
        fine resolution is set to N_fine = round(Lx/target_h) − 1, discarding
        the power-of-two constraint (which binds only the quantum solvers, not
        this purely classical reference). Provided to reproduce literature-
        specific meshes such as h = 1/153.

    Mode 3 — Integer factorial refinement (default)
        N_fine = N · refine_factor. Integer scaling is deliberate: it maximises
        the likelihood that coarse nodes coincide exactly with fine nodes, in
        which case sampling is exact and no interpolation error is introduced.

    Nodal alignment
    ---------------
    Both meshes are vertex-centred with the boundary excluded, so the coarse
    node i sits at x_i = i·Lx/(N+1) and the fine node k at x_k = k·Lx/(N_fine+1).
    Exact coincidence therefore requires the stride

        s = (N_fine + 1) / (N + 1)

    to be an integer, in which case coarse node i is fine node s·i and sampling
    is a pure array slice. When s is not an integer the coarse nodes fall
    between fine nodes and bilinear interpolation is used instead, contributing
    an O(h_fine²) error that remains negligible against the coarse field.

    Parameters
    ----------
    f_fn : Callable
        Source term f(X, Y) of ∇²u = f, accepting two (Nx, Ny) coordinate
        matrices and returning an array of the same shape. Evaluated on the
        *fine* mesh in the physical (unscaled) convention of `PoissonLine2D` —
        that is, without the h² factor of the scaled formulation.
    N : int
        Number of coarse interior nodes per direction; the returned field is
        (N, N).
    Lx, Ly : float
        Physical domain extents [m], or unity for the non-dimensional unit
        square. Default 1.0.
    bc_x0, bc_x1, bc_y0, bc_y1 : float or Callable
        Dirichlet boundary data on the edges x=0, x=Lx, y=0, y=Ly respectively.
        A scalar is broadcast along the whole edge. A callable is evaluated on
        the *fine* mesh's edge coordinates and must accept a length-N_fine
        coordinate vector — the transverse coordinate y for bc_x0/bc_x1 and x
        for bc_y0/bc_y1 — returning an array of the same length. A callable is
        mandatory for spatially varying boundary data: a pre-evaluated coarse
        array cannot be reused, because the fine mesh has different, and far
        more numerous, boundary nodes.
    refine_factor : int
        Integer mesh multiplier for Mode 3. Default REFINE_FACTOR = 19.
    target_h : float, optional
        Absolute target fine spacing for Mode 2 [same units as Lx].
    analytical_fn : Callable, optional
        Closed-form solution u(X, Y) for Mode 1, evaluated at the coarse nodes.
    tol : float
        Relative Euclidean residual tolerance for the fine solve.
    max_cycles : int
        FMG V-cycle ceiling for the fine solve.

    Returns
    -------
    u_coarse : np.ndarray
        (N, N) reference solution field evaluated at the coarse interior nodes,
        indexed [i, j] with i along x and j along y ('ij' meshgrid convention).

    Raises
    ------
    ValueError
        If the requested fine resolution is coarser than N, which would make
        the "reference" less accurate than the field it is meant to certify.

    Warns
    -----
    RuntimeWarning
        If the fine solve fails to reach `tol` within `max_cycles`. The
        partially converged field is still returned, but every metric derived
        from it is then limited by the reference's own algebraic error.
    """
    # ── Mode 1: Analytical Solution ───────────────────────────────────────────
    if analytical_fn is not None:
        X, Y = _interior_grid(N, N, Lx, Ly)
        return np.asarray(analytical_fn(X, Y), dtype=float)

    # ── Mode 2 / 3: Fine Resolution Selection ─────────────────────────────────
    if target_h is not None:
        N_fine = max(N, int(round(Lx / target_h)) - 1)
    else:
        N_fine = N * int(refine_factor)

    if N_fine < N:
        raise ValueError(
            f"Computed fine resolution N_fine={N_fine} is inferior to the base "
            f"resolution N={N}. Augment refine_factor or diminish target_h."
        )

    # ── Fine-Mesh Solve ───────────────────────────────────────────────────────
    X_fine, Y_fine = _interior_grid(N_fine, N_fine, Lx, Ly)
    x_fine, y_fine = X_fine[:, 0], Y_fine[0, :]

    problem_fine = PoissonLine2D(
        np.asarray(f_fn(X_fine, Y_fine), dtype=float),
        Lx=Lx, Ly=Ly,
        bc_x0=_edge_data(bc_x0, y_fine),
        bc_x1=_edge_data(bc_x1, y_fine),
        bc_y0=_edge_data(bc_y0, x_fine),
        bc_y1=_edge_data(bc_y1, x_fine),
    )

    result = solve(
        problem_fine,
        inner="thomas",
        scheme=_reference_scheme(problem_fine),
        tol=tol,
        **_reference_scheme_options(problem_fine, max_cycles),
    )

    if not result.converged:
        warnings.warn(
            f"Fine-mesh reference solve terminated without meeting the "
            f"tolerance (N_fine={N_fine}, stop_reason='{result.stop_reason}', "
            f"terminal relative residual = {result.residual:.3e}, "
            f"requested tol = {tol:.1e}). Reference accuracy is limited by "
            f"algebraic error; raise max_cycles or relax tol.",
            RuntimeWarning,
        )

    # ── Coarse Node Extraction ────────────────────────────────────────────────
    stride = (N_fine + 1) / (N + 1)

    if abs(stride - round(stride)) < 1e-9:
        s = int(round(stride))
        coarse_indices = np.array([s * i - 1 for i in range(1, N + 1)])
        return result.u[np.ix_(coarse_indices, coarse_indices)]

    return _bilinear_sample(result.u, N, N_fine)


# ── Private Utility Methods ───────────────────────────────────────────────────

def _edge_data(bc, coords: np.ndarray):
    """
    Resolves one edge's Dirichlet data onto the fine mesh's boundary nodes.

    Scalars pass through unaltered, to be broadcast by `PoissonLine2D`. Callables
    are evaluated at the supplied coordinates, which are those of the fine mesh
    — the whole point of the indirection, since the caller cannot know N_fine in
    advance and a coarse array would fail to broadcast.

    Parameters
    ----------
    bc : float or Callable
        Scalar boundary value, or a function of the edge coordinate.
    coords : np.ndarray
        Length-N_fine coordinate vector along the edge in question.

    Returns
    -------
    float or np.ndarray
        The scalar unchanged, or the length-N_fine evaluated edge values.

    Raises
    ------
    ValueError
        If a callable returns an array whose length does not match `coords`.
    """
    if not callable(bc):
        return bc
    values = np.asarray(bc(coords), dtype=float)
    if values.shape != coords.shape:
        raise ValueError(
            f"Boundary callable returned shape {values.shape}; expected "
            f"{coords.shape} to match the fine mesh edge."
        )
    return values


def _interior_grid(
    Nx: int,
    Ny: int,
    Lx: float,
    Ly: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constructs the vertex-centred interior coordinate matrices of a mesh.

    Boundary nodes are excluded; their contribution is absorbed into the
    right-hand side by `PoissonLine2D`. Spacings are dx = Lx/(Nx+1) and
    dy = Ly/(Ny+1), matching `PoissonLine2D.grid()` exactly.

    Parameters
    ----------
    Nx, Ny : int
        Interior node counts along x and y.
    Lx, Ly : float
        Domain extents.

    Returns
    -------
    X, Y : np.ndarray
        (Nx, Ny) coordinate matrices in 'ij' indexing order.
    """
    x = np.arange(1, Nx + 1) * (Lx / (Nx + 1))
    y = np.arange(1, Ny + 1) * (Ly / (Ny + 1))
    return np.meshgrid(x, y, indexing="ij")


def _reference_scheme(problem: PoissonLine2D) -> str:
    """
    Selects the outer scheme for the fine solve.

    FMG is used wherever a multigrid hierarchy exists. `solvers.outer.solve`
    deliberately raises rather than silently degrading when a problem admits no
    coarse level, so the (rare) uncoarsenable case is detected here and routed
    to line SOR instead. This occurs only for meshes at or below the minimum
    strip length, where the O(N²) sweep count of a stationary scheme is
    immaterial.
    """
    return "fmg" if problem.coarsen() is not None else "sor"


def _reference_scheme_options(
    problem:    PoissonLine2D,
    max_cycles: int,
) -> dict:
    """
    Assembles the scheme-specific keyword arguments for the fine solve.

    Multigrid and stationary schemes expose different iteration ceilings
    (`max_cycles` versus `max_iter`), so the correct one is selected alongside
    the scheme itself. `patience` is set beyond the iteration ceiling in both
    cases: stagnation detection exists to bound *quantum* runs against an inner
    solver's error floor, and the direct Thomas solve used here has no such
    floor, so an early stagnation stop could only truncate a healthy — if
    slow — convergence.
    """
    if _reference_scheme(problem) == "fmg":
        return {"max_cycles": max_cycles, "patience": max_cycles + 1}
    max_iter = max(20000, max_cycles)
    return {"max_iter": max_iter, "patience": max_iter + 1}


def _bilinear_sample(
    u_fine:   np.ndarray,
    N_coarse: int,
    N_fine:   int,
) -> np.ndarray:
    """
    Interpolates a fine-mesh field bilinearly onto the coarse interior nodes.

    Invoked exclusively when the fine mesh spacing does not partition the
    coarse spacing uniformly, so that coarse coordinates fall strictly between
    fine nodal intersections. Both meshes are assumed square and vertex-centred
    with homogeneous exterior padding implied by the clamping at the edges.

    The interpolation is performed in normalised index space, which makes it
    independent of the physical domain extent: a coarse node at index i has
    fine-index coordinate (i+1)·(N_fine+1)/(N_coarse+1) − 1.

    Parameters
    ----------
    u_fine : np.ndarray
        (N_fine, N_fine) solution field on the refined mesh.
    N_coarse : int
        Interior node count of the target coarse mesh.
    N_fine : int
        Interior node count of the source fine mesh.

    Returns
    -------
    u_coarse : np.ndarray
        (N_coarse, N_coarse) field sampled at the coarse interior nodes.

    Notes
    -----
    Complexity is O(N_coarse²) evaluations, each O(1); memory is O(N_fine²) for
    the input field alone.
    """
    scale = (N_fine + 1) / (N_coarse + 1)

    # Fine-index coordinates of every coarse node, along one axis.
    fi = np.arange(1, N_coarse + 1) * scale - 1.0

    i0 = np.clip(np.floor(fi).astype(int), 0, N_fine - 1)
    i1 = np.clip(i0 + 1, 0, N_fine - 1)
    t  = fi - np.floor(fi)

    # Separable bilinear weighting: interpolate along x, then along y.
    w0 = (1.0 - t)[:, None]
    w1 = t[:, None]

    along_x = w0 * u_fine[i0, :] + w1 * u_fine[i1, :]
    return (w0.T * along_x[:, i0] + w1.T * along_x[:, i1])
