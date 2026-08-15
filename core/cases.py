"""
Canonical registry of benchmark cases.

A *case* is a fully specified boundary value problem: domain, source term,
boundary data, discretisation convention, and the means by which its ground
truth is obtained. It is deliberately not a solver configuration — precision
parameters, outer tolerances and polynomial degree caps belong to whoever runs
the case, not to the case itself, since the same problem is legitimately solved
to different tolerances by different studies.

Purpose
-------
Before this module existed, each case was defined inside the script that
happened to run it. Around twenty-five had exactly one definition site, so
deleting a script deleted physics; several others were duplicated three to five
times, and in two instances the *same name* denoted different mathematics (see
`Name collisions` below). This registry is the single definition site for all of
them.

Design
------
The interface mirrors `solvers/outer/inner.py`, which registers inner solvers by
name with introspectable options. The same idiom applies here — `register`,
`get`, `available`, `describe` — so a reader who understands one registry
understands the other.

`Case` is a wider type than `SimConfig1D`/`SimConfig2D`, which remain the solver
configuration objects. Those dataclasses cannot express most of this inventory:
they carry scalar Dirichlet data only, on the unit domain, with no periodicity,
no Neumann condition, no attached manufactured solution and no statement of how
a reference solution is obtained. All of those are required here.

Every case builds to a `BuiltCase`, whose populated fields depend on dimension:
1D cases carry the assembled `(A, b)` system directly, whilst 2D and 3D cases
carry a `LineProblem2D`/`LineProblem3D` for the outer iteration to decompose
into strips.

Grid conventions
----------------
Two conventions coexist and the distinction is load-bearing:

    "interior"        h = 1/(N+1), nodes at x = h, 2h, …, Nh. The boundaries are
                      not unknowns. Used by every pure-Dirichlet case.
    "including-origin"  h = 1/N, nodes at x = 0, h, …, (N−1)h. Used only by the
                      Neumann–Dirichlet case, where φ(0) is an unknown. Applying
                      a Neumann row to index 0 of the interior grid would impose
                      the condition at x = h rather than x = 0.

Reference strategies
--------------------
How ground truth is obtained, recorded per case because it has drifted:
the fine-mesh refinement factor alone appeared as 19, 17 and 9 at different
sites for nominally the same measurement.

    "analytical"  Closed-form solution; `exact` is populated.
    "manufactured"  Solution chosen first, source derived from it (MMS).
    "thomas"      No closed form; the classical direct solve is the reference.
    "fine_mesh"   Fine-mesh solve sampled back onto the coarse nodes;
                  `ref_params["refine"]` gives the per-direction factor.
    "fourier"     Truncated Fourier series on a fine grid;
                  `ref_params` gives `modes` and `n_fine`.
    "quadrature"  Numerical integration of the source.

Name collisions
---------------
Two labels denoted genuinely different mathematics before consolidation. Both
readings are preserved under unambiguous names, with their exact numerics
unchanged, so that results produced under either remain reproducible:

    poisson_2d_sin_pi        f = sin(πx)sin(πy), closed form known
    poisson_2d_fS_10sin2pi   f = 10 sin(2πx)cos(2πy), no closed form
                             — the legacy `SOURCE_FUNCTIONS_2D["fS"]`, used by
                             the 2D sweeps E and F

    het_1d_3a_linear         f = 2x − 1, exact x³/3 − x²/2 + x/6
    het_1d_linear_scaled     f = −α ρ₀ x with α ≈ 5.65×10⁴, exact
                             α ρ₀ x(1 − x²)/6 — the `HETConfig` reading

Selecting by the old short label is therefore no longer possible, which is the
intent: the ambiguity cannot be hit by accident.

References
----------
Ghafourpour & Laizet, Phys. Rev. Applied 24, 024032 (2025) — the generic
    Poisson sweeps this registry reproduces.
Boeuf & Garrigues, J. Appl. Phys. 84(7), 3541-3554 (1998) — the HET plasma
    model and its parameter table.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import numpy as np

from core import het_geometry as geom


# -- Built Case ----------------------------------------------------------------

@dataclass
class BuiltCase:
    """
    A case instantiated at a particular resolution.

    Which fields are populated depends on the dimension of the originating
    case, because 1D problems are solved as a matrix system directly whilst 2D
    and 3D problems are decomposed into strips by ``solvers.outer``.

    Attributes
    ----------
    coords : tuple of np.ndarray
        Node coordinates. Length-1 tuple ``(x,)`` in 1D with `x` of length N;
        ``(X, Y)`` in 2D and ``(X, Y, Z)`` in 3D, each of the full field shape
        and built with ``indexing="ij"``.
    spacings : tuple of float
        Mesh spacing per axis, in the physical units of the case domain [m] for
        HET cases and dimensionless for the generic Poisson cases.
    f_values : np.ndarray
        Source term sampled at the nodes, of the field shape. Note this is the
        raw source f, *not* the boundary-absorbed right-hand side.
    exact : np.ndarray or None
        Reference solution at the nodes, of the field shape, when the case
        declares an analytical, manufactured or quadrature reference. None when
        the reference must be computed by a solve.
    A : np.ndarray or None
        (N, N) system matrix. Populated for 1D cases only.
    b : np.ndarray or None
        Length-N right-hand side with boundary data absorbed. 1D only.
    problem : LineProblem2D or LineProblem3D or None
        Line-decomposed problem object. Populated for 2D and 3D cases only.
    kappa : float or None
        Condition number of the operator actually solved: κ(A) in 1D, κ of the
        strip operator in 2D and 3D.
    f_boundary : tuple of float or None
        Source term evaluated *on* the two boundaries, (f(0), f(1)). 1D only,
        and populated only where the analytical source is known.

        Unused by the second-order discretisation, whose boundary rows need
        only the Dirichlet data. The fourth-order ghost-node closure of
        ``problems.poisson_1d_4th`` additionally needs the governing equation
        on the boundary, u″ = f, and so requires these two values; without them
        it must extrapolate from the interior samples, which is asymptotically
        adequate but inaccurate on a sharply peaked source at the coarse
        resolutions the fourth-order sweep uses.
    f_faces : tuple or None
        The 2D/3D counterpart of `f_boundary`: the source evaluated *on* the
        domain faces, as ``(lo, hi)`` with one entry per axis. In 2D
        ``lo = (f_x0, f_y0)`` and ``hi = (f_x1, f_y1)``, each a length-N array
        over the interior nodes of the other axis; in 3D each entry is a 2D
        array over the face. Entries are None where the case cannot supply
        them.

        Required by the fourth-order closures of ``problems.poisson_line_2d_4th``
        and ``problems.poisson_line_3d_4th`` — but note that what those classes
        need is ∂²u/∂n² on the face, and in more than one dimension that is
        *not* the source: ∇²u = f gives ∂²u/∂n² = f − Σ_t ∂²u/∂t², the
        tangential terms being second derivatives of the Dirichlet data. The
        classes perform that subtraction themselves, so what belongs here is
        the plain source, exactly as in 1D.
    """

    coords:   tuple[np.ndarray, ...]
    spacings: tuple[float, ...]
    f_values: np.ndarray
    exact:    Optional[np.ndarray] = None
    A:        Optional[np.ndarray] = None
    b:        Optional[np.ndarray] = None
    problem:  Any = None
    kappa:    Optional[float] = None
    f_boundary: Optional[tuple[float, float]] = None
    f_faces:  Optional[tuple] = None


# -- Case Declaration ----------------------------------------------------------

@dataclass(frozen=True)
class Case:
    """
    One benchmark problem, completely specified.

    Attributes
    ----------
    name : str
        Unique registry key, lower_snake_case, prefixed by family and dimension
        (e.g. ``"het_1d_3c_neumann"``).
    dim : int
        Spatial dimension, 1, 2 or 3.
    family : str
        ``"poisson"`` for the generic benchmarks, ``"het"`` for the Hall Effect
        Thruster plasma application.
    summary : str
        One-line description, shown by ``describe``.
    build : callable
        ``(N: int) -> BuiltCase``. Assembles the case at resolution N.
    lengths : tuple of float
        Physical domain extents per axis. Unity for the non-dimensional
        benchmarks; metres [m] for the dimensional HET cases.
    periodic : tuple of bool
        Per-axis periodicity. All-False except the azimuthal axis of the 3D HET
        slab, where periodicity is what makes the domain a thruster channel
        rather than a box.
    grid : str
        ``"interior"`` or ``"including-origin"``; see the module docstring.
    reference : str
        How ground truth is obtained; see the module docstring.
    ref_params : dict
        Parameters of the reference strategy, e.g. ``{"refine": 19}`` or
        ``{"modes": 50, "n_fine": 200}``. Empty when the strategy takes none.
    default_N : tuple of int
        Resolutions at which the case is conventionally swept. Every entry must
        be a power of two, as quantum amplitude encoding requires log₂(N)
        qubits.
    notes : str
        Provenance: literature reference, the site the case was extracted from,
        legacy labels, physical units, and any recorded defect or caveat.
    """

    name:       str
    dim:        int
    family:     str
    summary:    str
    build:      Callable[[int], BuiltCase]
    lengths:    tuple[float, ...] = (1.0,)
    periodic:   tuple[bool, ...] = (False,)
    grid:       str = "interior"
    reference:  str = "analytical"
    ref_params: dict = field(default_factory=dict)
    default_N:  tuple[int, ...] = (4, 8, 16, 32)
    notes:      str = ""


_REGISTRY: dict[str, Case] = {}

_VALID_REFERENCES = frozenset({
    "analytical", "manufactured", "thomas", "fine_mesh", "fourier", "quadrature",
})


def register(case: Case) -> Case:
    """
    Adds a case to the registry.

    Parameters
    ----------
    case : Case
        The case to register.

    Returns
    -------
    Case
        The same case, so that registration may be used inline.

    Raises
    ------
    ValueError
        If the name is already registered, if the dimension is not 1, 2 or 3,
        if the reference strategy is unrecognised, if the grid convention is
        unrecognised, or if any resolution in ``default_N`` is not a power of
        two.
    """
    if case.name in _REGISTRY:
        raise ValueError(
            f"Case '{case.name}' is already registered. Names must be unique; "
            f"if two definitions genuinely differ, give each its own name "
            f"rather than overwriting."
        )
    if case.dim not in (1, 2, 3):
        raise ValueError(f"Case '{case.name}': dim must be 1, 2 or 3, got {case.dim}.")
    if case.family not in ("poisson", "het"):
        raise ValueError(
            f"Case '{case.name}': family must be 'poisson' or 'het', "
            f"got '{case.family}'."
        )
    if case.reference not in _VALID_REFERENCES:
        raise ValueError(
            f"Case '{case.name}': unrecognised reference strategy "
            f"'{case.reference}'. Valid: {sorted(_VALID_REFERENCES)}."
        )
    if case.grid not in ("interior", "including-origin"):
        raise ValueError(
            f"Case '{case.name}': unrecognised grid convention '{case.grid}'."
        )
    for N in case.default_N:
        if N <= 0 or (N & (N - 1)) != 0:
            raise ValueError(
                f"Case '{case.name}': default_N entry {N} is not a positive "
                f"power of two, which quantum amplitude encoding requires."
            )
    _REGISTRY[case.name] = case
    return case


def get(name: str) -> Case:
    """
    Retrieves a registered case by name.

    Parameters
    ----------
    name : str
        Registry key.

    Returns
    -------
    Case
        The registered case.

    Raises
    ------
    KeyError
        If the name is not registered. The message lists the valid names, as
        an unrecognised case is far more often a typo or a stale legacy label
        than a genuinely missing definition.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown case '{name}'. Registered cases: {sorted(_REGISTRY)}."
        )
    return _REGISTRY[name]


def available(dim: int | None = None, family: str | None = None) -> list[str]:
    """
    Lists registered case names, optionally filtered.

    Parameters
    ----------
    dim : int, optional
        Restrict to this spatial dimension.
    family : str, optional
        Restrict to ``"poisson"`` or ``"het"``.

    Returns
    -------
    list of str
        Matching case names, sorted.
    """
    names = []
    for name, case in _REGISTRY.items():
        if dim is not None and case.dim != dim:
            continue
        if family is not None and case.family != family:
            continue
        names.append(name)
    return sorted(names)


def describe(name: str | None = None) -> str:
    """
    Renders a human-readable description of one case or of the whole registry.

    Parameters
    ----------
    name : str, optional
        Case to describe. When omitted, tabulates every registered case.

    Returns
    -------
    str
        Formatted description, ready to print.
    """
    if name is not None:
        c = get(name)
        lines = [
            f"{c.name}",
            f"  {c.summary}",
            "",
            f"  dimension   : {c.dim}D",
            f"  family      : {c.family}",
            f"  domain      : {' x '.join(f'{L:g}' for L in c.lengths)}",
            f"  periodic    : {c.periodic}",
            f"  grid        : {c.grid}",
            f"  reference   : {c.reference}"
            + (f"  {c.ref_params}" if c.ref_params else ""),
            f"  default N   : {c.default_N}",
        ]
        if c.notes:
            lines += ["", "  Notes"]
            lines += [f"    {ln}" for ln in c.notes.strip().split("\n")]
        return "\n".join(lines)

    header = f"  {'case':<34}  {'dim':>3}  {'family':<8}  {'reference':<12}  summary"
    rows = [header, "  " + "-" * 110]
    for n in available():
        c = _REGISTRY[n]
        rows.append(
            f"  {c.name:<34}  {c.dim:>2}D  {c.family:<8}  "
            f"{c.reference:<12}  {c.summary}"
        )
    return "\n".join(rows)


# -- 1D Assembly Helpers -------------------------------------------------------

def _grid_1d(N: int) -> tuple[np.ndarray, float]:
    """
    Interior grid for pure-Dirichlet 1D problems.

    N unknowns at x = h, 2h, …, Nh with h = 1/(N+1); the boundaries x = 0 and
    x = 1 are not unknowns.

    Parameters
    ----------
    N : int
        Number of interior nodes.

    Returns
    -------
    x : np.ndarray
        Length-N vector of node coordinates.
    h : float
        Mesh spacing.
    """
    h = 1.0 / (N + 1)
    return np.arange(1, N + 1) * h, h


def _f_boundary_1d(
    source: Callable[[np.ndarray], np.ndarray]
) -> tuple[float, float]:
    """
    Evaluates a 1D source term on the two boundaries, (f(0), f(1)).

    Required by the fourth-order ghost-node closure of
    ``problems.poisson_1d_4th``, which eliminates the ghost node using the
    governing equation u″ = f evaluated on the boundary. The second-order
    discretisation does not use these values.

    Parameters
    ----------
    source : callable
        f(x), the same callable the interior samples are drawn from. Must
        accept and return a NumPy array.

    Returns
    -------
    tuple of float
        (f(0), f(1)).
    """
    edges = np.asarray(source(np.array([0.0, 1.0])), dtype=float)
    return (float(edges[0]), float(edges[1]))


def _f_faces_2d(
    source: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    Lx: float,
    Ly: float,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """
    Evaluates a 2D source term on the four domain faces.

    The 2D counterpart of `_f_boundary_1d`, required by the fourth-order
    ghost-node closure of ``problems.poisson_line_2d_4th``. The second-order
    discretisation does not use these values.

    Each face is sampled at the *interior* nodes of the tangential axis, which
    is the grid the closure's correction is applied on. The corners are not
    sampled: no boundary row references them.

    Parameters
    ----------
    source : callable
        f(X, Y), the same callable the interior samples are drawn from.
    x, y : np.ndarray
        Length-N interior coordinates along each axis.
    Lx, Ly : float
        Domain extents [m].

    Returns
    -------
    tuple
        ``((f_x0, f_y0), (f_x1, f_y1))``, matching `BuiltCase.f_faces`.
    """
    zeros_x, zeros_y = np.zeros_like(x), np.zeros_like(y)
    return (
        (np.asarray(source(zeros_y, y), dtype=float),
         np.asarray(source(x, zeros_x), dtype=float)),
        (np.asarray(source(zeros_y + Lx, y), dtype=float),
         np.asarray(source(x, zeros_x + Ly), dtype=float)),
    )


def _f_faces_3d(
    source: Callable[..., np.ndarray],
    axes: Sequence[np.ndarray],
    lengths: Sequence[float],
    periodic: Sequence[bool],
) -> tuple[tuple, tuple]:
    """
    Evaluates a 3D source term on the six domain faces.

    The 3D counterpart of `_f_faces_2d`. A periodic axis has no faces, so its
    entries are None and the fourth-order closure never consults them.

    Parameters
    ----------
    source : callable
        f(X, Y, Z), the same callable the interior samples are drawn from.
    axes : Sequence[np.ndarray]
        Per-axis node coordinates, as used to build the interior grid.
    lengths : Sequence[float]
        Domain extent per axis [m].
    periodic : Sequence[bool]
        Per-axis periodicity.

    Returns
    -------
    tuple
        ``(lo, hi)``, each a length-3 tuple whose entry for axis d is the source
        on that face — a 2D array over the other two axes, in their original
        order — or None for a periodic axis.
    """
    lo: list = [None, None, None]
    hi: list = [None, None, None]
    for ax in range(3):
        if periodic[ax]:
            continue
        others = [d for d in range(3) if d != ax]
        G0, G1 = np.meshgrid(axes[others[0]], axes[others[1]], indexing="ij")
        for end, store in ((0.0, lo), (float(lengths[ax]), hi)):
            args: list = [None, None, None]
            args[ax] = np.full(G0.shape, end)
            args[others[0]], args[others[1]] = G0, G1
            store[ax] = np.asarray(source(*args), dtype=float)
    return (tuple(lo), tuple(hi))


def _tst(N: int) -> np.ndarray:
    """
    N×N Toeplitz Symmetric Tridiagonal operator: main diagonal −2, off-diagonals
    +1. The 1/h² factor is folded into the right-hand side instead, matching
    ``problems/poisson_1d.build_tst_matrix``.

    Parameters
    ----------
    N : int
        System dimension.

    Returns
    -------
    np.ndarray
        N×N dense TST matrix.
    """
    return (-2.0 * np.eye(N)
            + np.diag(np.ones(N - 1), 1)
            + np.diag(np.ones(N - 1), -1))


def _kappa(A: np.ndarray) -> float:
    """
    Condition number λ_max / λ_min of a symmetric operator.

    ``np.linalg.eigvalsh`` reads only one triangle of its argument, so handing
    it a non-symmetric matrix silently returns the spectrum of a different
    operator. The symmetry test makes that failure loud.

    Parameters
    ----------
    A : np.ndarray
        N×N operator.

    Returns
    -------
    float
        κ(A), computed from eigenvalues when symmetric and from singular values
        otherwise.
    """
    if np.allclose(A, A.T, atol=1e-12):
        eigs = np.abs(np.linalg.eigvalsh(A))
    else:
        eigs = np.linalg.svd(A, compute_uv=False)
    return float(eigs.max() / eigs.min())


def _dirichlet_1d(
    N:      int,
    source: Callable[[np.ndarray], np.ndarray],
    exact:  Callable[[np.ndarray], np.ndarray] | None = None,
    alpha:  float = 0.0,
    beta:   float = 0.0,
) -> BuiltCase:
    """
    Assembles a 1D Dirichlet problem u″ = f on the interior grid.

    The boundary data is absorbed into the terminal entries of the right-hand
    side, b[0] −= α and b[−1] −= β, which is what keeps the operator a pure TST
    matrix independent of the boundary values.

    Parameters
    ----------
    N : int
        Number of interior nodes.
    source : callable
        f(x), evaluated at the interior nodes.
    exact : callable, optional
        Closed-form u(x); when omitted the case has no analytical reference.
    alpha, beta : float
        Dirichlet data at x = 0 and x = 1.

    Returns
    -------
    BuiltCase
        With `A`, `b`, `coords`, `f_values`, `f_boundary` and `kappa`
        populated, and `exact` populated when a closed form was supplied.
    """
    x, h = _grid_1d(N)
    A = _tst(N)
    f = source(x)
    b = h**2 * f
    b[0] -= alpha
    b[-1] -= beta
    return BuiltCase(
        coords=(x,), spacings=(h,), f_values=f,
        exact=None if exact is None else exact(x),
        A=A, b=b, kappa=_kappa(A),
        f_boundary=_f_boundary_1d(source),
    )


def _het_config_case(N: int, profile: str, V_d: float) -> BuiltCase:
    """
    Assembles a 1D HET case from the non-dimensional `HETConfig` model.

    The charge density profile is one of the parameterised analytical forms in
    `core/source_functions.py`, scaled by the Debye group α = (L/λ_D)² ≈ 5.65×10⁴
    at the reference parameters. The anode potential enters as the
    non-dimensional α_bc = V_d / φ₀.

    Parameters
    ----------
    N : int
        Number of interior nodes.
    profile : str
        Charge density profile: 'gaussian', 'linear' or 'step'.
    V_d : float
        Discharge (anode) potential [V]. Zero gives homogeneous conditions, the
        only regime in which the linear profile admits a closed form.

    Returns
    -------
    BuiltCase
        With `A`, `b` and `kappa` populated, and `exact` populated only for the
        linear profile under homogeneous conditions.
    """
    # Deferred import: `problems` imports `core`, so importing it at module
    # scope here would close an import cycle. This is the exception permitted
    # for genuine circular dependencies.
    from core.het_config import HETConfig
    from problems.het_plasma_1d import HETPoissonProblem1D

    cfg  = HETConfig(N=N, epsilon=0.01, rho_profile=profile, V_discharge=V_d)
    prob = HETPoissonProblem1D(cfg)
    return BuiltCase(
        coords=(prob.x,), spacings=(prob.dx,),
        f_values=prob.b / prob.dx**2,
        exact=prob.analytical_solution(),
        A=prob.A, b=prob.b, kappa=prob.kappa,
    )


# -- 1D Source Terms and Closed Forms ------------------------------------------
#
# The generic trio is re-stated here rather than imported from
# `core/source_functions.py` so that each case is self-describing and the two
# registries are free to diverge — `SOURCE_FUNCTIONS_2D["fS"]` already denotes a
# different function from the 2D sinusoid used everywhere else. The definitions
# below are numerically identical to `core/source_functions.py` and to the
# copies in the HPC driver, which the case-equivalence tests assert.

def _f_sin(x): return np.sin(np.pi * x)
def _f_lin(x): return 10.0 * x
def _f_hev(x): return np.where(x >= 0.5, 1.0, -1.0)


def _u_sin(x): return -np.sin(np.pi * x) / np.pi**2
def _u_lin(x): return 5.0 * x * (x**2 - 1.0) / 3.0
def _u_hev(x):
    return np.where(x < 0.5, -x**2 / 2.0 + x / 4.0,
                             x**2 / 2.0 - 3.0 * x / 4.0 + 1.0 / 4.0)


def _u_sin_nonhom(x, alpha=1.0, beta=2.0):
    """Closed form for u″ = sin(πx), u(0) = α, u(1) = β."""
    return -np.sin(np.pi * x) / np.pi**2 + (beta - alpha) * x + alpha


def _f_het_linear_3a(x: np.ndarray) -> np.ndarray:
    """Sub-case 3a source: linear density profile, f(x) = 2x − 1."""
    return 2.0 * x - 1.0


def _u_het_linear_3a(x: np.ndarray) -> np.ndarray:
    """Sub-case 3a closed form for u″ = 2x − 1, u(0) = u(1) = 0."""
    return x**3 / 3.0 - x**2 / 2.0 + x / 6.0


def _f_het_gaussian_3b(
    x:     np.ndarray,
    L:     float = geom.L_Z,
    sigma: float = 0.005,
) -> np.ndarray:
    """
    Sub-case 3b source: Gaussian electron density over a uniform ion background,

        n_e(x) = n₀ exp(−(xL − x₀)² / 2σ²),   f(x) = −(e/ε₀) n_e

    with n₀ = 10¹⁷ m⁻³ and x₀ = 0.6 L, placing the peak near the exit plane.

    Values are returned in physical units, of order 10⁹ [V/m²]; they are not
    normalised. Any normalisation a quantum solver interface requires is applied
    inside that solver.

    Parameters
    ----------
    x : np.ndarray
        Non-dimensional node coordinates on [0, 1].
    L : float
        Physical channel length [m].
    sigma : float
        Physical Gaussian width [m].

    Returns
    -------
    np.ndarray
        Length-N source values [V/m²].
    """
    e    = 1.602e-19
    eps0 = 8.854e-12
    n0   = 1e17            # m⁻³, representative channel density
    x0   = 0.6 * L         # peak near the exit plane
    n_e  = n0 * np.exp(-((x * L - x0)**2) / (2 * sigma**2))
    return -(e / eps0) * n_e


def _f_het_neumann_3c(
    x:          np.ndarray,
    sigma_norm: float = 0.2,
    x0:         float = 0.6,
) -> np.ndarray:
    """
    Sub-case 3c source: normalised Gaussian on the unit interval,

        f(x) = −exp(−(x − x₀)² / 2σ²)

    σ is already expressed in normalised units. An earlier revision accepted a
    physical σ and divided by L, which at σ = 0.2 and L = 0.025 gave an effective
    width of 8 — an almost flat source — whilst the analytical reference used
    0.2. The discrete and exact problems were therefore not the same problem,
    which was one cause of an anomalous ~60% Thomas error.

    Parameters
    ----------
    x : np.ndarray
        Node coordinates on [0, 1].
    sigma_norm : float
        Gaussian width in normalised units.
    x0 : float
        Peak location in normalised units.

    Returns
    -------
    np.ndarray
        Source values at the nodes.
    """
    return -np.exp(-((x - x0) ** 2) / (2.0 * sigma_norm ** 2))


def _u_het_neumann_3c(
    x:          np.ndarray,
    sigma_norm: float = 0.2,
    x0:         float = 0.6,
) -> np.ndarray:
    """
    Quadrature reference for φ″ = f, φ′(0) = 0, φ(1) = 0.

    By direct integration,

        φ′(x) = φ′(0) + ∫₀ˣ f(t) dt = ∫₀ˣ f(t) dt        [Neumann]
        φ(x)  = φ(0)  + ∫₀ˣ φ′(s) ds

    with the additive constant fixed by φ(1) = 0. Neither integral carries a
    leading minus sign: the convention is φ″ = f, consistent with the discrete
    system Au = h²f.

    Quadrature is used because the Gaussian has no closed-form finite-limit
    integral. The 10⁴-point grid is far denser than any N in the sweeps, so the
    quadrature error is negligible against the discretisation error.

    Parameters
    ----------
    x : np.ndarray
        Node coordinates at which to sample the reference.
    sigma_norm : float
        Gaussian width, matching the source.
    x0 : float
        Peak location, matching the source.

    Returns
    -------
    np.ndarray
        Reference potential at the requested nodes.
    """
    from scipy.integrate import cumulative_trapezoid

    x_fine = np.linspace(0.0, 1.0, 10000)
    f_fine = -np.exp(-((x_fine - x0)**2) / (2.0 * sigma_norm**2))

    dphi_fine = cumulative_trapezoid(f_fine, x_fine, initial=0.0)      # φ′
    phi_fine  = cumulative_trapezoid(dphi_fine, x_fine, initial=0.0)   # φ
    phi_fine -= phi_fine[-1]                                           # φ(1) = 0

    return np.interp(x, x_fine, phi_fine)


def _build_3b(N: int) -> BuiltCase:
    """
    Assembles sub-case 3b: Gaussian profile with a 300 V anode.

    The anode potential is absorbed into the first row of the right-hand side,
    b[0] −= V_d. The cathode contributes nothing, φ(1) = 0. There is no closed
    form, so the Thomas solve is the reference.

    Parameters
    ----------
    N : int
        Number of interior nodes.

    Returns
    -------
    BuiltCase
        With `exact` left as None.
    """
    V_d  = geom.V_ANODE
    x, h = _grid_1d(N)
    A    = _tst(N)
    f    = _f_het_gaussian_3b(x)
    b    = h**2 * f
    b[0]  -= V_d       # φ(0) = V_d, the anode
    b[-1] -= 0.0       # φ(1) = 0, the cathode plane; retained for symmetry
    return BuiltCase(
        coords=(x,), spacings=(h,), f_values=f,
        exact=None, A=A, b=b, kappa=_kappa(A),
        f_boundary=_f_boundary_1d(_f_het_gaussian_3b),
    )


def _build_3c(N: int, sigma_norm: float = 0.2) -> BuiltCase:
    """
    Assembles sub-case 3c: φ″ = f, φ′(0) = 0 (Neumann), φ(1) = 0 (Dirichlet).

    Grid. A Neumann condition at x = 0 makes φ(0) an unknown, so the node set
    must include it: unknowns φ₀ … φ_{N−1} at xᵢ = ih with h = 1/N, and x_N = 1
    is the Dirichlet boundary absorbed into the last row. This differs
    deliberately from the interior grid used by every other 1D case — applying a
    Neumann row to index 0 of that grid would impose the condition at x = h.

    Neumann row. The ghost point φ₋₁ = φ₁ follows from the centred difference
    φ′(0) = (φ₁ − φ₋₁)/2h = 0, so the node-0 equation φ₋₁ − 2φ₀ + φ₁ = h²f₀
    becomes 2φ₁ − 2φ₀ = h²f₀. Halving gives −φ₀ + φ₁ = h²f₀/2, i.e.
    A[0,0] = −1, A[0,1] = +1, b[0] = h²f₀/2, which is symmetric. The unhalved
    form has the same solution but is not Hermitian, so HHL and QSVT would not
    be valid on it and the eigvalsh-based κ would describe a different matrix.

    Parameters
    ----------
    N : int
        Number of unknowns, including the node at x = 0.
    sigma_norm : float
        Gaussian width in normalised units.

    Returns
    -------
    BuiltCase
        With the Neumann-modified operator, its right-hand side, and the
        quadrature reference.
    """
    h = 1.0 / N
    x = np.arange(N) * h

    A = _tst(N).astype(float)
    A[0, 0] = -1.0
    A[0, 1] = +1.0

    f = _f_het_neumann_3c(x, sigma_norm=sigma_norm)
    b = h**2 * f
    b[0] *= 0.5          # the same halving applied to the right-hand side

    return BuiltCase(
        coords=(x,), spacings=(h,), f_values=f,
        exact=_u_het_neumann_3c(x, sigma_norm=sigma_norm),
        A=A, b=b, kappa=_kappa(A),
    )


# -- 1D Generic Poisson Cases --------------------------------------------------

register(Case(
    name="poisson_1d_fS_hom",
    dim=1, family="poisson",
    summary="u″ = sin(πx), homogeneous Dirichlet",
    build=lambda N: _dirichlet_1d(N, _f_sin, _u_sin),
    default_N=(4, 8, 16, 32, 64),
    notes=(
        "Reference Section IV A. Smooth source; the benchmark's best case for\n"
        "quantum solvers, as the spectrum of the right-hand side is a single\n"
        "mode. Legacy label 'fS'; HPC case id '1D_Poisson_fS_hom'."
    ),
))

register(Case(
    name="poisson_1d_fL_hom",
    dim=1, family="poisson",
    summary="u″ = 10x, homogeneous Dirichlet",
    build=lambda N: _dirichlet_1d(N, _f_lin, _u_lin),
    default_N=(4, 8, 16, 32, 64),
    notes=(
        "Reference Section IV A. Legacy label 'fL'; HPC case id\n"
        "'1D_Poisson_fL_hom'."
    ),
))

register(Case(
    name="poisson_1d_fH_hom",
    dim=1, family="poisson",
    summary="u″ = 2H(x−0.5)−1 (Heaviside step), homogeneous Dirichlet",
    build=lambda N: _dirichlet_1d(N, _f_hev, _u_hev),
    default_N=(4, 8, 16, 32, 64),
    notes=(
        "Reference Section IV A. Discontinuous source, so the discretisation\n"
        "error dominates and the O(h²) rate is not attained; this is the\n"
        "intended stress case. Legacy label 'fH'; HPC id '1D_Poisson_fH_hom'."
    ),
))

register(Case(
    name="poisson_1d_fS_nonhom",
    dim=1, family="poisson",
    summary="u″ = sin(πx), u(0) = 1, u(1) = 2",
    build=lambda N: _dirichlet_1d(N, _f_sin, _u_sin_nonhom, alpha=1.0, beta=2.0),
    default_N=(4, 8, 16, 32, 64),
    notes=(
        "Reference Section IV B. The only non-homogeneous 1D case with a closed\n"
        "form: u = −sin(πx)/π² + (β−α)x + α. The 2D sweep C configurations use\n"
        "α ∈ {0, −0.5}, β = 0.5 and have no closed form, so they are measured\n"
        "against Thomas instead.\n"
        "Sole prior definition site: scripts/run_hpc_1Dfull.py:954-979."
    ),
))


# -- 1D HET Cases: HPC Sub-Case Family -----------------------------------------
#
# These three are the HET axial sub-cases of the 1D HPC sweep. They are a
# distinct family from the `HETConfig` cases below: different sources, different
# scalings, different closed forms. See the module docstring on name collisions.

register(Case(
    name="het_1d_3a_linear",
    dim=1, family="het",
    summary="HET axial, linear profile f = 2x − 1, homogeneous Dirichlet",
    build=lambda N: _dirichlet_1d(N, _f_het_linear_3a, _u_het_linear_3a),
    lengths=(geom.L_Z,),
    default_N=(4, 8, 16, 32, 64),
    notes=(
        "Exact solution u = x³/3 − x²/2 + x/6.\n"
        "NOT the same problem as het_1d_linear_scaled, which uses f = −αρ₀x with\n"
        "α ≈ 5.65×10⁴ and a different closed form. Both were previously called\n"
        "'HET linear'.\n"
        "Sole prior definition site: scripts/run_hpc_1Dfull.py:499, 1002.\n"
        "HPC case id 'HET_1D_3a_linear_hom'."
    ),
))

register(Case(
    name="het_1d_3b_gaussian_Vd300",
    dim=1, family="het",
    summary="HET axial, physical Gaussian density, 300 V anode",
    build=_build_3b,
    lengths=(geom.L_Z,),
    reference="thomas",
    default_N=(4, 8, 16, 32, 64),
    notes=(
        "Source in physical units [V/m²], magnitude ~10⁹: n₀ = 10¹⁷ m⁻³,\n"
        "x₀ = 0.6 L, σ = 5 mm. Anode absorbed as b[0] −= 300.\n"
        "No closed form; the Thomas solve is the reference. The large ‖b‖ is why\n"
        "HHL recovers its proportionality constant against the normalised system\n"
        "A/‖A‖₂ rather than the raw one.\n"
        "Sole prior definition site: scripts/run_hpc_1Dfull.py:477, 1013.\n"
        "HPC case id 'HET_1D_3b_gaussian_Vd300'."
    ),
))

register(Case(
    name="het_1d_3c_neumann",
    dim=1, family="het",
    summary="HET axial, Gaussian density, Neumann(x=0) – Dirichlet(x=1)",
    build=_build_3c,
    lengths=(geom.L_Z,),
    grid="including-origin",
    reference="quadrature",
    default_N=(4, 8, 16, 32, 64),
    notes=(
        "The only Neumann boundary condition anywhere in the repository, and the\n"
        "only case using the h = 1/N grid that includes the node at x = 0.\n"
        "The Neumann row is halved to keep the operator symmetric, without which\n"
        "HHL and QSVT would not be valid on it.\n"
        "Sole prior definition site: scripts/run_hpc_1Dfull.py:504, 553, 1033.\n"
        "HPC case id 'HET_1D_3c_gaussian_NeumannDirichlet'."
    ),
))


# -- 1D HET Cases: Non-Dimensional HETConfig Family ----------------------------
#
# The parameterised-profile model of `core/het_config.py`, non-dimensionalised
# by the Debye length and the electron thermal voltage. Sweeps H1-H4 drew on
# these.

register(Case(
    name="het_1d_linear_scaled",
    dim=1, family="het",
    summary="HET axial, linear profile −αρ₀x, homogeneous (V_d = 0)",
    build=lambda N: _het_config_case(N, "linear", 0.0),
    lengths=(geom.L_Z,),
    default_N=(4, 8, 16, 32),
    notes=(
        "Exact solution αρ₀x(1−x²)/6, valid only at V_d = 0; the closed form is\n"
        "suppressed once α_bc ≠ 0. α = (L/λ_D)² ≈ 5.65×10⁴ at the reference\n"
        "parameters.\n"
        "NOT the same problem as het_1d_3a_linear — see the module docstring."
    ),
))

register(Case(
    name="het_1d_gaussian_hom",
    dim=1, family="het",
    summary="HET axial, Gaussian profile, homogeneous (V_d = 0)",
    build=lambda N: _het_config_case(N, "gaussian", 0.0),
    lengths=(geom.L_Z,),
    reference="thomas",
    default_N=(4, 8),
    notes=(
        "Sweep H1. Verifies solver fidelity against a classical baseline before\n"
        "the non-homogeneous anode term is introduced.\n"
        "Sole prior definition site: scripts/run_het_benchmark.py:188-192.\n"
        "The module docstring of that script claimed N ∈ {4, 8, 16}; the code used\n"
        "(4, 8), which is what is recorded here."
    ),
))

register(Case(
    name="het_1d_gaussian_Vd300_scaled",
    dim=1, family="het",
    summary="HET axial, Gaussian profile, physical anode V_d = 300 V",
    build=lambda N: _het_config_case(N, "gaussian", 300.0),
    lengths=(geom.L_Z,),
    reference="thomas",
    default_N=(4, 8),
    notes=(
        "Sweep H2. α_bc = V_d/φ₀ = 15 at φ₀ = 20 V. Evaluates stability under\n"
        "non-homogeneous constraints.\n"
        "Prior definition site: scripts/run_het_benchmark.py:216-220."
    ),
))

register(Case(
    name="het_1d_step_scaled",
    dim=1, family="het",
    summary="HET axial, step profile −αρ₀ sgn(x − x_ion), V_d = 300 V",
    build=lambda N: _het_config_case(N, "step", 300.0),
    lengths=(geom.L_Z,),
    reference="thomas",
    default_N=(4, 8),
    notes=(
        "Sweep H3, the only exercise of the 'step' charge density profile\n"
        "anywhere in the repository. Discontinuous source, so it plays the same\n"
        "stress-case role for the HET family that fH does for generic Poisson.\n"
        "Sole prior definition site: scripts/run_het_benchmark.py:243-247."
    ),
))


# -- 1D Boeuf & Garrigues (1998) Figure 5 axial profile ------------------------
#
# Boeuf & Garrigues do not solve Poisson's equation for the potential at all:
# their quasineutral model derives the field from the electron momentum
# equation instead, stated explicitly in their §III.G ("the electric field in
# a quasineutral model cannot be obtained from Poisson's equation"). There is
# therefore no charge-density source term to extract from their model for a
# Poisson benchmark — every other HET case in this registry (delta_0 sheath
# models, HETConfig's gaussian/linear/step profiles) is a generic, physically
# motivated stand-in, not something read off this paper.
#
# What the paper does report directly is the computed potential and field
# profile itself (Fig. 5(a), p. 3547, for their V_a = 200 V, SPT-100
# operating point): phi sits close to the anode potential across most of the
# channel (the "conduction zone", where electron conductivity is large
# because the magnetic field is small), then drops steeply over roughly the
# last quarter of the channel (the "acceleration region", where the field
# needed to sustain the current grows because conductivity there is small).
# The profile below is a smooth logistic fit to that curve, used as a
# manufactured solution: phi is declared directly, its second derivative is
# exact and closed-form, and the discrete Poisson solve is expected to
# reproduce phi to O(h^2).
#
# Read directly off Fig. 5(a): the transition centres close to x = 3.5 cm out
# of d = 4 cm (x-tilde = 0.875), consistent with the independently stated
# ionisation-zone location, "the maximum ion production occurs at x ~ 3 cm,
# i.e., the entrance to the acceleration region" (p. 3547) sitting just
# upstream of it, and the peak field is ~2-2.5e4 V/m. The logistic width
# below (w = 0.054) is chosen to hit that peak field magnitude; this is not a
# least-squares fit to pixel data extracted from the scan, since the paper's
# own authors note that "it is difficult to exhibit meaningful quantitative
# comparisons" (p. 3547) for exactly this reason — the fit is anchored to the
# figure's reported shape and the peak-field order of magnitude, not to
# individually digitised points.
#
# One consequence of the fit is not a defect: phi(x-tilde=1) comes out to
# ~18 V rather than exactly 0. Boeuf & Garrigues' own voltage accounting
# explains why a nonzero residual here is physically correct rather than a
# fitting artefact — "the discharge voltage is the sum of the cathode fall
# voltage (on the order of 10-20 V) and the possible voltage drop in the
# plasma region... The voltage V which is imposed in the model is the voltage
# drop along the column, and not the discharge voltage" (p. 3545). 18 V sits
# inside their own stated 10-20 V cathode-fall range.

_BG1998_VA: float = 200.0     # Column voltage for the Fig. 5 operating point [V]
_BG1998_XC: float = 0.875     # Non-dimensional transition centre (x = 3.5 cm of 4 cm)
_BG1998_W:  float = 0.054     # Non-dimensional transition width; sets the peak field


def _bg1998_sigmoid(x: np.ndarray) -> np.ndarray:
    """Logistic weight, 1 near the anode (x=0) and ~0 past the transition."""
    z = (_BG1998_XC - x) / _BG1998_W
    return 1.0 / (1.0 + np.exp(-z))


def _u_het_bg1998(x: np.ndarray) -> np.ndarray:
    """
    Closed-form target potential [V]: a logistic fit to Boeuf & Garrigues
    (1998) Fig. 5(a). See the module comment above for the fit's provenance.
    """
    return _BG1998_VA * _bg1998_sigmoid(x)


def _f_het_bg1998(x: np.ndarray) -> np.ndarray:
    """
    Closed-form source d^2(phi)/d(x-tilde)^2 [V] for `_u_het_bg1998`, exact
    (not finite-differenced): with s = sigmoid((x_c-x)/w),

        phi''(x) = (V_a / w^2) * s * (1-s) * (1-2s).
    """
    s = _bg1998_sigmoid(x)
    return (_BG1998_VA / _BG1998_W**2) * s * (1.0 - s) * (1.0 - 2.0 * s)


register(Case(
    name="het_1d_bg1998_fig5_profile",
    dim=1, family="het",
    summary="HET axial potential fit to Boeuf-Garrigues (1998) Fig. 5(a), V_a=200V",
    build=lambda N: _dirichlet_1d(
        N, source=_f_het_bg1998, exact=_u_het_bg1998,
        alpha=float(_u_het_bg1998(np.array(0.0))),
        beta=float(_u_het_bg1998(np.array(1.0))),
    ),
    lengths=(geom.L_Z,),
    reference="manufactured",
    default_N=(4, 8, 16, 32, 64),
    notes=(
        "New in this consolidation, not a prior duplicate. Every other HET\n"
        "case's charge density is a generic, physically motivated stand-in\n"
        "(delta_0 sheath models, HETConfig's gaussian/linear/step profiles);\n"
        "this one is instead anchored to the specific potential/field curve\n"
        "Boeuf & Garrigues (1998) actually report in Fig. 5(a), because their\n"
        "quasineutral transport model does not solve Poisson's equation for\n"
        "the field at all (their Sec. III.G) and so has no charge-density\n"
        "source term to extract in the first place. See the module comment\n"
        "above `_BG1998_VA` for the full derivation, including why phi(1) is\n"
        "~18V rather than exactly 0 (the paper's own unmodelled cathode-fall\n"
        "voltage, stated as 10-20V) and why this is a shape/order-of-\n"
        "magnitude fit rather than a pixel-digitised one (the paper's own\n"
        "authors call meaningful quantitative comparison 'difficult').\n"
        "V_a = 200V matches the specific Fig. 5 operating point, not the\n"
        "300V discharge voltage used by this registry's other HET cases."
    ),
))


# -- 2D Assembly Helpers -------------------------------------------------------

def _grid_2d(
    N:  int,
    Lx: float = 1.0,
    Ly: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Vertex-centred interior grid on a rectangle, matching `PoissonLine2D`.

    Nodes at xᵢ = i·Δx for i = 1 … N with Δx = Lx/(N+1), and likewise in y. The
    boundaries are excluded, their data entering through the absorbed
    right-hand side instead.

    Parameters
    ----------
    N : int
        Nodes per direction.
    Lx, Ly : float
        Domain extents.

    Returns
    -------
    X, Y : np.ndarray
        (N, N) coordinate fields, built with ``indexing="ij"``.
    dx, dy : float
        Mesh spacings.
    """
    dx, dy = Lx / (N + 1), Ly / (N + 1)
    x = np.arange(1, N + 1) * dx
    y = np.arange(1, N + 1) * dy
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X, Y, dx, dy


def _line_2d(
    N:      int,
    Lx:     float,
    Ly:     float,
    source: Callable[[np.ndarray, np.ndarray], np.ndarray],
    exact:  Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    **bcs,
) -> BuiltCase:
    """
    Assembles a 2D line-decomposed problem.

    Parameters
    ----------
    N : int
        Nodes per direction.
    Lx, Ly : float
        Domain extents [m] for the dimensional HET cases, unity otherwise.
    source : callable
        f(X, Y) on the (N, N) interior grid.
    exact : callable, optional
        Manufactured or analytical solution φ(X, Y).
    **bcs
        Dirichlet data passed through to `PoissonLine2D`: ``bc_x0``, ``bc_x1``,
        ``bc_y0``, ``bc_y1``, each scalar or a length-N array.

    Returns
    -------
    BuiltCase
        With `problem` populated and `kappa` set to the strip condition number
        κ(A_row), which tends to 3⁻ as N → ∞ and is therefore far better
        conditioned than the O(N²) 1D operator.
    """
    # Deferred import: `problems` imports `core`, so a module-scope import here
    # would close an import cycle.
    from problems.poisson_line_2d import PoissonLine2D

    X, Y, dx, dy = _grid_2d(N, Lx, Ly)
    f = source(X, Y)
    prob = PoissonLine2D(f, Lx=Lx, Ly=Ly, **bcs)
    return BuiltCase(
        coords=(X, Y), spacings=(dx, dy), f_values=f,
        exact=None if exact is None else exact(X, Y),
        problem=prob, kappa=prob.kappa_row(),
        f_faces=_f_faces_2d(source, X[:, 0], Y[0, :], Lx, Ly),
    )


def _f_two_gaussian(x, y, Lx=0.01, Ly=0.01):
    """
    Two-Gaussian space charge, the PlasmaNet benchmark source.

        f = ∇²φ = −ρ/ε₀

    with ρ the sum of two Gaussian blobs of width σ = 0.1 Lx centred at
    (0.3, 0.3) and (0.7, 0.7) in units of the domain, peak density
    n₀ = 10¹⁶ m⁻³.

    Parameters
    ----------
    x, y : np.ndarray
        Coordinate fields [m].
    Lx, Ly : float
        Domain extents [m].

    Returns
    -------
    np.ndarray
        Source field [V/m²].
    """
    sigma, n0, e, eps0 = 0.1 * Lx, 1e16, 1.602e-19, 8.854e-12
    rho  = n0 * e * np.exp(-((x - 0.3 * Lx)**2 + (y - 0.3 * Ly)**2) / (2 * sigma**2))
    rho += n0 * e * np.exp(-((x - 0.7 * Lx)**2 + (y - 0.7 * Ly)**2) / (2 * sigma**2))
    return -rho / eps0


def _two_gaussian_reference(N, Lx=0.01, Ly=0.01, n_fine=200, modes=50):
    """
    Truncated Fourier reference for the two-Gaussian source, sampled at the
    coarse nodes.

    The fine grid is deliberately independent of the solver resolution.
    Computing the coefficients on the solver grid introduces a large quadrature
    error at small N — about 32% at N = 4, where four points per direction
    cannot resolve a Gaussian of width σ = 0.1 Lx.

    Parameters
    ----------
    N : int
        Coarse resolution at which to sample the reference.
    Lx, Ly : float
        Domain extents [m].
    n_fine : int
        Fine-grid resolution per direction on which the coefficients are formed.
    modes : int
        Number of Fourier modes retained per direction.

    Returns
    -------
    np.ndarray
        (N, N) reference potential at the coarse interior nodes.
    """
    from scipy.interpolate import RegularGridInterpolator

    dx_f, dy_f = Lx / (n_fine + 1), Ly / (n_fine + 1)
    x_f = np.arange(1, n_fine + 1) * dx_f
    y_f = np.arange(1, n_fine + 1) * dy_f
    xf, yf = np.meshgrid(x_f, y_f, indexing="ij")
    f_fine = _f_two_gaussian(xf, yf, Lx, Ly)

    phi_fine = np.zeros((n_fine, n_fine))
    for n in range(1, modes + 1):
        sin_x = np.sin(n * np.pi * xf / Lx)
        for m in range(1, modes + 1):
            sin_y = np.sin(m * np.pi * yf / Ly)
            R_nm  = (4.0 / (Lx * Ly)) * np.sum(f_fine * sin_x * sin_y) * dx_f * dy_f
            denom = -np.pi**2 * (n**2 / Lx**2 + m**2 / Ly**2)
            phi_fine += (R_nm / denom) * sin_x * sin_y

    interp = RegularGridInterpolator((x_f, y_f), phi_fine, method="linear",
                                     bounds_error=False, fill_value=0.0)
    X, Y, _, _ = _grid_2d(N, Lx, Ly)
    return interp(np.stack([X.ravel(), Y.ravel()], axis=-1)).reshape(N, N)


def _build_two_gaussian(N: int) -> BuiltCase:
    """
    Assembles the two-Gaussian PlasmaNet benchmark with its Fourier reference.

    Parameters
    ----------
    N : int
        Nodes per direction.

    Returns
    -------
    BuiltCase
        With `exact` holding the truncated Fourier reference.
    """
    Lx = Ly = 0.01
    built = _line_2d(N, Lx, Ly, lambda X, Y: _f_two_gaussian(X, Y, Lx, Ly))
    built.exact = _two_gaussian_reference(N, Lx, Ly)
    return built


def _build_het_2d_mms(N: int) -> BuiltCase:
    """
    Assembles the 2D HET manufactured solution on the axial-radial slice.

        φ(z, r) = φ₀ sin(πz/L_z) cos(πr/2L_r)

    The cosine radial profile is non-zero at the inner wall, so r = 0 carries
    the axial profile as Dirichlet data whilst the anode, cathode and outer wall
    are all zero.

    Parameters
    ----------
    N : int
        Nodes per direction.

    Returns
    -------
    BuiltCase
        With the inner-wall boundary data applied as `bc_y0`.
    """
    Lz, Lr, phi0 = geom.L_Z, geom.L_R, geom.PHI_0

    def phi(z, r):
        return phi0 * np.sin(np.pi * z / Lz) * np.cos(np.pi * r / (2 * Lr))

    def f(z, r):
        coeff = -phi0 * np.pi**2 * (1.0 / Lz**2 + 1.0 / (4.0 * Lr**2))
        return coeff * np.sin(np.pi * z / Lz) * np.cos(np.pi * r / (2 * Lr))

    dz = Lz / (N + 1)
    z_pts = np.arange(1, N + 1) * dz
    bc_inner = phi0 * np.sin(np.pi * z_pts / Lz)
    return _line_2d(N, Lz, Lr, f, phi, bc_y0=bc_inner)


def _build_het_2d_bg(N: int) -> BuiltCase:
    """
    Assembles the 2D Boeuf-Garrigues case: the analytical sheath charge density
    on the axial-radial channel with physical electrode potentials.

    No closed form exists, so the reference is a classical solve on a mesh
    refined by 9 per direction, with the source re-evaluated on the refined mesh
    rather than interpolated, so that the reference carries only its own
    O(h_fine²) truncation error.

    Boundary conditions are the anode at α_bc, the cathode at zero, and both
    radial walls grounded. The grounded inner wall is the documented physics; an
    earlier revision applied the anode term at the inner wall whilst scoring the
    residual against zero there, so the reported residual belonged to a
    different system than the one solved.

    Parameters
    ----------
    N : int
        Nodes per direction.

    Returns
    -------
    BuiltCase
        With `exact` left as None; the reference requires a solve and is
        obtained through `benchmark.reference_2d.fine_mesh_reference`.
    """
    # Deferred import: `problems` imports `core`, closing a cycle if imported at
    # module scope.
    from core.het_config import HETConfig2D
    from problems.het_plasma_2d import build_het_problem

    cfg  = HETConfig2D(V_discharge=300.0)
    prob = build_het_problem(cfg, N)
    X, Y = cfg.grid(N)
    return BuiltCase(
        coords=(X, Y),
        spacings=(cfg.L_x / (N + 1), cfg.L_y / (N + 1)),
        f_values=cfg.poisson_source_at(X, Y),
        exact=None, problem=prob, kappa=prob.kappa_row(),
        f_faces=_f_faces_2d(cfg.poisson_source_at, X[:, 0], Y[0, :],
                            cfg.L_x, cfg.L_y),
    )


# -- 2D Generic Poisson Cases --------------------------------------------------

register(Case(
    name="poisson_2d_sin_pi",
    dim=2, family="poisson",
    summary="∇²φ = sin(πx)sin(πy) on the unit square, homogeneous Dirichlet",
    build=lambda N: _line_2d(
        N, 1.0, 1.0,
        lambda X, Y: np.sin(np.pi * X) * np.sin(np.pi * Y),
        lambda X, Y: -np.sin(np.pi * X) * np.sin(np.pi * Y) / (2.0 * np.pi**2),
    ),
    lengths=(1.0, 1.0), periodic=(False, False),
    default_N=(4, 8, 16, 32, 64),
    notes=(
        "Closed form φ = −f/(2π²). The standard 2D verification case, previously\n"
        "duplicated in four places: scripts/debug_outer_2d.py:79,\n"
        "scripts/run_hpc_2Dfull.py:308-313, tests/conftest.py:89 and\n"
        "scripts/archive/debug_2d_solvers.py:84 — all numerically identical.\n"
        "NOT the same source as poisson_2d_fS_10sin2pi, despite both having been\n"
        "described as 'the 2D sinusoidal source'.\n"
        "HPC case id '2D_Poisson_sin_hom'."
    ),
))

register(Case(
    name="poisson_2d_fS_10sin2pi",
    dim=2, family="poisson",
    summary="∇²φ = 10 sin(2πx)cos(2πy) on the unit square",
    build=lambda N: _line_2d(
        N, 1.0, 1.0,
        lambda X, Y: 10.0 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y),
    ),
    lengths=(1.0, 1.0), periodic=(False, False),
    reference="fine_mesh", ref_params={"refine": 19},
    default_N=(4, 8, 16, 32),
    notes=(
        "The legacy SOURCE_FUNCTIONS_2D['fS'] of core/source_functions.py:90,\n"
        "used by the 2D sweeps E and F. Has no closed form, unlike\n"
        "poisson_2d_sin_pi, so it is measured against a fine-mesh reference.\n"
        "The refinement factor is 19, the default of\n"
        "benchmark/reference_2d.py:REFINE_FACTOR; 17 and 9 also appear at other\n"
        "sites for nominally the same measurement."
    ),
))

register(Case(
    name="poisson_2d_two_gaussian_plasmanet",
    dim=2, family="poisson",
    summary="Two Gaussian charge blobs on a 10 mm square, Fourier reference",
    build=_build_two_gaussian,
    lengths=(0.01, 0.01), periodic=(False, False),
    reference="fourier", ref_params={"modes": 50, "n_fine": 200},
    default_N=(4, 8, 16, 32, 64),
    notes=(
        "The PlasmaNet benchmark. σ = 0.1 Lx, n₀ = 10¹⁶ m⁻³, blobs at (0.3, 0.3)\n"
        "and (0.7, 0.7) in domain units.\n"
        "The Fourier coefficients are formed on a 200² grid independent of the\n"
        "solver resolution: computing them on the solver grid gives ~32% error at\n"
        "N = 4, where four points per direction cannot resolve the blob.\n"
        "Sole prior definition site: scripts/run_hpc_2Dfull.py:318-352.\n"
        "HPC case id '2D_Poisson_TwoGaussian_PlasmaNet'."
    ),
))

register(Case(
    name="poisson_2d_single_mode_n1m1",
    dim=2, family="poisson",
    summary="Single Fourier mode n = m = 1 on the unit square",
    build=lambda N: _line_2d(
        N, 1.0, 1.0,
        lambda X, Y: -np.sin(np.pi * X) * np.sin(np.pi * Y),
        lambda X, Y: (1.0 / (np.pi**2 * 2.0))
        * np.sin(np.pi * X) * np.sin(np.pi * Y),
    ),
    lengths=(1.0, 1.0), periodic=(False, False),
    default_N=(4, 8, 16, 32, 64),
    notes=(
        "R_nm = 1. An exact eigenmode of the discrete operator up to truncation,\n"
        "so it isolates algebraic from discretisation error.\n"
        "Sole prior definition site: scripts/run_hpc_2Dfull.py:357-363.\n"
        "HPC case id '2D_Poisson_SingleMode_n1m1'."
    ),
))


# -- 2D HET Cases --------------------------------------------------------------

register(Case(
    name="het_2d_mms_spt100",
    dim=2, family="het",
    summary="HET axial-radial manufactured solution, φ₀ sin(πz/L_z)cos(πr/2L_r)",
    build=_build_het_2d_mms,
    lengths=(geom.L_Z, geom.L_R), periodic=(False, False),
    reference="manufactured",
    default_N=(4, 8, 16, 32, 64),
    notes=(
        "Uses the single canonical geometry in core/het_geometry.py, read\n"
        "directly from Boeuf & Garrigues (1998): L_z = 40 mm, L_r = 20 mm.\n"
        "This case previously used a legacy L_r = 20 mm whilst the 3D\n"
        "manufactured solution used a 'corrected' 15 mm derived from a\n"
        "secondary-source R_in = 35 mm; checking the primary source directly\n"
        "(2026-08-07) showed the 2D value had been right all along and the 3D\n"
        "'correction' was itself the error. Both now agree on 20 mm, and L_z\n"
        "changed from the old 25 mm to the paper's stated 40 mm at the same time\n"
        "— see core/het_geometry.py's module docstring for the full derivation.\n"
        "The cosine radial profile is non-zero at the inner wall, hence the\n"
        "array-valued bc_y0. This differs from the 3D HET manufactured solution,\n"
        "which uses sin(πr/L_r) and so vanishes at both walls.\n"
        "Prior sites: scripts/run_hpc_2Dfull.py:368-374 and 708-717,\n"
        "scripts/debug_outer_2d.py:89-104.  HPC case id '2D_HET_MMS_SPT100'."
    ),
))

register(Case(
    name="het_2d_sin_meeting_report",
    dim=2, family="het",
    summary="HET axial-radial sinusoid, φ₀ = 20 V, both walls grounded",
    build=lambda N: _line_2d(
        N, geom.L_Z, geom.L_R,
        lambda X, Y: (-20.0 * np.pi**2 * (1.0 / geom.L_Z**2 + 1.0 / geom.L_R**2))
        * np.sin(np.pi * X / geom.L_Z) * np.sin(np.pi * Y / geom.L_R),
        lambda X, Y: 20.0
        * np.sin(np.pi * X / geom.L_Z) * np.sin(np.pi * Y / geom.L_R),
    ),
    lengths=(geom.L_Z, geom.L_R), periodic=(False, False),
    reference="manufactured",
    default_N=(4, 8, 16, 32, 64),
    notes=(
        "Dimensional, with amplitude φ₀ = 20 V rather than the 300 V discharge\n"
        "scale. Distinct from the non-dimensional unit-square sinusoid built by\n"
        "problems/het_plasma_2d.py:118, which has amplitude 1.\n"
        "Domain extents now read from core/het_geometry.py (L_z = 40 mm,\n"
        "L_r = 20 mm) rather than a locally hardcoded (25, 20) mm literal, so\n"
        "this case tracks the single canonical SPT-100 geometry along with\n"
        "every other HET case, even though its source term is a generic\n"
        "sinusoid unrelated to the channel's actual plasma physics.\n"
        "Sole prior definition site: scripts/run_hpc_2Dfull.py:379-385, 720-726.\n"
        "HPC case id '2D_HET_Sin_MeetingReport'."
    ),
))

register(Case(
    name="het_2d_boeuf_garrigues",
    dim=2, family="het",
    summary="HET axial-radial generic sheath-charge model, V_d = 300 V",
    build=_build_het_2d_bg,
    lengths=(geom.L_Z, geom.L_R), periodic=(False, False),
    reference="fine_mesh", ref_params={"refine": 9},
    default_N=(4, 8, 16),
    notes=(
        "NOT a literal reproduction of Boeuf & Garrigues' computed field: their\n"
        "quasineutral transport model does not solve Poisson's equation for the\n"
        "potential at all (explicit in their Sec. III.G), so there is no charge\n"
        "density in their paper to extract for a 2D Poisson benchmark, and no\n"
        "2D result in the paper to compare against either — their model is 1D.\n"
        "This case's source (core/het_config.py:HETConfig2D, an axial bipolar\n"
        "sheath term times a radial wall-sheath modulation) is a generic,\n"
        "physically motivated stand-in carrying the paper's scalar parameters\n"
        "(T_e, n_0, V_discharge) but not its reported profile shape. For an\n"
        "actual fit to what the paper reports, see the 1D case\n"
        "het_1d_bg1998_fig5_profile, which is anchored to their Fig. 5(a).\n"
        "Anode at α_bc, cathode and BOTH radial walls grounded; the grounded\n"
        "inner wall is the documented physics. An earlier revision applied the\n"
        "anode term b −= α_bc at j = 0 whilst scoring the residual against\n"
        "bc_y0 = 0, so the reported residual described a different system than\n"
        "the one solved; results predating that correction are not comparable.\n"
        "refine_factor = 9 here, against the default 19 used elsewhere.\n"
        "Sole prior definition site: scripts/run_het_2d_benchmark.py:296-340."
    ),
))


# -- 3D Assembly Helpers -------------------------------------------------------
#
# Physical constants. The benchmark drivers use these four-significant-figure
# values, whilst core/het_config.py carries the full CODATA figures
# (1.602176634e-19 C, 8.854187817e-12 F/m). The truncated values are reproduced
# here so that migrated cases remain bit-identical to the results already
# produced with them; the discrepancy is ~1.1e-4 relative and is far below the
# discretisation error, but it is not zero.

_Q_E_DRIVER: float = 1.602e-19       # Elementary charge [C], as used by the drivers
_EPS0_DRIVER: float = 8.854e-12      # Vacuum permittivity [F/m], likewise


def _cube_grid(N: int) -> tuple[tuple[np.ndarray, ...], float, np.ndarray]:
    """
    Interior grid on the unit cube: h = 1/(N+1), nodes at h … Nh per axis.

    Parameters
    ----------
    N : int
        Nodes per direction.

    Returns
    -------
    coords : tuple of np.ndarray
        (X, Y, Z), each (N, N, N), built with ``indexing="ij"``.
    h : float
        Mesh spacing, identical on all three axes.
    p : np.ndarray
        The length-N 1D node vector, needed to build face data.
    """
    h = 1.0 / (N + 1)
    p = np.arange(1, N + 1) * h
    return tuple(np.meshgrid(p, p, p, indexing="ij")), h, p


def _het_grid_3d(N: int):
    """
    Unwrapped SPT-100 channel grid.

    The axial and radial axes carry Dirichlet boundaries and so use the interior
    convention h = L/(N+1). The azimuthal axis is periodic and therefore has no
    boundary node: it uses ds = L_s/N with nodes at 0, ds, …, (N−1)ds.

    Parameters
    ----------
    N : int
        Nodes per direction.

    Returns
    -------
    coords : tuple of np.ndarray
        (Z, R, S), each (N, N, N).
    spacings : tuple of float
        (dz, dr, ds) [m].
    """
    dz, dr = geom.L_Z / (N + 1), geom.L_R / (N + 1)
    ds = geom.L_S / N
    z = np.arange(1, N + 1) * dz
    r = np.arange(1, N + 1) * dr
    s = np.arange(N) * ds
    return tuple(np.meshgrid(z, r, s, indexing="ij")), (dz, dr, ds)


def _line_3d(coords, spacings, f, exact, lengths, periodic,
             source=None, **kw) -> BuiltCase:
    """
    Wraps assembled 3D field data into a `BuiltCase`.

    Parameters
    ----------
    coords : tuple of np.ndarray
        (X, Y, Z) coordinate fields.
    spacings : tuple of float
        Mesh spacing per axis.
    f : np.ndarray
        Source field.
    exact : np.ndarray or None
        Manufactured solution, when one exists.
    lengths : tuple of float
        Domain extents per axis.
    periodic : tuple of bool
        Per-axis periodicity.
    source : callable, optional
        f(X, Y, Z), the same callable `f` was sampled from. Given, the source is
        additionally evaluated on the six faces and recorded in
        `BuiltCase.f_faces` for the fourth-order closure. Callers should pass
        the identical callable they built `f` with rather than a transcription
        of it, so the two cannot drift apart.
    **kw
        Passed to `PoissonLine3D`, principally ``bc_lo`` and ``bc_hi``.

    Returns
    -------
    BuiltCase
        With `problem` populated and `kappa` the strip condition number, which
        tends to 2⁻ as N → ∞ in 3D.
    """
    # Deferred import: `problems` imports `core`, closing a cycle otherwise.
    from problems.poisson_line_3d import PoissonLine3D

    prob = PoissonLine3D(f, lengths=lengths, periodic=periodic, **kw)
    faces = None
    if source is not None:
        axes = (coords[0][:, 0, 0], coords[1][0, :, 0], coords[2][0, 0, :])
        faces = _f_faces_3d(source, axes, lengths, periodic)
    return BuiltCase(
        coords=coords, spacings=spacings, f_values=f, exact=exact,
        problem=prob, kappa=prob.kappa_row(), f_faces=faces,
    )


def _build_cube_3d(N: int) -> BuiltCase:
    """Triple-sin manufactured solution on the unit cube."""
    (X, Y, Z), h, _ = _cube_grid(N)

    # Grouped exactly as the original expression was - the sines multiplied
    # together first, then scaled - because floating-point multiplication is
    # not associative and this source feeds a published sweep.
    def src(x, y, z):
        return (-3.0 * np.pi**2) * (np.sin(np.pi * x) * np.sin(np.pi * y)
                                    * np.sin(np.pi * z))

    phi = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    return _line_3d((X, Y, Z), (h, h, h), src(X, Y, Z), phi,
                    (1.0, 1.0, 1.0), (False, False, False), source=src)


def _build_het_3d_mms(N: int, m: int = 1) -> BuiltCase:
    """
    Manufactured solution on the unwrapped SPT-100 channel,

        φ = φ₀ sin(πz/L_z) sin(πr/L_r) cos(2πm s/L_s)

    which vanishes at the anode, cathode and both walls and is exactly periodic
    in s, so the periodic stencil and the periodic grid-transfer operators are
    genuinely exercised rather than merely present.

    Parameters
    ----------
    N : int
        Nodes per direction.
    m : int
        Azimuthal mode number.

    Returns
    -------
    BuiltCase
    """
    (Zg, Rg, Sg), sp = _het_grid_3d(N)
    lap = -geom.PHI_0 * np.pi**2 * (1.0 / geom.L_Z**2 + 1.0 / geom.L_R**2
                                    + 4.0 * m**2 / geom.L_S**2)

    def src(z, r, s):
        return lap * (np.sin(np.pi * z / geom.L_Z)
                      * np.sin(np.pi * r / geom.L_R)
                      * np.cos(2.0 * np.pi * m * s / geom.L_S))

    profile = (np.sin(np.pi * Zg / geom.L_Z) * np.sin(np.pi * Rg / geom.L_R)
               * np.cos(2.0 * np.pi * m * Sg / geom.L_S))
    phi = geom.PHI_0 * profile
    return _line_3d((Zg, Rg, Sg), sp, src(Zg, Rg, Sg), phi,
                    (geom.L_Z, geom.L_R, geom.L_S), (False, False, True),
                    source=src)


def _build_het_3d_spoke(N: int) -> BuiltCase:
    """
    Rotating-spoke potential structure, manufactured so an exact solution exists,

        φ = φ₀ sin(πz/L_z) sin(πr/L_r) [1 + ε cos(2πm s/L_s)]

    an axial-radial potential well modulated azimuthally at relative amplitude
    ε = 0.30 and mode m = 2. Applying the Laplacian term by term gives the
    unmodulated well plus the azimuthal curvature of the spoke.

    Because the mode amplitude is known exactly, the azimuthal-mode relative
    error measures directly whether a solver reproduces the spoke or merely
    smears it — a stricter test than any pointwise norm, since a solver that
    damps or phase-shifts the azimuthal structure can still look acceptable in
    L∞ whilst being useless for instability work.

    Parameters
    ----------
    N : int
        Nodes per direction.

    Returns
    -------
    BuiltCase
    """
    m, eps = geom.SPOKE_MODE_M, geom.SPOKE_EPSILON
    (Zg, Rg, Sg), sp = _het_grid_3d(N)

    def src(z, r, s):
        b = np.sin(np.pi * z / geom.L_Z) * np.sin(np.pi * r / geom.L_R)
        a = np.cos(2.0 * np.pi * m * s / geom.L_S)
        return geom.PHI_0 * b * (
            -(np.pi**2 / geom.L_Z**2 + np.pi**2 / geom.L_R**2)
            * (1.0 + eps * a)
            - eps * (2.0 * np.pi * m / geom.L_S) ** 2 * a)

    base = np.sin(np.pi * Zg / geom.L_Z) * np.sin(np.pi * Rg / geom.L_R)
    azim = np.cos(2.0 * np.pi * m * Sg / geom.L_S)
    phi = geom.PHI_0 * base * (1.0 + eps * azim)
    return _line_3d((Zg, Rg, Sg), sp, src(Zg, Rg, Sg), phi,
                    (geom.L_Z, geom.L_R, geom.L_S), (False, False, True),
                    source=src)


def _build_het_3d_discharge(N: int) -> BuiltCase:
    """
    Realistic SPT-100 discharge at nominal 300 V, the production case.

    Solves ∇²φ = −ρ/ε₀ with the actual operating boundary conditions: anode
    +300 V at z = 0, cathode 0 V at z = L_z, both walls at the floating
    potential −20 V, and the azimuthal direction periodic.

    The bulk plasma is quasi-neutral, but charge separation develops near the
    exit plane where ions are accelerated out faster than electrons can follow.
    That region is modelled as a Gaussian in z centred at z = 0.8 L_z with
    σ_z = 0.12 L_z, tapered radially and modulated azimuthally by the spoke, at
    peak density n₀ = 10¹⁶ m⁻³ — about 1% of the ~10¹⁸ m⁻³ bulk, giving a
    space-charge perturbation of order 10 V on top of the 300 V applied, which
    is the correct order of magnitude for a real device.

    No closed form exists, so the Thomas solve is the reference.

    Parameters
    ----------
    N : int
        Nodes per direction.

    Returns
    -------
    BuiltCase
        With `exact` left as None.
    """
    m = geom.SPOKE_MODE_M
    (Zg, Rg, Sg), sp = _het_grid_3d(N)
    z_acc   = 0.8 * geom.L_Z      # acceleration region sits near the exit plane
    sigma_z = 0.12 * geom.L_Z
    n0      = 1.0e16              # peak net charge-carrier density [m⁻³]

    def src(z, r, s):
        n_d = (n0 * np.exp(-((z - z_acc) ** 2) / (2.0 * sigma_z**2))
               * np.sin(np.pi * r / geom.L_R)
               * (1.0 + geom.SPOKE_EPSILON
                  * np.cos(2.0 * np.pi * m * s / geom.L_S)))
        return -(_Q_E_DRIVER * n_d) / _EPS0_DRIVER

    f = src(Zg, Rg, Sg)

    bc_anode    = np.full((N, N), geom.V_ANODE)      # face z = 0,   shape (r, s)
    bc_cathode  = np.full((N, N), geom.V_CATHODE)    # face z = L_z
    bc_wall_in  = np.full((N, N), geom.V_WALL)       # face r = 0,   shape (z, s)
    bc_wall_out = np.full((N, N), geom.V_WALL)       # face r = L_r

    return _line_3d(
        (Zg, Rg, Sg), sp, f, None,
        (geom.L_Z, geom.L_R, geom.L_S), (False, False, True),
        source=src,
        bc_lo=(bc_anode, bc_wall_in, 0.0),
        bc_hi=(bc_cathode, bc_wall_out, 0.0),
    )


def _build_laplace_3d(N: int) -> BuiltCase:
    """
    Laplace equation: homogeneous PDE with non-homogeneous Dirichlet data,

        ∇²φ = 0,   φ = sin(πx) sin(πy) sinh(kz)/sinh(k),   k = √2 π

    Harmonic by construction: the two negative curvatures in x and y (−π² each)
    are exactly cancelled by the positive curvature of sinh in z (+k² = +2π²).
    Boundary data is zero on five faces and sin(πx)sin(πy) on z = L_z.

    This case closes a real gap in coverage. Every other 3D case with an exact
    solution carries zero boundary data, and the one case with real boundary
    data — the discharge — has no closed form. Without it the boundary
    absorption path in `PoissonLine3D` is never checked against a known answer
    anywhere in 3D, so a defect there would surface only as a plausible-looking
    wrong field in the production case.

    Parameters
    ----------
    N : int
        Nodes per direction.

    Returns
    -------
    BuiltCase
    """
    (X, Y, Z), h, p = _cube_grid(N)
    k = np.sqrt(2.0) * np.pi
    phi = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sinh(k * Z) / np.sinh(k)
    f = np.zeros_like(phi)
    face_xy = np.sin(np.pi * p)[:, None] * np.sin(np.pi * p)[None, :]
    return _line_3d((X, Y, Z), (h, h, h), f, phi,
                    (1.0, 1.0, 1.0), (False, False, False),
                    source=lambda x, y, z: np.zeros_like(x),
                    bc_hi=(0.0, 0.0, face_xy))


_GAUSS_SIGMA_3D = 0.12
_GAUSS_CENTRES_3D = ((0.3, 0.3, 0.35), (0.7, 0.65, 0.6))
_GAUSS_AMPS_3D = (1.0, -0.8)


def _gauss_phi_3d(X, Y, Z):
    """Sum of two Gaussian blobs of opposite sign on the unit cube."""
    out = np.zeros_like(X)
    for A, (cx, cy, cz) in zip(_GAUSS_AMPS_3D, _GAUSS_CENTRES_3D):
        out += A * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
                          / (2.0 * _GAUSS_SIGMA_3D**2))
    return out


def _gauss_src_3d(X, Y, Z):
    """
    Analytic Laplacian of the Gaussian sum. In 3D,
    ∇² exp(−r²/2σ²) = exp(−r²/2σ²)(r²/σ⁴ − 3/σ²).
    """
    out = np.zeros_like(X)
    for A, (cx, cy, cz) in zip(_GAUSS_AMPS_3D, _GAUSS_CENTRES_3D):
        r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
        out += (A * np.exp(-r2 / (2.0 * _GAUSS_SIGMA_3D**2))
                * (r2 / _GAUSS_SIGMA_3D**4 - 3.0 / _GAUSS_SIGMA_3D**2))
    return out


def _build_gaussian_3d(N: int) -> BuiltCase:
    """
    Two localised Gaussian blobs of opposite sign with exact non-homogeneous
    Dirichlet data on all six faces.

    The 3D analogue of the two-Gaussian PlasmaNet benchmark, and the standard
    shape of a plasma space-charge source: compact, steep and poorly resolved on
    coarse grids. Unlike the 2D version, which needs a 200² Fourier reference,
    this one is manufactured — φ is the Gaussian sum and f its analytic
    Laplacian — so the exact solution costs nothing.

    Two things are tested that no other case covers together: a source with real
    spatial structure (σ = 0.12 against h = 1/33 at N = 32, so only ~4 cells per
    standard deviation), and non-homogeneous data on every face rather than one.
    Its truncation error is correspondingly larger than the smooth sinusoidal
    cases; that is the point, not a defect.

    Boundary values are taken at the true boundary planes, coordinate 0 and L,
    not at the first interior node. Using the latter is an easy mistake that
    silently destroys second-order convergence.

    Parameters
    ----------
    N : int
        Nodes per direction.

    Returns
    -------
    BuiltCase
    """
    (X, Y, Z), h, p = _cube_grid(N)
    phi = _gauss_phi_3d(X, Y, Z)
    f = _gauss_src_3d(X, Y, Z)

    A, B = np.meshgrid(p, p, indexing="ij")
    zeros, ones = np.zeros_like(A), np.ones_like(A)
    bc_lo = (_gauss_phi_3d(zeros, A, B), _gauss_phi_3d(A, zeros, B),
             _gauss_phi_3d(A, B, zeros))
    bc_hi = (_gauss_phi_3d(ones, A, B), _gauss_phi_3d(A, ones, B),
             _gauss_phi_3d(A, B, ones))
    return _line_3d((X, Y, Z), (h, h, h), f, phi,
                    (1.0, 1.0, 1.0), (False, False, False),
                    source=_gauss_src_3d, bc_lo=bc_lo, bc_hi=bc_hi)


def _build_highmode_3d(N: int) -> BuiltCase:
    """
    A single high-wavenumber Fourier eigenmode, (n, m, l) = (2, 3, 4),

        φ = sin(2πx) sin(3πy) sin(4πz),   f = −π²(4+9+16) φ = −29π² φ

    The triple-sin cube is the (1,1,1) mode — the smoothest solution the grid
    can carry, which every iterative scheme handles best. This is the opposite
    end: at N = 8 the l = 4 mode has only two cells per half-wavelength, so it
    sits near the resolution limit.

    Two things are probed. Discretisation: the h² error constant scales with the
    fourth derivative, so this case shows the true accuracy cost of an
    under-resolved solution. And multigrid: high-frequency error components are
    exactly what the smoother must remove and what the coarse grid cannot
    represent, so a defective smoother or transfer operator degrades here first
    whilst looking acceptable on the (1,1,1) mode.

    Parameters
    ----------
    N : int
        Nodes per direction.

    Returns
    -------
    BuiltCase
    """
    n, m, l = 2, 3, 4
    (X, Y, Z), h, _ = _cube_grid(N)

    # Grouped as the original was: the scalar prefactor formed first, then
    # applied to the product of sines. Not associative in floating point.
    def src(x, y, z):
        return (-np.pi**2 * (n * n + m * m + l * l)) * (
            np.sin(n * np.pi * x) * np.sin(m * np.pi * y)
            * np.sin(l * np.pi * z))

    phi = (np.sin(n * np.pi * X) * np.sin(m * np.pi * Y)
           * np.sin(l * np.pi * Z))
    return _line_3d((X, Y, Z), (h, h, h), src(X, Y, Z), phi,
                    (1.0, 1.0, 1.0), (False, False, False), source=src)


# -- 3D Generic Poisson Cases --------------------------------------------------

register(Case(
    name="poisson_3d_triple_sin_cube",
    dim=3, family="poisson",
    summary="∇²φ = −3π² sin(πx)sin(πy)sin(πz) on the unit cube",
    build=_build_cube_3d,
    lengths=(1.0, 1.0, 1.0), periodic=(False, False, False),
    reference="manufactured",
    default_N=(4, 8, 16, 32),
    notes=(
        "The canonical 3D verification case and what the order-of-accuracy check\n"
        "is run on. Previously duplicated in three places: run_hpc_3Dfull.py:338,\n"
        "debug_outer_3d.py:82 and tests/conftest.py:115 — all identical.\n"
        "HPC case id '3D_Poisson_TripleSin_cube'."
    ),
))

register(Case(
    name="poisson_3d_laplace_bc_driven",
    dim=3, family="poisson",
    summary="∇²φ = 0 with sin(πx)sin(πy) on z = L; harmonic sinh solution",
    build=_build_laplace_3d,
    lengths=(1.0, 1.0, 1.0), periodic=(False, False, False),
    reference="analytical",
    default_N=(4, 8, 16, 32),
    notes=(
        "The only 3D case that checks the boundary-absorption path against a\n"
        "known answer: every other case with an exact solution carries zero\n"
        "boundary data, and the discharge case, which has real boundary data, has\n"
        "no closed form.\n"
        "Also the closest generic analogue of the physics — a real HET discharge\n"
        "is dominated by the 300 V applied across the channel, not by the\n"
        "space-charge source, so a BC-driven solution is the regime that matters.\n"
        "Sole prior definition site: scripts/run_hpc_3Dfull.py:485-519."
    ),
))

register(Case(
    name="poisson_3d_two_gaussian_cube",
    dim=3, family="poisson",
    summary="Two opposite-sign Gaussian blobs, non-homogeneous data on all faces",
    build=_build_gaussian_3d,
    lengths=(1.0, 1.0, 1.0), periodic=(False, False, False),
    reference="manufactured",
    default_N=(4, 8, 16, 32),
    notes=(
        "σ = 0.12, centres (0.3, 0.3, 0.35) and (0.7, 0.65, 0.6), amplitudes\n"
        "1.0 and −0.8. The only 3D case combining a structured source with\n"
        "non-homogeneous data on every face.\n"
        "Sole prior definition site: scripts/run_hpc_3Dfull.py:524-582."
    ),
))

register(Case(
    name="poisson_3d_high_mode_n2m3l4",
    dim=3, family="poisson",
    summary="High-wavenumber eigenmode (n, m, l) = (2, 3, 4) on the unit cube",
    build=_build_highmode_3d,
    lengths=(1.0, 1.0, 1.0), periodic=(False, False, False),
    reference="manufactured",
    default_N=(8, 16, 32),
    notes=(
        "f = −29π²φ. Near the resolution limit at N = 8, where the l = 4 mode has\n"
        "only two cells per half-wavelength. Degrades first under a defective\n"
        "smoother or transfer operator, whilst the (1,1,1) mode still looks fine.\n"
        "Sole prior definition site: scripts/run_hpc_3Dfull.py:587-617."
    ),
))


# -- 3D HET Cases --------------------------------------------------------------

register(Case(
    name="het_3d_mms_spt100",
    dim=3, family="het",
    summary="HET unwrapped-channel MMS, azimuthal mode m = 1, periodic in s",
    build=_build_het_3d_mms,
    lengths=(geom.L_Z, geom.L_R, geom.L_S), periodic=(False, False, True),
    reference="manufactured",
    default_N=(4, 8, 16, 32),
    notes=(
        "φ = φ₀ sin(πz/L_z) sin(πr/L_r) cos(2πm s/L_s), exactly periodic in s.\n"
        "The radial profile is sin(πr/L_r), vanishing at both walls — unlike the\n"
        "2D HET manufactured solution, which uses cos(πr/2L_r) and therefore\n"
        "requires inner-wall boundary data. The two are different manufactured\n"
        "solutions sharing a family name.\n"
        "Severely anisotropic at the real channel aspect ratio, ds/dr ≈ 19 at\n"
        "N = 16, so this is the case that genuinely tests anisotropic\n"
        "semi-coarsening in the multigrid hierarchy.\n"
        "Prior sites: run_hpc_3Dfull.py:359-382, debug_outer_3d.py:111.\n"
        "HPC case id '3D_HET_MMS_SPT100'."
    ),
))

register(Case(
    name="het_3d_slab_m4",
    dim=3, family="het",
    summary="HET unwrapped-channel MMS at azimuthal mode m = 4",
    build=lambda N: _build_het_3d_mms(N, m=4),
    lengths=(geom.L_Z, geom.L_R, geom.L_S), periodic=(False, False, True),
    reference="manufactured",
    default_N=(4, 8, 16, 32),
    notes=(
        "Identical geometry to het_3d_mms_spt100 at a higher azimuthal mode, so\n"
        "it probes whether the periodic transfer operators preserve structure the\n"
        "coarse grid can barely represent.\n"
        "Sole prior definition site: scripts/debug_outer_3d.py:115-116, where it\n"
        "was the 'slab' case."
    ),
))

register(Case(
    name="het_3d_rotating_spoke",
    dim=3, family="het",
    summary="HET rotating spoke, m = 2 at 30% relative amplitude",
    build=_build_het_3d_spoke,
    lengths=(geom.L_Z, geom.L_R, geom.L_S), periodic=(False, False, True),
    reference="manufactured",
    default_N=(4, 8, 16, 32),
    notes=(
        "The rotating spoke is a large-scale, low-mode coherent azimuthal\n"
        "structure observed in essentially every Hall thruster since Janes &\n"
        "Lowder (1966), characterised by McDonald & Gallimore (2011) and Sekerak\n"
        "et al. (2015). It rotates in the E×B direction at a few km/s and carries\n"
        "a substantial fraction of the discharge current, so capturing it is the\n"
        "strongest physical argument for simulating a HET in 3D rather than 2D.\n"
        "Manufactured, so the mode amplitude is known exactly and the azimuthal\n"
        "mode error measures whether a solver reproduces the spoke or smears it.\n"
        "Sole prior definition site: scripts/run_hpc_3Dfull.py:387-424.\n"
        "HPC case id '3D_HET_RotatingSpoke_SPT100'."
    ),
))

register(Case(
    name="het_3d_discharge_spt100",
    dim=3, family="het",
    summary="SPT-100 discharge at 300 V: anode/cathode/floating walls, periodic",
    build=_build_het_3d_discharge,
    lengths=(geom.L_Z, geom.L_R, geom.L_S), periodic=(False, False, True),
    reference="thomas",
    default_N=(4, 8, 16, 32),
    notes=(
        "The production case, and the only one anywhere in the repository with\n"
        "real HET operating boundary conditions: anode +300 V, cathode 0 V, both\n"
        "walls floating at −20 V, azimuthal periodic.\n"
        "Space charge n₀ = 10¹⁶ m⁻³ at z = 0.8 L_z, σ_z = 0.12 L_z, modulated by\n"
        "the m = 2 spoke. No closed form; Thomas is the reference.\n"
        "Uses the drivers' truncated constants e = 1.602e-19, ε₀ = 8.854e-12\n"
        "rather than the CODATA values in core/het_config.py, for bit-exact\n"
        "agreement with results already produced.\n"
        "This is the case whose cost predicts what a quantum-in-the-loop HET\n"
        "simulation would pay per timestep.\n"
        "Sole prior definition site: scripts/run_hpc_3Dfull.py:429-480."
    ),
))
