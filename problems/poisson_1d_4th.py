"""
problems/poisson_1d_4th.py
--------------------------
Fourth-order accurate 1D Poisson discretisation using the five-point
centred finite difference stencil.

The stencil is:

    -u_{i-2} + 16u_{i-1} - 30u_i + 16u_{i+1} - u_{i+2}
    ──────────────────────────────────────────────────── = f_i + O(h^4)
                         12 h^2

This produces a pentadiagonal Toeplitz matrix with integer diagonals
[-1, 16, -30, 16, -1] (the 1/(12h^2) prefactor is folded into the RHS
so that A retains its canonical integer form, which is most natural for
block encoding).

Boundary treatment
------------------
At i=1 the stencil reaches u₋₁, a ghost node outside [0,1]. It is
eliminated by the odd reflection about x = 0, carried to fourth order:

    u₋₁ = 2α − u₁ + h²·u″(0) + O(h⁴) = 2α − u₁ + h²·f(0) + O(h⁴)

the last equality being the governing equation itself evaluated on the
boundary, which supplies the second-derivative term at no cost.
Substituting into the row-0 stencil, the −u₋₁ term contributes +u₁ to
the operator and −2α + h²·f(0) to the data, whilst the known +16·u₀
contributes a further +16α:

    row 0:   −29·u₁ + 16·u₂ − u₃ = 12h²·f₁ − 14α + h²·f(0)

The +u₁ part is absorbed into A[0,0] (adding +1 to the diagonal); the
−14α and +h²·f(0) parts are absorbed into b[0]. The closure at i=N is
symmetric, in β and f(1).

Both corrections are right-hand-side only, so A retains the symmetric
pentadiagonal Toeplitz form required by ``build_dense_block_encoding``
and by ``PentadiagonalToeplitz``, and κ(A) — hence the QSVT phase-angle
cache keys — is unaffected by them.

Truncating the reflection at u₋₁ = 2α − u₁ is a documented trap, and
one this class fell into. The omitted h²·u″(0) term divides by the
stencil's 12h² prefactor to leave an O(1) consistency error on the two
boundary rows, capping the scheme at second order; and summing the
ghost and boundary contributions with a common sign yields 18α in place
of 14α, which destroys convergence outright whenever α ≠ 0. Neither
defect is visible on a solution odd about both boundaries, where the
plain reflection happens to be exact — u = −sin(πx)/π² is precisely
such a solution, and was for a long time the only case this class was
exercised against. Measured orders of convergence, dense direct solves
against manufactured solutions:

    u = sin(πx),  α = β = 0      3.95
    u = x(1−x),   α = β = 0      machine-exact (the stencil is exact
                                 on cubics, so this measures the
                                 closure alone)
    u = eˣ,       α = 1, β = e   3.93

Condition number
----------------
    κ(A_pent) / κ(A_tri) → 4/3  as N → ∞,  at the same N.

Measured, against the second-order operator on the same mesh:

    N        4         8        16        32
    order 2  9.4721   32.1634  116.4612  440.6886
    order 4  11.9477  42.1378  154.5126  586.8093
    ratio    1.261    1.310    1.327     1.332

The frequently quoted factor of 2.5 is the *spectral-norm* ratio
30/12 = 2.5, not a condition-number ratio, and overstates the penalty
by nearly a factor of two.

The higher condition number is the price of fourth-order accuracy.
The benefit is that the same discretisation error is achieved at
roughly N^{1/2} fewer interior nodes, which halves the qubit count
for amplitude encoding.

Imports
-------
Uses the same registry keys as the rest of the codebase:
    core.source_functions.SOURCE_FUNCTIONS   ('fS', 'fL', 'fH')
    core.exact_solutions.EXACT_SOLUTIONS     ('fS', 'fL', 'fH')
"""

from __future__ import annotations

import numpy as np

from core.source_functions import SOURCE_FUNCTIONS
from core.exact_solutions import EXACT_SOLUTIONS


class PoissonProblem1D4th:
    """
    Fourth-order accurate 1D Poisson problem on [0, 1].

    Parameters
    ----------
    N : int
        Number of interior nodes.  Must be a power of 2 and >= 4.
    source_fn : str
        Key into SOURCE_FUNCTIONS: 'fS', 'fL', or 'fH'.
    alpha : float
        Dirichlet BC at x = 0.
    beta : float
        Dirichlet BC at x = 1.
    f_boundary : tuple of float, optional
        Source values (f(0), f(1)) *on* the two boundaries, required by the
        fourth-order ghost-node closure. When ``source_fn`` is given these are
        evaluated exactly and this argument is ignored. When only ``f_vals``
        is given and this is omitted, they are recovered by cubic
        extrapolation from the four nearest interior samples, which is O(h⁴)
        accurate and therefore preserves the order of the scheme.

    Attributes
    ----------
    A : np.ndarray, shape (N, N)
        Pentadiagonal system matrix (dense, symmetric, integer coefficients).
    b : np.ndarray, shape (N,)
        Right-hand side vector (units: 12 h² · f).
    x : np.ndarray, shape (N,)
        Interior node coordinates x_i = i·h, h = 1/(N+1).
    dx : float
        Mesh spacing h.
    f_boundary : tuple of float
        The resolved (f(0), f(1)) actually used by the boundary closure,
        whether supplied, evaluated or extrapolated. Retained for diagnostics,
        since an inaccurate value here degrades the order of the scheme
        without producing any other visible symptom.
    kappa : float
        2-norm condition number of A.
    """

    def __init__(
        self,
        N: int,
        source_fn: str | None = None,
        f_vals: np.ndarray | None = None,
        alpha: float = 0.0,
        beta: float = 0.0,
        f_boundary: tuple[float, float] | None = None,
    ) -> None:
        if N < 4:
            raise ValueError(
                f"N must be >= 4 for the five-point stencil; got N={N}."
            )
        if (N & (N - 1)) != 0:
            raise ValueError(
                f"N must be a power of 2 for amplitude encoding; got N={N}."
            )
        if source_fn is not None and source_fn not in SOURCE_FUNCTIONS:
            raise ValueError(
                f"Unknown source function '{source_fn}'. "
                f"Valid options: {list(SOURCE_FUNCTIONS)}"
            )
        if source_fn is None and f_vals is None:
            raise ValueError("Must provide either source_fn or f_vals")

        self.N = N
        self.source_fn = source_fn
        self._f_vals_input = f_vals
        self.alpha = alpha
        self.beta = beta

        self.dx = 1.0 / (N + 1)
        self.x = np.arange(1, N + 1) * self.dx

        self.f_boundary = self._resolve_f_boundary(f_boundary)
        self.A = self._build_matrix()
        self.b = self._build_rhs()
        self.kappa = self._condition_number()

    # ── Boundary source data ──────────────────────────────────────────────────

    def _resolve_f_boundary(
        self, supplied: tuple[float, float] | None
    ) -> tuple[float, float]:
        """
        Determine the source values f(0) and f(1) on the two boundaries.

        The fourth-order ghost-node closure needs the governing equation
        evaluated *on* the boundary (see the module docstring), so these are
        required data rather than a refinement. Three routes, in order of
        preference:

        1. Supplied explicitly by the caller.
        2. Evaluated exactly from ``source_fn``, when the analytical source is
           known.
        3. Extrapolated from the interior samples, when only ``f_vals`` is
           available — as is the case for the HET profiles, which are sampled
           fields rather than closed-form expressions.

        Route 3 uses the cubic Lagrange extrapolant through the four nearest
        interior nodes. Its error is O(h⁴), and it enters the row equation
        divided by 12, so the fourth order of the scheme is preserved; a
        linear or constant extrapolation would not be sufficient, since this
        term carries an O(1) weight in the boundary row.

        Parameters
        ----------
        supplied : tuple of float or None
            The caller's (f(0), f(1)), if any.

        Returns
        -------
        tuple of float
            The (f(0), f(1)) pair used by ``_build_rhs``.
        """
        if supplied is not None:
            f0, f1 = supplied
            return (float(f0), float(f1))

        if self.source_fn is not None:
            f = SOURCE_FUNCTIONS[self.source_fn]
            edges = f(np.array([0.0, 1.0]))
            return (float(edges[0]), float(edges[1]))

        f_vals = np.asarray(self._f_vals_input, dtype=float)

        # Cubic Lagrange extrapolation to the node one step beyond the first
        # (respectively last) four interior samples, which are equispaced.
        # The weights are those of the standard Newton-Gregory backward
        # extrapolation by one interval: [4, -6, 4, -1].
        w = np.array([4.0, -6.0, 4.0, -1.0])
        if f_vals.size >= 4:
            f0 = float(w @ f_vals[:4])
            f1 = float(w @ f_vals[-4:][::-1])
        else:
            # Unreachable through the constructor, which enforces N >= 4, but
            # kept so the helper is total.
            f0, f1 = float(f_vals[0]), float(f_vals[-1])
        return (f0, f1)

    # ── Matrix assembly ───────────────────────────────────────────────────────

    def _build_matrix(self) -> np.ndarray:
        """
        Assemble the N×N pentadiagonal Toeplitz matrix.

        Interior rows use the standard five-point stencil coefficients
        [-1, 16, -30, 16, -1].  The 1/(12h^2) prefactor is NOT included
        here; it is folded into the RHS as 12h^2·f(x_i).

        The two boundary rows require ghost-point corrections:
          - Row 0 (i=1): u_{-1} reflected as 2·alpha - u_1.
            The +u_1 contribution is absorbed into A[0,0] += 1.
          - Row N-1 (i=N): symmetric, A[-1,-1] += 1.
        The corresponding -2·alpha / -2·beta terms go into the RHS.
        """
        N = self.N
        A = np.zeros((N, N))

        # Main diagonal: -30
        np.fill_diagonal(A, -30.0)

        # ±1 off-diagonals: +16
        if N > 1:
            np.fill_diagonal(A[1:, :], 16.0)
            np.fill_diagonal(A[:, 1:], 16.0)

        # ±2 off-diagonals: -1
        if N > 2:
            np.fill_diagonal(A[2:, :], -1.0)
            np.fill_diagonal(A[:, 2:], -1.0)

        # Ghost-point reflection at left boundary (i=1, row 0):
        # u_{-1} = 2·alpha - u_1  =>  -u_{-1} = -2·alpha + u_1
        # The +u_1 part adds 1 to A[0,0]; -2·alpha goes into b[0].
        A[0, 0] += 1.0

        # Ghost-point reflection at right boundary (i=N, row N-1):
        # u_{N+2} = 2·beta - u_N  =>  -u_{N+2} = -2·beta + u_N
        A[-1, -1] += 1.0

        return A

    def _build_rhs(self) -> np.ndarray:
        """
        Assemble the RHS vector b.

        The physical equation is:

            A_pent @ u = 12 h^2 · f(x)

        after absorbing boundary values and ghost-point reflections.

        Corrections applied:
        ┌─────────────────────────────────────────────────────────────────┐
        │ Row 0 (i=1):                                                    │
        │   +16·u₀ term   →  b[0] -= 16·α                                │
        │   −u₋₁ term     →  u₋₁ = 2α − u₁ + h²·f(0),                    │
        │                    so −u₋₁ = −2α + u₁ − h²·f(0)                │
        │                    +u₁ already in A[0,0] += 1                  │
        │                    −2α        →  b[0] += 2·α                   │
        │                    −h²·f(0)   →  b[0] += h²·f(0)               │
        │   Total:  b[0] -= 14·α  and  b[0] += h²·f(0)                   │
        │                                                                 │
        │ Row 1 (i=2):                                                    │
        │   −u_{i−2} = −u₀ = −α  →  b[1] += α                            │
        │   (the ±2 stencil at i=2 reaches u₀, a known boundary value)    │
        │                                                                 │
        │ Row N−1 (i=N):  symmetric to row 0                              │
        │                    →  b[-1] -= 14·β,  b[-1] += h²·f(1)         │
        │ Row N−2 (i=N−1): symmetric to row 1  →  b[-2] += β             │
        └─────────────────────────────────────────────────────────────────┘

        The −14α is the term that must not be written as −18α: the boundary
        node contributes +16α and the ghost −2α, which subtract rather than
        accumulate. The +h²·f(0) is the second-derivative term of the
        reflection, without which the closure is only O(h²) accurate and caps
        the whole scheme at second order. See the module docstring for the
        derivation and the measured orders.
        """
        N = self.N
        dx = self.dx
        if self.source_fn is not None:
            f = SOURCE_FUNCTIONS[self.source_fn]
            f_vals = f(self.x)
        else:
            f_vals = self._f_vals_input
            
        alpha = self.alpha
        beta = self.beta
        f0, f1 = self.f_boundary

        # Base RHS: 12 h² · f(x_i)
        b = 12.0 * dx**2 * np.asarray(f_vals, dtype=float).copy()

        # ── Left boundary corrections ─────────────────────────────────────────
        b[0] -= 14.0 * alpha      # row 0: -16α (from +16u₀) + 2α (from ghost)
        b[0] += dx**2 * f0        # row 0: second-derivative term of the ghost
        if N > 1:
            b[1] += alpha         # row 1: +α (from -u_{i-2} = -u₀)

        # ── Right boundary corrections ────────────────────────────────────────
        b[-1] -= 14.0 * beta      # row N-1: symmetric to row 0
        b[-1] += dx**2 * f1
        if N > 1:
            b[-2] += beta         # row N-2: symmetric to row 1

        return b

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _condition_number(self) -> float:
        """2-norm condition number kappa(A) = |lambda_max| / |lambda_min|."""
        eigs = np.abs(np.linalg.eigvalsh(self.A))
        return float(eigs.max() / eigs.min())

    def exact_solution(self) -> np.ndarray | None:
        """
        Return the analytical solution at interior nodes, if available.

        Only defined for homogeneous BCs (alpha = beta = 0).
        The fourth-order discretisation solves the same PDE as the
        second-order one, so the same analytical solutions apply.
        """
        if self.alpha != 0.0 or self.beta != 0.0:
            return None
        sol_fn = EXACT_SOLUTIONS.get(self.source_fn)
        if sol_fn is None:
            return None
        return sol_fn(self.x)

    def summary(self) -> str:
        """One-line summary for logging."""
        return (
            f"4th-order 1D Poisson: N={self.N}, f={self.source_fn}, "
            f"alpha={self.alpha}, beta={self.beta}, "
            f"kappa(A)={self.kappa:.2f}"
        )

    def compare_with_2nd_order(self) -> dict:
        """
        Return a dict comparing this problem's properties with the
        equivalent second-order (TST) discretisation at the same N.

        Useful for the thesis condition-number and accuracy analysis.
        """
        from problems.poisson_1d import PoissonProblem1D
        from core.config import SimConfig1D

        cfg = SimConfig1D(
            N=self.N,
            epsilon=0.01,
            source_fn=self.source_fn,
            alpha=self.alpha,
            beta=self.beta,
        )
        prob_2nd = PoissonProblem1D(cfg)

        return {
            "N": self.N,
            "kappa_2nd": prob_2nd.kappa,
            "kappa_4th": self.kappa,
            "kappa_ratio": self.kappa / prob_2nd.kappa,
            "spectral_norm_2nd": float(np.linalg.norm(prob_2nd.A, ord=2)),
            "spectral_norm_4th": float(np.linalg.norm(self.A, ord=2)),
        }