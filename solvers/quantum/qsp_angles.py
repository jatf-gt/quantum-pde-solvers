"""
Quantum Signal Processing (QSP) phase angle computation for the matrix
inversion polynomial used in QSVT.

Mathematical foundation
-----------------------
QSP realises a degree-d polynomial p(x) of definite parity (even or odd) with
|p(x)| ≤ 1 on [-1, 1] as a sequence of d+1 phase angles interleaved with the
signal unitary. For matrix inversion the target is

    p(x) ≈ 1/x    for x ∈ [1/κ, 1]

Phase computation uses pyqsp's `sym_qsp` method (Dong, Lin, Ni & Wang,
arXiv:2307.12468), which finds the *reduced* (parity-folded) phase sequence via
Newton iteration and reconstructs the full phase sequence via
`SymmetricQSPProtocol.full_phases`.

Why this module does NOT hand-roll the phase reconstruction
-----------------------------------------------------------
An earlier version of this module reconstructed the full phase sequence manually
from the reduced (half) sequence returned by the Newton solver, using formulas
such as `concatenate([reduced, reduced[::-1]])`. This is the *mirror image* of
pyqsp's own convention. Inspecting
`pyqsp.sym_qsp_opt.SymmetricQSPProtocol.__init__`, the correct reconstruction
for odd parity is

    full_phases = concatenate([flip(reduced), reduced])   # reversed FIRST

not `concatenate([reduced, flip(reduced)])`. Getting this backwards still
produces the right *length*, which is why it was so easily mistaken for a
different bug (a length mismatch) — but it swaps which end of the polynomial
domain corresponds to which index, which is exactly the "reflected about the
midpoint" symptom observed in the HET solutions. This module therefore never
concatenates reduced phases by hand: both the direct and warm-started code paths
obtain the full phase sequence from `SymmetricQSPProtocol.full_phases` /
`update_reduced_phases`, which perform this reconstruction internally and
correctly.

Warm start — tried, and deliberately NOT included
-------------------------------------------------
An earlier version of this module explored warm-starting the Newton solve from a
cheap low-degree "pilot" solution, interpolated up to the target degree, to cut
the iteration count at large κ. This was re-tested directly: for κ = 9.47,
ε = 0.01 (degree 1181), pyqsp's own default initial guess
(`reduced_phases = coef/2`) converges cleanly in 6 iterations to a residual of
1e-14. Seeding the same solver from a degree-63 pilot's phases, linearly
interpolated up to the target's 591 reduced phases, does NOT converge within 15
iterations — the residual grows (3.2 → 8.7, then oscillates around 4–5) instead
of shrinking. The naive interpolated guess knocks the iteration out of the
solver's basin of attraction; it is worse than doing nothing.

Iteration count is essentially degree-independent for this problem (5–6
iterations from degree ~500 to ~4000 in the logged runs, and the same 6 at
degree 1181) **when using the uncapped PolyOneOverX.generate() path**. What
does grow with degree is the per-iteration cost (`gen_jacobian`, ~O(d^2.5)
empirically) and the memory (O(d²)). Capping `max_degree` is the only lever
that helps at large κ; see below.

CRITICAL: the 5–6 iteration claim does NOT extend to the capped path
(`_fit_capped_reduced_coefs`). PolyOneOverX.generate() returns a polynomial
that is analytically guaranteed to satisfy the QSP realizability conditions,
placing the coef/2 initial guess in the Newton basin of attraction. The capped
Chebyshev least-squares fit carries no such guarantee. At degree/κ < 11 (the
accuracy degradation threshold documented in session notes §3) the approximation
is too poor to produce a useful initial guess, and Newton diverges or oscillates
rather than converging. At degree=14999 each Newton iteration costs ~12–23 min
(empirically, from CX3 timing of N=16/32 at the same degree), so maxiter=100
non-converging iterations amounts to 20–38 h per epsilon, all before a single
cache file is written. This is the root cause of the N=64 order-4 precompute
hanging without output. The fix is stagnation-based early stopping in
`_newton_solve` (patience=5 for the capped path): exits in at most 6 non-
improving steps, matching the normal convergence iteration count and capping
the per-epsilon cost at ~2 h rather than ~38 h.

Degree capping
--------------
For κ large enough (roughly N ≥ 32 for the 1D Poisson TST matrix), the
*uncapped* polynomial degree makes both runtime (Newton iteration cost
~O(d^2.5) empirically) and memory (the Newton Jacobian working array is O(d²))
impractical — days of walltime and tens to hundreds of GB of RAM. If
`max_degree` is supplied, the target Chebyshev polynomial is truncated to that
degree before solving. This cap is applied to the polynomial actually solved;
in an earlier version it was computed as a warning and then silently ignored.

If `compute_inversion_angles` is called with `method='auto'` and no
`max_degree`, and the estimated degree exceeds `_DEGREE_SANITY_LIMIT`,
it raises rather than silently launching what could be a multi-day,
possibly OOM-inducing computation. Pass `max_degree` explicitly to
proceed anyway.

Caching
-------
Three levels: in-memory dict → on-disk .npz cache → compute. Reading from disk
is enabled by default, so a run automatically picks up anything precomputed
offline via `hpc/runners/precompute_phases.py`. Writing to disk is off by
default, so that ad hoc and interactive runs do not silently populate the cache
directory; the precompute script enables writing explicitly.

The cache key is built from a *canonical* method name, so that 'auto',
'sym_qsp_wrapper' and 'sym_qsp_direct' — which now all compute the identical
thing — cannot disagree about a key and miss each other. The key is
(round(κ,4), round(ε,8), method, max_degree): note that a κ differing in the
fourth decimal is a silent cache miss, which is why κ is always derived from the
same problem classes the solvers use rather than from a table.

References
----------
Dong, Y., Lin, L., Ni, H. & Wang, J. (2023). Robust iterative method for
    symmetric quantum signal processing in all parameter regimes.
    arXiv:2307.12468.
Martyn, J. M., Rossi, Z. M., Tan, A. K. & Chuang, I. L. (2021). Grand
    unification of quantum algorithms. PRX Quantum, 2, 040203.
Gilyén, A., Su, Y., Low, G. H. & Wiebe, N. (2019). Quantum singular value
    transformation. STOC 2019, pp. 193-204.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np

# ── Cache ─────────────────────────────────────────────────────────────────────

_PHASE_CACHE: dict[tuple, tuple[np.ndarray, int]] = {}

# Anchor to the repo root (this file lives at solvers/quantum/qsp_angles.py, so
# three parents up), NOT the process CWD. A relative path here silently
# produces a cache miss -- and a from-scratch recompute -- whenever the caller
# runs from anywhere other than the repo root.
_DISK_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "qsvt_phase_cache"

# Reading precomputed results is always on: a plain benchmark run should
# transparently pick up anything hpc/runners/precompute_phases.py has
# already computed, with zero code changes at the call site.
_ENABLE_DISK_READ: bool = True

# Writing is opt-in. Flip this (e.g. from precompute_qsvt_phases.py) to
# populate the cache; leave it off for interactive/exploratory runs so
# they don't silently write files.
_ENABLE_DISK_WRITE: bool = False

# Degree above which an *uncapped* solve is refused unless the caller
# explicitly supplies max_degree. Calibrated against measured runtimes:
# ~15,000 is roughly the N=16 scale (~a few hours, still tractable as a
# one-off precompute); N=32/64 (tens of thousands to ~250k) would be
# days-to-months and tens-to-hundreds of GB of RAM.
_DEGREE_SANITY_LIMIT = 15_000


# ── Public interface ──────────────────────────────────────────────────────────

def compute_inversion_angles(
    kappa      : float,
    epsilon    : float,
    method     : str           = "auto",
    max_degree : Optional[int] = None,
) -> tuple[np.ndarray, int]:
    """
    Compute QSP phase angles for p(x) ~= 1/(kappa*x) on [1/kappa, 1].

    Parameters
    ----------
    kappa : float
        Condition number. Determines the approximation domain [1/kappa, 1].
    epsilon : float
        Target approximation error.
    method : str
        'auto'            -- direct Newton solve (see module docstring for
                              why this deliberately does not warm-start).
        'sym_qsp_wrapper' / 'sym_qsp_direct'
                          -- accepted as aliases of 'auto' for backward
                             compatibility with earlier scripts/configs.
                             All three canonicalise to the SAME cache key.
        'precomputed'     -- disk cache only; raises if not found.
    max_degree : int or None
        If given, the target polynomial is truncated to this degree
        before solving (trades a small amount of approximation error
        for tractable runtime/memory at large kappa). If not given and
        the estimated degree exceeds _DEGREE_SANITY_LIMIT, raises.

    Returns
    -------
    angles : np.ndarray, shape (d+1,)
        Phase angles, negated for Qiskit's Rz sign convention.
    degree : int
        Actual polynomial degree solved for (== max_degree if capped).
    """
    if kappa < 1.0:
        raise ValueError(f"kappa must be >= 1, got {kappa:.4f}.")
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}.")

    canonical_method = _canonicalise_method(method)

    cache_key = (round(kappa, 4), round(epsilon, 8), canonical_method,
                 max_degree if max_degree is not None else -1)

    if cache_key in _PHASE_CACHE:
        return _PHASE_CACHE[cache_key]

    result = _load_disk(cache_key)
    if result is not None:
        _PHASE_CACHE[cache_key] = result
        return result

    if canonical_method == "precomputed":
        raise RuntimeError(
            f"method='precomputed' but no cached phases found for "
            f"kappa={kappa:.4f}, epsilon={epsilon:.4e}, max_degree={max_degree}. "
            f"Run hpc/runners/precompute_phases.py first."
        )

    if max_degree is None:
        est_degree = polynomial_degree_estimate(kappa, epsilon)
        if est_degree > _DEGREE_SANITY_LIMIT:
            raise RuntimeError(
                f"Estimated polynomial degree {est_degree} exceeds the "
                f"sanity limit ({_DEGREE_SANITY_LIMIT}) for kappa={kappa:.2f}, "
                f"epsilon={epsilon:.4e}. An uncapped solve at this degree is "
                f"expected to take from hours to months and may need tens to "
                f"hundreds of GB of RAM (the Newton Jacobian array is "
                f"O(degree^2)). Pass max_degree explicitly to proceed anyway, "
                f"or run hpc/runners/precompute_phases.py --max-degree ..."
            )

    result = _compute(kappa, epsilon, max_degree)
    _PHASE_CACHE[cache_key] = result
    _save_disk(cache_key, result)
    return result


def polynomial_degree_estimate(kappa: float, epsilon: float) -> int:
    """
    Rough guide only -- used for the warm-start/sanity-limit thresholds,
    NOT to control the actual polynomial degree pyqsp solves for (that is
    determined internally by PolyOneOverX.generate()). Calibrated against
    observed degrees for this problem's kappa range; expect O(1) factor
    error, not O(kappa) error.
    """
    d = int(np.ceil(13.0 * kappa * np.log(kappa / epsilon)))
    return d if d % 2 == 1 else d + 1


def evaluate_inversion_polynomial(angles: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate Im(<0|U_Phi(x)|0>) at the given points (verification helper)."""
    x   = np.atleast_1d(x)
    out = np.zeros(len(x))
    for k, xk in enumerate(x):
        if abs(xk) <= 1.0:
            U = _qsp_unitary(angles, float(xk))
            out[k] = float(np.imag(U[0, 0]))
    return out


# ── Private: method canonicalisation ──────────────────────────────────────────

def _canonicalise_method(method: str) -> str:
    """
    Map all equivalent method spellings to one cache-key string, so a
    precompute script and a live solver can never silently disagree
    about what was cached. This is the fix for the cache-miss bug where
    the precompute script's default ('sym_qsp_direct') differed from the
    solver configs' default ('auto') even though both computed the exact
    same thing.
    """
    if method in ("auto", "sym_qsp_direct", "sym_qsp_wrapper", "reduced_degree"):
        return "auto"
    if method == "precomputed":
        return "precomputed"
    raise ValueError(
        f"Unknown method '{method}'. Valid: 'auto', 'precomputed' "
        f"('sym_qsp_direct'/'sym_qsp_wrapper'/'reduced_degree' accepted "
        f"as aliases of 'auto')."
    )


# ── Private: computation ──────────────────────────────────────────────────────

def _fit_capped_reduced_coefs(
    kappa: float, epsilon: float, degree: int,
) -> np.ndarray:
    """
    Build a degree-`degree` odd-parity Chebyshev approximation to
    1/(kappa*x) directly, via a single linear least-squares solve on
    samples restricted to [1/kappa, 1] -- WITHOUT calling
    PolyOneOverX.generate(), whose internal cost is O(kappa^2 log(kappa/
    epsilon)) regardless of the degree eventually wanted (see module
    docstring / commit notes). This function's cost depends only on
    `degree`, not on kappa, which is the whole point: it's what makes
    max_degree actually bound the runtime for large kappa.

    This does not carry PolyOneOverX's analytic, provable boundedness
    guarantee -- boundedness is checked and enforced numerically on a
    dense grid afterward instead. Only used when a cap is requested,
    i.e. only when the caller has already accepted a precision/runtime
    tradeoff.
    """
    if degree % 2 == 0:
        degree += 1

    n_pts = max(2 * degree, 2000)
    k_idx = np.arange(1, n_pts + 1)
    x_eval = (
        0.5 * (1.0 / kappa + 1.0)
        + 0.5 * (1.0 - 1.0 / kappa) * np.cos(np.pi * (2 * k_idx - 1) / (2 * n_pts))
    )
    x_eval = np.sort(x_eval)

    target_raw = 1.0 / (kappa * x_eval)
    scale = 0.9 / float(np.max(target_raw))
    target = target_raw * scale

    # Fit ONLY the odd-order Chebyshev basis -- enforces odd parity
    # exactly (coefficients at even indices are exactly zero, not just
    # numerically small), rather than fitting all orders and hoping.
    odd_orders = np.arange(1, degree + 1, 2)
    full_vander = np.polynomial.chebyshev.chebvander(x_eval, degree)
    basis_odd = full_vander[:, odd_orders]

    coefs_odd, *_ = np.linalg.lstsq(basis_odd, target, rcond=None)

    coef_array = np.zeros(degree + 1)
    coef_array[odd_orders] = coefs_odd

    # Global boundedness check over the full domain (both branches by
    # odd symmetry, and through the [-1/kappa, 1/kappa] dead zone that
    # wasn't part of the fit).
    x_check = np.linspace(-1.0, 1.0, 2000)
    p_max = float(np.max(np.abs(
        np.polynomial.chebyshev.chebval(x_check, coef_array)
    )))
    if p_max > 0.9:
        coef_array *= 0.9 / p_max

    # Diagnostic: measure SHAPE error with the best global scale factor
    # projected out. QSVT recovers the proportionality constant downstream,
    # so a uniform rescale of the polynomial (which the boundedness step above
    # deliberately applies) is benign and must not be reported as fit error.
    # Measuring it naively reports ~27% for a fit that is actually exact.
    fit_vals = np.polynomial.chebyshev.chebval(x_eval, coef_array)
    denom = float(np.dot(fit_vals, fit_vals))
    if denom > 0.0:
        s_opt = float(np.dot(fit_vals, target)) / denom      # best global scale
        shape_err = np.max(np.abs(s_opt * fit_vals - target) / np.abs(target))
    else:
        shape_err = float("inf")

    if shape_err > 0.05:
        warnings.warn(
            f"Capped-degree fit (degree={degree}, kappa={kappa:.2f}) has "
            f"max shape-relative error {shape_err:.2%} on [1/kappa, 1] after "
            f"removing the global scale factor -- max_degree may be too small "
            f"for this kappa.",
            RuntimeWarning,
        )

    return coef_array


def _target_reduced_coefs(
    kappa: float, epsilon: float, cap: Optional[int],
) -> tuple[np.ndarray, int, int]:
    """
    Build the Chebyshev coefficients of the 1/(kappa*x) approximation,
    then reduce to the parity-folded coefficient array pyqsp's Newton
    solver expects.

    If `cap` is given, PolyOneOverX.generate() is bypassed entirely (see
    _fit_capped_reduced_coefs) -- its cost is O(kappa^2 log(kappa/
    epsilon)) and is paid in full BEFORE any post-hoc truncation could
    help, so truncating its output does not bound runtime. If `cap` is
    None, PolyOneOverX's exact/rigorous construction is used, as before.

    Returns (reduced_coefs, parity, degree).
    """
    if cap is not None:
        coef_array = _fit_capped_reduced_coefs(kappa, epsilon, cap)
    else:
        from pyqsp.poly import PolyOneOverX

        poly = PolyOneOverX()
        poly_coef_raw = poly.generate(
            kappa=kappa, epsilon=epsilon,
            return_coef=True, ensure_bounded=True, chebyshev_basis=True,
        )
        coef_array = np.asarray(
            poly_coef_raw.coef if hasattr(poly_coef_raw, "coef") else poly_coef_raw,
            dtype=float,
        )

    degree = len(coef_array) - 1
    is_even = np.max(np.abs(coef_array[0::2])) > 1e-8
    is_odd  = np.max(np.abs(coef_array[1::2])) > 1e-8
    if (is_even and is_odd) or not (is_even or is_odd):
        raise RuntimeError(
            f"Target polynomial does not have definite parity "
            f"(kappa={kappa}, epsilon={epsilon}, cap={cap})."
        )
    parity = 0 if is_even else 1
    reduced_coefs = coef_array[parity::2]
    return reduced_coefs, parity, degree


def _newton_solve(
    reduced_coefs       : np.ndarray,
    parity              : int,
    crit                : float = 1e-12,
    maxiter             : int   = 100,
    init_reduced_phases : Optional[np.ndarray] = None,
    stagnation_patience : int   = 8,
) -> tuple[np.ndarray, float, int]:
    """
    Reimplementation of pyqsp.sym_qsp_opt.newton_solver's loop, but with
    an overridable initial guess (pyqsp's own function hardcodes
    reduced_phases = coef/2 with no way to change it). Uses
    SymmetricQSPProtocol throughout, so full-phase reconstruction is
    always pyqsp's own (correct) implementation -- never hand-rolled.

    Parameters
    ----------
    reduced_coefs : np.ndarray
        Parity-folded Chebyshev coefficients of the target polynomial.
    parity : int
        Parity of the target polynomial (0 = even, 1 = odd).
    crit : float
        Convergence criterion on the L1 residual norm.
    maxiter : int
        Hard upper bound on Newton iterations.
    init_reduced_phases : np.ndarray or None
        Initial guess for the reduced phase sequence. Defaults to
        ``reduced_coefs / 2``, the same starting point used by pyqsp's
        own newton_solver and shown to converge in 5–6 iterations for
        polynomials produced by PolyOneOverX.generate().
    stagnation_patience : int
        Maximum number of consecutive iterations for which the L1
        residual does not strictly improve before the solver exits early.
        For quadratic Newton convergence (error reducing by ~100× per
        step) this threshold is never reached. For the capped-path case
        where the polynomial approximation is in the degradation regime
        (degree / κ < 11), the solve may diverge or oscillate; the
        patience cap prevents the worst-case 100-iteration catastrophe
        (~38 h at degree 14999) by exiting after a handful of
        non-improving steps. Set to ``maxiter`` to disable.

    Returns
    -------
    best_phases : np.ndarray
        Full phase angles corresponding to the iteration with the lowest
        residual seen during the solve. Returning the *best-seen* rather
        than the *final* phases means that a diverging Newton step cannot
        corrupt the result beyond what was achieved at the best iterate.
    best_err : float
        L1 residual at ``best_phases``.
    n_iter : int
        Total number of Newton iterations executed (including post-best
        iterations that triggered the stagnation exit).
    """
    from pyqsp.sym_qsp_opt import SymmetricQSPProtocol

    if init_reduced_phases is None:
        init_reduced_phases = reduced_coefs / 2

    qsp = SymmetricQSPProtocol(reduced_phases=init_reduced_phases, parity=parity)
    curr_iter = 0
    err = float("inf")

    # Track the best residual and corresponding phases seen across all
    # iterations. A diverging Newton step may leave qsp.full_phases in a
    # worse state than a prior iterate; returning the best-seen phases
    # ensures the caller always receives the most accurate result found.
    best_err    = float("inf")
    best_phases = np.asarray(qsp.full_phases, dtype=float).copy()
    stall_count = 0

    while True:
        Fval, DFval = qsp.gen_jacobian()
        res = Fval - reduced_coefs
        err = float(np.linalg.norm(res, ord=1))
        curr_iter += 1

        if err < best_err:
            best_err    = err
            best_phases = np.asarray(qsp.full_phases, dtype=float).copy()
            stall_count = 0
        else:
            stall_count += 1

        lin_sol = np.linalg.solve(DFval, res)
        qsp.update_reduced_phases(qsp.reduced_phases - lin_sol)

        if curr_iter >= maxiter or best_err < crit or stall_count >= stagnation_patience:
            break

    return best_phases, best_err, curr_iter


def _compute(
    kappa: float, epsilon: float, max_degree: Optional[int],
) -> tuple[np.ndarray, int]:
    """
    Solve for the QSP phases at the given (possibly capped) degree.

    The stagnation patience passed to the Newton solver is set tighter
    for the capped path (max_degree is not None) than for the uncapped
    path.  The uncapped path uses PolyOneOverX.generate(), whose output
    is analytically guaranteed to satisfy the QSP realizability
    conditions and always converges in 5–6 iterations from the
    coef/2 initial guess.  The capped path fits the polynomial via
    least-squares and carries no such guarantee; at degree/κ < 11 the
    fit quality is poor and the Newton solve may diverge or oscillate
    rather than converge, running all maxiter=100 iterations at
    ~12–23 min each (empirically, from degree-14999 timing on CX3).
    A patience of 5 exits in at most 5+1 non-improving steps -- the
    same count at which quadratic convergence has already reached
    machine precision -- capping the per-epsilon cost at ~2 h rather
    than ~38 h.
    """
    reduced_coefs, parity, degree = _target_reduced_coefs(kappa, epsilon, max_degree)

    # Tighter stagnation budget for the capped (approximate) path.
    stagnation_patience = 5 if max_degree is not None else 8
    full_phases, err, n_iter = _newton_solve(
        reduced_coefs, parity, stagnation_patience=stagnation_patience,
    )

    if err > 1e-8:
        warnings.warn(
            f"sym_qsp Newton solve finished with residual {err:.3e} "
            f"(kappa={kappa:.2f}, epsilon={epsilon:.4e}, degree={degree}) "
            f"after {n_iter} iterations -- did not fully converge.",
            RuntimeWarning,
        )

    angles = -np.asarray(full_phases, dtype=float)  # Qiskit Rz sign convention
    return angles, degree


# ── Private: disk cache ───────────────────────────────────────────────────────

def _cache_key_to_filename(key: tuple) -> Path:
    kappa, epsilon, method, max_deg = key
    tag = f"k{kappa}_e{epsilon}_{method}_d{max_deg}".replace(".", "p")
    return _DISK_CACHE_DIR / f"{tag}.npz"


def _load_disk(key: tuple):
    if not _ENABLE_DISK_READ:
        return None
    path = _cache_key_to_filename(key)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        return np.array(data["angles"]), int(data["degree"])
    except Exception:
        return None


def _save_disk(key: tuple, result: tuple[np.ndarray, int]) -> None:
    if not _ENABLE_DISK_WRITE:
        return
    _DISK_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_key_to_filename(key)
    angles, degree = result
    kappa, epsilon, method, max_deg = key
    np.savez_compressed(
        path, angles=angles, degree=np.array(degree),
        kappa=np.array(kappa), epsilon=np.array(epsilon),
        method=np.array(method), max_deg=np.array(max_deg),
    )


# ── Private: circuit-convention unitary (verification only) ───────────────────

def _qsp_unitary(angles: np.ndarray, x: float) -> np.ndarray:
    """2x2 QSP unitary matching Qiskit's Rz convention (verification helper)."""
    sx = np.sqrt(max(0.0, 1.0 - x**2))
    W  = np.array([[x, 1j*sx], [1j*sx, x]], dtype=complex)
    U  = np.diag([np.exp(-1j*angles[0]), np.exp(1j*angles[0])])
    for phi in angles[1:]:
        R = np.diag([np.exp(-1j*phi), np.exp(1j*phi)])
        U = R @ W @ U
    return U