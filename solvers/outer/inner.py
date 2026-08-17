"""
Inner (strip) solver registry.

Adapts the existing 1-D solvers in ``solvers/classical`` and
``solvers/quantum`` to the single ``(A, b) -> x`` signature that every outer
scheme expects, and records per-call diagnostics that the schemes themselves
have no business knowing about (VQLS cost, QSVT polynomial degree, HHL
proportionality constant).

Options
-------
Every solver declares its tunable parameters explicitly, with a type and a
one-line description.  Three properties follow, and all three matter:

*   Options are **validated**.  An unrecognised key raises, listing what is
    valid.  A registry that silently absorbs unknown keyword arguments is
    actively dangerous here: ``qsvt_max_degrees=500`` would be accepted,
    ignored, and the run would quietly cost ten times what was intended
    while appearing to honour the setting.

*   Options are **introspectable**.  ``describe("qsvt")`` prints the full set
    with defaults, so a runner script can expose them without hard-coding a
    list that drifts out of date.

*   An option left unset means *use the underlying solver's own default*,
    not a default invented here.  This module does not silently re-specify
    the behaviour of QSVTConfig1D or VQLSConfig1D.

Registering a new solver:

    @register("my_solver", options={
        "tol": Option(float, help="convergence tolerance"),
    })
    def _make(tol=None, **_):
        def solve(A, b):
            ...
            return x, {}          # (solution, diagnostics)
        return solve
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np


# -- Option declaration --------------------------------------------------------

@dataclass(frozen=True)
class Option:
    """
    One tunable parameter of an inner solver.

    default = None means "do not pass this through; let the underlying solver
    use its own default".  This keeps the wrapper honest: it cannot silently
    change the behaviour of a solver it merely adapts.
    """
    type:    type
    default: Any = None
    help:    str = ""
    choices: Optional[tuple] = None

    def coerce(self, value):
        if value is None:
            return None
        if self.type is bool and isinstance(value, str):
            if value.lower() in ("true", "1", "yes", "on"):
                value = True
            elif value.lower() in ("false", "0", "no", "off"):
                value = False
            else:
                raise ValueError(f"cannot interpret {value!r} as a boolean")
        try:
            value = self.type(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"expected {self.type.__name__}, got {value!r}") from exc
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"must be one of {self.choices}, got {value!r}")
        return value


_FACTORIES: dict[str, Callable] = {}
_OPTIONS: dict[str, dict[str, Option]] = {}


def register(name: str, options: dict[str, Option] | None = None):
    """Decorator registering a factory ``(**resolved_options) -> raw solver``."""
    def _wrap(factory):
        _FACTORIES[name] = factory
        _OPTIONS[name] = dict(options or {})
        return factory
    return _wrap


def available() -> list[str]:
    return sorted(_FACTORIES)


def available_options(name: str) -> dict[str, Option]:
    if name not in _OPTIONS:
        raise ValueError(f"Unknown inner solver {name!r}. "
                         f"Available: {', '.join(available())}")
    return dict(_OPTIONS[name])


def describe(name: str | None = None) -> str:
    """Human-readable table of every option, for --list-options in a runner."""
    names = available() if name is None else [name]
    out = []
    for n in names:
        opts = available_options(n)
        out.append(f"{n}:")
        if not opts:
            out.append("    (no tunable options)")
        for k, o in sorted(opts.items()):
            d = "solver default" if o.default is None else repr(o.default)
            ch = f" choices={list(o.choices)}" if o.choices else ""
            out.append(f"    {k:<16} {o.type.__name__:<6} default={d:<16}{ch}")
            if o.help:
                out.append(f"    {'':<16} {o.help}")
    return "\n".join(out)


def resolve_options(name: str, options: dict | None) -> dict:
    """
    Validate and type-coerce a user option dict against a solver's declared
    options.  Unknown keys raise; unset keys are omitted entirely so the
    underlying solver keeps its own defaults.
    """
    spec = available_options(name)
    resolved: dict[str, Any] = {}
    for k, o in spec.items():
        if o.default is not None:
            resolved[k] = o.default
    for k, v in (options or {}).items():
        if k not in spec:
            raise ValueError(
                f"Unknown option {k!r} for inner solver {name!r}. "
                f"Valid options: {', '.join(sorted(spec)) or '(none)'}")
        try:
            coerced = spec[k].coerce(v)
        except ValueError as exc:
            raise ValueError(f"Option {name}.{k}: {exc}") from exc
        if coerced is None:
            resolved.pop(k, None)
        else:
            resolved[k] = coerced
    return resolved


class InnerConfig(dict):
    """
    Per-solver options, for runners that sweep several solvers at once.

        cfg = InnerConfig(qsvt={"max_degree": 500},
                          hhl={"epsilon": 0.01},
                          vqls={"n_layers": 6, "n_restarts": 5})

        for name in ("hhl", "vqls", "qsvt"):
            solve(problem, inner=name, scheme="fmg", inner_options=cfg)

    ``solve`` recognises this type and hands each solver only its own
    section, so one configuration object can drive a whole sweep.
    """

    def for_solver(self, name: str) -> dict:
        return dict(self.get(name, {}))


# -- Wrapper: timing, diagnostics, failure fallback ----------------------------

class InnerSolverWrapper:
    """
    Wraps a raw ``(A, b) -> (x, extra)`` callable with timing, diagnostics
    collection and optional failure fallback.
    """

    def __init__(self, name, fn, fallback=None, collect=True, options=None):
        self.name = name
        self.options = dict(options or {})
        self._fn = fn
        self._fallback = fallback
        self._collect = collect
        self.calls = 0
        self.total_time = 0.0
        self.failures = 0
        self.records: list[dict] = []

    def __call__(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        # A strip whose right-hand side is numerically zero has the zero
        # solution; skipping it avoids a pointless (and for VQLS ill-posed)
        # quantum call.
        if np.linalg.norm(b) < 1e-14:
            return np.zeros(len(b))

        t0 = time.perf_counter()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x, extra = self._fn(A, b)
        except Exception as exc:
            self.failures += 1
            if self._fallback is None:
                raise
            x, _ = self._fallback(A, b)
            extra = {"error": str(exc), "fallback": True}
        dt = time.perf_counter() - t0

        self.calls += 1
        self.total_time += dt
        if self._collect:
            extra = dict(extra)
            extra["time_s"] = dt
            extra["n"] = len(b)
            self.records.append(extra)
        return x

    def summary(self) -> dict:
        d = {"inner_calls": self.calls,
             "inner_total_s": self.total_time,
             "inner_failures": self.failures,
             "inner_options": dict(self.options)}
        if not self.records:
            return d
        t = np.array([r["time_s"] for r in self.records])
        d["inner_mean_s"] = float(t.mean())
        d["inner_max_s"] = float(t.max())
        for key in ("final_cost", "polynomial_degree", "circuit_depth",
                    "prop_const", "n_circuit_evals"):
            vals = [r[key] for r in self.records if r.get(key) is not None]
            vals = [float(v) for v in vals if np.isfinite(float(v))]
            if vals:
                d[f"{key}_mean"] = float(np.mean(vals))
                d[f"{key}_max"] = float(np.max(vals))
        return d


# -- Built-in solvers ----------------------------------------------------------

@register("thomas")
def _thomas(**_):
    """Direct tridiagonal solve. The exact reference for every scheme."""
    def solve(A, b):
        # Fallback to general solver for pentadiagonal matrices
        if np.any(np.abs(np.triu(A, 2)) > 1e-12) or np.any(np.abs(np.tril(A, -2)) > 1e-12):
            return np.linalg.solve(A, b), {}
            
        n = len(b)
        m = A.diagonal(0).copy()
        up = A.diagonal(1).copy()
        lo = A.diagonal(-1).copy()
        d = b.copy()
        for i in range(1, n):
            w = lo[i - 1] / m[i - 1]
            m[i] -= w * up[i - 1]
            d[i] -= w * d[i - 1]
        x = np.zeros(n)
        x[-1] = d[-1] / m[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (d[i] - up[i] * x[i + 1]) / m[i]
        return x, {}
    return solve


@register("perturbed", options={
    "delta": Option(float, 0.0, "relative size of the systematic operator "
                                "perturbation; surrogate for quantum error"),
    "seed":  Option(int, 0, "perturbation seed"),
})
def _perturbed(delta=0.0, seed=0, **_):
    """
    Exact solve of a deterministically perturbed operator (A + E), with
    ||E|| = delta * ||A||.

    The surrogate used to characterise how much inner-solver error each outer
    scheme tolerates, without paying for quantum simulation.  It models a
    *systematic* approximation error - fixed Trotter truncation in HHL, fixed
    polynomial degree in QSVT - rather than shot noise, which is not the
    dominant error mode under statevector simulation.
    """
    cache: dict = {}

    def solve(A, b):
        key = (len(b), round(delta, 12))
        if key not in cache:
            if delta <= 0.0:
                cache[key] = np.linalg.inv(A)
            else:
                rng = np.random.default_rng(seed + len(b))
                E = rng.standard_normal((len(b), len(b)))
                E = 0.5 * (E + E.T)
                E *= delta * np.linalg.norm(A) / np.linalg.norm(E)
                cache[key] = np.linalg.inv(A + E)
        return cache[key] @ b, {}
    return solve


@register("hhl", options={
    "epsilon": Option(float, 0.01,
                      "overall algorithm precision. Apportioned by HHL as "
                      "eps/3 each to the reciprocal rotation and the state "
                      "preparation and eps/6 to the Hamiltonian simulation, "
                      "from which the Trotter step count follows. Dominant "
                      "cost driver: circuit depth grows as 1/epsilon."),
    "trotter_steps": Option(int,
                            help="Hamiltonian-simulation step count, fixed "
                                 "exactly and overriding the count epsilon "
                                 "implies. Unset = derived from epsilon. Valid "
                                 "only for a Toeplitz tridiagonal strip; a "
                                 "non-Toeplitz operator is simulated by exact "
                                 "matrix exponentiation and raises."),
})
def _hhl(epsilon=0.01, trotter_steps=None, **_):
    from solvers.quantum.hhl_1d import hhl_solve_system

    def solve(A, b):
        out = hhl_solve_system(A, b, epsilon, trotter_steps=trotter_steps)
        u = np.asarray(out[0], dtype=float)
        c = float(out[2]) if len(out) > 2 else float("nan")
        return u, {"prop_const": c}
    return solve


@register("vqls", options={
    "n_layers":    Option(int, help="ansatz entangling layers; parameter count "
                                    "is n_qubits*(n_layers+1)"),
    "n_restarts":  Option(int, help="random restarts; main defence against "
                                    "local minima, and a direct cost multiplier"),
    "max_iter":    Option(int, help="optimiser iterations per restart"),
    "tol":         Option(float, help="optimiser convergence tolerance"),
    "optimiser":   Option(str, help="classical optimiser, e.g. COBYLA"),
    "random_seed": Option(int, help="parameter initialisation seed"),
    "device_name": Option(str, help="PennyLane device"),
    "verbose":     Option(bool, help="per-solve optimiser logging"),
})
def _vqls(**opts):
    import dataclasses
    from solvers.quantum.vqls_1d import vqls_solve_system, VQLSConfig1D

    declared = {f.name for f in dataclasses.fields(VQLSConfig1D)}
    unknown = set(opts) - declared
    if unknown:
        raise ValueError(f"VQLSConfig1D has no field(s) {sorted(unknown)}; "
                         f"the option registry is out of date with vqls_1d.py")
    cfg = VQLSConfig1D(**opts) if opts else None

    def solve(A, b):
        res = (vqls_solve_system(A, b, config=cfg) if cfg is not None
               else vqls_solve_system(A, b))
        return (np.asarray(res.u, dtype=float),
                {"final_cost": float(getattr(res, "final_cost", np.nan)),
                 "n_circuit_evals": getattr(res, "n_circuit_evals", None)})
    return solve


@register("qsvt", options={
    "max_degree":     Option(int, help="cap on the QSP polynomial degree. The "
                                       "dominant cost driver: circuit depth is "
                                       "O(degree). Unset = uncapped."),
    "epsilon":        Option(float, help="target inversion accuracy; sets the "
                                         "required degree when uncapped"),
    "angle_method":   Option(str, help="QSP angle solver, e.g. sym_qsp_direct"),
    "max_degree_cap": Option(int, help="hard ceiling used during angle finding"),
    "device_name":    Option(str, help="simulator backend"),
    "verbose":        Option(bool, help="per-solve angle-finding logging"),
    "label":          Option(str, help="diagnostic label identifying this "
                                       "problem instance in the proportionality-"
                                       "recovery output, e.g. a case name. Purely "
                                       "cosmetic - does not affect the solution."),
})
def _qsvt(**opts):
    import dataclasses
    from solvers.quantum.qsvt_1d import qsvt_solve_system, QSVTConfig1D

    declared = {f.name for f in dataclasses.fields(QSVTConfig1D)}
    unknown = set(opts) - declared
    if unknown:
        raise ValueError(f"QSVTConfig1D has no field(s) {sorted(unknown)}; "
                         f"the option registry is out of date with qsvt_1d.py")
    # No options set -> use the module's own DEFAULT_QSVT_CONFIG untouched.
    cfg = QSVTConfig1D(**opts) if opts else None

    def solve(A, b):
        res = (qsvt_solve_system(A, b, config=cfg) if cfg is not None
               else qsvt_solve_system(A, b))
        return (np.asarray(res.u, dtype=float),
                {"polynomial_degree": getattr(res, "polynomial_degree", None),
                 "circuit_depth": getattr(res, "circuit_depth", None)})
    return solve


# -- Fourth-order (pentadiagonal) inner solvers --------------------------------
#
# Registered separately rather than folded into the 2nd-order factories above,
# because the two differ in the operator they may legally be given, not merely in
# a parameter. `hhl_solve_system` and `qsvt_solve_system` reconstruct their operator
# from A[0,0] and A[0,1] alone; handed the pentadiagonal strip matrix they discarded
# the ±2 band and solved a *tridiagonal* system instead. That produced errors of
# 52 % (N=4), 237 % (N=8) and 117 % (N=16) against the true pentadiagonal solution
# — and reported none of it, since the residual was computed against the truncated
# operator. Both now raise on a wider band rather than truncating.
#
# Thomas is absent by design: `solvers/classical/thomas.py` implements the
# tridiagonal algorithm specifically and cannot factor a pentadiagonal matrix. The
# 4th-order classical reference is the direct dense solve, reached through the
# ordinary "thomas" entry's own guard.

@register("hhl_4th", options={
    "epsilon": Option(float, 0.01,
                      "overall algorithm precision, apportioned as for the "
                      "2nd-order entry. Dominant cost driver: circuit depth "
                      "grows as 1/epsilon."),
    "trotter_steps": Option(int,
                            help="Hamiltonian-simulation step count, fixed "
                                 "exactly and overriding the count epsilon "
                                 "implies. Unset = derived from epsilon."),
})
def _hhl_4th(epsilon=0.01, trotter_steps=None, **_):
    from solvers.quantum.hhl_1d_4th import hhl_solve_system_4th

    def solve(A, b):
        res = hhl_solve_system_4th(A, b, epsilon=epsilon,
                                   trotter_steps=trotter_steps)
        return (np.asarray(res.u, dtype=float),
                {"prop_const": float(getattr(res, "prop_const", np.nan))})
    return solve


@register("qsvt_4th", options={
    "max_degree":     Option(int, help="cap on the QSP polynomial degree. The "
                                       "dominant cost driver: circuit depth is "
                                       "O(degree). Unset = uncapped."),
    "epsilon":        Option(float, help="target inversion accuracy; sets the "
                                         "required degree when uncapped"),
    "angle_method":   Option(str, help="QSP angle solver, e.g. sym_qsp_direct"),
    "max_degree_cap": Option(int, help="hard ceiling used during angle finding"),
    "device_name":    Option(str, help="simulator backend"),
    "verbose":        Option(bool, help="per-solve angle-finding logging"),
    "label":          Option(str, help="diagnostic label identifying this "
                                       "problem instance in the proportionality-"
                                       "recovery output, e.g. a case name. Purely "
                                       "cosmetic - does not affect the solution."),
})
def _qsvt_4th(**opts):
    import dataclasses
    from solvers.quantum.qsvt_1d import QSVTConfig1D
    from solvers.quantum.qsvt_1d_4th import (QSVTConfig1D4th,
                                             qsvt_solve_system_4th)

    declared = {f.name for f in dataclasses.fields(QSVTConfig1D)}
    unknown = set(opts) - declared
    if unknown:
        raise ValueError(f"QSVTConfig1D has no field(s) {sorted(unknown)}; "
                         f"the option registry is out of date with qsvt_1d.py")
    # Unset -> the 4th-order defaults, which are looser than the 2nd-order ones:
    # the pentadiagonal operator's condition number is 4/3 of the tridiagonal one
    # in the asymptotic limit, so the same epsilon costs a proportionally higher
    # polynomial degree.
    cfg = QSVTConfig1D(**opts) if opts else QSVTConfig1D4th().to_qsvt_config()

    def solve(A, b):
        res = qsvt_solve_system_4th(A, b, config=cfg)
        return (np.asarray(res.u, dtype=float),
                {"polynomial_degree": getattr(res, "polynomial_degree", None),
                 "circuit_depth": getattr(res, "circuit_depth", None)})
    return solve


# -- Factory -------------------------------------------------------------------

def get_inner(name: str, fallback_to_thomas: bool = True,
              **options) -> InnerSolverWrapper:
    """
    Build an inner solver by name.

    Parameters
    ----------
    name : one of ``available()``.
    fallback_to_thomas : if True, a strip solve that raises is retried with
        the direct solver rather than aborting the run.  Counted in
        ``.failures`` so the substitution is never silent.
    **options : validated against ``available_options(name)``.  Unknown keys
        raise rather than being ignored.
    """
    if name not in _FACTORIES:
        raise ValueError(f"Unknown inner solver {name!r}. "
                         f"Available: {', '.join(available())}")
    resolved = resolve_options(name, options)
    fn = _FACTORIES[name](**resolved)
    fb = _FACTORIES["thomas"]() if (fallback_to_thomas and name != "thomas") else None
    return InnerSolverWrapper(name, fn, fallback=fb, options=resolved)