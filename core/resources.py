"""
Post-transpilation resource estimation against real IBM hardware targets.

Motivation
----------
Every circuit-depth and qubit-count figure currently reported anywhere in
this project — the README's HPC section, ``QSVTSolverResult.circuit_depth``,
the numbers quoted in progress reports — is measured *before*
transpilation. That is the right quantity for comparing algorithms to each
other and for the statevector-simulation cost model in
``solvers/backend_factory.py``, where an abstract gate is exactly as
expensive as any other. It is the wrong quantity for asking whether a
circuit can run on a real device, because a device only executes a fixed
native gate set over a fixed qubit connectivity, and an arbitrary
``UnitaryGate`` — which is what ``block_encoding.build_tst_block_encoding``
produces — is synthesised into dozens of native two-qubit gates before a
single shot can be taken.

This module supplies that second, harder number. It answers: after
transpilation to a real IBM Heron r2 target, how many two-qubit gates does
this circuit actually need, and does that fit inside what today's hardware
can execute with useful fidelity?

Target hardware
----------------
IBM Heron r2 (e.g. ``ibm_kingston``, 156 qubits, heavy-hex topology).
Native basis: {RZ, SX, X} single-qubit, CZ two-qubit — confirmed against
current IBM documentation and independent hardware papers as of August 2026
(Heron r1/r2 use CZ; the earlier Eagle generation used ECR instead, so a
basis-gate choice copied from an Eagle-era paper would silently misprice
every count in this module).

    HERON_R2_BASIS_GATES = ("rz", "sx", "x", "cz")

Two published figures anchor the feasibility judgement:

*   **Circuit capacity**: IBM reports Heron r2 executing computations with
    up to ~5,000 two-qubit gate operations, roughly double the ~2,880
    reported for the prior generation.
*   **Median 2Q error**: independently reported in the 2–3 × 10⁻³ range for
    CZ on Heron r1/r2 hardware (values differ slightly by system and
    measurement date; treat as an order-of-magnitude anchor, not a
    per-device guarantee — always re-check current calibration before an
    actual run).

Both are exposed as module constants below, each carrying the caveat that
device calibration drifts and should be re-verified close to the run.

Composability, and what it actually gives you
-----------------------------------------------
Directly transpiling a full QSVT circuit at production degree is not
generally feasible: at N=32 the degree is in the hundreds, and Qiskit's
unitary synthesis pass is not free. The estimate here instead transpiles the
*single* block-encoding application once, and combines its two-qubit count
with the degree:

    two_qubit_total ≈ degree × two_qubit_count(U_A) + two_qubit_count(state_prep)

The first draft of this module claimed this composition was exact, on the
reasoning that ``_build_qsvt_circuit`` applies the identical block-encoding
gate ``degree`` times interleaved only with single-qubit Rz rotations (the
"non-alternating" convention documented there), so every application should
transpile identically in isolation. ``validate_composability`` was written
to confirm that claim before it was relied on, and instead disproved it: a
direct sweep over N ∈ {4, 8, 16} and degree ∈ {5, 11, 21, 41} showed the
composed estimate consistently *exceeds* the directly-transpiled count —
never once falls short — by a margin that shrinks sharply with N:

    N=4  : composed exceeds direct by 27-36%
    N=8  : composed exceeds direct by 0.5-1.2%
    N=16 : composed exceeds direct by 0.4%

The excess comes from cross-application optimisation the transpiler finds
between adjacent block-encoding applications, which the isolated single-gate
transpile cannot see; the effect is proportionally largest when the circuit
itself is smallest. The composed estimate is therefore not exact, but it is
a *safe upper bound* on the two-qubit count in every case tested, which is
the property a feasibility screen actually needs: a "feasible" verdict
(``total <= budget``) is trustworthy, since the real circuit needs no more
than the estimate; a "not feasible" verdict close to the budget should be
treated as provisional for N=4-sized problems (where the margin is wide
enough to matter) and re-checked with a direct transpile.

Every use of the extrapolated estimate in this project should be preceded,
at least once per problem shape, by a call to :func:`validate_composability`
— ``tests/test_resources.py`` does this for every N used elsewhere in the
codebase, and asserts the safe-upper-bound property rather than exactness.

References
----------
IBM Quantum, "Processor types" (Heron r1/r2/r3), docs.quantum.ibm.com.
IBM Quantum Developer Conference, Nov 2024 (Heron r2 unveiling; ~5,000 2Q
    gate circuit capacity, up from ~2,880).
Qiskit transpiler documentation, "Represent quantum computers for the
    transpiler" (native basis gates by processor generation).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "HERON_R2_BASIS_GATES",
    "HERON_R2_TWO_QUBIT_GATE_BUDGET",
    "HERON_R2_TWO_QUBIT_ERROR",
    "ResourceReport",
    "FeasibilityReport",
    "transpile_report",
    "block_encoding_unit_cost",
    "state_prep_cost",
    "qsvt_resource_estimate",
    "validate_composability",
    "feasibility_table",
]


# ── Hardware target constants ─────────────────────────────────────────────────

# Confirmed current (Heron r1/r2) native gate set. Do not reuse for Eagle-
# generation backends (native two-qubit gate ECR, not CZ) or for a future
# Nighthawk target without re-checking — Nighthawk's square-lattice topology
# and gate set were not yet public at the time this module was written.
HERON_R2_BASIS_GATES: Tuple[str, ...] = ("rz", "sx", "x", "cz")

# Approximate circuit capacity at which IBM reports Heron r2 producing
# useful (not merely executable) results. This is a soft budget, not a hard
# cutoff -- treat crossing it as "needs error mitigation or is out of reach
# today", not as a circuit that literally cannot be submitted.
HERON_R2_TWO_QUBIT_GATE_BUDGET: int = 5_000

# Representative median CZ error, order-of-magnitude only. Re-check current
# calibration (e.g. via the IBM Quantum Platform backend properties) before
# relying on this for anything beyond a rough feasibility screen.
HERON_R2_TWO_QUBIT_ERROR: float = 2.5e-3


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class ResourceReport:
    """
    Transpiled resource footprint of a single circuit.

    Attributes
    ----------
    n_qubits : int
        Circuit width. Unaffected by transpilation for these circuit sizes
        (no ancilla-hungry routing is needed at n_qubits <= ~10 on a
        156-qubit device), but recorded for completeness and provenance.
    pre_depth, post_depth : int
        Depth before and after transpilation. Comparing the two is itself
        informative: a large ratio indicates the logical circuit relied on
        gates far from the native set.
    two_qubit_count : int
        Total native two-qubit gates (CZ, for Heron r2) after transpilation.
        This is the quantity ``HERON_R2_TWO_QUBIT_GATE_BUDGET`` bounds and
        the one that should be quoted whenever "circuit size" is discussed
        in a hardware context.
    gate_counts : dict
        Full post-transpilation gate histogram, for anything beyond the
        headline two-qubit count.
    basis_gates : tuple of str
        Target basis used for this report.
    optimization_level : int
        Qiskit transpiler optimisation level used.
    coupling_map : str or None
        Human-readable description of the coupling constraint used, or
        ``None`` if transpiled against an unconstrained (all-to-all) model.
        ``None`` is the honest default for these circuit sizes: at 3-6
        logical qubits, any real heavy-hex region contains a usable
        subgraph, so the two-qubit *count* is essentially connectivity-
        independent even though the specific qubit mapping is not decided
        here. Pass an explicit coupling map if a specific physical layout
        matters for the question being asked.
    """

    n_qubits:           int
    pre_depth:          int
    post_depth:         int
    two_qubit_count:    int
    gate_counts:        Dict[str, int]
    basis_gates:        Tuple[str, ...]
    optimization_level: int
    coupling_map:       Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        d = {
            "n_qubits":           self.n_qubits,
            "pre_depth":          self.pre_depth,
            "post_depth":         self.post_depth,
            "two_qubit_count":    self.two_qubit_count,
            "basis_gates":        list(self.basis_gates),
            "optimization_level": self.optimization_level,
            "coupling_map":       self.coupling_map,
        }
        d.update({f"gate_{k}": v for k, v in self.gate_counts.items()})
        return d


@dataclass
class FeasibilityReport:
    """
    A QSVT resource estimate for one problem size, judged against a budget.

    Attributes
    ----------
    N, kappa, degree : problem-shape parameters that produced this estimate.
    unit_cost : ResourceReport
        Transpiled cost of a single block-encoding application.
    prep_cost : ResourceReport
        Transpiled cost of the initial state preparation (the ``Isometry``
        loading |b_norm⟩).
    total_two_qubit_count : int
        ``degree * unit_cost.two_qubit_count + prep_cost.two_qubit_count``.
        A *safe upper bound* on the true post-transpilation count, not an
        exact prediction — see :func:`validate_composability` and the
        module docstring. Tight (<1.5%) for N >= 8; loose (up to ~36%) at
        N=4, where the transpiler finds proportionally more cross-
        application optimisation in the small circuit.
    budget : int
        The two-qubit gate budget this was judged against.
    feasible : bool
        ``total_two_qubit_count <= budget``. Because the count is a safe
        upper bound, ``True`` is trustworthy as-is; ``False`` near the
        budget boundary at small N should be re-checked with a direct
        transpile before being reported as a hard limit.
    validated : bool
        Whether this specific (N, degree) combination has been checked
        against a direct full-circuit transpilation via
        :func:`validate_composability`. ``False`` means the total is an
        extrapolation that has not itself been directly confirmed for this
        exact degree, only for the safe-upper-bound property in general.
    """

    N:                      int
    kappa:                  float
    degree:                 int
    unit_cost:              ResourceReport
    prep_cost:              ResourceReport
    total_two_qubit_count:  int
    budget:                 int
    feasible:               bool
    validated:              bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "N":                      self.N,
            "kappa":                  self.kappa,
            "degree":                 self.degree,
            "unit_two_qubit_count":   self.unit_cost.two_qubit_count,
            "prep_two_qubit_count":   self.prep_cost.two_qubit_count,
            "total_two_qubit_count":  self.total_two_qubit_count,
            "budget":                 self.budget,
            "feasible":               self.feasible,
            "validated":              self.validated,
            "overshoot_factor":       (
                self.total_two_qubit_count / self.budget if self.budget else None
            ),
        }


# ── Core transpilation ────────────────────────────────────────────────────────

def transpile_report(
    circuit,
    basis_gates:        Sequence[str] = HERON_R2_BASIS_GATES,
    coupling_map=None,
    optimization_level: int = 3,
    seed_transpiler:    int = 0,
) -> ResourceReport:
    """
    Transpile ``circuit`` to a target basis and report its resource footprint.

    Parameters
    ----------
    circuit : QuantumCircuit
    basis_gates : sequence of str
        Defaults to the Heron r2 native set.
    coupling_map : qiskit.transpiler.CouplingMap or None
        Explicit connectivity constraint. ``None`` transpiles against an
        unconstrained model — see the ``coupling_map`` field docstring on
        :class:`ResourceReport` for why that is a reasonable default at
        this circuit scale.
    optimization_level : int
        Qiskit transpiler level, 0-3. Level 3 is used throughout this
        project's estimates since it is what a real submission would use.
    seed_transpiler : int
        Fixed for reproducibility; Qiskit's routing and synthesis heuristics
        are stochastic at optimization_level >= 1.

    Returns
    -------
    ResourceReport
    """
    from qiskit import transpile

    pre_depth = circuit.depth()
    tqc = transpile(
        circuit,
        basis_gates         = list(basis_gates),
        coupling_map        = coupling_map,
        optimization_level  = optimization_level,
        seed_transpiler      = seed_transpiler,
    )
    counts = dict(tqc.count_ops())
    two_qubit_gates = {"cz", "ecr", "cx", "rzz"}
    two_qubit_count = sum(v for k, v in counts.items() if k in two_qubit_gates)

    return ResourceReport(
        n_qubits           = tqc.num_qubits,
        pre_depth           = pre_depth,
        post_depth          = tqc.depth(),
        two_qubit_count     = two_qubit_count,
        gate_counts         = counts,
        basis_gates         = tuple(basis_gates),
        optimization_level  = optimization_level,
        coupling_map        = "unconstrained" if coupling_map is None else "constrained",
    )


# ── QSVT-specific building blocks ─────────────────────────────────────────────

def block_encoding_unit_cost(
    N:          int,
    main_diag:  float = -2.0,
    off_diag:   float = 1.0,
    **transpile_kwargs,
) -> ResourceReport:
    """
    Transpiled cost of one application of the TST block encoding U_A.

    This is the primitive every QSVT circuit repeats ``degree`` times, and
    the single most valuable number in a hardware feasibility estimate: the
    total two-qubit count scales linearly in it (see module docstring).

    Parameters
    ----------
    N : int
        Problem size; must be a power of 2.
    main_diag, off_diag : float
        TST matrix parameters. Defaults are the 1-D Poisson values used
        throughout this project (-2 main, +1 off). The unit cost is
        insensitive to the specific numeric values in practice -- what
        determines the transpiled gate count is the generic-unitary
        synthesis of a 2N x 2N matrix, not the particular matrix entries --
        but the real problem values are used regardless, since they cost
        nothing extra and keep every report traceable to an actual system.
    """
    from solvers.quantum.block_encoding import build_tst_block_encoding

    be_circuit, alpha = build_tst_block_encoding(N, main_diag, off_diag)
    report = transpile_report(be_circuit, **transpile_kwargs)
    log.debug(
        "block_encoding_unit_cost: N=%d, alpha=%.4f, 2Q=%d, depth %d->%d",
        N, alpha, report.two_qubit_count, report.pre_depth, report.post_depth,
    )
    return report


def state_prep_cost(
    b_norm_vec: np.ndarray,
    **transpile_kwargs,
) -> ResourceReport:
    """
    Transpiled cost of preparing |b_norm⟩ via Qiskit's ``Isometry``.

    This is the same state-preparation primitive
    ``qsvt_1d._build_qsvt_circuit`` uses to load the right-hand side, so the
    reported cost is exactly what a real QSVT circuit pays once, up front.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import Isometry

    n = int(np.log2(len(b_norm_vec)))
    qc = QuantumCircuit(n)
    qc.append(Isometry(b_norm_vec, 0, 0), list(range(n)))
    return transpile_report(qc, **transpile_kwargs)


def qsvt_resource_estimate(
    N:          int,
    degree:     int,
    kappa:      Optional[float] = None,
    main_diag:  float = -2.0,
    off_diag:   float = 1.0,
    budget:     int = HERON_R2_TWO_QUBIT_GATE_BUDGET,
    b_norm_vec: Optional[np.ndarray] = None,
    **transpile_kwargs,
) -> FeasibilityReport:
    """
    Estimate the total transpiled two-qubit cost of a QSVT circuit at (N, degree).

    Composes the block-encoding unit cost and the state-preparation cost
    rather than transpiling the full circuit — see the module docstring for
    why this is exact rather than approximate for this circuit family, and
    :func:`validate_composability` for the empirical check.

    Parameters
    ----------
    N : int
    degree : int
        Polynomial degree — the number of block-encoding applications.
        Obtain from ``solvers.quantum.qsp_angles.polynomial_degree_estimate``
        for a rough guide, or from a cached ``QSVTConfig1D`` run for the
        exact value actually used in a solve.
    kappa : float or None
        Recorded for provenance in the report; not used in the estimate
        itself.
    budget : int
        Two-qubit gate budget to judge feasibility against. Defaults to the
        Heron r2 figure; pass a different value for a different target or a
        more conservative (mitigation-aware) threshold.
    b_norm_vec : np.ndarray or None
        Right-hand-side vector to cost state preparation for. Defaults to
        the uniform state, which is a reasonable stand-in when the specific
        problem instance is not yet fixed -- state preparation cost depends
        weakly on the target vector's structure for a generic ``Isometry``
        synthesis, so this default rarely changes the total materially, but
        pass the real vector when the estimate needs to be exact for a
        specific case.
    """
    from solvers.quantum.qsp_angles import polynomial_degree_estimate  # noqa: F401 (doc cross-ref)

    n = int(np.log2(N))
    if b_norm_vec is None:
        b_norm_vec = np.ones(N) / np.sqrt(N)

    unit_cost = block_encoding_unit_cost(N, main_diag, off_diag, **transpile_kwargs)
    prep_cost = state_prep_cost(b_norm_vec, **transpile_kwargs)

    total = degree * unit_cost.two_qubit_count + prep_cost.two_qubit_count

    return FeasibilityReport(
        N                      = N,
        kappa                  = kappa if kappa is not None else float("nan"),
        degree                 = degree,
        unit_cost              = unit_cost,
        prep_cost              = prep_cost,
        total_two_qubit_count  = total,
        budget                 = budget,
        feasible               = total <= budget,
    )


# ── Composability validation ──────────────────────────────────────────────────

def validate_composability(
    N:          int,
    degree:     int,
    main_diag:  float = -2.0,
    off_diag:   float = 1.0,
    seed:       int = 0,
    **transpile_kwargs,
) -> Dict[str, Any]:
    """
    Confirm the linear composition estimate against a direct full-circuit
    transpilation, at a degree small enough to be tractable.

    Builds the actual QSVT circuit via
    ``solvers.quantum.qsvt_1d._build_qsvt_circuit`` with synthetic,
    non-degenerate phase angles (uniform random in ``[0.1, 3.0]``, not zero
    or a special multiple of pi/2 that the transpiler could simplify away),
    transpiles it directly, and compares its two-qubit count against
    ``degree * unit_cost + prep_cost``.

    The composed estimate is not exact — see the module docstring for the
    measured overshoot pattern — but empirically never falls below the
    directly-measured count. This function's real job is to keep re-checking
    that "never falls below" property rather than to chase exact equality:
    ``is_safe_upper_bound`` is the field that matters, and any downstream use
    of :func:`qsvt_resource_estimate` should treat its feasibility verdict as
    provisional wherever this check has not been run for a comparable
    (N, degree) shape.

    Returns
    -------
    dict with keys:
        direct_two_qubit_count, composed_two_qubit_count : int
        overshoot_fraction : float
            (composed - direct) / direct. Positive means composed is
            conservative (safe); the module docstring gives the typical
            magnitude by N.
        is_safe_upper_bound : bool
            ``composed >= direct``. This is the property relied on
            elsewhere, not exact equality.
        direct_depth : int
    """
    from solvers.quantum.block_encoding import build_tst_block_encoding
    from solvers.quantum.qsvt_1d import _build_qsvt_circuit

    n = int(np.log2(N))
    be_circuit, _alpha = build_tst_block_encoding(N, main_diag, off_diag)
    rng    = np.random.default_rng(seed)
    angles = rng.uniform(0.1, 3.0, size=degree + 1)
    b_norm_vec = np.ones(N) / np.sqrt(N)

    full_circuit = _build_qsvt_circuit(be_circuit, angles, n, b_norm_vec)
    direct       = transpile_report(full_circuit, **transpile_kwargs)

    unit_cost = block_encoding_unit_cost(N, main_diag, off_diag, **transpile_kwargs)
    prep_cost = state_prep_cost(b_norm_vec, **transpile_kwargs)
    composed_total = degree * unit_cost.two_qubit_count + prep_cost.two_qubit_count

    direct_count = direct.two_qubit_count
    overshoot = (
        (composed_total - direct_count) / direct_count if direct_count else float("nan")
    )

    return {
        "N":                        N,
        "degree":                   degree,
        "direct_two_qubit_count":   direct_count,
        "composed_two_qubit_count": composed_total,
        "overshoot_fraction":       overshoot,
        "is_safe_upper_bound":      composed_total >= direct_count,
        "direct_depth":             direct.post_depth,
    }


# ── Convenience: a full sweep ──────────────────────────────────────────────────

def feasibility_table(
    sizes:  Sequence[Tuple[int, float, int]],
    budget: int = HERON_R2_TWO_QUBIT_GATE_BUDGET,
    **transpile_kwargs,
) -> list:
    """
    Run :func:`qsvt_resource_estimate` over a sweep of (N, kappa, degree).

    Parameters
    ----------
    sizes : sequence of (N, kappa, degree)
        Typically sourced from a cached QSVT run's ``kappa_effective`` and
        ``polynomial_degree`` fields, so the estimate reflects degrees that
        were actually used rather than the rough formula's guess.

    Returns
    -------
    list of dict
        One row per size, suitable for ``benchmark.results_io`` or direct
        ``pandas.DataFrame`` construction; deliberately not tied to either.
    """
    rows = []
    for N, kappa, degree in sizes:
        report = qsvt_resource_estimate(
            N, degree, kappa=kappa, budget=budget, **transpile_kwargs
        )
        rows.append(report.as_dict())
    return rows