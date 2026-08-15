"""
Execution against real IBM Quantum hardware, via Qiskit Runtime.

Scope, decided deliberately
----------------------------
This module does not attempt to reconstruct a solution vector from real
hardware. ``core.noise.NoiseExecutor`` (Phase 3) already established why:
amplitude reconstruction needs a phase reference, which a real device does
not have. What a real device *can* do without qualification is answer two
kinds of question, and this module is built around exactly those two:

1. **Post-selection statistics** — genuine measurement counts, no
   reconstruction needed. ``hardware_postselection_sample`` is the
   real-hardware counterpart of ``core.noise.sample_postselection``, same
   interface, same ``PostSelectionSample`` return type, just backed by
   ``SamplerV2`` instead of Aer.

2. **Observable expectation values** — ``hardware_estimate`` for a single
   Pauli-sum observable, and ``hardware_estimate_batch`` for many at once.
   The batching matters: a loop of single-PUB calls was measured directly
   against a local-testing backend and found impractically slow (a 40-circuit
   loop did not finish in the time a single 40-PUB batched job took ~22s to
   complete). Any caller needing more than a handful of expectation values —
   which includes essentially every genuinely interesting experiment this
   project would run — must batch through ``hardware_estimate_batch``, not
   call ``hardware_estimate`` in a loop.

``hardware_fidelity_estimate`` builds on (2): fidelity against a known
target state, decomposed into Pauli terms via
``vqls_utils.pauli_decompose_matrix`` (reused directly, not reimplemented)
and measured as a weighted sum of Pauli expectation values — the standard
Direct Fidelity Estimation identity, confirmed here against a classical
statevector calculation on two independent random states before being
trusted (see this module's test file). This is the tool for the
block-encoding fidelity experiment described in the original hardware-
scoping discussion: prepare |b_norm>, apply the block encoding once, and
measure how close the result is to the classically-known target M|b>/alpha.

What is explicitly out of scope here
---------------------------------------
VQLS's Hadamard-test cost function (``solvers.quantum.vqls_hadamard``) is
*not* wired to real hardware in this module. Its circuits are built in
PennyLane and were validated there; routing them through PennyLane's
``qiskit.remote`` device in a loop — one PennyLane QNode call per required
circuit — hit exactly the per-call overhead problem described above, since
VQLS's cost function needs `circuit_count(L)` = 2L + 2L^2 circuits per
single cost evaluation. Making that practical needs the Hadamard-test
circuits rebuilt natively in Qiskit so they can be submitted as one batched
job via ``hardware_estimate_batch``, which is a distinct piece of work with
its own wire-convention risks (Qiskit and PennyLane order qubits
oppositely — see ``core.execution``'s and ``vqls_hadamard``'s module
docstrings for how much that has mattered every time it has come up) and is
better done, and validated, as its own follow-up rather than rushed in here.

Credentials and setup
------------------------
This module never touches IBM Quantum credentials. Before any real
(non-Fake) backend can be used, run once, outside of this project, in your
own Python session:

    from qiskit_ibm_runtime import QiskitRuntimeService
    QiskitRuntimeService.save_account(
        channel="ibm_quantum", token="<your Premium-plan API token>"
    )

After that, ``QiskitRuntimeService()`` with no arguments loads the saved
account automatically, which is what ``HardwareContext.real`` does below.

Validation
--------------
Everything in this module except the actual network call to IBM's servers
has been tested here against ``FakeTorino`` (Heron r1, the newest
generation reachable under this project's qiskit==1.4.5 pin — see
``core.noise.fake_backend_noise_model`` for the version-compatibility
finding this repeats) in Qiskit Runtime's local-testing mode: same
``SamplerV2``/``EstimatorV2`` primitives, same options surface, same PUB
format, pointed at a local simulator carrying real calibration data instead
of a live queue. That is the strongest test possible without a live
service connection, and it is what every function here was actually run
against before being included.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.execution import PostSelectSpec

log = logging.getLogger(__name__)

__all__ = [
    "HardwareContext",
    "JobProvenance",
    "PostSelectionSample",
    "hardware_postselection_sample",
    "hardware_estimate",
    "hardware_estimate_batch",
    "hardware_fidelity_estimate",
]


# -- Context: backend, mode, mitigation, all in one place ----------------------

@dataclass
class HardwareContext:
    """
    Everything needed to run a circuit somewhere: which backend, in what
    mode, with what error mitigation. One object threaded through every
    function in this module, so a script can switch from safe local testing
    to a real queued job by changing exactly one line.

    Parameters
    ----------
    backend : BackendV1/V2 or None
        The target. A ``FakeTorino()``-style object for local testing (no
        credentials, no queue, no cost — the default this module was
        validated against), or a real backend obtained from
        ``HardwareContext.real(...)`` for an actual hardware run.
    resilience_level : int
        0 = none, 1 = measurement error mitigation (TREX), 2 = adds
        zero-noise extrapolation, 3 = adds probabilistic error cancellation.
        Passed straight through to ``EstimatorOptions.resilience_level``.
        Not used by the Sampler path (post-selection counting has no
        equivalent notion of resilience level in this API).
    dynamical_decoupling : bool
        Idle-qubit DD sequences, cheap and broadly useful; on by default.
    default_shots : int
        Shots per circuit when not overridden per-call.
    seed : int or None
        Simulator seed. Only meaningful for local-testing (Fake/Aer)
        backends; a real backend ignores it.

    Class methods
    -------------
    ``HardwareContext.local_testing(backend)`` — the safe default, no
    credentials needed, what every example in this module's test file uses.

    ``HardwareContext.real(backend_name=None, resilience_level=1)`` — loads
    the saved account via ``QiskitRuntimeService()`` (must already exist —
    see module docstring) and either picks the named backend or the least-
    busy one available on the account's instance. This is the only place in
    this module that touches ``QiskitRuntimeService`` at all.
    """

    backend:               Any
    resilience_level:      int  = 1
    dynamical_decoupling:  bool = True
    default_shots:         int  = 4096
    seed:                   Optional[int] = None
    _service:               Any = field(default=None, repr=False)

    @property
    def is_local_testing(self) -> bool:
        """True if this context targets a Fake/local backend, not a live queue."""
        return type(self.backend).__module__.startswith("qiskit_ibm_runtime.fake_provider")

    @classmethod
    def local_testing(cls, backend=None, **kwargs) -> "HardwareContext":
        """
        Build a context against a Fake backend — no credentials, no queue,
        no cost. Defaults to ``FakeTorino`` (see module docstring on the
        qiskit==1.4.5 version constraint this repeats from
        ``core.noise.fake_backend_noise_model``).
        """
        if backend is None:
            from qiskit_ibm_runtime.fake_provider import FakeTorino
            backend = FakeTorino()
        return cls(backend=backend, **kwargs)

    @classmethod
    def real(
        cls,
        backend_name: Optional[str] = None,
        resilience_level: int = 1,
        min_num_qubits: Optional[int] = None,
        **kwargs,
    ) -> "HardwareContext":
        """
        Build a context against a real backend via a saved IBM Quantum
        account. Requires ``QiskitRuntimeService.save_account(...)`` to have
        already been run once — see module docstring.

        Parameters
        ----------
        backend_name : str or None
            A specific backend (e.g. ``'ibm_kingston'``). If ``None``, picks
            the least-busy backend meeting ``min_num_qubits`` on the
            account's default instance.
        min_num_qubits : int or None
            Only used when ``backend_name`` is ``None``.
        """
        from qiskit_ibm_runtime import QiskitRuntimeService

        service = QiskitRuntimeService()
        if backend_name is not None:
            backend = service.backend(backend_name)
        else:
            backend = service.least_busy(min_num_qubits=min_num_qubits)

        log.info("HardwareContext.real: using backend %s", backend.name)
        ctx = cls(backend=backend, resilience_level=resilience_level, **kwargs)
        ctx._service = service
        return ctx

    # -- Primitive factories --------------------------------------------------

    def sampler(self):
        from qiskit_ibm_runtime import SamplerV2
        s = SamplerV2(mode=self.backend)
        # resilience_level/dynamical_decoupling genuinely have no effect in
        # local-testing mode -- the Runtime SDK warns on every job if they
        # are set anyway (observed: over a million warnings across a modest
        # test suite, since the warning fires per-PUB, not once). Skipping
        # them here is not an optimisation, it is the correct behaviour: a
        # Fake backend has no dynamical-decoupling hardware to configure.
        if not self.is_local_testing:
            s.options.dynamical_decoupling.enable = self.dynamical_decoupling
        if self.seed is not None and self.is_local_testing:
            s.options.simulator.seed_simulator = self.seed
        return s

    def estimator(self):
        from qiskit_ibm_runtime import EstimatorV2
        e = EstimatorV2(mode=self.backend)
        if not self.is_local_testing:
            e.options.resilience_level = self.resilience_level
            e.options.dynamical_decoupling.enable = self.dynamical_decoupling
        if self.seed is not None and self.is_local_testing:
            e.options.simulator.seed_simulator = self.seed
        return e


# -- Provenance -----------------------------------------------------------------

@dataclass
class JobProvenance:
    """
    What actually happened, recorded rather than assumed.

    Fields that a local-testing (Fake) backend cannot populate are left
    ``None`` — see the class docstring on ``core.execution.ExecutionRecord``
    for the same "don't fabricate a plausible-looking default" principle
    applied there.

    Attributes
    ----------
    job_id : str
    backend_name, backend_version : str
    is_local_testing : bool
    resilience_level : int or None
        ``None`` for a Sampler-based job, which has no resilience level.
    shots : int or None
    wall_time_s : float
        Measured here, not reported by the job -- the elapsed time of the
        ``.run()`` call, submission through result retrieval.
    """
    job_id:             str
    backend_name:        str
    backend_version:     Optional[str]
    is_local_testing:    bool
    resilience_level:    Optional[int]
    shots:               Optional[int]
    wall_time_s:         float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "job_id":            self.job_id,
            "backend_name":      self.backend_name,
            "backend_version":   self.backend_version,
            "is_local_testing":  self.is_local_testing,
            "resilience_level":  self.resilience_level,
            "shots":             self.shots,
            "wall_time_s":       self.wall_time_s,
        }


def _backend_version(backend) -> Optional[str]:
    return getattr(backend, "backend_version", None)


# -- Post-selection sampling ----------------------------------------------------

@dataclass
class PostSelectionSample:
    """
    Real-hardware counterpart of ``core.noise.PostSelectionSample`` — same
    fields and meaning, with a ``provenance`` field added since a hardware
    run needs its job traced, unlike a local Aer call.
    """
    shots:       int
    n_accepted:  int
    probability: float
    ci_low:      float
    ci_high:     float
    provenance:  JobProvenance

    @property
    def shot_overhead(self) -> float:
        return 1.0 / self.probability if self.probability > 0 else float("inf")


def hardware_postselection_sample(
    circuit,
    spec:     PostSelectSpec,
    context:  HardwareContext,
    shots:    Optional[int] = None,
) -> PostSelectionSample:
    """
    Real-hardware (or local-testing) post-selection statistics.

    Same acceptance-counting logic as ``core.noise.sample_postselection``:
    measure every qubit, parse each returned bitstring directly as this
    project's little-endian basis-state index (no reversal — see the
    detailed correction note in ``core.noise.sample_postselection``, which
    this function repeats exactly, having been copied from the version that
    was actually fixed and tested there).

    Parameters
    ----------
    circuit : QuantumCircuit
        Not yet transpiled or measured; both are done here.
    spec : PostSelectSpec
    context : HardwareContext
    shots : int or None
        Defaults to ``context.default_shots``.
    """
    from qiskit import transpile

    shots = shots if shots is not None else context.default_shots

    meas = circuit.copy()
    meas.measure_all()
    tqc = transpile(meas, backend=context.backend, optimization_level=1)

    sampler = context.sampler()
    t0 = time.time()
    job = sampler.run([tqc], shots=shots)
    result = job.result()
    wall_time = time.time() - t0

    counts = result[0].data.meas.get_counts()

    n_accepted = 0
    for bitstring, count in counts.items():
        idx = int(bitstring.replace(" ", ""), 2)
        if spec.accepts(idx):
            n_accepted += count

    p_hat = n_accepted / shots
    ci_low, ci_high = _wilson_interval(n_accepted, shots)

    provenance = JobProvenance(
        job_id            = job.job_id(),
        backend_name      = context.backend.name,
        backend_version   = _backend_version(context.backend),
        is_local_testing  = context.is_local_testing,
        resilience_level  = None,
        shots             = shots,
        wall_time_s       = wall_time,
    )

    return PostSelectionSample(
        shots=shots, n_accepted=n_accepted, probability=p_hat,
        ci_low=ci_low, ci_high=ci_high, provenance=provenance,
    )


def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion k/n. Matches core.noise."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


# -- Observable estimation -------------------------------------------------------

@dataclass
class EstimateResult:
    """One expectation value, its standard error, and where it came from."""
    value:       float
    std_error:   float
    provenance:  JobProvenance


def hardware_estimate(
    circuit,
    observable,
    context:  HardwareContext,
    shots:    Optional[int] = None,
) -> EstimateResult:
    """
    A single Pauli-sum expectation value on real (or local-testing) hardware.

    For more than one observable, or more than one circuit, use
    :func:`hardware_estimate_batch` instead — see module docstring for why a
    loop of single calls is not a scaling strategy here.

    Parameters
    ----------
    circuit : QuantumCircuit
        Not yet transpiled.
    observable : qiskit.quantum_info.SparsePauliOp
        Not yet layout-mapped; this function transpiles the circuit and
        applies the resulting layout to the observable, so pass the
        observable in the circuit's original (un-transpiled) qubit order.
    """
    results = hardware_estimate_batch([(circuit, observable)], context, shots=shots)
    return results[0]


def hardware_estimate_batch(
    circuit_observable_pairs: Sequence[Tuple[Any, Any]],
    context:  HardwareContext,
    shots:    Optional[int] = None,
) -> List[EstimateResult]:
    """
    Many (circuit, observable) pairs, submitted as one batched Runtime job.

    This is the primitive every multi-measurement experiment in this
    project should be built on. Measured directly: a 40-PUB batch completed
    in ~22s against a local-testing backend; the equivalent single-PUB loop
    did not complete in over two minutes for the same 40 circuits. The
    difference is per-job overhead, not per-circuit compute, and it compounds
    with every additional circuit a loop submits separately.

    Each pair may share a circuit (different observables on the same
    prepared state) or not; nothing here assumes they are related.

    Parameters
    ----------
    circuit_observable_pairs : sequence of (QuantumCircuit, SparsePauliOp)
        Each circuit is transpiled independently; each observable is
        layout-mapped to its own circuit's transpiled layout.
    context : HardwareContext
    shots : int or None
        Applied uniformly to every PUB in the batch via
        ``EstimatorOptions.default_shots``.

    Returns
    -------
    list of EstimateResult, same order as the input pairs. All share one
    ``JobProvenance`` (one job id, one wall time for the whole batch) except
    that each ``EstimateResult.provenance`` is a distinct object -- callers
    inspecting provenance per-result still get a consistent, complete record
    for the batch each result came from.
    """
    from qiskit import transpile

    shots = shots if shots is not None else context.default_shots

    pubs = []
    for qc, obs in circuit_observable_pairs:
        tqc = transpile(qc, backend=context.backend, optimization_level=1)
        obs_mapped = obs.apply_layout(tqc.layout)
        pubs.append((tqc, obs_mapped))

    estimator = context.estimator()
    estimator.options.default_shots = shots

    t0 = time.time()
    job = estimator.run(pubs)
    result = job.result()
    wall_time = time.time() - t0

    provenance = JobProvenance(
        job_id            = job.job_id(),
        backend_name      = context.backend.name,
        backend_version   = _backend_version(context.backend),
        is_local_testing  = context.is_local_testing,
        # Assigned None during local-testing mode: this configuration option
        # is non-functional in that context (see HardwareContext.estimator),
        # hence recording a nominal value would misrepresent execution conditions.
        resilience_level  = None if context.is_local_testing else context.resilience_level,
        shots             = shots,
        wall_time_s       = wall_time,
    )

    return [
        EstimateResult(
            value      = float(pub_result.data.evs),
            std_error  = float(pub_result.data.stds),
            provenance = provenance,
        )
        for pub_result in result
    ]


# -- Fidelity estimation (block-encoding hardware experiment) -----------------

def hardware_fidelity_estimate(
    circuit,
    target_state: np.ndarray,
    context:      HardwareContext,
    shots:        Optional[int] = None,
) -> Tuple[EstimateResult, int]:
    """
    Fidelity of ``circuit``'s output against a classically-known
    ``target_state``, via Direct Fidelity Estimation.

    This is the tool for the block-encoding hardware experiment from the
    original hardware-scoping discussion: prepare |b_norm>, apply the block
    encoding once, and measure how close the hardware output is to the
    classically-computed target M|b>/alpha.

    Method: decompose the rank-1 projector |target><target| into Pauli terms
    via ``vqls_utils.pauli_decompose_matrix`` (reused directly), then measure
    every nonzero term as one batched job via
    :func:`hardware_estimate_batch`. The fidelity is the coefficient-weighted
    sum of the measured expectation values — confirmed exactly against a
    classical statevector calculation on independent random states in this
    module's test file, before being trusted here.

    Not importance-sampled: uses every nonzero Pauli term in the
    decomposition, not a statistically-efficient subset. Reasonable at the
    circuit sizes this project targets (a handful of qubits, a few dozen
    terms at most); would need revisiting before scaling to more qubits,
    where the term count grows as 4^n.

    Parameters
    ----------
    circuit : QuantumCircuit
        Not yet transpiled or measured.
    target_state : np.ndarray, shape (2**n,)
        Unit-normalised. Not checked for normalisation here — pass a
        pre-normalised vector.

    Returns
    -------
    (EstimateResult, n_terms)
        ``EstimateResult.value`` is the fidelity estimate in [0, 1]
        (approximately; measurement noise can push it slightly outside).
        ``EstimateResult.std_error`` is the propagated standard error from
        the batch's individual term uncertainties. ``n_terms`` is the number
        of Pauli terms the estimate was built from, i.e. the size of the
        batched job -- worth reporting alongside the fidelity, since it is
        the circuit-cost this experiment actually paid.
    """
    from qiskit.quantum_info import SparsePauliOp
    from solvers.quantum.vqls_utils import pauli_decompose_matrix

    projector = np.outer(target_state, target_state.conj())
    terms = pauli_decompose_matrix(projector)

    pairs = [
        (circuit, SparsePauliOp(pauli_str))
        for _coeff, pauli_str in terms
    ]
    results = hardware_estimate_batch(pairs, context, shots=shots)

    fidelity = sum(
        coeff.real * r.value for (coeff, _s), r in zip(terms, results)
    )
    # Independent-term error propagation: Var(sum c_l X_l) = sum c_l^2 Var(X_l)
    # for independently measured terms, which these are (separate PUBs).
    std_error = float(np.sqrt(sum(
        (coeff.real * r.std_error) ** 2 for (coeff, _s), r in zip(terms, results)
    )))

    combined = EstimateResult(
        value      = float(fidelity),
        std_error  = std_error,
        provenance = results[0].provenance,   # all results share one batch/job
    )
    return combined, len(terms)