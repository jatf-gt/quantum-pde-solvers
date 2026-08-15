"""
Noisy and shot-sampled execution backends.

Extends the ``core.execution`` abstraction with the two robustness-study
tools identified in the hardware-scoping discussion: a way to see how
solution quality degrades under realistic gate noise (parametric depolarising
sweeps, and real IBM calibration data via a fake backend), and a way to see
the genuine finite-shot cost of post-selection, independent of gate error.

These are deliberately two separate questions with two separate mechanisms,
not one noisy executor trying to do both at once:

*   **Gate noise** (this module's ``NoiseExecutor``) is modelled via Aer's
    ``density_matrix`` simulation method with a ``NoiseModel`` attached. This
    computes the *exact* mixed state reachable under the given noise
    channel — no shot variance, no sampling error. It answers "how much does
    the *algorithm's output* degrade at this error rate", cleanly separated
    from "how many shots would I need". This is the right tool for the
    parametric depolarising sweep and for a fake-backend (real calibration
    data) comparison.

*   **Shot statistics** (this module's ``sample_postselection``) is modelled
    via genuine measurement and repeated shots. It answers "how many runs
    does it actually take to get one accepted sample", which is a sampling
    question with no meaningful "exact" answer — the whole point is the
    variance. This is the right tool for the HHL 1/κ² shot-overhead study.

Solution-vector reconstruction under noise, and its real limitation
---------------------------------------------------------------------
``StatevectorExecutor`` (core.execution) reads the solution off directly:
the post-selected subspace is exactly one-dimensional (pure), so masking the
statevector *is* the answer. Under noise the post-selected subspace is no
longer exactly pure — the density submatrix has some spread across several
eigenvalues — so "the solution" has to mean something more specific:
``NoiseExecutor`` takes the *leading eigenvector* of the post-selected,
renormalised density submatrix as the best available pure-state estimate,
and reports the leading eigenvalue as ``purity`` in the execution record (1.0
= no degradation; falling below that quantifies exactly how much the
post-selected state has been mixed by the noise channel).

That reconstruction carries a genuine, unavoidable limitation, stated here
rather than glossed over: the leading eigenvector is defined only up to a
global phase, and eigensolvers do not return a physically meaningful one.
``NoiseExecutor`` resolves this by comparing against a reference state from
an exact (noiseless) run of the *same* circuit and rotating the extracted
eigenvector to align with it — see ``validate_zero_noise_recovery`` in
``tests/test_noise.py`` for the empirical confirmation that this reproduces
``StatevectorExecutor`` exactly when the noise model is trivial. This makes
``NoiseExecutor`` a *validation* tool, not a hardware-realistic one: a real
device has no such reference and cannot self-calibrate its phase this way.
That is fine for this project's purpose — quantifying algorithmic robustness
before committing QPU time — but it should not be read as "what a real
hardware run would report" for the solution vector itself. The
post-selection *probability*, by contrast, needs no reference and is exactly
as a real device would report it; that is why the shot-overhead study
(``sample_postselection``) does not share this limitation.

References
----------
Nielsen & Chuang, §8.2.3 (density matrices, purity) and §2.2.6 (mixed-state
    fidelity).
Qiskit Aer documentation, "Density matrix simulation method".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from core.execution import (
    Executor,
    ExecutionRecord,
    PostSelectSpec,
    StatevectorExecutor,
)

log = logging.getLogger(__name__)

__all__ = [
    "HERON_R2_SINGLE_QUBIT_ERROR",
    "depolarizing_noise_model",
    "fake_backend_noise_model",
    "NoiseExecutor",
    "PostSelectionSample",
    "sample_postselection",
    "depolarizing_sweep",
]


# Representative median single-qubit (sx/x) error on Heron r2, order-of-
# magnitude only -- see core.resources for the two-qubit figure and the same
# re-check-before-relying-on-it caveat.
HERON_R2_SINGLE_QUBIT_ERROR: float = 3.0e-4


# -- Noise model builders ------------------------------------------------------

def depolarizing_noise_model(
    single_qubit_error: float = HERON_R2_SINGLE_QUBIT_ERROR,
    two_qubit_error:    float = None,
    basis_gates:        Sequence[str] = ("rz", "sx", "x", "cz"),
):
    """
    Build a uniform depolarising ``NoiseModel`` over the given basis.

    Applies a single-qubit depolarising channel to every one-qubit gate in
    ``basis_gates`` and a two-qubit depolarising channel to every two-qubit
    gate, at every qubit (no connectivity dependence — appropriate at the
    circuit sizes used in this project, per the same reasoning given in
    ``core.resources.ResourceReport.coupling_map``).

    ``rz`` is deliberately left unmodelled here: on real hardware, virtual
    Z rotations are implemented as a frame change and carry no gate error,
    so including a depolarising channel on ``rz`` would overstate the noise
    budget. This mirrors standard practice in IBM's own device noise models.

    Parameters
    ----------
    single_qubit_error : float
        Depolarising parameter for one-qubit gates.
    two_qubit_error : float or None
        Depolarising parameter for two-qubit gates. Defaults to
        ``core.resources.HERON_R2_TWO_QUBIT_ERROR`` if not given.
    basis_gates : sequence of str
        Gate names to attach the channel to. Splits automatically into
        one- and two-qubit gates by name against the known Heron r2 basis;
        pass a custom sequence for a different target and note that any gate
        name not recognised as one of {rz, sx, x} or {cz, cx, ecr} is
        skipped with a warning rather than silently ignored.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    if two_qubit_error is None:
        from core.resources import HERON_R2_TWO_QUBIT_ERROR
        two_qubit_error = HERON_R2_TWO_QUBIT_ERROR

    one_qubit_names = {"sx", "x", "id"}   # rz excluded deliberately, see above
    two_qubit_names = {"cz", "cx", "ecr"}

    model = NoiseModel(basis_gates=list(basis_gates))
    applied_1q, applied_2q, skipped = [], [], []

    for gate in basis_gates:
        if gate in one_qubit_names:
            model.add_all_qubit_quantum_error(
                depolarizing_error(single_qubit_error, 1), [gate]
            )
            applied_1q.append(gate)
        elif gate in two_qubit_names:
            model.add_all_qubit_quantum_error(
                depolarizing_error(two_qubit_error, 2), [gate]
            )
            applied_2q.append(gate)
        elif gate != "rz":
            skipped.append(gate)

    if skipped:
        log.warning(
            "depolarizing_noise_model: gate(s) %s not recognised as 1Q or "
            "2Q and were left unmodelled. Pass an explicit error rate via a "
            "different construction if this is not intentional.", skipped
        )

    log.debug(
        "depolarizing_noise_model: 1Q gates %s at p=%.2e, 2Q gates %s at p=%.2e",
        applied_1q, single_qubit_error, applied_2q, two_qubit_error,
    )
    return model


def fake_backend_noise_model(name: str = "FakeTorino"):
    """
    Build a ``NoiseModel`` from real IBM calibration data via a Qiskit fake
    backend.

    Requires the optional dependency ``qiskit-ibm-runtime`` (not in this
    project's ``requirements.txt``, since it is needed only for this
    validation path, not for any solver).

    **Version constraint, confirmed by direct testing, not assumed:**
    this project pins ``qiskit==1.4.5``. The current ``qiskit-ibm-runtime``
    release (0.48.0 at time of writing) requires ``qiskit>=2.3.0`` and will
    not import against that pin. ``qiskit-ibm-runtime==0.29.0`` is the
    newest release that declares ``qiskit>=1.1.0`` and does import cleanly
    here -- but that generation's fake-backend roster only goes up to
    ``FakeTorino`` (Heron r1, 133 qubits). ``FakeKingston`` (Heron r2, the
    generation targeted everywhere else in this project via
    ``core.resources.HERON_R2_BASIS_GATES``) was added in a later
    ``qiskit-ibm-runtime`` release that requires the qiskit 2.x line.

    Practically: install with

        pip install qiskit-ibm-runtime==0.29.0

    to get a real, working, Heron-generation calibration snapshot
    (``FakeTorino``) without disturbing this project's qiskit pin. Getting
    ``FakeKingston`` specifically means upgrading qiskit to >=2.3.0 project-
    wide, which is a major version bump with its own compatibility surface
    against ``qiskit_algorithms``, ``quantum_linear_solvers`` and this
    project's own circuit-construction code -- not something to do casually
    mid-thesis, and out of scope for this function. The default here is
    ``FakeTorino`` for exactly this reason: it is what actually works
    against the environment this project is pinned to.

    Parameters
    ----------
    name : str
        Fake backend class name from ``qiskit_ibm_runtime.fake_provider``.
        Confirmed available under ``qiskit-ibm-runtime==0.29.0`` as of
        August 2026: ``'FakeTorino'`` (Heron r1, 133 qubits). Pass a
        different name at your own risk of hitting the version wall above;
        the error message will list what is actually importable in your
        environment either way.

    Returns
    -------
    NoiseModel
        Built via ``NoiseModel.from_backend``, so it carries whatever the
        snapshot's calibration reported at capture time — a real, but
        dated, error profile. Not live calibration.
    """
    try:
        from qiskit_ibm_runtime import fake_provider
    except ImportError as exc:
        raise ImportError(
            "fake_backend_noise_model requires the optional dependency "
            "qiskit-ibm-runtime. Install with: pip install qiskit-ibm-runtime"
        ) from exc

    from qiskit_aer.noise import NoiseModel

    try:
        backend_cls = getattr(fake_provider, name)
    except AttributeError as exc:
        available = sorted(
            n for n in dir(fake_provider) if n.startswith("Fake")
        )
        raise ValueError(
            f"No fake backend named {name!r} in qiskit_ibm_runtime.fake_provider. "
            f"Available: {available}"
        ) from exc

    backend = backend_cls()
    log.info(
        "fake_backend_noise_model: built from %s (%d qubits)",
        name, backend.num_qubits,
    )
    return NoiseModel.from_backend(backend)


# -- Noisy solution-vector executor ---------------------------------------------

class NoiseExecutor:
    """
    Extract a solution vector under a gate noise model, via exact density-
    matrix evolution (no shot variance — see module docstring).

    Not exact in the way ``StatevectorExecutor`` is exact: under a nontrivial
    noise model, the post-selected subspace is mixed, and this executor
    reports the best pure-state estimate (leading eigenvector) together with
    ``purity`` (the leading eigenvalue) so that "how much was thrown away by
    this approximation" is always visible in the returned
    :class:`~core.execution.ExecutionRecord`, not hidden.

    Parameters
    ----------
    noise_model : NoiseModel or None
        Aer noise model to apply. ``None`` runs the *mechanism* (density
        matrix, eigenvector extraction, phase alignment) with no actual
        noise — useful only as a self-test that this executor reproduces
        ``StatevectorExecutor`` in the trivial limit; see
        ``tests/test_noise.py``.
    reference_executor : Executor or None
        Used once per ``extract`` call to obtain the phase-alignment
        reference (see module docstring). Defaults to a fresh
        ``StatevectorExecutor``. This is what makes ``NoiseExecutor`` a
        validation tool rather than a hardware-standalone one: it always
        needs one exact simulation of the same circuit to calibrate against.
    basis_gates : sequence of str
        Basis to transpile into before density-matrix simulation. Aer's
        density-matrix method does not natively support every instruction
        this project's circuits use (notably ``Isometry`` for state
        preparation), so transpilation is not optional here.
    zero_atol : float
        Null-vector threshold, matching ``StatevectorExecutor``'s default.
    """

    mode = "noisy"

    def __init__(
        self,
        noise_model         = None,
        reference_executor: Optional[Executor] = None,
        basis_gates:        Sequence[str] = ("rz", "sx", "x", "cz"),
        zero_atol:           float = 1e-10,
    ):
        self.noise_model         = noise_model
        self.reference_executor  = reference_executor or StatevectorExecutor()
        self.basis_gates         = tuple(basis_gates)
        self.zero_atol           = zero_atol

    def extract(
        self,
        circuit,
        spec: PostSelectSpec,
    ) -> Tuple[np.ndarray, ExecutionRecord]:
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        n_total = circuit.num_qubits
        N       = 2 ** spec.n_data

        # Reference for phase alignment -- see module docstring. Computed
        # first and cheaply reused; this is an exact statevector run of the
        # *unnoised* circuit, independent of self.noise_model.
        ref_amplitudes = self._reference_accepted_amplitudes(circuit, spec)

        tqc = transpile(circuit, basis_gates=list(self.basis_gates), optimization_level=1)
        tqc.save_density_matrix()
        sim = AerSimulator(method="density_matrix", noise_model=self.noise_model)
        rho = np.asarray(sim.run(tqc).result().data(0)["density_matrix"])

        indices    = np.arange(2 ** n_total, dtype=np.int64)
        accepted   = (indices & spec.mask) == spec.target
        sub        = rho[np.ix_(accepted, accepted)]
        prob       = float(np.trace(sub).real)

        record_common = dict(
            mode                   = self.mode,
            n_qubits               = n_total,
            backend                = (
                "density_matrix+noise" if self.noise_model is not None
                else "density_matrix+ideal"
            ),
            circuit_depth          = tqc.depth(),
            shots                  = None,
            postselect_probability = prob,
            n_accepted             = None,
        )

        if prob < self.zero_atol:
            raise RuntimeError(
                f"{spec.label or 'NoiseExecutor'}: post-selected subspace has "
                f"vanishing weight (p={prob:.3e}) under the given noise model. "
                f"Either the circuit itself is broken, or the noise level is "
                f"unrealistically severe for this problem."
            )

        sub_norm         = sub / prob
        eigvals, eigvecs = np.linalg.eigh(sub_norm)
        top              = int(np.argmax(eigvals))
        purity           = float(eigvals[top])
        leading           = eigvecs[:, top]

        # Match StatevectorExecutor's sub-normalised convention (raw
        # amplitudes scaled by sqrt(acceptance probability), not unit norm)
        # -- see module docstring / tests/test_noise.py for why this matters.
        data_index = indices[accepted] & (N - 1)
        amplitudes = np.zeros(N, dtype=complex)
        amplitudes[data_index] = leading * np.sqrt(prob)

        amplitudes = _align_phase(amplitudes, ref_amplitudes)
        x_raw      = _project(amplitudes, spec.component)

        if _is_null(x_raw, self.zero_atol):
            raise RuntimeError(
                f"{spec.label or 'NoiseExecutor'}: extraction returned a "
                f"null vector after post-selection (purity={purity:.4f})."
            )

        record = ExecutionRecord(**record_common, extra={"purity": purity})
        return x_raw, record

    # -- Internals -------------------------------------------------------------

    def _reference_accepted_amplitudes(self, circuit, spec: PostSelectSpec) -> np.ndarray:
        """
        Exact (noiseless) post-selected, sub-normalised amplitude vector for
        the same circuit -- used only to fix the eigenvector's global phase.
        See module docstring for why this makes NoiseExecutor a validation
        tool rather than a standalone hardware executor.
        """
        from qiskit.quantum_info import Statevector

        sv       = np.asarray(Statevector(circuit).data, dtype=complex)
        n_total  = circuit.num_qubits
        N        = 2 ** spec.n_data
        indices  = np.arange(2 ** n_total, dtype=np.int64)
        accepted = (indices & spec.mask) == spec.target

        ref = np.zeros(N, dtype=complex)
        ref[indices[accepted] & (N - 1)] = sv[accepted]
        return ref


def _align_phase(amplitudes: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rotate ``amplitudes`` so its overlap with ``reference`` is real positive."""
    overlap = np.vdot(reference, amplitudes)
    if abs(overlap) < 1e-14:
        return amplitudes  # reference too small to calibrate against; leave as-is
    phase = overlap / abs(overlap)
    return amplitudes / phase


def _project(amplitudes: np.ndarray, component: str) -> np.ndarray:
    if component == "real":
        return np.real(amplitudes)
    if component == "imag":
        return np.imag(amplitudes)
    return amplitudes


def _is_null(x: np.ndarray, atol: float) -> bool:
    return bool(np.allclose(x, 0.0, atol=atol))


# -- Shot-sampled post-selection statistics ------------------------------------

@dataclass
class PostSelectionSample:
    """
    Genuine finite-shot post-selection statistics for one circuit.

    Unlike everything above, this carries real sampling variance -- it is
    the answer to "if I actually ran this on a shot-based backend, how many
    of my shots would survive post-selection", which is exactly the
    quantity behind HHL's 1/κ² overhead claim.

    Attributes
    ----------
    shots, n_accepted : int
    probability : float
        ``n_accepted / shots``, the point estimate.
    ci_low, ci_high : float
        Wilson score 95% confidence interval on the true acceptance
        probability. Reported because a point estimate from a few hundred
        shots at a probability like 1/κ² ~ 0.01 carries real uncertainty
        that a bare ratio hides.
    shot_overhead : float
        ``1 / probability`` — expected total shots per accepted sample.
    """
    shots:       int
    n_accepted:  int
    probability: float
    ci_low:      float
    ci_high:     float

    @property
    def shot_overhead(self) -> float:
        return 1.0 / self.probability if self.probability > 0 else float("inf")


def sample_postselection(
    circuit,
    spec:        PostSelectSpec,
    shots:       int = 8192,
    noise_model = None,
    seed:        int = 0,
) -> PostSelectionSample:
    """
    Run ``circuit`` with real measurement and ``shots`` repetitions, and
    report how many survive the post-selection condition in ``spec``.

    This measures every qubit in the computational basis, exactly as a real
    device submission would, and counts acceptance from the resulting
    bitstring counts -- no density matrix, no eigenvector reconstruction,
    because none is needed: acceptance is a computational-basis event.

    Parameters
    ----------
    shots : int
        Default 8192, matching a common IBM Runtime default job size.
    noise_model : NoiseModel or None
        If given, applied during the shot-based run (readout error included,
        if present in the model) -- unlike ``NoiseExecutor``, this can
        legitimately combine gate noise and shot noise in one run.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    meas = circuit.copy()
    meas.measure_all()

    tqc = transpile(
        meas, basis_gates=["rz", "sx", "x", "cz"], optimization_level=1
    )
    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    counts = sim.run(tqc, shots=shots).result().get_counts()

    n_total = circuit.num_qubits
    n_accepted = 0
    for bitstring, count in counts.items():
        # Qiskit count keys are strings read left-to-right as clbit[n-1]...
        # clbit[0] (MSB first); measure_all() maps qubit i -> clbit i, and
        # this project's basis-state index convention (core.execution) is
        # exactly "bit i of the integer index is qubit i" -- so the string,
        # parsed directly as a binary integer, already IS that index. No
        # reversal. (Verified directly against a known state in
        # tests/test_noise.py -- an earlier draft of this function reversed
        # the string on a mistaken assumption and silently inverted every
        # acceptance decision; catching that required checking a circuit
        # with a known answer, not trusting the docstring reasoning alone.)
        idx = int(bitstring.replace(" ", ""), 2)
        if spec.accepts(idx):
            n_accepted += count

    p_hat = n_accepted / shots
    ci_low, ci_high = _wilson_interval(n_accepted, shots)

    return PostSelectionSample(
        shots=shots, n_accepted=n_accepted, probability=p_hat,
        ci_low=ci_low, ci_high=ci_high,
    )


def _wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion k/n. No scipy dependency."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom))


# -- Convenience: a parametric sweep --------------------------------------------

def depolarizing_sweep(
    circuit,
    spec:              PostSelectSpec,
    error_rates:       Sequence[float],
    single_qubit_error: float = HERON_R2_SINGLE_QUBIT_ERROR,
    reference_amplitudes: Optional[np.ndarray] = None,
    basis_gates:       Sequence[str] = ("rz", "sx", "x", "cz"),
) -> list:
    """
    Run :class:`NoiseExecutor` across a sweep of two-qubit depolarising
    error rates, reporting fidelity against the exact (zero-noise) solution
    at each point.

    This is the "parametric depolarising sweep" step of the robustness
    programme: an error-vs-fidelity curve, cheap enough to run entirely on
    CX3 or a laptop, no QPU time required.

    Read the ``error_rates=0.0`` row carefully: it is not a true zero-noise
    baseline. ``single_qubit_error`` defaults to the realistic Heron r2
    floor and is held fixed across the whole sweep (only the two-qubit rate
    varies), so the first row already carries whatever purity loss that
    floor produces once compounded over every single-qubit gate the circuit
    transpiles into. For a QSVT circuit of any real depth this is not
    negligible: at ``degree=11``, ``N=4``, a floor of 3e-4 alone was
    observed to reduce purity to ~0.68, purely from single-qubit gate count
    — before a single two-qubit error is added. Pass
    ``single_qubit_error=0.0`` explicitly for a genuine all-zero reference
    point.

    Parameters
    ----------
    error_rates : sequence of float
        Two-qubit depolarising probabilities to sweep.
    single_qubit_error : float
        Held fixed across the sweep; see the warning above. Defaults to
        ``HERON_R2_SINGLE_QUBIT_ERROR``.
    reference_amplitudes : np.ndarray or None
        Exact accepted-subspace amplitudes to compare against. Computed
        automatically from a fresh ``StatevectorExecutor`` run if not given;
        pass it explicitly to avoid recomputing across many sweep points
        sharing the same circuit.

    Returns
    -------
    list of dict, each with: two_qubit_error, single_qubit_error, purity,
    fidelity_vs_ideal, postselect_probability
    """
    ref_executor = StatevectorExecutor(diagnostics=False)
    if reference_amplitudes is None:
        x_ref, _ = ref_executor.extract(circuit, spec)
        reference_amplitudes = x_ref  # already the projected component

    rows = []
    for p2 in error_rates:
        noise_model = depolarizing_noise_model(
            single_qubit_error=single_qubit_error,
            two_qubit_error=p2,
            basis_gates=basis_gates,
        )
        executor = NoiseExecutor(noise_model=noise_model, basis_gates=basis_gates)
        x_noisy, record = executor.extract(circuit, spec)

        denom = np.linalg.norm(reference_amplitudes) * np.linalg.norm(x_noisy)
        fidelity = (
            float((np.dot(reference_amplitudes, x_noisy) / denom) ** 2)
            if denom > 0 else 0.0
        )
        rows.append({
            "two_qubit_error":        p2,
            "single_qubit_error":     single_qubit_error,
            "purity":                 record.extra["purity"],
            "fidelity_vs_ideal":      fidelity,
            "postselect_probability": record.postselect_probability,
        })
    return rows