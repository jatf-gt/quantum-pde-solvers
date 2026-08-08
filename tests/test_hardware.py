"""
Tests for core.hardware.

Everything here runs against FakeTorino (local testing mode): real Runtime
primitives (SamplerV2/EstimatorV2), real PUB submission, real calibration-
based noise, zero credentials, zero network access, zero queue time. This
is the strongest test possible without a live IBM Quantum connection, and
it is exactly what every function in core.hardware was run against before
being trusted (see that module's docstring).

Kept deliberately small: each job submission carries measurable fixed
overhead even in local-testing mode (a single-circuit job was observed to
take ~15-20s here), so this suite favours few, well-chosen cases over broad
parametrisation. Broader coverage belongs in an experiment script run
on-demand, not in a suite that runs on every commit.
"""
from __future__ import annotations

import numpy as np
import pytest

qiskit_ibm_runtime = pytest.importorskip("qiskit_ibm_runtime")

from qiskit import QuantumCircuit                              # noqa: E402
from qiskit.quantum_info import SparsePauliOp, Statevector      # noqa: E402

from core.execution import PostSelectSpec                       # noqa: E402
from core.hardware import (                                     # noqa: E402
    HardwareContext,
    hardware_estimate_batch,
    hardware_fidelity_estimate,
    hardware_postselection_sample,
)


@pytest.fixture(scope="module")
def ctx():
    return HardwareContext.local_testing()


# ── HardwareContext ────────────────────────────────────────────────────────────

class TestHardwareContext:

    def test_local_testing_flag(self, ctx):
        assert ctx.is_local_testing is True

    def test_local_testing_backend_is_fake_torino(self, ctx):
        assert ctx.backend.name == "fake_torino"

    def test_real_requires_no_credentials_at_construction_time(self):
        # HardwareContext.real() is not called here (it needs a saved
        # account and network access, neither available in this sandbox).
        # This test only confirms local_testing() -- the path this whole
        # suite exercises -- touches no credential machinery at all.
        ctx2 = HardwareContext.local_testing()
        assert ctx2._service is None


# ── Post-selection sampling ───────────────────────────────────────────────────

@pytest.mark.quantum
class TestHardwarePostselectionSample:

    def test_deterministic_acceptance(self, ctx):
        # Same regression this exact check caught a real bug for in
        # core.noise.sample_postselection: qubit 2 forced to |1>, spec
        # requires qubit2=1 -- should accept essentially every shot.
        qc = QuantumCircuit(3)
        qc.x(2)
        spec = PostSelectSpec(n_data=2, conditions=((2, 1),))
        sample = hardware_postselection_sample(qc, spec, ctx, shots=300)
        assert sample.probability > 0.85  # allow for realistic device noise

    def test_deterministic_rejection(self, ctx):
        qc = QuantumCircuit(3)
        qc.x(2)
        spec = PostSelectSpec(n_data=2, conditions=((2, 0),))
        sample = hardware_postselection_sample(qc, spec, ctx, shots=300)
        assert sample.probability < 0.15

    def test_provenance_populated(self, ctx):
        qc = QuantumCircuit(2)
        spec = PostSelectSpec(n_data=1, conditions=((1, 0),))
        sample = hardware_postselection_sample(qc, spec, ctx, shots=200)
        assert sample.provenance.job_id
        assert sample.provenance.backend_name == "fake_torino"
        assert sample.provenance.is_local_testing is True
        assert sample.provenance.shots == 200
        assert sample.provenance.wall_time_s > 0


# ── Observable estimation ─────────────────────────────────────────────────────

@pytest.mark.quantum
class TestHardwareEstimateBatch:

    def test_bell_state_correlations(self, ctx):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        pairs = [(qc, SparsePauliOp("ZZ")), (qc, SparsePauliOp("XX"))]
        results = hardware_estimate_batch(pairs, ctx, shots=1500)
        # Ideal Bell state: <ZZ> = <XX> = 1. Allow generous margin for
        # realistic device noise on a local-testing backend.
        assert results[0].value > 0.5
        assert results[1].value > 0.5

    def test_batch_shares_one_job(self, ctx):
        qc = QuantumCircuit(1)
        qc.h(0)
        pairs = [(qc, SparsePauliOp("Z")), (qc, SparsePauliOp("X"))]
        results = hardware_estimate_batch(pairs, ctx, shots=500)
        assert results[0].provenance.job_id == results[1].provenance.job_id

    def test_resilience_level_is_none_in_local_testing(self, ctx):
        # Local testing mode genuinely ignores resilience_level -- the
        # provenance should say so honestly (None), not echo back the
        # configured value as if it took effect. See HardwareContext.estimator
        # and hardware_estimate_batch's provenance construction.
        qc = QuantumCircuit(1)
        qc.h(0)
        results = hardware_estimate_batch(
            [(qc, SparsePauliOp("Z"))], ctx, shots=200
        )
        assert ctx.is_local_testing is True
        assert results[0].provenance.resilience_level is None


# ── Fidelity estimation ───────────────────────────────────────────────────────

class TestFidelityFormulaClassical:
    """
    The Direct Fidelity Estimation identity itself, checked with plain
    linear algebra -- no hardware, no Runtime primitives, just confirming
    the formula hardware_fidelity_estimate relies on before trusting any
    hardware-backed use of it.
    """

    def test_matches_direct_fidelity_on_random_states(self):
        from solvers.quantum.vqls_utils import (
            pauli_decompose_matrix,
            _pauli_string_to_matrix,
        )

        n = 2
        rng = np.random.default_rng(0)
        psi_target = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        psi_target /= np.linalg.norm(psi_target)
        psi_actual = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
        psi_actual /= np.linalg.norm(psi_actual)

        projector = np.outer(psi_target, psi_target.conj())
        terms = pauli_decompose_matrix(projector)

        via_paulis = sum(
            c * np.vdot(psi_actual, _pauli_string_to_matrix(s) @ psi_actual)
            for c, s in terms
        )
        direct = abs(np.vdot(psi_target, psi_actual)) ** 2
        assert via_paulis.real == pytest.approx(direct, abs=1e-9)

    def test_sparse_pauli_op_label_matches_pauli_string_to_matrix(self):
        # Foundational assumption for hardware_fidelity_estimate: the Pauli
        # strings pauli_decompose_matrix returns can be fed directly into
        # SparsePauliOp without any reordering.
        from solvers.quantum.vqls_utils import _pauli_string_to_matrix

        for s in ["II", "XI", "IX", "XY", "YZ", "ZXI", "IYX"]:
            assert np.allclose(
                _pauli_string_to_matrix(s), SparsePauliOp(s).to_matrix()
            )


@pytest.mark.quantum
class TestHardwareFidelityEstimate:

    def test_self_fidelity_exceeds_orthogonal_fidelity(self, ctx):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        target_self = np.asarray(Statevector(qc).data)
        # Orthogonal to (|00>+|11>)/sqrt2: (|00>-|11>)/sqrt2
        target_orthogonal = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)

        fid_self, _ = hardware_fidelity_estimate(qc, target_self, ctx, shots=1500)
        fid_orth, _ = hardware_fidelity_estimate(qc, target_orthogonal, ctx, shots=1500)

        assert fid_self.value > fid_orth.value
        assert fid_self.value > 0.5
        assert fid_orth.value < 0.3

    def test_reports_term_count(self, ctx):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        target = np.asarray(Statevector(qc).data)
        _result, n_terms = hardware_fidelity_estimate(qc, target, ctx, shots=500)
        assert n_terms > 0
        assert n_terms <= 4 ** 2  # at most 4^n Pauli terms for n=2 qubits