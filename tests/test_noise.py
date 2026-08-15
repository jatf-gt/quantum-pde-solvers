"""
Tests for core.noise.

Two categories of thing are pinned here, matching how this module's
correctness was actually established: not by reasoning about the code, but
by running it against circuits with a known answer and catching two real
bugs that reasoning alone missed (a normalisation mismatch in the
eigenvector reconstruction, and an inverted bit-order assumption in the
shot-counting path). Both are pinned as regression tests below so neither
can silently return.
"""
from __future__ import annotations

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")
qiskit_aer = pytest.importorskip("qiskit_aer")

from qiskit import QuantumCircuit, QuantumRegister              # noqa: E402

from core.execution import (                                    # noqa: E402
    PostSelectSpec,
    StatevectorExecutor,
    hhl_spec,
    qsvt_spec,
)
from core.noise import (                                        # noqa: E402
    HERON_R2_SINGLE_QUBIT_ERROR,
    NoiseExecutor,
    depolarizing_noise_model,
    depolarizing_sweep,
    sample_postselection,
)


def _hhl_shaped_circuit(seed: int = 0) -> QuantumCircuit:
    """A fixed, nontrivial 7-qubit HHL-layout circuit for cross-checks."""
    qc = QuantumCircuit(
        QuantumRegister(2, "b"), QuantumRegister(3, "l"),
        QuantumRegister(1, "anc"), QuantumRegister(1, "flag"),
    )
    qc.h(range(7))
    for i in range(6):
        qc.cx(i, i + 1)
    qc.rz(1.1, 6)
    return qc


def _real_qsvt_circuit(degree: int = 11, seed: int = 0):
    from solvers.quantum.block_encoding import build_tst_block_encoding
    from solvers.quantum.qsvt_1d import _build_qsvt_circuit

    n = 2
    be_circuit, _alpha = build_tst_block_encoding(N=4, main_diag=-2.0, off_diag=1.0)
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.1, 3.0, size=degree + 1)
    b_norm_vec = np.ones(4) / 2.0
    qc = _build_qsvt_circuit(be_circuit, angles, n, b_norm_vec)
    return qc, qsvt_spec(n, 1)


# -- Noise model builders ------------------------------------------------------

class TestDepolarizingNoiseModel:

    def test_excludes_rz_from_error(self):
        # Virtual-Z rotations carry no gate error on real hardware; a model
        # that puts a channel on rz would overstate the noise budget.
        model = depolarizing_noise_model(single_qubit_error=0.5, two_qubit_error=0.5)
        assert "rz" not in model.noise_instructions

    def test_default_uses_resources_two_qubit_error(self):
        from core.resources import HERON_R2_TWO_QUBIT_ERROR
        model_default = depolarizing_noise_model()
        model_explicit = depolarizing_noise_model(two_qubit_error=HERON_R2_TWO_QUBIT_ERROR)
        # Both should be constructible without error and target the same gates.
        assert set(model_default.noise_instructions) == set(model_explicit.noise_instructions)


@pytest.mark.quantum
class TestFakeBackendNoiseModel:
    """
    qiskit-ibm-runtime's current release requires qiskit>=2.3.0 and will not
    import against this project's qiskit==1.4.5 pin -- confirmed by direct
    installation attempts, not assumed. qiskit-ibm-runtime==0.29.0 is the
    newest release compatible with the pin, and its fake-backend roster only
    reaches FakeTorino (Heron r1), not FakeKingston (Heron r2). These tests
    target what is actually reachable in this project's environment.
    """

    def test_torino_loads_or_skips_cleanly(self):
        pytest.importorskip("qiskit_ibm_runtime")
        from core.noise import fake_backend_noise_model
        model = fake_backend_noise_model("FakeTorino")
        assert "cz" in model.noise_instructions

    def test_default_is_torino_not_kingston(self):
        # The default was deliberately changed from FakeKingston (unreachable
        # under this project's qiskit pin) to FakeTorino (confirmed reachable)
        # -- pinned here so it isn't quietly changed back without re-checking
        # the version constraint this test's class docstring documents.
        import inspect
        from core.noise import fake_backend_noise_model
        default = inspect.signature(fake_backend_noise_model).parameters["name"].default
        assert default == "FakeTorino"

    def test_unknown_backend_name_lists_available(self):
        pytest.importorskip("qiskit_ibm_runtime")
        from core.noise import fake_backend_noise_model
        with pytest.raises(ValueError, match="Available:"):
            fake_backend_noise_model("FakeNotARealBackend")


# -- sample_postselection: the bit-order regression ----------------------------

@pytest.mark.quantum
class TestSamplePostselection:
    """
    An earlier draft of sample_postselection reversed the counts-key bit
    order on a mistaken assumption and silently inverted every acceptance
    decision -- it still ran, still returned plausible-looking numbers, and
    was wrong. These pin the fix against circuits with a known, unambiguous
    answer, which is what actually caught it.
    """

    def test_accepts_matching_condition(self):
        qc = QuantumCircuit(3)
        qc.x(2)
        spec = PostSelectSpec(n_data=2, conditions=((2, 1),))
        sample = sample_postselection(qc, spec, shots=200)
        assert sample.n_accepted == 200

    def test_rejects_non_matching_condition(self):
        qc = QuantumCircuit(3)
        qc.x(2)
        spec = PostSelectSpec(n_data=2, conditions=((2, 0),))
        sample = sample_postselection(qc, spec, shots=200)
        assert sample.n_accepted == 0

    def test_multi_qubit_condition(self):
        # bits 3 and 4 both required =1; only bit 3 is set here.
        qc = QuantumCircuit(5)
        qc.x(3)
        spec = PostSelectSpec(n_data=3, conditions=((3, 1), (4, 1)))
        sample = sample_postselection(qc, spec, shots=200)
        assert sample.n_accepted == 0

    def test_shot_estimate_brackets_exact_probability(self):
        qc = _hhl_shaped_circuit()
        spec = hhl_spec(qc, 2)
        _x, record = StatevectorExecutor(diagnostics=False).extract(qc, spec)
        sample = sample_postselection(qc, spec, shots=20_000, seed=1)
        assert sample.ci_low <= record.postselect_probability <= sample.ci_high

    def test_shot_overhead_is_reciprocal_of_probability(self):
        qc = QuantumCircuit(2)
        spec = PostSelectSpec(n_data=1, conditions=((1, 0),))
        sample = sample_postselection(qc, spec, shots=1000)
        if sample.probability > 0:
            assert sample.shot_overhead == pytest.approx(1.0 / sample.probability)


# -- NoiseExecutor: zero-noise recovery (the normalisation regression) ---------

@pytest.mark.quantum
class TestNoiseExecutorZeroNoise:
    """
    An earlier draft compared a unit-normalised eigenvector directly against
    StatevectorExecutor's sub-normalised (probability-encoding) output,
    producing a ~35% spurious 'error' at exactly zero noise. Pinned here
    against both a toy circuit and a real QSVT circuit.
    """

    def test_hhl_shaped_circuit_matches_statevector_executor(self):
        qc = _hhl_shaped_circuit()
        spec = hhl_spec(qc, 2)
        x_exact, _ = StatevectorExecutor(diagnostics=False).extract(qc, spec)
        x_noisy, record = NoiseExecutor(noise_model=None).extract(qc, spec)
        assert record.extra["purity"] == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(x_noisy, x_exact, atol=1e-6)

    def test_real_qsvt_circuit_matches_statevector_executor(self):
        qc, spec = _real_qsvt_circuit()
        x_exact, _ = StatevectorExecutor(diagnostics=False).extract(qc, spec)
        x_noisy, record = NoiseExecutor(noise_model=None).extract(qc, spec)
        assert record.extra["purity"] == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(x_noisy, x_exact, atol=1e-6)

    def test_explicit_zero_rate_model_also_recovers_exactly(self):
        qc, spec = _real_qsvt_circuit()
        x_exact, _ = StatevectorExecutor(diagnostics=False).extract(qc, spec)
        zero_model = depolarizing_noise_model(single_qubit_error=0.0, two_qubit_error=0.0)
        x_noisy, record = NoiseExecutor(noise_model=zero_model).extract(qc, spec)
        assert record.extra["purity"] == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(x_noisy, x_exact, atol=1e-6)


# -- NoiseExecutor under real noise --------------------------------------------

@pytest.mark.quantum
class TestNoiseExecutorUnderNoise:

    def test_purity_decreases_with_two_qubit_error(self):
        qc, spec = _real_qsvt_circuit()
        rows = depolarizing_sweep(
            qc, spec, error_rates=[0.0, 0.01, 0.05], single_qubit_error=0.0
        )
        purities = [r["purity"] for r in rows]
        assert purities == sorted(purities, reverse=True)
        assert purities[0] == pytest.approx(1.0, abs=1e-6)  # true zero baseline
        assert purities[-1] < purities[0]

    def test_single_qubit_floor_alone_measurably_degrades_deep_circuit(self):
        # The finding that motivated exposing single_qubit_error explicitly:
        # even the tiny realistic floor, compounded over a real QSVT
        # circuit's gate count, is not negligible.
        qc, spec = _real_qsvt_circuit(degree=11)
        rows = depolarizing_sweep(
            qc, spec, error_rates=[0.0],
            single_qubit_error=HERON_R2_SINGLE_QUBIT_ERROR,
        )
        assert rows[0]["purity"] < 0.95, (
            "expected the single-qubit floor to measurably degrade a "
            "degree-11 QSVT circuit; if this now passes at >=0.95 either "
            "the circuit construction changed or the floor value did"
        )

    def test_fidelity_vs_ideal_responds_to_noise(self):
        qc, spec = _real_qsvt_circuit()
        rows = depolarizing_sweep(
            qc, spec, error_rates=[0.0, 0.1], single_qubit_error=0.0
        )
        assert rows[0]["fidelity_vs_ideal"] > rows[1]["fidelity_vs_ideal"]

    def test_vanishing_postselect_probability_raises(self):
        # A spec whose condition the circuit can never satisfy: qubit 1 is
        # forced to |1> deterministically, but the spec requires |0>.
        qc = QuantumCircuit(2)
        qc.x(1)
        spec = PostSelectSpec(n_data=1, conditions=((1, 0),))
        with pytest.raises(RuntimeError, match="vanishing weight"):
            NoiseExecutor(noise_model=None, zero_atol=1e-6).extract(qc, spec)