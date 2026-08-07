"""
Equivalence tests for the execution abstraction.

These tests exist to defend one claim: introducing ``core.execution`` did not
change any number the thesis reports. They do that by carrying a verbatim
transcription of the pre-refactor inline extraction logic and asserting that
``StatevectorExecutor`` reproduces it *exactly* — ``array_equal``, not
``allclose`` — on randomly seeded states across every register layout the
project uses.

The transcriptions below are frozen. They must not be "tidied up" to match
the new implementation: their whole value is that they were written
independently of it. If a future change makes them disagree, the change is
wrong until proven otherwise.

Scope note
----------
These tests cover the extraction step in isolation, on random statevectors.
End-to-end solver outputs are pinned separately in
``tests/test_regression_baseline.py``, which is the test that actually
guarantees replication of the thesis results.
"""
from __future__ import annotations

import numpy as np
import pytest

qiskit = pytest.importorskip("qiskit")

from qiskit import QuantumCircuit, QuantumRegister                # noqa: E402
from qiskit.quantum_info import Statevector, random_statevector   # noqa: E402

from core.execution import (                                      # noqa: E402
    ExecutionRecord,
    PostSelectSpec,
    StatevectorExecutor,
    default_executor,
    execution_context,
    hhl_spec,
    qsvt_spec,
)


# ── Frozen transcriptions of the pre-refactor inline logic ────────────────────

def _original_hhl_extraction(circuit, num_qubits: int) -> np.ndarray:
    """Verbatim copy of ``hhl_1d._extract_solution_statevector`` before the refactor."""
    N         = 2 ** num_qubits
    n_total   = circuit.num_qubits
    n_b       = circuit.qregs[0].size
    n_l       = circuit.qregs[1].size
    n_ancilla = n_total - 1 - n_b - n_l

    flag_bit_pos        = n_total - 1
    clock_start         = n_b
    ancilla_start       = n_b + n_l
    non_b_non_flag_mask = (
        ((1 << n_l)       - 1) << clock_start
        | ((1 << n_ancilla) - 1) << ancilla_start
    )

    sv    = Statevector(circuit).data
    x_raw = np.zeros(N, dtype=complex)
    for idx in range(2 ** n_total):
        flag_bit    = (idx >> flag_bit_pos) & 1
        middle_bits = idx & non_b_non_flag_mask
        b_reg_idx   = idx & (N - 1)
        if flag_bit == 1 and middle_bits == 0:
            x_raw[b_reg_idx] = sv[idx]
    return np.real(x_raw)


def _original_qsvt_extraction(sv: np.ndarray, n: int, n_a: int) -> np.ndarray:
    """Verbatim copy of ``qsvt_1d._extract_solution`` before the refactor."""
    N       = 2 ** n
    n_total = n + n_a
    anc_bit = n
    x_raw   = np.zeros(N, dtype=complex)
    for idx in range(2 ** n_total):
        if ((idx >> anc_bit) & 1) == 0:
            x_raw[idx & (N - 1)] = sv[idx]
    return np.imag(x_raw)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_hhl_circuit(n_b: int, n_l: int, n_anc: int, seed: int):
    qc = QuantumCircuit(
        QuantumRegister(n_b,   "b"),
        QuantumRegister(n_l,   "l"),
        QuantumRegister(n_anc, "anc"),
        QuantumRegister(1,     "flag"),
    )
    w = qc.num_qubits
    qc.initialize(random_statevector(2 ** w, seed=seed).data, range(w))
    return qc


def _random_qsvt_circuit(n: int, n_a: int, seed: int):
    qc = QuantumCircuit(
        QuantumRegister(n,   "data"),
        QuantumRegister(n_a, "anc"),
    )
    w = qc.num_qubits
    qc.initialize(random_statevector(2 ** w, seed=seed).data, range(w))
    return qc


# ── Bit-level equivalence ─────────────────────────────────────────────────────

@pytest.mark.quantum
class TestStatevectorExecutorEquivalence:
    """The executor must reproduce the original extraction exactly."""

    @pytest.mark.parametrize("n_b,n_l,n_anc", [
        (2, 3, 1),   # N=4  production layout
        (2, 4, 1),   # N=4  finer clock register
        (3, 3, 1),   # N=8  production layout
        (3, 5, 2),   # N=8  deeper clock, two MCMT ancillae
        (2, 2, 1),   # minimal
    ])
    def test_hhl_extraction_is_bit_identical(self, n_b, n_l, n_anc):
        qc       = _random_hhl_circuit(n_b, n_l, n_anc, seed=20260807)
        expected = _original_hhl_extraction(qc, n_b)
        actual, record = StatevectorExecutor().extract(qc, hhl_spec(qc, n_b))

        assert np.array_equal(actual, expected), (
            "HHL extraction diverged from the pre-refactor implementation"
        )
        assert actual.dtype == expected.dtype
        assert isinstance(record, ExecutionRecord)
        assert record.mode == "statevector"

    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6])
    def test_qsvt_extraction_is_bit_identical(self, n):
        # n_a is fixed at the production value: the Sz.-Nagy dilation of a
        # Hermitian matrix needs exactly one ancilla.
        from solvers.quantum.block_encoding import _N_ANCILLA_BE

        qc       = _random_qsvt_circuit(n, _N_ANCILLA_BE, seed=31415 + n)
        expected = _original_qsvt_extraction(
            Statevector(qc).data, n, _N_ANCILLA_BE
        )
        actual, _ = StatevectorExecutor().extract(qc, qsvt_spec(n, _N_ANCILLA_BE))

        assert np.array_equal(actual, expected)

    def test_qsvt_multi_ancilla_postselects_all_ancillae(self):
        """
        Regression guard for a latent defect in the original extraction.

        ``_extract_solution`` accepted an ``n_a`` argument but post-selected
        only on qubit ``n``, silently ignoring any further ancillae. At the
        production value ``n_a = 1`` the two behaviours coincide, so no
        thesis number is affected. They diverge for ``n_a > 1``, which is
        reachable as soon as the block encoding needs more than one ancilla —
        the pentadiagonal fourth-order operator being the obvious case.

        ``qsvt_spec`` post-selects every ancilla, which is the correct
        behaviour. This test pins that, and pins the divergence so nobody
        later "restores compatibility" by reintroducing the defect.
        """
        n, n_a = 2, 2
        qc     = _random_qsvt_circuit(n, n_a, seed=2718)
        sv     = Statevector(qc).data

        lenient  = _original_qsvt_extraction(sv, n, n_a)   # first ancilla only
        strict, _ = StatevectorExecutor().extract(qc, qsvt_spec(n, n_a))

        assert not np.array_equal(strict, lenient), (
            "expected the strict specification to differ from the original "
            "single-ancilla mask at n_a=2"
        )
        # The strict result must be a sub-selection of the lenient one: every
        # basis state it accepts, the lenient mask also accepted.
        mask   = qsvt_spec(n, n_a)
        manual = np.zeros(2 ** n, dtype=complex)
        for idx in range(2 ** qc.num_qubits):
            if mask.accepts(idx):
                manual[idx & (2 ** n - 1)] = sv[idx]
        assert np.array_equal(strict, np.imag(manual))


# ── Post-selection bookkeeping ────────────────────────────────────────────────

@pytest.mark.quantum
class TestPostSelectionProbability:
    """
    The success probability is new information, so it needs its own checks.

    It is the bridge between a simulated result and its hardware shot cost,
    and it is reported in the thesis, so it must be right.
    """

    def test_probability_matches_accepted_born_weight(self):
        qc   = _random_hhl_circuit(2, 3, 1, seed=99)
        spec = hhl_spec(qc, 2)
        sv   = Statevector(qc).data

        expected = sum(
            abs(sv[i]) ** 2
            for i in range(2 ** qc.num_qubits)
            if spec.accepts(i)
        )
        _, record = StatevectorExecutor().extract(qc, spec)
        assert record.postselect_probability == pytest.approx(expected, rel=1e-12)

    def test_shot_overhead_is_reciprocal(self):
        qc = _random_qsvt_circuit(3, 1, seed=1234)
        _, record = StatevectorExecutor().extract(qc, qsvt_spec(3, 1))
        assert record.shot_overhead == pytest.approx(
            1.0 / record.postselect_probability, rel=1e-12
        )

    def test_unconditioned_spec_accepts_everything(self):
        qc = _random_qsvt_circuit(3, 0, seed=5)
        spec = PostSelectSpec(n_data=3, conditions=(), component="real")
        _, record = StatevectorExecutor().extract(qc, spec)
        assert record.postselect_probability == pytest.approx(1.0, rel=1e-12)


# ── Specification validation ──────────────────────────────────────────────────

class TestPostSelectSpecValidation:
    """Invalid specifications must fail loudly at construction, not silently."""

    def test_rejects_condition_on_data_register(self):
        with pytest.raises(ValueError, match="collides with the data register"):
            PostSelectSpec(n_data=3, conditions=((1, 0),))

    def test_rejects_unknown_component(self):
        with pytest.raises(ValueError, match="component must be"):
            PostSelectSpec(n_data=2, component="magnitude")

    def test_rejects_non_binary_required_bit(self):
        with pytest.raises(ValueError, match="must be 0 or 1"):
            PostSelectSpec(n_data=2, conditions=((3, 2),))

    def test_mask_and_target(self):
        spec = PostSelectSpec(n_data=2, conditions=((2, 0), (3, 1), (4, 1)))
        assert spec.mask   == 0b11100
        assert spec.target == 0b11000
        assert spec.accepts(0b11000 | 0b01)
        assert not spec.accepts(0b11100 | 0b01)

    def test_hhl_spec_rejects_mismatched_register(self):
        qc = _random_hhl_circuit(2, 3, 1, seed=1)
        with pytest.raises(ValueError, match="does not match num_qubits"):
            hhl_spec(qc, num_qubits=3)


# ── Default executor and scoping ──────────────────────────────────────────────

class TestDefaultExecutor:
    """The default must be exact, and any override must be strictly scoped."""

    def test_default_is_statevector(self):
        assert default_executor().mode == "statevector"

    def test_context_manager_restores_previous(self):
        original = default_executor()
        sentinel = StatevectorExecutor(diagnostics=False)
        with execution_context(sentinel):
            assert default_executor() is sentinel
        assert default_executor() is original

    def test_context_manager_restores_on_exception(self):
        original = default_executor()
        with pytest.raises(RuntimeError):
            with execution_context(StatevectorExecutor(diagnostics=False)):
                raise RuntimeError("boom")
        assert default_executor() is original


# ── Failure path ──────────────────────────────────────────────────────────────

@pytest.mark.quantum
class TestNullExtraction:
    """A null post-selected subspace must raise, not return zeros."""

    def test_raises_on_empty_subspace(self):
        # |0...0> on every qubit: the HHL flag is never |1>, so nothing survives.
        qc = QuantumCircuit(
            QuantumRegister(2, "b"),
            QuantumRegister(2, "l"),
            QuantumRegister(1, "anc"),
            QuantumRegister(1, "flag"),
        )
        qc.x(0)  # non-trivial data register, still no flag
        with pytest.raises(RuntimeError, match="null vector under post-selection"):
            StatevectorExecutor(diagnostics=False).extract(qc, hhl_spec(qc, 2))