"""
Tests for solvers.quantum.vqls_hadamard.

The central claim this module makes is that its circuit-level cost
reproduces vqls_utils.build_cost_function exactly in analytic mode. That
claim is what these tests exist to keep true — not by construction, but by
comparison, which is the only way the denominator bug documented in
vqls_hadamard's module docstring was ever caught in the first place.
"""
from __future__ import annotations

import numpy as np
import pytest

pennylane = pytest.importorskip("pennylane")

from solvers.quantum.vqls_utils import (                    # noqa: E402
    build_ansatz,
    build_cost_function,
    pauli_decompose_matrix,
)
from solvers.quantum.vqls_hadamard import (                  # noqa: E402
    build_hadamard_cost_function,
    circuit_count,
)


def _tst_matrix(N: int, main_diag: float, off_diag: float) -> np.ndarray:
    """
    Local helper reproducing the TST matrix that ``pauli_decompose_tst``
    used to build internally, before ``vqls_utils.pauli_decompose_matrix``
    generalised the decomposition to take an arbitrary matrix directly
    (needed for the pentadiagonal fourth-order operator, which is not TST).
    Kept local to this test file rather than imported from production code,
    since it exists only to reconstruct the specific matrix shape these
    tests were originally validated against.
    """
    return (
        main_diag * np.eye(N)
        + off_diag * np.diag(np.ones(N - 1), k=1)
        + off_diag * np.diag(np.ones(N - 1), k=-1)
    )


# ── Exact-mode equivalence: the central claim ─────────────────────────────────

@pytest.mark.quantum
class TestExactModeMatchesClassicalReference:

    @pytest.mark.parametrize("n_qubits,n_layers,seed", [
        (2, 1, 3),
        (2, 2, 7),
        (3, 2, 42),
    ])
    def test_matches_across_random_parameter_draws(self, n_qubits, n_layers, seed):
        N = 2 ** n_qubits
        pauli_terms = pauli_decompose_matrix(_tst_matrix(N, -2.0, 1.0))

        rng = np.random.default_rng(seed)
        b = rng.normal(size=N)
        b_norm = b / np.linalg.norm(b)

        reference = build_cost_function(pauli_terms, b_norm, n_qubits, n_layers)
        hadamard  = build_hadamard_cost_function(
            pauli_terms, b_norm, n_qubits, n_layers, shots=None
        )

        for _ in range(3):
            params = rng.uniform(0, 2 * np.pi, size=n_qubits * (n_layers + 1))
            c_ref = reference(params)
            c_had = hadamard(params)
            assert c_had == pytest.approx(c_ref, abs=1e-8), (
                f"Hadamard-test cost diverged from the classical reference: "
                f"{c_had} vs {c_ref} at n_qubits={n_qubits}, n_layers={n_layers}"
            )

    def test_denominator_imaginary_part_vanishes(self):
        # Guaranteed by real Pauli coefficients (see module docstring) -- a
        # second, independent check beyond matching the reference cost.
        n_qubits, n_layers = 2, 1
        N = 2 ** n_qubits
        pauli_terms = pauli_decompose_matrix(_tst_matrix(N, -2.0, 1.0))
        rng = np.random.default_rng(11)
        b_norm = rng.normal(size=N)
        b_norm /= np.linalg.norm(b_norm)
        params = rng.uniform(0, 2 * np.pi, size=n_qubits * (n_layers + 1))

        hadamard = build_hadamard_cost_function(
            pauli_terms, b_norm, n_qubits, n_layers,
            shots=None, return_diagnostics=True,
        )
        _cost, diag = hadamard(params)
        assert abs(diag.denominator_imag_residual) < 1e-9


# ── The specific bug this module's docstring documents ───────────────────────

class TestDenominatorAnsatzMustBeUnconditional:
    """
    Regression guard for the exact bug found during development: wrapping
    the ansatz in the controlled block for the denominator's pairwise terms
    (by incorrect analogy with the numerator) silently produces a
    plausible-looking but wrong cost. This test reconstructs the *broken*
    variant inline and confirms it disagrees with the classical reference,
    so nobody re-introduces the same construction believing it to be a
    harmless simplification.
    """

    def test_controlled_ansatz_denominator_variant_disagrees_with_reference(self):
        import pennylane as qml
        from solvers.quantum.vqls_hadamard import _apply_controlled_pauli

        n_qubits, n_layers = 2, 1
        N = 2 ** n_qubits
        pauli_terms = pauli_decompose_matrix(_tst_matrix(N, -2.0, 1.0))
        rng = np.random.default_rng(3)
        b = rng.normal(size=N)
        b_norm = b / np.linalg.norm(b)
        params = rng.uniform(0, 2 * np.pi, size=n_qubits * (n_layers + 1))

        reference = build_cost_function(pauli_terms, b_norm, n_qubits, n_layers)
        c_ref = reference(params)

        device  = qml.device("default.qubit", wires=n_qubits + 1)
        ancilla = n_qubits

        def broken_denominator_term(sl, sl2, part):
            @qml.qnode(device)
            def circuit():
                qml.Hadamard(wires=ancilla)
                # BUG: ansatz wrapped in the controlled block, by incorrect
                # analogy with the numerator. Do not "fix" this test by
                # correcting it -- its whole purpose is to stay broken.
                qml.ctrl(build_ansatz, control=ancilla)(params, n_qubits, n_layers)
                _apply_controlled_pauli(sl2, ancilla)
                _apply_controlled_pauli(sl, ancilla)
                if part == "imag":
                    qml.adjoint(qml.S)(wires=ancilla)
                qml.Hadamard(wires=ancilla)
                return qml.expval(qml.PauliZ(ancilla))
            return float(circuit())

        # Reuse the correct numerator machinery (that part was never buggy)
        # via the real module, only substituting the broken denominator.
        from solvers.quantum.vqls_hadamard import _numerator_hadamard_term
        gamma = []
        for _c, s in pauli_terms:
            re = _numerator_hadamard_term(
                device, ancilla, n_qubits, n_layers, params, s, b_norm, "real", None
            )
            im = _numerator_hadamard_term(
                device, ancilla, n_qubits, n_layers, params, s, b_norm, "imag", None
            )
            gamma.append(re + 1j * im)
        numerator = abs(sum(c * g for (c, _s), g in zip(pauli_terms, gamma))) ** 2

        denom = 0j
        for cl, sl in pauli_terms:
            for cl2, sl2 in pauli_terms:
                re = broken_denominator_term(sl, sl2, "real")
                im = broken_denominator_term(sl, sl2, "imag")
                denom += cl * cl2 * (re + 1j * im)
        denom_real = float(np.real(denom))
        c_broken = 1.0 if denom_real < 1e-14 else float(1.0 - numerator / denom_real)

        assert abs(c_broken - c_ref) > 1e-3, (
            "expected the controlled-ansatz denominator variant to disagree "
            "with the classical reference; if this now passes, something "
            "about the ansatz or Pauli decomposition changed in a way that "
            "coincidentally masks the bug -- investigate before assuming "
            "this test is simply obsolete"
        )


# ── circuit_count() ────────────────────────────────────────────────────────────

class TestCircuitCount:

    @pytest.mark.parametrize("n_terms,expected", [
        (1, 2 + 2),
        (4, 8 + 32),
        (8, 16 + 128),
    ])
    def test_matches_formula(self, n_terms, expected):
        assert circuit_count(n_terms) == expected

    def test_matches_actual_number_of_circuit_evaluations(self, monkeypatch):
        n_qubits, n_layers = 2, 1
        N = 2 ** n_qubits
        pauli_terms = pauli_decompose_matrix(_tst_matrix(N, -2.0, 1.0))
        rng = np.random.default_rng(0)
        b_norm = rng.normal(size=N)
        b_norm /= np.linalg.norm(b_norm)
        params = rng.uniform(0, 2 * np.pi, size=n_qubits * (n_layers + 1))

        import solvers.quantum.vqls_hadamard as vh
        calls = {"n": 0}
        real_num = vh._numerator_hadamard_term
        real_den = vh._denominator_hadamard_term

        def counting_num(*args, **kwargs):
            calls["n"] += 1
            return real_num(*args, **kwargs)

        def counting_den(*args, **kwargs):
            calls["n"] += 1
            return real_den(*args, **kwargs)

        monkeypatch.setattr(vh, "_numerator_hadamard_term", counting_num)
        monkeypatch.setattr(vh, "_denominator_hadamard_term", counting_den)

        cost_fn = vh.build_hadamard_cost_function(
            pauli_terms, b_norm, n_qubits, n_layers, shots=None
        )
        cost_fn(params)
        assert calls["n"] == circuit_count(len(pauli_terms))


# ── Shot-based evaluation ─────────────────────────────────────────────────────

@pytest.mark.quantum
class TestShotBasedEvaluation:

    def test_shots_introduce_variance(self):
        n_qubits, n_layers = 2, 1
        N = 2 ** n_qubits
        pauli_terms = pauli_decompose_matrix(_tst_matrix(N, -2.0, 1.0))
        rng = np.random.default_rng(5)
        b_norm = rng.normal(size=N)
        b_norm /= np.linalg.norm(b_norm)
        params = rng.uniform(0, 2 * np.pi, size=n_qubits * (n_layers + 1))

        hadamard = build_hadamard_cost_function(
            pauli_terms, b_norm, n_qubits, n_layers, shots=200
        )
        values = [hadamard(params) for _ in range(5)]
        assert len(set(values)) > 1, (
            "shot-based evaluation returned identical values across repeated "
            "calls -- shots may be silently falling back to analytic mode"
        )

    def test_shot_mean_converges_towards_exact_value(self):
        # Law-of-large-numbers check: averaging many shot-based evaluations
        # should approach the exact cost. Loose tolerance -- this is a
        # statistical sanity check, not a precision test.
        n_qubits, n_layers = 2, 1
        N = 2 ** n_qubits
        pauli_terms = pauli_decompose_matrix(_tst_matrix(N, -2.0, 1.0))
        rng = np.random.default_rng(9)
        b_norm = rng.normal(size=N)
        b_norm /= np.linalg.norm(b_norm)
        params = rng.uniform(0, 2 * np.pi, size=n_qubits * (n_layers + 1))

        exact_fn = build_hadamard_cost_function(
            pauli_terms, b_norm, n_qubits, n_layers, shots=None
        )
        c_exact = exact_fn(params)

        shot_fn = build_hadamard_cost_function(
            pauli_terms, b_norm, n_qubits, n_layers, shots=500
        )
        samples = [shot_fn(params) for _ in range(30)]
        assert np.mean(samples) == pytest.approx(c_exact, abs=0.15)

    def test_diagnostics_returned_when_requested(self):
        n_qubits, n_layers = 2, 1
        N = 2 ** n_qubits
        pauli_terms = pauli_decompose_matrix(_tst_matrix(N, -2.0, 1.0))
        rng = np.random.default_rng(1)
        b_norm = rng.normal(size=N)
        b_norm /= np.linalg.norm(b_norm)
        params = rng.uniform(0, 2 * np.pi, size=n_qubits * (n_layers + 1))

        fn = build_hadamard_cost_function(
            pauli_terms, b_norm, n_qubits, n_layers,
            shots=None, return_diagnostics=True,
        )
        cost, diag = fn(params)
        assert isinstance(cost, float)
        assert diag.n_circuits == circuit_count(len(pauli_terms))

    def test_default_return_is_bare_float(self):
        # A plain float is required for direct use with an optimiser
        # expecting params -> float (e.g. scipy.optimize.minimize).
        n_qubits, n_layers = 2, 1
        N = 2 ** n_qubits
        pauli_terms = pauli_decompose_matrix(_tst_matrix(N, -2.0, 1.0))
        rng = np.random.default_rng(2)
        b_norm = rng.normal(size=N)
        b_norm /= np.linalg.norm(b_norm)
        params = rng.uniform(0, 2 * np.pi, size=n_qubits * (n_layers + 1))

        fn = build_hadamard_cost_function(pauli_terms, b_norm, n_qubits, n_layers)
        result = fn(params)
        assert isinstance(result, float)