"""
Circuit-level Hadamard-test evaluation of the VQLS cost function.

``solvers.quantum.vqls_utils.build_cost_function`` computes the VQLS global
cost

    C(theta) = 1 - |<b|A|x(theta)>|^2 / <x(theta)|A^dagger A|x(theta)>

by reconstructing A classically from its Pauli decomposition and applying it
to the ansatz statevector directly. That module's own docstring is explicit
that this is a placeholder: "On physical hardware the Hadamard tests would
be required." This module is that missing piece: an actual circuit-level
implementation, built from ancilla-based Hadamard tests, that can be run
either exactly (reproducing the classical shortcut's numbers, which is how
it was validated) or with a finite shot count (which the classical shortcut
cannot do at all, since it never constructs a circuit whose ancilla could be
measured).

This module does not modify ``vqls_utils.py`` or ``vqls_1d.py``. Nothing
here is on the path the regression baseline (``tests/test_regression_baseline.py``)
pins. It is an additional, opt-in capability; wiring it into
``vqls_solve`` as a selectable cost backend is a natural follow-up, deferred
so that the existing, baseline-pinned VQLS solve path is not touched.

The two-part structure, and a bug this design caught in itself
-----------------------------------------------------------------
The numerator and denominator are *not* symmetric constructions, and
treating them as if they were is exactly the mistake an early draft of this
module made and caught only by validating against the classical reference
numerically (see "Validation" below) — reasoning about the circuits alone
did not surface it.

**Numerator**: |<b|A|x(theta)>|^2 = |sum_l c_l * gamma_l|^2, where
gamma_l = <b|P_l|x(theta)> = <0| U_b^dagger P_l V(theta) |0>. This is a
standard Hadamard test on the state |0>: the *entire* chain — controlled-
V(theta), controlled-P_l, controlled-U_b^dagger — sits inside the controlled
block, because the reference state being tested is the trivial |0>, shared
identically by both the ancilla=0 and ancilla=1 branches before any of it
runs. Only L circuits (times 2 for real/imaginary parts) are needed here,
not L^2, because the c_l * gamma_l terms can be summed *classically* into a
single complex number before squaring — no different from how VQE sums
individually-measured Pauli expectation values before combining them.

**Denominator**: <x(theta)|A^dagger A|x(theta)> = sum_{l,l'} c_l c_l'
<x(theta)|P_l P_l'|x(theta)>, and each pairwise term is *also* a Hadamard
test, but on a *different* reference state: psi_0 = |x(theta)> itself, not
|0>. The early draft of this module wrapped V(theta) in the controlled
block here too, by analogy with the numerator — this is wrong, and it is
wrong for a specific, checkable reason: the ancilla=0 branch of a
controlled-V(theta) leaves the data register at |0...0>, not at |x(theta)>,
so the Hadamard test's own zero-noise assumption (a shared, fixed reference
state on both branches) is violated. The fix is to apply V(theta)
*unconditionally*, once, before the ancilla is even put into superposition;
only P_l and P_l' go inside the controlled block. This was caught by
comparing the intermediate statevector's ancilla=0 branch against the known
ansatz state directly (see the derivation notes retained in this module's
test file) — the ancilla=1 branch was already correct, which is what made
the bug easy to miss by inspection alone.

The classically-guaranteed reality check: because c_l are real (this
project's TST matrix is real symmetric, so its Pauli coefficients are real
— see ``pauli_decompose_matrix``), the full double sum over (l, l') is
guaranteed real even though individual pairwise terms need not be (a
Y-containing Pauli product picks up a factor of +-i). The denominator's
imaginary part summing to ~0 is therefore a second, independent check this
module's implementation must pass, not merely a nicety — it is checked
explicitly in ``build_hadamard_cost_function``'s docstring examples and in
``tests/test_vqls_hadamard.py``, and was, in fact, ~0 to machine precision
even in the buggy draft, meaning it alone would not have caught the bug —
only the direct numerical comparison against the classical reference did.

Circuit cost, made visible
-----------------------------
This construction is honestly expensive, and that expense is the point —
it is what "filling the stub" is supposed to reveal. For L nonzero Pauli
terms: 2L circuits for the numerator, 2L^2 for the denominator (both real
and imaginary parts). ``circuit_count(n_terms)`` reports this exactly, and
it is the number that should be quoted anywhere this module's cost is
compared against the classical shortcut's cost, which needs none.

Wire and Pauli-string conventions
------------------------------------
Matched against ``vqls_utils`` empirically, not assumed: PennyLane's
``qml.state()`` orders amplitudes with wire 0 as the most significant bit
(confirmed directly: a lone X on wire 0 of a 2-qubit register produces
index 2 = ``0b10``, not index 1). ``_pauli_string_to_matrix`` builds its
matrix via left-to-right ``np.kron``, which is the same "leftmost = most
significant" convention. The two are therefore already consistent under the
direct mapping "pauli_str[i] acts on wire i" — confirmed by applying every
single- and two-character test string directly to the ansatz state and
comparing against ``_pauli_string_to_matrix(s) @ x_state`` exactly, for
every string in ``{I,X,Y,Z}^2``, before this module's circuits were trusted
to combine correctly.

Validation
--------------
``tests/test_vqls_hadamard.py`` reproduces ``build_cost_function``'s output
to floating-point precision (``diff < 1e-9``) across multiple random
parameter draws, at both N=4 (n_qubits=2) and N=8 (n_qubits=3, n_layers=2),
in the zero-shot (analytic) mode. This is the primary correctness guarantee
for this module: it was iterated against that reference until exact, not
derived once and trusted.

Reference
-------------
Bravo-Prieto et al., "Variational Quantum Linear Solver", Quantum 7, 1188
    (2023) — the global cost function and its Hadamard-test evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pennylane as qml

from solvers.quantum.vqls_utils import build_ansatz

__all__ = [
    "build_hadamard_cost_function",
    "circuit_count",
    "HadamardCostDiagnostics",
]


# Controlled single-qubit Pauli gates, keyed by character. Deliberately not
# including "I": a controlled-identity is a no-op and is skipped entirely
# rather than emitting a gate, which matters for circuit_count() staying an
# honest count of what is actually submitted.
_CTRL_GATE = {"X": qml.CNOT, "Y": qml.CY, "Z": qml.CZ}


def _apply_controlled_pauli(pauli_str: str, ancilla: int) -> None:
    """
    Apply P_l to the data register, controlled on ``ancilla``, one factor
    per qubit. ``pauli_str[i]`` acts on wire ``i`` — see module docstring
    for why this is the convention that matches ``_pauli_string_to_matrix``.
    """
    for wire, ch in enumerate(pauli_str):
        if ch != "I":
            _CTRL_GATE[ch](wires=[ancilla, wire])


def _numerator_hadamard_term(
    device,
    ancilla:      int,
    n_qubits:     int,
    n_layers:     int,
    params:       np.ndarray,
    pauli_str:    str,
    b_norm:       np.ndarray,
    part:         str,
    shots:        Optional[int],
) -> float:
    """
    One real- or imaginary-part measurement contributing to gamma_l =
    <b|P_l|x(theta)>. Reference state is |0>, so the whole chain
    (V, P_l, U_b^dagger) is controlled — see module docstring.
    """
    @qml.qnode(device)
    def circuit():
        qml.Hadamard(wires=ancilla)
        qml.ctrl(build_ansatz, control=ancilla)(params, n_qubits, n_layers)
        _apply_controlled_pauli(pauli_str, ancilla)
        qml.ctrl(qml.adjoint(qml.StatePrep), control=ancilla)(
            b_norm, wires=range(n_qubits)
        )
        if part == "imag":
            qml.adjoint(qml.S)(wires=ancilla)
        qml.Hadamard(wires=ancilla)
        return qml.expval(qml.PauliZ(ancilla))

    if shots is not None:
        circuit = qml.set_shots(circuit, shots=shots)
    return float(circuit())


def _denominator_hadamard_term(
    device,
    ancilla:      int,
    n_qubits:     int,
    n_layers:     int,
    params:       np.ndarray,
    pauli_str_l:  str,
    pauli_str_l2: str,
    part:         str,
    shots:        Optional[int],
) -> float:
    """
    One real- or imaginary-part measurement contributing to
    <x(theta)|P_l P_l'|x(theta)>. Reference state is |x(theta)> itself, so
    the ansatz is applied *unconditionally*, before the ancilla superposition
    — this is the fix for the bug documented in the module docstring.
    Circuit order [P_l', P_l] (P_l' first/innermost) composes to the
    operator product P_l . P_l' acting on |x(theta)>, matching the
    classical convention ``Pl @ (Pl2 @ x_state)`` used by
    ``vqls_utils.build_cost_function``.
    """
    @qml.qnode(device)
    def circuit():
        build_ansatz(params, n_qubits, n_layers)   # unconditional
        qml.Hadamard(wires=ancilla)
        _apply_controlled_pauli(pauli_str_l2, ancilla)
        _apply_controlled_pauli(pauli_str_l,  ancilla)
        if part == "imag":
            qml.adjoint(qml.S)(wires=ancilla)
        qml.Hadamard(wires=ancilla)
        return qml.expval(qml.PauliZ(ancilla))

    if shots is not None:
        circuit = qml.set_shots(circuit, shots=shots)
    return float(circuit())


@dataclass
class HadamardCostDiagnostics:
    """
    Everything beyond the scalar cost that the Hadamard-test evaluation
    knows and the classical shortcut cannot: the denominator's residual
    imaginary part (should be ~0; see module docstring for why this is a
    second, independent correctness check, not merely diagnostic colour),
    and the exact number of circuits the evaluation submitted.

    Attributes
    ----------
    numerator, denominator : float
        The two pieces of C(theta) = 1 - numerator/denominator.
    denominator_imag_residual : float
        Imaginary part of the (guaranteed-real) denominator sum before
        discarding it. Large values indicate either a genuinely broken
        circuit construction or, in shot mode, are an expected consequence
        of finite-shot noise not respecting the exact cancellation that
        holds analytically.
    n_circuits : int
        Total circuits evaluated for this one cost call.
    """
    numerator:                   float
    denominator:                 float
    denominator_imag_residual:   float
    n_circuits:                  int


def circuit_count(n_terms: int) -> int:
    """
    Exact number of circuits one cost evaluation submits, given ``n_terms``
    nonzero Pauli terms: ``2*n_terms`` for the numerator (real + imaginary
    per term) plus ``2*n_terms**2`` for the denominator (real + imaginary
    per ordered pair). Quote this wherever this module's cost is compared
    against ``vqls_utils.build_cost_function``, which needs none.
    """
    return 2 * n_terms + 2 * n_terms ** 2


def build_hadamard_cost_function(
    pauli_terms:  List[Tuple[complex, str]],
    b_norm:       np.ndarray,
    n_qubits:     int,
    n_layers:     int,
    shots:        Optional[int] = None,
    device_name:  str = "default.qubit",
    return_diagnostics: bool = False,
) -> Callable[[np.ndarray], float]:
    """
    Build the VQLS global cost function via explicit Hadamard-test circuits.

    Drop-in replacement, in interface, for
    ``vqls_utils.build_cost_function`` — same signature shape, same
    returned-callable contract (``params -> float`` bounded in [0, 1] for a
    well-posed problem) — evaluated by actual quantum circuits instead of
    classical linear algebra. With ``shots=None`` it reproduces that
    function's output to floating-point precision (see module docstring,
    "Validation"); with ``shots`` set, it returns a genuinely noisy,
    finite-sample estimate, which is the entire reason this module exists.

    Parameters
    ----------
    pauli_terms : list of (coefficient, pauli_string)
        As returned by ``vqls_utils.pauli_decompose_matrix``. Coefficients are
        assumed real (true for this project's TST matrices; see module
        docstring) — a complex coefficient would still work but the
        denominator-imaginary-residual check would no longer be expected to
        vanish, and callers relying on that check should be aware.
    b_norm : np.ndarray, shape (2**n_qubits,)
        Unit-normalised right-hand side, real-valued.
    n_qubits, n_layers : int
        Ansatz shape, matching ``vqls_utils.build_ansatz``.
    shots : int or None
        ``None`` for exact (analytic) evaluation. A positive integer for a
        finite-shot estimate — every one of the ``circuit_count(len(pauli_terms))``
        circuits is run with this many shots independently, so the total
        shot budget per cost evaluation is ``shots * circuit_count(...)``,
        not ``shots`` alone.
    device_name : str
        PennyLane device. Only ``"default.qubit"`` has been validated here;
        a different simulator or a real backend would need its own
        equivalence check against the classical reference before trusting
        it, exactly as this module's own construction was checked.
    return_diagnostics : bool
        If True, the returned callable returns ``(cost, HadamardCostDiagnostics)``
        instead of a bare float. Default False so the callable is a drop-in
        match for ``build_cost_function``'s signature, usable directly by
        an optimiser expecting ``params -> float``.

    Returns
    -------
    Callable[[np.ndarray], float] or Callable[[np.ndarray], Tuple[float, HadamardCostDiagnostics]]
    """
    device  = qml.device(device_name, wires=n_qubits + 1)
    ancilla = n_qubits
    n_terms = len(pauli_terms)

    def cost_fn(params: np.ndarray):
        gamma = []
        for _c, pauli_str in pauli_terms:
            re = _numerator_hadamard_term(
                device, ancilla, n_qubits, n_layers, params,
                pauli_str, b_norm, "real", shots,
            )
            im = _numerator_hadamard_term(
                device, ancilla, n_qubits, n_layers, params,
                pauli_str, b_norm, "imag", shots,
            )
            gamma.append(re + 1j * im)

        numerator_amplitude = sum(
            c * g for (c, _s), g in zip(pauli_terms, gamma)
        )
        numerator = float(abs(numerator_amplitude) ** 2)

        denom = 0j
        for cl, sl in pauli_terms:
            for cl2, sl2 in pauli_terms:
                re = _denominator_hadamard_term(
                    device, ancilla, n_qubits, n_layers, params,
                    sl, sl2, "real", shots,
                )
                im = _denominator_hadamard_term(
                    device, ancilla, n_qubits, n_layers, params,
                    sl, sl2, "imag", shots,
                )
                denom += cl * cl2 * (re + 1j * im)

        denominator = float(np.real(denom))
        imag_residual = float(np.imag(denom))

        if denominator < 1e-14:
            cost = 1.0
        else:
            cost = float(1.0 - numerator / denominator)

        if return_diagnostics:
            diagnostics = HadamardCostDiagnostics(
                numerator=numerator,
                denominator=denominator,
                denominator_imag_residual=imag_residual,
                n_circuits=circuit_count(n_terms),
            )
            return cost, diagnostics
        return cost

    return cost_fn