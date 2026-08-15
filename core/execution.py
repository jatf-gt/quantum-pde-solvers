"""
Execution backend abstraction for quantum solver state extraction.

Motivation
----------
Every quantum solver in this project ends the same way: a circuit is built,
evolved, and a solution vector is recovered from the amplitudes of a *data*
register, conditional on a set of *ancilla* registers being found in a
prescribed state. Until now that final step was performed inline in two
places — ``hhl_1d._extract_solution`` and ``qsvt_1d._extract_solution`` —
each calling ``Statevector(circuit).data`` directly and slicing the result.

Exact statevector slicing is optimal for classical simulation and forms the basis
for all baseline results in the thesis. It is, however, physically unrealisable on
quantum hardware. On physical devices, the equivalent operation is:

    prepare → measure ancillas → discard shots that failed post-selection
            → reconstruct the surviving data-register amplitudes by tomography

This represents a distinct computational process with associated costs and error
models. This module abstracts the operation, enabling the simulator and the
device to act as two implementations of a unified interface rather than
independent code paths.

The post-selection seam
-----------------------
Post-selection is the right place to cut, and it is worth being explicit
about why. It is not merely the last step before the answer; it is the step
whose cost separates simulation from execution:

*   On a statevector, post-selection is a *mask*. Rejected amplitudes are
    skipped. The operation is free and the success probability is never
    consulted.

*   On hardware, post-selection is *rejection sampling*. A run that succeeds
    with probability p requires O(1/p) shots to yield the same statistics.
    For HHL, p ~ 1/κ², so the shot overhead grows quadratically in the
    condition number — a cost that is invisible in simulation and dominant
    on a device.

``StatevectorExecutor`` therefore reports ``postselect_probability`` even
though it does not need it, because that single number is the bridge between
the simulated result and its hardware cost, and it is available for free.

Guarantee
---------
``StatevectorExecutor`` is the default and reproduces the previous inline
code exactly — same masking logic, same component extraction, same failure
diagnostics, same dtype. It is not a reimplementation with equivalent intent
but a transcription. ``tests/test_execution.py`` asserts bit-level equality
against the original routines; ``tests/test_regression_baseline.py`` pins the
solver outputs themselves. Any future executor that changes a baseline number
is a bug in that executor, not a revision of the baseline.

References
----------
Nielsen & Chuang, *Quantum Computation and Quantum Information*, §8.4
    (measurement, post-selection and the associated sampling overhead).
Harrow, Hassidim & Lloyd, Phys. Rev. Lett. 103, 150502 (2009)
    (the 1/κ² post-selection success probability of the HHL ancilla).
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, Sequence, Tuple

import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    "PostSelectSpec",
    "ExecutionRecord",
    "Executor",
    "StatevectorExecutor",
    "default_executor",
    "set_default_executor",
    "execution_context",
]


# -- Post-selection specification ----------------------------------------------

@dataclass(frozen=True)
class PostSelectSpec:
    """
    Declarative description of how a solution vector sits inside a circuit.

    A specification says three things: which qubits carry the answer, what
    the remaining qubits must be measured to be for a run to count, and which
    real component of the surviving amplitudes is the physical solution.

    Qubit indices follow Qiskit's little-endian convention, so index 0 is the
    least significant bit of the integer basis-state label. The data register
    is assumed to occupy indices ``0 … n_data-1``, which is true of every
    circuit built in this project; ``conditions`` then refers to indices at
    or above ``n_data``.

    Attributes
    ----------
    n_data : int
        Number of data-register qubits. The extracted vector has length
        ``2**n_data``.
    conditions : tuple of (int, int)
        Post-selection conditions as ``(qubit_index, required_bit)`` pairs.
        A basis state is accepted only if every listed qubit holds its
        required value. An empty tuple accepts everything.
    component : {'real', 'imag', 'complex'}
        Which part of the accepted amplitudes constitutes the solution.
        HHL uses ``'real'``; QSVT under the Wx convention uses ``'imag'``,
        because the QSP sequence realises Im(P(A/α))|b⟩ ≈ κ⁻¹A⁻¹|b⟩ there.
    label : str
        Diagnostic identifier carried into error messages and logs, e.g.
        ``'HHL'`` or ``'QSVT-row-3'``.

    Notes
    -----
    The component choice is a property of the algorithm's convention, not of
    the backend, so it belongs here rather than in the executor. A hardware
    executor must reconstruct the *complex* amplitudes by tomography and then
    apply the same projection, which is why the field is part of the shared
    specification.
    """

    n_data:     int
    conditions: Tuple[Tuple[int, int], ...] = ()
    component:  str = "real"
    label:      str = ""

    def __post_init__(self) -> None:
        if self.n_data < 1:
            raise ValueError(f"n_data must be >= 1, got {self.n_data}")
        if self.component not in ("real", "imag", "complex"):
            raise ValueError(
                f"component must be 'real', 'imag' or 'complex', "
                f"got {self.component!r}"
            )
        for qubit, bit in self.conditions:
            if qubit < self.n_data:
                raise ValueError(
                    f"post-selection condition on qubit {qubit} collides with "
                    f"the data register (indices 0..{self.n_data - 1}). The "
                    f"data register cannot be post-selected on."
                )
            if bit not in (0, 1):
                raise ValueError(
                    f"required bit must be 0 or 1, got {bit} on qubit {qubit}"
                )

    # -- Derived bit-masks -----------------------------------------------------

    @property
    def mask(self) -> int:
        """Bit-mask selecting every post-selected qubit."""
        m = 0
        for qubit, _ in self.conditions:
            m |= (1 << qubit)
        return m

    @property
    def target(self) -> int:
        """Required value of the masked bits for a basis state to be accepted."""
        t = 0
        for qubit, bit in self.conditions:
            if bit:
                t |= (1 << qubit)
        return t

    def accepts(self, index: int) -> bool:
        """True if basis state ``index`` survives post-selection."""
        return (index & self.mask) == self.target

    # -- Convenience constructors ----------------------------------------------

    @classmethod
    def from_registers(
        cls,
        n_data:     int,
        zeroed:     Sequence[int] = (),
        ones:       Sequence[int] = (),
        component:  str = "real",
        label:      str = "",
    ) -> "PostSelectSpec":
        """
        Build a specification from two lists of qubit indices.

        Parameters
        ----------
        n_data : int
            Size of the data register.
        zeroed : sequence of int
            Qubit indices required to be |0⟩ — cleared clock registers,
            returned MCMT ancillae, block encoding ancillae.
        ones : sequence of int
            Qubit indices required to be |1⟩ — the HHL eigenvalue-inversion
            flag being the only instance in this project.
        component : {'real', 'imag', 'complex'}
        label : str
        """
        conditions = tuple(
            [(int(q), 0) for q in zeroed] + [(int(q), 1) for q in ones]
        )
        return cls(
            n_data     = n_data,
            conditions = conditions,
            component  = component,
            label      = label,
        )


# -- Execution diagnostics -----------------------------------------------------

@dataclass
class ExecutionRecord:
    """
    Provenance and cost metadata for one state extraction.

    Every field is optional because not every backend can supply every
    quantity, and a field that a backend cannot honestly fill is left
    ``None`` rather than given a plausible-looking default. Anything written
    into a results file must be traceable to something that was actually
    measured or computed.

    Attributes
    ----------
    mode : str
        Execution mode: ``'statevector'``, ``'sampled'`` or ``'hardware'``.
    backend : str or None
        Backend identifier. ``None`` for exact statevector evolution, which
        uses no backend object at all.
    n_qubits : int
        Total circuit width.
    circuit_depth : int or None
        Depth as constructed, before transpilation. Comparable across
        backends; not a hardware cost. See ``core.resources`` (Phase 2) for
        post-transpilation figures, which are the ones a device cares about.
    shots : int or None
        Shots requested. ``None`` under exact evolution.
    postselect_probability : float or None
        Probability that a single run survives post-selection, i.e. the
        summed Born weight of the accepted subspace. On hardware the
        expected shot overhead to obtain one accepted sample is its
        reciprocal. For HHL this tracks 1/κ².
    n_accepted : int or None
        Shots that actually survived post-selection. ``None`` under exact
        evolution, where no sampling occurs.
    extra : dict
        Backend-specific fields — mitigation settings, calibration
        timestamps, job identifiers — kept out of the typed surface so that
        adding a backend never requires editing this dataclass.
    """

    mode:                    str
    n_qubits:                int
    backend:                 Optional[str]   = None
    circuit_depth:           Optional[int]   = None
    shots:                   Optional[int]   = None
    postselect_probability:  Optional[float] = None
    n_accepted:              Optional[int]   = None
    extra:                   Dict[str, Any]  = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        """Flat JSON-serialisable mapping, for ``benchmark.results_io``."""
        out = {
            "mode":                   self.mode,
            "backend":                self.backend,
            "n_qubits":               self.n_qubits,
            "circuit_depth":          self.circuit_depth,
            "shots":                  self.shots,
            "postselect_probability": self.postselect_probability,
            "n_accepted":             self.n_accepted,
        }
        out.update({f"exec_{k}": v for k, v in self.extra.items()})
        return out

    @property
    def shot_overhead(self) -> Optional[float]:
        """
        Expected runs per accepted sample, 1/p.

        This is the quantity that makes a simulated HHL result and its
        hardware cost commensurable, and it is the headline number of the
        post-selection cost study.
        """
        p = self.postselect_probability
        if p is None or p <= 0.0:
            return None
        return 1.0 / p


# -- Executor interface --------------------------------------------------------

class Executor(Protocol):
    """
    Interface every execution backend implements.

    Deliberately narrow. An executor answers exactly one question — *given
    this circuit and this description of where the answer lives, what is the
    solution vector and what did obtaining it cost* — and knows nothing about
    Poisson problems, proportionality recovery or outer iterations. That
    separation is what allows a hardware backend to be added without any
    solver being aware of it.
    """

    mode: str

    def extract(
        self,
        circuit: Any,
        spec:    PostSelectSpec,
    ) -> Tuple[np.ndarray, ExecutionRecord]:
        """
        Recover the post-selected data-register vector from ``circuit``.

        Returns
        -------
        x_raw : np.ndarray, shape (2**spec.n_data,)
            Real-valued unless ``spec.component == 'complex'``. Unnormalised:
            proportionality recovery remains the caller's responsibility, as
            it is algorithm-specific.
        record : ExecutionRecord
        """
        ...


# -- Exact statevector executor ------------------------------------------------

class StatevectorExecutor:
    """
    Exact statevector evolution — the project default and the thesis baseline.

    Reproduces the previous inline extraction in ``hhl_1d`` and ``qsvt_1d``
    exactly. No sampling, no noise, no transpilation: the circuit is evolved
    as an exact unitary and the accepted amplitudes are read off directly.
    This is the noiseless upper bound against which every noisy or hardware
    result is to be compared, and it is the mode in which all replication
    runs must be performed.

    Parameters
    ----------
    zero_atol : float
        Threshold below which the extracted vector is judged null, triggering
        the post-selection diagnostic dump. Matches the previous inline value
        of 1e-12 in both call sites.
    diagnostics : bool
        Emit the dominant-amplitude table when extraction fails. Previously
        unconditional in HHL; retained as the default so that failure output
        is unchanged.
    """

    mode = "statevector"

    def __init__(self, zero_atol: float = 1e-12, diagnostics: bool = True):
        self.zero_atol   = zero_atol
        self.diagnostics = diagnostics

    def extract(
        self,
        circuit: Any,
        spec:    PostSelectSpec,
    ) -> Tuple[np.ndarray, ExecutionRecord]:
        from qiskit.quantum_info import Statevector

        sv      = np.asarray(Statevector(circuit).data, dtype=complex)
        n_total = circuit.num_qubits

        amplitudes, prob = self._mask(sv, spec, n_total)
        x_raw            = _project(amplitudes, spec.component)

        record = ExecutionRecord(
            mode                   = self.mode,
            n_qubits               = n_total,
            backend                = None,
            circuit_depth          = circuit.depth(),
            shots                  = None,
            postselect_probability = prob,
            n_accepted             = None,
        )

        if _is_null(x_raw, self.zero_atol):
            if self.diagnostics:
                _dump_amplitudes(sv, spec, circuit)
            raise RuntimeError(
                f"{spec.label or 'Solver'} extraction returned a null vector "
                f"under post-selection.\n"
                f"  component        : {spec.component}\n"
                f"  conditions       : {list(spec.conditions)}\n"
                f"  accepted weight  : {prob:.3e}\n"
                f"  n_data / n_total : {spec.n_data} / {n_total}"
            )

        return x_raw, record

    # -- Internals -------------------------------------------------------------

    @staticmethod
    def _mask(
        sv:      np.ndarray,
        spec:    PostSelectSpec,
        n_total: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Collapse the full statevector onto the accepted data-register subspace.

        Equivalent to the explicit loops previously written inline, but
        vectorised. The accepted Born weight is accumulated alongside, which
        the inline versions did not do and which costs nothing here.
        """
        N        = 2 ** spec.n_data
        indices  = np.arange(2 ** n_total, dtype=np.int64)
        accepted = (indices & spec.mask) == spec.target

        amplitudes = np.zeros(N, dtype=complex)
        data_index = indices[accepted] & (N - 1)
        amplitudes[data_index] = sv[accepted]

        prob = float(np.sum(np.abs(sv[accepted]) ** 2))
        return amplitudes, prob


# -- Shared helpers, reused by every executor ----------------------------------

def _project(amplitudes: np.ndarray, component: str) -> np.ndarray:
    """Take the algorithm-specified component of the accepted amplitudes."""
    if component == "real":
        return np.real(amplitudes)
    if component == "imag":
        return np.imag(amplitudes)
    return amplitudes


def _is_null(x: np.ndarray, atol: float) -> bool:
    return bool(np.allclose(x, 0.0, atol=atol))


def _dump_amplitudes(
    sv:      np.ndarray,
    spec:    PostSelectSpec,
    circuit: Any,
    top:     int = 10,
) -> None:
    """
    Print the dominant statevector amplitudes and their post-selection status.

    Preserves the diagnostic that HHL previously emitted inline, generalised
    to any specification. The point of the table is to show *which* register
    failed to clear, so the per-condition breakdown matters more than the
    amplitudes themselves.
    """
    regs = [(r.name, r.size) for r in circuit.qregs]
    print(
        f"\n  {spec.label or 'Solver'} post-selection diagnostics — "
        f"dominant amplitudes by magnitude:"
    )
    print(f"    registers: {regs}")

    magnitudes = np.abs(sv)
    for idx in np.argsort(magnitudes)[::-1][:top]:
        idx  = int(idx)
        bits = " ".join(
            f"q{q}={(idx >> q) & 1}(want {want})"
            for q, want in spec.conditions
        )
        status = "accept" if spec.accepts(idx) else "reject"
        print(
            f"    idx={idx:6d}  |amp|={magnitudes[idx]:.6f}  "
            f"data={idx & ((1 << spec.n_data) - 1):4d}  "
            f"[{status}]  {bits}"
        )


# -- Default executor, with scoped override ------------------------------------

_DEFAULT: Executor = StatevectorExecutor()


def default_executor() -> Executor:
    """
    The executor used when a caller supplies none.

    Exact statevector evolution unless deliberately changed, so that any code
    path not explicitly opted into a noisy or hardware backend produces
    baseline numbers.
    """
    return _DEFAULT


def set_default_executor(executor: Executor) -> Executor:
    """
    Replace the process-wide default, returning the previous one.

    Prefer :func:`execution_context` in scripts. Direct use is intended for
    HPC runner entry points, where the mode is fixed once from the command
    line before any solve begins.
    """
    global _DEFAULT
    previous, _DEFAULT = _DEFAULT, executor
    log.info("Default executor set to mode=%r", getattr(executor, "mode", "?"))
    return previous


@contextmanager
def execution_context(executor: Executor):
    """
    Temporarily install ``executor`` as the default.

    A context manager rather than a threaded-through configuration field, and
    that is a deliberate trade. Threading an executor argument through
    ``qsvt_solve → qsvt_solve_system → _extract_solution`` and the matching
    HHL and outer-scheme call chains would mean editing a dozen signatures
    that the baseline regression tests currently pin — precisely the churn
    that risks perturbing results the thesis depends on. Scoping the default
    instead keeps every existing signature and every existing default intact.

    The cost is process-global state, which is unsafe to mutate concurrently.
    That matters here because the 2-D and 3-D solvers parallelise strip solves
    across a ``ProcessPoolExecutor``. It is safe as used: each worker is a
    separate process with its own module state, and the mode is set once at
    entry rather than switched mid-run. Do not set the default from inside a
    worker or from a thread.

    Examples
    --------
    >>> from core.execution import execution_context, StatevectorExecutor
    >>> with execution_context(StatevectorExecutor(diagnostics=False)):
    ...     result = qsvt_solve(problem)
    """
    previous = set_default_executor(executor)
    try:
        yield executor
    finally:
        set_default_executor(previous)


# -- Specification builders for the solvers in this project --------------------

def hhl_spec(circuit: Any, num_qubits: int, label: str = "HHL") -> PostSelectSpec:
    """
    Post-selection specification for an HHL output circuit.

    Register layout, from ``circuit.qregs`` in Qiskit little-endian order:

        qregs[0] : b-register (solution), n_b qubits
        qregs[1] : l-register (clock),    n_l qubits
        qregs[2] : MCMT ancillae,         n_a qubits
        qregs[3] : flag qubit,            1 qubit, most significant

    A run counts only if the flag is |1⟩ — the eigenvalue inversion succeeded
    — and the clock and MCMT ancillae have returned to |0…0⟩, so that the
    data register is unentangled from them. The solution is the real part.

    The success probability of this condition scales as 1/κ², which is the
    dominant hardware cost of HHL and is recorded by the executor.
    """
    n_total   = circuit.num_qubits
    n_b       = circuit.qregs[0].size
    n_l       = circuit.qregs[1].size
    n_ancilla = n_total - 1 - n_b - n_l

    if n_b != num_qubits:
        raise ValueError(
            f"b-register size {n_b} does not match num_qubits={num_qubits}"
        )
    if n_ancilla < 0:
        raise ValueError(
            f"inconsistent register layout: n_total={n_total}, n_b={n_b}, "
            f"n_l={n_l} leaves {n_ancilla} ancilla qubits"
        )

    clock_bits   = range(n_b, n_b + n_l)
    ancilla_bits = range(n_b + n_l, n_b + n_l + n_ancilla)

    return PostSelectSpec.from_registers(
        n_data    = n_b,
        zeroed    = [*clock_bits, *ancilla_bits],
        ones      = [n_total - 1],
        component = "real",
        label     = label,
    )


def qsvt_spec(n: int, n_a: int = 1, label: str = "QSVT") -> PostSelectSpec:
    """
    Post-selection specification for a QSVT output circuit.

    Post-selects the block encoding ancilla on |0⟩ and returns the imaginary
    part. Under the Wx convention used by ``qsp_angles``, the QSP sequence
    realises Im(P(A/α))|b⟩ ≈ κ_eff⁻¹ A⁻¹|b⟩, so the imaginary part — not the
    real part — is the solution.

    Parameters
    ----------
    n : int
        Data-register qubits; N = 2ⁿ.
    n_a : int
        Block encoding ancilla qubits. One, for the Sz.-Nagy dilation.
    """
    return PostSelectSpec.from_registers(
        n_data    = n,
        zeroed    = range(n, n + n_a),
        component = "imag",
        label     = label,
    )