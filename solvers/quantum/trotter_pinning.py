"""
Trotter-step pinning for the vendored Hamiltonian-simulation matrix classes.

Purpose
-------
The matrix classes in `quantum_linear_solvers.linear_solvers.matrices` derive
their Trotter step count from the error tolerance, and they re-derive it inside
the `evolution_time` setter:

    @evolution_time.setter
    def evolution_time(self, evolution_time):
        self._evolution_time = evolution_time
        self.trotter_steps = ceil(sqrt((evolution_time·|b|)³ / 2 / tolerance))

`HHL.solve` assigns `matrix_circuit.tolerance = self._epsilon_a` and then
`matrix_circuit.evolution_time = …`, in that order, on every solve. Any step
count supplied to the constructor is therefore **discarded before a single gate
is built**, silently: the object still reports the value it was given until the
first solve, and the solve returns a result identical to one obtained with any
other value.

This was not a hypothesis. At N = 4 and N = 8, step counts of 10 through 1000
returned solution vectors identical bit-for-bit, with flat wall time.

What this module provides
-------------------------
`pin_trotter_steps(matrix, n)` returns a matrix object whose step count survives
`HHL.solve`. It is implemented by binding a subclass that overrides the
`evolution_time` setter to record the new evolution time *without* re-deriving
the step count, leaving every other behaviour of the vendored class untouched.

Why a subclass rather than a patch to the vendored library
----------------------------------------------------------
The vendored tree is third-party code, patched only for compatibility, and its
derivation is correct for its own purpose: given a tolerance, it computes the
step count that meets it, which is what a caller who supplies only a tolerance
wants. The defect is that a caller who supplies a *count* cannot be heard. A
subclass adds that channel without changing what the library does for anyone
else, and leaves the vendored file free to be re-synced from upstream.

Which knob to use
-----------------
Two routes exist to the same quantity and they are not interchangeable:

  `HHL(epsilon=ε)`        The algorithm's overall precision. The library splits
                          it into ε_r (reciprocal rotation), ε_s (state
                          preparation) and ε_a = ε/6 (Hamiltonian simulation),
                          and derives the step count from ε_a. This is the
                          physically meaningful parameter and it needs no pinning
                          — it was simply never passed. Use it wherever the
                          quantity of interest is a target accuracy.

  `pin_trotter_steps`     Fixes the step count exactly, overriding whatever ε
                          implies. Use it wherever the step count is itself the
                          independent variable, as in a sensitivity sweep over
                          Hamiltonian-simulation depth at fixed ε.

Measured at N = 4 on the second-order operator, ε ∈ {1e-1, 1e-2, 1e-3} gives
step counts {3, 7, 21} and relative residuals {2.40e-2, 1.79e-2, 1.65e-2}. The
ε = 1e-2 row reproduces the archived sweeps exactly, because 1e-2 is also the
library default that those sweeps ran under.

References
----------
  Harrow, A. W., Hassidim, A. & Lloyd, S. (2009). Quantum algorithm for linear
      systems of equations. Phys. Rev. Lett. 103, 150502.
  Vázquez, A. C., Hiptmair, R. & Woerner, S. (2022). Enhancing the quantum
      linear systems algorithm using Richardson extrapolation.
      ACM Trans. Quantum Comput. 3, 1.
"""

from __future__ import annotations

from typing import TypeVar

T = TypeVar("T")

# Subclasses are built once per vendored class and cached, so that repeated
# strip solves in the outer iteration do not synthesise a new type per call.
# Constructing a type is cheap but not free, and a 2-D sweep performs one inner
# solve per strip per sweep — of order 10⁴ calls in a single run.
_PINNED_TYPES: dict[type, type] = {}


# ── Private Utility Methods ────────────────────────────────────────────────────

def _pinned_subclass(base: type) -> type:
    """
    Build (or retrieve) the pinning subclass of one vendored matrix class.

    The subclass overrides only the `evolution_time` setter. The getter, the
    tolerance property, the step-count property and every circuit-construction
    method are inherited unchanged, so a pinned object is in every other respect
    the object the library would have built.

    Parameters
    ----------
    base : type
        A `LinearSystemMatrix` subclass exposing `evolution_time` and
        `trotter_steps` properties — `TridiagonalToeplitz` or
        `PentadiagonalToeplitz` in this repository.

    Returns
    -------
    type
        The cached pinning subclass of `base`.
    """
    if base in _PINNED_TYPES:
        return _PINNED_TYPES[base]

    class _Pinned(base):                                   # type: ignore[misc, valid-type]
        """Vendored matrix whose Trotter step count survives `HHL.solve`."""

        # Class-level default so that the overridden setter is well defined
        # while the base class's own __init__ is still running, before any
        # instance attribute exists.
        _pinned_trotter_steps: int | None = None

        @property
        def evolution_time(self) -> float:
            """Time of the Hamiltonian evolution."""
            return self._evolution_time

        @evolution_time.setter
        def evolution_time(self, evolution_time: float) -> None:
            """
            Record a new evolution time, preserving a pinned step count.

            With no pin this defers to the base class and the step count is
            re-derived from the tolerance exactly as before, so an unpinned
            object is bit-for-bit the vendored one. With a pin the evolution
            time is stored directly and the step count is left alone, which is
            the whole purpose of the class.
            """
            if self._pinned_trotter_steps is None:
                base.evolution_time.fset(self, evolution_time)
                return
            self._evolution_time = evolution_time
            self._trotter_steps = self._pinned_trotter_steps

    _Pinned.__name__ = f"Pinned{base.__name__}"
    _Pinned.__qualname__ = _Pinned.__name__
    _PINNED_TYPES[base] = _Pinned
    return _Pinned


# ── Public Interface ───────────────────────────────────────────────────────────

def pinned_matrix_class(base: type) -> type:
    """
    Return the pinning subclass of a vendored matrix class.

    Construct through this in place of the vendored class wherever a caller may
    wish to fix the Trotter step count. Passing `trotter_steps=None` to
    `pin_trotter_steps` afterwards leaves the object behaving exactly as the
    vendored class does, so it is safe to use unconditionally.

    Parameters
    ----------
    base : type
        `TridiagonalToeplitz` or `PentadiagonalToeplitz`.

    Returns
    -------
    type
        A subclass accepting the same constructor arguments as `base`.
    """
    return _pinned_subclass(base)


def pin_trotter_steps(matrix: T, trotter_steps: int | None) -> T:
    """
    Fix the Trotter step count of a matrix object against `HHL.solve`.

    Parameters
    ----------
    matrix : LinearSystemMatrix
        A matrix object built from the class returned by `pinned_matrix_class`.
    trotter_steps : int or None
        Step count to enforce. None removes any pin and restores the vendored
        behaviour, in which the count is derived from the tolerance.

    Returns
    -------
    LinearSystemMatrix
        The same object, returned for call chaining.

    Raises
    ------
    TypeError
        If `matrix` was not built from a pinning subclass. Silently ignoring the
        request would reintroduce exactly the defect this module exists to
        remove — a parameter that appears to be set and is not.
    ValueError
        If `trotter_steps` is not a positive integer.
    """
    if not hasattr(type(matrix), "_pinned_trotter_steps"):
        raise TypeError(
            f"{type(matrix).__name__} does not support Trotter-step pinning. "
            "Build the object from pinned_matrix_class(...) rather than from "
            "the vendored class directly; on the vendored class the count is "
            "re-derived from the tolerance inside HHL.solve and any value set "
            "here is discarded without warning."
        )

    if trotter_steps is None:
        matrix._pinned_trotter_steps = None
        return matrix

    if not isinstance(trotter_steps, (int, float)) or int(trotter_steps) < 1:
        raise ValueError(
            f"trotter_steps must be a positive integer, got {trotter_steps!r}."
        )

    n = int(trotter_steps)
    matrix._pinned_trotter_steps = n
    matrix.trotter_steps = n
    return matrix
