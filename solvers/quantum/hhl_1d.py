"""
Harrow-Hassidim-Lloyd (HHL) quantum solver for 1D Poisson systems with a
Toeplitz Symmetric Tridiagonal (TST) matrix structure.

Public Interface
----------------
hhl_solve(problem) :
    High-level wrapper accepting a `PoissonProblem1D`, returning a standardised
    `SolverResult`.
hhl_solve_system(A, b, eps) :
    Low-level routine accepting raw NumPy arrays. Kept separate so that the
    outer iteration in `solvers/outer` can solve each strip without
    constructing a problem container per sub-problem.

Hamiltonian simulation is provided by the vendored `quantum_linear_solvers`
implementation of the TST evolution operator (Vázquez et al.), and all circuits
are evaluated by deterministic statevector simulation; no shot noise is introduced by the measurement process.
"""
from __future__ import annotations

import warnings

import numpy as np

from quantum_linear_solvers.linear_solvers.hhl import HHL
from quantum_linear_solvers.linear_solvers.matrices.tridiagonal_toeplitz import (
    TridiagonalToeplitz,
)
from quantum_linear_solvers.linear_solvers.matrices.numpy_matrix import (
    NumPyMatrix,
)
from qiskit.quantum_info import Statevector

from core.execution import default_executor, hhl_spec

from problems.poisson_1d import PoissonProblem1D
from solvers.quantum.block_encoding import is_toeplitz_tridiagonal
from solvers.quantum.result import SolverResult
from solvers.quantum.trotter_pinning import pin_trotter_steps, pinned_matrix_class


# -- Public High-Level Interface -----------------------------------------------

def hhl_solve(problem: PoissonProblem1D) -> SolverResult:
    """
    Solves the 1D Poisson system Au = b by the HHL algorithm.

    A wrapper around `hhl_solve_system` that unpacks the `PoissonProblem1D`
    container and packages the outputs into a `SolverResult`.

    Parameters
    ----------
    problem : PoissonProblem1D
        Discretised 1D problem supplying the N×N operator, the length-N
        right-hand side and the precision parameter ε.

    Returns
    -------
    result : SolverResult
        Solution vector, raw b-register amplitudes, proportionality constant
        and relative Euclidean residual.
    """
    u, x_raw, c = hhl_solve_system(
        problem.A,
        problem.b,
        problem.config.epsilon,
    )
    return SolverResult(
        u=u,
        solver="HHL",
        raw_state=x_raw,
        prop_const=c,
        euclidean_residual=_relative_residual(problem.A, u, problem.b),
    )


# -- Core Algorithmic Sub-Routine ----------------------------------------------

def hhl_solve_system(
    A:             np.ndarray,
    b:             np.ndarray,
    epsilon:       float,
    trotter_steps: int | None = None,
    diagnostics:   dict | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solves the linear system Au = b by the HHL algorithm, operating directly on
    raw NumPy arrays.

    This is the primary quantum execution routine. It is invoked once per strip
    per sweep by the outer iteration in `solvers/outer`, and holds no dependency
    on any problem container class.

    Parameters
    ----------
    A : np.ndarray
        N×N TST system matrix, assumed Hermitian. Spectrally normalised
        internally so that its eigenvalues lie within (-1, 1].
    b : np.ndarray
        Length-N right-hand side vector.
    epsilon : float
        Overall precision parameter of the algorithm, ε. Supplied to `HHL`,
        which apportions it as ε_r = ε_s = ε/3 to the reciprocal rotation and
        the state preparation and ε_a = ε/6 to the Hamiltonian simulation, and
        derives the Trotter step count from ε_a and the evolution time. The
        default of 0.01 used throughout this repository coincides with the
        library default, so a solve at that value reproduces every recorded
        sweep exactly.

        Measured at N = 4 on the second-order operator: ε ∈ {1e-1, 1e-2, 1e-3}
        gives step counts {3, 7, 21} and relative residuals
        {2.40e-2, 1.79e-2, 1.65e-2}.
    trotter_steps : int or None
        Hamiltonian-simulation step count, fixed exactly and overriding the
        count ε would imply. None derives it from ε in the ordinary way. Set
        this only where the step count is itself the independent variable, as in
        a sensitivity sweep over simulation depth at fixed ε; see
        `solvers/quantum/trotter_pinning.py` for why the vendored class cannot
        honour a step count supplied to its constructor.
    diagnostics : dict or None
        Optional output mapping, updated in place with quantities that are
        settled inside the solve and are otherwise unobservable from the return
        tuple:

          ``trotter_steps``   The step count actually simulated. This is *not*
                              recoverable from ε by the caller: the library
                              derives it as ⌈√((t·|b_off|)³ / 2ε_a)⌉ with an
                              evolution time t that HHL itself computes from the
                              spectral bounds. Recording ⌈1/ε⌉ instead, as the
                              sweeps did, reports a number no circuit ever used.
          ``evolution_time``  The evolution time HHL selected, in the same units.
          ``simulation``      ``"trotter"`` on the Toeplitz branch,
                              ``"exact_exponential"`` on the general branch,
                              where there is no Trotter error to control.

        Supplying nothing leaves the solve unchanged; the return tuple is
        identical either way.

    Returns
    -------
    u : np.ndarray
        Length-N recovered physical solution vector.
    x_raw : np.ndarray
        Length-N raw quantum state amplitudes extracted from the b-register.
    c : float
        Proportionality constant satisfying c·A·x_raw ≈ b.

    Raises
    ------
    ValueError
        If the right-hand side b is numerically zero, precluding state
        normalisation. Callers in the outer layer must detect this and assign a
        zero-vector solution directly.
    RuntimeError
        If statevector extraction yields a null vector under the strict
        post-selection criteria, or if proportionality recovery degenerates.
    """
    N = len(b)

    # -- Phase 0: Structural Precondition --------------------------------------
    # `TridiagonalToeplitz` is constructed from A[0,0] and A[0,1] only, so any wider
    # band -- or any diagonal that is not constant -- would be discarded without
    # trace and a different system solved. Where that reconstruction is exact the
    # Toeplitz operator is kept, because its Hamiltonian simulation is the
    # structured one the published figures were produced with; where it is not, the
    # general `NumPyMatrix` simulation encodes A in full.
    #
    # The distinction is not academic. Sub-case 3c's halved Neumann row gives
    # A[0,0] = -1 against -2 elsewhere, so the reconstruction silently built a
    # uniformly shifted operator and HHL returned ~100 % error at every N while
    # appearing entirely healthy. See `is_toeplitz_tridiagonal`.
    use_toeplitz = is_toeplitz_tridiagonal(A)

    # -- Phase 1: Spectral Normalisation ---------------------------------------
    A_norm_factor = float(np.linalg.norm(A, ord=2))
    b_norm_factor = float(np.linalg.norm(b))

    if b_norm_factor < 1e-14:
        raise ValueError(
            "RHS vector b is numerically zero; state normalisation for amplitude "
            "encoding cannot proceed. The calling namespace must detect this "
            "condition and assign a zero-vector solution directly."
        )

    b_norm = b / b_norm_factor

    # -- Phase 2: Operator Construction ----------------------------------------
    num_qubits = int(np.log2(N))

    if use_toeplitz:
        a_norm = A[0, 0] / A_norm_factor   # Principal diagonal of normalised A
        b_off  = A[0, 1] / A_norm_factor   # Off-diagonal of normalised A

        # Built from the pinning subclass unconditionally. With no pin requested
        # it behaves exactly as `TridiagonalToeplitz`; with one it is the only
        # way the request can survive, because `HHL.solve` re-derives the step
        # count from the tolerance inside the `evolution_time` setter. The step
        # count passed here is a placeholder in either case, overwritten by that
        # derivation before any gate is built.
        matrix = pinned_matrix_class(TridiagonalToeplitz)(
            num_state_qubits=num_qubits,
            main_diag=a_norm,
            off_diag=b_off,
            trotter_steps=1,
        )
        pin_trotter_steps(matrix, trotter_steps)
    else:
        # `NumPyMatrix` exponentiates the supplied operator directly, via
        # `scipy.linalg.expm`, rather than exploiting a banded structure. It
        # accepts any Hermitian A at the cost of a denser evolution circuit, and
        # A is normalised by ‖A‖₂ exactly as in the Toeplitz branch, so every
        # downstream stage -- the QPE register sizing, the post-selection, the
        # proportionality recovery below -- is unchanged.
        #
        # Its Hamiltonian simulation carries **no Trotter error**: the
        # exponential is formed exactly. A step count is therefore not merely
        # unsupported on this branch but meaningless, and is rejected rather than
        # accepted and ignored. `tolerance` is likewise not a simulation
        # parameter here; it is supplied for consistency with the base class and
        # is overwritten by `HHL.solve` in any case, ε reaching the algorithm
        # through the `HHL` constructor below.
        if trotter_steps is not None:
            raise ValueError(
                f"trotter_steps={trotter_steps} was requested, but this operator "
                "is not Toeplitz tridiagonal and is simulated by exact matrix "
                "exponentiation, which has no Trotter decomposition to control. "
                "Vary epsilon instead, or restrict the sweep to Toeplitz cases."
            )
        matrix = NumPyMatrix(A / A_norm_factor, tolerance=epsilon)

    # -- Phase 3: Algorithm Execution ------------------------------------------
    # ε is supplied to the algorithm rather than to the matrix. `HHL.solve`
    # assigns `matrix.tolerance = self._epsilon_a` before every solve, so a
    # tolerance set on the matrix is discarded; routing ε through the constructor
    # is the only path by which it reaches the Hamiltonian simulation, the
    # reciprocal rotation and the state preparation. The library default is
    # itself 0.01, so the repository default reproduces every recorded sweep.
    hhl = HHL(epsilon=epsilon)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solution = hhl.solve(matrix, b_norm)

    # Read back after the solve, not before: `HHL.solve` fixes the evolution time
    # from the spectral bounds and re-derives the step count from it, so the
    # values the matrix carried at construction are not the ones simulated.
    if diagnostics is not None:
        diagnostics["evolution_time"] = float(
            getattr(matrix, "evolution_time", float("nan"))
        )
        if use_toeplitz:
            diagnostics["simulation"] = "trotter"
            diagnostics["trotter_steps"] = int(matrix.trotter_steps)
        else:
            diagnostics["simulation"] = "exact_exponential"
            diagnostics["trotter_steps"] = None

    # -- Phase 4: Statevector Extraction ---------------------------------------
    x_raw = _extract_solution_statevector(solution.state, num_qubits)

    # -- Phase 5: Dimensionality Recovery --------------------------------------
    # The scaling constant is recovered against the *normalised* system, so that
    # quantum error is not amplified by the geometric factor ||b||_2 / ||A||_2.
    # This matters for physically scaled domains — notably the HET case, where
    # ||b|| >> 1 owing to the alpha = L^2 / lambda_D^2 prefactor.
    #
    # The normalised geometric relation is
    #   A_norm . x_raw ~ c_norm . b_norm
    # with A_norm = A / ||A||_2 and b_norm = b / ||b||_2, from which the
    # physical solution is recovered as
    #   u = c_norm . x_raw . (||b||_2 / ||A||_2)

    A_norm = A / A_norm_factor
    b_norm_vec = b / b_norm_factor

    Ax_norm = A_norm @ x_raw
    denom   = float(np.dot(Ax_norm, Ax_norm))

    if denom < 1e-14:
        raise RuntimeError(
            "HHL proportionality recovery failed: ||A_norm·x_raw||² "
            "is numerically zero."
        )

    c_norm = float(np.dot(b_norm_vec, Ax_norm) / denom)
    scale  = b_norm_factor / A_norm_factor
    u      = c_norm * scale * x_raw
    c      = c_norm * scale

    return u, x_raw, c


# -- Private Utility Methods ---------------------------------------------------

def _relative_residual(
    A: np.ndarray,
    u: np.ndarray,
    b: np.ndarray,
) -> float:
    """Computes the relative Euclidean residual ‖Au - b‖₂ / ‖b‖₂."""
    return float(np.linalg.norm(A @ u - b) / np.linalg.norm(b))


def _extract_solution_statevector(
    circuit,
    num_qubits: int,
) -> np.ndarray:
    """
    Extracts the solution vector from the HHL output quantum circuit.

    Register layout (from circuit.qregs, Qiskit little-endian ordering):

        qregs[0] : b-register (solution), n_b qubits, indices [0, n_b - 1]
        qregs[1] : l-register (clock),    n_l qubits, indices [n_b, n_b+n_l-1]
        qregs[2] : MCMT ancilla,          n_a qubits
        qregs[3] : flag qubit (ancilla),  index [n_total - 1]

    Strict post-selection criteria:

        - flag qubit evaluates to |1⟩
        - clock (l-register) is cleared to |0…0⟩
        - MCMT ancillae are returned to |0…0⟩

    Parameters
    ----------
    circuit : QuantumCircuit
        HHL output circuit carrying the solution register.
    num_qubits : int
        Number of b-register qubits; N = 2^num_qubits.

    Returns
    -------
    x_raw : np.ndarray
        Length-N real part of the post-selected b-register amplitudes.

    Raises
    ------
    RuntimeError
        If post-selection yields a null vector. A diagnostic table of the ten
        dominant statevector amplitudes is emitted first, to identify which
        register failed to clear.
    """
    # Routed through core.execution so that the same solver body runs under
    # exact statevector evolution (the default and the thesis baseline), under
    # a shot-based noisy simulator, or on hardware. The default executor
    # reproduces the previous inline masking and diagnostics bit-for-bit; the
    # register layout documented above is now expressed by ``hhl_spec``.
    x_raw_real, _record = default_executor().extract(
        circuit,
        hhl_spec(circuit, num_qubits),
    )
    return x_raw_real
