"""
solvers/quantum/hhl_1d_4th.py
------------------------------
HHL solver for the fourth-order 1D Poisson system.

Uses PentadiagonalToeplitz for efficient Hamiltonian simulation of the
five-point stencil matrix, in place of the TridiagonalToeplitz used by
the second-order solver.

Architecture
------------
This module mirrors hhl_1d.py exactly, with two changes:
  1. The matrix class is PentadiagonalToeplitz instead of TridiagonalToeplitz.
  2. The normalised diagonal values passed to the class are derived from the
     pentadiagonal stencil coefficients (a=-30, b1=16, b2=-1) divided by the
     spectral norm alpha = ||A||_2.

Everything else — QPE, controlled rotation, proportionality recovery,
solution extraction — is inherited from the existing HHL infrastructure
unchanged.

Usage
-----
    from problems.poisson_1d_4th import PoissonProblem1D4th
    from solvers.quantum.hhl_1d_4th import hhl_solve_4th

    prob   = PoissonProblem1D4th(N=4, source_fn='fS')
    result = hhl_solve_4th(prob)
    print(result.u)
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

from problems.poisson_1d_4th import PoissonProblem1D4th
from solvers.quantum.result import SolverResult
from solvers.quantum.trotter_pinning import pin_trotter_steps, pinned_matrix_class

# -- Submodule path ------------------------------------------------------------
# The quantum_linear_solvers submodule lives at the repo root.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_QLS_PATH = _REPO_ROOT / "quantum_linear_solvers"
if str(_QLS_PATH) not in sys.path:
    sys.path.insert(0, str(_QLS_PATH))


def hhl_solve_4th(
    problem: PoissonProblem1D4th,
    epsilon: float = 0.01,
    trotter_steps: int | None = None,
) -> SolverResult:
    """
    Solve the fourth-order 1D Poisson system using HHL, from a problem object.

    Convenience wrapper over `hhl_solve_system_4th`; see that function for the
    normalisation and proportionality-recovery derivations.

    Parameters
    ----------
    problem : PoissonProblem1D4th
        The fourth-order discretised problem.
    epsilon : float
        HHL error tolerance (controls QPE register size and Trotter steps).
        Default 0.01.
    trotter_steps : int or None
        Number of Trotter steps for the Hamiltonian simulation. If None, the
        `PentadiagonalToeplitz` class auto-computes a value.

    Returns
    -------
    SolverResult
        Physical solution and diagnostics.
    """
    return hhl_solve_system_4th(problem.A, problem.b, epsilon=epsilon,
                                trotter_steps=trotter_steps)


def hhl_solve_system_4th(
    A: np.ndarray,
    b: np.ndarray,
    epsilon: float = 0.01,
    trotter_steps: int | None = None,
) -> SolverResult:
    """
    Solve a pentadiagonal system Au = b using HHL, on raw NumPy arrays.

    The array-level interface, matching the ``(A, b) -> result`` shape that
    `solvers/outer/inner.py` adapts into a strip solver. Registered there as
    ``"hhl_4th"``; without it, a 4th-order 2-D or 3-D solve would draw the
    2nd-order factory from the registry, whose `TridiagonalToeplitz` discards the
    ±2 band and solves a different system on every strip.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Hermitian pentadiagonal system matrix, N a power of 2.
    b : np.ndarray, shape (N,)
        Right-hand side vector.
    epsilon : float
        HHL error tolerance (controls QPE register size and Trotter steps).
        Default 0.01.
    trotter_steps : int or None
        Number of Trotter steps for the Hamiltonian simulation, fixed exactly
        and overriding the count epsilon would imply. If None, the count is
        derived from epsilon and the evolution time in the ordinary way; for
        small N (4, 8) the derived value is typically 1–3.

        Enforced through `solvers/quantum/trotter_pinning.py`. Assigning the
        attribute on the vendored object has no effect: `HHL.solve` sets
        `evolution_time` on every solve, and that setter re-derives the count
        from the tolerance, discarding whatever was assigned.

    Returns
    -------
    SolverResult
        Same result type as the second-order HHL solver, with fields:
        u, residual, wall_time, proportionality_constant.

    Notes
    -----
    Normalisation
    ~~~~~~~~~~~~~
    The HHL algorithm requires the matrix eigenvalues to lie in (-1, 1].
    We normalise A by its spectral norm alpha = ||A||_2, giving:

        A_norm = A / alpha,   b_norm = b / ||b||_2

    The three diagonal values passed to PentadiagonalToeplitz are therefore:

        main_diag  = a  / alpha  (a  = -30 for the integer-coefficient form)
        off_diag_1 = b1 / alpha  (b1 = +16)
        off_diag_2 = b2 / alpha  (b2 = -1)

    Proportionality recovery
    ~~~~~~~~~~~~~~~~~~~~~~~~
    The HHL output |x_raw> satisfies c·A_norm·|x_raw> ≈ |b_norm> for some
    scalar c.  We recover c via the least-squares projection:

        c_norm = <b_norm | A_norm | x_raw> / ||A_norm | x_raw>||^2

    and then rescale to physical units:

        u = c_norm * (||b|| / alpha) * x_raw

    This is identical to the proportionality recovery in hhl_1d.py.
    """
    from quantum_linear_solvers.linear_solvers import HHL
    from quantum_linear_solvers.linear_solvers.matrices.pentadiagonal_toeplitz import (
        PentadiagonalToeplitz,
    )

    t_start = time.perf_counter()

    A = np.asarray(A)
    b = np.asarray(b)
    N = len(b)

    # -- Normalise -------------------------------------------------------------
    alpha = float(np.linalg.norm(A, ord=2))   # spectral norm = ||A||_2
    b_norm_factor = float(np.linalg.norm(b))

    A_norm = A / alpha
    b_norm = b / b_norm_factor

    # -- Build the PentadiagonalToeplitz matrix object -------------------------
    # The integer stencil coefficients are a=-30, b1=16, b2=-1.
    # After normalisation by alpha, the values passed to the class are:
    #   main_diag  = -30 / alpha
    #   off_diag_1 = +16 / alpha
    #   off_diag_2 =  -1 / alpha
    #
    # Note: problem.A uses the convention that the RHS absorbs the 1/(12h^2)
    # prefactor, so A has integer entries [-1, 16, -30, 16, -1].  The spectral
    # norm alpha therefore reflects these integer values directly.
    num_qubits = int(np.log2(N))

    # The interior values are the true Toeplitz entries; boundary rows
    # differ due to ghost-point reflection. The INTERIOR values are supplied
    # to PentadiagonalToeplitz (which models the ideal Toeplitz operator)
    # and rely on the Hamiltonian simulation being a good approximation
    # of the full operator including boundary corrections.
    # The interior main diagonal is A[1,1] (not A[0,0] which has +1 added).
    # The ±1 off-diagonal is A[0,1] = 16 (unchanged by boundary correction).
    # The ±2 off-diagonal is A[0,2] = -1 (unchanged by boundary correction).
    a_norm  = float(A_norm[1, 1])      # interior main diagonal: -30/alpha
    b1_norm = float(A_norm[0, 1])      # ±1 off-diagonal: +16/alpha
    b2_norm = float(A_norm[0, 2])      # ±2 off-diagonal: -1/alpha

    # Built from the pinning subclass. `PentadiagonalToeplitz` re-derives its
    # step count inside the `evolution_time` setter, and `HHL.solve` assigns that
    # attribute on every solve, so a count assigned to the vendored object here
    # is discarded before a single gate is built. `pin_trotter_steps` is the only
    # channel that survives; with `trotter_steps=None` the object behaves exactly
    # as the vendored class does. See `solvers/quantum/trotter_pinning.py`.
    matrix = pinned_matrix_class(PentadiagonalToeplitz)(
        num_state_qubits=num_qubits,
        main_diag=a_norm,
        off_diag_1=b1_norm,
        off_diag_2=b2_norm,
        tolerance=epsilon / 6.0,   # epsilon_a = epsilon / 6, as in HHL
    )

    pin_trotter_steps(matrix, trotter_steps)

    # -- Run HHL ---------------------------------------------------------------
    hhl = HHL(epsilon=epsilon)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solution = hhl.solve(matrix, b_norm)

    # -- Extract raw solution vector from statevector --------------------------
    x_raw = _extract_solution_vector(solution, num_qubits)

    # Normalise to unit vector — post-selection amplitude is suppressed
    # by sqrt(p_success) which is unknown; only the direction matters.
    x_raw_norm = np.linalg.norm(x_raw)
    if x_raw_norm < 1e-14:
        return SolverResult(
            u=np.zeros(N), solver="HHL-4th",
            raw_state=x_raw, prop_const=0.0, euclidean_residual=1.0,
        )
    x_raw_unit = x_raw / x_raw_norm

    # print(f"\n  [HHL-4th diagnostic]")
    # print(f"    num_qubits (data register) = {num_qubits}")
    # print(f"    total circuit qubits       = {solution.state.num_qubits}")
    # print(f"    ||x_raw||                  = {np.linalg.norm(x_raw):.6e}")
    # print(f"    x_raw                      = {x_raw}")
    # print(f"    ||A_norm @ x_raw||         = {np.linalg.norm(A_norm @ x_raw):.6e}")
    # cos_angle = np.dot(b_norm, A_norm @ x_raw) / (
    #     np.linalg.norm(A_norm @ x_raw) * np.linalg.norm(b_norm) + 1e-300
    # )
    # print(f"    cos(A_norm x_raw, b_norm)  = {cos_angle:.6f}")
    # print(f"    u_thomas                   = {np.linalg.solve(A, b)}")

    # Recover physical scale: find c such that c * A @ x_raw_unit ≈ b
    # Projection employs b rather than Ax to avoid amplifying high-frequency QPE noise.
    Ax = A @ x_raw_unit
    denom = float(np.dot(b, Ax))
    numer = float(np.dot(b, b))
    # print(f"    denom_phys = {denom:.6e}")
    # print(f"    numer_phys = {numer:.6e}")
    # print(f"    c_phys     = {numer / denom:.6e}")
    c_phys = numer / denom
    u = c_phys * x_raw_unit

    residual = float(
        np.linalg.norm(A @ u - b) / (np.linalg.norm(b) + 1e-300)
    )
    # print(f"    residual = {residual:.6e}")
    wall = time.perf_counter() - t_start

    return SolverResult(
        u=u, solver="HHL-4th",
        raw_state=x_raw, prop_const=c_phys,
        euclidean_residual=residual,
    )

def _extract_solution_vector(solution, num_qubits: int) -> np.ndarray:
    """
    Extract the solution amplitudes from the HHL statevector.

    Post-selects on:
      - ancilla (flag) qubit = |1>
      - QPE (clock) register = |0...0>

    and reads off the data register amplitudes.

    This is identical to the extraction logic in hhl_1d.py.
    """
    from qiskit.quantum_info import Statevector

    qc = solution.state
    sv = np.array(Statevector(qc))

    N = 2 ** num_qubits
    n_total = qc.num_qubits
    n_b = num_qubits
    n_other = n_total - 1 - n_b   # clock + ancilla registers (minus the flag)

    x_raw = np.zeros(N, dtype=complex)

    for idx in range(2 ** n_total):
        # Qiskit little-endian: bit k of idx = qubit k
        ancilla_bit = (idx >> (n_total - 1)) & 1   # flag qubit (MSB)
        other_bits  = (idx >> n_b) & ((1 << n_other) - 1)
        b_reg_idx   = idx & (N - 1)

        if ancilla_bit == 1 and other_bits == 0:
            x_raw[b_reg_idx] = sv[idx]

    x_raw_real = np.real(x_raw)

    if np.allclose(x_raw_real, 0.0):
        # Fallback: try reading directly from solution.euclidean_norm scaling
        # This can happen if the qubit ordering differs from the assumed layout.
        import warnings
        warnings.warn(
            "HHL-4th: solution extraction returned all-zero vector. "
            "The qubit register layout may differ from the assumed convention. "
            "Returning zero vector — check solution.state.draw() for layout.",
            RuntimeWarning,
        )

    return x_raw_real