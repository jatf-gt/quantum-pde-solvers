"""
Extended metric dataclasses and computation utilities for the quantum PDE
solver benchmarking framework.

This module defines the canonical data contract for all benchmark results.
Every field is documented with its physical or mathematical meaning, units,
and the conditions under which it may be None. The design principle is that
all metrics are *measured* from the actual solver output, never inferred or
estimated from input parameters.

Mathematical context
────────────────────
For a linear system  A u = b  with solution  u*:

  Relative residual:   r  = ‖A û - b‖₂ / ‖b‖₂
  Max relative error:  e∞ = max_i |û_i - u*_i| / |u*_i|   (where |u*_i| > tol)
  Max absolute error:  ea = max_i |û_i - u*_i|
  VQLS cost bound:     C  ≥ r² / κ²   (Bravo-Prieto et al., Quantum 7, 1188, 2023)

The VQLS cost bound implies that a cost value C does NOT directly bound the
residual r without knowledge of κ. The residual must always be computed
explicitly from the returned solution vector.

References
──────────
  Bravo-Prieto et al. (2023) Quantum 7, 1188.  doi:10.22331/q-2023-11-22-1188
  Ghafourpour & Laizet (2025) Phys. Rev. Applied 24, 024032.
  Morales et al. (2026) Rev. Mod. Phys. 98, 025005.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ── Tolerance for near-zero masking in relative error computation ─────────────
_REL_ERR_MASK_TOL: float = 1.0e-10


# ── Public API ────────────────────────────────────────────────────────────────

@dataclass
class CircuitMetrics:
    """
    Circuit-level resource metrics extracted from a Qiskit QuantumCircuit.

    All depth values are gate counts after transpilation at a fixed
    optimisation level. Qubit count is the total register width including
    ancillae. These are *logical* resource estimates on a noise-free device;
    physical resource estimates on specific hardware topologies are out of
    scope for this module.

    Attributes
    ----------
    n_qubits : int
        Total number of qubits in the circuit (solution register + ancillae
        + clock register for HHL, or ancilla only for QSVT).
    depth_raw : int
        Circuit depth before transpilation (logical depth, gate count).
    depth_opt0 : int
        Circuit depth after transpilation at Qiskit optimisation level 0
        (no gate cancellation; faithful representation of the logical circuit).
    depth_opt1 : int
        Circuit depth after transpilation at optimisation level 1
        (light optimisation; representative of a realistic compilation).
    n_cx_gates : int
        Number of two-qubit (CNOT/CX) gates at optimisation level 1.
        Two-qubit gates dominate the error budget on real hardware.
    transpile_time_s : float
        Wall time [s] consumed by the Qiskit transpilation step itself.
        Excluded from solver wall time.
    optimisation_level : int
        The primary optimisation level used for depth_opt reporting.
        Set to 1 throughout this framework for consistency.
    """

    n_qubits:           int
    depth_raw:          int
    depth_opt0:         int
    depth_opt1:         int
    n_cx_gates:         int
    transpile_time_s:   float
    optimisation_level: int = 1


@dataclass
class BenchmarkResult:
    """
    Complete record for a single (solver, problem, N) benchmark run.

    This dataclass is the canonical unit of storage for the benchmarking
    framework. Every field that can be None is documented with the condition
    under which it is absent. Fields are grouped by category for clarity.

    Serialisation
    ─────────────
    All fields are JSON-serialisable (None, int, float, str, list of float).
    numpy arrays are converted to Python lists before storage.

    Attributes — Problem identification
    ────────────────────────────────────
    case_id : str
        Unique identifier for the problem case (e.g. '1D_Poisson_fS_hom').
    solver : str
        Algorithm name: 'thomas' | 'hhl' | 'vqls' | 'qsvt'.
    N : int
        Number of interior nodes (problem size).
    discretisation_order : int
        Spatial discretisation order: 2 (tridiagonal TST) or 4 (pentadiagonal).
    kappa : float
        2-norm condition number κ(A), computed from eigendecomposition.
    source_fn : str
        Source function key: 'fS' | 'fL' | 'fH' | 'gaussian' | 'linear' | 'step'.
    alpha_bc : float
        Left Dirichlet boundary value φ(0).
    beta_bc : float
        Right Dirichlet boundary value φ(1).

    Attributes — Accuracy metrics
    ──────────────────────────────
    residual : float
        Relative residual r = ‖Aû - b‖₂ / ‖b‖₂. Always computed from the
        returned solution vector; never inferred from solver parameters.
    max_rel_err_vs_exact : Optional[float]
        Maximum relative error [%] against the analytical solution.
        None if no analytical solution exists for this case.
    max_abs_err_vs_exact : Optional[float]
        Maximum absolute error against the analytical solution.
        None if no analytical solution exists.
    max_rel_err_vs_thomas : float
        Maximum relative error [%] against the Thomas algorithm solution.
        Always available; separates algorithmic error from discretisation error.
    max_abs_err_vs_thomas : float
        Maximum absolute error against the Thomas algorithm solution.
    err_disc : Optional[float]
        Discretisation error [%]: max relative error of Thomas vs exact.
        None if no analytical solution exists.
    err_alg : Optional[float]
        Algorithmic error [%]: approximate quantum-specific error component,
        estimated as max_rel_err_vs_exact - err_disc. This is an upper bound,
        not an exact decomposition. None if err_disc is None.
    proportionality_residual : Optional[float]
        Residual of the proportionality recovery step for HHL and QSVT:
        ‖A(c·x_raw) - b‖₂ / ‖b‖₂ where c is the recovered scalar.
        Measures the error introduced by the recovery step independently of
        the circuit error. None for Thomas and VQLS.

    Attributes — Timing
    ────────────────────
    wall_time_s : float
        Total solver wall time [s] measured with time.perf_counter().
        For VQLS, includes the full optimisation loop.
        For QSVT, includes phase-angle lookup (not precomputation).
        Does NOT include circuit transpilation time (see circuit_metrics).
    phase_lookup_time_s : Optional[float]
        Time [s] spent on QSVT phase-angle disk cache lookup or computation.
        Subset of wall_time_s. None for HHL, VQLS, Thomas.

    Attributes — Circuit resources
    ───────────────────────────────
    circuit_metrics : Optional[CircuitMetrics]
        Full circuit resource record. None for Thomas (no circuit).
        See CircuitMetrics docstring for field definitions.

    Attributes — Algorithm-specific parameters
    ────────────────────────────────────────────
    hhl_epsilon : Optional[float]
        HHL QPE precision parameter ε. None for other solvers.
    hhl_trotter_steps : Optional[int]
        Number of Lie–Trotter–Suzuki steps used in Hamiltonian simulation.
        In the current implementation, trotter_steps = max(1, ceil(1/epsilon)).
        None for other solvers.
    vqls_n_layers : Optional[int]
        Number of ansatz layers in the VQLS parameterised circuit.
        None for other solvers.
    vqls_n_restarts : Optional[int]
        Number of COBYLA restart stages used. None for other solvers.
    vqls_cost_final : Optional[float]
        Final VQLS cost function value C(θ*) at convergence.
        Note: C ≥ r²/κ² (Bravo-Prieto et al. 2023), so this does NOT
        directly bound the residual without knowledge of κ.
    vqls_n_evaluations : Optional[int]
        Total number of cost function evaluations by the COBYLA optimiser.
        None for other solvers.
    vqls_converged : Optional[bool]
        True if the COBYLA optimiser reported convergence within the
        specified tolerance. False indicates the iteration limit was reached.
        None for other solvers.
    qsvt_polynomial_degree : Optional[int]
        Degree of the Chebyshev polynomial approximation to 1/(κ·x) used
        in the QSVT inversion. None for other solvers.
    qsvt_max_degree_cap : Optional[int]
        Maximum degree cap applied during phase-angle computation.
        None if uncapped (full-precision phases used). None for other solvers.
    qsvt_subnormalisation : Optional[float]
        Block encoding subnormalisation factor α = ‖A‖₂. The effective
        condition number driving the QSVT polynomial degree is κ_eff = κ·α.
        None for other solvers.
    qsvt_kappa_eff : Optional[float]
        Effective condition number κ_eff = κ · α for QSVT.
        None for other solvers.
    qsvt_angle_method : Optional[str]
        Phase-angle computation method: 'auto' | 'symqsp_wrapper'.
        None for other solvers.
    qsvt_phase_from_cache : Optional[bool]
        True if phase angles were loaded from the disk cache.
        False if computed at runtime. None for other solvers.

    Attributes — Sensitivity study metadata
    ─────────────────────────────────────────
    sensitivity_param : Optional[str]
        Name of the parameter being varied in a sensitivity sweep.
        None for primary benchmark runs.
    sensitivity_value : Optional[float]
        Value of the sensitivity parameter for this run.
        None for primary benchmark runs.
    r_target : Optional[float]
        Target residual for equal-accuracy protocol runs.
        None for primary benchmark (fixed-parameter) runs.

    Attributes — Hardware execution metadata
    ─────────────────────────────────────────
    backend_name : str
        Execution backend: 'aer_statevector' | 'aer_gpu' | 'ibm_<device>'.
    backend_shots : Optional[int]
        Number of measurement shots. None for statevector (exact) simulation.
    error_mitigation : Optional[str]
        Error mitigation technique applied, if any: 'zne' | 'pec' | None.
        Always None for statevector simulation.
    hardware_run : bool
        True if executed on real quantum hardware. False for simulation.
        When True, backend_shots and error_mitigation must be non-None.
    """

    # ── Problem identification ────────────────────────────────────────────────
    case_id:               str
    solver:                str
    N:                     int
    discretisation_order:  int
    kappa:                 float
    source_fn:             str
    alpha_bc:              float
    beta_bc:               float

    # ── Accuracy metrics ──────────────────────────────────────────────────────
    residual:                    float
    max_rel_err_vs_exact:        Optional[float]
    max_abs_err_vs_exact:        Optional[float]
    max_rel_err_vs_thomas:       float
    max_abs_err_vs_thomas:       float
    err_disc:                    Optional[float]
    err_alg:                     Optional[float]
    proportionality_residual:    Optional[float]

    # ── Timing ────────────────────────────────────────────────────────────────
    wall_time_s:          float
    phase_lookup_time_s:  Optional[float]

    # ── Circuit resources ─────────────────────────────────────────────────────
    circuit_metrics:  Optional[CircuitMetrics]

    # ── HHL parameters ────────────────────────────────────────────────────────
    hhl_epsilon:        Optional[float]
    hhl_trotter_steps:  Optional[int]

    # ── VQLS parameters ───────────────────────────────────────────────────────
    vqls_n_layers:       Optional[int]
    vqls_n_restarts:     Optional[int]
    vqls_cost_final:     Optional[float]
    vqls_n_evaluations:  Optional[int]
    vqls_converged:      Optional[bool]

    # ── QSVT parameters ───────────────────────────────────────────────────────
    qsvt_polynomial_degree:  Optional[int]
    qsvt_max_degree_cap:     Optional[int]
    qsvt_subnormalisation:   Optional[float]
    qsvt_kappa_eff:          Optional[float]
    qsvt_angle_method:       Optional[str]
    qsvt_phase_from_cache:   Optional[bool]

    # ── Sensitivity study metadata ────────────────────────────────────────────
    sensitivity_param:  Optional[str]
    sensitivity_value:  Optional[float]
    r_target:           Optional[float]

    # ── Hardware execution metadata ───────────────────────────────────────────
    backend_name:       str   = "aer_statevector"
    backend_shots:      Optional[int]  = None
    error_mitigation:   Optional[str]  = None
    hardware_run:       bool  = False

    def to_dict(self) -> dict:
        """
        Serialise to a JSON-compatible dictionary.

        CircuitMetrics is flattened into the top-level dict with a
        'circuit_' prefix to avoid nested JSON structures.
        """
        d: dict = {}
        for fname, fval in self.__dict__.items():
            if fname == "circuit_metrics":
                if fval is not None:
                    for cname, cval in fval.__dict__.items():
                        d[f"circuit_{cname}"] = cval
                else:
                    for cname in CircuitMetrics.__dataclass_fields__:
                        d[f"circuit_{cname}"] = None
            else:
                d[fname] = fval
        return d


# ── Computation utilities ─────────────────────────────────────────────────────

def compute_residual(
    A: np.ndarray,
    u: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Compute the relative residual r = ‖Au - b‖₂ / ‖b‖₂.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        System matrix.
    u : np.ndarray, shape (N,)
        Candidate solution vector.
    b : np.ndarray, shape (N,)
        Right-hand side vector.

    Returns
    -------
    float
        Relative residual. Returns inf if ‖b‖₂ < 1e-300 (degenerate RHS).
    """
    b_norm = float(np.linalg.norm(b))
    if b_norm < 1.0e-300:
        return float("inf")
    return float(np.linalg.norm(A @ u - b)) / b_norm


def compute_max_rel_err(
    u: np.ndarray,
    u_ref: np.ndarray,
    mask_tol: float = _REL_ERR_MASK_TOL,
) -> float:
    """
    Compute the maximum relative error, excluding near-zero reference nodes.

    Nodes where |u_ref_i| < mask_tol are excluded from the maximum to
    prevent division by near-zero values from dominating the metric.
    If all reference values are below the threshold, the maximum absolute
    error is returned instead (with a warning logged to stderr).

    Parameters
    ----------
    u : np.ndarray, shape (N,)
        Candidate solution vector.
    u_ref : np.ndarray, shape (N,)
        Reference solution vector (analytical or Thomas).
    mask_tol : float, optional
        Threshold below which reference values are excluded. Default 1e-10.

    Returns
    -------
    float
        Maximum relative error as a decimal fraction (not percentage).
        Multiply by 100 for percentage.
    """
    mask = np.abs(u_ref) > mask_tol
    if not np.any(mask):
        import sys
        print(
            "  [WARN] compute_max_rel_err: all reference values below "
            f"mask_tol={mask_tol:.0e}; returning max absolute error instead.",
            file=sys.stderr,
        )
        return float(np.max(np.abs(u - u_ref)))
    return float(np.max(np.abs((u[mask] - u_ref[mask]) / u_ref[mask])))


def compute_max_abs_err(
    u: np.ndarray,
    u_ref: np.ndarray,
) -> float:
    """
    Compute the maximum absolute error max_i |û_i - u*_i|.

    Parameters
    ----------
    u : np.ndarray, shape (N,)
        Candidate solution vector.
    u_ref : np.ndarray, shape (N,)
        Reference solution vector.

    Returns
    -------
    float
        Maximum absolute error.
    """
    return float(np.max(np.abs(u - u_ref)))


def extract_circuit_metrics(
    circuit,
    optimisation_level: int = 1,
) -> CircuitMetrics:
    """
    Extract circuit resource metrics from a Qiskit QuantumCircuit.

    Transpiles the circuit at two optimisation levels (0 and the specified
    level) to provide both the raw logical depth and a realistic compiled
    depth. The two-qubit gate count is extracted at the specified level.

    Parameters
    ----------
    circuit : qiskit.QuantumCircuit
        The quantum circuit to analyse. Must be a valid Qiskit circuit object.
    optimisation_level : int, optional
        Primary Qiskit transpilation optimisation level (0–3). Default 1.
        Level 0: no gate cancellation (faithful logical representation).
        Level 1: light optimisation (recommended for fair benchmarking).
        Level 3: aggressive optimisation (not recommended for comparison).

    Returns
    -------
    CircuitMetrics
        Populated CircuitMetrics dataclass.

    Raises
    ------
    ImportError
        If qiskit or qiskit_aer is not installed.
    """
    try:
        from qiskit import transpile
        from qiskit_aer import AerSimulator
    except ImportError as exc:
        raise ImportError(
            "Qiskit and qiskit-aer are required for circuit metric extraction."
        ) from exc

    backend = AerSimulator(method="statevector")
    n_qubits = circuit.num_qubits

    t0 = time.perf_counter()

    # Raw depth (no transpilation, just count gates)
    depth_raw = circuit.depth()

    # Optimisation level 0: no gate cancellation
    qc_opt0 = transpile(circuit, backend=backend, optimization_level=0)
    depth_opt0 = qc_opt0.depth()

    # Specified optimisation level
    qc_opt = transpile(circuit, backend=backend,
                       optimization_level=optimisation_level)
    depth_opt = qc_opt.depth()

    # Two-qubit gate count at the specified level
    cx_count = qc_opt.count_ops().get("cx", 0)
    cx_count += qc_opt.count_ops().get("ecr", 0)   # IBM native two-qubit gate
    cx_count += qc_opt.count_ops().get("cz", 0)

    transpile_time = time.perf_counter() - t0

    return CircuitMetrics(
        n_qubits=n_qubits,
        depth_raw=depth_raw,
        depth_opt0=depth_opt0,
        depth_opt1=depth_opt,
        n_cx_gates=cx_count,
        transpile_time_s=transpile_time,
        optimisation_level=optimisation_level,
    )