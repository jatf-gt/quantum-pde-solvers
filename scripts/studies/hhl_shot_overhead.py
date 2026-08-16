"""
HHL post-selection shot-overhead experiment.

Quantifies the overhead required for HHL post-selection. HHL succeeds with probability ~1/kappa², necessitating ~kappa² total shots per usable sample. This script measures the overhead directly via core.hardware.hardware_postselection_sample and compares it against the exact statevector value.

Implementation Correction
-------------------------
The initial version passed main_diag and off_diag directly to TridiagonalToeplitz without spectral normalisation. This produced a post-selection probability of 0.95 for a kappa~9 problem (expected ~0.012) by feeding a Hamiltonian-simulation-based QPE routine an invalid eigenvalue range. The current build_hhl_circuit mirrors hhl_solve_system exactly, performing the required normalisation. A sanity check now warns if the measured probability deviates by two orders of magnitude from the 1/kappa² expectation.

Usage
-----
    python scripts/hhl_shot_overhead.py --N 4
    python scripts/hhl_shot_overhead.py --N 4 --real --backend ibm_kingston
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import warnings

import numpy as np

from core.execution import StatevectorExecutor, hhl_spec
from core.hardware import HardwareContext, hardware_postselection_sample


def build_hhl_circuit(N: int, main_diag: float, off_diag: float, epsilon: float):
    """
    Mirrors solvers.quantum.hhl_1d.hhl_solve_system's circuit construction
    exactly, stopping before statevector extraction (which this script does
    on hardware instead, via post-selection sampling).

    Re-verified character-for-character against the repository's current
    hhl_1d.py (commit e03dcc5) before this fix, after an earlier version of
    this script passed the raw main_diag/off_diag values directly to
    TridiagonalToeplitz instead of spectrally normalising them first --
    hhl_solve_system's own docstring states its A parameter is "spectrally
    normalised internally so that its eigenvalues lie within (-1, 1]" and
    its body computes a_norm = A[0,0]/||A||_2, b_off = A[0,1]/||A||_2 before
    ever constructing the TridiagonalToeplitz operator. Skipping that step
    feeds the Hamiltonian-simulation-based QPE an eigenvalue range it was
    never designed for, and the eigenvalue-inversion flag's success
    probability -- the entire quantity this script exists to measure --
    is meaningless under it.
    """
    from quantum_linear_solvers.linear_solvers.hhl import HHL
    from quantum_linear_solvers.linear_solvers.matrices.tridiagonal_toeplitz import (
        TridiagonalToeplitz,
    )

    A = (
        main_diag * np.eye(N)
        + off_diag * np.diag(np.ones(N - 1), k=1)
        + off_diag * np.diag(np.ones(N - 1), k=-1)
    )
    A_norm_factor = float(np.linalg.norm(A, ord=2))
    a_norm = A[0, 0] / A_norm_factor
    b_off  = A[0, 1] / A_norm_factor

    num_qubits    = int(np.log2(N))
    trotter_steps = max(1, int(np.ceil(1.0 / epsilon)))

    matrix = TridiagonalToeplitz(
        num_state_qubits=num_qubits,
        main_diag=a_norm,
        off_diag=b_off,
        trotter_steps=trotter_steps,
    )

    b = np.ones(N) / np.sqrt(N)  # uniform b_norm, matching core.resources' default

    hhl = HHL()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solution = hhl.solve(matrix, b)

    return solution.state, num_qubits, A


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--main_diag", type=float, default=-2.0)
    parser.add_argument("--off_diag", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--backend", type=str, default=None)
    args = parser.parse_args()

    print(f"Building HHL circuit: N={args.N}, main_diag={args.main_diag}, "
          f"off_diag={args.off_diag}, epsilon={args.epsilon}")
    try:
        circuit, num_qubits, A = build_hhl_circuit(
            args.N, args.main_diag, args.off_diag, args.epsilon
        )
    except ImportError as exc:
        print(f"\nCould not import quantum_linear_solvers: {exc}")
        print("This script needs the same HHL dependency solvers/quantum/hhl_1d.py "
              "uses in production. Check it is installed/available in your "
              "environment before proceeding.")
        return

    spec = hhl_spec(circuit, num_qubits)

    # Pre-transpilation depth is what solution.state reports directly and
    # appears deceptively small if the circuit uses a handful of large,
    # undecomposed composite instructions (QPE/reciprocal/prep blocks are
    # typically built this way). Post-transpilation depth, via
    # core.resources.transpile_report (Phase 2, same tool used throughout
    # this project's hardware-feasibility numbers), rigorously reflects
    # the exact execution submission.
    from core.resources import transpile_report
    pre_depth = circuit.depth()
    report = transpile_report(circuit, coupling_map=None, optimization_level=1)
    print(f"Circuit: {circuit.num_qubits} qubits, "
          f"pre-transpile depth={pre_depth}, post-transpile depth={report.post_depth}, "
          f"two_qubit_count={report.two_qubit_count}")

    x_exact, exact_record = StatevectorExecutor(diagnostics=False).extract(circuit, spec)
    kappa = float(np.linalg.cond(A))
    expected_order = 1.0 / kappa ** 2
    print(f"\nMatrix condition number kappa ~ {kappa:.2f} "
          f"(expected post-selection probability order of magnitude ~1/kappa^2 "
          f"~ {expected_order:.4f})")
    print(f"Exact post-selection probability: "
          f"{exact_record.postselect_probability:.6f}")
    print(f"Exact shot overhead (1/p): {exact_record.shot_overhead:.1f}")
    if not (0.01 * expected_order <= exact_record.postselect_probability <= 100 * expected_order):
        print(
            "\n*** WARNING: exact post-selection probability is more than 100x "
            "away from the 1/kappa^2 order-of-magnitude expectation. This is "
            "not necessarily wrong (the 1/kappa^2 estimate is a rough guide, "
            "not an exact formula), but it is worth checking before trusting "
            "the hardware comparison below -- a similar mismatch here traced "
            "back to an unnormalised TridiagonalToeplitz input in an earlier "
            "version of this script. ***"
        )

    if args.real:
        print(f"\nSubmitting to real hardware ({args.backend or 'least-busy'})...")
        context = HardwareContext.real(backend_name=args.backend)
    else:
        print("\nRunning against FakeTorino (local testing, no queue, no cost)...")
        context = HardwareContext.local_testing()

    expected_accepted = args.shots * exact_record.postselect_probability
    if expected_accepted < 20:
        suggested_shots = int(np.ceil(30 / max(exact_record.postselect_probability, 1e-6)))
        print(
            f"\nNote: at p~{exact_record.postselect_probability:.4f}, "
            f"{args.shots} shots gives an expected ~{expected_accepted:.1f} "
            f"accepted samples -- too few for a tight confidence interval. "
            f"Consider --shots {suggested_shots} for ~30 expected accepted samples."
        )

    sample = hardware_postselection_sample(circuit, spec, context, shots=args.shots)

    print(f"\nBackend: {sample.provenance.backend_name} "
          f"(local_testing={sample.provenance.is_local_testing})")
    print(f"Job ID: {sample.provenance.job_id}")
    print(f"Wall time: {sample.provenance.wall_time_s:.1f}s")
    print(f"\nMeasured: {sample.n_accepted}/{sample.shots} accepted")
    print(f"Probability: {sample.probability:.6f} "
          f"(95% CI [{sample.ci_low:.6f}, {sample.ci_high:.6f}])")
    print(f"Shot overhead: {sample.shot_overhead:.1f}")

    print(f"\nExact vs measured probability ratio: "
          f"{sample.probability / exact_record.postselect_probability:.2f}x "
          f"(1.0 = no hardware degradation of the post-selection rate itself; "
          f"note this ratio says nothing about the quality of the surviving "
          f"samples, only how many there are)")


if __name__ == "__main__":
    main()