"""
HHL post-selection shot-overhead experiment.

Measures the real cost the original hardware-scoping discussion flagged for
HHL: post-selection succeeds with probability ~1/kappa^2, so obtaining one
usable sample requires ~kappa^2 total shots. This script measures that
overhead directly via core.hardware.hardware_postselection_sample, and
reports it alongside the exact (statevector) value for comparison.

IMPORTANT -- one part of this script is untested here
----------------------------------------------------------
Everything else in this Phase 5 delivery (core/hardware.py,
block_encoding_fidelity.py) was validated directly against FakeTorino in
this development environment. The HHL circuit construction below could not
be: quantum_linear_solvers (hhl_1d.py's dependency) is present in the repo
tree but its submodule contents were not populated in the sandbox this was
built in, so `HHL().solve(...)` was never actually run here. The code below
mirrors solvers.quantum.hhl_1d.hhl_solve_system's construction line for
line, and hhl_spec (core.execution, already validated) handles the
post-selection specification -- but please run this once yourself and sanity-
check the exact-probability printout against a value you trust (e.g. from an
existing HHL solve) before relying on the hardware numbers it produces.

Usage
-----
    python scripts/hhl_shot_overhead.py --N 4
    python scripts/hhl_shot_overhead.py --N 4 --real --backend ibm_kingston
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
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
    """
    from quantum_linear_solvers.linear_solvers.hhl import HHL
    from quantum_linear_solvers.linear_solvers.matrices.tridiagonal_toeplitz import (
        TridiagonalToeplitz,
    )

    num_qubits = int(np.log2(N))
    trotter_steps = max(1, int(np.ceil(1.0 / epsilon)))

    matrix = TridiagonalToeplitz(
        num_state_qubits=num_qubits,
        main_diag=main_diag,
        off_diag=off_diag,
        trotter_steps=trotter_steps,
    )

    b = np.ones(N) / np.sqrt(N)  # uniform b_norm, matching core.resources' default

    hhl = HHL()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solution = hhl.solve(matrix, b)

    return solution.state, num_qubits


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
        circuit, num_qubits = build_hhl_circuit(
            args.N, args.main_diag, args.off_diag, args.epsilon
        )
    except ImportError as exc:
        print(f"\nCould not import quantum_linear_solvers: {exc}")
        print("This script needs the same HHL dependency solvers/quantum/hhl_1d.py "
              "uses in production. Check it is installed/available in your "
              "environment before proceeding.")
        return

    spec = hhl_spec(circuit, num_qubits)
    print(f"Circuit: {circuit.num_qubits} qubits, depth {circuit.depth()}")

    x_exact, exact_record = StatevectorExecutor(diagnostics=False).extract(circuit, spec)
    print(f"\nExact post-selection probability: "
          f"{exact_record.postselect_probability:.6f}")
    print(f"Exact shot overhead (1/p): {exact_record.shot_overhead:.1f}")

    if args.real:
        print(f"\nSubmitting to real hardware ({args.backend or 'least-busy'})...")
        context = HardwareContext.real(backend_name=args.backend)
    else:
        print("\nRunning against FakeTorino (local testing, no queue, no cost)...")
        context = HardwareContext.local_testing()

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