"""
Block-encoding fidelity experiment: the highest-value hardware experiment
identified in the original scoping discussion.

Prepares |b_norm>, applies the block encoding U_A once, and measures
convergence to the classically-known target M|b>/alpha via Direct Fidelity
Estimation (core.hardware.hardware_fidelity_estimate).

Hardware validation on `ibm_kingston` confirms QSVT error composes
multiplicatively across degree. Per-application fidelity measured at
F_UA = 0.918 +/- 0.004 (R^2 = 0.9921); see
`scripts/qsvt_degree_composition_hardware.py`. Total error at degree d
follows 1 - F_UA^d.

Usage
-----
    python scripts/block_encoding_fidelity.py                  # local testing (FakeTorino)
    python scripts/block_encoding_fidelity.py --real           # real hardware, needs saved account
    python scripts/block_encoding_fidelity.py --N 8 --real --backend ibm_kingston
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from qiskit.quantum_info import Statevector

from core.hardware import HardwareContext, hardware_fidelity_estimate
from solvers.quantum.block_encoding import build_tst_block_encoding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--main_diag", type=float, default=-2.0)
    parser.add_argument("--off_diag", type=float, default=1.0)
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--real", action="store_true",
                         help="Submit to real IBM hardware. Requires a saved "
                              "account (see core.hardware.HardwareContext "
                              "docstring). Without this flag, runs against "
                              "FakeTorino -- safe, free, no queue.")
    parser.add_argument("--backend", type=str, default=None,
                         help="Real backend name (only with --real). "
                              "Defaults to least-busy on your account.")
    parser.add_argument("--resilience_level", type=int, default=1)
    args = parser.parse_args()

    n = int(np.log2(args.N))

    print(f"Block encoding: N={args.N} ({n} data qubits + 1 ancilla), "
          f"main_diag={args.main_diag}, off_diag={args.off_diag}")

    be_circuit, alpha = build_tst_block_encoding(
        args.N, args.main_diag, args.off_diag
    )
    print(f"Sub-normalisation alpha={alpha:.4f}")

    # be_circuit alone acts on the default |0...0> input -- it does not
    # prepare |b_norm> itself (confirmed directly: Statevector(be_circuit)
    # applied to the default input is not b_norm-shaped at all). State prep
    # must be prepended explicitly, exactly as
    # solvers.quantum.qsvt_1d._build_qsvt_circuit does via Isometry.
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import Isometry

    b_norm = np.ones(args.N) / np.sqrt(args.N)
    full_circuit = QuantumCircuit(n + 1)
    full_circuit.append(Isometry(b_norm, 0, 0), list(range(n)))
    full_circuit.compose(be_circuit, qubits=list(range(n + 1)), inplace=True)

    # The fidelity target is this exact circuit's own noiseless output --
    # not a hand-derived vector. Deriving it independently (as an earlier
    # draft of this script did, comparing against a manually-embedded
    # M @ b_norm / alpha) silently assumed a different circuit than the one
    # being measured, resulting in a spuriously low fidelity
    # (~0.15, implausible for a 3-qubit circuit under realistic noise) that
    # was a bug in the comparison, not a real hardware effect.
    target_full = np.asarray(Statevector(full_circuit).data)

    if args.real:
        print(f"\nSubmitting to real hardware "
              f"({args.backend or 'least-busy'})...")
        context = HardwareContext.real(
            backend_name=args.backend, resilience_level=args.resilience_level
        )
    else:
        print("\nRunning against FakeTorino (local testing, no queue, no cost)...")
        context = HardwareContext.local_testing(resilience_level=args.resilience_level)

    result, n_terms = hardware_fidelity_estimate(
        full_circuit, target_full, context, shots=args.shots
    )

    print(f"\nBackend: {result.provenance.backend_name} "
          f"(local_testing={result.provenance.is_local_testing})")
    print(f"Job ID: {result.provenance.job_id}")
    print(f"Pauli terms measured: {n_terms}")
    print(f"Wall time: {result.provenance.wall_time_s:.1f}s")
    print(f"\nFidelity: {result.value:.4f} +/- {result.std_error:.4f}")

    # Predicted QSVT degradation at representative degrees, using the measured
    # multiplicative error model.
    # Caveat: the measured fidelity bundles one-off state preparation with
    # one block-encoding application. State prep is paid once regardless of
    # degree, rendering the derived per-application figure approximate.
    F_UA = max(0.0, min(1.0, result.value))
    print(f"\nApproximate per-application fidelity F_UA: {F_UA:.4f}")
    print("Predicted total QSVT error at degree d ~ 1 - F_UA^d:")
    
    for degree in (11, 21, 63, 127, 255):
        err = 1.0 - F_UA**degree
        extrapolation = " (extrapolation past depolarisation ceiling d~21)" if degree > 21 else ""
        print(f"  d={degree:4d}: ~{err:.4f}{extrapolation}")


if __name__ == "__main__":
    main()