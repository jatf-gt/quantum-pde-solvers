"""
Robustness programme, steps 1-3: shot noise, depolarising sweep, fake backend.

Answers the "before spending QPU time, how does algorithmic error compose
with device error" question directly, entirely on CX3 or a laptop.

Usage
-----
    python scripts/robustness_sweep_1d.py
    python scripts/robustness_sweep_1d.py --N 8 --degree 21 --out results/robustness/

Steps run, in the order the robustness programme specifies:

1. Shot noise only -- genuine finite-shot post-selection statistics via
   sample_postselection, no gate error, isolating sampling cost.
2. Parametric depolarising sweep -- error-vs-fidelity curve via
   depolarizing_sweep, both with and without the realistic single-qubit
   floor, so the floor's own contribution is visible rather than folded in.
3. Fake backend (real calibration data) -- FakeTorino, the newest Heron-
   generation snapshot reachable under this project's qiskit==1.4.5 pin
   (see core.noise.fake_backend_noise_model for why not FakeKingston).

Step 4 (real hardware) is out of scope for this script by construction --
see core/execution.py's HardwareExecutor (Phase 5) once available.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from core.execution import qsvt_spec, StatevectorExecutor
from core.noise import (
    HERON_R2_SINGLE_QUBIT_ERROR,
    depolarizing_sweep,
    sample_postselection,
)
from core.resources import HERON_R2_TWO_QUBIT_ERROR


def build_qsvt_circuit(N: int, degree: int, seed: int = 0):
    """
    Build a QSVT circuit at production shape but with synthetic phase
    angles, exactly as core.resources.validate_composability does. This
    sweep is about noise response, not inversion accuracy, so the specific
    (non-degenerate) angle values don't matter -- see core/resources.py's
    module docstring for the same reasoning applied to gate counting.
    """
    from solvers.quantum.block_encoding import build_tst_block_encoding
    from solvers.quantum.qsvt_1d import _build_qsvt_circuit

    n = int(np.log2(N))
    be_circuit, _alpha = build_tst_block_encoding(N, main_diag=-2.0, off_diag=1.0)
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.1, 3.0, size=degree + 1)
    b_norm_vec = np.ones(N) / np.sqrt(N)
    return _build_qsvt_circuit(be_circuit, angles, n, b_norm_vec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--degree", type=int, default=11,
                         help="QSVT polynomial degree (default 11, tractable "
                              "for a direct density-matrix sweep at N=4)")
    parser.add_argument("--shots", type=int, default=8192)
    parser.add_argument("--out", type=Path, default=Path("results/robustness_1d"))
    args = parser.parse_args()

    qc   = build_qsvt_circuit(args.N, args.degree)
    spec = qsvt_spec(int(np.log2(args.N)), 1)

    print(f"QSVT circuit: N={args.N}, degree={args.degree}, "
          f"{qc.num_qubits} qubits, pre-transpile depth={qc.depth()}\n")

    # -- Step 1: shot noise only --------------------------------------------
    print("Step 1: shot noise only (no gate error)")
    x_exact, rec_exact = StatevectorExecutor(diagnostics=False).extract(qc, spec)
    sample = sample_postselection(qc, spec, shots=args.shots)
    print(f"  exact postselect probability : {rec_exact.postselect_probability:.4f}")
    print(f"  {args.shots}-shot estimate    : {sample.probability:.4f} "
          f"(95% CI [{sample.ci_low:.4f}, {sample.ci_high:.4f}])")
    print(f"  shot overhead (1/p)          : {sample.shot_overhead:.1f}\n")

    # -- Step 2: parametric depolarising sweep ------------------------------
    print("Step 2: parametric depolarising sweep")
    error_rates = [0.0, 1e-3, HERON_R2_TWO_QUBIT_ERROR, 5e-3, 1e-2, 2e-2, 5e-2]

    print("  with realistic single-qubit floor (heron_r2 = "
          f"{HERON_R2_SINGLE_QUBIT_ERROR:.1e}):")
    rows_floor = depolarizing_sweep(
        qc, spec, error_rates=error_rates,
        single_qubit_error=HERON_R2_SINGLE_QUBIT_ERROR,
        reference_amplitudes=x_exact,
    )
    for r in rows_floor:
        print(f"    p2={r['two_qubit_error']:.4f}  purity={r['purity']:.4f}  "
              f"fidelity={r['fidelity_vs_ideal']:.4f}")

    print("  true zero single-qubit baseline (isolates two-qubit error alone):")
    rows_clean = depolarizing_sweep(
        qc, spec, error_rates=error_rates, single_qubit_error=0.0,
        reference_amplitudes=x_exact,
    )
    for r in rows_clean:
        print(f"    p2={r['two_qubit_error']:.4f}  purity={r['purity']:.4f}  "
              f"fidelity={r['fidelity_vs_ideal']:.4f}")
    print()

    # -- Step 3: fake backend (real calibration data) -----------------------
    print("Step 3: fake backend (real IBM calibration data)")
    try:
        from core.noise import fake_backend_noise_model, NoiseExecutor
        noise_model = fake_backend_noise_model("FakeTorino")
        x_fake, rec_fake = NoiseExecutor(noise_model=noise_model).extract(qc, spec)
        denom = np.linalg.norm(x_exact) * np.linalg.norm(x_fake)
        fidelity = float((np.dot(x_exact, x_fake) / denom) ** 2) if denom > 0 else 0.0
        print(f"  FakeTorino (Heron r1 calibration snapshot):")
        print(f"    purity   = {rec_fake.extra['purity']:.4f}")
        print(f"    fidelity = {fidelity:.4f}")
    except ImportError as exc:
        print(f"  Skipped: {exc}")
    print()

    # -- Save -------------------------------------------------------------
    args.out.mkdir(parents=True, exist_ok=True)
    import json
    payload = {
        "N": args.N, "degree": args.degree,
        "shot_sample": {
            "shots": sample.shots, "n_accepted": sample.n_accepted,
            "probability": sample.probability,
            "ci": [sample.ci_low, sample.ci_high],
            "exact_probability": rec_exact.postselect_probability,
        },
        "depolarizing_sweep_with_floor": rows_floor,
        "depolarizing_sweep_clean":      rows_clean,
    }
    out_file = args.out / "robustness_results.json"
    out_file.write_text(json.dumps(payload, indent=2))
    print(f"Written: {out_file}")


if __name__ == "__main__":
    main()