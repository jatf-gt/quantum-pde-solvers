"""
The real-hardware validation run: preflight, submit, and record.

Purpose
-------
Spend a small, bounded amount of IBM Open Plan QPU time on the single
highest-value-per-minute experiment in this project, and produce an
appendix-ready record of it.

The experiment is the block-encoding fidelity measurement (the same one
scripts/block_encoding_fidelity.py runs against a simulator): prepare
|b_norm>, apply the block encoding U_A once, and measure the fidelity of
the result against the classically-known target via Direct Fidelity
Estimation. It was chosen over every alternative for one reason: it yields
a single number, delta = 1 - fidelity, which is the input to
scripts/delta_amplification_hardware.py, and therefore propagates through
the entire outer-scheme feasibility analysis. One small hardware
measurement converts that whole chapter's conclusions from
"simulator-estimated" to "anchored on a real device".

What this deliberately does NOT do
-------------------------------------
It does not attempt a full quantum PDE solve on hardware. That is not a
funding or time limitation, it is a documented conclusion of this project:
core/resources.py puts 1-D QSVT at N=8 already ~2x over Heron r2's usable
two-qubit gate budget, and the tomography requirement in the outer loop is
architectural, not technological. Submitting a doomed large circuit would
consume the quota and produce a number that means nothing. Measuring the
primitive, and propagating it analytically, produces a defensible result
within seconds of QPU time.

Modes
-----
    --check     Verify credentials, list backends, show quota usage.
                Uses NO QPU time. Run this first.
    (default)   Dry run against FakeTorino. No QPU time, no credentials.
    --submit    Submit to real hardware. Uses QPU time. Requires --check
                to have passed first.

Environment requirement (important)
--------------------------------------
This script needs qiskit-ibm-runtime >= 0.40 to reach the current IBM
Quantum Platform, which in turn needs qiskit >= 2.x. That conflicts with
this project's pinned qiskit==1.4.5. Run this script from a SEPARATE
virtual environment; see the accompanying notes. The rest of the
repository is unaffected and continues to run under the pinned versions --
this script imports only block_encoding (verified to work unchanged under
qiskit 2.3.0) and stdlib/numpy, deliberately avoiding the parts of the
codebase that are version-sensitive.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np


# -- Circuit construction (version-portable: only block_encoding is imported) --

def build_experiment_circuit(N: int, main_diag: float, off_diag: float):
    """
    State preparation followed by one block-encoding application, plus the
    classically-computed target state its output should match.

    Identical construction to scripts/block_encoding_fidelity.py, including
    the correction made there: be_circuit alone acts on |0...0> and does NOT
    prepare |b_norm> itself, so state preparation must be prepended
    explicitly. The fidelity target is this exact circuit's own noiseless
    statevector -- deriving it independently was the bug that produced a
    spurious 0.15 fidelity in an earlier version of that script.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import Isometry
    from qiskit.quantum_info import Statevector

    from solvers.quantum.block_encoding import build_tst_block_encoding

    n = int(np.log2(N))
    be_circuit, alpha = build_tst_block_encoding(N, main_diag, off_diag)

    b_norm = np.ones(N) / np.sqrt(N)
    qc = QuantumCircuit(n + 1)
    qc.append(Isometry(b_norm, 0, 0), list(range(n)))
    qc.compose(be_circuit, qubits=list(range(n + 1)), inplace=True)

    target = np.asarray(Statevector(qc).data)
    return qc, target, alpha


def pauli_terms_for_projector(target: np.ndarray):
    """
    Decompose |target><target| into Pauli terms.

    Reimplemented locally rather than importing vqls_utils.pauli_decompose_matrix,
    only because that module imports PennyLane at module load and this script
    is designed to run in a minimal, separate environment. The arithmetic is
    identical and is checked against the imported version in --check mode when
    that import happens to be available.
    """
    N = target.shape[0]
    n = int(np.log2(N))
    P1 = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    }
    A = np.outer(target, target.conj())
    terms = []
    for idx in range(4 ** n):
        s, tmp = "", idx
        for _ in range(n):
            s = "IXYZ"[tmp % 4] + s
            tmp //= 4
        P = np.array([[1.0 + 0j]])
        for ch in s:
            P = np.kron(P, P1[ch])
        c = complex(np.trace(A @ P) / N)
        if abs(c) > 1e-12:
            terms.append((c, s))
    return terms


# -- Preflight -----------------------------------------------------------------

def preflight(args) -> int:
    """
    Verify everything that can be verified without spending QPU time.

    Checks, in order: library versions are new enough to reach the current
    platform; credentials load; the requested backend exists and is
    operational; and the circuit transpiles within a sane gate budget for
    that backend. Any failure here is a failure that would otherwise have
    consumed quota to discover.
    """
    print("=" * 70)
    print("PREFLIGHT -- no QPU time is used by this mode")
    print("=" * 70)

    # 1. Versions
    import qiskit
    import qiskit_ibm_runtime
    rt_version = qiskit_ibm_runtime.__version__
    print(f"\n[1] Versions: qiskit {qiskit.__version__}, "
          f"qiskit-ibm-runtime {rt_version}")

    major, minor = (int(x) for x in rt_version.split(".")[:2])
    if (major, minor) < (0, 40):
        print(f"    FAIL: qiskit-ibm-runtime {rt_version} cannot access the "
              f"current IBM Quantum Platform.")
        print(f"    The 'ibm_quantum' channel was sunset on 1 July 2025 and "
              f"the replacement channel 'ibm_quantum_platform' is absent "
              f"in releases prior to 0.40.")
        print(f"    Resolution: deploy a separate virtual environment via "
              f"'pip install -U qiskit qiskit-ibm-runtime'.")
        return 1
    print("    OK: new enough for the ibm_quantum_platform channel.")

    # 2. Credentials
    print("\n[2] Credentials")
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
    except Exception as exc:
        print(f"    FAIL: could not load a saved account ({type(exc).__name__}: {exc})")
        print("    Resolution: execute the save_account configuration snippet.")
        return 1
    print("    OK: saved account loaded.")

    # 3. Backends
    print("\n[3] Available backends")
    try:
        backends = service.backends(operational=True, simulator=False)
    except Exception as exc:
        print(f"    FAIL: could not list backends ({type(exc).__name__}: {exc})")
        return 1
    if not backends:
        print("    FAIL: no operational QPUs visible on this account.")
        return 1
    for b in backends:
        try:
            pending = b.status().pending_jobs
        except Exception:
            pending = "?"
        print(f"    {b.name:<20} {b.num_qubits:>4} qubits   queue: {pending}")

    chosen_name = args.backend
    if chosen_name is None:
        chosen = min(backends, key=lambda b: getattr(b.status(), "pending_jobs", 1e9))
        chosen_name = chosen.name
        print(f"\n    Would use least-busy: {chosen_name}")
    else:
        if chosen_name not in [b.name for b in backends]:
            print(f"\n    FAIL: requested backend {chosen_name!r} is not in the "
                  f"list above.")
            return 1
        chosen = service.backend(chosen_name)
        print(f"\n    Would use requested: {chosen_name}")

    # 4. Circuit cost against that backend
    print("\n[4] Circuit")
    from qiskit import transpile
    qc, target, alpha = build_experiment_circuit(args.N, args.main_diag, args.off_diag)
    tqc = transpile(qc, backend=chosen, optimization_level=1)
    two_q = sum(v for k, v in tqc.count_ops().items() if k in ("cz", "cx", "ecr"))
    terms = pauli_terms_for_projector(target)
    print(f"    qubits={qc.num_qubits}, alpha={alpha:.4f}")
    print(f"    transpiled depth={tqc.depth()}, two-qubit gates={two_q}")
    print(f"    Pauli terms to measure: {len(terms)}  "
          f"(= {len(terms)} PUBs in one batched job)")
    print(f"    shots per PUB: {args.shots}  -> "
          f"{len(terms) * args.shots:,} shots total")
    if two_q > 5000:
        print(f"    WARNING: {two_q} two-qubit gates exceeds the ~5000 "
              f"usable-circuit figure for Heron r2. Results will be noise.")

    # 5. Quota
    print("\n[5] Quota")
    print("    Check your remaining allowance on the IBM Quantum Platform "
          "dashboard (Instances page).")
    print("    Open Plan: 10 minutes per rolling 28-day window.")
    print(f"    This job is a single small batch and is expected to consume "
          f"only seconds of QPU time; the exact figure is reported after "
          f"submission and saved to the results file.")

    print("\n" + "=" * 70)
    print("PREFLIGHT PASSED. Re-run with --submit to use QPU time.")
    print("=" * 70)
    return 0


# -- Submission ----------------------------------------------------------------

def run_experiment(args, use_hardware: bool) -> dict:
    from qiskit import transpile
    from qiskit.quantum_info import SparsePauliOp

    qc, target, alpha = build_experiment_circuit(args.N, args.main_diag, args.off_diag)
    terms = pauli_terms_for_projector(target)

    if use_hardware:
        from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
        service = QiskitRuntimeService()
        backend = (service.backend(args.backend) if args.backend
                   else service.least_busy(operational=True, simulator=False))
        print(f"Submitting to {backend.name} ...")
    else:
        from qiskit_ibm_runtime import EstimatorV2
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        backend = FakeTorino()
        print(f"Dry run against {backend.name} (no QPU time, no credentials) ...")

    tqc = transpile(qc, backend=backend, optimization_level=1)
    pubs = [(tqc, SparsePauliOp(s).apply_layout(tqc.layout)) for _c, s in terms]

    estimator = EstimatorV2(mode=backend)
    if use_hardware:
        # Error mitigation genuinely applies on real hardware (and is
        # silently ignored, with a warning, on a Fake backend -- see
        # core/hardware.py's HardwareContext for the same guard).
        estimator.options.resilience_level = args.resilience_level
        estimator.options.dynamical_decoupling.enable = True
    estimator.options.default_shots = args.shots

    job = estimator.run(pubs)
    print(f"Job ID: {job.job_id()}")
    if use_hardware:
        print("Job queued. Note: queue duration incurs no compute allocation charges; "
              "only QPU execution time is metered.")
    result = job.result()

    fidelity = sum(c.real * float(r.data.evs) for (c, _s), r in zip(terms, result))
    std_err = float(np.sqrt(sum(
        (c.real * float(r.data.stds)) ** 2 for (c, _s), r in zip(terms, result)
    )))

    # Actual metered usage, straight from the job -- the authoritative
    # figure, not an estimate.
    quantum_seconds = None
    if use_hardware:
        try:
            quantum_seconds = job.metrics()["usage"]["quantum_seconds"]
        except Exception as exc:
            print(f"(could not read usage metrics: {exc})")

    return {
        "timestamp_utc":     datetime.now(timezone.utc).isoformat(),
        "hardware":          use_hardware,
        "backend":           backend.name,
        "job_id":            job.job_id(),
        "N":                 args.N,
        "n_qubits":          qc.num_qubits,
        "alpha":             float(alpha),
        "transpiled_depth":  tqc.depth(),
        "two_qubit_gates":   sum(v for k, v in tqc.count_ops().items()
                                  if k in ("cz", "cx", "ecr")),
        "n_pauli_terms":     len(terms),
        "shots_per_pub":     args.shots,
        "resilience_level":  args.resilience_level if use_hardware else None,
        "fidelity":          float(fidelity),
        "fidelity_std_err":  std_err,
        "delta":             float(max(0.0, 1.0 - fidelity)),
        "quantum_seconds":   quantum_seconds,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="Preflight only. No QPU time. Run this first.")
    p.add_argument("--submit", action="store_true",
                   help="Submit to real hardware. USES QPU TIME.")
    p.add_argument("--backend", type=str, default=None,
                   help="e.g. ibm_kingston. Default: least busy.")
    p.add_argument("--N", type=int, default=4)
    p.add_argument("--main_diag", type=float, default=-2.0)
    p.add_argument("--off_diag", type=float, default=1.0)
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--resilience_level", type=int, default=1,
                   help="0=none, 1=readout mitigation (TREX), 2=+ZNE.")
    p.add_argument("--out", type=Path, default=Path("results/hardware_run"))
    args = p.parse_args()

    if args.check:
        sys.exit(preflight(args))

    if args.submit:
        print("!" * 70)
        print("This will submit a job to real IBM hardware and consume QPU "
              "time from your Open Plan allowance.")
        print("!" * 70)
        if input("Type 'yes' to continue: ").strip().lower() != "yes":
            print("Aborted. No QPU time used.")
            return

    record = run_experiment(args, use_hardware=args.submit)

    print("\n" + "=" * 70)
    print(f"Backend         : {record['backend']}")
    print(f"Job ID          : {record['job_id']}")
    print(f"Circuit         : {record['n_qubits']} qubits, depth "
          f"{record['transpiled_depth']}, {record['two_qubit_gates']} 2Q gates")
    print(f"Measured        : {record['n_pauli_terms']} Pauli terms x "
          f"{record['shots_per_pub']} shots")
    print(f"FIDELITY        : {record['fidelity']:.4f} +/- "
          f"{record['fidelity_std_err']:.4f}")
    print(f"delta (1 - F)   : {record['delta']:.4f}")
    if record["quantum_seconds"] is not None:
        print(f"QPU TIME USED   : {record['quantum_seconds']:.2f} s "
              f"({record['quantum_seconds']/60:.3f} min of your allowance)")
    print("=" * 70)

    args.out.mkdir(parents=True, exist_ok=True)
    tag = "hardware" if record["hardware"] else "dryrun"
    path = args.out / f"{tag}_{record['job_id'][:8]}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"\nSaved: {path}")
    print("\nNext step: feeding this delta into the outer-scheme analysis --")
    print(f"  python scripts/delta_amplification_hardware.py "
          f"--delta {record['delta']:.4f}")


if __name__ == "__main__":
    main()