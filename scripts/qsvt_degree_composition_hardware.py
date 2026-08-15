"""
Does hardware error compose as F^d across QSVT degree? A real-device test
of the extrapolation model this entire project's feasibility analysis rests on.

The question
------------
Every hardware-feasibility claim in this project extrapolates from a single
block-encoding application to a degree-d QSVT circuit:

    core/resources.py          composes two-qubit gate counts as d * count(U_A)
    qsvt_2d_line_degree_sweep  models total error as 1 - F_1^d
    block_encoding_fidelity    predicts QSVT error at degree d as d * (1 - F_1)

All three assume errors from repeated applications of the same unitary
accumulate independently. That assumption has been flagged as unvalidated in
each of those modules' docstrings and has never been checked against a real
device -- only against simulators whose noise models are, by construction,
built from independent per-gate channels and therefore cannot possibly
falsify it.

This script measures F_d directly, for d = 1, 3, 5, ..., on the actual 2-D
line row operator, and compares against the F_1^d prediction. Three outcomes,
all worth reporting:

    F_d ~= F_1^d   The extrapolations are sound; say so with evidence.
    F_d >  F_1^d   Coherent errors partially cancel across repetitions. A
                   positive result: QSVT is more hardware-robust than an
                   independent-error model predicts, and every feasibility
                   limit in this project is conservative.
    F_d <  F_1^d   Errors compound worse than modelled; the feasibility
                   limits are optimistic and should be restated.

Why the row operator, not the 1-D Poisson operator
-----------------------------------------------------
A_row carries the -2/dy^2 diagonal shift from transverse coupling, pinning
kappa(A_row) -> 3 independent of N (measured: 2.36 at Nx=4). This is the
only operator in the project whose QSVT circuit is shallow enough to run at
several degrees within a small QPU budget, and it is the one the whole 2-D/
3-D architecture actually uses. Measuring the composition law here is
therefore both affordable and directly relevant.

Cost control
------------
QPU time is metered per second of execution, not per job, and every mode
below reports actual usage from job.metrics(). A reference point from this
project: 20 PUBs x 4096 shots on a 3-qubit circuit consumed 34 s on
ibm_kingston. Budget accordingly, and note that resilience_level=2 (ZNE)
multiplies execution time roughly 3x because it runs each circuit at
several noise amplification factors.

Modes
-----
    --calibration   Dump the backend's current calibration snapshot.
                    Uses NO QPU time. Run this on the same day as any
                    hardware job: device error rates drift daily, and a
                    fidelity number is not interpretable without the
                    device state that produced it.
    (default)       Dry run against FakeTorino. No QPU time, no credentials.
    --submit        Real hardware. Prints a budget estimate and requires
                    typed confirmation before spending anything.
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


# ── Row operator (local, to keep this script importable in a minimal env) ─────

def row_operator(Nx: int, Ly_over_Lx: float = 1.0):
    """
    The 2-D line/strip operator from problems/poisson_line_2d.py, rebuilt
    locally so this script runs in the separate qiskit 2.x environment
    without importing the PennyLane-dependent parts of the codebase.

        A_row = tridiag(1/dx^2, -2(1/dx^2 + 1/dy^2), 1/dx^2)

    Matches PoissonLine2D._build_row_matrix exactly for a square grid
    (Nx = Ny, Lx = Ly), which is the benchmark case used throughout.
    """
    dx = 1.0 / (Nx + 1)
    dy = Ly_over_Lx / (Nx + 1)
    a = -2.0 * (1.0 / dx ** 2 + 1.0 / dy ** 2)
    b = 1.0 / dx ** 2
    A = (a * np.eye(Nx)
         + b * (np.diag(np.ones(Nx - 1), 1) + np.diag(np.ones(Nx - 1), -1)))
    e = np.abs(np.linalg.eigvalsh(A))
    return A, float(e.max() / e.min())


def build_degree_circuit(Nx: int, degree: int, seed: int = 0):
    """
    A QSVT-shaped circuit at the given degree: state preparation, then the
    block encoding applied `degree` times with single-qubit Rz rotations
    between applications.

    Uses synthetic, non-degenerate phase angles rather than real QSP angles.
    This is deliberate and matters for the validity of the measurement: the
    question here is how *hardware error* composes across repeated
    applications of U_A, which depends on the gate sequence, not on the
    particular rotation angles between them. Real QSP angles would need
    pyqsp in this minimal environment and would change nothing about the
    error-composition physics. The same reasoning, and the same choice, is
    documented in core/resources.py::validate_composability.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import Isometry
    from qiskit.quantum_info import Statevector

    from solvers.quantum.block_encoding import build_tst_block_encoding

    A, _kappa = row_operator(Nx)
    n = int(np.log2(Nx))
    be, alpha = build_tst_block_encoding(Nx, float(A[0, 0]), float(A[0, 1]))
    be_gate = be.to_gate(label="U_A")

    rng = np.random.default_rng(seed)
    angles = rng.uniform(0.1, 3.0, size=degree + 1)

    b = np.ones(Nx) / np.sqrt(Nx)
    qc = QuantumCircuit(n + 1)
    qc.append(Isometry(b, 0, 0), list(range(n)))
    qc.rz(float(angles[0]), n)
    for k in range(degree):
        qc.append(be_gate, list(range(n + 1)))
        qc.rz(float(angles[k + 1]), n)

    return qc, np.asarray(Statevector(qc).data), alpha


def pauli_terms_for_projector(target: np.ndarray):
    """Decompose |target><target| into Pauli terms (see ibm_hardware_run.py)."""
    N = target.shape[0]
    n = int(np.log2(N))
    P1 = {"I": np.eye(2, dtype=complex),
          "X": np.array([[0, 1], [1, 0]], dtype=complex),
          "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
          "Z": np.array([[1, 0], [0, -1]], dtype=complex)}
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


# ── Calibration provenance (zero QPU cost) ────────────────────────────────────

def capture_calibration(args) -> int:
    """
    Record the backend's current calibration.

    Device error rates drift day to day, so a fidelity measurement is only
    interpretable alongside the device state that produced it. Run this on
    the same day as any hardware job and keep the output with the results;
    it is what makes the measurement reproducible-in-principle and
    citable in a thesis appendix.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService

    service = QiskitRuntimeService()
    backend = (service.backend(args.backend) if args.backend
               else service.least_busy(operational=True, simulator=False))
    props = backend.properties()

    record = {
        "captured_utc":   datetime.now(timezone.utc).isoformat(),
        "backend":        backend.name,
        "num_qubits":     backend.num_qubits,
        "basis_gates":    sorted(backend.operation_names),
    }
    if props is not None:
        record["calibration_timestamp"] = str(getattr(props, "last_update_date", None))
        one_q, two_q, readout, t1s, t2s = [], [], [], [], []
        for q in range(backend.num_qubits):
            for name, store in (("sx", one_q), ("readout_error", readout)):
                try:
                    store.append(props.gate_error(name, q) if name == "sx"
                                 else props.readout_error(q))
                except Exception:
                    pass
            for name, store in (("T1", t1s), ("T2", t2s)):
                try:
                    store.append(getattr(props, name.lower())(q))
                except Exception:
                    pass
        for gate in ("cz", "ecr", "cx"):
            for pair in getattr(backend, "coupling_map", []) or []:
                try:
                    two_q.append(props.gate_error(gate, list(pair)))
                except Exception:
                    pass
            if two_q:
                record["two_qubit_gate"] = gate
                break

        def _stats(v):
            return ({"median": float(np.median(v)), "min": float(np.min(v)),
                     "max": float(np.max(v)), "n": len(v)} if v else None)

        record["single_qubit_error_sx"] = _stats(one_q)
        record["two_qubit_error"]       = _stats(two_q)
        record["readout_error"]         = _stats(readout)
        record["T1_us"]  = _stats([t * 1e6 for t in t1s]) if t1s else None
        record["T2_us"]  = _stats([t * 1e6 for t in t2s]) if t2s else None

    print(json.dumps(record, indent=2))
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = args.out / f"calibration_{backend.name}_{stamp}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"\nSaved: {path}")
    print("Keep this alongside your fidelity results -- it is the device "
          "state they were measured against.")
    return 0


# ── The sweep ─────────────────────────────────────────────────────────────────

def run_sweep(args, use_hardware: bool) -> dict:
    from qiskit import transpile
    from qiskit.quantum_info import SparsePauliOp
    from qiskit_ibm_runtime import EstimatorV2

    if use_hardware:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()
        backend = (service.backend(args.backend) if args.backend
                   else service.least_busy(operational=True, simulator=False))
    else:
        from qiskit_ibm_runtime.fake_provider import FakeTorino
        backend = FakeTorino()

    _A, kappa = row_operator(args.Nx)
    print(f"Row operator: Nx={args.Nx}, kappa(A_row)={kappa:.4f}")
    print(f"Backend: {backend.name}"
          f"{'' if use_hardware else '  (dry run -- no QPU time)'}\n")

    rows, total_seconds = [], 0.0
    for degree in args.degrees:
        qc, target, _alpha = build_degree_circuit(args.Nx, degree)
        terms = pauli_terms_for_projector(target)
        tqc = transpile(qc, backend=backend, optimization_level=1)
        two_q = sum(v for k, v in tqc.count_ops().items()
                    if k in ("cz", "cx", "ecr"))

        pubs = [(tqc, SparsePauliOp(s).apply_layout(tqc.layout)) for _c, s in terms]
        est = EstimatorV2(mode=backend)
        if use_hardware:
            est.options.resilience_level = args.resilience_level
            est.options.dynamical_decoupling.enable = True
        est.options.default_shots = args.shots

        job = est.run(pubs)
        result = job.result()
        fid = sum(c.real * float(r.data.evs) for (c, _s), r in zip(terms, result))
        err = float(np.sqrt(sum((c.real * float(r.data.stds)) ** 2
                                for (c, _s), r in zip(terms, result))))

        qs = None
        if use_hardware:
            try:
                qs = job.metrics()["usage"]["quantum_seconds"]
                total_seconds += qs
            except Exception:
                pass

        rows.append({"degree": degree, "two_qubit_gates": two_q,
                     "transpiled_depth": tqc.depth(),
                     "fidelity": float(fid), "fidelity_std_err": err,
                     "job_id": job.job_id(), "quantum_seconds": qs})
        print(f"  d={degree:3d}  2Q={two_q:5d}  F={fid:.4f} +/- {err:.4f}"
              + (f"  ({qs:.1f}s)" if qs else ""))

    return {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "hardware": use_hardware, "backend": backend.name,
            "Nx": args.Nx, "kappa_row": kappa, "shots": args.shots,
            "resilience_level": args.resilience_level if use_hardware else None,
            "rows": rows, "total_quantum_seconds": total_seconds}


def analyse(record: dict) -> None:
    """
    Test whether hardware error composes multiplicatively across degree.

    If errors from repeated U_A applications accumulate independently, then

        F_d = F_prep * (F_UA)^d      i.e.    ln F_d = ln F_prep + d * ln F_UA

    so ln F is LINEAR in d. That linearity is the testable content of the
    model, and a least-squares fit of ln F against d is the right way to
    test it -- both because it uses every sweep point rather than two, and
    because the fit quality (R^2) is itself the answer.

    Two earlier versions of this function got the baseline wrong, in ways
    worth recording since both produced confident, opposite, and false
    findings on a simulator whose noise is independent by construction:

      1. Modelling F_d as F_1^d raised the ONE-OFF state-preparation error
         to the d-th power, manufacturing a spurious "errors cancel"
         verdict (ratio rising to 2.0 at d=7).
      2. Correcting that with a two-point estimate F_UA = F_1/F_0 was
         unstable: at Nx=4 the d=1 circuit adds only ~19 two-qubit gates on
         top of a state preparation that already dominates the error, so
         F_1 and F_0 are nearly equal and their ratio came out as 1.0043 --
         a per-application fidelity above 1, which is unphysical, and which
         then made every higher degree look anomalously "WORSE".

    The fit below avoids both failure modes. Degree 0 (state preparation
    alone, zero two-qubit gates) is included so the intercept is measured
    rather than assumed.
    """
    rows = sorted(record["rows"], key=lambda r: r["degree"])
    pts = [(r["degree"], r["fidelity"]) for r in rows if r["fidelity"] > 1e-6]
    if len(pts) < 3:
        print("\nNeed at least 3 sweep points with positive fidelity to fit.")
        return

    d = np.array([p[0] for p in pts], dtype=float)
    F = np.array([p[1] for p in pts], dtype=float)
    lnF = np.log(F)

    slope, intercept = np.polyfit(d, lnF, 1)
    pred = intercept + slope * d
    ss_res = float(np.sum((lnF - pred) ** 2))
    ss_tot = float(np.sum((lnF - lnF.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan")

    F_prep = float(np.exp(intercept))
    F_UA = float(np.exp(slope))

    print(f"\nLeast-squares fit of ln F against degree:")
    print(f"  intercept -> state-prep fidelity F_prep = {F_prep:.4f}")
    print(f"  slope     -> per-application  F_UA      = {F_UA:.4f}")
    print(f"  R^2 (linearity of ln F in d)            = {r2:.4f}")

    print(f"\n{'degree':>7} {'F_measured':>11} {'F_fitted':>10} {'residual':>10}")
    for dd, ff, pp in zip(d, F, np.exp(pred)):
        print(f"{int(dd):7d} {ff:11.4f} {pp:10.4f} {ff - pp:+10.4f}")

    print()
    if not np.isfinite(r2):
        print("Fit degenerate; cannot conclude.")
    elif r2 > 0.98:
        print(f"FINDING: ln F is linear in degree to R^2 = {r2:.4f}. Hardware "
              f"error composes multiplicatively across repeated U_A "
              f"applications, exactly as the independent-error model in "
              f"core/resources.py and qsvt_2d_line_degree_sweep.py assumes. "
              f"Those extrapolations are VALIDATED on this backend for this "
              f"circuit family, with per-application fidelity {F_UA:.4f}.")
    elif r2 > 0.90:
        print(f"FINDING: ln F is approximately linear in degree "
              f"(R^2 = {r2:.4f}), with visible curvature. The multiplicative "
              f"model is a reasonable but imperfect description; quote "
              f"F_UA = {F_UA:.4f} with that caveat.")
    else:
        curve = "above" if float(np.sum(F - np.exp(pred))) > 0 else "below"
        print(f"FINDING: ln F is NOT linear in degree (R^2 = {r2:.4f}); "
              f"measured fidelity falls {curve} the multiplicative model. "
              f"Errors from repeated U_A applications do not accumulate "
              f"independently on this backend, so the F^d extrapolation "
              f"used throughout this project does not hold for this circuit "
              f"family and should be replaced by the measured curve.")

    print("\nSanity check: F_UA must lie in (0, 1]. A fitted value above 1, "
          "or an R^2 far from 1 on a SIMULATOR (whose noise is independent "
          "per gate by construction), indicates a problem with the analysis "
          "rather than a physical finding -- both failure modes are "
          "documented in this function's docstring.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--calibration", action="store_true",
                   help="Dump backend calibration. No QPU time.")
    p.add_argument("--submit", action="store_true", help="USES QPU TIME.")
    p.add_argument("--backend", type=str, default=None)
    p.add_argument("--Nx", type=int, default=4)
    p.add_argument("--degrees", type=int, nargs="+", default=[0, 1, 3, 5, 7])
    p.add_argument("--shots", type=int, default=4096)
    p.add_argument("--resilience_level", type=int, default=1,
                   help="0=none, 1=readout mitigation, 2=+ZNE (~3x the time).")
    p.add_argument("--seconds_per_job", type=float, default=34.0,
                   help="Measured cost of one comparable job, for the budget "
                        "estimate. Default from this project's ibm_kingston run.")
    p.add_argument("--out", type=Path, default=Path("results/degree_composition"))
    args = p.parse_args()

    if args.calibration:
        sys.exit(capture_calibration(args))

    if args.submit:
        est = len(args.degrees) * args.seconds_per_job
        if args.resilience_level >= 2:
            est *= 3.0
        print("!" * 70)
        print(f"{len(args.degrees)} jobs at resilience_level="
              f"{args.resilience_level}")
        print(f"ROUGH budget estimate: ~{est:.0f} s (~{est/60:.1f} min) of QPU "
              f"allowance.")
        print("Deeper circuits run longer per shot, so treat this as a lower "
              "bound. Actual usage is reported per job as it completes.")
        print("!" * 70)
        if input("Type 'yes' to continue: ").strip().lower() != "yes":
            print("Aborted. No QPU time used.")
            return

    record = run_sweep(args, use_hardware=args.submit)
    analyse(record)

    if record["total_quantum_seconds"]:
        print(f"\nTOTAL QPU TIME USED: {record['total_quantum_seconds']:.1f} s "
              f"({record['total_quantum_seconds']/60:.2f} min)")

    args.out.mkdir(parents=True, exist_ok=True)
    tag = "hardware" if record["hardware"] else "dryrun"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = args.out / f"{tag}_{stamp}.json"
    path.write_text(json.dumps(record, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()