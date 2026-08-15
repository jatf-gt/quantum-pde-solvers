"""
Assesses whether hardware error composes as F^d across QSVT degree. Provides a real-device test of the extrapolation model underlying the project's feasibility analysis.

Rationale
---------
These models assume errors from repeated U_A applications accumulate independently. Previous validation relied on simulators, whose noise models assume independent per-gate channels by construction and cannot falsify the premise. This script provides the required real-device measurement.

Measures F_d directly at d = 1, 3, 5... for the 2-D line row operator, comparing against the F_1^d prediction. Potential outcomes:

    F_d ~= F_1^d   Extrapolations hold.
    F_d >  F_1^d   Coherent errors partially cancel. QSVT is more robust than predicted; feasibility limits remain conservative.
    F_d <  F_1^d   Errors compound super-linearly. Feasibility limits require revision.

Operator Selection
------------------
The transverse coupling in A_row provides a diagonal shift pinning kappa(A_row) near 3, independent of N. This yields the only QSVT circuit shallow enough for degree sweeps within a standard QPU budget, while remaining the core operator for the 2-D and 3-D architectures.

Execution Cost
--------------
QPU time is metered per execution second. Every mode reports usage from job.metrics(). For reference: 20 PUBs with 4096 shots on a 3-qubit circuit consume ~34 s on ibm_kingston. Resilience level 2 (ZNE) increases execution time roughly threefold.

Operation Modes
---------------
    --calibration   Dumps the backend calibration snapshot. Consumes zero QPU time. Required alongside hardware runs, as daily error rate drift renders fidelity figures uninterpretable without device state.
    (default)       Dry run against FakeTorino. No QPU time, no credentials.
    --submit        Executes on hardware. Prints a budget estimate and requires confirmation.
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
    Builds a QSVT circuit at the specified degree: state preparation followed by degree applications of the block encoding interleaved with single-qubit Rz rotations.

    Employs synthetic, non-degenerate phase angles instead of true QSP angles. Hardware error composition depends on the gate sequence, not the specific rotation angles. True QSP angles require pyqsp, adding dependency overhead without altering the underlying error physics (see core/resources.py::validate_composability).
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
    Records the backend's current calibration.

    Device error rates drift daily. Fidelity measurements require contemporaneous device state for valid interpretation and reproducibility. Execute alongside any hardware job.
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
    Tests whether hardware error composes multiplicatively across degree.

    Independent accumulation of errors yields F_d = F_prep * (F_UA)^d, or ln F_d = ln F_prep + d * ln F_UA. The testable content is the linearity of ln F in d. A least-squares fit leverages all sweep points; the resulting R² quantifies model validity.

    Previous versions exhibited two failure modes:
      1. Modelling F_d as F_1^d raised the one-off state-preparation error to the d-th power, producing a spurious "errors cancel" verdict.
      2. A two-point estimate (F_UA = F_1/F_0) proved unstable. At Nx=4, the d=1 circuit adds few gates to a state preparation that already dominates the error. F_1 and F_0 were nearly equal, producing an unphysical per-application fidelity above 1.

    The present fit avoids both modes. Degree 0 (state preparation alone) is included to measure, rather than assume, the intercept.
    """
    rows = sorted(record["rows"], key=lambda r: r["degree"])

    # ── Depolarisation floor ─────────────────────────────────────────────────
    # A fully depolarised state rho = I/2^n has DFE fidelity <t|I/2^n|t> =
    # 1/2^n against ANY target. Once a circuit is deep enough to decohere
    # completely, the measured "fidelity" stops decaying and sits at that
    # floor -- it is no longer measuring anything about the circuit.
    #
    # An earlier version of this function fitted every point regardless, and
    # on a real ibm_kingston sweep to d=63 that produced a confident and
    # FALSE verdict ("errors do not accumulate independently, the F^d
    # extrapolation does not hold"). The tail it fitted was 0.1118, 0.1250,
    # 0.1222 for a 3-qubit circuit -- all within one error bar of
    # 1/2^3 = 0.125. Nothing had broken; the measurement had saturated.
    # Points at or near the floor are therefore excluded from the fit and
    # reported separately.
    n_qubits = int(np.log2(max(2, record.get("Nx", 4)))) + 1
    floor = 1.0 / (2 ** n_qubits)
    typical_err = float(np.median([r.get("fidelity_std_err", 0.007) or 0.007
                                   for r in rows]))
    live = [r for r in rows if r["fidelity"] - floor > 3 * typical_err]
    saturated = [r for r in rows if r not in live]

    print(f"\nDepolarisation floor for {n_qubits} qubits: 1/2^{n_qubits} = "
          f"{floor:.4f}")
    if saturated:
        print(f"  SATURATED (excluded from fit): "
              f"degrees {[r['degree'] for r in saturated]} with fidelities "
              f"{[round(r['fidelity'], 4) for r in saturated]}")
        print(f"  These circuits are indistinguishable from the maximally "
              f"mixed state. This is an instrumental limit, not a failure "
              f"of the composition model.")
        deepest_live = max((r["degree"] for r in live), default=None)
        if deepest_live is not None:
            print(f"  Usable QSVT depth on this backend for this circuit "
                  f"family: d <~ {deepest_live}")

    # Fit the floor-subtracted signal over the live points only.
    pts = [(r["degree"], r["fidelity"] - floor) for r in live
           if r["fidelity"] - floor > 1e-6]
    if len(pts) < 3:
        print("\nFewer than 3 unsaturated points; cannot fit a composition "
              "law. Re-run with shallower degrees -- the deep points carry "
              "no information.")
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
    print(f"  intercept -> (F_prep - floor)            = {F_prep:.4f}")
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
    p.add_argument("--reanalyse", type=Path, default=None,
                   help="Re-run the analysis on a previously saved results "
                        "JSON. No QPU time, no credentials -- the fidelities "
                        "are already measured and are not re-acquired. Use "
                        "this after any change to analyse(), rather than "
                        "repeating the sweep on hardware.")
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

    if args.reanalyse:
        record = json.loads(args.reanalyse.read_text())
        print(f"Re-analysing {args.reanalyse.name}  "
              f"(backend={record.get('backend')}, "
              f"resilience_level={record.get('resilience_level')})")
        print("No QPU time is used: these fidelities were already measured.")
        for r in sorted(record["rows"], key=lambda x: x["degree"]):
            print(f"  d={r['degree']:3d}  2Q={r.get('two_qubit_gates', 0):5d}  "
                  f"F={r['fidelity']:.4f} +/- {r.get('fidelity_std_err', float('nan')):.4f}")
        analyse(record)
        return

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