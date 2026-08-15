"""
2-D line QSVT degree sweep: quantifies the intersection of algorithmic accuracy and hardware fidelity for the core sub-problem.

Operator Properties
-------------------
The row operator A_row (from problems/poisson_line_2d.py) carries a -2/dy² diagonal shift from transverse coupling, pinning kappa(A_row) near 3 regardless of N (measured: kappa=2.36 at Nx=4). This well-conditioned operator requires only a low-degree QSVT polynomial. A degree of 5 yields machine-precision algorithmic accuracy (rel_error ~ 1e-15).

Evaluation Metrics
------------------
The script measures three independent quantities per degree:

1. **Algorithmic accuracy**: Real QSP angles (solvers.quantum.qsvt_1d.qsvt_solve_system) against a direct solve. Provides the noiseless upper bound.
2. **Circuit cost**: Post-transpilation two-qubit gate count (core.resources).
3. **Hardware fidelity**: Measured via core.hardware.hardware_fidelity_estimate against FakeTorino (default) or real hardware (--real).

Hardware Validation
-------------------
Hardware error composes multiplicatively. Validated on `ibm_kingston`, measured per-application fidelity is F_UA = 0.918 (R² = 0.9921). Modelled hardware error is thus 1 - F_UA^d.

The crossover degree identifies where hardware fidelity restricts accuracy more than polynomial degree improves it. Note that fidelity saturates at the depolarisation floor for d <~ 21. Crossovers computed beyond this depth represent extrapolations past measurable physical limits on this backend.

Usage
-----
    python scripts/qsvt_2d_line_degree_sweep.py
    python scripts/qsvt_2d_line_degree_sweep.py --Nx 8 --degrees 1 3 5 7 11 21
    python scripts/qsvt_2d_line_degree_sweep.py --real --backend ibm_kingston
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from problems.poisson_line_2d import PoissonLine2D


def build_row_problem(Nx: int):
    """
    A representative row/strip sub-problem from the unit-square benchmark:
    Nx interior nodes along the strip, Ny=Nx transverse strips (only the
    row operator and one strip's RHS are used here, so Ny's exact value
    does not matter beyond satisfying PoissonLine2D's 2-D input shape).
    """
    x = np.arange(1, Nx + 1) / (Nx + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    f = np.sin(np.pi * X) * np.sin(np.pi * Y)
    prob = PoissonLine2D(f)
    A_row = prob.row_matrix()
    b_row = prob.rhs()[:, 0]
    return prob, A_row, b_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Nx", type=int, default=4)
    parser.add_argument("--degrees", type=int, nargs="+",
                         default=[1, 3, 5, 7, 11, 21])
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--backend", type=str, default=None)
    args = parser.parse_args()

    prob, A_row, b_row = build_row_problem(args.Nx)
    kappa = prob.kappa_row()
    print(f"Row problem: Nx={args.Nx}, kappa(A_row)={kappa:.4f} "
          f"(project's line-decomposition claim: kappa -> 3 independent of N)")

    from core.resources import transpile_report
    from core.hardware import HardwareContext, hardware_fidelity_estimate
    from solvers.quantum.qsvt_1d import qsvt_solve_system, QSVTConfig1D
    from solvers.quantum.block_encoding import build_tst_block_encoding
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import Isometry
    from qiskit.quantum_info import Statevector

    n = int(np.log2(args.Nx))
    main_diag = float(A_row[0, 0])
    off_diag  = float(A_row[0, 1])
    exact_solution = np.linalg.solve(A_row, b_row)

    if args.real:
        print(f"\nMeasuring hardware fidelity on real hardware "
              f"({args.backend or 'least-busy'})...")
        context = HardwareContext.real(backend_name=args.backend)
    else:
        print("\nMeasuring hardware fidelity on FakeTorino (local testing)...")
        context = HardwareContext.local_testing()

    print(f"\n{'degree':>7} {'alg_rel_err':>12} {'2Q_gates':>9} "
          f"{'hw_fidelity':>12} {'hw_std_err':>10}")

    rows = []
    for degree in args.degrees:
        # 1. Algorithmic accuracy -- real QSP angles, real solve.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cfg = QSVTConfig1D(max_degree=degree)
            res = qsvt_solve_system(A_row, b_row, config=cfg)
        alg_err = float(np.linalg.norm(res.u - exact_solution)
                         / np.linalg.norm(exact_solution))
        actual_degree = res.polynomial_degree

        # 2. Circuit cost -- post-transpilation 2Q count (Phase 2).
        be_circuit, _alpha = build_tst_block_encoding(args.Nx, main_diag, off_diag)
        unit_report = transpile_report(be_circuit)
        two_q_estimate = actual_degree * unit_report.two_qubit_count

        # 3. Hardware fidelity -- state prep + one block-encoding application
        # (see block_encoding_fidelity.py for why this, not the full
        # degree-many-application circuit, is what's measured: a full
        # degree-21 circuit's DFE would need re-deriving the target through
        # the whole QSP sequence, which is a separate, larger undertaking).
        # This measures the *per-application* fidelity once per sweep point
        # only to confirm it is stable across the sweep, not because it
        # depends on degree.
        b_norm = b_row / np.linalg.norm(b_row)
        full_circuit = QuantumCircuit(n + 1)
        full_circuit.append(Isometry(b_norm, 0, 0), list(range(n)))
        full_circuit.compose(be_circuit, qubits=list(range(n + 1)), inplace=True)
        target_full = np.asarray(Statevector(full_circuit).data)

        fid_result, _n_terms = hardware_fidelity_estimate(
            full_circuit, target_full, context, shots=args.shots
        )

        rows.append(dict(
            degree=actual_degree, algorithmic_rel_error=alg_err,
            two_qubit_gates=two_q_estimate,
            hardware_fidelity=fid_result.value,
            hardware_std_error=fid_result.std_error,
        ))
        print(f"{actual_degree:7d} {alg_err:12.2e} {two_q_estimate:9d} "
              f"{fid_result.value:12.4f} {fid_result.std_error:10.4f}")

    # Per-application infidelity is degree-independent (same circuit
    # measured each time); the DEGREE-DEPENDENT total hardware error is
    # modelled as 1 - (fidelity)^degree, i.e. the block encoding applied
    # `degree` times independently -- the same first-order composition
    # used in block_encoding_fidelity.py.
    mean_fidelity = float(np.mean([r["hardware_fidelity"] for r in rows]))
    print(f"\nMean per-application hardware fidelity across sweep: "
          f"{mean_fidelity:.4f}")
    print(f"\n{'degree':>7} {'alg_rel_err':>12} {'modelled_hw_err':>16} "
          f"{'crossover?':>10}")
    for r in rows:
        modelled_hw_err = 1.0 - mean_fidelity ** r["degree"]
        crossover = "hw binds" if modelled_hw_err > r["algorithmic_rel_error"] else "alg binds"
        print(f"{r['degree']:7d} {r['algorithmic_rel_error']:12.2e} "
              f"{modelled_hw_err:16.4f} {crossover:>10}")

    print("\n'alg binds' means the algorithmic (polynomial truncation) error "
          "still exceeds the modelled hardware error at this degree -- going "
          "deeper would help. 'hw binds' means hardware noise already "
          "dominates -- going deeper would only add circuit cost for no "
          "accuracy gain, and error mitigation (Phase 5's resilience_level) "
          "or a shallower degree is the better lever.")


if __name__ == "__main__":
    main()