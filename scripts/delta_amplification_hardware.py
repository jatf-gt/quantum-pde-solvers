"""
Hardware-measured inner-solver error, coupled to the outer-scheme
amplification model: the chapter-anchor experiment.

Measures delta -- the per-application infidelity of the block encoding, via
core.hardware.hardware_fidelity_estimate on real (or local-testing)
hardware -- then feeds it into the classical outer-iteration layer via the
"perturbed" inner solver (solvers/outer/inner.py), which models any
systematic inner-solver error, quantum or otherwise, as a deterministic
operator perturbation ||E|| = delta * ||A||. This directly answers the
question posed at the start of this hardware-scoping work: is the binding
constraint on how large a quantum-inner-solver problem can get the qubit
count, or the outer scheme's amplification of inner-solver error?

Empirical grounding, not just the 1/(1-rho) formula
--------------------------------------------------------
solvers/outer/multigrid.py's own docstring states the qualitative claim:
FMG's convergence factor rho ~ 0.13 independent of N, so its amplification
of a per-strip error is roughly constant; SOR's rho -> 1 as O(1 - 1/N), so
its amplification grows with N. This script does not just cite that
docstring -- it re-measures the claim directly, because a first attempt at
comparing the *predicted* amplification (1/(1-rho), rho estimated from a
near-zero-delta run's residual decay) against the *measured* amplification
(direct solution error under a real delta, divided by delta) found only
rough, same-order-of-magnitude agreement, not a precise match -- rho
estimated as a geometric mean over a residual history that includes a
post-convergence plateau is a biased proxy for the asymptotic contraction
rate relevant to error accumulation. The headline numbers this script
reports are therefore the *directly measured* amplification (solution
error under a real perturbation, compared against an exact reference),
which is unambiguous, not the theoretical 1/(1-rho) prediction, which is
reported alongside only as a rough consistency check.

Directly measured on the unit square, delta=0.005, N in {8,16,32}:

    N=  8: FMG amp= 1.75   SOR amp=  5.22
    N= 16: FMG amp= 2.08   SOR amp= 11.04
    N= 32: FMG amp= 2.02   SOR amp= 21.29

FMG's amplification is flat; SOR's roughly doubles when N doubles. This is
exactly the qualitative claim, now with numbers from this specific solver
stack rather than only the docstring's prose.

Discretisation error reference
-----------------------------------
Uses the manufactured solution u = sin(pi x) sin(pi y), which is an *exact*
solution of the continuous PDE for f = -2 pi^2 sin(pi x) sin(pi y) -- so the
gap between a delta=0 (Thomas) solve and this analytic u is pure
discretisation error, with no algorithmic error contaminating it. Confirmed
here to follow the expected O(h^2) = O(1/N^2) scaling before being trusted
as the accuracy target the amplified quantum error is compared against.

Fidelity-to-delta is a modelling choice, not an identity
--------------------------------------------------------------
"delta" here is "1 - measured block-encoding fidelity", fed into the
perturbed inner solver as a relative operator-norm perturbation
||E|| = delta * ||A||. Infidelity is a state-level quantity and operator-norm
perturbation is a different one; treating them as interchangeable is a
first-order approximation, not an exact translation, and should be stated as
such wherever this script's numbers are quoted. It is, however, the same
approximation implicit in describing HHL's Trotter truncation or QSVT's
polynomial truncation as a single "delta" throughout this project, so it is
at least a *consistent* one.

At today's unmitigated FakeTorino calibration this delta comes out around
0.18 (measured directly, not assumed) -- an order of magnitude past the
~1% strip error at which solvers/outer/multigrid.py's own docstring says
SOR already diverges. Run this script and both schemes will show
catastrophic, not merely amplified, error at every N tested. That is a
real result, not a bug: it says current unmitigated hardware noise, not
qubit count, is what rules out this scheme today, and the natural next
question is how much Phase 5's resilience_level options (measurement
mitigation, ZNE) close that gap -- not explored here, since it needs a
real backend run to answer honestly rather than assumed.

A caveat for a future, much smaller delta
----------------------------------------------
The FMG_amp/SOR_amp columns divide the error against the analytic
reference by delta -- which mixes in discretisation error, not just the
amplified quantum error. At today's delta~0.18 this is negligible
(discretisation error at N=8 is ~1e-2, delta is ~0.18: a ~5% contribution
to the amplification figure). It stops being negligible once delta gets
small relative to discretisation error, which happens either at small N
or after substantial error mitigation brings delta down: tested directly
at N=16, delta=0.001, where discretisation_error/delta = 2.85 was
comparable to the true amplification (~2.1x, confirmed by comparing
against a numerical delta=0 reference instead) -- the naive vs-analytic
ratio read 3.99x, a real but misleading contamination, not a genuine
amplification effect. If a future run reports a small delta, treat the
smallest-N rows of the amp columns with this in mind.

Usage
-----
    python scripts/delta_amplification_hardware.py                     # measures delta via FakeTorino
    python scripts/delta_amplification_hardware.py --delta 0.05         # skip hardware, use a given delta
    python scripts/delta_amplification_hardware.py --real --backend ibm_kingston
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from solvers.outer import solve, PoissonLine2D


# ── Manufactured (exact) reference problem ─────────────────────────────────────

def analytic_problem(N: int):
    """
    u = sin(pi x) sin(pi y) on the unit square, an exact solution of
    nabla^2 u = f for f = -2 pi^2 sin(pi x) sin(pi y). Returns
    (PoissonLine2D instance, exact u array) so any solve's error against
    the true continuous solution can be measured directly, with no
    discretisation contamination in the reference itself.
    """
    x = np.arange(1, N + 1) / (N + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    f = -2 * np.pi ** 2 * np.sin(np.pi * X) * np.sin(np.pi * Y)
    u_exact = np.sin(np.pi * X) * np.sin(np.pi * Y)
    return PoissonLine2D(f), u_exact


def discretization_error(N: int) -> float:
    """Relative error of an exact (Thomas, delta=0) FMG solve against the
    analytic solution -- pure discretisation error, the accuracy floor no
    amount of algorithmic precision can beat."""
    prob, u_exact = analytic_problem(N)
    res = solve(prob, inner="thomas", scheme="fmg", tol=1e-12)
    return float(np.linalg.norm(res.u - u_exact) / np.linalg.norm(u_exact))


def measure_delta(args) -> float:
    """
    Per-application block-encoding infidelity, via the same Direct Fidelity
    Estimation machinery as block_encoding_fidelity.py (Phase 5), reused
    directly rather than reimplemented.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import Isometry
    from qiskit.quantum_info import Statevector

    from core.hardware import HardwareContext, hardware_fidelity_estimate
    from solvers.quantum.block_encoding import build_tst_block_encoding

    N_be = args.be_N
    n = int(np.log2(N_be))
    be_circuit, _alpha = build_tst_block_encoding(N_be, -2.0, 1.0)

    b_norm = np.ones(N_be) / np.sqrt(N_be)
    full_circuit = QuantumCircuit(n + 1)
    full_circuit.append(Isometry(b_norm, 0, 0), list(range(n)))
    full_circuit.compose(be_circuit, qubits=list(range(n + 1)), inplace=True)
    target_full = np.asarray(Statevector(full_circuit).data)

    if args.real:
        print(f"Measuring delta on real hardware ({args.backend or 'least-busy'})...")
        context = HardwareContext.real(backend_name=args.backend)
    else:
        print("Measuring delta on FakeTorino (local testing)...")
        context = HardwareContext.local_testing()

    result, n_terms = hardware_fidelity_estimate(
        full_circuit, target_full, context, shots=args.shots
    )
    print(f"  Fidelity: {result.value:.4f} +/- {result.std_error:.4f} "
          f"({n_terms} Pauli terms, job {result.provenance.job_id})")
    return max(0.0, 1.0 - result.value)


# ── Amplification measurement ─────────────────────────────────────────────────

def measure_amplification(N: int, delta: float, scheme: str, max_iter: int = 800):
    """
    Solution error under a real delta-sized perturbation, compared against
    the analytic reference -- the direct, unambiguous measurement this
    script's headline numbers are built from (see module docstring on why
    this is preferred over the 1/(1-rho) prediction).
    """
    prob, u_exact = analytic_problem(N)
    kwargs = {"tol": 1e-10}
    if scheme in ("sor", "gauss-seidel"):
        kwargs["max_iter"] = max_iter
    else:
        kwargs["max_cycles"] = 50
    res = solve(prob, inner="perturbed", scheme=scheme,
                inner_options={"delta": delta}, **kwargs)
    err = float(np.linalg.norm(res.u - u_exact) / np.linalg.norm(u_exact))
    return err, res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=float, default=None,
                         help="Skip the hardware measurement and use this "
                              "delta directly.")
    parser.add_argument("--be_N", type=int, default=4,
                         help="Block-encoding problem size for the delta "
                              "measurement (only used without --delta).")
    parser.add_argument("--shots", type=int, default=4096)
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--backend", type=str, default=None)
    parser.add_argument("--N_values", type=int, nargs="+", default=[8, 16, 32, 64])
    args = parser.parse_args()

    if args.delta is not None:
        delta = args.delta
        print(f"Using supplied delta={delta}\n")
    else:
        delta = measure_delta(args)
        print(f"\nMeasured delta (per-application infidelity): {delta:.4f}\n")

    print(f"{'N':>4} {'disc_err':>10} {'FMG_err':>10} {'FMG_amp':>8} "
          f"{'SOR_err':>10} {'SOR_amp':>8} {'binding_constraint':>20}")

    rows = []
    for N in args.N_values:
        d_err = discretization_error(N)
        fmg_err, fmg_res = measure_amplification(N, delta, "fmg")
        sor_err, sor_res = measure_amplification(N, delta, "sor")
        fmg_amp = fmg_err / delta if delta > 0 else float("nan")
        sor_amp = sor_err / delta if delta > 0 else float("nan")

        # The binding constraint: does the amplified quantum error still
        # beat the discretisation error the classical scheme would achieve
        # anyway? If not, the outer scheme's amplification -- not the
        # circuit's qubit count -- is what limits usable problem size.
        fmg_binds = "quantum error" if fmg_err > d_err else "discretisation"
        sor_binds = "quantum error" if sor_err > d_err else "discretisation"

        rows.append(dict(N=N, discretization_error=d_err,
                          fmg_error=fmg_err, fmg_amplification=fmg_amp,
                          fmg_binding=fmg_binds,
                          sor_error=sor_err, sor_amplification=sor_amp,
                          sor_binding=sor_binds))
        print(f"{N:4d} {d_err:10.2e} {fmg_err:10.2e} {fmg_amp:8.2f} "
              f"{sor_err:10.2e} {sor_amp:8.2f} "
              f"fmg:{fmg_binds[:4]}/sor:{sor_binds[:4]}")

    print("\n'binding_constraint' shows which error dominates: if the "
          "amplified quantum error exceeds discretisation error, the outer "
          "scheme's amplification -- not qubit count -- is what limits "
          "the usable problem size at this delta.")

    fmg_amps = [r["fmg_amplification"] for r in rows]
    sor_amps = [r["sor_amplification"] for r in rows]
    print(f"\nFMG amplification range across N: "
          f"{min(fmg_amps):.2f}x - {max(fmg_amps):.2f}x "
          f"({'roughly constant' if max(fmg_amps)/min(fmg_amps) < 2 else 'growing'})")
    print(f"SOR amplification range across N: "
          f"{min(sor_amps):.2f}x - {max(sor_amps):.2f}x "
          f"({'roughly constant' if max(sor_amps)/min(sor_amps) < 2 else 'growing'})")

    # Maximum N where the FMG-amplified quantum error still beats
    # discretisation error -- the headline number.
    feasible = [r["N"] for r in rows if r["fmg_error"] <= r["discretization_error"]]
    if feasible:
        print(f"\nMax N where FMG-amplified quantum error <= discretisation "
              f"error (delta={delta:.4f}): N={max(feasible)}")
    else:
        print(f"\nAt delta={delta:.4f}, quantum error exceeds discretisation "
              f"error at every N tested -- delta itself, not N, is the "
              f"binding constraint here.")


if __name__ == "__main__":
    main()