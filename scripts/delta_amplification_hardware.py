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

A second, more serious issue surfaced the same way: at delta large enough
(confirmed directly at delta ~ 0.165, this project's own unmitigated
hardware measurement), the outer iteration does not converge to a
*stable, merely large* fixed point -- it diverges outright. Confirmed by
inspecting residual_history directly: at N=32, delta=0.165, FMG's residual
grew monotonically and explosively (38 -> 65 -> 93 -> ... -> 631 over 10
iterations) while StagnationMonitor still classified the run as
"stagnated", since its median-window test is built to detect a lack of
improvement, not active blow-up. Reporting err/delta from a diverged run
as "amplification" would present a snapshot of an undefined process as if
it were a physical quantity. measure_amplification therefore checks for
this directly (residual growing over the run's second half) and reports
diverged runs as DIVERGED, with amplification omitted (NaN) rather than a
large, meaningless number.

At small delta (0.005, well inside both schemes' stability region),
directly measured on the unit square, N in {8,16,32}:

    N=  8: FMG amp= 1.75   SOR amp=  5.22
    N= 16: FMG amp= 2.08   SOR amp= 11.04
    N= 32: FMG amp= 2.02   SOR amp= 21.29

FMG's amplification is flat; SOR's roughly doubles when N doubles -- the
qualitative claim holds cleanly here. It does NOT hold, and should not be
expected to, once delta pushes the iteration past its stability threshold;
that is a different, more severe regime with its own (DIVERGED) label, not
a continuation of the same amplification trend to larger numbers.

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
Even within the stable (non-diverged) regime, the FMG_amp/SOR_amp columns
divide the error against the analytic reference by delta -- which mixes in
discretisation error, not just the amplified quantum error. This is
negligible once delta is comfortably larger than discretisation error, but
was confirmed directly to matter at N=16, delta=0.001, where
discretisation_error/delta = 2.85 was comparable to the true amplification
(~2.1x, confirmed by comparing against a numerical delta=0 reference
instead) -- the naive vs-analytic ratio read 3.99x, a real but misleading
contamination. If a future, well-mitigated run reports a small delta,
treat the smallest-N rows of the amp columns with this in mind.

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


# -- Manufactured (exact) reference problem -------------------------------------

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


# -- Amplification measurement -------------------------------------------------

def measure_amplification(N: int, delta: float, scheme: str, max_iter: int = 800):
    """
    Solution error under a real delta-sized perturbation, compared against
    the analytic reference, WITH explicit divergence detection.

    Why this check exists: an earlier version of this function reported the
    error at whatever point the outer iteration stopped, unconditionally,
    labelling err/delta as "amplification" regardless of what the iteration
    was actually doing. At delta ~ 0.16 (this project's own unmitigated
    hardware measurement), that number is not a stable amplified fixed
    point -- it is a snapshot of active divergence. Confirmed directly at
    N=32: residual_history grew monotonically and explosively (38 -> 65 ->
    93 -> ... -> 631 over 10 iterations), while StagnationMonitor's
    median-window test still classified the run as "stagnated", because a
    smooth exponential-looking growth curve does not trip a test built to
    detect a *lack* of improvement, not active blow-up. Reporting
    "amplification = 14328x" from a diverged run is not a finding about
    amplification; it is a snapshot of an undefined quantity mislabelled as
    one. solvers/outer/multigrid.py's own docstring already flags that both
    schemes have a finite stability threshold ("SOR diverges at 1% strip
    error... multigrid still converges to ~5%"); this check is what
    respects that threshold instead of silently reporting through it.

    Divergence test: compare the residual at the end of the run against the
    residual at the midpoint. A ratio > 1.5 means the residual grew rather
    than plateaued -- confirmed empirically to cleanly separate the stable
    cases (ratio ~1.00 at delta up to 0.10 in testing) from the diverging
    one (ratio 2.70 at delta=0.165) for this problem.

    Returns
    -------
    (error, OuterResult, diverged: bool)
    """
    prob, u_exact = analytic_problem(N)
    kwargs = {"tol": 1e-10}
    if scheme in ("sor", "gauss-seidel"):
        kwargs["max_iter"] = max_iter
    else:
        kwargs["max_cycles"] = 50
    res = solve(prob, inner="perturbed", scheme=scheme,
                inner_options={"delta": delta}, **kwargs)

    h = res.residual_history
    diverged = False
    if len(h) > 2:
        mid = h[len(h) // 2]
        if mid > 0 and h[-1] / mid > 1.5:
            diverged = True

    err = float(np.linalg.norm(res.u - u_exact) / np.linalg.norm(u_exact))
    return err, res, diverged


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

    print(f"{'N':>4} {'disc_err':>10} {'FMG_err':>10} {'FMG_amp':>10} "
          f"{'SOR_err':>10} {'SOR_amp':>10} {'binding_constraint':>20}")

    rows = []
    for N in args.N_values:
        d_err = discretization_error(N)
        fmg_err, fmg_res, fmg_diverged = measure_amplification(N, delta, "fmg")
        sor_err, sor_res, sor_diverged = measure_amplification(N, delta, "sor")

        # Amplification is only a meaningful quantity for a run that
        # actually stabilised. A diverged run's "error" is a snapshot of
        # blow-up, not a fixed-point offset -- reporting err/delta for it
        # would present an artifact of *when the loop stopped* as if it
        # were a physical amplification factor. See measure_amplification's
        # docstring for the divergence this was built to catch.
        fmg_amp = (fmg_err / delta) if (delta > 0 and not fmg_diverged) else float("nan")
        sor_amp = (sor_err / delta) if (delta > 0 and not sor_diverged) else float("nan")

        # The binding constraint: does the amplified quantum error still
        # beat the discretisation error the classical scheme would achieve
        # anyway? Divergence is reported as its own, more severe category --
        # it means no usable solution is obtained at all, not merely that
        # quantum error exceeds discretisation error.
        def _binding(err, d_err, diverged):
            if diverged:
                return "DIVERGED"
            return "quantum error" if err > d_err else "discretisation"

        fmg_binds = _binding(fmg_err, d_err, fmg_diverged)
        sor_binds = _binding(sor_err, d_err, sor_diverged)

        rows.append(dict(N=N, discretization_error=d_err,
                          fmg_error=fmg_err, fmg_amplification=fmg_amp,
                          fmg_diverged=fmg_diverged, fmg_binding=fmg_binds,
                          sor_error=sor_err, sor_amplification=sor_amp,
                          sor_diverged=sor_diverged, sor_binding=sor_binds))

        fmg_amp_str = "DIVERGED" if fmg_diverged else f"{fmg_amp:.2f}"
        sor_amp_str = "DIVERGED" if sor_diverged else f"{sor_amp:.2f}"
        print(f"{N:4d} {d_err:10.2e} {fmg_err:10.2e} {fmg_amp_str:>10} "
              f"{sor_err:10.2e} {sor_amp_str:>10} "
              f"fmg:{fmg_binds[:8]}/sor:{sor_binds[:8]}")

    n_diverged = sum(r["fmg_diverged"] or r["sor_diverged"] for r in rows)
    if n_diverged > 0:
        print(f"\n*** {n_diverged}/{len(rows)} N-values show at least one "
              f"scheme DIVERGING (residual growing, not merely large) at "
              f"delta={delta:.4f}. This is a stronger finding than 'quantum "
              f"error exceeds discretisation error': it means no usable "
              f"solution is obtained via this outer scheme at this delta, "
              f"for these N. Amplification figures are omitted (NaN) for "
              f"diverged runs -- they are not meaningful quantities to "
              f"report, see measure_amplification's docstring. ***")

    print("\n'binding_constraint' shows which error dominates: if the "
          "amplified quantum error exceeds discretisation error, the outer "
          "scheme's amplification -- not qubit count -- is what limits "
          "the usable problem size at this delta. DIVERGED is a distinct, "
          "more severe outcome (see above).")

    fmg_amps = [r["fmg_amplification"] for r in rows if not r["fmg_diverged"]]
    sor_amps = [r["sor_amplification"] for r in rows if not r["sor_diverged"]]

    if fmg_amps:
        print(f"\nFMG amplification range across N (non-diverged runs only): "
              f"{min(fmg_amps):.2f}x - {max(fmg_amps):.2f}x "
              f"({'roughly constant' if max(fmg_amps)/min(fmg_amps) < 2 else 'growing'})")
    else:
        print("\nFMG diverged at every N tested -- no amplification range to report.")

    if sor_amps:
        print(f"SOR amplification range across N (non-diverged runs only): "
              f"{min(sor_amps):.2f}x - {max(sor_amps):.2f}x "
              f"({'roughly constant' if max(sor_amps)/min(sor_amps) < 2 else 'growing'})")
    else:
        print("SOR diverged at every N tested -- no amplification range to report.")

    # Maximum N where the FMG-amplified quantum error still beats
    # discretisation error -- the headline number. A diverged run never
    # counts as feasible, regardless of where its (meaningless) error
    # snapshot happens to sit relative to discretisation error.
    feasible = [r["N"] for r in rows
                if not r["fmg_diverged"] and r["fmg_error"] <= r["discretization_error"]]
    if feasible:
        print(f"\nMax N where FMG-amplified quantum error <= discretisation "
              f"error (delta={delta:.4f}): N={max(feasible)}")
    elif any(r["fmg_diverged"] for r in rows):
        print(f"\nAt delta={delta:.4f}, FMG diverges at every N tested -- "
              f"delta is past FMG's stability threshold here, not merely "
              f"large enough to dominate discretisation error. Error "
              f"mitigation to reduce delta is necessary before any N is "
              f"usable, not just before N can be made larger.")
    else:
        print(f"\nAt delta={delta:.4f}, quantum error exceeds discretisation "
              f"error at every N tested -- delta itself, not N, is the "
              f"binding constraint here.")


if __name__ == "__main__":
    main()