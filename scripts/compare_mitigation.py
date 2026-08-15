"""
Compare a mitigated and an unmitigated hardware sweep: how much does error
mitigation actually buy, and is the F > 1 overshoot real?

Purpose
-------
Two open questions in this project are answered by the same pair of runs,
and neither needs new circuits -- only the same sweep repeated at a
different resilience_level:

1.  **How much does error mitigation close the gap?**
    scripts/delta_amplification_hardware.py's docstring explicitly defers
    this ("not explored here, since it needs a real backend run to answer
    honestly rather than assumed"). It is the single largest lever on
    every feasibility conclusion in this project: delta enters the outer-
    scheme amplification analysis directly, and the difference between
    delta = 0.08 and delta = 0.05 is the difference between FMG being
    marginal and being comfortable at a given N.

2.  **Is the F > 1 measurement an artifact or an error?**
    The first ibm_kingston sweep returned F = 1.0141 +/- 0.0038 at degree 0
    and a fitted intercept F_prep = 1.0325 -- both above the physical
    bound of 1. Readout error mitigation (TREX, resilience_level >= 1)
    produces an *unbiased* estimator of the noiseless expectation value,
    which is explicitly allowed to overshoot: it inverts a measured
    response matrix, and inversion of a noisy matrix does not respect the
    [-1, 1] range of the underlying observable. An unmitigated run
    (resilience_level = 0) has no such inversion step and should therefore
    return F <= 1 throughout. If it does, the overshoot is confirmed as a
    mitigation artifact affecting the intercept only. If it does not, the
    problem lies upstream -- in the Direct Fidelity Estimation sum or the
    target state -- and every fidelity number in this project needs
    revisiting.

Why the headline finding survives either way
-----------------------------------------------
The composition result is a statement about the SLOPE of ln F against
degree, and a mitigation bias that multiplies every fidelity by a common
factor c shifts ln F by the constant ln c -- changing the intercept and
leaving the slope untouched. Checked directly on the ibm_kingston data:
fitting as-measured, rescaling so F(0) = 1, and dropping degree 0 entirely
give F_UA = 0.9053, 0.9053 and 0.9020 respectively, a spread of 0.003. The
per-application fidelity is robust to this artifact; only F_prep is not,
and F_prep is not a quantity this project relies on.

Usage
-----
    # having run the sweep twice, at --resilience_level 1 and 0:
    python scripts/compare_mitigation.py \\
        --mitigated results/degree_composition/hardware_<A>.json \\
        --unmitigated results/degree_composition/hardware_<B>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def load(path: Path) -> dict:
    record = json.loads(Path(path).read_text())
    if not record.get("hardware"):
        print(f"NOTE: {path.name} is a dry run, not a hardware run. "
              f"Comparing simulator output tells you nothing about "
              f"mitigation, which has no effect in local testing mode.")
    return record


def fit(rows: list) -> dict:
    """Least-squares fit of ln F against degree. See the composition script."""
    pts = sorted(((r["degree"], r["fidelity"]) for r in rows
                  if r["fidelity"] > 1e-6), key=lambda p: p[0])
    d = np.array([p[0] for p in pts], float)
    F = np.array([p[1] for p in pts], float)
    if len(d) < 3:
        return {}
    slope, intercept = np.polyfit(d, np.log(F), 1)
    pred = intercept + slope * d
    ss_res = float(np.sum((np.log(F) - pred) ** 2))
    ss_tot = float(np.sum((np.log(F) - np.log(F).mean()) ** 2))
    return {
        "F_prep": float(np.exp(intercept)),
        "F_UA":   float(np.exp(slope)),
        "r2":     1.0 - ss_res / ss_tot if ss_tot > 1e-15 else float("nan"),
        "degrees": d, "fidelities": F,
        "max_fidelity": float(F.max()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mitigated", type=Path, required=True)
    p.add_argument("--unmitigated", type=Path, required=True)
    args = p.parse_args()

    on, off = load(args.mitigated), load(args.unmitigated)
    fit_on, fit_off = fit(on["rows"]), fit(off["rows"])
    if not fit_on or not fit_off:
        print("Need at least 3 sweep points in each run.")
        return

    print("=" * 72)
    print(f"Backend: {on['backend']}   Nx={on['Nx']}  "
          f"kappa(A_row)={on['kappa_row']:.4f}")
    print(f"Mitigated run   : resilience_level={on.get('resilience_level')}")
    print(f"Unmitigated run : resilience_level={off.get('resilience_level')}")
    print("=" * 72)

    on_by_d = {r["degree"]: r for r in on["rows"]}
    off_by_d = {r["degree"]: r for r in off["rows"]}
    shared = sorted(set(on_by_d) & set(off_by_d))

    print(f"\n{'degree':>7} {'F_unmitigated':>15} {'F_mitigated':>13} "
          f"{'gain':>9}")
    for d in shared:
        f_off, f_on = off_by_d[d]["fidelity"], on_by_d[d]["fidelity"]
        gain = (f_on - f_off)
        print(f"{d:7d} {f_off:15.4f} {f_on:13.4f} {gain:+9.4f}")

    print(f"\n{'':<22}{'unmitigated':>13} {'mitigated':>12}")
    print(f"{'per-application F_UA':<22}{fit_off['F_UA']:13.4f} "
          f"{fit_on['F_UA']:12.4f}")
    print(f"{'state-prep F_prep':<22}{fit_off['F_prep']:13.4f} "
          f"{fit_on['F_prep']:12.4f}")
    print(f"{'R^2 (linearity)':<22}{fit_off['r2']:13.4f} {fit_on['r2']:12.4f}")

    # ── Question 1: mitigation benefit, in the units this project uses ──
    delta_off = 1.0 - fit_off["F_UA"]
    delta_on = 1.0 - fit_on["F_UA"]
    print(f"\nPer-application delta (1 - F_UA): "
          f"{delta_off:.4f} unmitigated -> {delta_on:.4f} mitigated")
    if delta_off > 1e-9:
        factor = delta_off / delta_on if delta_on > 1e-9 else float("inf")
        print(f"Mitigation reduces per-application error by {factor:.2f}x.")
        print(f"Feed both into the outer-scheme analysis to see what this "
              f"buys in usable problem size:")
        print(f"  python scripts/delta_amplification_hardware.py --delta {delta_off:.4f}")
        print(f"  python scripts/delta_amplification_hardware.py --delta {delta_on:.4f}")

    # ── Question 2: is the overshoot a mitigation artifact? ──
    print("\n" + "-" * 72)
    print("F > 1 diagnosis")
    print("-" * 72)
    over_on = fit_on["max_fidelity"] > 1.0 or fit_on["F_prep"] > 1.0
    over_off = fit_off["max_fidelity"] > 1.0 or fit_off["F_prep"] > 1.0

    if over_on and not over_off:
        print("CONFIRMED as a mitigation artifact: the mitigated run exceeds "
              "the physical bound F <= 1, the unmitigated run does not.\n"
              "Readout mitigation inverts a measured response matrix, and "
              "that inversion does not respect the observable's [-1, 1] "
              "range, so an unbiased mitigated estimator is allowed to "
              "overshoot. It biases the fitted intercept F_prep, NOT the "
              "slope: report F_UA as the result and note the intercept is "
              "not physically meaningful under mitigation.")
    elif over_on and over_off:
        print("NOT a mitigation artifact: F > 1 appears in the UNMITIGATED "
              "run too. Something upstream is wrong -- most likely the "
              "Direct Fidelity Estimation coefficient sum or the target "
              "state. Every fidelity number in this project should be "
              "re-checked before use. Start by confirming that the Pauli "
              "coefficients of |target><target| sum correctly and that the "
              "target matches the transpiled circuit's own statevector.")
    elif not over_on and not over_off:
        print("No overshoot in either run; both respect F <= 1. Nothing to "
              "explain.")
    else:
        print("Unexpected: the UNMITIGATED run overshoots but the mitigated "
              "one does not. This is not the behaviour either mechanism "
              "predicts; treat both runs as suspect and investigate before "
              "quoting either.")

    # ── Slope robustness, the reason the headline survives ──
    print("\n" + "-" * 72)
    print("Slope robustness (the headline finding)")
    print("-" * 72)
    for label, f in (("unmitigated", fit_off), ("mitigated", fit_on)):
        d, F = f["degrees"], f["fidelities"]
        s_raw, _ = np.polyfit(d, np.log(F), 1)
        s_resc, _ = np.polyfit(d, np.log(F / F[0]), 1)
        mask = d > 0
        s_nod0 = (np.polyfit(d[mask], np.log(F[mask]), 1)[0]
                  if mask.sum() >= 2 else np.nan)
        vals = [np.exp(s) for s in (s_raw, s_resc, s_nod0) if np.isfinite(s)]
        print(f"  {label:<12} F_UA as-measured={np.exp(s_raw):.4f}  "
              f"rescaled={np.exp(s_resc):.4f}  "
              f"excl. d=0={np.exp(s_nod0):.4f}  "
              f"spread={max(vals)-min(vals):.4f}")
    print("\nA spread of a few times 1e-3 means the per-application fidelity "
          "is insensitive to how the intercept anomaly is treated, which is "
          "what makes it safe to quote as the result.")

    total = (on.get("total_quantum_seconds") or 0) + (off.get("total_quantum_seconds") or 0)
    if total:
        print(f"\nCombined QPU time across both runs: {total:.1f} s "
              f"({total/60:.2f} min)")


if __name__ == "__main__":
    main()