#!/usr/bin/env python3
"""
Post-processing plots for a run_hpc_2Dfull.py sweep.

Reads results/2Dhpc_run/ (results_full.json plus the archived
solutions_{case}_{solver}_N{N}.npz files) and produces PNGs into
results/2Dhpc_run/plots/.  Nothing here re-runs a solve; it only reads what
the runner already wrote, so it is cheap and safe to re-run at any time.

These are diagnostic plots, not thesis figures - the priority is catching a
wrong field or a suspicious trend quickly, not polish.

Plots produced
--------------
1. Solution fields  (the most important one - "does the field look right")
   For every (case, N): exact / Thomas / each quantum solver as pcolormesh,
   with a signed error map underneath each.  This is the plot that catches a
   sign error, a misplaced boundary condition, or a solver that converged to
   the wrong fixed point - things a table of norms can hide.

2. Convergence history - residual vs outer iteration/cycle, all solvers and
   schemes overlaid, log scale.  Shows stagnation directly.

3. Accuracy vs N - log-log, one line per solver, with an O(h^2) reference
   slope.  A solver that doesn't track the reference slope is either poorly
   resolved or has a bug independent of the outer scheme.

4. Cost vs N - weighted strip-solve cost and wall time, log-log.  This is
   the plot that answers "does the quantum solver's cost scale acceptably".

5. Quantum overhead vs Thomas - wall-clock ratio, log scale, per case.

6. Error decomposition - err_vs_thomas (algorithmic/quantum error) against
   err_thomas_vs_exact (discretisation error), so it is visible which one
   dominates at each N.

Usage
-----
    python scripts/plot_hpc_2Dfull_results.py                  # everything found
    python scripts/plot_hpc_2Dfull_results.py --case 2D_HET_MMS_SPT100
    python scripts/plot_hpc_2Dfull_results.py --list            # show what's available

Author : Juan Antonio Trobajo Flecha
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results" / "2Dhpc_run"
PLOTS_DIR = RESULTS_DIR / "plots"

SOLVER_COLOUR = {"Thomas": "#444444", "HHL": "#d62728",
                 "VQLS": "#2ca02c", "QSVT": "#1f77b4"}
SOLVER_ORDER = ["Thomas", "HHL", "VQLS", "QSVT"]


# ============================================================================
#  Loading
# ============================================================================

def load_results() -> list[dict]:
    path = RESULTS_DIR / "results_full.json"
    if not path.exists():
        raise SystemExit(f"No results found at {path}. Run run_hpc_2Dfull.py first.")
    with open(path) as fh:
        return json.load(fh)


def load_solution(case: str, solver: str, N: int) -> dict | None:
    path = RESULTS_DIR / f"solutions_{case}_{solver}_N{N}.npz"
    if not path.exists():
        return None
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def _sort_key(s: str) -> tuple:
    return (SOLVER_ORDER.index(s) if s in SOLVER_ORDER else 99, s)


def group_by_case_N(rows: list[dict]) -> dict[tuple, list[dict]]:
    """{(case, N): [row, row, ...]}, solvers ordered Thomas/HHL/VQLS/QSVT."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("notes", "").startswith("scheme_comparison"):
            continue           # these belong to the --compare-schemes study
        groups.setdefault((r["case"], r["N"]), []).append(r)
    for key in groups:
        groups[key].sort(key=lambda r: _sort_key(r["solver"]))
    return groups


def group_by_case_solver(rows: list[dict]) -> dict[tuple, list[dict]]:
    """{(case, solver): [row, ...]} sorted by N, for the vs-N plots."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r.get("notes", "").startswith("scheme_comparison"):
            continue
        groups.setdefault((r["case"], r["solver"]), []).append(r)
    for key in groups:
        groups[key].sort(key=lambda r: r["N"])
    return groups


# ============================================================================
#  Plot 1 - solution fields
# ============================================================================

def plot_fields(case: str, N: int, rows: list[dict], plt, TwoSlopeNorm) -> Path | None:
    """
    Exact (if available) plus one column per solver: field on top, signed
    error underneath.  The single most important plot in this script.
    """
    sols = {}
    exact = None
    for r in rows:
        sol = load_solution(case, r["solver"], N)
        if sol is None:
            continue
        sols[r["solver"]] = sol
        if exact is None and "phi_exact" in sol:
            exact = sol["phi_exact"]
    if not sols:
        return None

    any_sol = next(iter(sols.values()))
    x, y = any_sol["x"], any_sol["y"]
    ref = exact if exact is not None else sols.get("Thomas", any_sol)["phi_solver"]
    ref_is_exact = exact is not None

    names = list(sols)
    n_cols = 1 + len(names)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7), squeeze=False)

    vmin, vmax = float(ref.min()), float(ref.max())
    im = axes[0, 0].pcolormesh(x, y, ref, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                               shading="auto")
    axes[0, 0].set_title("Exact" if ref_is_exact else "Thomas (reference)",
                         fontweight="bold")
    axes[0, 0].set_aspect("equal")
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
    axes[1, 0].axis("off")

    for ci, name in enumerate(names, start=1):
        phi = sols[name]["phi_solver"]
        im = axes[0, ci].pcolormesh(x, y, phi, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                                    shading="auto")
        axes[0, ci].set_title(name, fontweight="bold",
                              color=SOLVER_COLOUR.get(name, "black"))
        axes[0, ci].set_aspect("equal")
        plt.colorbar(im, ax=axes[0, ci], shrink=0.8)

        err = phi - ref
        abs_max = max(float(np.abs(err).max()), 1e-12)
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
        im2 = axes[1, ci].pcolormesh(x, y, err, cmap="seismic", norm=norm,
                                     shading="auto")
        pct = (np.max(np.abs(err)) / (np.max(np.abs(ref)) + 1e-300)) * 100.0
        label = "vs exact" if ref_is_exact else "vs Thomas"
        axes[1, ci].set_title(f"Error {label} ({pct:.3f}%)")
        axes[1, ci].set_aspect("equal")
        plt.colorbar(im2, ax=axes[1, ci], shrink=0.8)

    fig.suptitle(f"{case}  N={N}", fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"fields_{case}_N{N}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
#  Plot 2 - convergence history
# ============================================================================

def plot_convergence(case: str, N: int, rows: list[dict], plt) -> Path | None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    any_curve = False
    for r in rows:
        sol = load_solution(case, r["solver"], N)
        if sol is None or "residual_history" not in sol:
            continue
        h = sol["residual_history"]
        if len(h) == 0:
            continue
        ax.semilogy(range(1, len(h) + 1), h, lw=1.8,
                    color=SOLVER_COLOUR.get(r["solver"]),
                    label=f"{r['solver']} ({r['scheme']})")
        any_curve = True
    if not any_curve:
        plt.close(fig)
        return None
    ax.set_xlabel("outer iteration (sweep or cycle)")
    ax.set_ylabel(r"$\|b - Au\|_2 / \|b\|_2$")
    ax.set_title(f"Convergence - {case}, N={N}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = PLOTS_DIR / f"convergence_{case}_N{N}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
#  Plot 3 - accuracy vs N
# ============================================================================

def plot_accuracy_vs_n(case: str, by_solver: dict, plt) -> Path | None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    any_line = False
    N_all = []
    for (c, solver), rs in sorted(by_solver.items()):
        if c != case:
            continue
        Ns = [r["N"] for r in rs if r.get("linf_err") is not None]
        errs = [r["linf_err"] for r in rs if r.get("linf_err") is not None]
        if not Ns:
            continue
        ax.loglog(Ns, errs, "o-", lw=1.8, color=SOLVER_COLOUR.get(solver),
                  label=solver)
        N_all.extend(Ns)
        any_line = True
    if not any_line:
        plt.close(fig)
        return None
    N_all = sorted(set(N_all))
    if len(N_all) >= 2:
        ref = [(N_all[0] / n) ** 2 for n in N_all]
        ref_all = [r * (max(ax.get_ylim()) * 0.5 / max(ref)) for r in ref]
        ax.loglog(N_all, ref_all, "k--", lw=1, alpha=0.5, label=r"$O(h^2)$ ref.")
    ax.set_xlabel("N")
    ax.set_ylabel(r"$L_\infty$ error vs. reference (%)")
    ax.set_title(f"Accuracy vs N - {case}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = PLOTS_DIR / f"accuracy_vs_N_{case}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
#  Plot 4 - cost vs N
# ============================================================================

def plot_cost_vs_n(case: str, by_solver: dict, plt) -> Path | None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    any_line = False
    for (c, solver), rs in sorted(by_solver.items()):
        if c != case:
            continue
        Ns = [r["N"] for r in rs]
        wc = [r["weighted_cost"] for r in rs if r.get("weighted_cost") is not None]
        wt = [r["wall_time_s"] for r in rs]
        if wc and len(wc) == len(Ns):
            axes[0].loglog(Ns, wc, "o-", color=SOLVER_COLOUR.get(solver), label=solver)
        axes[1].loglog(Ns, wt, "o-", color=SOLVER_COLOUR.get(solver), label=solver)
        any_line = True
    if not any_line:
        plt.close(fig)
        return None
    axes[0].set_xlabel("N"); axes[0].set_ylabel("weighted strip-solve cost")
    axes[0].set_title("Cost (finest-solve units)")
    axes[1].set_xlabel("N"); axes[1].set_ylabel("wall time (s)")
    axes[1].set_title("Wall time")
    for a in axes:
        a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
    fig.suptitle(f"Cost vs N - {case}", fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"cost_vs_N_{case}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
#  Plot 5 - quantum overhead vs Thomas
# ============================================================================

def plot_overhead(case: str, by_N: dict, plt) -> Path | None:
    xs, ys, cs, labels = [], [], [], []
    for (c, N), rs in sorted(by_N.items()):
        if c != case:
            continue
        base = next((r["wall_time_s"] for r in rs if r["solver"] == "Thomas"), None)
        if not base:
            continue
        for r in rs:
            if r["solver"] == "Thomas" or not r["wall_time_s"]:
                continue
            xs.append(N); ys.append(r["wall_time_s"] / base)
            cs.append(SOLVER_COLOUR.get(r["solver"], "gray")); labels.append(r["solver"])
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    seen = set()
    for x, y, c, l in zip(xs, ys, cs, labels):
        ax.semilogy(x, y, "o", color=c, label=l if l not in seen else None,
                   markersize=8)
        seen.add(l)
    ax.set_xlabel("N")
    ax.set_ylabel("wall time / Thomas wall time")
    ax.set_title(f"Quantum overhead vs Thomas - {case}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = PLOTS_DIR / f"overhead_vs_thomas_{case}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
#  Plot 6 - error decomposition
# ============================================================================

def plot_error_decomposition(case: str, by_solver: dict, plt) -> Path | None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    any_line = False
    disc_plotted = False
    for (c, solver), rs in sorted(by_solver.items()):
        if c != case or solver == "Thomas":
            continue
        Ns = [r["N"] for r in rs if r.get("err_vs_thomas") is not None]
        alg = [r["err_vs_thomas"] for r in rs if r.get("err_vs_thomas") is not None]
        if Ns:
            ax.semilogy(Ns, [max(v, 1e-6) for v in alg], "o-",
                       color=SOLVER_COLOUR.get(solver), label=f"{solver} (vs Thomas)")
            any_line = True
        if not disc_plotted:
            Nd = [r["N"] for r in rs if r.get("err_thomas_vs_exact") is not None]
            disc = [r["err_thomas_vs_exact"] for r in rs
                   if r.get("err_thomas_vs_exact") is not None]
            if Nd:
                ax.semilogy(Nd, [max(v, 1e-6) for v in disc], "k--", lw=1.6,
                           label="discretisation (Thomas vs exact)")
                disc_plotted = True
    if not any_line:
        plt.close(fig)
        return None
    ax.set_xlabel("N")
    ax.set_ylabel("error (%)")
    ax.set_title(f"Error decomposition - {case}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = PLOTS_DIR / f"error_decomposition_{case}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
#  Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default=None, help="restrict to one case")
    ap.add_argument("--N", type=int, default=None, help="restrict to one N")
    ap.add_argument("--list", action="store_true",
                    help="list available (case, N, solver) combinations and exit")
    args = ap.parse_args()

    rows = load_results()
    if args.case:
        rows = [r for r in rows if r["case"] == args.case]
    if args.N:
        rows = [r for r in rows if r["N"] == args.N]
    if not rows:
        raise SystemExit("No matching rows.")

    if args.list:
        for r in sorted(rows, key=lambda r: (r["case"], r["N"], r["solver"])):
            print(f"  {r['case']:<38} N={r['N']:<4} {r['solver']:<8} "
                 f"scheme={r['scheme']:<10} {r['stop_reason']}")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        raise SystemExit("matplotlib is required for this script.")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    by_case_N = group_by_case_N(rows)
    by_case_solver = group_by_case_solver(rows)
    cases = sorted({c for c, _ in by_case_N})

    made = []
    for (case, N), case_rows in sorted(by_case_N.items()):
        p = plot_fields(case, N, case_rows, plt, TwoSlopeNorm)
        if p: made.append(p)
        p = plot_convergence(case, N, case_rows, plt)
        if p: made.append(p)

    for case in cases:
        for fn in (plot_accuracy_vs_n, plot_cost_vs_n, plot_error_decomposition):
            p = fn(case, by_case_solver, plt)
            if p: made.append(p)
        p = plot_overhead(case, by_case_N, plt)
        if p: made.append(p)

    print(f"Wrote {len(made)} plot(s) to {PLOTS_DIR}")
    for p in made:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()