#!/usr/bin/env python3
"""
example_report.py — copy-me template for a small, laptop-scale report.

Modelled on the structure of scripts/archive/run_meeting5.py (the Meeting 5
progress report: run a section, print a table, save a figure, repeat) but
rebuilt on the current architecture - the case registry (core/cases.py) and
the outer-iteration layer (solvers/outer) - rather than the retired
PoissonProblem2D/thomas_solve_2d/vqls_solve_2d/qsvt_solve_2d stack
run_meeting5.py used, which predates both.

How to use this file
---------------------
Copy it to a new name (e.g. scripts/report_meeting9.py) and edit the lines
marked "# CHANGE ME" below. Everything else is plumbing you can leave alone:
argument-free case building via core.cases, dispatch via
solvers.outer.get_inner (1-D) / solvers.outer.solve (2-D/3-D), a comparison
table, one figure, and one CSV. Runtime at the defaults below is well under a
minute on a laptop with no quantum backend beyond statevector simulation.

Output
------
    results/example_report/report_1d.png
    results/example_report/report_2d.png
    results/example_report/report_metrics.csv
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import cases
from benchmark.diagnostics import BOLD, RED, RESET, colour, rel_err
from solvers.outer import get_inner, solve

RESULTS_DIR = REPO_ROOT / "results" / "example_report"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -- CHANGE ME: pick the cases and resolutions this report covers -------------
# Full case lists: core.cases.available(dim=1), .available(dim=2), .available(dim=3)
CASE_1D = "poisson_1d_fS_hom"
N_1D = 8
SOLVERS_1D = ["thomas", "hhl", "qsvt"]      # any of solvers.outer.available_inner()

CASE_2D = "poisson_2d_sin_pi"
N_2D = 8
INNER_2D = "qsvt"                           # cheapest quantum solver per strip
SCHEME_2D = "fmg"


# -- 1-D section ----------------------------------------------------------------

def run_1d() -> list[dict]:
    """
    Solves CASE_1D at N_1D with every solver in SOLVERS_1D, via the same
    (A, b) -> x inner-solver registry the outer iteration uses per strip.

    Returns
    -------
    list of dict
        One row per solver: name, wall time, residual, error vs the case's
        exact solution (NaN if it has none) and vs the Thomas reference.
    """
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  1-D: {CASE_1D}   N={N_1D}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    case = cases.get(CASE_1D)
    built = case.build(N_1D)
    A, b, u_exact = built.A, built.b, built.exact

    ref = None
    rows = []
    for name in SOLVERS_1D:
        t0 = time.perf_counter()
        try:
            u = get_inner(name, fallback_to_thomas=False)(A, b)
        except Exception as exc:
            print(f"  {RED}[FAIL] {name}: {exc}{RESET}")
            continue
        wall = time.perf_counter() - t0
        if ref is None and name == "thomas":
            ref = u
        residual = float(np.linalg.norm(A @ u - b) / np.linalg.norm(b))
        e_exact = rel_err(u, u_exact) if u_exact is not None else float("nan")
        e_ref = rel_err(u, ref) if ref is not None else float("nan")
        rows.append({"section": "1D", "case": CASE_1D, "N": N_1D, "solver": name,
                     "wall_s": wall, "residual": residual,
                     "err_vs_exact_pct": e_exact, "err_vs_thomas_pct": e_ref,
                     "u": u})
        print(f"  {name:<8} time={wall:>7.3f}s  residual={residual:.3e}  "
              f"{colour(e_exact)}vs exact={e_exact:>7.3f}%{RESET}  "
              f"{colour(e_ref, 0.5, 2.0)}vs Thomas={e_ref:>7.3f}%{RESET}")

    _plot_1d(built.coords[0], u_exact, rows)
    return rows


def _plot_1d(x: np.ndarray, u_exact, rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib unavailable - skipping the 1D figure")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    if u_exact is not None:
        ax1.plot(x, u_exact, "k-", lw=2, label="exact")
    for row in rows:
        ax1.plot(x, row["u"], "o--", ms=4, label=row["solver"])
        ref = u_exact if u_exact is not None else rows[0]["u"]
        ax2.semilogy(x, np.abs(row["u"] - ref) + 1e-18, "o-", ms=3, label=row["solver"])

    ax1.set_xlabel("x"); ax1.set_ylabel("u"); ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_title(f"1D {CASE_1D}, N={N_1D}")
    ax2.set_xlabel("x"); ax2.set_ylabel("|error|"); ax2.legend(); ax2.grid(alpha=0.3)
    ax2.set_title("Pointwise error")

    plt.tight_layout()
    out = RESULTS_DIR / "report_1d.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# -- 2-D section ----------------------------------------------------------------

def run_2d() -> list[dict]:
    """
    Solves CASE_2D at N_2D with Thomas and INNER_2D under SCHEME_2D.

    Returns
    -------
    list of dict
        One row per solver, matching the shape of run_1d()'s rows.
    """
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  2-D: {CASE_2D}   N={N_2D}   scheme={SCHEME_2D}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    case = cases.get(CASE_2D)
    built = case.build(N_2D)
    prob, u_exact = built.problem, built.exact

    ref = None
    rows = []
    for name in ("thomas", INNER_2D):
        t0 = time.perf_counter()
        try:
            res = solve(prob, inner=name, scheme=SCHEME_2D, tol=1e-4)
        except Exception as exc:
            print(f"  {RED}[FAIL] {name}: {exc}{RESET}")
            continue
        wall = time.perf_counter() - t0
        if ref is None and name == "thomas":
            ref = res.u
        e_exact = rel_err(res.u, u_exact) if u_exact is not None else float("nan")
        e_ref = rel_err(res.u, ref) if ref is not None else float("nan")
        rows.append({"section": "2D", "case": CASE_2D, "N": N_2D, "solver": name,
                     "wall_s": wall, "residual": res.residual,
                     "err_vs_exact_pct": e_exact, "err_vs_thomas_pct": e_ref,
                     "u": res.u})
        print(f"  {name:<8} outer={res.n_outer:>4}  time={wall:>7.3f}s  "
              f"{colour(e_exact)}vs exact={e_exact:>7.3f}%{RESET}  "
              f"{colour(e_ref, 0.5, 2.0)}vs Thomas={e_ref:>7.3f}%{RESET}")

    _plot_2d(prob, u_exact, rows)
    return rows


def _plot_2d(prob, u_exact, rows: list[dict]) -> None:
    if u_exact is None or not rows:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib unavailable - skipping the 2D figure")
        return

    x, y = prob.grid()
    fig, axes = plt.subplots(1, 1 + len(rows), figsize=(4 * (1 + len(rows)), 4))
    vmin, vmax = float(u_exact.min()), float(u_exact.max())

    im = axes[0].pcolormesh(x, y, u_exact, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    axes[0].set_title("Exact"); axes[0].set_aspect("equal")
    plt.colorbar(im, ax=axes[0], shrink=0.8)

    for ax, row in zip(axes[1:], rows):
        im = ax.pcolormesh(x, y, row["u"], cmap="RdBu_r", vmin=vmin, vmax=vmax)
        ax.set_title(row["solver"]); ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"2D {CASE_2D}, N={N_2D}")
    plt.tight_layout()
    out = RESULTS_DIR / "report_2d.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# -- CSV export -----------------------------------------------------------------

def write_csv(rows: list[dict]) -> None:
    out = RESULTS_DIR / "report_metrics.csv"
    fields = ["section", "case", "N", "solver", "wall_s", "residual",
             "err_vs_exact_pct", "err_vs_thomas_pct"]
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\n  saved {out}")


# -- Main ----------------------------------------------------------------------

def main() -> None:
    t0 = time.perf_counter()
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  EXAMPLE REPORT — copy scripts/example_report.py and edit "
          f"the CHANGE ME lines{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    rows = run_1d() + run_2d()
    write_csv(rows)

    print(f"\n  Total elapsed: {time.perf_counter() - t0:.1f}s")
    print(f"  All outputs saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
