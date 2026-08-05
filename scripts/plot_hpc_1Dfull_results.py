#!/usr/bin/env python3
"""
Post-processing script: reads the JSON/CSV output from run_hpc_full.py
and produces publication-quality plots for thesis Chapter 5.

Plots produced:
  1. Solution profiles: Thomas vs HHL vs VQLS vs QSVT (per case, per N)
  2. Max relative error vs N: all solvers on one axis (log-log)
  3. Residual vs N: all solvers (log-log)
  4. Wall time vs N: all solvers (log-log)
  5. HET 1D: potential profiles and electric field (sub-cases 3a, 3b, 3c)

Usage:
  python plot_hpc_results.py --results-dir results_hpc/hpc_run
  python plot_hpc_results.py --results-dir results_hpc/hpc_run --save-pdf

Author : Juan Antonio Trobajo Flecha
Date   : July 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Use a non-interactive backend when running on HPC (no display).
matplotlib.use("Agg")

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.labelsize":    12,
    "axes.titlesize":    12,
    "legend.fontsize":   10,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "lines.linewidth":   1.8,
    "lines.markersize":  4,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
})

# ── Colour / marker scheme (consistent across all plots) ─────────────────────
SOLVER_STYLE = {
    "Thomas": {"color": "#1f77b4", "marker": "o",  "ls": "-",  "label": "Thomas (classical)"},
    "HHL":    {"color": "#ff7f0e", "marker": "s",  "ls": "--", "label": "HHL"},
    "VQLS":   {"color": "#2ca02c", "marker": "^",  "ls": "-.", "label": "VQLS"},
    "QSVT":   {"color": "#d62728", "marker": "D",  "ls": ":",  "label": "QSVT"},
}

CASE_LABELS = {
    "1D_Poisson_fS_hom":              r"1D Poisson, $f_S$, hom. BCs",
    "1D_Poisson_fL_hom":              r"1D Poisson, $f_L$, hom. BCs",
    "1D_Poisson_fH_hom":              r"1D Poisson, $f_H$, hom. BCs",
    "1D_Poisson_fS_nonhom":           r"1D Poisson, $f_S$, non-hom. BCs",
    "HET_1D_3a_linear_hom":           r"HET 1D, linear profile, hom. BCs",
    "HET_1D_3b_gaussian_Vd300":       r"HET 1D, Gaussian, $V_d=300$ V",
    "HET_1D_3c_gaussian_NeumannDirichlet": r"HET 1D, Gaussian, Neumann–Dirichlet BCs",
}


# ============================================================================
#  Data loading
# ============================================================================

def load_results(results_dir: Path) -> list[dict]:
    json_path = results_dir / "results_full.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Results file not found: {json_path}")
    with open(json_path) as f:
        return json.load(f)


def load_solution(results_dir: Path, case: str, solver: str, N: int) -> dict | None:
    """Load a solution NPZ file. Returns dict with x, u, u_exact (or None)."""
    fname = results_dir / f"solutions_{case}_{solver}_N{N}.npz"
    if not fname.exists():
        return None
    data = np.load(fname)
    out = {"x": data["x"], "u": data["u_solver"]}
    if "u_exact" in data:
        out["u_exact"] = data["u_exact"]
    return out


def group_results(results: list[dict]) -> dict:
    """
    Group results by (case, solver) -> list of dicts sorted by N.
    Returns: {case: {solver: [result_dicts sorted by N]}}
    """
    grouped: dict = {}
    for r in results:
        case   = r["case"]
        solver = r["solver"]
        grouped.setdefault(case, {}).setdefault(solver, []).append(r)
    for case in grouped:
        for solver in grouped[case]:
            grouped[case][solver].sort(key=lambda x: x["N"])
    return grouped


# ============================================================================
#  Plot 1: Solution profiles (Thomas vs quantum solvers)
# ============================================================================

def plot_solution_profiles(
    results_dir: Path,
    grouped: dict,
    save_pdf: bool,
    N_plot: int = 8,
) -> None:
    """
    For each case, plot the solution profiles at N=N_plot for all solvers.
    One figure per case, two panels: solution + pointwise error.
    """
    for case, solver_data in grouped.items():
        # Collect available solvers for this case at N_plot.
        available = {}
        for solver in ["Thomas", "HHL", "VQLS", "QSVT"]:
            if solver not in solver_data:
                continue
            sol = load_solution(results_dir, case, solver, N_plot)
            if sol is None:
                continue
            available[solver] = sol

        if not available:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        ax_sol, ax_err = axes

        # Reference exact solution (from Thomas or NPZ).
        u_exact = None
        if "Thomas" in available and "u_exact" in available["Thomas"]:
            u_exact = available["Thomas"]["u_exact"]
            x_ref   = available["Thomas"]["x"]

        for solver, sol in available.items():
            st = SOLVER_STYLE[solver]
            ax_sol.plot(sol["x"], sol["u"],
                        color=st["color"], marker=st["marker"],
                        ls=st["ls"], label=st["label"],
                        markevery=max(1, len(sol["x"]) // 32))

        if u_exact is not None:
            ax_sol.plot(x_ref, u_exact, "k--", lw=1.2, label="Exact", zorder=0)

        ax_sol.set_xlabel(r"$x$")
        ax_sol.set_ylabel(r"$u(x)$")
        ax_sol.set_title(f"{CASE_LABELS.get(case, case)}\n$N={N_plot}$")
        ax_sol.legend(fontsize=9)

        # Pointwise absolute error vs Thomas reference.
        u_thomas = available.get("Thomas", {}).get("u")
        if u_thomas is not None:
            for solver, sol in available.items():
                if solver == "Thomas":
                    continue
                st = SOLVER_STYLE[solver]
                err = np.abs(sol["u"] - u_thomas)
                ax_err.semilogy(sol["x"], err + 1e-16,
                                color=st["color"], marker=st["marker"],
                                ls=st["ls"], label=st["label"],
                                markevery=max(1, len(sol["x"]) // 8))

        ax_err.set_xlabel(r"$x$")
        ax_err.set_ylabel(r"$|u_\mathrm{solver} - u_\mathrm{Thomas}|$")
        ax_err.set_title("Pointwise absolute error vs Thomas")
        ax_err.legend(fontsize=9)

        fig.tight_layout()
        stem = f"fig_profiles_{case}_N{N_plot}"
        _save_fig(fig, results_dir, stem, save_pdf)
        plt.close(fig)


# ============================================================================
#  Plot 2: Max relative error vs N (convergence plot)
# ============================================================================

def plot_error_vs_N(
    grouped: dict,
    results_dir: Path,
    save_pdf: bool,
    cases_to_plot: list[str] | None = None,
) -> None:
    """
    Log-log plot of max relative error vs N for all solvers.
    One figure per case (or a combined figure for the generic Poisson cases).
    """
    if cases_to_plot is None:
        cases_to_plot = list(grouped.keys())

    for case in cases_to_plot:
        if case not in grouped:
            continue
        solver_data = grouped[case]

        fig, ax = plt.subplots(figsize=(7, 5))

        for solver in ["Thomas", "HHL", "VQLS", "QSVT"]:
            if solver not in solver_data:
                continue
            rows = solver_data[solver]
            Ns    = [r["N"] for r in rows if r["max_rel_err"] is not None]
            errs  = [r["max_rel_err"] for r in rows if r["max_rel_err"] is not None]
            if not Ns:
                continue
            st = SOLVER_STYLE[solver]
            ax.loglog(Ns, errs,
                      color=st["color"], marker=st["marker"],
                      ls=st["ls"], label=st["label"])

        # Reference O(N^-2) line.
        Ns_ref = np.array([4, 8, 16, 32, 64])
        ax.loglog(Ns_ref, 10.0 / Ns_ref**2, "k:", lw=1.0, label=r"$\mathcal{O}(N^{-2})$")

        ax.set_xlabel(r"$N$ (system size)")
        ax.set_ylabel(r"Max relative error (\%)")
        ax.set_title(f"Convergence: {CASE_LABELS.get(case, case)}")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks([4, 8, 16, 32, 64])
        ax.legend()
        fig.tight_layout()
        _save_fig(fig, results_dir, f"fig_error_vs_N_{case}", save_pdf)
        plt.close(fig)


# ============================================================================
#  Plot 3: Residual vs N
# ============================================================================

def plot_residual_vs_N(
    grouped: dict,
    results_dir: Path,
    save_pdf: bool,
) -> None:
    """Log-log plot of ||Au-b||/||b|| vs N for all solvers and cases."""
    for case, solver_data in grouped.items():
        fig, ax = plt.subplots(figsize=(7, 5))
        any_data = False

        for solver in ["Thomas", "HHL", "VQLS", "QSVT"]:
            if solver not in solver_data:
                continue
            rows = solver_data[solver]
            Ns   = [r["N"] for r in rows if r["residual"] is not None
                    and not np.isnan(float(r["residual"]))]
            res  = [r["residual"] for r in rows if r["residual"] is not None
                    and not np.isnan(float(r["residual"]))]
            if not Ns:
                continue
            st = SOLVER_STYLE[solver]
            ax.loglog(Ns, res,
                      color=st["color"], marker=st["marker"],
                      ls=st["ls"], label=st["label"])
            any_data = True

        if not any_data:
            plt.close(fig)
            continue

        ax.set_xlabel(r"$N$ (system size)")
        ax.set_ylabel(r"Relative residual $\|Au - b\| / \|b\|$")
        ax.set_title(f"Residual: {CASE_LABELS.get(case, case)}")
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks([4, 8, 16, 32, 64])
        ax.legend()
        fig.tight_layout()
        _save_fig(fig, results_dir, f"fig_residual_vs_N_{case}", save_pdf)
        plt.close(fig)


# ============================================================================
#  Plot 4: Wall time vs N
# ============================================================================

def plot_time_vs_N(
    grouped: dict,
    results_dir: Path,
    save_pdf: bool,
) -> None:
    """Log-log plot of wall time vs N for all solvers."""
    # Aggregate across all cases for a single summary plot.
    fig, ax = plt.subplots(figsize=(7, 5))

    # Use the generic Poisson fS case as representative.
    case = "1D_Poisson_fS_hom"
    if case not in grouped:
        plt.close(fig)
        return

    solver_data = grouped[case]
    for solver in ["Thomas", "HHL", "VQLS", "QSVT"]:
        if solver not in solver_data:
            continue
        rows = solver_data[solver]
        Ns   = [r["N"] for r in rows if r["wall_time_s"] > 0]
        ts   = [r["wall_time_s"] for r in rows if r["wall_time_s"] > 0]
        if not Ns:
            continue
        st = SOLVER_STYLE[solver]
        ax.loglog(Ns, ts,
                  color=st["color"], marker=st["marker"],
                  ls=st["ls"], label=st["label"])

    ax.set_xlabel(r"$N$ (system size)")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(r"Computational cost: 1D Poisson, $f_S$, homogeneous BCs")
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks([4, 8, 16, 32, 64])
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, results_dir, "fig_time_vs_N", save_pdf)
    plt.close(fig)


# ============================================================================
#  Plot 5: HET 1D — potential profiles and electric field
# ============================================================================

def plot_het_1d(
    results_dir: Path,
    grouped: dict,
    save_pdf: bool,
    N_plot: int = 8,
) -> None:
    """
    Three-panel figure for the HET 1D cases:
    Left: sub-case 3a potential; Centre: sub-case 3b potential + E-field;
    Right: sub-case 3c potential (new Neumann-Dirichlet benchmark).
    """
    het_cases = [
        "HET_1D_3a_linear_hom",
        "HET_1D_3b_gaussian_Vd300",
        "HET_1D_3c_gaussian_NeumannDirichlet",
    ]
    panel_titles = [
        r"Sub-case 3a: linear profile, hom. BCs",
        r"Sub-case 3b: Gaussian, $V_d=300$ V",
        r"Sub-case 3c: Gaussian, Neumann–Dirichlet (new)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, case, title in zip(axes, het_cases, panel_titles):
        if case not in grouped:
            ax.set_title(title + "\n(no data)")
            continue

        solver_data = grouped[case]
        for solver in ["Thomas", "HHL", "VQLS"]:
            if solver not in solver_data:
                continue
            sol = load_solution(results_dir, case, solver, N_plot)
            if sol is None:
                continue
            st = SOLVER_STYLE[solver]
            ax.plot(sol["x"], sol["u"],
                    color=st["color"], marker=st["marker"],
                    ls=st["ls"], label=st["label"],
                    markevery=max(1, len(sol["x"]) // 6))

        # Exact solution overlay if available.
        thomas_sol = load_solution(results_dir, case, "Thomas", N_plot)
        if thomas_sol is not None and "u_exact" in thomas_sol:
            x_fine = np.linspace(thomas_sol["x"][0], thomas_sol["x"][-1], 300)
            # Interpolate exact from the saved NPZ.
            u_ex_interp = np.interp(x_fine, thomas_sol["x"], thomas_sol["u_exact"])
            ax.plot(x_fine, u_ex_interp, "k--", lw=1.2, label="Exact", zorder=0)

        ax.set_xlabel(r"$x / L$")
        ax.set_ylabel(r"$\phi$ (normalised)")
        ax.set_title(title + f"\n$N={N_plot}$")
        ax.legend(fontsize=8)

    fig.suptitle("HET 1D Axial Poisson — Potential Profiles", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, results_dir, f"fig_het_1d_profiles_N{N_plot}", save_pdf)
    plt.close(fig)

    # Separate electric field plot for sub-case 3b.
    case = "HET_1D_3b_gaussian_Vd300"
    if case not in grouped:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    for solver in ["Thomas", "HHL", "VQLS"]:
        sol = load_solution(results_dir, case, solver, N_plot)
        if sol is None:
            continue
        E = -np.gradient(sol["u"], sol["x"])
        st = SOLVER_STYLE[solver]
        ax.plot(sol["x"], np.abs(E),
                color=st["color"], marker=st["marker"],
                ls=st["ls"], label=st["label"],
                markevery=max(1, len(sol["x"]) // 6))

    ax.set_xlabel(r"$x / L$")
    ax.set_ylabel(r"$|E|$ (V/m)")
    ax.set_title(r"HET 1D: Electric field magnitude, $V_d=300$ V, $N=" + str(N_plot) + r"$")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, results_dir, f"fig_het_1d_Efield_N{N_plot}", save_pdf)
    plt.close(fig)


# ============================================================================
#  Plot 6: Combined summary table figure
# ============================================================================

def plot_summary_table(
    grouped: dict,
    results_dir: Path,
    save_pdf: bool,
) -> None:
    """
    A 2×2 grid of error-vs-N plots for the four main generic Poisson cases,
    suitable for a single thesis figure.
    """
    cases = [
        "1D_Poisson_fS_hom",
        "1D_Poisson_fL_hom",
        "1D_Poisson_fH_hom",
        "1D_Poisson_fS_nonhom",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes_flat = axes.flatten()

    for ax, case in zip(axes_flat, cases):
        if case not in grouped:
            ax.set_title(CASE_LABELS.get(case, case) + "\n(no data)")
            continue
        solver_data = grouped[case]
        for solver in ["Thomas", "HHL", "VQLS", "QSVT"]:
            if solver not in solver_data:
                continue
            rows = solver_data[solver]
            Ns   = [r["N"] for r in rows if r["max_rel_err"] is not None]
            errs = [r["max_rel_err"] for r in rows if r["max_rel_err"] is not None]
            if not Ns:
                continue
            st = SOLVER_STYLE[solver]
            ax.loglog(Ns, errs,
                      color=st["color"], marker=st["marker"],
                      ls=st["ls"], label=st["label"])

        Ns_ref = np.array([4, 8, 16, 32, 64])
        ax.loglog(Ns_ref, 10.0 / Ns_ref**2, "k:", lw=1.0, label=r"$\mathcal{O}(N^{-2})$")
        ax.set_xlabel(r"$N$")
        ax.set_ylabel(r"Max rel. error (\%)")
        ax.set_title(CASE_LABELS.get(case, case))
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks([4, 8, 16, 32])
        ax.legend(fontsize=8)

    fig.suptitle("1D Poisson: Algorithm Comparison — All Cases", fontsize=13)
    fig.tight_layout()
    _save_fig(fig, results_dir, "fig_summary_generic_poisson", save_pdf)
    plt.close(fig)


# ============================================================================
#  Utility
# ============================================================================

def _save_fig(fig: plt.Figure, results_dir: Path, stem: str, save_pdf: bool) -> None:
    """Save figure as PNG (always) and PDF (if requested)."""
    png_path = results_dir / f"{stem}.png"
    fig.savefig(png_path)
    print(f"  Saved: {png_path}")
    if save_pdf:
        pdf_path = results_dir / f"{stem}.pdf"
        fig.savefig(pdf_path)
        print(f"  Saved: {pdf_path}")


# ============================================================================
#  Main
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Post-process and plot HPC benchmark results."
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results_hpc/results/1Dhpc_run"),
        help="Directory containing results_full.json and solution NPZ files."
    )
    parser.add_argument(
        "--save-pdf", action="store_true",
        help="Also save figures as PDF (in addition to PNG)."
    )
    parser.add_argument(
        "--N-profile", type=int, default=32,
        help="N value to use for solution profile plots (default: 32)."
    )
    args = parser.parse_args()

    print(f"Loading results from: {args.results_dir}")
    results = load_results(args.results_dir)
    grouped = group_results(results)

    print(f"Found {len(results)} result rows across {len(grouped)} cases.")
    print("Generating plots...")

    plot_solution_profiles(args.results_dir, grouped, args.save_pdf, N_plot=args.N_profile)
    plot_error_vs_N(grouped, args.results_dir, args.save_pdf)
    plot_residual_vs_N(grouped, args.results_dir, args.save_pdf)
    plot_time_vs_N(grouped, args.results_dir, args.save_pdf)
    plot_het_1d(args.results_dir, grouped, args.save_pdf, N_plot=args.N_profile)
    plot_summary_table(grouped, args.results_dir, args.save_pdf)

    print(f"\nAll figures saved to: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()