"""
Provides graphical visualisation utilities for the 1D Poisson benchmark evaluations.

This module leverages Matplotlib to generate comparative visualisations of 
the spatial solution curves and associated error profiles, structurally mirroring 
the figures presented in the primary reference literature.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark.metrics import BenchmarkResult


# Define the output directory for generated figure assets.
RESULTS_DIR = Path("results")


# ── Graphical Visualisation ───────────────────────────────────────────────────

def plot_solution_comparison(
    thomas_br: BenchmarkResult,
    hhl_br: BenchmarkResult,
    save_fig: bool = False,
) -> None:
    """
    Generates a two-panel comparative visualisation of the benchmark outcomes.

    The left panel displays the spatial solution curves (Analytical, Thomas, 
    and HHL algorithms). The right panel delineates the corresponding error 
    profile (either absolute relative error or absolute deviation from the 
    classical reference, contingent upon boundary condition homogeneity).

    Parameters
    ----------
    thomas_br : BenchmarkResult
        Data structure containing the classical Thomas solver metrics.
    hhl_br : BenchmarkResult
        Data structure containing the quantum HHL solver metrics.
    save_fig : bool, default=False
        If True, the generated figure is exported to the local results directory 
        as a PNG file prior to rendering.
    """
    cfg = hhl_br.config
    x   = hhl_br.x

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # ── Left Panel: Spatial Solution Curves ───────────────────────────────────
    if hhl_br.u_exact is not None:
        ax1.plot(x, hhl_br.u_exact, "k-", lw=2, label="Analytical")
        
    ax1.plot(x, thomas_br.u_solver, "g-o", lw=1.5, ms=5, label="Thomas")
    ax1.plot(x, hhl_br.u_solver, "r--*", lw=1.5, ms=5, label="HHL")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    ax1.set_title(
        f"f = {cfg.source_fn},  N = {cfg.N},  ε = {cfg.epsilon},  "
        f"α = {cfg.alpha},  β = {cfg.beta}"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Right Panel: Error Profile ────────────────────────────────────────────
    if hhl_br.rel_error is not None:
        # Near-zero nodes (NaN) are assigned a 0.0 value purely to maintain 
        # visual continuity within the plotted curve.
        rel_err_plot = np.where(np.isnan(hhl_br.rel_error), 0.0, hhl_br.rel_error)
        ax2_color = "r"
        ax2.plot(x, rel_err_plot, color=ax2_color, lw=1.5)
        ax2.set_ylabel("HHL absolute relative error (%)", color=ax2_color)
        ax2.tick_params(axis="y", labelcolor=ax2_color)
    else:
        # Evaluations featuring non-homogeneous boundary conditions utilise 
        # the absolute deviation from the classical Thomas algorithm.
        ax2.plot(x, hhl_br.abs_error, "r-", lw=1.5)
        ax2.set_ylabel("|u_HHL − u_Thomas|")

    ax2.set_xlabel("x")
    ax2.set_title("HHL error profile")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_fig:
        RESULTS_DIR.mkdir(exist_ok=True)
        fname = (
            RESULTS_DIR /
            f"{cfg.source_fn}_N{cfg.N}_eps{cfg.epsilon:.4f}"
            f"_a{cfg.alpha}_b{cfg.beta}.png"
        )
        plt.savefig(fname, dpi=150)
        print(f"  Figure exported to {fname}")

    plt.show()


def plot_sweep_pairs(
    results: list[BenchmarkResult],
    save_fig: bool = False,
) -> None:
    """
    Sequentially visualises a collection of paired benchmark results.

    This routine iterates through a flat array of BenchmarkResult objects, 
    presuming a strict alternating structure of [Thomas, HHL, Thomas, HHL, ...], 
    which corresponds to the standard output format of the automated sweep drivers.

    Parameters
    ----------
    results : list[BenchmarkResult]
        Flat list containing paired algorithmic outcomes.
    save_fig : bool, default=False
        Flag passed to the underlying plotting routine to command file exportation.
    """
    for i in range(0, len(results), 2):
        thomas_br = results[i]
        hhl_br    = results[i + 1]
        
        # Visualisation is strictly bypassed if neither an analytical nor a 
        # classical reference is available for comparative assessment.
        if hhl_br.u_exact is not None or hhl_br.u_thomas is not None:
            plot_solution_comparison(thomas_br, hhl_br, save_fig=save_fig)