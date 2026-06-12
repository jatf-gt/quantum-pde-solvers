"""
Provides graphical visualisation utilities for the 1D and 2D Poisson benchmarks.

This module leverages Matplotlib to generate comparative visualisations of 
spatial solution curves, contour mappings, and convergence histories, structurally 
mirroring the figures presented in the primary reference literature.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark.metrics import BenchmarkResult, BenchmarkResult2D

# Define the output directory for generated figure assets.
RESULTS_DIR = Path("results")


# ── 1D Graphical Visualisation ────────────────────────────────────────────────

def plot_solution_comparison_1d(
    thomas_br: BenchmarkResult,
    hhl_br: BenchmarkResult,
    save_fig: bool = False,
) -> None:
    """
    Generates a two-panel comparative visualisation of the 1D benchmark outcomes.

    The left panel displays the spatial solution curves (Analytical, Thomas, 
    and HHL algorithms). The right panel delineates the corresponding error 
    profile.

    Parameters
    ----------
    thomas_br : BenchmarkResult
        Data structure containing the classical Thomas solver metrics.
    hhl_br : BenchmarkResult
        Data structure containing the quantum HHL solver metrics.
    save_fig : bool, default=False
        If True, exports the generated figure to the local results directory.
    """
    cfg = hhl_br.config
    x   = hhl_br.x

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # ── Left Panel: Solution Curves ───────────────────────────────────────────
    if hhl_br.u_exact is not None:
        ax1.plot(x, hhl_br.u_exact, "k-", lw=2, label="Analytical")
        
    ax1.plot(x, thomas_br.u_solver, "g-o", lw=1.5, ms=5, label="Thomas")
    ax1.plot(x, hhl_br.u_solver, "r--*", lw=1.5, ms=5, label="HHL")
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    ax1.set_title(
        f"f={cfg.source_fn},  N={cfg.N},  ε={cfg.epsilon},  "
        f"α={cfg.alpha},  β={cfg.beta}"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Right Panel: Error Profile ────────────────────────────────────────────
    if hhl_br.rel_error is not None:
        rel_plot = np.where(np.isnan(hhl_br.rel_error), 0.0, hhl_br.rel_error)
        ax2_color = "r"
        ax2.plot(x, rel_plot, color=ax2_color, lw=1.5)
        ax2.set_ylabel("HHL absolute relative error (%)", color=ax2_color)
        ax2.tick_params(axis="y", labelcolor=ax2_color)
    else:
        ax2.plot(x, hhl_br.abs_error, "r-", lw=1.5)
        ax2.set_ylabel("|u_HHL − u_Thomas|")

    ax2.set_xlabel("x")
    ax2.set_title("HHL error profile")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    _maybe_save(
        fig, save_fig,
        f"1d_{cfg.source_fn}_N{cfg.N}_eps{cfg.epsilon:.4f}"
        f"_a{cfg.alpha}_b{cfg.beta}.png"
    )
    plt.show()


# ── 2D Graphical Visualisation ────────────────────────────────────────────────

def plot_solution_contours_2d(
    thomas_br: BenchmarkResult2D,
    hhl_br: BenchmarkResult2D,
    use_relative_error: bool = True,
    save_fig: bool = False,
) -> None:
    """
    Generates a four-panel contour mapping to evaluate 2D benchmark outcomes.

    Replicates the structural layout of Figures 10, 12, and 14 from the primary 
    reference:
        (a) Top-Left: Thomas solution contour mapping.
        (b) Top-Right: HHL solution contour mapping.
        (c) Bottom-Left: Thomas spatial error contour.
        (d) Bottom-Right: HHL spatial error contour.

    Parameters
    ----------
    thomas_br : BenchmarkResult2D
        Data structure containing classical metrics.
    hhl_br : BenchmarkResult2D
        Data structure containing quantum metrics.
    use_relative_error : bool, default=True
        Dictates error topography. If False, absolute deviation is plotted. This 
        is explicitly required for non-homogeneous constraints where artificial 
        near-zero evaluations inflate relative metrics.
    save_fig : bool, default=False
        If True, exports the generated figure to the local results directory.
    """
    cfg = hhl_br.config
    X   = hhl_br.X
    Y   = hhl_br.Y

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"2D Poisson — f={cfg.source_fn}, N={cfg.N}, ε={cfg.epsilon}",
        fontsize=13,
    )

    # Establish an equivalent colormap spectrum across solution panels.
    u_all = np.concatenate([thomas_br.u_solver.ravel(), hhl_br.u_solver.ravel()])
    u_min, u_max = u_all.min(), u_all.max()
    levels_u = np.linspace(u_min, u_max, 20)

    # (a) Thomas Spatial Solution
    _contour_panel(
        axes[0, 0], X, Y, thomas_br.u_solver,
        levels_u, "Thomas solution", fig,
    )

    # (b) HHL Spatial Solution
    _contour_panel(
        axes[0, 1], X, Y, hhl_br.u_solver,
        levels_u, "HHL solution", fig,
    )

    # (c) Thomas Spatial Error Profile
    if use_relative_error and thomas_br.rel_error is not None:
        err_thomas = np.where(
            np.isnan(thomas_br.rel_error), 0.0, thomas_br.rel_error
        )
        err_label = "Relative error (%)"
    else:
        err_thomas = thomas_br.abs_error
        err_label  = "Absolute error"

    _error_panel(
        axes[1, 0], X, Y, err_thomas,
        f"Thomas {err_label}", fig,
    )

    # (d) HHL Spatial Error Profile
    if use_relative_error and hhl_br.rel_error is not None:
        err_hhl = np.where(
            np.isnan(hhl_br.rel_error), 0.0, hhl_br.rel_error
        )
    else:
        err_hhl = hhl_br.abs_error

    _error_panel(
        axes[1, 1], X, Y, err_hhl,
        f"HHL {err_label}", fig,
    )

    plt.tight_layout()
    _maybe_save(
        fig, save_fig,
        f"2d_{cfg.source_fn}_N{cfg.N}_eps{cfg.epsilon:.4f}.png"
    )
    plt.show()


def plot_convergence_history(
    results: list[BenchmarkResult2D],
    save_fig: bool = False,
) -> None:
    """
    Visualises the natural logarithmic iteration error decay against step progression.

    This routine facilitates comparative evaluations of algorithmic stability 
    across disparate convergence thresholds or solver typologies (e.g., assessing 
    the impact of Trotterisation noise on line-Jacobi execution).

    Parameters
    ----------
    results : list[BenchmarkResult2D]
        Array of aggregated benchmark execution data structures.
    save_fig : bool, default=False
        If True, exports the generated figure to the local results directory.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for r in results:
        cfg    = r.config
        errors = np.array(r.iteration_errors)

        # Enforce strict domain requirements for logarithmic operations.
        valid_mask = errors > 0
        iters      = np.where(valid_mask)[0] + 1
        ln_errors  = np.log(errors[valid_mask])

        label = (
            f"{r.solver}, f={cfg.source_fn}, "
            f"N={cfg.N}, ε={cfg.epsilon}"
        )
        ax.plot(iters, ln_errors, lw=1.5, label=label)

        # Delineate terminal convergence states visually.
        if r.converged and r.iterations <= len(errors):
            conv_iter = r.iterations
            conv_err  = errors[conv_iter - 1]
            if conv_err > 0:
                ax.axvline(
                    conv_iter, color="grey",
                    linestyle="--", alpha=0.5,
                )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("ln(iteration error)")
    ax.set_title("Line-Jacobi convergence history")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if results:
        cfg = results[0].config
        fname = f"convergence_{cfg.source_fn}_N{cfg.N}.png"
    else:
        fname = "convergence.png"

    _maybe_save(fig, save_fig, fname)
    plt.show()


def plot_sweep_pairs_1d(
    results: list[BenchmarkResult],
    save_fig: bool = False,
) -> None:
    """
    Sequentially visualises a collection of paired 1D benchmark results.

    Iterates through a flat array of `BenchmarkResult` objects, presupposing 
    a strict alternating architecture of [Thomas, HHL, Thomas, ...].
    """
    for i in range(0, len(results), 2):
        thomas_br = results[i]
        hhl_br    = results[i + 1]
        
        if hhl_br.u_exact is not None or hhl_br.u_thomas is not None:
            plot_solution_comparison_1d(thomas_br, hhl_br, save_fig=save_fig)


# ── Private Utility Methods ───────────────────────────────────────────────────

def _contour_panel(ax, X, Y, Z, levels, title, fig) -> None:
    """Renders a filled topological contour plot with overlaid delineations."""
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="viridis")
    ax.contour(X, Y, Z, levels=levels, colors="white", linewidths=0.4, alpha=0.5)
    fig.colorbar(cf, ax=ax, shrink=0.85)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")


def _error_panel(ax, X, Y, Z, title, fig) -> None:
    """Renders an error topography map restricted to sequential colour palettes."""
    levels = np.linspace(0, np.max(Z) if np.max(Z) > 0 else 1.0, 20)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="hot_r")
    fig.colorbar(cf, ax=ax, shrink=0.85)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")


def _maybe_save(fig, save_fig: bool, filename: str) -> None:
    """Handles operational file I/O for asset export directives."""
    if save_fig:
        RESULTS_DIR.mkdir(exist_ok=True)
        path = RESULTS_DIR / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Figure exported to {path}")