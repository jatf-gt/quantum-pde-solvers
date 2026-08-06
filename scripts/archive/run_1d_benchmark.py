"""
Execution entry point for the 1D Poisson quantum linear solver benchmark.

This script orchestrates the systematic execution of the 1D simulated sweeps
(Sections IV A-D), ensuring appropriate system path resolution prior to
invoking the primary benchmark modules. It automates algorithmic data generation,
CSV serialisation, and Matplotlib visualisations without exposing the internal
module architecture.
"""
import sys
from pathlib import Path

# ── System Path Resolution ────────────────────────────────────────────────────

# Dynamically resolve the project root directory (one level up from this script)
# and append it to the system path to enable absolute imports.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from benchmark.runner import (
    sweep_a, sweep_b, sweep_c, sweep_d,
    save_to_csv_1d, _plot_1d_pairs,
    OUTPUT_CSV, SAVE_FIGS,
)
from benchmark.reporting import print_result_table


# ── Primary Execution Sequence ────────────────────────────────────────────────

if __name__ == "__main__":

    sweep_d()

    results_a = sweep_a()
    print("\n--- Sweep A ---")
    print_result_table(results_a)
    if OUTPUT_CSV:
        save_to_csv_1d(results_a, "sweep_a_homogeneous_1d.csv")
    _plot_1d_pairs(results_a, save_fig=SAVE_FIGS)

    results_b = sweep_b()
    print("\n--- Sweep B ---")
    print_result_table(results_b)
    if OUTPUT_CSV:
        save_to_csv_1d(results_b, "sweep_b_epsilon_1d.csv")

    results_c = sweep_c()
    print("\n--- Sweep C ---")
    print_result_table(results_c)
    if OUTPUT_CSV:
        save_to_csv_1d(results_c, "sweep_c_nonhomogeneous_1d.csv")
    _plot_1d_pairs(results_c, save_fig=SAVE_FIGS)