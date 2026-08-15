"""
Execution entry point for the 2D Poisson quantum linear solver benchmark.

This script orchestrates the systematic execution of the 2D simulated sweeps
detailed in Sections IV E-F of the primary reference literature.

Execution Time Note
-------------------
An HHL solve at N=8, ε=0.01 typically requires 50 to 100 line-Jacobi iterations,
each containing 8 HHL circuit simulations — approximately 10 to 30 minutes per
configuration on local hardware. Run `sweep_g` together with a single `sweep_e`
configuration to verify correctness before committing to the full suite.
"""
import sys
from pathlib import Path

# -- System Path Resolution ----------------------------------------------------

# Dynamically resolve the project root directory (one level up from this script)
# and append it to the system path to enable absolute imports.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from benchmark.runner import (
    sweep_e, sweep_f, sweep_g,
    save_to_csv_2d, _plot_2d_pairs,
    OUTPUT_CSV, SAVE_FIGS,
)
from benchmark.reporting import print_result_table_2d


# -- Primary Execution Sequence ------------------------------------------------

if __name__ == "__main__":

    # Evaluate condition number scaling limits (rapid execution, no HHL).
    sweep_g()

    # Section IV E: Homogeneous boundary conditions.
    results_e = sweep_e()
    print("\n--- Sweep E ---")
    print_result_table_2d(results_e)
    if OUTPUT_CSV:
        save_to_csv_2d(results_e, "sweep_e_homogeneous_2d.csv")
    _plot_2d_pairs(results_e, use_relative_error=True, save_fig=SAVE_FIGS)

    # Section IV F: Non-homogeneous boundary conditions.
    results_f = sweep_f()
    print("\n--- Sweep F ---")
    print_result_table_2d(results_f)
    if OUTPUT_CSV:
        save_to_csv_2d(results_f, "sweep_f_nonhomogeneous_2d.csv")

    # Non-homogeneous evaluations prioritise absolute error mappings to preserve
    # strict alignment with the graphical outputs of the primary literature.
    _plot_2d_pairs(results_f, use_relative_error=False, save_fig=SAVE_FIGS)