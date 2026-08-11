"""
Benchmarking framework for quantum PDE solver evaluation.

Public API
──────────
  metrics         BenchmarkResult, CircuitMetrics, compute_* utilities.
  equal_accuracy  Equal-accuracy protocol: sweep_*_equal_accuracy.
  sensitivity     OAT sensitivity: sensitivity_sweep_*, run_all_sensitivity_sweeps.
  runner          Top-level sweep drivers: run_primary_1d, run_equal_accuracy_1d,
                  run_sensitivity_1d.
  results_io      SweepArchive: on-disk persistence layer.
  tables          LaTeX and console table generation.
  plotting        Publication-standard figure generation.
  reporting       Console reporting and diagnostic alerts.
  hardware        Real hardware execution interface and feasibility estimation.
"""

from benchmark.metrics import (
    BenchmarkResult,
    CircuitMetrics,
    compute_residual,
    compute_max_rel_err,
    compute_max_abs_err,
    extract_circuit_metrics,
)
from benchmark.equal_accuracy import (
    EqualAccuracyResult,
    sweep_hhl_equal_accuracy,
    sweep_vqls_equal_accuracy,
    sweep_qsvt_equal_accuracy,
)
from benchmark.sensitivity import (
    SensitivitySweepResult,
    sensitivity_sweep_hhl,
    sensitivity_sweep_vqls,
    sensitivity_sweep_qsvt,
    run_all_sensitivity_sweeps,
)
from benchmark.runner import (
    run_primary_1d,
    run_equal_accuracy_1d,
    run_sensitivity_1d,
)
from benchmark.results_io import SweepArchive
from benchmark import tables, plotting, reporting

__all__ = [
    "BenchmarkResult", "CircuitMetrics",
    "compute_residual", "compute_max_rel_err", "compute_max_abs_err",
    "extract_circuit_metrics",
    "EqualAccuracyResult",
    "sweep_hhl_equal_accuracy", "sweep_vqls_equal_accuracy",
    "sweep_qsvt_equal_accuracy",
    "SensitivitySweepResult",
    "sensitivity_sweep_hhl", "sensitivity_sweep_vqls",
    "sensitivity_sweep_qsvt", "run_all_sensitivity_sweeps",
    "run_primary_1d", "run_equal_accuracy_1d", "run_sensitivity_1d",
    "SweepArchive",
    "tables", "plotting", "reporting",
]