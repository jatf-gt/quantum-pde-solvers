"""
On-disk persistence layer for the benchmarking framework.

Schema contract
---------------
All benchmark output is written to a structured directory tree:

  results/<run_tag>/
    ├-- results_full.json          Complete BenchmarkResult list (primary).
    ├-- results_summary.csv        Flat CSV of all primary fields.
    ├-- equal_accuracy.json        EqualAccuracyResult list.
    ├-- sensitivity_<solver>.json  SensitivitySweepResult list per solver.
    ├-- run_metadata.json          Run configuration and environment info.
    ├-- tables/                    LaTeX .tex files (from benchmark/tables.py).
    ├-- figures/                   Saved figures (.pdf, .png).
    └-- solutions/                 Per-solve .npz archives.
        └-- <case_id>_<solver>_N<N>.npz

The schema is intentionally flat: all BenchmarkResult fields are stored
at the top level of each JSON record (CircuitMetrics is flattened with a
'circuit_' prefix). This allows the CSV to be opened directly in a
spreadsheet without nested parsing.

Backward compatibility
----------------------
The reader functions accept both the new BenchmarkResult schema (this
module) and the legacy schema from earlier runner versions. Legacy fields
are mapped to their new equivalents where possible; unmapped fields are
silently ignored.

References
----------
  Ghafourpour & Laizet (2025) Phys. Rev. Applied 24, 024032.
"""

from __future__ import annotations

import csv
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np

from benchmark.metrics import BenchmarkResult, CircuitMetrics
from benchmark.equal_accuracy import EqualAccuracyResult
from benchmark.sensitivity import SensitivitySweepResult


# -- Legacy field aliases ------------------------------------------------------
# The following mapping reconciles field names introduced in earlier runner
# versions with their current equivalents. Applied during deserialisation to
# maintain backward compatibility with pre-existing archives.

_LEGACY_ALIASES: dict[str, str] = {
    "max_rel_err":           "max_rel_err_vs_exact",
    "max_abs_err":           "max_abs_err_vs_exact",
    "wall_time":             "wall_time_s",
    "converged":             "vqls_converged",
    "final_cost":            "vqls_cost_final",
    "degree":                "qsvt_polynomial_degree",
    "circuit_depth":         "circuit_depth_opt1",   # flattened
    "n_qubits":              "circuit_n_qubits",     # flattened
    "epsilon":               "hhl_epsilon",
    "trotter_steps":         "hhl_trotter_steps",
    "n_layers":              "vqls_n_layers",
    "n_restarts":            "vqls_n_restarts",
}


# -- BenchmarkResult serialisation ---------------------------------------------

def _benchmark_result_to_dict(result: BenchmarkResult) -> dict:
    """Serialise a BenchmarkResult to a JSON-compatible flat dictionary."""
    return result.to_dict()


def _dict_to_benchmark_result(d: dict) -> BenchmarkResult:
    """
    Deserialise a dictionary to a BenchmarkResult.

    Handles both the current schema and legacy schemas from earlier runner
    versions via _LEGACY_ALIASES.
    """
    # Resolve legacy field aliases to their current equivalents.
    for old, new in _LEGACY_ALIASES.items():
        if old in d and new not in d:
            d[new] = d.pop(old)

    # Reconstruct the CircuitMetrics object from the flattened circuit_* columns.
    cm = None
    circuit_fields = {
        k[len("circuit_"):]: v
        for k, v in d.items()
        if k.startswith("circuit_") and v is not None
    }
    if circuit_fields:
        try:
            cm = CircuitMetrics(**circuit_fields)
        except TypeError:
            cm = None

    # Strip the flattened circuit_* keys from the working dictionary; the
    # reconstructed CircuitMetrics object is injected below.
    d_clean = {k: v for k, v in d.items() if not k.startswith("circuit_")}

    # Populate absent optional fields with None to satisfy the dataclass
    # constructor.
    all_fields = BenchmarkResult.__dataclass_fields__
    for fname in all_fields:
        if fname not in d_clean:
            d_clean[fname] = None

    # Assign sentinel values to required non-optional fields that are absent,
    # preventing constructor rejection of malformed records.
    for fname in ("case_id", "solver", "N", "discretisation_order",
                  "kappa", "source_fn", "alpha_bc", "beta_bc",
                  "residual", "max_rel_err_vs_thomas",
                  "max_abs_err_vs_thomas", "wall_time_s"):
        if d_clean.get(fname) is None:
            d_clean[fname] = 0 if fname in ("N", "discretisation_order") else (
                "" if fname in ("case_id", "solver", "source_fn") else 0.0
            )

    # Reconstructed from the flattened circuit_* columns above, so any value
    # already carried under this key is stale and is replaced rather than merged.
    # The assignment previously stood immediately BEFORE a pop of the same key,
    # which removed it again and left the constructor without a required
    # argument -- every read of an equal-accuracy or sensitivity archive raised.
    d_clean["circuit_metrics"] = cm

    try:
        result = BenchmarkResult(**{
            k: v for k, v in d_clean.items()
            if k in all_fields
        })
        result.circuit_metrics = cm
        return result
    except Exception as exc:
        raise ValueError(
            f"Failed to deserialise BenchmarkResult from dict: {exc}\n"
            f"Dict keys: {list(d_clean.keys())}"
        ) from exc


# -- SweepArchive --------------------------------------------------------------

class SweepArchive:
    """
    Manages the on-disk archive for a complete benchmark sweep.

    Provides methods to write and read all result types (primary,
    equal-accuracy, sensitivity) and to check archive completeness.

    Parameters
    ----------
    root : Path or str
        Root directory for this sweep's output.
    run_tag : str
        Short identifier for this run (e.g. '1D_primary_2026-08-10').
    """

    def __init__(self, root: Path | str, run_tag: str = "") -> None:
        self.root = Path(root)
        self.run_tag = run_tag
        self.solutions_dir = self.root / "solutions"
        self.tables_dir    = self.root / "tables"
        self.figures_dir   = self.root / "figures"

        for d in (self.root, self.solutions_dir,
                  self.tables_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)

    # -- Primary results -------------------------------------------------------

    def write_primary(self, results: list[BenchmarkResult]) -> None:
        """Write primary BenchmarkResult list to JSON and CSV."""
        # Serialise to JSON.
        json_path = self.root / "results_full.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                [_benchmark_result_to_dict(r) for r in results],
                f, indent=2, default=str,
            )

        # Write the flat CSV representation.
        csv_path = self.root / "results_summary.csv"
        if results:
            rows = [_benchmark_result_to_dict(r) for r in results]
            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

    def read_primary(self) -> list[BenchmarkResult]:
        """Read primary results from JSON. Returns empty list if file absent."""
        json_path = self.root / "results_full.json"
        if not json_path.exists():
            return []
        with open(json_path, encoding="utf-8") as f:
            raw = json.load(f)
        return [_dict_to_benchmark_result(d) for d in raw]

    def append_primary(self, new_results: list[BenchmarkResult]) -> None:
        """
        Append new results to the existing primary JSON, avoiding duplicates.

        A result is considered a duplicate if it has the same
        (case_id, solver, N, discretisation_order, sensitivity_param,
        sensitivity_value) as an existing entry.
        """
        existing = self.read_primary()
        existing_keys = {
            (r.case_id, r.solver, r.N, r.discretisation_order,
             r.sensitivity_param, r.sensitivity_value)
            for r in existing
        }
        to_add = [
            r for r in new_results
            if (r.case_id, r.solver, r.N, r.discretisation_order,
                r.sensitivity_param, r.sensitivity_value)
            not in existing_keys
        ]
        self.write_primary(existing + to_add)

    # -- Solution archives -----------------------------------------------------

    def write_solution(
        self,
        case_id: str,
        solver: str,
        N: int,
        x: np.ndarray,
        u_solver: np.ndarray,
        u_exact: Optional[np.ndarray] = None,
        u_thomas: Optional[np.ndarray] = None,
        discretisation_order: int = 2,
    ) -> Path:
        """
        Save solution vectors to a compressed NPZ archive.

        Parameters
        ----------
        x : np.ndarray, shape (N,)
            Interior node coordinates.
        u_solver : np.ndarray, shape (N,)
            Solver solution vector.
        u_exact : np.ndarray or None
            Analytical solution, if available.
        u_thomas : np.ndarray or None
            Thomas reference solution, if available.

        Returns
        -------
        Path
            Path of the written .npz file.
        """
        fname = (
            self.solutions_dir
            / f"{case_id}_{solver}_N{N}_ord{discretisation_order}.npz"
        )
        arrays: dict[str, np.ndarray] = {"x": x, "u_solver": u_solver}
        if u_exact is not None:
            arrays["u_exact"] = u_exact
        if u_thomas is not None:
            arrays["u_thomas"] = u_thomas
        np.savez_compressed(fname, **arrays)
        return fname

    def read_solution(
        self,
        case_id: str,
        solver: str,
        N: int,
        discretisation_order: int = 2,
    ) -> Optional[dict[str, np.ndarray]]:
        """
        Load a solution archive. Returns None if the file does not exist.
        """
        fname = (
            self.solutions_dir
            / f"{case_id}_{solver}_N{N}_ord{discretisation_order}.npz"
        )
        if not fname.exists():
            # Try legacy filename convention (no _ord suffix)
            fname_legacy = (
                self.solutions_dir / f"{case_id}_{solver}_N{N}.npz"
            )
            if fname_legacy.exists():
                fname = fname_legacy
            else:
                return None
        data = np.load(fname, allow_pickle=False)
        return dict(data)

    # -- Equal-accuracy results ------------------------------------------------

    def write_equal_accuracy(self, ea_results: list[EqualAccuracyResult]) -> None:
        """Write equal-accuracy results to JSON, overwriting existing."""
        path = self.root / "equal_accuracy.json"
        records = []
        for ear in ea_results:
            records.append({
                "solver":             ear.solver,
                "r_target":           ear.r_target,
                "band_factor":        ear.band_factor,
                "in_band":            ear.in_band,
                "n_solver_calls":     ear.n_solver_calls,
                "total_sweep_time_s": ear.total_sweep_time_s,
                "notes":              ear.notes,
                "best_result":        _benchmark_result_to_dict(ear.best_result),
                "all_results":        [
                    _benchmark_result_to_dict(r) for r in ear.all_results
                ],
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)

    def append_equal_accuracy(self, new_results: list[EqualAccuracyResult]) -> None:
        """
        Append new equal-accuracy results, replacing existing ones with the same
        (case_id, solver, N).
        """
        existing = self.read_equal_accuracy()
        # Find which case/solver combos are in new_results
        new_keys = {
            (r.best_result.case_id, r.solver, r.best_result.N)
            for r in new_results
        }
        # Keep existing results that are not being replaced
        to_keep = [
            r for r in existing
            if (r.best_result.case_id, r.solver, r.best_result.N) not in new_keys
        ]
        self.write_equal_accuracy(to_keep + new_results)

    def read_equal_accuracy(self) -> list[EqualAccuracyResult]:
        """Read equal-accuracy results from JSON."""
        path = self.root / "equal_accuracy.json"
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        results = []
        for rec in raw:
            best = _dict_to_benchmark_result(rec["best_result"])
            all_r = [_dict_to_benchmark_result(d) for d in rec["all_results"]]
            results.append(EqualAccuracyResult(
                solver=rec["solver"],
                r_target=rec["r_target"],
                band_factor=rec["band_factor"],
                in_band=rec["in_band"],
                best_result=best,
                all_results=all_r,
                n_solver_calls=rec["n_solver_calls"],
                total_sweep_time_s=rec["total_sweep_time_s"],
                notes=rec.get("notes", ""),
            ))
        return results

    # -- Sensitivity results ---------------------------------------------------

    def write_sensitivity(
        self,
        solver: str,
        sweeps: list[SensitivitySweepResult],
    ) -> None:
        """Write sensitivity sweep results to JSON for one solver."""
        path = self.root / f"sensitivity_{solver}.json"
        records = []
        for sweep in sweeps:
            records.append({
                "solver":             sweep.solver,
                "param_name":         sweep.param_name,
                "param_values":       sweep.param_values,
                "baseline_config":    sweep.baseline_config,
                "n_solver_calls":     sweep.n_solver_calls,
                "total_sweep_time_s": sweep.total_sweep_time_s,
                "results":            [
                    _benchmark_result_to_dict(r) for r in sweep.results
                ],
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, default=str)

    def read_sensitivity(self, solver: str) -> list[SensitivitySweepResult]:
        """Read sensitivity results for one solver from JSON."""
        path = self.root / f"sensitivity_{solver}.json"
        if not path.exists():
            return []
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        sweeps = []
        for rec in raw:
            results = [_dict_to_benchmark_result(d) for d in rec["results"]]
            sweeps.append(SensitivitySweepResult(
                solver=rec["solver"],
                param_name=rec["param_name"],
                param_values=rec["param_values"],
                results=results,
                baseline_config=rec.get("baseline_config", {}),
                n_solver_calls=rec["n_solver_calls"],
                total_sweep_time_s=rec["total_sweep_time_s"],
            ))
        return sweeps

    # -- Run metadata ----------------------------------------------------------

    def write_metadata(self, config: dict) -> None:
        """
        Write run metadata (configuration, environment) to JSON.

        Parameters
        ----------
        config : dict
            Run configuration dictionary. Merged with environment info.
        """
        meta = {
            "run_tag":         self.run_tag,
            "timestamp":       time.strftime("%Y-%m-%dT%H:%M:%S"),
            "python_version":  sys.version,
            "platform":        platform.platform(),
            "config":          config,
        }
        path = self.root / "run_metadata.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

    # -- Archive completeness --------------------------------------------------

    def missing(
        self,
        expected_cases: list[str],
        expected_solvers: list[str],
        expected_N: list[int],
    ) -> list[tuple[str, str, int]]:
        """
        Report (case_id, solver, N) combinations present in the primary
        JSON but whose solution archive (.npz) is absent.

        Parameters
        ----------
        expected_cases : list[str]
            Case identifiers expected in the sweep.
        expected_solvers : list[str]
            Solver names expected.
        expected_N : list[int]
            Problem sizes expected.

        Returns
        -------
        list[tuple[str, str, int]]
            Missing (case_id, solver, N) combinations.
        """
        existing = self.read_primary()
        existing_keys = {
            (r.case_id, r.solver, r.N) for r in existing
        }
        missing = []
        for case_id in expected_cases:
            for solver in expected_solvers:
                for N in expected_N:
                    if (case_id, solver, N) not in existing_keys:
                        missing.append((case_id, solver, N))
        return missing