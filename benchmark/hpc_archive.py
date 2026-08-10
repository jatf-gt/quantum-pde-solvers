"""
On-disk schema for a completed HPC sweep: the contract between the runners that
write results and the post-processing that reads them.

Relationship to ``benchmark/results_io.py``
-------------------------------------------
The package now carries **two** archive abstractions, and they are not
interchangeable. Both expose a class called ``SweepArchive``; they are told apart
by module, and the wrong one will fail loudly on its constructor rather than
silently mis-read a sweep.

===============  =========================================================
This module      The *legacy* schema: the existing per-dimension sweep
``hpc_archive``  directories ``results/{1,2,3}Dhpc_run/``, produced by
                 ``hpc/runners/run_{1,2,3}d.py``. Read-only, dimension-aware
                 (``SweepArchive(dir, dim=…)``), row-oriented (plain dicts),
                 and tolerant of the three historical spellings of the
                 solution field through ``FIELD_ALIASES``.

``results_io``   The *publication* archive introduced with the professional
                 benchmarking framework: ``results/<run_tag>/`` with
                 ``solutions/``, ``tables/`` and ``figures/`` subdirectories.
                 Read-write, run-tag-oriented
                 (``SweepArchive(root, run_tag=…)``), and typed — it stores
                 ``BenchmarkResult`` objects rather than dicts.
===============  =========================================================

This module must keep working unchanged. It is what
``scripts/gap_analysis.py`` reads to build the rerun manifests, and those
manifests are the only thing standing between a resubmission and the
recomputation of individual rows costing 25–38 hours of cluster time. It is also
what ``benchmark/hpc_plotting.py`` reads to regenerate every published HPC
figure. The archives it describes already exist on disk and cannot be rewritten,
so its contract is fixed by data rather than by preference.

New work should target ``results_io``; this module is retained because the
sweeps it describes are already paid for.

Why this module exists
----------------------
The schema was previously defined nowhere. It was triplicated on the write side
— three result dataclasses, three solution writers, three metadata writers — and
re-derived on the read side from bare string literals scattered across
seventeen plotting functions. The two halves were coupled by nothing but those
literals, with three consequences:

*   The same quantity carries three names. The 1D driver writes ``u_solver``,
    the 2D driver ``phi_solver`` and the 3D driver ``phi``. The 2D writer papers
    over this with an explicit alias so that the 1D-era reader keeps working;
    the alias, rather than any declaration, is the contract.
*   A filename change breaks the reader silently. Every caller treats a missing
    archive as "not run", so a renamed file yields an empty figure set rather
    than an error.
*   ``run_metadata.json`` is written by all three drivers and read by none.

This module is the single declaration. Reading goes through `SweepArchive`;
field names go through `FIELD_ALIASES` rather than being spelled at the point of
use; and `SweepArchive.missing` distinguishes "not run" from "not found",
which is what makes a silently empty figure set impossible.

Scope
-----
Reader side only, for now. The functions here are written so that the HPC
runners can adopt them for writing — `solution_filename` is the same convention
they build by hand — but they are deliberately not yet wired into those drivers,
because 2D and 3D sweeps are in flight on the cluster and changing what they
write would strand the results already produced. That migration belongs with the
move of the runners into `hpc/`.

This module must import no plotting stack. `benchmark/hpc_plotting.py` defers its
Matplotlib import specifically so that the Agg backend is not forced on a process
that wants an interactive one, and listing a sweep's contents must not require a
plotting stack to be installed at all.

Schema summary
--------------
Each sweep directory contains:

    results_full.json     One record per (case, solver, N). Ten fields are
                          common to all three dimensions; the rest are
                          dimension-specific — see `COMMON_ROW_FIELDS`.
    results_summary.csv   The same rows, flattened. Written but never read.
    <prefix>_{case}_{solver}_N{N}.npz
                          One archive per solution, written as it is produced,
                          so a walltime kill loses the summary but not the data.
    run_metadata.json     Environment and configuration provenance.

The 1D driver additionally writes ``all_solutions.npz``, a consolidated archive
that nothing reads.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np


# ── Schema Constants ──────────────────────────────────────────────────────────

SOLVER_ORDER: tuple[str, ...] = ("Thomas", "HHL", "VQLS", "QSVT")
"""Canonical solver ordering. Fixes column order in every field figure, so the
same solver occupies the same position throughout a sweep. Unrecognised labels
sort last rather than raising: a sweep may legitimately introduce a solver this
module has not been told about."""

SOLUTION_PREFIX: dict[int, str] = {1: "solutions", 2: "solutions", 3: "solution3d"}
"""Filename stem of the solution archives, per dimension. 1D and 2D share a
stem, so nothing but the containing directory distinguishes their archives."""

COMMON_ROW_FIELDS: frozenset[str] = frozenset({
    "case", "solver", "N", "max_rel_err", "max_abs_err", "residual",
    "wall_time_s", "converged", "notes", "rel_l2_err", "rms_err",
})
"""The eleven summary fields every dimension writes. Everything else is
dimension-specific: 1D alone carries circuit metrics (`n_qubits`,
`circuit_depth`, `success_probability`); 2D and 3D alone carry outer-iteration
metrics (`scheme`, `n_outer`, `weighted_cost`, `stop_reason`, `linf_err`,
`err_vs_thomas`). Reading a dimension-specific field from the wrong dimension is
a KeyError, which is why `row_field` exists."""

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "solution": ("u_solver", "phi_solver", "phi"),
    "exact":    ("u_exact", "phi_exact"),
    "source":   ("f_vals", "f"),
}
"""
Names under which the same physical quantity is archived, in preference order.

`solution` is the computed field: 1D writes ``u_solver``, 2D ``phi_solver`` and
3D ``phi``. The 2D driver also writes a ``u_solver`` alias for backward
compatibility, so both spellings appear in a 2D archive and the preference order
decides. `field` resolves these; no caller should spell them directly.
"""


# ── Filename Convention ───────────────────────────────────────────────────────

def solution_filename(case: str, solver: str, N: int, dim: int = 1) -> str:
    """
    Builds the archive filename for one solution.

    This is the convention the three runners currently construct independently.
    Routing both sides through this function is what makes a rename a loud
    failure rather than a silently empty figure set.

    Parameters
    ----------
    case : str
        Case identifier as recorded in the summary.
    solver : str
        Solver label.
    N : int
        Resolution.
    dim : int
        Spatial dimension, selecting the filename stem.

    Returns
    -------
    str
        Filename including the ``.npz`` extension.

    Raises
    ------
    KeyError
        If `dim` is not 1, 2 or 3.
    """
    return f"{SOLUTION_PREFIX[dim]}_{case}_{solver}_N{N}.npz"


def field(data: dict, kind: str) -> Optional[np.ndarray]:
    """
    Resolves an aliased field from a loaded archive.

    Parameters
    ----------
    data : dict
        Archive contents as returned by `SweepArchive.solution`.
    kind : str
        Logical field name, a key of `FIELD_ALIASES`: ``"solution"``,
        ``"exact"`` or ``"source"``.

    Returns
    -------
    np.ndarray or None
        The first alias present in the archive, or None if none is.

    Raises
    ------
    KeyError
        If `kind` is not a recognised logical field. Unknown field names are
        rejected rather than returning None, since silently absorbing a typo
        would present as "this solution has no solution field".
    """
    if kind not in FIELD_ALIASES:
        raise KeyError(
            f"Unknown field '{kind}'. Valid: {sorted(FIELD_ALIASES)}."
        )
    for name in FIELD_ALIASES[kind]:
        if name in data:
            return data[name]
    return None


def row_field(row: dict, name: str, default: Any = None) -> Any:
    """
    Reads a summary field, tolerating its absence in other dimensions.

    Parameters
    ----------
    row : dict
        One record from `results_full.json`.
    name : str
        Field name.
    default : Any
        Value returned when the field is absent.

    Returns
    -------
    Any
        The field value, or `default`.

    Raises
    ------
    KeyError
        If `name` is one of `COMMON_ROW_FIELDS` and is nonetheless missing.
        A common field absent from a row means the file is malformed, which
        should not be papered over with a default.
    """
    if name in COMMON_ROW_FIELDS and name not in row:
        raise KeyError(
            f"Row is missing the common field '{name}'; the summary is "
            f"malformed. Row keys: {sorted(row)}."
        )
    return row.get(name, default)


# ── Sweep Archive ─────────────────────────────────────────────────────────────

class SweepArchive:
    """
    Reader for the output directory of a single HPC sweep.

    Holds the three things that differ between the 1D, 2D and 3D drivers — the
    results directory, the naming convention of the archived solutions, and
    where figures are written — so that the post-processing above is written
    once and works for all three.

    Attributes
    ----------
    results_dir : Path
        Directory containing `results_full.json` and the per-solution `.npz`
        archives.
    dim : int
        Spatial dimension of the sweep, selecting the archive filename stem.
    solution_prefix : str
        Filename stem of the solution archives, derived from `dim` unless
        overridden.
    plots_dir : Path
        Destination for figures. The 2D and 3D drivers use a `plots/`
        subdirectory; the 1D driver writes alongside its results, and its
        figures are referenced in written work under those names.
    skip_scheme_comparison : bool
        Whether to exclude rows tagged `scheme_comparison` from the grouped
        views. Those rows belong to the `--compare-schemes` study and would
        otherwise appear as spurious duplicate solvers in the vs-N plots.
    """

    def __init__(
        self,
        results_dir:            Path,
        dim:                    int = 1,
        solution_prefix:        str | None = None,
        plots_subdir:           str | None = "plots",
        skip_scheme_comparison: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        results_dir : Path
            Sweep output directory.
        dim : int
            Spatial dimension, 1, 2 or 3.
        solution_prefix : str, optional
            Overrides the stem implied by `dim`. Present for sweeps written
            before the convention settled.
        plots_subdir : str or None
            Subdirectory for figures; None writes alongside the results.
        skip_scheme_comparison : bool
            Exclude `scheme_comparison` rows from the grouped views.
        """
        self.results_dir = Path(results_dir)
        self.dim = int(dim)
        self.solution_prefix = (solution_prefix if solution_prefix is not None
                                else SOLUTION_PREFIX[self.dim])
        self.plots_dir = (self.results_dir / plots_subdir if plots_subdir
                          else self.results_dir)
        self.skip_scheme_comparison = skip_scheme_comparison

    # ── Loading ───────────────────────────────────────────────────────────────

    def rows(self) -> list[dict]:
        """
        Reads the sweep summary.

        Returns
        -------
        list of dict
            One record per (case, solver, N) combination.

        Raises
        ------
        SystemExit
            If the summary is absent. A walltime-killed job writes its
            per-solution archives but never its summary, so this is a routine
            outcome rather than an error worth a traceback.
        """
        path = self.results_dir / "results_full.json"
        if not path.exists():
            raise SystemExit(
                f"No results found at {path}. Run the corresponding HPC driver "
                f"first, or point --results-dir at a completed sweep."
            )
        with open(path) as fh:
            return json.load(fh)

    def solution(self, case: str, solver: str, N: int) -> dict | None:
        """
        Loads one archived solution.

        Parameters
        ----------
        case : str
            Case identifier as recorded in the summary.
        solver : str
            Solver label.
        N : int
            Resolution.

        Returns
        -------
        dict or None
            Every array in the archive, keyed by name, or None if the
            combination was not run.
        """
        path = self.results_dir / f"{self.solution_prefix}_{case}_{solver}_N{N}.npz"
        if not path.exists():
            return None
        with np.load(path) as d:
            return {k: d[k] for k in d.files}

    def metadata(self) -> dict:
        """
        Reads the run provenance record.

        Returns
        -------
        dict
            Environment and configuration as recorded by the driver, or an empty
            dict if absent. Absence is not an error: the record is provenance,
            not data, and a sweep remains fully interpretable without it.
        """
        path = self.results_dir / "run_metadata.json"
        if not path.exists():
            return {}
        with open(path) as fh:
            return json.load(fh)

    def missing(self, rows: list[dict]) -> list[tuple[str, str, int]]:
        """
        Reports summary rows whose solution archive is absent.

        This is the check that distinguishes a partial sweep from a broken
        filename convention. Every plotting caller treats a missing archive as
        "not run" and simply omits it, so without this a renamed or relocated
        archive presents as a quietly incomplete figure set.

        Parameters
        ----------
        rows : list of dict
            Summary records, from `rows`.

        Returns
        -------
        list of tuple
            (case, solver, N) for each row with no archive on disk, in summary
            order.
        """
        gaps = []
        for r in rows:
            case, solver, N = r["case"], r["solver"], r["N"]
            if self.solution(case, solver, N) is None:
                gaps.append((case, solver, int(N)))
        return gaps

    # ── Grouping ──────────────────────────────────────────────────────────────

    def _keep(self, row: dict) -> bool:
        """Whether a summary row belongs in the grouped views."""
        if self.skip_scheme_comparison:
            return not row.get("notes", "").startswith("scheme_comparison")
        return True

    def group_by_case_N(self, rows: list[dict]) -> dict[tuple, list[dict]]:
        """
        Groups as {(case, N): [row, …]}, solvers in canonical order.

        The ordering matters for the field plots: it fixes the column order so
        the same solver occupies the same position in every figure of a sweep.

        Parameters
        ----------
        rows : list of dict
            Summary records.

        Returns
        -------
        dict
            Keyed by (case, N).
        """
        groups: dict[tuple, list[dict]] = {}
        for r in rows:
            if not self._keep(r):
                continue
            groups.setdefault((r["case"], r["N"]), []).append(r)
        for key in groups:
            groups[key].sort(key=lambda r: solver_sort_key(r["solver"]))
        return groups

    def group_by_case_solver(self, rows: list[dict]) -> dict[tuple, list[dict]]:
        """
        Groups as {(case, solver): [row, …]} sorted by N, for vs-N plots.

        Parameters
        ----------
        rows : list of dict
            Summary records.

        Returns
        -------
        dict
            Keyed by (case, solver).
        """
        groups: dict[tuple, list[dict]] = {}
        for r in rows:
            if not self._keep(r):
                continue
            groups.setdefault((r["case"], r["solver"]), []).append(r)
        for key in groups:
            groups[key].sort(key=lambda r: r["N"])
        return groups

    def group_nested(self, rows: list[dict]) -> dict:
        """
        Groups as {case: {solver: [row, …]}} sorted by N.

        The nested form the 1D plots are written against, iterating cases as
        figures and solvers as curves within them.

        Parameters
        ----------
        rows : list of dict
            Summary records.

        Returns
        -------
        dict
            Nested by case then solver.
        """
        grouped: dict = {}
        for r in rows:
            grouped.setdefault(r["case"], {}).setdefault(r["solver"], []).append(r)
        for case in grouped:
            for solver in grouped[case]:
                grouped[case][solver].sort(key=lambda x: x["N"])
        return grouped

    def series(
        self,
        grouped: dict[tuple, list[dict]],
        case:    str,
        exclude: tuple[str, ...] = (),
    ) -> Iterator[tuple[str, list[dict]]]:
        """
        Iterates (solver, rows) for one case from a case-solver grouping.

        Every vs-N plot repeated the same inner filter — loop the whole
        grouping, skip keys whose case does not match — in four places.

        Solvers are yielded in **lexical** order, not the canonical
        `SOLVER_ORDER`. That is deliberate: the call sites this replaces
        iterate `sorted(grouped.items())`, which sorts by the (case, solver)
        key and therefore alphabetically. Imposing the canonical order here
        would reorder every legend and line in the existing 2-D and 3-D
        figures — a restyling, not a refactor. Use `group_by_case_N`, which
        does sort canonically, where column order matters.

        Parameters
        ----------
        grouped : dict
            Output of `group_by_case_solver`.
        case : str
            Case to select.
        exclude : tuple of str
            Solver labels to skip, e.g. the classical reference in a plot that
            measures everything relative to it.

        Yields
        ------
        tuple
            (solver, rows), lexically ordered by solver.
        """
        for (c, solver), rs in sorted(grouped.items()):
            if c == case and solver not in exclude:
                yield solver, rs


def solver_sort_key(s: str) -> tuple:
    """
    Orders solvers canonically, with unrecognised labels last.

    Parameters
    ----------
    s : str
        Solver label.

    Returns
    -------
    tuple
        Sort key.
    """
    return (SOLVER_ORDER.index(s) if s in SOLVER_ORDER else 99, s)


# ── Writing ───────────────────────────────────────────────────────────────────
#
# Provided so the HPC runners can adopt a single declaration of the schema they
# currently each construct by hand. Not yet wired into those drivers: 2D and 3D
# sweeps are in flight, and changing what they write would strand results
# already produced. See the module docstring.

def save_summary(results_dir: Path, rows: list[dict]) -> None:
    """
    Writes `results_full.json` and `results_summary.csv`.

    Parameters
    ----------
    results_dir : Path
        Sweep output directory; created if absent.
    rows : list of dict
        Result records, one per (case, solver, N).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "results_full.json", "w") as fh:
        json.dump(rows, fh, indent=2, default=str)

    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(results_dir / "results_summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_solution(
    results_dir: Path,
    case:        str,
    solver:      str,
    N:           int,
    dim:         int,
    **arrays:    np.ndarray,
) -> Path:
    """
    Writes one solution archive.

    Parameters
    ----------
    results_dir : Path
        Sweep output directory; created if absent.
    case, solver : str
        Identifiers.
    N : int
        Resolution.
    dim : int
        Spatial dimension, selecting the filename stem.
    **arrays
        Arrays to archive. Field names should be the first entry of the
        corresponding `FIELD_ALIASES` tuple for the dimension in question.

    Returns
    -------
    Path
        Path written.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / solution_filename(case, solver, N, dim)
    np.savez_compressed(path, **arrays)
    return path
