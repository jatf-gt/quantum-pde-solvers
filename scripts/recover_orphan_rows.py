#!/usr/bin/env python3
"""
Reconstructs summary rows for solution archives whose rows were lost.

Why this exists
---------------
A sweep writes each solution to its own ``.npz`` as soon as it is produced, but a
job killed mid-work-unit returns nothing to the parent, so the corresponding row
never reaches ``results_full.json``. The field survives; the record of it does not.
`scripts/gap_analysis.py` reports the discrepancy as an *unexplained orphan* and
would otherwise schedule the combination for recomputation.

Two such orphans exist in the 3-D archive, both HHL at N=16 and each roughly six
hours of statevector simulation:

    solution3d_3D_Laplace_BCdriven_cube_HHL_N16.npz
    solution3d_3D_Poisson_TripleSin_cube_HHL_N16.npz

Every accuracy metric in the schema is a function of the stored fields (``phi``,
``phi_exact``, the source and the grid) together with the case definition, which is
reproducible from `core.cases`. Recovering them therefore costs no quantum
simulation at all, against roughly twelve hours to reproduce them.

What is and is not recoverable
------------------------------
Recoverable, because it is a function of the stored field:

    max_rel_err, max_abs_err, rel_l2_err, rms_err, linf_err, residual,
    err_vs_thomas, err_thomas_vs_exact, peak_E_field, peak_E_axial,
    peak_E_rel_err, azimuthal_mode_amp, azimuthal_mode_rel_err, kappa_row,
    shape, n_unknowns, anisotropy, n_outer, convergence_factor

**Not** recoverable, because it existed only in the killed process's memory:

    wall_time_s, strip_solves, strip_solves_by_size, weighted_cost,
    mean_strip_size, s_per_strip_solve, solves_per_digit, inner_calls,
    inner_failures, inner_total_s, inner_mean_s, inner_max_s, hhl_scale_c,
    level_shapes, level_kappas, n_levels, stop_reason

Those are left as ``None`` rather than zero. A zero would read as "this row was
free", which is the opposite of the truth and would corrupt any cost comparison
that aggregates the column. Every recovered row additionally carries
``notes="recovered_from_archive"``, so a reader — human or the plotting layer —
can always tell an inferred row from an instrumented one.

The residual is recomputed from the stored field against the freshly assembled
operator, rather than read from ``residual_history``, so that it is a property of
the solution rather than a value inherited from a process that did not survive to
report it. ``residual_history`` is used only for ``n_outer`` and the convergence
factor, both of which it determines exactly.

Usage
-----
    python scripts/recover_orphan_rows.py --dim 3 --dry-run
    python scripts/recover_orphan_rows.py --dim 3

``--dry-run`` prints the rows it would add and touches nothing. Without it, the
rows are merged into ``results_full.json`` under the same supersession rule the
runners use — an existing row for the same (case, solver, N) is left alone unless
``--force`` is given, since an instrumented row is always preferable to an
inferred one.

Date   : August 2026
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ``pytest.ini`` sets ``pythonpath = .``, but a bare ``python3 scripts/...`` puts
# ``scripts/`` on ``sys.path[0]`` rather than the repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from solvers.outer.core import OuterResult, WorkLog     # noqa: E402
from scripts.gap_analysis import STALE_GEOMETRY_CASES   # noqa: E402


# -- Archive naming ------------------------------------------------------------

ARCHIVE_PREFIX: dict[int, str] = {2: "solutions_", 3: "solution3d_"}
"""
Filename prefix per dimension, as written by the runners.

Declared here rather than inferred, because the two schemas differ and a wrong
guess would silently find nothing. `benchmark/results_io.py` is the authority on
the convention; this mirrors only the part needed to parse a name back into a
(case, solver, N) triple.
"""

RESULTS_DIR: dict[int, str] = {2: "results/2Dhpc_run", 3: "results/3Dhpc_run"}


def parse_archive_name(path: Path, dim: int) -> Optional[tuple[str, str, int]]:
    """
    Recover the (case, solver, N) triple from an archive filename.

    Parameters
    ----------
    path : Path
        Archive path, e.g. ``solution3d_3D_Poisson_TripleSin_cube_HHL_N16.npz``.
    dim : int
        Sweep dimension, selecting the expected filename prefix.

    Returns
    -------
    tuple of (str, str, int), optional
        (case identifier, solver label, N), or None if the name does not match the
        convention — which is how a stray file in the directory is skipped rather
        than misparsed.
    """
    prefix = ARCHIVE_PREFIX[dim]
    stem = path.stem
    if not stem.startswith(prefix):
        return None
    body = stem[len(prefix):]
    parts = body.rsplit("_", 2)
    if len(parts) != 3 or not parts[2].startswith("N"):
        return None
    case, solver, n_token = parts
    try:
        return case, solver, int(n_token[1:])
    except ValueError:
        return None


# -- Case lookup ---------------------------------------------------------------

def build_case_index(runner, N: int) -> dict:
    """
    Assemble every case the runner defines at resolution `N`, indexed by its identifier.

    The runners record a display identifier (``3D_Poisson_TripleSin_cube``) that is
    not the `core.cases` registry key, and no mapping between the two is stored in
    the results. Rather than guess at one by string similarity — which mismatches
    silently — this calls the runner's own section builders, each of which returns
    the identifier alongside the assembled problem. The runner is thus the single
    authority on what a given identifier means, exactly as it is during a sweep.

    Assembly only; no solve is performed, so the cost is negligible.

    Parameters
    ----------
    runner : module
        `hpc.runners.run_2d` or `hpc.runners.run_3d`.
    N : int
        Resolution at which to assemble.

    Returns
    -------
    dict
        ``{case_id: (problem, phi_ref, f_values, mode_m)}`` for every section that
        assembles successfully. A section that raises is omitted with a warning
        rather than aborting the recovery of the others.
    """
    index: dict = {}
    for section, builder in runner.SECTIONS.items():
        try:
            prob, phi_ref, f_vals, case_id, mode_m = builder(N)
        except Exception as exc:              # noqa: BLE001 - reported, not raised
            print(f"    (section {section} at N={N} did not assemble: "
                  f"{type(exc).__name__}: {exc})")
            continue
        index[case_id] = (prob, phi_ref, f_vals, mode_m)
    return index


# -- Row reconstruction --------------------------------------------------------

def synthesise_outer_result(archive: dict, scheme: str = "") -> OuterResult:
    """
    Wrap a stored field in an `OuterResult` so the runner's own recorder can be reused.

    Reusing `run_3d._record` rather than recomputing the metrics here is deliberate:
    the accuracy definitions (which norm, measured against what, in which units)
    must match the instrumented rows exactly, or the recovered rows would not be
    comparable with the ones beside them. Reimplementing them invites precisely
    that divergence.

    The `WorkLog` is left empty and `wall_time_s` at zero; the caller overwrites the
    resulting fields with None, since neither quantity was recorded.

    Parameters
    ----------
    archive : dict
        Arrays read from the ``.npz``.
    scheme : str
        Outer scheme name, if known; empty when the archive does not record it.

    Returns
    -------
    OuterResult
        Carrying the stored field, residual history, and nothing else.
    """
    hist = [float(v) for v in archive.get("residual_history", [])]
    return OuterResult(
        u=archive["phi"],
        scheme=scheme,
        inner="hhl",
        converged=False,
        n_outer=max(len(hist) - 1, 0),
        residual=hist[-1] if hist else float("nan"),
        residual_history=hist,
        work=WorkLog(),
        wall_time_s=0.0,
        stop_reason="",
        diagnostics={},
    )


UNRECOVERABLE_FIELDS: tuple[str, ...] = (
    "wall_time_s", "strip_solves", "strip_solves_by_size", "weighted_cost",
    "mean_strip_size", "s_per_strip_solve", "solves_per_digit", "inner_calls",
    "inner_failures", "inner_total_s", "inner_mean_s", "inner_max_s",
    "hhl_scale_c", "level_shapes", "level_kappas", "n_levels", "stop_reason",
    "inner_options", "vqls_final_cost", "qsvt_degree", "qsvt_depth",
    "n_circuit_evals",
)
"""
Fields nulled on a recovered row because they existed only in the lost process.

Set to None rather than left at their dataclass defaults: a ``strip_solves`` of 0
or a ``wall_time_s`` of 0.0 would read as a free solve and would silently skew any
aggregate over the column.
"""


def main() -> int:
    """
    Entry point.

    Returns
    -------
    int
        0 on success, 1 if any orphan could not be reconstructed.
    """
    ap = argparse.ArgumentParser(
        description="Rebuild summary rows for orphaned solution archives.")
    ap.add_argument("--dim", type=int, choices=[2, 3], required=True)
    ap.add_argument("--results-dir", default=None,
                    help="Sweep directory (default: per --dim).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be recovered; write nothing.")
    ap.add_argument("--force", action="store_true",
                    help="Replace an existing row. Off by default: an "
                         "instrumented row is always preferable to an inferred one.")
    args = ap.parse_args()

    results_dir = Path(args.results_dir or RESULTS_DIR[args.dim])
    summary_path = results_dir / "results_full.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} does not exist.")
        return 1

    with open(summary_path) as fh:
        rows = json.load(fh)
    recorded = {(r["case"], r["solver"], int(r["N"])) for r in rows}

    archives = sorted(results_dir.glob(f"{ARCHIVE_PREFIX[args.dim]}*.npz"))
    orphans, superseded = [], []
    for path in archives:
        triple = parse_archive_name(path, args.dim)
        if triple is None:
            continue
        if triple[0] in STALE_GEOMETRY_CASES:
            # Superseded residue, not a lost row. These archives predate the
            # SPT-100 correction, so their fields are wrong however complete they
            # look; the wave-1 rerun replaces them. Recovering a row from one would
            # manufacture a plausible record of a solve to the wrong problem.
            superseded.append(triple)
            continue
        if triple not in recorded or args.force:
            orphans.append((path, triple))

    if superseded:
        print(f"Ignoring {len(superseded)} archive(s) from geometry-superseded "
              f"cases; the wave-1 rerun replaces those, and their stored fields "
              f"solve the pre-correction problem.\n")

    if not orphans:
        print("No orphaned archives: every archive on disk has a summary row.")
        return 0

    print(f"Found {len(orphans)} archive(s) with no summary row:")
    for path, (case, solver, N) in orphans:
        print(f"  {case:34s} {solver:7s} N={N:<3d}  {path.name}")
    print()

    # Deferred: importing the runner pulls in the whole solver stack, and this
    # script's parsing and reporting paths must stay usable without it.
    if args.dim == 3:
        from hpc.runners import run_3d as runner
    else:
        from hpc.runners import run_2d as runner

    # One index per resolution, since assembling a case is N-dependent.
    indices: dict[int, dict] = {}

    recovered, failed = [], []
    for path, (case_id, solver, N) in orphans:
        if N not in indices:
            print(f"  Assembling the runner's cases at N={N} ...")
            indices[N] = build_case_index(runner, N)
        entry = indices[N].get(case_id)
        if entry is None:
            print(f"  SKIP {case_id}: no section of the runner produces this "
                  f"identifier at N={N}, so the operator cannot be reassembled "
                  f"and no metric is derivable.")
            failed.append((case_id, solver, N))
            continue
        print(f"  {case_id} {solver} N={N}: rebuilding ...")
        try:
            row = _recover_one(runner, path, case_id, solver, N, entry)
        except Exception as exc:              # noqa: BLE001 - reported, not raised
            print(f"    FAILED: {type(exc).__name__}: {exc}")
            failed.append((case_id, solver, N))
            continue
        recovered.append(row)
        print(f"    linf_err={row.get('linf_err')}  "
              f"residual={row.get('residual')}  n_outer={row.get('n_outer')}")

    if not recovered:
        print("\nNothing recovered.")
        return 1 if failed else 0

    if args.dry_run:
        print(f"\n--dry-run: {len(recovered)} row(s) would be added to "
              f"{summary_path}; nothing written.")
        return 1 if failed else 0

    # Merged under the same supersession rule the runners apply, so a summary that
    # already carries duplicates from an --append run predating that rule is
    # collapsed here too. Reported separately from the recovered count, because a
    # net row count that falls while rows are being added is otherwise alarming.
    keep = {(r["case"], r["solver"], int(r["N"])): r for r in rows}
    collapsed = len(rows) - len(keep)
    for row in recovered:
        keep[(row["case"], row["solver"], int(row["N"]))] = row
    merged = list(keep.values())
    with open(summary_path, "w") as fh:
        json.dump(merged, fh, indent=2, default=str)
    print(f"\n{summary_path}: {len(rows)} -> {len(merged)} rows "
          f"({len(recovered)} recovered, {collapsed} pre-existing duplicate(s) "
          f"on (case, solver, N) collapsed).")
    print("Recovered rows carry notes='recovered_from_archive' and null cost "
          "columns. Re-run scripts/gap_analysis.py to confirm the orphans clear.")
    return 1 if failed else 0


def _recover_one(runner, path: Path, case_id: str, solver: str, N: int,
                 entry: tuple) -> dict:
    """
    Rebuild one row from its archive.

    Parameters
    ----------
    runner : module
        `hpc.runners.run_2d` or `run_3d`, supplying `_record` so the metric
        definitions match the instrumented rows exactly.
    path : Path
        Archive to read.
    case_id : str
        Identifier to record.
    solver : str
        Solver label as recorded (e.g. ``"HHL"``).
    N : int
        Resolution.
    entry : tuple
        ``(problem, phi_ref, f_values, mode_m)`` as assembled by
        `build_case_index`, i.e. by the runner's own section builder.

    Returns
    -------
    dict
        The reconstructed row, with unrecoverable columns set to None.

    Raises
    ------
    KeyError
        If the archive lacks the solution field.
    """
    from dataclasses import asdict

    with np.load(path, allow_pickle=False) as z:
        archive = {k: z[k] for k in z.files}
    if "phi" not in archive:
        raise KeyError(f"{path.name} has no 'phi' field")

    prob, built_ref, built_f, mode_m = entry
    # The reference is taken from the freshly assembled case, not from the archive:
    # a field stored under a superseded definition would silently score the recovered
    # solution against the wrong truth. The archive's own copy is used only when the
    # case has no analytical reference at all.
    phi_ref = built_ref if built_ref is not None else archive.get("phi_exact")
    f_vals = built_f if built_f is not None else archive.get("f")

    # The Thomas field for the same (case, N) supplies err_vs_thomas. It is a
    # separate archive, and its absence is not fatal - the column simply stays
    # None, which is what _record does when no reference is passed.
    thomas_path = path.with_name(
        path.name.replace(f"_{solver}_N{N}", f"_Thomas_N{N}"))
    phi_thomas = None
    if thomas_path.exists() and thomas_path != path:
        with np.load(thomas_path, allow_pickle=False) as z:
            phi_thomas = z["phi"] if "phi" in z.files else None

    res = synthesise_outer_result(archive)
    # The residual is recomputed against the freshly assembled operator, so it is a
    # property of the recovered field rather than a number inherited from a process
    # that never reported it.
    rhs = prob.rhs()
    b_norm = float(np.linalg.norm(rhs)) or 1.0
    res.residual = float(np.linalg.norm(rhs - prob.apply(res.u)) / b_norm)

    cfg = runner.SweepConfig()
    # Derived from the recomputed residual against the sweep's own tolerance, rather
    # than asserted. The killed process's verdict is not available, and stating one
    # without evidence would be the sort of plausible fabrication this whole script
    # is written to avoid.
    res.converged = bool(res.residual <= cfg.tol)

    rows: list = []
    runner._record(rows, case_id, solver.lower(), N, prob, res, phi_ref, f_vals,
                   phi_thomas, mode_m, cfg, notes="recovered_from_archive")
    if not rows:
        raise RuntimeError("_record produced no row")

    row = asdict(rows[0])
    row["solver"] = solver
    for field_name in UNRECOVERABLE_FIELDS:
        if field_name in row:
            row[field_name] = None
    return row


if __name__ == "__main__":
    raise SystemExit(main())
