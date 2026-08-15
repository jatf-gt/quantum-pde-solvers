"""
Determines precisely which (case, solver, N) combinations an HPC sweep still owes,
and — equally important — which it must never recompute.

Why this exists
---------------
Resubmission has repeatedly been driven from hand-written resolution ranges and from
directory listings. Both are unsound:

*   A ``.npz`` archive on disk does not imply a row in ``results_full.json``. A job
    killed mid-work-unit leaves the archive but never records the row, so a listing
    overstates completion. ``results_full.json`` is the only authority.
*   A blanket rerun destroys expensive correct work. Single rows in this archive cost
    25-38 h of wall time; recomputing one because a neighbouring row failed is the
    most costly mistake available.

This module classifies every combination, emits a machine-readable manifest of the
work genuinely outstanding, and prints a keep-list ordered by the wall time already
invested, so the cost of any proposed recomputation is explicit before submission.

Classification
--------------
good
    Converged, terminated on tolerance, carries a plausible error metric, and was not
    produced under a superseded geometry. Never rerun.
missing
    No row in the summary.
degraded
    Present but untrustworthy: failed to converge, hit a wall-clock or iteration cap,
    stagnated, recorded an exception, lacks an error metric, or reports an error too
    large to be credible for the case.

Notes
-----
The expected case list is recovered from the runner module by scanning its source for
case-identifier literals, rather than duplicated here, so that adding a section to a
runner cannot silently leave a hole in the analysis. Any identifier appearing in the
results but not in the runner is reported as drift rather than ignored.
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

# ``pytest.ini`` sets ``pythonpath = .``, but a bare ``python3 scripts/gap_analysis.py``
# puts ``scripts/`` on ``sys.path[0]`` rather than the repository root, so the local
# import below fails however sound the working directory. The PBS submission scripts
# invoke this module exactly that way for their post-run analysis, and the step is
# guarded with ``|| true``, so the failure was silent: the manifest failed to
# appear. Resolving the root from ``__file__`` makes the invocation location
# irrelevant, matching what every module under ``hpc/runners/`` already does.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark.hpc_archive import SweepArchive  # noqa: E402

# -- Policy --------------------------------------------------------------------

RUNNER_FOR_DIM: dict[int, str] = {
    1: "hpc/runners/run_1d.py",
    2: "hpc/runners/run_2d.py",
    3: "hpc/runners/run_3d.py",
}

CASES_UNSUPPORTED_AT_ORDER_4: frozenset[str] = frozenset({
    "HET_1D_3c_gaussian_NeumannDirichlet",
})
"""
Cases a 4th-order sweep does not attempt, and which must therefore not be reported
as outstanding when one is analysed.

``PoissonProblem1D4th`` implements no Neumann closure, so sub-case 3c is skipped by
``run_1d.py --order 4`` by design. The case list is recovered by scanning the runner
source for identifier literals, which cannot see a runtime exclusion, so without
this the analysis reports one phantom row per solver per resolution — 12 of the 24
entries in a 1-D order-4 manifest, all of them uncomputable.

Exclusions belong here rather than in ``--cases`` at the call site: a hand-written
case list is exactly the hand-maintained scope this module exists to replace.
"""

CASE_ID_PREFIXES: tuple[str, ...] = ("1D_", "2D_", "3D_", "HET_1D_")
"""
Prefixes marking a string literal in a runner as a recorded case identifier.

The runners spell the identifier they record in three different shapes — a literal
argument to ``_run_case`` in 2-D, an element of the tuple returned by a section
function in 3-D, and an assignment to ``case_id`` in 1-D — so matching on the value
rather than the syntactic position is what keeps this uniform across all three.
"""

STOP_REASONS_TRUNCATED: frozenset[str] = frozenset({"wall_time_exceeded"})
"""
Terminations that genuinely cut a solve short, so the field is unfinished.

Only the wall-clock cap qualifies. Stagnation is deliberately excluded: the outer
schemes detect it precisely so that a quantum solver sitting at its inner-solver
noise floor stops rather than burning thousands of futile strip solves. A stagnated
run with a sound field is the *designed* outcome, not a failure, and re-running it
reproduces the same stagnation at the same cost.
"""

STOP_REASONS_SOFT: frozenset[str] = frozenset({
    "stagnated", "max_iter", "max_cycles",
})
"""
Terminations recorded for visibility but not, by themselves, grounds to recompute.

A quantum inner solver rarely drives the outer residual to tolerance; it converges to
its own attractor a fixed distance from the truth. Treating that as a failure would
condemn most of the archive. Escalate with ``--strict`` when the intent is to
recompute anything that did not cleanly meet tolerance.
"""

EXCEPTION_MARKERS: tuple[str, ...] = (
    "No module named", "Error", "error", "Traceback", "Exception",
)
"""Substrings in ``notes`` betraying a recorded exception rather than a real result."""

TIMEOUT_MARKERS: tuple[str, ...] = ("hhl_timeout", "timeout", "timed_out")
"""
Substrings in ``notes`` marking a solve that *reached its budget*, not one that broke.

A timeout is a measurement. `hpc/runners/run_1d.py::_run_hhl` imposes a hard
one-hour budget per solve, and at N=32 and N=64 the 1-D operator's condition number
reaches ~1.7e3, growing HHL's clock register to the point where the statevector
simulation does not finish inside it. Recording that is the point of the benchmark;
recomputing it spends another hour to obtain the same row.

These markers are tested **before** `EXCEPTION_MARKERS`, because the substring
``"error"`` would otherwise capture the historical note ``"solver_error"`` that both
outcomes used to share.
"""

LEGACY_HHL_TIMEOUT_S: float = 3600.0
"""
The per-solve HHL budget **under which the existing 1-D archive was produced**.

A historical constant, not a mirror of the runner's current default. It exists only
to classify rows written before `_run_hhl` distinguished its two failure modes:
those carry nothing but ``notes="solver_error"``, and the sole surviving evidence
that the solve ran to its budget rather than raising is a ``wall_time_s`` at or above
it. The thirteen such rows all record 3600.2 s.

Deliberately **decoupled** from `run_1d.HHL_TIMEOUT_S`, which is now a
``--hhl-timeout-s`` default and expected to be raised: at 3600 s HHL does not
complete for N>=32, so finding where it does means increasing it. Rows produced at a
raised budget carry an explicit ``hhl_timeout:<budget>s`` marker and never reach this
inference, so the two values may diverge freely. Tying them together would have
meant that raising the runtime budget silently reclassified the historical rows as
genuine errors and scheduled 13 h of recomputation.
"""

STALE_GEOMETRY_CASES: frozenset[str] = frozenset({
    "HET_1D_3b_gaussian_Vd300",
    "2D_HET_MMS_SPT100",
    "2D_HET_Sin_MeetingReport",
    "3D_HET_MMS_SPT100",
    "3D_HET_RotatingSpoke_SPT100",
    "3D_HET_Discharge_SPT100",
})
"""
Cases proven to change under the SPT-100 geometry correction (commit ``861ff46``).

Regenerate with ``python scripts/check_geometry_impact.py --dim {1,2,3}``, which
compares the assembled operator, the strip operator, the spacings, the source and the
reference across both geometries. Note that the 1-D HET cases other than 3b, and
``het_2d_boeuf_garrigues``, are provably unaffected and must **not** be listed here —
adding them would trigger needless recomputation.
"""

IMPLAUSIBLE_REL_ERR_PCT: float = 100.0
"""
Relative error above which a quantum solve is treated as having failed.

Measured against **Thomas**, not against the exact solution, wherever
``err_vs_thomas`` is recorded. The distinction is essential: at N=4 a deliberately
under-resolved case such as the two-Gaussian or high-wavenumber source carries 40-50 %
discretisation error in *every* solver including Thomas, which is genuine truncation
error and not a defect — recomputing it changes nothing. What does indicate failure is
a quantum solver diverging from the classical reference on the same mesh, which is
exactly what ``err_vs_thomas`` isolates.
"""


# -- Expected grid -------------------------------------------------------------

def discover_case_ids(runner_path: Path) -> list[str]:
    """
    Recovers the case identifiers a runner records, by scanning its source.

    Parameters
    ----------
    runner_path : Path
        Path to the runner module.

    Returns
    -------
    list of str
        Identifiers, sorted and de-duplicated.

    Raises
    ------
    FileNotFoundError
        If the runner module does not exist.
    """
    tree = ast.parse(runner_path.read_text(encoding="utf-8"), str(runner_path))
    found = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        and node.value.startswith(CASE_ID_PREFIXES)
    }
    return sorted(found)


def merge_case_ids(discovered: Iterable[str], observed: Iterable[str]) -> list[str]:
    """
    Combines statically discovered identifiers with those actually recorded.

    Two corrections are applied. Identifiers only ever seen in the results — a case
    whose rows exist but whose runner spelling the scan missed — are added, so a real
    hole is never hidden. Identifiers that are a proper prefix of another are then
    dropped: the 1-D runner composes some case identifiers with an f-string
    (``f"1D_Poisson_{src_key}_hom"``), and the scan can only recover the literal
    fragment ``1D_Poisson_``, which would otherwise be reported as an entire missing
    case.

    Parameters
    ----------
    discovered : iterable of str
        Identifiers recovered from the runner source.
    observed : iterable of str
        Identifiers present in the recorded summary.

    Returns
    -------
    list of str
        Merged identifiers, sorted.
    """
    merged = set(discovered) | set(observed)
    return sorted(
        candidate for candidate in merged
        if not any(other != candidate and other.startswith(candidate)
                   for other in merged)
    )


# -- Classification ------------------------------------------------------------

def classify_row(row: dict, strict: bool = False, dim: int = 1
                 ) -> tuple[list[str], list[str]]:
    """
    Separates grounds for recomputation from merely notable properties of a row.

    Parameters
    ----------
    row : dict
        One record from ``results_full.json``.
    strict : bool
        Escalate the soft flags into recompute reasons, so that anything which did
        not cleanly meet tolerance is rerun.
    dim : int
        Spatial dimension of the sweep, 1, 2 or 3. Required because
        ``wall_time_s`` means different things across the two schemas: in 1-D it is
        one solve, and so is comparable against `HHL_TIMEOUT_S`; in 2-D and 3-D it
        is an entire outer iteration over N (or N²) strip solves, which routinely
        exceeds an hour in the normal course of a sound run. Applying the 1-D
        timeout inference there would mark most large-N HHL rows as timed out and
        suppress the very reasons that schedule them for rerun.

    Returns
    -------
    tuple of (list of str, list of str)
        (reasons, flags). A row is sound exactly when `reasons` is empty; `flags`
        are recorded either way so that an accepted stagnation stays visible.
    """
    reasons: list[str] = []
    flags: list[str] = []
    is_thomas = str(row.get("solver", "")).lower() == "thomas"

    if row.get("case") in STALE_GEOMETRY_CASES:
        reasons.append("stale_geometry")

    notes = str(row.get("notes") or "")
    # Order matters: a timed-out solve is a terminal measurement, whereas a raised
    # exception is a defect, and the historical note "solver_error" was written for
    # both. Test for the timeout first, then fall back to the wall-clock inference
    # for rows predating the explicit marker, and only then treat it as an error.
    wall = row.get("wall_time_s")
    timed_out = (
        any(marker in notes for marker in TIMEOUT_MARKERS)
        or (dim == 1 and str(row.get("solver", "")).lower() == "hhl"
            and wall is not None and float(wall) >= LEGACY_HHL_TIMEOUT_S)
    )
    if timed_out:
        flags.append("solver_timeout")
    elif any(marker in notes for marker in EXCEPTION_MARKERS):
        reasons.append("solver_error")

    stop = str(row.get("stop_reason") or "")
    if stop in STOP_REASONS_TRUNCATED:
        reasons.append(stop)
    elif stop in STOP_REASONS_SOFT:
        flags.append(stop)

    if not row.get("converged", False):
        flags.append("not_converged")

    # Accuracy is judged against Thomas on the same mesh where that is recorded,
    # isolating solver failure from shared discretisation error. Thomas itself is the
    # reference and so is never judged on magnitude.
    if not is_thomas:
        err = row.get("err_vs_thomas")
        basis = "vs_thomas"
        if err is None:
            err, basis = row.get("max_rel_err"), "vs_exact"
        if err is None:
            # A timed-out solve has no error metric *because* it produced no
            # solution, which is the recorded outcome rather than a missing
            # measurement. Escalating it here would reinstate by the back door the
            # recomputation the timeout classification above exists to prevent.
            if timed_out:
                flags.append("no_error_metric")
            else:
                reasons.append("no_error_metric")
        elif float(err) > IMPLAUSIBLE_REL_ERR_PCT:
            # A flag, not a reason. A large error can be the finding rather than a
            # fault: in 1-D the operator's condition number reaches ~1.7e3 by N=64,
            # and the degradation of HHL and QSVT with kappa is precisely what the
            # benchmark exists to measure. Recomputing reproduces the same number at
            # the same cost. Only --strict escalates these.
            flags.append(f"large_error_{basis}")

    if strict:
        reasons.extend(flags)
        flags = []

    return reasons, flags


def analyse(
    archive:  SweepArchive,
    cases:    Iterable[str],
    n_values: Iterable[int],
    solvers:  Iterable[str],
    strict:   bool = False,
    dim:      int = 1,
) -> dict:
    """
    Classifies every expected combination against the recorded summary.

    Parameters
    ----------
    archive : SweepArchive
        Reader for the sweep directory.
    cases, n_values, solvers : iterable
        The expected grid.
    strict : bool
        Treat any row that did not cleanly meet tolerance as outstanding.
    dim : int
        Spatial dimension, forwarded to `classify_row`, whose wall-clock timeout
        inference is valid only for the 1-D schema.

    Returns
    -------
    dict
        Manifest with ``rerun``, ``keep`` and ``drift`` entries.
    """
    rows = archive.rows()

    # Index by (case, solver, N). A case may legitimately appear more than once when
    # a sweep was resumed; the last row written is the operative one.
    indexed: dict[tuple[str, str, int], dict] = {}
    for row in rows:
        if str(row.get("notes") or "").startswith("scheme_comparison"):
            continue
        indexed[(row["case"], row["solver"], int(row["N"]))] = row

    rerun, keep = [], []
    for case in cases:
        for N in n_values:
            for solver in solvers:
                row = indexed.get((case, solver, int(N)))
                if row is None:
                    rerun.append({"case": case, "solver": solver, "N": int(N),
                                  "reasons": ["missing"], "flags": [],
                                  "wall_time_s": None})
                    continue
                reasons, flags = classify_row(row, strict=strict, dim=dim)
                entry = {"case": case, "solver": solver, "N": int(N),
                         "flags": flags, "wall_time_s": row.get("wall_time_s")}
                if reasons:
                    rerun.append({**entry, "reasons": reasons})
                else:
                    keep.append(entry)

    expected = {(c, s, int(n)) for c in cases for s in solvers for n in n_values}
    drift = sorted({key[0] for key in indexed if key not in expected})

    return {"rerun": rerun, "keep": keep, "drift": drift,
            "n_rows_recorded": len(indexed),
            "orphan_archives": _orphan_archives(archive, indexed)}


def _orphan_archives(archive: SweepArchive,
                     indexed: dict[tuple[str, str, int], dict]) -> list[str]:
    """
    Solution archives on disk with no corresponding row in the summary.

    A populated archive whose row is absent means the summary does not describe the
    work actually done. There are two causes and both are dangerous here: a job
    killed mid-work-unit writes the ``.npz`` but never records the row, and — more
    insidiously — the runners open ``results_full.json`` in truncating mode, so a
    *concurrently running* job momentarily presents an almost-empty summary. Acting on
    that summary would schedule a full recomputation of work that already exists.

    Parameters
    ----------
    archive : SweepArchive
        Reader for the sweep directory.
    indexed : dict
        Rows keyed by (case, solver, N).

    Orphans are split by whether they are explained. An archive belonging to a case
    whose rows were deliberately removed by ``scripts/cleanup_stale_geometry.py`` is
    expected residue: the row is gone because the result is superseded, and the
    archive should simply be purged. An orphan of any other case is unexplained and
    is the signature of the dangerous condition.

    Returns
    -------
    dict
        ``{"stale": [...], "unexplained": [...]}``, each sorted.
    """
    prefix = archive.solution_prefix
    recorded = {f"{prefix}_{case}_{solver}_N{N}.npz"
                for case, solver, N in indexed}
    on_disk = {path.name for path in archive.results_dir.glob(f"{prefix}_*.npz")}
    orphans = sorted(on_disk - recorded)

    stale, unexplained = [], []
    for name in orphans:
        body = name[len(prefix) + 1:].removesuffix(".npz")
        if any(body.startswith(f"{case}_") for case in STALE_GEOMETRY_CASES):
            stale.append(name)
        else:
            unexplained.append(name)
    return {"stale": stale, "unexplained": unexplained}


# -- Reporting -----------------------------------------------------------------

def _print_report(manifest: dict, show_keep: int) -> None:
    """Prints the human-readable summary."""
    rerun, keep = manifest["rerun"], manifest["keep"]

    tally: dict[str, int] = {}
    for entry in rerun:
        for reason in entry["reasons"]:
            tally[reason] = tally.get(reason, 0) + 1

    flag_tally: dict[str, int] = {}
    for entry in keep:
        for flag in entry.get("flags", ()):
            flag_tally[flag] = flag_tally.get(flag, 0) + 1

    print()
    print(f"  Recorded rows      : {manifest['n_rows_recorded']}")
    print(f"  Sound  (keep)      : {len(keep)}")
    print(f"  Outstanding (rerun): {len(rerun)}")
    print()
    if tally:
        print("  Grounds for recomputation")
        for reason, count in sorted(tally.items(), key=lambda kv: -kv[1]):
            print(f"    {reason:<26} {count:>5}")
        print()
    if flag_tally:
        print("  Accepted without recomputation (recorded for visibility)")
        for flag, count in sorted(flag_tally.items(), key=lambda kv: -kv[1]):
            print(f"    {flag:<26} {count:>5}")
        print("    Re-run these too with --strict.")
        print()

    if rerun:
        print(f"  {'Case':<34}{'Solver':<9}{'N':>5}   Reasons")
        print("  " + "-" * 76)
        for entry in sorted(rerun, key=lambda e: (e["case"], e["N"], e["solver"])):
            print(f"  {entry['case']:<34}{entry['solver']:<9}{entry['N']:>5}   "
                  f"{','.join(entry['reasons'])}")
        print()

    # Ordered by the wall time already invested: the rows at the top of this list are
    # the ones a careless blanket rerun would destroy.
    priced = sorted((e for e in keep if e.get("wall_time_s")),
                    key=lambda e: -float(e["wall_time_s"]))
    if priced:
        print(f"  Most expensive sound results - MUST NOT be recomputed "
              f"(top {min(show_keep, len(priced))}):")
        print(f"  {'Case':<34}{'Solver':<9}{'N':>5}{'hours':>10}")
        print("  " + "-" * 60)
        for entry in priced[:show_keep]:
            hours = float(entry["wall_time_s"]) / 3600.0
            print(f"  {entry['case']:<34}{entry['solver']:<9}"
                  f"{entry['N']:>5}{hours:>10.2f}")
        total = sum(float(e["wall_time_s"]) for e in priced) / 3600.0
        print(f"  {'':<48}{total:>10.2f}  total preserved")
        print()

    if manifest["drift"]:
        print("  Recorded cases absent from the expected grid (check --cases):")
        for case in manifest["drift"]:
            print(f"    {case}")
        print()

    orphans = manifest.get("orphan_archives") or {}
    stale, unexplained = orphans.get("stale", []), orphans.get("unexplained", [])

    if stale:
        print(f"  {len(stale)} superseded archive(s) remain from cases stripped by")
        print("  cleanup_stale_geometry.py. Expected residue - purge them so a failed")
        print("  rerun cannot leave a stale field behind an absent row:")
        print(f"      rm {manifest.get('results_dir', '.')}/<case>_*.npz")
        print()

    if unexplained:
        print("  " + "!" * 74)
        print(f"  UNSAFE: {len(unexplained)} archive(s) have no summary row and no")
        print("  known explanation. The summary does not describe the work present.")
        print("  Either a job is writing here right now (the runners truncate")
        print("  results_full.json on start), or the sync from the cluster is partial.")
        print("  Submitting from this manifest would recompute existing results.")
        print("  Stop the job, re-sync, and re-run this analysis before submitting.")
        for name in unexplained[:10]:
            print(f"      {name}")
        if len(unexplained) > 10:
            print(f"      ... and {len(unexplained) - 10} more")
        print("  " + "!" * 74)
        print()


def _parse_list(text: Optional[str]) -> Optional[list[str]]:
    """Splits a comma-separated option value, tolerating spaces."""
    if not text:
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Report outstanding and untrustworthy rows in an HPC sweep.")
    parser.add_argument("--dim", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--n-values", default="4,8,16,32,64",
                        help="Expected resolutions (default: 4,8,16,32,64).")
    parser.add_argument("--solvers", default="Thomas,HHL,VQLS,QSVT",
                        help="Expected solver labels.")
    parser.add_argument("--cases", default=None,
                        help="Override the case list; defaults to those discovered "
                             "in the corresponding runner module.")
    parser.add_argument("--order", type=int, choices=(2, 4), default=2,
                        help="Discretisation order of the sweep being analysed "
                             "(default: 2). At order 4 the cases the runner does "
                             "not implement are dropped from the expected set, "
                             "which otherwise reports them as missing rows that "
                             "cannot be produced.")
    parser.add_argument("--show-keep", type=int, default=10,
                        help="How many of the costliest sound rows to list.")
    parser.add_argument("--strict", action="store_true",
                        help="Also recompute rows that stagnated or hit an iteration "
                             "cap. Off by default: stagnation is the designed "
                             "terminal state for a quantum solver at its noise floor.")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Write the manifest JSON here.")
    args = parser.parse_args()

    runner = Path(RUNNER_FOR_DIM[args.dim])
    n_values = [int(v) for v in _parse_list(args.n_values)]
    solvers = _parse_list(args.solvers)

    archive = SweepArchive(args.results_dir, dim=args.dim,
                           skip_scheme_comparison=True)
    observed = {row["case"] for row in archive.rows()}
    cases = (_parse_list(args.cases)
             or merge_case_ids(discover_case_ids(runner), observed))

    # An explicit --cases list is taken as given; the order-4 exclusion applies only
    # to the discovered set, which is what cannot see a runtime skip.
    excluded: list[str] = []
    if args.order == 4 and not args.cases:
        excluded = sorted(set(cases) & CASES_UNSUPPORTED_AT_ORDER_4)
        cases = [c for c in cases if c not in CASES_UNSUPPORTED_AT_ORDER_4]

    print()
    print("=" * 78)
    print(f"  GAP ANALYSIS  -  {args.dim}D  -  order {args.order}  -  {args.results_dir}")
    print(f"  cases from {runner} + recorded rows: {len(cases)}")
    if excluded:
        print(f"  excluded at order 4 (unimplemented): {', '.join(excluded)}")
    print(f"  N={n_values}  solvers={solvers}")
    print("=" * 78)
    manifest = analyse(archive, cases, n_values, solvers, strict=args.strict,
                       dim=args.dim)
    manifest.update({
        "generated":   datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dim":         args.dim,
        "order":       args.order,
        "results_dir": str(args.results_dir),
        "strict":      args.strict,
        "expected":    {"cases": cases, "n_values": n_values, "solvers": solvers,
                        "excluded_unimplemented": excluded},
    })

    _print_report(manifest, args.show_keep)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(manifest, handle, indent=2)
        print(f"  Manifest written to {args.output}")
        print()


if __name__ == "__main__":
    main()
