#!/usr/bin/env python3
"""
Recompute the accuracy metrics of recovered rows through the runner's own definitions.

Why this exists
---------------
Rows reconstructed from a solution archive after a job was killed carry accuracy
metrics computed by whatever ad-hoc script performed the reconstruction, not by
the sweep runner. In `results/2Dhpc_run/results_full.json` the twenty-four rows
tagged ``recovered_from_npz`` disagree with the instrumented rows in two ways at
once:

  **Scale.** ``max_rel_err`` was recorded as a fraction where every instrumented
  row records per cent. A curve drawn from that column therefore has a
  hundred-fold dip at exactly the resolutions the reconstruction covered — at
  N = 64 the recovered Thomas row reads 1.96e-4 against 7.56e-2 at N = 32, which
  is 386x lower where a second-order scheme should give four times lower.

  **Masking.** `hpc/runners/run_2d.py::_max_rel` excludes nodes at which the
  reference is smaller than 1e-10 before dividing. This matters for the HET
  manufactured solution, whose profile passes through zero at the anode, the
  cathode and the outer wall; including those nodes makes the metric diverge on a
  perfectly sound field. The reconstruction instead floored the denominator at
  1e-12, which does not exclude those nodes but inflates them.

Multiplying the column by a hundred would fix the first and leave the second. The
metrics are therefore recomputed from the archived field, through the runner's
own metric functions imported directly, so there is one definition rather than
two that have to be kept in agreement.

What is and is not touched
--------------------------
Recomputed, because each is a pure function of the stored field and the stored
reference:

    max_rel_err, max_abs_err, rel_l2_err, rms_err, linf_err

Left exactly as found:

    Every other column, every row not tagged as recovered, and every ``.npz``.
    In particular the instrumentation that existed only in the killed process —
    wall time, strip-solve counts, the residual — stays absent rather than being
    invented.

Self-check
----------
Before touching anything the script recomputes the metrics of rows that were
*instrumented*, whose values are known good, and reports the agreement. If the
recomputation cannot reproduce the runner's own numbers on those rows it does not
reproduce them on the recovered ones either, and the run aborts rather than
writing. Agreement is expected to machine precision: the same function is being
called on the same array.

Usage
-----
    python scripts/utils/normalise_recovered_metrics.py --dim 2 --dry-run
    python scripts/utils/normalise_recovered_metrics.py --dim 2

``--dry-run`` prints the before-and-after table and writes nothing. Without it a
timestamped backup of ``results_full.json`` is written alongside the original
before any change, and ``results_summary.csv`` is regenerated to match.

The operation is idempotent: a second run finds every recovered row already in
agreement and reports no changes.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hpc.runners.run_2d import _accuracy                             # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("normalise")

# Sweep directory and solution-archive prefix per dimension. 2-D writes
# `solutions_*.npz`; 3-D writes `solution3d_*.npz`.
SWEEPS: dict[int, list[tuple[str, str]]] = {
    2: [("results/2Dhpc_run", "solutions"),
        ("results/2Dhpc_run_4th", "solutions")],
    3: [("results/3Dhpc_run", "solution3d"),
        ("results/3Dhpc_run_4th", "solution3d")],
}

# Fields this script owns. Every one is a pure function of the stored field and
# the stored reference, and none exists only in a running process.
RECOMPUTED: tuple[str, ...] = (
    "max_rel_err", "max_abs_err", "rel_l2_err", "rms_err", "linf_err",
)

# Field-name aliases across sweep generations, as declared in
# `benchmark/results_io.py`. The solution key differs by dimension and by the
# vintage of the writer.
SOLVER_KEYS: tuple[str, ...] = ("phi_solver", "u_solver", "phi")
EXACT_KEYS: tuple[str, ...] = ("phi_exact", "u_exact")

# Relative agreement required of the self-check. The same function is applied to
# the same array, so anything beyond floating-point noise indicates the archive
# and the summary have drifted apart and the recomputation cannot be trusted.
SELF_CHECK_TOL: float = 1.0e-9


# ── Private Utility Methods ────────────────────────────────────────────────────

def _first(data, keys: tuple[str, ...]) -> Optional[np.ndarray]:
    """
    Return the first array present under any of a set of alias keys.

    Parameters
    ----------
    data : np.lib.npyio.NpzFile
        Opened solution archive.
    keys : tuple of str
        Alias keys, in order of preference.

    Returns
    -------
    np.ndarray or None
        The array, or None where the archive carries none of the aliases.
    """
    for key in keys:
        if key in data.files:
            return data[key]
    return None


def _archive_for(sweep: Path, prefix: str, row: dict) -> Optional[Path]:
    """
    Locate the solution archive belonging to one summary row.

    Parameters
    ----------
    sweep : Path
        Sweep directory.
    prefix : str
        Archive filename prefix for this dimension.
    row : dict
        Summary row carrying `case`, `solver` and `N`.

    Returns
    -------
    Path or None
        The archive path, or None where it is absent.
    """
    path = sweep / f"{prefix}_{row['case']}_{row['solver']}_N{row['N']}.npz"
    return path if path.exists() else None


def _recompute(path: Path) -> Optional[dict]:
    """
    Recompute the accuracy metrics of one archived solution.

    Delegates to `hpc.runners.run_2d._accuracy`, the function the sweep itself
    used, rather than restating the definitions. The masking convention, the
    per-cent scaling and the zero-guard therefore cannot drift from the runner's.

    Parameters
    ----------
    path : Path
        Solution archive.

    Returns
    -------
    dict or None
        Mapping of the recomputed fields, or None where the archive carries no
        reference solution and the metrics are consequently undefined.
    """
    with np.load(path, allow_pickle=False) as data:
        u = _first(data, SOLVER_KEYS)
        ref = _first(data, EXACT_KEYS)
        if u is None or ref is None:
            return None
        return _accuracy(np.asarray(u, dtype=float),
                         np.asarray(ref, dtype=float))


def _agrees(a: Optional[float], b: Optional[float]) -> bool:
    """
    Whether two metric values agree to within the self-check tolerance.

    Two absent values agree; one absent value does not agree with a present one,
    since that is exactly the discrepancy worth reporting.

    Parameters
    ----------
    a, b : float or None
        Values to compare.

    Returns
    -------
    bool
        Whether the values agree.
    """
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    a, b = float(a), float(b)
    if np.isnan(a) and np.isnan(b):
        return True
    return abs(a - b) <= SELF_CHECK_TOL * max(1.0, abs(a), abs(b))


def _write_csv(json_path: Path, csv_path: Path) -> None:
    """
    Regenerate the flat CSV from the JSON, preserving the union of all columns.

    Parameters
    ----------
    json_path : Path
        Source summary.
    csv_path : Path
        Destination CSV.
    """
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── Driver ─────────────────────────────────────────────────────────────────────

def process(sweep: Path, prefix: str, dry_run: bool) -> tuple[int, int, int]:
    """
    Normalise one sweep directory.

    Parameters
    ----------
    sweep : Path
        Sweep directory holding `results_full.json` and the solution archives.
    prefix : str
        Archive filename prefix for this dimension.
    dry_run : bool
        If True, report without writing.

    Returns
    -------
    checked : int
        Instrumented rows used for the self-check.
    changed : int
        Recovered rows whose metrics were corrected.
    skipped : int
        Recovered rows whose archive or reference was unavailable.

    Raises
    ------
    SystemExit
        If the self-check fails, which means the recomputation does not
        reproduce the runner's own numbers and must not be applied.
    """
    json_path = sweep / "results_full.json"
    if not json_path.exists():
        log.info("  %-28s no results_full.json; skipped", str(sweep))
        return 0, 0, 0

    rows = json.loads(json_path.read_text(encoding="utf-8"))

    # -- Self-check against instrumented rows ---------------------------------
    checked = mismatched = 0
    for row in rows:
        if "recovered" in str(row.get("notes") or ""):
            continue
        if row.get("max_rel_err") is None:
            continue
        path = _archive_for(sweep, prefix, row)
        if path is None:
            continue
        fresh = _recompute(path)
        if not fresh:
            continue
        checked += 1
        for field in RECOMPUTED:
            if not _agrees(fresh.get(field), row.get(field)):
                mismatched += 1
                log.warning(
                    "    self-check MISMATCH %s/%s/N%s %s: archive %.6g, "
                    "recomputed %.6g",
                    row["case"], row["solver"], row["N"], field,
                    float(row.get(field) or float("nan")),
                    float(fresh.get(field) or float("nan")),
                )
                break
        if mismatched > 3:
            break

    if checked == 0:
        log.warning("  %-28s no instrumented row could be re-derived; "
                    "self-check inconclusive, refusing to write", str(sweep))
        return 0, 0, 0
    if mismatched:
        raise SystemExit(
            f"Self-check failed on {mismatched} of {checked} instrumented rows "
            f"in {sweep}. The recomputation does not reproduce the runner's own "
            f"metrics, so it must not be applied to the recovered rows. "
            f"Nothing was written."
        )
    log.info("  %-28s self-check passed on %d instrumented row(s)",
             str(sweep), checked)

    # -- Correct the recovered rows -------------------------------------------
    changed = skipped = 0
    for row in rows:
        if "recovered" not in str(row.get("notes") or ""):
            continue
        path = _archive_for(sweep, prefix, row)
        if path is None:
            skipped += 1
            log.warning("    no archive for %s/%s/N%s; left unchanged",
                        row["case"], row["solver"], row["N"])
            continue
        fresh = _recompute(path)
        if not fresh:
            skipped += 1
            log.warning("    %s carries no reference solution; left unchanged",
                        path.name)
            continue

        deltas = [f for f in RECOMPUTED
                  if not _agrees(fresh.get(f), row.get(f))]
        if not deltas:
            continue

        log.info("    %-34s %-7s N=%-4s %s",
                 row["case"][:34], row["solver"], row["N"],
                 "  ".join(
                     f"{f}: {row.get(f)!r} -> {fresh.get(f):.6g}"
                     for f in deltas if fresh.get(f) is not None
                 ))
        if not dry_run:
            row.update({f: fresh[f] for f in RECOMPUTED if f in fresh})
        changed += 1

    if changed and not dry_run:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        backup = json_path.with_name(f"results_full.{stamp}.pre-normalise.json")
        shutil.copy2(json_path, backup)
        json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        _write_csv(json_path, sweep / "results_summary.csv")
        log.info("  %-28s wrote %d corrected row(s); backup at %s",
                 str(sweep), changed, backup.name)

    return checked, changed, skipped


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dim", type=int, choices=(2, 3), default=2,
                    help="Which dimension's sweeps to normalise.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the corrections without writing.")
    args = ap.parse_args()

    log.info("=" * 78)
    log.info("  RECOVERED-ROW METRIC NORMALISATION  -  %d-D%s",
             args.dim, "  (dry run)" if args.dry_run else "")
    log.info("=" * 78)

    total_changed = total_skipped = 0
    for rel, prefix in SWEEPS[args.dim]:
        checked, changed, skipped = process(
            REPO_ROOT / rel, prefix, args.dry_run)
        total_changed += changed
        total_skipped += skipped

    log.info("-" * 78)
    log.info("  %d row(s) %s, %d skipped",
             total_changed,
             "would be corrected" if args.dry_run else "corrected",
             total_skipped)
    log.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
