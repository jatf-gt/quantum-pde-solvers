"""
Assembles the per-solution field archives into a Zenodo deposit.

Why this exists
---------------
The repository tracks every summary, table and tidy figure dataset — some 7.7 MB
across 209 files — but not the per-solution `.npz` archives, which hold the
solution, exact and Thomas fields of each solve and total some 180 MB. Git
retains every version of a binary permanently and the repository is public, so
committing them is irreversible in practice. They are, at the same time, the one
output that cannot be regenerated without days of cluster time, and the only
artefact from which a solution field can be re-rendered or a metric derived that
was not recorded at the time. A data repository with a DOI is the appropriate
home for them; this module builds the deposit.

What is deposited
-----------------
The solution archives of the six primary sweeps and of the uniform-degree QSVT
ladder, one zip per sweep, paths relative to `results/`. Excluded:

*   `results/qsvt_phase_cache/` — 428 `.npz` of QSP phase angles, 3.6 MB, already
    tracked in the repository.
*   Figures and logs — regenerable from the summaries by
    `hpc/runners/plot_results.py`, and in the 2-D case running to 156 MB of logs.

Each zip carries the sweep's `results_full.json` and `run_metadata.json`
alongside its fields, so that a single downloaded archive is self-describing: the
row that produced each field, and the environment and git commit that produced
the row.

Products, under the output directory:

  <sweep>.zip      One per sweep, fields plus that sweep's summary and metadata.
  MANIFEST.csv     One row per deposited field: sweep, case, solver, N, size,
                   SHA-256. The integrity record, and the index a reader needs to
                   find one solve without unzipping 180 MB.
  README.md        The deposit's own description: provenance, schema, and how to
                   read a field back.

Usage
-----
    python scripts/utils/make_zenodo_package.py --out <dir>
    python scripts/utils/make_zenodo_package.py --out <dir> --dry-run

`--dry-run` reports the inventory and writes nothing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ── Deposit scope ─────────────────────────────────────────────────────────────

# The six primary sweeps plus the uniform-degree ladder. Named explicitly rather
# than globbed: `results/` also holds the studies directories, whose archives
# carry no fields, and the phase cache, which is tracked in the repository.
SWEEPS: tuple[str, ...] = (
    "1Dhpc_run",
    "1Dhpc_run_4th",
    "1Dhpc_run_degcap5000",
    "2Dhpc_run",
    "2Dhpc_run_4th",
    "3Dhpc_run",
    "3Dhpc_run_4th",
)

# Carried into each zip beside the fields, so one archive is self-describing.
COMPANION_FILES: tuple[str, ...] = ("results_full.json", "run_metadata.json")

log = logging.getLogger("zenodo")


# ── Inventory ─────────────────────────────────────────────────────────────────

def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """
    Hash one file.

    Parameters
    ----------
    path : Path
        File to hash.
    chunk : int, optional
        Read size in bytes; the default of 1 MiB keeps a 137 MB sweep off the
        heap without measurable cost.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _parse_name(stem: str) -> tuple[str, str, str]:
    """
    Recover (case, solver, N) from a solution archive's filename.

    The convention is `solutions_<case>_<solver>_N<n>` in 1-D and
    `solution2d_<case>_<solver>_N<n>` / `solution3d_...` in 2-D and 3-D, which is
    the naming `benchmark/results_io.py` declares. Parsing is best-effort and
    positional from the right: the case identifier itself contains underscores,
    so only the trailing two fields can be taken positionally.

    Parameters
    ----------
    stem : str
        Filename without its `.npz` suffix.

    Returns
    -------
    tuple of str
        (case, solver, N). Any field the name does not carry is returned empty,
        which is the situation for the aggregate `all_solutions` archives.
    """
    parts = stem.split("_")
    if len(parts) < 3 or not parts[-1].startswith("N"):
        return ("", "", "")
    n_value = parts[-1][1:]
    solver = parts[-2]
    case = "_".join(parts[1:-2])
    return (case, solver, n_value)


def inventory(results_dir: Path) -> list[dict]:
    """
    List every field archive to be deposited, with its identity and digest.

    Parameters
    ----------
    results_dir : Path
        The repository's `results/` directory.

    Returns
    -------
    list of dict
        One record per `.npz`, carrying sweep, case, solver, N, byte size and
        SHA-256. Ordered by sweep then filename, so the manifest is stable
        across runs and a re-deposit diffs cleanly against its predecessor.
    """
    records: list[dict] = []
    for sweep in SWEEPS:
        sweep_dir = results_dir / sweep
        if not sweep_dir.is_dir():
            log.warning("  %-24s absent; skipped", sweep)
            continue
        for npz in sorted(sweep_dir.rglob("*.npz")):
            case, solver, n_value = _parse_name(npz.stem)
            records.append({
                "sweep":    sweep,
                "path":     npz.relative_to(results_dir).as_posix(),
                "case":     case,
                "solver":   solver,
                "N":        n_value,
                "bytes":    npz.stat().st_size,
                "sha256":   _sha256(npz),
            })
    return records


# ── Products ──────────────────────────────────────────────────────────────────

def write_manifest(records: list[dict], out_dir: Path) -> Path:
    """
    Write the deposit's integrity record and index.

    Parameters
    ----------
    records : list of dict
        Output of `inventory`.
    out_dir : Path
        Directory to write into.

    Returns
    -------
    Path
        The manifest written.
    """
    path = out_dir / "MANIFEST.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["sweep", "path", "case", "solver", "N", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(records)
    return path


def write_zips(records: list[dict], results_dir: Path, out_dir: Path) -> dict[str, int]:
    """
    Write one zip per sweep, fields plus that sweep's summary and metadata.

    `.npz` is already a compressed container, so the archives are stored rather
    than deflated: deflating them again costs minutes of CPU and returns under a
    per cent, while `ZIP_STORED` lets a reader extract one field without
    decompressing the whole archive.

    Parameters
    ----------
    records : list of dict
        Output of `inventory`.
    results_dir : Path
        The repository's `results/` directory.
    out_dir : Path
        Directory to write the zips into.

    Returns
    -------
    dict
        {sweep: bytes written}, for the report.
    """
    sizes: dict[str, int] = {}
    for sweep in SWEEPS:
        members = [r for r in records if r["sweep"] == sweep]
        if not members:
            continue
        zip_path = out_dir / f"{sweep}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED,
                             allowZip64=True) as zf:
            for record in members:
                zf.write(results_dir / record["path"], record["path"])
            for name in COMPANION_FILES:
                companion = results_dir / sweep / name
                if companion.exists():
                    zf.write(companion, f"{sweep}/{name}")
        sizes[sweep] = zip_path.stat().st_size
        log.info("    %-28s %6.1f MB  (%d fields)",
                 zip_path.name, sizes[sweep] / 1048576, len(members))
    return sizes


def write_readme(records: list[dict], sizes: dict[str, int], out_dir: Path) -> Path:
    """
    Write the deposit's own description.

    A data record is read by someone who does not have the repository open, so
    the description must state what a field is, what units it carries and how to
    read one back, without reference to the source tree.

    Parameters
    ----------
    records : list of dict
        Output of `inventory`.
    sizes : dict
        {sweep: bytes}, from `write_zips`.
    out_dir : Path
        Directory to write into.

    Returns
    -------
    Path
        The README written.
    """
    total_mb = sum(r["bytes"] for r in records) / 1048576
    lines = [
        "# Per-solution field archives",
        "",
        "Solution fields from the quantum linear-solver benchmarks of "
        "`quantum-pde-solvers` — HHL, VQLS and QSVT against the classical Thomas "
        "algorithm, on the Poisson equation in one, two and three dimensions and "
        "on a Hall-effect thruster electrostatics model.",
        "",
        f"{len(records)} archives, {total_mb:.1f} MB, across {len(sizes)} sweeps.",
        "",
        "## Contents",
        "",
        "| archive | fields | size |",
        "|---|---|---|",
    ]
    for sweep in SWEEPS:
        members = [r for r in records if r["sweep"] == sweep]
        if not members:
            continue
        lines.append(f"| `{sweep}.zip` | {len(members)} | "
                     f"{sizes.get(sweep, 0) / 1048576:.1f} MB |")
    lines += [
        "",
        "Each zip also carries that sweep's `results_full.json` (one row per "
        "(case, solver, N), with every recorded metric) and `run_metadata.json` "
        "(environment, resolved configuration and git commit), so an archive is "
        "self-describing without the others.",
        "",
        "`MANIFEST.csv` indexes every field by sweep, case, solver and resolution, "
        "with byte size and SHA-256.",
        "",
        "## Reading a field",
        "",
        "```python",
        "import numpy as np",
        "d = np.load('1Dhpc_run/solutions_1D_Poisson_fS_nonhom_QSVT_N16.npz')",
        "d.files              # the arrays present",
        "d['u_solver']        # the quantum solution on the solver's nodes",
        "```",
        "",
        "Field names differ by dimension and by the era of the run: the solution is "
        "`u_solver` in one dimension and `phi_solver` or `phi` in two and three. "
        "The aliases are declared once, in `benchmark/results_io.py` in the source "
        "repository, and every reader there resolves them.",
        "",
        "## What is not here",
        "",
        "The QSP phase-angle cache (428 archives, 3.6 MB) is tracked in the source "
        "repository, as are all summaries, tables and the tidy datasets behind every "
        "figure. Figures and job logs are omitted: the first regenerate from the "
        "summaries, and the second run to 156 MB with no analytical content.",
        "",
        "## Source",
        "",
        "https://github.com/jatf-gt/quantum-pde-solvers",
        "",
        "Produced on Imperial College London's CX3 cluster. All circuits are "
        "deterministic statevector simulations; no shot noise.",
        "",
    ]
    path = out_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    """
    Build the deposit, or report what it would contain.

    Returns
    -------
    int
        0 on success; 1 where the results directory holds no field archive.
    """
    ap = argparse.ArgumentParser(
        description="Assemble the per-solution field archives into a Zenodo deposit.")
    ap.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results",
                    help="Source directory (default: the repository's results/).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Directory to build the deposit in. Created if absent.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report the inventory and write nothing.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    log.info("=" * 78)
    log.info("  ZENODO DEPOSIT%s", "  (dry run)" if args.dry_run else "")
    log.info("=" * 78)
    log.info("  source    : %s", args.results_dir)

    records = inventory(args.results_dir)
    if not records:
        log.error("  No field archives found under %s.", args.results_dir)
        return 1

    total = sum(r["bytes"] for r in records)
    log.info("  fields    : %d, %.1f MB", len(records), total / 1048576)
    for sweep in SWEEPS:
        members = [r for r in records if r["sweep"] == sweep]
        if members:
            log.info("    %-24s %4d fields  %7.1f MB", sweep, len(members),
                     sum(r["bytes"] for r in members) / 1048576)

    if args.dry_run:
        log.info("-" * 78)
        log.info("  Dry run: nothing written.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    log.info("-" * 78)
    log.info("  Building in %s", args.out)
    sizes = write_zips(records, args.results_dir, args.out)
    log.info("    %s", write_manifest(records, args.out).name)
    log.info("    %s", write_readme(records, sizes, args.out).name)
    log.info("-" * 78)
    log.info("  Deposit size: %.1f MB in %d files",
             sum(sizes.values()) / 1048576, len(sizes) + 2)
    log.info("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
