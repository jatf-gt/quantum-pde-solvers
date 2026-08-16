#!/usr/bin/env python3
"""
cleanup_stale_geometry.py
===========================
Removes rows for cases affected by the SPT-100 geometry correction
(core/het_geometry.py: R_in 35mm->30mm, L_z 25mm->40mm, per Boeuf &
Garrigues 1998) from results_full.json / results_summary.csv, BEFORE
re-running them.

Why this step is necessary and --append alone is not enough: --append
merges with whatever is already on disk. It has no notion that a row for
case "2D_HET_MMS_SPT100" computed under the OLD geometry is now wrong, not
merely incomplete - the case ID string does not change even though the
physical problem it describes did. Re-running with --append after the
geometry fix, without this cleanup, would leave both the old (wrong) and
new (correct) rows for the same (case, N, solver) sitting side by side in
results_full.json, silently ambiguous to any later analysis.

The corresponding .npz solution files do NOT need manual deletion - a
re-run writes to the same filename and overwrites them automatically.

Usage:
    python3 cleanup_stale_geometry.py results/2Dhpc_run/results_full.json
    python3 cleanup_stale_geometry.py results/3Dhpc_run/results_full.json
    python3 cleanup_stale_geometry.py --dry-run results/2Dhpc_run/results_full.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Case IDs affected by the geometry correction, cross-checked against
# core/cases.py by AST-searching every case-building function for a
# reference to core.het_geometry (or the HET config structures that import
# it) rather than assumed from names.
AFFECTED_CASES = {
    "2D_HET_MMS_SPT100",           # _build_het_2d_mms
    "2D_HET_Sin_MeetingReport",    # uses geom.L_Z, geom.L_R directly
    "3D_HET_MMS_SPT100",           # _build_het_3d_mms
    "3D_HET_RotatingSpoke_SPT100", # _build_het_3d_spoke
    "3D_HET_Discharge_SPT100",     # _build_het_3d_discharge
    # NOT affected (verified): sin_hom, TwoGaussian, SingleMode (2D);
    # TripleSin_cube, Laplace_BCdriven_cube, TwoGaussian_cube,
    # HighMode_n2m3l4 (3D) - none reference core.het_geometry.
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_full_json", type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be removed without writing")
    args = ap.parse_args()

    path = args.results_full_json
    with open(path) as fh:
        rows = json.load(fh)

    kept = [r for r in rows if r["case"] not in AFFECTED_CASES]
    removed = [r for r in rows if r["case"] in AFFECTED_CASES]

    print(f"{path}: {len(rows)} rows total")
    print(f"  removing {len(removed)} rows for affected cases:")
    from collections import Counter
    for case, n in Counter(r["case"] for r in removed).items():
        print(f"    {case}: {n} rows")
    print(f"  keeping {len(kept)} rows for unaffected cases")

    if args.dry_run:
        print("--dry-run: nothing written.")
        return

    with open(path, "w") as fh:
        json.dump(kept, fh, indent=2, default=str)

    csv_path = path.with_name(path.name.replace("results_full.json",
                                                 "results_summary.csv"))
    if csv_path == path:
        print(f"  WARNING: could not derive a distinct CSV path from "
             f"{path.name} (expected a filename containing "
             f"'results_full.json'); CSV not updated - update it by hand "
             f"or rename the file to match the usual convention.")
    elif csv_path.exists() and kept:
        fieldnames = list(kept[0].keys())
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in kept:
                w.writerow(r)
        print(f"  also rewrote {csv_path}")

    print("Operation complete. The extracted configurations are prepared for execution with "
         "--append (ensuring collision-free deployment).")


if __name__ == "__main__":
    main()