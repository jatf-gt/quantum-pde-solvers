"""
Quantifies which registered cases are numerically altered by a change to the
shared SPT-100 geometry.

Purpose
-------
The channel geometry in `core/het_geometry.py` was corrected against the primary
source (Boeuf & Garrigues 1998) in commit ``861ff46``: the axial length L_z was
revised 25 mm → 40 mm and the inner radius R_in 35 mm → 30 mm. Every HET result
computed before that commit is therefore suspect, and the naive remedy — recompute
the entire HET sweep — costs hours of cluster time for cases whose assembled system
the correction never touches.

This module answers the question exactly rather than conservatively. It builds each
case twice, once under the current constants and once under the superseded ones,
and compares the assembled operator A, the right-hand side b, the sampled source f
and the reference solution element-wise. A case whose four arrays are bit-identical
is provably unaffected and must not be recomputed.

Method
------
The case registry reads the geometry constants at *registration* time, not at build
time, so patching `core.het_geometry` in place is insufficient — `core.cases` must be
reloaded so that its module-level ``register`` calls re-evaluate the lengths against
the patched values. The comparison is run in a single process in that order:
current geometry first, then patched, since reloading is one-directional.

Interpretation
--------------
A nonzero relative difference in ``b`` or ``f`` with ``A`` unchanged is the expected
signature of a source term positioned against a physical length: the operator is the
dimensionless TST matrix in 1-D and so is geometry-independent, whilst a source such
as a Gaussian sited a fixed fraction along the channel moves with L_z. Cases in the
non-dimensional ``*_scaled`` family normalise L out and are correspondingly immune.

A difference in ``A`` — expected in 2-D and 3-D, where the strip operator absorbs a
1/dy² diagonal shift and therefore depends on the aspect ratio — additionally
invalidates any cached QSVT phase angles keyed on that case's condition number.

Notes
-----
The comparison uses an exact (zero-tolerance) test rather than `np.allclose`. The
question is whether the correction reaches the case at all, not whether the two
results agree to some engineering tolerance, so any nonzero difference counts.
"""
from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ``pytest.ini`` sets ``pythonpath = .``, but a bare ``python3 scripts/check_geometry_impact.py``
# puts ``scripts/`` on ``sys.path[0]`` rather than the repository root, so the local
# imports below fail however sound the working directory. Resolving the root from
# ``__file__`` makes the invocation location irrelevant, matching what every module
# under ``hpc/runners/`` already does.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import core.cases as cases          # noqa: E402
import core.het_geometry as geom    # noqa: E402

# -- Superseded geometry -------------------------------------------------------
# The values in force before commit 861ff46. R_OUT was already correct at 50 mm;
# L_R, R_MEAN and L_S are derived and so are recomputed rather than listed.

PRE_CORRECTION: dict[str, float] = {
    "L_Z":   0.025,     # Axial channel length [m], was drawn from secondary sources
    "R_IN":  0.035,     # Inner channel radius [m]
    "R_OUT": 0.050,     # Outer channel radius [m], unchanged by the correction
}

COMPARED_FIELDS: tuple[str, ...] = ("A", "row", "h", "b", "f", "exact")
"""
Arrays compared per case.

``A`` is the assembled system matrix, populated in 1-D only. ``row`` is the strip
operator of the 2-D/3-D line-decomposed problems, which is where an aspect-ratio
change registers: it absorbs a 1/dy² diagonal shift, so κ(A_row) depends on L_z/L_r
even when the source is unchanged. Comparing ``b``/``f``/``exact`` alone would
therefore miss precisely the effect that invalidates a cached QSVT phase angle.
``h`` is the mesh spacing per axis, which changes with any domain length.
"""

NOISE_TOLERANCE: float = 1e-12
"""
Relative difference below which two arrays are treated as identical.

Rebuilding a case re-evaluates its source expression, so differences at the level
of floating-point round-off (~1e-16) appear even where the geometry is irrelevant
to the result. Anything at or below this threshold is round-off, not physics; a
genuine geometry effect is O(0.1) or larger in every case observed.
"""


# -- Case selection ------------------------------------------------------------

def het_case_ids(dim: Optional[int] = None) -> list[str]:
    """
    Identifiers of every registered HET case, optionally restricted by dimension.

    Parameters
    ----------
    dim : int, optional
        Spatial dimension to select. None returns every dimension.

    Returns
    -------
    list of str
        Case identifiers, sorted.
    """
    ids = []
    for case_id in cases.available():
        try:
            spec = cases.get(case_id)
        except Exception:                                        # noqa: BLE001
            continue
        if "het" not in case_id.lower():
            continue
        if dim is not None and getattr(spec, "dim", None) != dim:
            continue
        ids.append(case_id)
    return sorted(ids)


def _snapshot(case_ids: list[str], N: int) -> dict[str, dict]:
    """
    Builds each case at resolution N and retains the arrays under comparison.

    Parameters
    ----------
    case_ids : list of str
        Cases to build.
    N : int
        Resolution, a power of two.

    Returns
    -------
    dict
        Keyed by case identifier; each value holds the compared arrays, or an
        ``error`` entry when the case could not be built. A build failure is
        recorded rather than raised so that one unbuildable case does not mask the
        verdict for every other.
    """
    out: dict[str, dict] = {}
    for case_id in case_ids:
        try:
            built = cases.get(case_id).build(N)
            problem = getattr(built, "problem", None)
            row = (problem.row_matrix()
                   if problem is not None and hasattr(problem, "row_matrix")
                   else None)
            out[case_id] = {
                "A":     _as_array(built.A),
                "row":   _as_array(row),
                "h":     _as_array(built.spacings),
                "b":     _as_array(getattr(built, "b", None)),
                "f":     _as_array(built.f_values),
                "exact": _as_array(built.exact),
            }
        except Exception as exc:                                 # noqa: BLE001
            out[case_id] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def _as_array(value) -> Optional[np.ndarray]:
    """Returns a float copy of `value`, or None when the field is absent."""
    return None if value is None else np.array(value, dtype=float)


def _relative_difference(before: Optional[np.ndarray],
                         after:  Optional[np.ndarray]) -> Optional[float]:
    """
    Maximum relative difference between two arrays.

    Parameters
    ----------
    before, after : np.ndarray or None
        Arrays under comparison, of identical shape.

    Returns
    -------
    float or None
        Max |before - after| normalised by max|after|; None when the field is
        absent from both; ``inf`` when present in only one, or when the shapes
        disagree, since either constitutes a change.
    """
    if before is None and after is None:
        return None
    if before is None or after is None or before.shape != after.shape:
        return math.inf
    denominator = max(float(np.abs(after).max()), 1e-300)
    return float(np.abs(before - after).max() / denominator)


# -- Geometry patching ---------------------------------------------------------

def apply_pre_correction_geometry() -> None:
    """
    Reverts `core.het_geometry` to its pre-861ff46 state and re-registers cases.

    Mutates the module in place and reloads `core.cases`, because the registry
    evaluates the geometry constants when the module is executed rather than when a
    case is built. The operation is one-directional within a process: restoring the
    corrected values would require a second reload, which the caller does not need
    since the corrected snapshot is always taken first.
    """
    geom.L_Z   = PRE_CORRECTION["L_Z"]
    geom.R_IN  = PRE_CORRECTION["R_IN"]
    geom.R_OUT = PRE_CORRECTION["R_OUT"]
    geom.L_R   = geom.R_OUT - geom.R_IN
    geom.R_MEAN = 0.5 * (geom.R_IN + geom.R_OUT)
    geom.L_S    = 2.0 * math.pi * geom.R_MEAN
    geom.ASPECT_RADIAL_AZIMUTHAL = geom.L_R / geom.L_S
    importlib.reload(cases)


# -- Reporting -----------------------------------------------------------------

def compare(case_ids: list[str], N: int) -> tuple[list[str], list[str]]:
    """
    Builds every case under both geometries and reports the element-wise verdict.

    Parameters
    ----------
    case_ids : list of str
        Cases to compare.
    N : int
        Resolution at which to build.

    Returns
    -------
    tuple of (list of str, list of str)
        (unchanged, changed) case identifiers.
    """
    print(f"  Corrected geometry : L_Z={geom.L_Z:.4f} m  R_IN={geom.R_IN:.4f} m  "
          f"R_OUT={geom.R_OUT:.4f} m  L_R={geom.L_R:.4f} m")
    corrected = _snapshot(case_ids, N)

    apply_pre_correction_geometry()
    print(f"  Superseded geometry: L_Z={geom.L_Z:.4f} m  R_IN={geom.R_IN:.4f} m  "
          f"R_OUT={geom.R_OUT:.4f} m  L_R={geom.L_R:.4f} m")
    superseded = _snapshot(case_ids, N)
    print()

    header = (f"  {'Case':<42}" + "".join(f"{f:>12}" for f in COMPARED_FIELDS)
              + "   Verdict")
    print(header)
    print("  " + "-" * (len(header) - 2))

    unchanged, changed = [], []
    for case_id in case_ids:
        before, after = superseded[case_id], corrected[case_id]
        if "error" in before or "error" in after:
            message = before.get("error") or after.get("error")
            print(f"  {case_id:<42}{message[:55]}")
            continue

        cells, differs = [], False
        for field_name in COMPARED_FIELDS:
            rel = _relative_difference(before[field_name], after[field_name])
            if rel is None:
                cells.append("-")
            elif rel == 0.0:
                cells.append("same")
            elif rel <= NOISE_TOLERANCE:
                cells.append("~same")
            else:
                cells.append(f"{rel:.2e}")
                differs = True

        (changed if differs else unchanged).append(case_id)
        verdict = "CHANGED - rerun" if differs else "identical - keep"
        print(f"  {case_id:<42}" + "".join(f"{c:>12}" for c in cells)
              + f"   {verdict}")

    return unchanged, changed


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Report which HET cases the SPT-100 geometry correction alters.")
    parser.add_argument("--dim", type=int, choices=(1, 2, 3), default=None,
                        help="Restrict to one spatial dimension (default: all).")
    parser.add_argument("--N", type=int, default=16,
                        help="Resolution at which to build each case (default: 16).")
    args = parser.parse_args()

    case_ids = het_case_ids(args.dim)
    scope = "all dimensions" if args.dim is None else f"{args.dim}D"
    print()
    print(f"  HET geometry impact - {scope}, N={args.N}, "
          f"{len(case_ids)} case(s)")
    print()

    if not case_ids:
        print("  No HET cases registered for that selection.")
        return

    unchanged, changed = compare(case_ids, args.N)

    print()
    print(f"  Unaffected (no rerun required): {len(unchanged)}")
    print(f"  Altered    (rerun required)   : {len(changed)}")
    for case_id in changed:
        print(f"      {case_id}")
    print()


if __name__ == "__main__":
    main()
