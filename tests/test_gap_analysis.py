"""
Tests for `scripts/gap_analysis.py`, the tool that decides what an HPC sweep owes.

Why these are worth having
--------------------------
This module's output drives resubmission, and both of its possible errors are
expensive in opposite directions. Scheduling a sound row wastes cluster time — single
rows in this archive cost 25-38 h. Omitting a genuinely absent row leaves a hole in
the published results. The classification policy has already had to be corrected
three times against real data (stagnation, error magnitude, and per-solve timeouts),
each time because a defensible-looking rule destroyed or withheld real work.

The tests therefore pin the *policy decisions*, not the formatting: which outcomes
are grounds to recompute, and which are merely notable.
"""
from __future__ import annotations

import pytest

from scripts.gap_analysis import (LEGACY_HHL_TIMEOUT_S, classify_row,
                                  STALE_GEOMETRY_CASES)


def _row(**kwargs) -> dict:
    """
    Build a minimal sound row, overridden by `kwargs`.

    Returns
    -------
    dict
        A row that `classify_row` accepts with no reasons, so that each test
        perturbs exactly one property.
    """
    row = {
        "case": "3D_Poisson_TripleSin_cube",
        "solver": "HHL",
        "N": 8,
        "converged": True,
        "stop_reason": "tol_met",
        "notes": "",
        "wall_time_s": 120.0,
        "err_vs_thomas": 0.5,
        "max_rel_err": 0.5,
    }
    row.update(kwargs)
    return row


class TestSoundRows:
    """A row with nothing wrong with it must never be scheduled."""

    def test_clean_row_has_no_reasons(self):
        reasons, flags = classify_row(_row())
        assert reasons == []

    def test_thomas_is_never_judged_on_error_magnitude(self):
        # Thomas is the reference; a large error against the exact solution is
        # discretisation error shared by every solver on that mesh.
        reasons, _ = classify_row(
            _row(solver="Thomas", err_vs_thomas=None, max_rel_err=45.0))
        assert reasons == []


class TestSoftOutcomes:
    """Outcomes recorded for visibility that must not trigger recomputation."""

    def test_stagnation_is_a_flag_not_a_reason(self):
        # The outer schemes detect stagnation precisely so a quantum solver at its
        # inner-solver noise floor stops rather than burning futile strip solves.
        reasons, flags = classify_row(
            _row(stop_reason="stagnated", converged=False))
        assert reasons == []
        assert "stagnated" in flags

    def test_large_error_is_a_flag_not_a_reason(self):
        # In 1-D the operator reaches kappa ~ 1.7e3 by N=64; HHL's degradation with
        # kappa is what the benchmark measures, not a defect to be recomputed.
        reasons, flags = classify_row(_row(err_vs_thomas=250.0))
        assert reasons == []
        assert any(f.startswith("large_error") for f in flags)

    def test_strict_escalates_flags_into_reasons(self):
        reasons, flags = classify_row(
            _row(stop_reason="stagnated", converged=False), strict=True)
        assert "stagnated" in reasons
        assert flags == []


class TestTruncationIsAReason:
    """Outcomes that genuinely leave the field unfinished."""

    def test_wall_time_exceeded_is_a_reason(self):
        reasons, _ = classify_row(
            _row(stop_reason="wall_time_exceeded", converged=False))
        assert "wall_time_exceeded" in reasons

    def test_exception_is_a_reason(self):
        reasons, _ = classify_row(
            _row(notes="ModuleNotFoundError: No module named 'x'",
                 converged=False, err_vs_thomas=None, max_rel_err=None))
        assert "solver_error" in reasons

    def test_missing_error_metric_is_a_reason(self):
        reasons, _ = classify_row(
            _row(err_vs_thomas=None, max_rel_err=None, wall_time_s=12.0))
        assert "no_error_metric" in reasons


class TestPerSolveTimeout:
    """
    A solve that reached its own budget is a measurement, not a fault.

    `hpc/runners/run_1d.py::_run_hhl` imposes a hard per-solve budget. At N=32 and
    N=64 the 1-D operator's condition number reaches ~1.7e3 and the statevector
    simulation does not finish inside it. Thirteen such rows exist, every one with
    ``wall_time_s = 3600.2``; classifying them as errors scheduled 13 h of cluster
    time to reproduce thirteen identical timeouts.
    """

    def test_explicit_timeout_marker_is_a_flag(self):
        reasons, flags = classify_row(
            _row(notes="rel_vs_thomas;hhl_timeout:3600s", converged=False,
                 err_vs_thomas=None, max_rel_err=None,
                 wall_time_s=LEGACY_HHL_TIMEOUT_S), dim=1)
        assert reasons == []
        assert "solver_timeout" in flags

    def test_a_raised_budget_is_recognised_without_the_legacy_inference(self):
        """
        A row produced at a raised ``--hhl-timeout-s`` must classify on its marker.

        The budget is expected to be raised — at 3600 s HHL does not complete for
        N>=32, so finding where it does means increasing it. Such a row's wall time
        bears no relation to `LEGACY_HHL_TIMEOUT_S`, and in 2-D/3-D the inference is
        switched off entirely, so the explicit marker must carry the classification
        on its own.
        """
        reasons, flags = classify_row(
            _row(notes="hhl_timeout:21600s", converged=False, err_vs_thomas=None,
                 max_rel_err=None, wall_time_s=21600.4), dim=2)
        assert reasons == []
        assert "solver_timeout" in flags

    def test_legacy_1d_row_is_inferred_from_wall_time(self):
        # Rows written before the marker existed carry only "solver_error"; the sole
        # surviving evidence is a wall time at the budget.
        reasons, flags = classify_row(
            _row(case="1D_Poisson_fS_hom", notes="solver_error", converged=False,
                 err_vs_thomas=None, max_rel_err=None, wall_time_s=3600.2), dim=1)
        assert reasons == []
        assert "solver_timeout" in flags

    def test_a_genuine_1d_failure_inside_the_budget_is_still_a_reason(self):
        # HET_1D_3c HHL at N=32 died after 743 s, well inside the budget. That is a
        # real failure and must remain scheduled.
        reasons, _ = classify_row(
            _row(case="HET_1D_3c_gaussian_NeumannDirichlet", notes="solver_error",
                 converged=False, err_vs_thomas=None, max_rel_err=None,
                 wall_time_s=742.9), dim=1)
        assert "solver_error" in reasons

    @pytest.mark.parametrize("dim", [2, 3])
    def test_the_wall_time_inference_does_not_apply_in_2d_or_3d(self, dim):
        """
        The inference is valid only for the 1-D schema.

        In 1-D ``wall_time_s`` is one solve; in 2-D and 3-D it is a whole outer
        iteration over N (or N²) strip solves, which routinely exceeds an hour in a
        perfectly sound run. Applying the inference there would mark most large-N
        HHL rows as timed out and suppress the reasons that schedule them.
        """
        reasons, flags = classify_row(
            _row(notes="solver_error", converged=False, err_vs_thomas=None,
                 max_rel_err=None, wall_time_s=20000.0), dim=dim)
        assert "solver_timeout" not in flags
        assert "solver_error" in reasons


class TestStaleGeometry:
    """The SPT-100 correction invalidates specific cases and only those."""

    def test_a_stale_case_is_always_a_reason(self):
        reasons, _ = classify_row(_row(case="3D_HET_MMS_SPT100"))
        assert "stale_geometry" in reasons

    def test_unaffected_het_cases_are_absent_from_the_policy_set(self):
        # check_geometry_impact.py --dim 1 proves that only 3b moves: the 1-D
        # operator is the dimensionless TST matrix and the *_scaled family
        # normalises L out. Listing another one would force needless recomputation.
        assert "HET_1D_3a_linear_hom" not in STALE_GEOMETRY_CASES
        assert "HET_1D_3c_gaussian_NeumannDirichlet" not in STALE_GEOMETRY_CASES
        assert "HET_1D_3b_gaussian_Vd300" in STALE_GEOMETRY_CASES


def test_legacy_budget_describes_the_existing_archive():
    """
    `LEGACY_HHL_TIMEOUT_S` is a fact about recorded data, not a mirror of the runner.

    It must NOT be pinned to `run_1d.HHL_TIMEOUT_S`: that value is a
    ``--hhl-timeout-s`` default and is expected to be raised, whereas the thirteen
    rows this constant classifies were all produced at 3600 s and stay that way
    forever. Coupling them would mean that raising the runtime budget silently
    reclassified those rows as genuine errors and scheduled 13 h of recomputation.

    What is checked instead is the historical claim itself, against the archive.
    """
    import json
    from pathlib import Path

    assert LEGACY_HHL_TIMEOUT_S == 3600.0

    summary = Path("results/1Dhpc_run/results_full.json")
    if not summary.exists():                # pragma: no cover - archive not present
        pytest.skip("1D archive not present in this checkout")
    rows = json.loads(summary.read_text(encoding="utf-8"))
    timed_out = [r for r in rows
                 if r["solver"] == "HHL" and (r.get("wall_time_s") or 0) >= 3600.0]
    assert timed_out, "no timed-out HHL rows found; the constant's premise is gone"
    for row in timed_out:
        # Every one sits just above the budget, never far beyond it - which is what
        # makes the wall-clock inference sound for these rows.
        assert 3600.0 <= row["wall_time_s"] < 3660.0, row
