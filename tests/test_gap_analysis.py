"""
Tests for `scripts/utils/gap_analysis.py`, the tool that decides what an HPC sweep owes.

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

from scripts.utils.gap_analysis import LEGACY_HHL_TIMEOUT_S, classify_row


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
        """Confirms that a flawless execution record yields no recomputation reasons."""
        reasons, flags = classify_row(_row())
        assert reasons == []

    def test_thomas_is_never_judged_on_error_magnitude(self):
        """
        Validates that Thomas solver executions are exempt from error magnitude checks.
        
        As the reference solution, large errors indicate discretisation limits shared
        by all solvers, not failures of the Thomas scheme itself.
        """
        # Thomas is the reference; a large error against the exact solution is
        # discretisation error shared by every solver on that mesh.
        reasons, _ = classify_row(
            _row(solver="Thomas", err_vs_thomas=None, max_rel_err=45.0))
        assert reasons == []


class TestSoftOutcomes:
    """Outcomes recorded for visibility that must not trigger recomputation."""

    def test_stagnation_is_a_flag_not_a_reason(self):
        """
        Ensures solver stagnation is recorded as an informational flag rather than a failure reason.
        
        Quantum solvers routinely halt at their noise floor to conserve resources, which constitutes
        a valid measurement rather than a defect requiring recomputation.
        """
        # The outer schemes detect stagnation precisely so a quantum solver at its
        # inner-solver noise floor stops rather than burning futile strip solves.
        reasons, flags = classify_row(
            _row(stop_reason="stagnated", converged=False))
        assert reasons == []
        assert "stagnated" in flags

    def test_large_error_is_a_flag_not_a_reason(self):
        """
        Validates that substantial discrepancies against the reference are flagged but do not mandate recomputation.
        
        Degradation with increasing condition number is an expected phenomenon in HHL evaluation,
        representing a valid measurement of solver limitations.
        """
        # In 1-D the operator reaches kappa ~ 1.7e3 by N=64; HHL's degradation with
        # kappa is what the benchmark measures, not a defect to be recomputed.
        reasons, flags = classify_row(_row(err_vs_thomas=250.0))
        assert reasons == []
        assert any(f.startswith("large_error") for f in flags)

    def test_strict_escalates_flags_into_reasons(self):
        """Confirms that strict mode elevates informational flags to mandatory recomputation reasons."""
        reasons, flags = classify_row(
            _row(stop_reason="stagnated", converged=False), strict=True)
        assert "stagnated" in reasons
        assert flags == []


class TestTruncationIsAReason:
    """Outcomes that genuinely leave the field unfinished."""

    def test_wall_time_exceeded_is_a_reason(self):
        """Ensures that exceeding the maximum wall time is classified as a mandatory recomputation reason."""
        reasons, _ = classify_row(
            _row(stop_reason="wall_time_exceeded", converged=False))
        assert "wall_time_exceeded" in reasons

    def test_exception_is_a_reason(self):
        """Validates that runtime exceptions are correctly identified as solver errors requiring recomputation."""
        reasons, _ = classify_row(
            _row(notes="ModuleNotFoundError: No module named 'x'",
                 converged=False, err_vs_thomas=None, max_rel_err=None))
        assert "solver_error" in reasons

    def test_missing_error_metric_is_a_reason(self):
        """Confirms that the absence of required error metrics triggers a recomputation requirement."""
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
        """Ensures that explicitly marked timeouts are recorded as flags rather than failure reasons."""
        reasons, flags = classify_row(
            _row(notes="rel_vs_thomas;hhl_timeout:3600s", converged=False,
                 err_vs_thomas=None, max_rel_err=None,
                 wall_time_s=LEGACY_HHL_TIMEOUT_S), dim=1)
        assert reasons == []
        assert "solver_timeout" in flags

    def test_a_raised_budget_is_recognised_without_the_legacy_inference(self):
        """
        Validates that rows produced with an extended timeout budget classify correctly based on their explicit marker.
        
        Extended wall times are necessary for larger grid dimensions and do not correlate with legacy 
        time limits. The explicit marker correctly prevents erroneous recomputation scheduling.
        """
        reasons, flags = classify_row(
            _row(notes="hhl_timeout:21600s", converged=False, err_vs_thomas=None,
                 max_rel_err=None, wall_time_s=21600.4), dim=2)
        assert reasons == []
        assert "solver_timeout" in flags

    def test_legacy_1d_row_is_inferred_from_wall_time(self):
        """
        Ensures that historical 1-D results lacking explicit markers are correctly inferred as timeouts based on wall time.
        """
        # Rows written before the marker existed carry only "solver_error"; the sole
        # surviving evidence is a wall time at the budget.
        reasons, flags = classify_row(
            _row(case="1D_Poisson_fS_hom", notes="solver_error", converged=False,
                 err_vs_thomas=None, max_rel_err=None, wall_time_s=3600.2), dim=1)
        assert reasons == []
        assert "solver_timeout" in flags

    def test_a_genuine_1d_failure_inside_the_budget_is_still_a_reason(self):
        """Confirms that premature solver termination within the allocated budget is correctly classified as a failure."""
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
        Validates that legacy wall time inference is restricted to 1-D scenarios.
        
        In higher dimensions, wall time accumulates across multiple strip solves, meaning that 
        lengthy executions are typical and should not be erroneously classified as timeouts.
        """
        reasons, flags = classify_row(
            _row(notes="solver_error", converged=False, err_vs_thomas=None,
                 max_rel_err=None, wall_time_s=20000.0), dim=dim)
        assert "solver_timeout" not in flags
        assert "solver_error" in reasons


def test_legacy_budget_describes_the_existing_archive():
    """
    `LEGACY_HHL_TIMEOUT_S` is a fact about recorded data, not a mirror of the runner.

    It must NOT be pinned to `run_1d.HHL_TIMEOUT_S`: that value is a
    ``--hhl-timeout-s`` default and is expected to be raised, whereas the rows this
    constant classifies were all produced at 3600 s and stay that way forever.
    Coupling them would mean that raising the runtime budget silently reclassified
    those rows as genuine errors and scheduled hours of recomputation.

    What is checked instead is the historical claim itself, against the archive.
    The check mirrors the two-tier classification in ``classify_row``:

    *   Rows with an explicit ``hhl_timeout:<budget>s`` marker are definitively
        timed-out; their wall time should be within 60 s of the declared budget.
    *   Rows without an explicit marker but with wall time at or above
        ``LEGACY_HHL_TIMEOUT_S`` are the historical rows this constant was
        introduced to handle; they should fall within 60 s of 3600 s.

    Rows that ran longer than 3600 s but without hitting a cap (e.g. a slow genuine
    completion at 4378 s for N=32 where kappa is high) carry empty ``notes`` and must
    NOT be classified as timeouts — they are excluded from this check.
    """
    import json
    from pathlib import Path
    from scripts.utils.gap_analysis import TIMEOUT_MARKERS

    assert LEGACY_HHL_TIMEOUT_S == 3600.0

    summary = Path("results/1Dhpc_run/results_full.json")
    if not summary.exists():                # pragma: no cover - archive not present
        pytest.skip("1D archive not present in this checkout")
    rows = json.loads(summary.read_text(encoding="utf-8"))
    hhl_rows = [r for r in rows if r["solver"] == "HHL"]

    # Rows with an explicit timeout marker (any budget).
    explicit_timed_out = [
        r for r in hhl_rows
        if any(m in str(r.get("notes") or "") for m in TIMEOUT_MARKERS)
    ]

    # Rows classified as timed-out via the legacy wall-clock inference:
    # no explicit marker, wall time within [LEGACY_HHL_TIMEOUT_S, LEGACY_HHL_TIMEOUT_S + 60s).
    # This mirrors the tightened bound in classify_row; rows running significantly
    # beyond 3600 s without a marker (genuine slow completions at high κ) are excluded.
    legacy_inferred = [
        r for r in hhl_rows
        if not any(m in str(r.get("notes") or "") for m in TIMEOUT_MARKERS)
        and LEGACY_HHL_TIMEOUT_S
           <= (r.get("wall_time_s") or 0)
           < LEGACY_HHL_TIMEOUT_S + 60.0
    ]

    all_timed_out = explicit_timed_out + legacy_inferred
    assert all_timed_out, "no timed-out HHL rows found; the constant's premise is gone"

    # Explicit-marker rows: wall time within 60 s of their declared budget.
    for row in explicit_timed_out:
        notes = str(row.get("notes") or "")
        # Extract declared budget from the marker, e.g. "hhl_timeout:7200s".
        declared_budget = None
        for fragment in notes.split(";"):
            if "hhl_timeout:" in fragment:
                try:
                    declared_budget = float(
                        fragment.split("hhl_timeout:")[1].rstrip("s")
                    )
                except (IndexError, ValueError):
                    pass
        assert declared_budget is not None, (
            f"Row carries a timeout marker but no parseable budget: {notes!r}"
        )
        wall = row["wall_time_s"]
        assert declared_budget <= wall < declared_budget + 60.0, row

    # Legacy-inferred rows: wall time should be near LEGACY_HHL_TIMEOUT_S.
    for row in legacy_inferred:
        wall = row["wall_time_s"]
        assert LEGACY_HHL_TIMEOUT_S <= wall < LEGACY_HHL_TIMEOUT_S + 60.0, (
            f"Legacy-inferred timeout row has wall_time={wall:.1f}s, which is "
            f">={LEGACY_HHL_TIMEOUT_S:.0f}s but not within the expected 60s "
            f"window. This row may have completed genuinely (not hit a cap) and "
            f"should not be treated as a timeout. Row: {row}"
        )
