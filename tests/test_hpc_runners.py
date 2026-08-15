"""
Tests for the scope and supersession semantics of the HPC drivers.

Why these are worth having
--------------------------
Both behaviours pinned here are data-loss-class: getting them wrong destroys hours
of recorded cluster work rather than merely producing a wrong number, and neither
shows up as an exception.

``--append`` merges the rows already on disk ahead of the rows the current
invocation produces. Without supersession on ``(case, solver, N)`` a wave that
revisits a partially-completed unit writes a *second* row for every solver already
present, and nothing downstream can choose between the two. The 3-D summary was
found holding four such duplicate sets, from steps that re-ran the sub-second Thomas
reference each time they visited a section.

`RunSelection` restricts a run to the rows a gap analysis says are outstanding. A
solver excluded from the scope must be neither executed *nor recorded*: a
placeholder row for a skipped solver would supersede the sound row already on disk,
turning a scope restriction into deletion.

The runner modules are imported for real, so these also serve as import-time smoke
tests for three files that a PBS job is the usual first thing to execute.
"""
from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _quiet_runner_logs(monkeypatch):
    """
    Silence the runners' module loggers for the duration of a test.

    The drivers log to ``results/<sweep>/run.log`` in append mode, which is a real
    provenance artefact of past cluster runs. A test suite must not write into it.
    """
    null = logging.getLogger("test_hpc_runners_null")
    null.handlers = [logging.NullHandler()]
    null.propagate = False
    for module_name in ("hpc.runners.run_1d", "hpc.runners.run_2d",
                        "hpc.runners.run_3d"):
        try:
            module = __import__(module_name, fromlist=["log"])
        except ImportError:                     # pragma: no cover - env-dependent
            continue
        monkeypatch.setattr(module, "log", null, raising=False)


# -- Supersession --------------------------------------------------------------

class TestDedupe2D:
    """`run_2d._dedupe_results` keeps the newest row per identity."""

    @staticmethod
    def _row(module, case="c", solver="HHL", N=32, scheme="fmg", notes="",
             err=0.0):
        return module.RunResult2D(
            case=case, solver=solver, N=N, kappa_row=3.0, max_rel_err=err,
            max_abs_err=0.0, residual=0.0, wall_time_s=1.0, converged=True,
            n_jacobi_iters=1, notes=notes, scheme=scheme)

    def test_rerun_supersedes_the_earlier_row(self):
        """
        Validates that a newer solver run correctly supersedes an older entry for 
        the same (case, solver, N) tuple, ensuring failed or obsolete runs are 
        discarded in favour of their replacements.
        """
        from hpc.runners import run_2d

        rows = [self._row(run_2d, err=99.0, scheme="line-sor (fallback)"),
                self._row(run_2d, err=0.5, scheme="fmg")]
        out = run_2d._dedupe_results(rows)
        assert len(out) == 1
        # The scheme is deliberately excluded from the key, so a row recorded under
        # a fallback is replaced by its successful rerun rather than kept beside it.
        assert out[0].max_rel_err == 0.5
        assert out[0].scheme == "fmg"

    def test_distinct_triples_are_all_kept(self):
        """
        Ensures that runs differing in solver identity or resolution are all 
        retained during deduplication.
        """
        from hpc.runners import run_2d

        rows = [self._row(run_2d, solver="HHL"), self._row(run_2d, solver="VQLS"),
                self._row(run_2d, solver="HHL", N=64)]
        assert len(run_2d._dedupe_results(rows)) == 3

    def test_scheme_comparison_rows_survive_together(self):
        """
        Confirms that rows generated under the scheme comparison study are 
        differentiated by scheme, preventing legitimate comparative data 
        from being deduplicated away.
        """
        from hpc.runners import run_2d

        # --compare-schemes deliberately records several schemes for one triple, so
        # the scheme participates in the key for those rows only.
        rows = [self._row(run_2d),
                self._row(run_2d, notes="scheme_comparison:sor", scheme="sor"),
                self._row(run_2d, notes="scheme_comparison:jacobi",
                          scheme="jacobi")]
        assert len(run_2d._dedupe_results(rows)) == 3

    def test_order_is_first_seen(self):
        """
        Validates that the deduplication process maintains the original row 
        insertion order, replacing values in-place rather than appending.
        """
        from hpc.runners import run_2d

        rows = [self._row(run_2d, solver="HHL"), self._row(run_2d, solver="VQLS"),
                self._row(run_2d, solver="HHL", err=1.0)]
        out = run_2d._dedupe_results(rows)
        assert [r.solver for r in out] == ["HHL", "VQLS"]


class TestDedupe3D:
    """`run_3d._dedupe_results` applies the same rule to the 3-D schema."""

    @staticmethod
    def _row(module, case="c", solver="HHL", N=16, err=0.0):
        return module.RunResult3D(
            case=case, solver=solver, N=N, shape="16x16x16", n_unknowns=4096,
            kappa_row=2.0, max_rel_err=err, max_abs_err=0.0, residual=0.0,
            wall_time_s=1.0, converged=True, n_outer=1)

    def test_rerun_supersedes_the_earlier_row(self):
        """
        Verifies that 3D row deduplication discards older superseded entries 
        for a given problem configuration, aligning with 2D behaviour.
        """
        from hpc.runners import run_3d

        rows = [self._row(run_3d, err=99.0), self._row(run_3d, err=0.4)]
        out = run_3d._dedupe_results(rows)
        assert len(out) == 1 and out[0].max_rel_err == 0.4

    def test_the_observed_duplicate_thomas_rows_collapse(self):
        """
        Confirms that identical reference runs executed across multiple batch 
        steps are correctly collapsed into a single authoritative row.
        """
        from hpc.runners import run_3d

        # Reproduces what was found in results/3Dhpc_run: the same Thomas reference
        # recorded three times by three steps, differing only in wall time.
        rows = [self._row(run_3d, solver="Thomas", err=3.77) for _ in range(3)]
        assert len(run_3d._dedupe_results(rows)) == 1


# -- Scope selection -----------------------------------------------------------

class TestRunSelection:
    """`run_1d.RunSelection` decides what a gap-fill invocation touches."""

    def test_empty_case_filter_selects_everything(self):
        """
        Ensures that an empty run selection filter defaults to an inclusive 
        policy, permitting all cases to execute.
        """
        from hpc.runners.run_1d import RunSelection

        sel = RunSelection()
        assert sel.wants_case("HET_1D_3b_gaussian_Vd300")
        assert sel.wants_case("1D_Poisson_fS_hom")

    def test_substring_filter_selects_one_subcase(self):
        """
        Validates that specifying a substring case filter correctly restricts 
        execution exclusively to problems matching that substring.
        """
        from hpc.runners.run_1d import RunSelection

        # This is the 20-row 1-D wave-1 scope: sub-case 3b and nothing else.
        sel = RunSelection(cases=("3b",))
        assert sel.wants_case("HET_1D_3b_gaussian_Vd300")
        assert not sel.wants_case("HET_1D_3a_linear_hom")
        assert not sel.wants_case("HET_1D_3c_gaussian_NeumannDirichlet")

    def test_filter_is_case_insensitive(self):
        """
        Confirms that case filtering operates case-insensitively, preventing 
        typographical mismatches from excluding valid cases.
        """
        from hpc.runners.run_1d import RunSelection

        assert RunSelection(cases=("HET_1D_3B",)).wants_case(
            "het_1d_3b_gaussian_vd300")

    def test_glob_filter(self):
        """
        Verifies that glob-style patterns are correctly parsed and applied 
        to filter the execution scope.
        """
        from hpc.runners.run_1d import RunSelection

        sel = RunSelection(cases=("1d_poisson_f*_hom",))
        assert sel.wants_case("1D_Poisson_fS_hom")
        assert not sel.wants_case("1D_Poisson_fS_nonhom")

    def test_solver_selection(self):
        """
        Ensures that specifying a solver filter correctly restricts execution 
        to only the requested quantum or classical algorithms.
        """
        from hpc.runners.run_1d import RunSelection

        sel = RunSelection(solvers=("vqls", "qsvt"))
        assert sel.wants_solver("VQLS") and sel.wants_solver("qsvt")
        assert not sel.wants_solver("hhl")

    def test_is_hashable_and_frozen(self):
        """
        Validates that the selection configuration is immutable and hashable, 
        ensuring safe transfer across multiprocessing boundaries.
        """
        from hpc.runners.run_1d import RunSelection

        # Frozen so that it is unambiguously picklable across the
        # ProcessPoolExecutor boundary and no worker can mutate the parent's scope.
        sel = RunSelection(cases=("3b",))
        hash(sel)
        with pytest.raises(Exception):
            sel.cases = ("3a",)


class TestSectionFamilies:
    """The 1-D section labels must resolve to executable work-unit families."""

    def test_labels_match_the_dispatch_table(self):
        """
        Confirms that the section labels defined in the runner match the 
        internal dispatch mapping, preventing runtime family resolution errors.
        """
        from hpc.runners import run_1d

        # The same invariant the runner asserts at dispatch time. Checked here too,
        # because there it is discovered only once a job is already running.
        assert set(run_1d.SECTION_FAMILIES) == {"1", "1b", "2"}
        assert set(run_1d.SECTION_FAMILIES.values()) == {
            "generic_poisson", "generic_poisson_nonhom", "het_1d"}

    def test_every_family_has_a_runner(self):
        """
        Validates that every defined problem family successfully dispatches 
        to an executable work unit routine.
        """
        from hpc.runners import run_1d

        for family in run_1d.SECTION_FAMILIES.values():
            results, solutions = run_1d._execute_work_unit(
                family, 4, run_1d.RunSelection(cases=("__no_such_case__",)), 2)
            # Every case is filtered out, so the unit must complete having recorded
            # nothing - which also proves the family dispatches at all.
            assert results == []
