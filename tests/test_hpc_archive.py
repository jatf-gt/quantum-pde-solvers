"""
Tests for the legacy HPC sweep schema contract, `benchmark/hpc_archive.py`.

This module was `benchmark/results_io.py` until the professional benchmarking
framework introduced a second, differently-shaped archive under that name. The
two are distinguished in `benchmark/hpc_archive.py`'s docstring; the one tested
here is the read-only, dimension-aware reader for the sweep directories that
already exist on disk, and it is what `scripts/utils/gap_analysis.py` builds the rerun
manifests from.

Two things are worth guarding here, and neither is exercised by any other test.

The first is the field-alias resolution. The same physical quantity is archived
as `u_solver` in 1D, `phi_solver` in 2D and `phi` in 3D, and before this module
existed the reader spelled those names at seventeen separate call sites. A
reader that silently fails to find a field produces an empty figure rather than
an error, so the resolution is asserted directly.

The second is that importing this module must not drag in a plotting stack.
`benchmark/hpc_plotting.py` defers its Matplotlib import so that the Agg backend
is never forced on a process wanting an interactive one; that protection is
worthless if the module it imports for loading pulls Matplotlib itself.

The fixtures build sweep directories on disk rather than mocking, since the
contract under test *is* the on-disk layout.
"""
import json
import subprocess
import sys

import numpy as np
import pytest

from benchmark import hpc_archive as rio


# -- Fixtures ------------------------------------------------------------------

@pytest.fixture
def sweep_1d(tmp_path):
    """A minimal 1D sweep directory: two solvers at two resolutions."""
    rows = []
    for solver in ("Thomas", "HHL"):
        for N in (4, 8):
            rows.append({
                "case": "case_a", "solver": solver, "N": N,
                "max_rel_err": 1.0 / N, "max_abs_err": 1.0 / N,
                "residual": 1e-12, "wall_time_s": 0.5,
                "converged": True, "notes": "",
                "rel_l2_err": 1.0 / N, "rms_err": 1.0 / N,
            })
            rio.save_solution(
                tmp_path, "case_a", solver, N, dim=1,
                x=np.linspace(0, 1, N), u_solver=np.zeros(N),
                u_exact=np.ones(N),
            )
    rio.save_summary(tmp_path, rows)
    return tmp_path


@pytest.fixture
def sweep_3d(tmp_path):
    """A minimal 3D sweep, which uses a different stem and field names."""
    rows = [{
        "case": "cube", "solver": "Thomas", "N": 4,
        "max_rel_err": 0.1, "max_abs_err": 0.1, "residual": 1e-10,
        "wall_time_s": 1.0, "converged": True, "notes": "",
        "rel_l2_err": 0.1, "rms_err": 0.1,
    }]
    rio.save_solution(tmp_path, "cube", "Thomas", 4, dim=3,
                      phi=np.zeros((4, 4, 4)), phi_exact=np.ones((4, 4, 4)))
    rio.save_summary(tmp_path, rows)
    return tmp_path


# -- Import Hygiene ------------------------------------------------------------

def test_importing_hpc_archive_does_not_pull_matplotlib():
    """
    Validates that importing the `hpc_archive` module does not inadvertently
    load Matplotlib. Ensures the plotting stack is not prematurely pulled into 
    processes requiring lighter dependencies. Checked in a subprocess to avoid
    state contamination from prior imports.
    """
    code = (
        "import sys; import benchmark.hpc_archive; "
        "print('matplotlib' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_importing_hpc_plotting_does_not_pull_matplotlib():
    """
    Confirms that the `hpc_plotting` orchestration module defers its Matplotlib 
    imports, protecting environments where an interactive backend is undesirable.
    """
    code = (
        "import sys; import benchmark.hpc_plotting; "
        "print('matplotlib' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# -- Filename Convention -------------------------------------------------------

class TestFilenames:

    def test_1d_and_2d_share_a_stem(self):
        """
        Ensures that 1D and 2D archives use the identical naming stem, verifying 
        that differentiation is strictly contextual (by directory). Records this 
        as an explicit schema fact, a deliberate one.
        """
        assert (rio.solution_filename("c", "HHL", 8, dim=1)
                == rio.solution_filename("c", "HHL", 8, dim=2))

    def test_3d_has_its_own_stem(self):
        """
        Validates that 3D archives are allocated a distinct naming stem, separating 
        them from the 1D/2D naming convention.
        """
        assert rio.solution_filename("c", "HHL", 8, dim=3).startswith("solution3d_")

    def test_filename_round_trips_through_save(self, tmp_path):
        """
        Confirms that paths constructed for saving solutions correctly match the 
        expected schema filenames. Verifies physical creation on disk.
        """
        path = rio.save_solution(tmp_path, "c", "HHL", 8, dim=2, u=np.zeros(3))
        assert path.name == rio.solution_filename("c", "HHL", 8, dim=2)
        assert path.exists()

    def test_unknown_dimension_is_rejected(self):
        """
        Ensures that providing an unsupported physical dimension raises a KeyError, 
        preventing malformed file paths from being generated.
        """
        with pytest.raises(KeyError):
            rio.solution_filename("c", "HHL", 8, dim=4)


# -- Field Aliases -------------------------------------------------------------

class TestFieldAliases:

    def test_resolves_the_1d_spelling(self):
        """
        Validates that field alias resolution accurately extracts the 1D-specific 
        keys (`u_solver`, `u_exact`) mapped to the common semantic names.
        """
        data = {"u_solver": np.arange(3), "u_exact": np.ones(3)}
        assert rio.field(data, "solution").tolist() == [0, 1, 2]
        assert rio.field(data, "exact").tolist() == [1, 1, 1]

    def test_resolves_the_2d_spelling(self):
        """
        Validates that field alias resolution accurately extracts the 2D-specific 
        keys (`phi_solver`, `phi_exact`) mapped to the common semantic names.
        """
        data = {"phi_solver": np.arange(3), "phi_exact": np.ones(3)}
        assert rio.field(data, "solution").tolist() == [0, 1, 2]
        assert rio.field(data, "exact").tolist() == [1, 1, 1]

    def test_resolves_the_3d_spelling(self):
        """
        Validates that field alias resolution accurately extracts the 3D-specific 
        key (`phi`) mapped to the common semantic name.
        """
        assert rio.field({"phi": np.arange(3)}, "solution").tolist() == [0, 1, 2]

    def test_preference_order_prefers_the_native_name(self):
        """
        Confirms that when multiple alias keys are present, the original 1D 
        spelling takes precedence. Ensures backward compatibility for 2D archives 
        parsed by 1D-era loaders.
        """
        data = {"u_solver": np.zeros(3), "phi_solver": np.ones(3)}
        assert rio.field(data, "solution").tolist() == [0, 0, 0]

    def test_absent_field_returns_none(self):
        """
        Ensures that querying an unrepresented semantic field returns None, rather 
        than raising an error, when the data dictionary is structurally sound.
        """
        assert rio.field({"x": np.zeros(3)}, "exact") is None

    def test_unknown_field_name_raises(self):
        """
        Validates that querying a completely unknown semantic name raises a KeyError. 
        Protects against typographical errors masking as absent data.
        """
        with pytest.raises(KeyError, match="Unknown field"):
            rio.field({"u_solver": np.zeros(3)}, "solutoin")


class TestRowFields:

    def test_missing_common_field_raises(self):
        """
        Confirms that attempting to extract a mandatory common row field raises a 
        KeyError if it is absent from the underlying dictionary.
        """
        with pytest.raises(KeyError, match="common field"):
            rio.row_field({"solver": "HHL"}, "case")

    def test_missing_dimension_specific_field_returns_default(self):
        """
        Ensures that querying a dimension-specific field (e.g. `scheme` in 1D rows) 
        returns the supplied default value instead of failing.
        """
        assert rio.row_field({"case": "c"}, "scheme", "n/a") == "n/a"


# -- Sweep Archive -------------------------------------------------------------

class TestSweepArchive:

    def test_reads_rows_and_solutions(self, sweep_1d):
        """
        Validates that `SweepArchive` successfully reconstructs row data and 
        loads solution arrays from a valid on-disk hierarchy.
        """
        sw = rio.SweepArchive(sweep_1d, dim=1, plots_subdir=None)
        rows = sw.rows()
        assert len(rows) == 4
        sol = sw.solution("case_a", "HHL", 8)
        assert rio.field(sol, "solution").shape == (8,)

    def test_absent_solution_returns_none(self, sweep_1d):
        """
        Ensures that requesting a solution which has not been archived returns None, 
        maintaining robustness in incomplete sweeps.
        """
        sw = rio.SweepArchive(sweep_1d, dim=1, plots_subdir=None)
        assert sw.solution("case_a", "QSVT", 8) is None

    def test_absent_summary_exits_rather_than_tracebacks(self, tmp_path):
        """
        Confirms that reading from an archive lacking a summary file yields a clean 
        SystemExit. Provides expected behaviour for wall-time-killed jobs.
        """
        sw = rio.SweepArchive(tmp_path, dim=1)
        with pytest.raises(SystemExit, match="No results found"):
            sw.rows()

    def test_missing_reports_gaps(self, sweep_1d):
        """
        Validates the gap-detection logic by confirming it accurately identifies 
        unconverged rows lacking complete physical data.
        """
        sw = rio.SweepArchive(sweep_1d, dim=1, plots_subdir=None)
        rows = sw.rows()
        assert sw.missing(rows) == []

        rows.append({
            "case": "case_a", "solver": "QSVT", "N": 16,
            "max_rel_err": None, "max_abs_err": None, "residual": None,
            "wall_time_s": 0.0, "converged": False, "notes": "solver_error",
            "rel_l2_err": None, "rms_err": None,
        })
        assert sw.missing(rows) == [("case_a", "QSVT", 16)]

    def test_wrong_dimension_makes_every_archive_missing(self, sweep_1d):
        """
        Ensures that an archive parsed with incorrect dimensionality correctly 
        registers every data file as missing, signalling misconfiguration 
        rather than raising obscure errors.
        """
        sw = rio.SweepArchive(sweep_1d, dim=3)
        rows = sw.rows()
        assert len(sw.missing(rows)) == len(rows)

    def test_metadata_absence_is_not_an_error(self, sweep_1d):
        """
        Verifies that metadata absence is tolerated cleanly, as provenance files 
        are supplementary and not strictly required for data interpretation.
        """
        assert rio.SweepArchive(sweep_1d, dim=1).metadata() == {}

    def test_metadata_round_trips(self, sweep_1d):
        """
        Validates that written metadata JSON files are correctly parsed back 
        into python dictionaries by the reader.
        """
        (sweep_1d / "run_metadata.json").write_text(json.dumps({"dimension": 1}))
        assert rio.SweepArchive(sweep_1d, dim=1).metadata()["dimension"] == 1

    def test_plots_dir_defaults_to_a_subdirectory(self, sweep_1d):
        """
        Ensures the plots directory correctly resolves to the expected nested 
        'plots' path for modern (>=2D) layouts.
        """
        assert rio.SweepArchive(sweep_1d, dim=2).plots_dir.name == "plots"

    def test_plots_dir_can_be_the_results_dir(self, sweep_1d):
        """
        Confirms that 1D layouts correctly collocate figures alongside the 
        numerical results when explicitly configured.
        """
        sw = rio.SweepArchive(sweep_1d, dim=1, plots_subdir=None)
        assert sw.plots_dir == sw.results_dir


class TestGrouping:

    def test_group_by_case_solver_sorts_by_resolution(self, sweep_1d):
        """
        Validates that grouping results by problem case and solver correctly 
        orders the sub-elements monotonically by resolution (N).
        """
        sw = rio.SweepArchive(sweep_1d, dim=1, plots_subdir=None)
        grouped = sw.group_by_case_solver(sw.rows())
        assert [r["N"] for r in grouped[("case_a", "HHL")]] == [4, 8]

    def test_group_by_case_N_sorts_solvers_canonically(self, sweep_1d):
        """
        Validates that grouping results by problem case and resolution uses the 
        canonical solver ordering.
        """
        sw = rio.SweepArchive(sweep_1d, dim=1, plots_subdir=None)
        grouped = sw.group_by_case_N(sw.rows())
        assert [r["solver"] for r in grouped[("case_a", 4)]] == ["Thomas", "HHL"]

    def test_series_yields_lexical_order_not_canonical(self, sweep_1d):
        """
        Ensures that `series` yields items in lexical (alphabetical) order, 
        preserving backwards compatibility for legend construction in older plots.
        """
        sw = rio.SweepArchive(sweep_1d, dim=1, plots_subdir=None)
        grouped = sw.group_by_case_solver(sw.rows())
        assert [s for s, _ in sw.series(grouped, "case_a")] == ["HHL", "Thomas"]

    def test_series_can_exclude_a_solver(self, sweep_1d):
        """
        Confirms that the `series` iterator can successfully filter out explicitly 
        excluded solvers.
        """
        sw = rio.SweepArchive(sweep_1d, dim=1, plots_subdir=None)
        grouped = sw.group_by_case_solver(sw.rows())
        got = [s for s, _ in sw.series(grouped, "case_a", exclude=("Thomas",))]
        assert got == ["HHL"]

    def test_scheme_comparison_rows_are_droppable(self, tmp_path):
        """
        Verifies that rows relating to alternative schemes are dropped from standard 
        analyses to prevent spurious duplicates in core solver comparisons.
        """
        rows = [
            {"case": "c", "solver": "HHL", "N": 4, "max_rel_err": 0.1,
             "max_abs_err": 0.1, "residual": 0.0, "wall_time_s": 1.0,
             "converged": True, "notes": "", "rel_l2_err": 0.1, "rms_err": 0.1},
            {"case": "c", "solver": "HHL", "N": 4, "max_rel_err": 0.2,
             "max_abs_err": 0.2, "residual": 0.0, "wall_time_s": 1.0,
             "converged": True, "notes": "scheme_comparison:sor",
             "rel_l2_err": 0.2, "rms_err": 0.2},
        ]
        rio.save_summary(tmp_path, rows)
        sw = rio.SweepArchive(tmp_path, dim=2, skip_scheme_comparison=True)
        assert len(sw.group_by_case_solver(sw.rows())[("c", "HHL")]) == 1

        keep_all = rio.SweepArchive(tmp_path, dim=2)
        assert len(keep_all.group_by_case_solver(keep_all.rows())[("c", "HHL")]) == 2


class TestRoundTrip:
    """The writing helpers must produce what the reading helpers expect."""

    def test_summary_round_trips(self, tmp_path):
        """
        Verifies that writing a sweep summary to disk and subsequently parsing 
        it faithfully reproduces the exact dictionary sequence.
        """
        rows = [{"case": "c", "solver": "HHL", "N": 4, "max_rel_err": 0.5,
                 "max_abs_err": 0.5, "residual": 1e-9, "wall_time_s": 2.0,
                 "converged": True, "notes": "", "rel_l2_err": 0.5,
                 "rms_err": 0.5}]
        rio.save_summary(tmp_path, rows)
        assert rio.SweepArchive(tmp_path, dim=1).rows() == rows
        assert (tmp_path / "results_summary.csv").exists()

    def test_solution_round_trips_across_dimensions(self, tmp_path, sweep_3d):
        """
        Validates that writing and reading multi-dimensional numeric fields 
        preserves exact shapes and correctly integrates with gap reporting.
        """
        sw = rio.SweepArchive(sweep_3d, dim=3)
        sol = sw.solution("cube", "Thomas", 4)
        assert rio.field(sol, "solution").shape == (4, 4, 4)
        assert rio.field(sol, "exact").shape == (4, 4, 4)
        assert sw.missing(sw.rows()) == []
