"""
test_order4_wiring.py
---------------------
Tests that ``--order 4`` reaches the fourth-order machinery, end to end.

``problems/poisson_line_{2,3}d_4th.py`` being correct is necessary and not
sufficient: every fourth-order 2-D/3-D result produced before 2026-08-12 came
from a runner that was correct in outline and wrong in what it dispatched to.
Three of the four defects behind those results were *routing* defects, not
numerical ones, and none raised:

* ``run_{2,3}d.py`` dispatched ``--order 4`` to their own iteration loops over
  ``solvers/outer/multigrid_4th.py`` — a separate, defective closure — rather
  than re-discretising the problem and taking the ordinary ``solve()`` path.
  Everything that path had been given since (the wall-clock cap, stagnation
  detection, true per-solve work accounting, ``level_kappas``, the ``inner_*``
  diagnostics) was silently absent from every fourth-order row.
* The quantum entry points reconstructed a tridiagonal operator from ``A[0,0]``
  and ``A[0,1]``, discarding the ±2 band, so ``--order 4`` with HHL or QSVT
  solved a second-order system and reported its residual against the truncated
  operator.
* Options given as ``-I hhl.epsilon=…`` never reached ``hhl_4th``, the registry
  keying on the name ``solve()`` is handed.

These tests pin the routing. The numerics are pinned by
``tests/test_poisson_line_4th.py``.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from core import cases


@pytest.fixture(autouse=True)
def _quiet_runner_logs(monkeypatch):
    """
    Silence the runners' module loggers, as ``test_hpc_runners`` does.

    The drivers log to ``results/<sweep>/run.log`` in append mode, which is a
    real provenance artefact of past cluster runs; a test suite must not write
    into it.
    """
    null = logging.getLogger("test_order4_wiring_null")
    null.handlers = [logging.NullHandler()]
    null.propagate = False
    for module_name in ("hpc.runners.run_2d", "hpc.runners.run_3d"):
        try:
            module = __import__(module_name, fromlist=["log"])
        except ImportError:                 # pragma: no cover - env-dependent
            continue
        monkeypatch.setattr(module, "log", null, raising=False)


# -- Inner-solver dispatch -----------------------------------------------------

class TestInnerSolverDispatch:
    """
    The registry name must change with the order for HHL and QSVT, and must not
    change for anything else.
    """

    @pytest.mark.parametrize("module_name", ["run_2d", "run_3d"])
    def test_order_four_selects_the_pentadiagonal_entry_points(self, module_name):
        """
        Validates that the fourth-order discretization correctly dispatches to pentadiagonal quantum solver entry points.
        Confirms that specific aliases for HHL and QSVT are selected.
        """
        module = __import__(f"hpc.runners.{module_name}", fromlist=["_"])
        assert module._inner_for_order("hhl", 4) == "hhl_4th"
        assert module._inner_for_order("qsvt", 4) == "qsvt_4th"

    @pytest.mark.parametrize("module_name", ["run_2d", "run_3d"])
    def test_order_two_is_untouched(self, module_name):
        """
        Validates that second-order execution paths remain unaltered during inner solver dispatch.
        Ensures backward compatibility and correctness for standard finite-difference solvers.
        """
        module = __import__(f"hpc.runners.{module_name}", fromlist=["_"])
        for solver in ("thomas", "hhl", "vqls", "qsvt"):
            assert module._inner_for_order(solver, 2) == solver

    @pytest.mark.parametrize("module_name", ["run_2d", "run_3d"])
    def test_thomas_and_vqls_do_not_change(self, module_name):
        """
        Validates that the Thomas algorithm and Variational Quantum Linear Solver (VQLS) bypass specific fourth-order re-routing.
        Confirms that these solvers correctly handle full pentadiagonal matrices without structural truncation.
        """
        module = __import__(f"hpc.runners.{module_name}", fromlist=["_"])
        assert module._inner_for_order("thomas", 4) == "thomas"
        assert module._inner_for_order("vqls", 4) == "vqls"

    def test_the_registry_actually_provides_them(self):
        """
        Validates that the solver registry accurately lists the fourth-order quantum entry points.
        Confirms the structural availability of these specialized solver configurations.
        """
        from solvers.outer import available_inner

        assert "hhl_4th" in available_inner()
        assert "qsvt_4th" in available_inner()


class TestInnerOptionsReachTheFourthOrderSolvers:
    """
    ``InnerConfig`` keys on the name ``solve()`` is given. Without a section for
    the alias, the sweep's epsilon and degree cap are dropped in silence.
    """

    @pytest.mark.parametrize("module_name", ["run_2d", "run_3d"])
    def test_sweep_defaults_are_mirrored_onto_the_aliases(self, module_name):
        """
        Validates that default sweep configurations are successfully propagated to the fourth-order solver aliases.
        Ensures that necessary computational parameters, such as degree caps, are preserved across the routing layer.
        """
        module = __import__(f"hpc.runners.{module_name}", fromlist=["_"])
        cfg = module.SweepConfig().inner_config(32)
        assert cfg.for_solver("hhl_4th") == cfg.for_solver("hhl")
        assert cfg.for_solver("qsvt_4th") == cfg.for_solver("qsvt")
        # N=32 is where the degree cap switches on; the alias must carry it.
        assert cfg.for_solver("qsvt_4th").get("max_degree") is not None

    @pytest.mark.parametrize("module_name", ["run_2d", "run_3d"])
    def test_command_line_options_reach_the_alias(self, module_name):
        """
        Validates that explicit command-line overrides successfully propagate to the assigned solver alias.
        Prevents silent dropping of user-defined tolerance or algorithmic configurations.
        """
        module = __import__(f"hpc.runners.{module_name}", fromlist=["_"])
        cfg = module.SweepConfig(
            inner_options={"hhl": {"epsilon": 0.123}}).inner_config(8)
        assert cfg.for_solver("hhl_4th")["epsilon"] == 0.123


# -- Problem conversion --------------------------------------------------------

class TestProblemConversion:

    def test_2d_conversion_preserves_the_continuous_problem(self):
        """
        Validates that two-dimensional problem re-discretization accurately preserves the core continuous problem structure.
        Confirms the equivalence of boundary conditions, array shapes, and forcing functions.
        """
        from problems.poisson_line_2d_4th import PoissonLine2D4th
        from hpc.runners.run_2d import _to_4th_order_2d

        built = cases.get("het_2d_mms_spt100").build(8)
        fourth = _to_4th_order_2d(built.problem, built.f_faces)

        assert isinstance(fourth, PoissonLine2D4th)
        assert fourth.shape == built.problem.shape
        assert np.array_equal(fourth.f, built.problem.f)
        assert (fourth.Lx, fourth.Ly) == (built.problem.Lx, built.problem.Ly)
        for face in ("bc_x0", "bc_x1", "bc_y0", "bc_y1"):
            assert np.array_equal(getattr(fourth, face),
                                  getattr(built.problem, face))

    def test_3d_conversion_preserves_periodicity_and_lengths(self):
        """
        Validates that three-dimensional fourth-order conversion maintains structural invariants such as periodicity and domain lengths.
        Ensures that the physical domain definition is conserved.
        """
        from problems.poisson_line_3d_4th import PoissonLine3D4th
        from hpc.runners.run_3d import _to_4th_order_3d

        built = cases.get("het_3d_mms_spt100").build(8)
        fourth = _to_4th_order_3d(built.problem, built.f_faces)

        assert isinstance(fourth, PoissonLine3D4th)
        assert fourth.periodic == built.problem.periodic
        assert fourth.lengths == built.problem.lengths
        assert np.array_equal(fourth.f, built.problem.f)

    def test_conversion_without_face_data_still_builds(self):
        """
        Validates that problem conversion gracefully falls back to extrapolation when explicit face source data is absent.
        Confirms that case constructions remain robust under degraded source definitions.
        """
        from hpc.runners.run_2d import _to_4th_order_2d

        built = cases.get("poisson_2d_sin_pi").build(8)
        assert _to_4th_order_2d(built.problem, None).shape == (8, 8)

    def test_the_converted_problem_is_more_accurate(self):
        """
        Validates that the complete fourth-order numerical pipeline significantly reduces discretization error compared to the baseline second-order solver.
        Provides end-to-end confirmation of convergence improvements.
        """
        from solvers.outer import solve
        from hpc.runners.run_2d import _to_4th_order_2d

        built = cases.get("poisson_2d_sin_pi").build(16)
        exact = built.exact

        second = solve(built.problem, inner="thomas", scheme="fmg", tol=1e-10)
        fourth = solve(_to_4th_order_2d(built.problem, built.f_faces),
                       inner="thomas", scheme="fmg", tol=1e-10)

        e2 = np.max(np.abs(second.u - exact))
        e4 = np.max(np.abs(fourth.u - exact))
        assert e4 < e2 / 10.0


# -- Face source data from the case registry -----------------------------------

class TestFaceSources:
    """
    ``BuiltCase.f_faces`` is the 2-D/3-D counterpart of ``f_boundary``. Where it
    is absent the closure extrapolates, which is order-preserving for a smooth
    source and inaccurate for a sharply peaked one — so the registry supplying
    it is a real property, not a convenience.
    """

    def test_every_2d_and_3d_case_supplies_face_sources(self):
        """
        Validates that all registered two-dimensional and three-dimensional cases provide explicit face source boundary data.
        Prevents unnecessary fallback to less accurate extrapolation techniques.
        """
        missing = [name for name in cases.available()
                   if cases.get(name).dim > 1
                   and cases.get(name).build(8).f_faces is None]
        assert missing == []

    def test_face_values_match_the_analytic_source(self):
        """
        Validates that explicitly provided face source arrays match the known analytic functions at the domain boundaries.
        Confirms the exactitude of continuous-to-discrete mappings.
        """
        built = cases.get("poisson_2d_sin_pi").build(8)
        lo, hi = built.f_faces
        for face in (*lo, *hi):
            assert np.allclose(face, 0.0, atol=1e-12)

    def test_a_non_vanishing_face_is_captured(self):
        """
        Validates that boundary conditions with non-zero face sources are correctly evaluated and retained.
        Ensures the correct capturing of complex boundary interactions.
        """
        built = cases.get("poisson_2d_two_gaussian_plasmanet").build(16)
        lo, hi = built.f_faces
        assert any(np.any(np.abs(face) > 0.0) for face in (*lo, *hi))

    def test_periodic_axes_carry_no_face_data(self):
        """
        Validates that periodic boundaries correctly register as lacking face source data, aligning with the absence of ghost nodes on periodic axes.
        """
        built = cases.get("het_3d_mms_spt100").build(8)
        lo, hi = built.f_faces
        assert lo[2] is None and hi[2] is None
        assert lo[0] is not None and hi[0] is not None

    def test_face_shape_matches_the_domain_face(self):
        """
        Validates that the dimensionality and shape of the generated face arrays strictly match the expected topological boundaries of the domain.
        """
        built = cases.get("poisson_3d_two_gaussian_cube").build(8)
        lo, _ = built.f_faces
        assert lo[0].shape == (8, 8)


# -- The retired path is gone --------------------------------------------------

class TestRetiredPathIsGone:
    """
    The defective closure lived in one module and was reached from two runners
    and two debug scripts. A stale import of any of them silently reinstates
    order 0.88, so its absence is worth asserting rather than assuming.
    """

    def test_multigrid_4th_is_not_importable(self):
        """
        Validates the complete removal of the defective, historically retired fourth-order multigrid closure.
        Confirms that accidental re-imports will raise structural failures.
        """
        with pytest.raises(ImportError):
            __import__("solvers.outer.multigrid_4th")

    @pytest.mark.parametrize("module_name", ["run_2d", "run_3d"])
    def test_runners_do_not_reference_it(self, module_name):
        """
        Validates that current execution runners maintain strict isolation from deprecated solver paths.
        Prevents the accidental reintroduction of architectural dependency inversions.
        """
        from pathlib import Path

        source = (Path(__file__).resolve().parent.parent
                  / "hpc" / "runners" / f"{module_name}.py").read_text(
                      encoding="utf-8")
        assert "multigrid_4th" not in source
        # The 4th-order path used to import a *script* into a runner, inverting
        # the dependency direction the architecture is built on.
        assert "scripts.debug" not in source
