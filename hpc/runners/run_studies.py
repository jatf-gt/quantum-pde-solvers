#!/usr/bin/env python3
"""
Equal-accuracy and one-at-a-time sensitivity studies for the 1-D solvers.

Purpose
-------
The primary sweep (`hpc/runners/run_1d.py`) evaluates each solver at exactly one
parameter setting — HHL at ε = 0.01, VQLS at n_layers = max(6, 2n+2) with
n_restarts = max(3, 2n), QSVT at the degree cap `run_1d.qsvt_max_degree` returns.
One setting yields one point per curve, so neither of the studies below can be
recovered from it by interpolation: both must re-solve across a parameter grid.
This module is that driver.

  Equal accuracy (`benchmark/equal_accuracy.py`)
      A comparison at nominally equal precision parameters is unsound, because
      the three algorithms' precision knobs do not mean the same thing: the VQLS
      cost C bounds the residual only as C ≥ r²/κ², HHL's ε is coupled to the
      Trotter count as n_T = ⌈1/ε⌉, and the QSVT residual is non-monotone in
      polynomial degree through the oscillatory Chebyshev error. The protocol
      instead sweeps each solver's own knob until its residual lands in a band
      about a common target r_target, and reports the RESOURCE COST there. That
      is the comparison the thesis needs: cost at matched accuracy, not accuracy
      at matched nominal parameters.

  Sensitivity (`benchmark/sensitivity.py`)
      One-at-a-time variation about a fixed baseline, giving one curve per
      parameter. Chosen over a full factorial design because K parameters at M
      values cost K·M solves rather than M^K, and because the resulting curves
      are separately interpretable in a thesis figure.

Scope and cost
--------------
Both studies re-solve at every grid point, so their cost is a multiple of a
primary-sweep row: roughly 10 solves per case for HHL (5 equal-accuracy ε values
plus 5 sensitivity), 34 for VQLS (a 5×4 layer/restart grid plus 14 OAT points)
and 15 for QSVT. Run at ONE modest resolution across a few representative cases
— these characterise the algorithms, not the mesh — rather than across the full
sweep grid, where they would cost more than the benchmark they annotate.

The default scope is therefore N = 8 over three cases spanning the accuracy
range: a smooth sinusoid, a discontinuous source, and the HET application case.

Outputs
-------
Written through `benchmark/results_io.SweepArchive`, which is the schema for this
new material — NOT `benchmark/hpc_archive.py`, which is the read-only contract
fixed by the existing `results/{1,2,3}Dhpc_run/` sweeps and must not acquire
new-format files. Products, under ``results/1Dstudies/``:

  equal_accuracy.json      One record per (case, N, solver).
  sensitivity_<solver>.json   One record per (case, N, parameter).
  run_metadata.json        Environment, git state and the resolved grids.

The directory name carries the dimension, the discretisation order and the run
tag — `results/2Dstudies`, `results/2Dstudies_4th`, `results/3Dstudies_grid_fix`
— since none of the three is recoverable from the filenames within it. See
`results_dir_for`, and `--results-dir` to override.

References
----------
Bravo-Prieto, C., LaRose, R., Cerezo, M., Subasi, Y., Cincio, L. & Coles, P. J.
    (2023). Variational Quantum Linear Solver. Quantum, 7, 1188.
Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D.,
    Saisana, M. & Tarantola, S. (2008). Global Sensitivity Analysis: The Primer.
    Wiley.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# `pytest.ini` sets `pythonpath = .`, but a bare `python3 hpc/runners/run_studies.py`
# puts `hpc/runners/` on sys.path[0] rather than the repository root. Resolving the
# root from `__file__` decouples the import path from the invocation directory,
# consistent with every other module under `hpc/runners/`.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmark import equal_accuracy as ea                      # noqa: E402
from benchmark import sensitivity as sens                       # noqa: E402
from benchmark.results_io import SweepArchive                   # noqa: E402
from core import cases                                          # noqa: E402

# -- Output directory and logging ----------------------------------------------

RESULTS_DIR = Path("results") / "1Dstudies"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  pid=%(process)-6d %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.FileHandler(RESULTS_DIR / "studies.log", encoding="utf-8"),
              logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("studies")

# -- Default scope -------------------------------------------------------------
# Three cases spanning the accuracy range the primary sweep exhibits: a smooth
# sinusoid where every solver does well, a discontinuous source that is the
# designed stress case, and the HET application case whose right-hand side is
# physically scaled (‖b‖ ~ 700) and therefore exercises the proportionality
# recovery. Adding cases multiplies the cost linearly and buys little: these
# studies characterise the ALGORITHMS, not the case.
DEFAULT_CASES: tuple[str, ...] = (
    "poisson_1d_fS_hom",
    "poisson_1d_fH_hom",
    "het_1d_3a_linear",
)

# 2-D and 3-D scope. Two cases each rather than three, and one resolution: an
# outer iteration performs N (or N²) inner solves, so a single grid point here
# costs what an entire 1-D sweep does. One smooth manufactured case and one HET
# application case per dimension is enough to show whether the parameter response
# carries over from 1-D, which is the question these studies answer in 2-D/3-D.
DEFAULT_CASES_2D: tuple[str, ...] = (
    "poisson_2d_sin_pi",
    "het_2d_mms_spt100",
)
DEFAULT_CASES_3D: tuple[str, ...] = (
    "poisson_3d_triple_sin_cube",
    "het_3d_mms_spt100",
)

DEFAULT_CASES_BY_DIM: dict[int, tuple[str, ...]] = {
    1: DEFAULT_CASES, 2: DEFAULT_CASES_2D, 3: DEFAULT_CASES_3D,
}

# Default resolution per dimension.
#
# N=8 in 2-D and 3-D is the SMALLEST the default scheme admits, not merely a
# choice: FMG needs at least two grid levels, and a 4×4 (or 4×4×4) problem cannot
# be coarsened, so N=4 raises before any solve. Dropping to N=4 requires
# `--scheme sor`, which is a different outer iteration from the one the primary
# sweep records and would make the studies non-comparable with it. N=8 with FMG
# is preferred over N=4 with SOR for that reason.
DEFAULT_N_BY_DIM: dict[int, tuple[int, ...]] = {1: (8,), 2: (8,), 3: (8,)}

# Output directory per dimension, so a 2-D study cannot overwrite a 1-D one.
RESULTS_DIR_BY_DIM: dict[int, str] = {
    1: "1Dstudies", 2: "2Dstudies", 3: "3Dstudies",
}


def results_dir_for(dim: int, order: int, run_tag: str = "") -> Path:
    """
    Resolve the output directory for one studies invocation.

    The name carries the dimension, the discretisation order and the run tag,
    because none of the three is recorded in a filename: `write_sensitivity`
    overwrites `sensitivity_<solver>.json` wholesale, `append_equal_accuracy`
    merges on `(case_id, solver, N)` — a key that omits the order — and
    `run_metadata.json` is written before the first case. Two submissions
    differing only in `--order` that share a directory therefore destroy each
    other's records and cross-stamp each other's metadata, which is what
    happened to the second-order `grid_fix` pair of 2026-08-19.

    The fourth-order suffix is `_4th`, matching the directory names
    `hpc/runners/make_tables.py::STUDY_DIRS` already resolves for `--order 4`.

    Parameters
    ----------
    dim : int
        Spatial dimension, 1, 2 or 3.
    order : int
        Discretisation order, 2 or 4.
    run_tag : str, optional
        Short identifier for the invocation; appended when non-empty.

    Returns
    -------
    Path
        Directory under `results/`, e.g. `results/2Dstudies_4th_o4`.
    """
    name = RESULTS_DIR_BY_DIM[dim]
    if order != 2:
        name += f"_{order}th"
    if run_tag:
        name += f"_{run_tag}"
    return Path("results") / name

# HPC case identifiers, matching what run_1d.py records, so a studies row can be
# joined to its primary-sweep counterpart without a lookup table.
CASE_ID: dict[str, str] = {
    "poisson_1d_fS_hom":  "1D_Poisson_fS_hom",
    "poisson_1d_fH_hom":  "1D_Poisson_fH_hom",
    "poisson_1d_fL_hom":  "1D_Poisson_fL_hom",
    "poisson_1d_fS_nonhom": "1D_Poisson_fS_nonhom",
    "het_1d_3a_linear":   "HET_1D_3a_linear_hom",
    "het_1d_3b_gaussian_Vd300": "HET_1D_3b_gaussian_Vd300",
    "poisson_2d_sin_pi":  "2D_Poisson_sin_hom",
    "poisson_2d_single_mode_n1m1": "2D_Poisson_SingleMode_n1m1",
    "het_2d_mms_spt100":  "2D_HET_MMS_SPT100",
    "poisson_3d_triple_sin_cube": "3D_Poisson_TripleSin_cube",
    "het_3d_mms_spt100":  "3D_HET_MMS_SPT100",
}

# N = 8 by default. Large enough that κ (32.2 at order 2) is not degenerate and
# the solvers are genuinely separated, small enough that a 34-solve VQLS grid remains
# computationally tractable within a single HPC job allocation. The studies answer a
# question about parameters, and the parameter response does not change qualitatively
# with resolution.
DEFAULT_N: tuple[int, ...] = (8,)

SOLVERS: tuple[str, ...] = ("hhl", "vqls", "qsvt")

# Sub-case 3c is excluded unconditionally, at either order. Its quantum solves
# return ~100 % error at every N — the recovered proportionality constant diverges
# with N and rescaling still leaves ~2/3 of the error — so a parameter study over
# it would characterise a defect rather than an algorithm. See the note in
# docs/HPC_REPAIR_PLAN.md.
EXCLUDED_CASES: frozenset[str] = frozenset({"het_1d_3c_neumann"})


# -- Case assembly -------------------------------------------------------------

def _build(case_key: str, N: int, order: int) -> dict:
    """
    Assemble one case and its classical reference.

    Both study modules take ``(A, b, u_thomas, u_exact)`` plus the case metadata
    they record on every row, so this returns exactly that bundle. The Thomas
    solution is computed here rather than inside the study modules because both
    need it and it is the same solve.

    Parameters
    ----------
    case_key : str
        Registry key in `core.cases`.
    N : int
        Resolution; a power of two, as the amplitude encoding requires.
    order : {2, 4}
        Spatial discretisation order.

    Returns
    -------
    dict
        Keyword arguments common to every study entry point.

    Raises
    ------
    ValueError
        If the case has no analytical solution and no classical reference can be
        formed, leaving the accuracy metrics undefined.
    """
    built = cases.get(case_key).build(N)
    A = np.asarray(built.A, dtype=float)
    b = np.asarray(built.b, dtype=float)

    if order == 4:
        from problems.poisson_1d_4th import PoissonProblem1D4th

        # f_boundary carries f(0) and f(1), which the corrected 4th-order closure
        # requires as data: the ghost reflection needs u''|face, and in 1-D the PDE
        # supplies it as f|face. Falling back to extrapolation is order-preserving
        # but not accurate enough for the physically scaled HET sources.
        fb = getattr(built, "f_boundary", None)
        prob = PoissonProblem1D4th(
            N=N, f_vals=np.asarray(built.f_values, dtype=float),
            alpha=0.0, beta=0.0,
            **({"f_boundary": fb} if fb is not None else {}),
        )
        A, b = np.asarray(prob.A, dtype=float), np.asarray(prob.b, dtype=float)

    # The classical reference is obtained by a direct dense solve rather than
    # through `solvers.classical.thomas`, whose entry point takes an assembled
    # PoissonProblem1D and not an (A, b) pair, and whose elimination is valid only
    # for a tridiagonal operator — at order 4 the matrix is pentadiagonal. At the
    # resolutions these studies run at, a dense solve is microseconds and is exact
    # to round-off, which is all the reference has to be: it is the accuracy datum,
    # never a timing competitor.
    u_thomas = np.linalg.solve(A, b)
    u_exact = None if built.exact is None else np.asarray(built.exact, dtype=float)

    kappa = float(np.linalg.cond(A))
    return {
        "A": A, "b": b, "u_thomas": u_thomas, "u_exact": u_exact,
        "case_id": CASE_ID.get(case_key, case_key),
        "N": N, "kappa": kappa,
        "source_fn": case_key,
        "alpha_bc": 0.0, "beta_bc": 0.0,
        "discretisation_order": order,
    }


def _build_outer(case_key: str, N: int, order: int, scheme: str,
                 scheme_options: dict) -> dict:
    """
    Assemble one 2-D or 3-D case and its classical reference field.

    The reference is obtained from an outer solve with ``inner="thomas"`` rather
    than from a direct classical solve of a global operator. There is no global
    operator to solve: `solvers/outer` decomposes the domain into strips and
    couples them iteratively. Using the same scheme, tolerance and iteration
    count for the reference is also what makes the comparison isolate the INNER
    solver — any difference attributable to the outer iteration is common to both
    and cancels.

    Parameters
    ----------
    case_key : str
        Registry key in `core.cases`.
    N : int
        Resolution per direction.
    order : {2, 4}
        Spatial discretisation order.
    scheme : str
        Outer scheme, e.g. "fmg".
    scheme_options : dict
        Forwarded to `solve`, e.g. ``max_wall_s``.

    Returns
    -------
    dict
        Keyword arguments common to both outer study entry points.

    Raises
    ------
    RuntimeError
        If the classical reference solve does not converge, leaving nothing
        trustworthy to measure the quantum solves against.
    """
    from solvers.outer import solve

    built = cases.get(case_key).build(N)
    problem = built.problem

    if order == 4:
        # Re-discretisation is delegated to the runners' own helpers rather than
        # repeated here. The 4th-order closure needs the face source samples
        # (`f_faces`) because the ghost reflection requires ∂²u/∂n² on the face,
        # which in more than one dimension is NOT f alone but f minus the
        # tangential curvature of the Dirichlet data. A second implementation of
        # that is precisely the drift these studies must not introduce, since
        # their whole purpose is to be comparable with the primary sweep.
        is_3d = len(getattr(problem, "shape", (0, 0))) == 3
        if is_3d:
            from hpc.runners.run_3d import _to_4th_order_3d as _to_4th
        else:
            from hpc.runners.run_2d import _to_4th_order_2d as _to_4th
        problem = _to_4th(problem, getattr(built, "f_faces", None))

    ref = solve(problem, inner="thomas", scheme=scheme, **scheme_options)
    if not ref.converged:
        raise RuntimeError(
            f"classical reference did not converge for {case_key} at N={N} "
            f"(stop_reason={ref.stop_reason}); the quantum rows would have "
            f"nothing sound to be measured against.")

    exact = None if built.exact is None else np.asarray(built.exact, dtype=float)
    return {
        "problem": problem,
        "u_thomas": np.asarray(ref.u, dtype=float),
        "u_exact": exact,
        "case_id": CASE_ID.get(case_key, case_key),
        "N": N,
        "kappa": float(problem.kappa_row()),
        "source_fn": case_key,
        "discretisation_order": order,
        "scheme": scheme,
        "scheme_options": scheme_options,
    }


# -- QSVT degree grids ---------------------------------------------------------

def _qsvt_degree_grid(kappa: float, epsilon: float, declared: list) -> list:
    """
    Restrict a QSVT degree grid to the caps that actually change the polynomial.

    `benchmark/equal_accuracy.py` and `benchmark/sensitivity.py` declare absolute
    degree grids — 500 to 5000 plus an uncapped entry — chosen against the large-κ
    end of the 1-D sweep. Applied unfiltered at small κ they are both useless and
    ruinously expensive, because a cap ABOVE the natural degree is inactive:
    `qsp_angles._target_reduced_coefs` truncates the Chebyshev expansion of 1/x to
    the requested degree, and beyond the natural degree the discarded coefficients
    are already at round-off, so the polynomial is unchanged while the Newton solve
    still runs at the full requested degree at ~O(d^2.5).

    Concretely, at N=4 the operator has κ ≈ 9.5 and a natural degree near 845; the
    declared grid asks for 5000 and 2000 as well, neither cached, each costing tens
    of minutes to hours to produce a polynomial indistinguishable from the uncapped
    one. Measured: the unfiltered grid did not complete a single case in 9 minutes.

    Entries at or below the natural degree are retained, because that is the
    direction the study is actually interrogating — how accuracy degrades as the
    polynomial is truncated below what κ demands. The uncapped entry is always
    kept: it is the reference point the capped ones are compared against.

    Parameters
    ----------
    kappa : float
        Condition number of the operator under study.
    epsilon : float
        Target approximation error, fixing the natural degree with κ.
    declared : list
        The grid as declared in the benchmark module, possibly containing None.

    Returns
    -------
    list
        The retained subset, ordered as declared, always including None.
    """
    from solvers.quantum.qsp_angles import polynomial_degree_estimate

    natural = polynomial_degree_estimate(kappa, epsilon)
    kept = [d for d in declared if d is None or d <= natural]
    if not any(d is None for d in kept):
        kept = [None] + kept
    dropped = [d for d in declared if d is not None and d > natural]
    if dropped:
        log.info("    QSVT degree grid: dropped %s (above the natural degree %d "
                 "at kappa=%.2f, so inactive but not free)",
                 ", ".join(str(d) for d in dropped), natural, kappa)
    return kept


# -- Studies -------------------------------------------------------------------

def run_equal_accuracy(bundle: dict, solvers: tuple[str, ...],
                       r_target: float) -> list:
    """
    Equal-accuracy sweep over the requested solvers for one assembled case.

    Each solver's own precision knob is swept and the setting whose residual
    falls closest to `r_target` is selected, so that the reported resource cost
    is the cost AT MATCHED ACCURACY rather than at a matched nominal parameter.

    A solver that raises is logged and skipped rather than allowed to abort the
    study: the three sweeps are independent, and losing HHL must not discard the
    VQLS and QSVT results computed alongside it.

    Parameters
    ----------
    bundle : dict
        As returned by `_build`.
    solvers : tuple of str
        Subset of ('hhl', 'vqls', 'qsvt').
    r_target : float
        Common residual target.

    Returns
    -------
    list of EqualAccuracyResult
        One entry per solver that completed.
    """
    entry = {
        "hhl":  ea.sweep_hhl_equal_accuracy,
        "vqls": ea.sweep_vqls_equal_accuracy,
        "qsvt": ea.sweep_qsvt_equal_accuracy,
    }
    out = []
    for name in solvers:
        extra: dict = {}
        if name == "qsvt":
            extra["max_degree_grid"] = _qsvt_degree_grid(
                bundle["kappa"], sens.QSVT_BASELINE["epsilon"],
                ea.QSVT_MAXDEGREE_GRID)
        t0 = time.perf_counter()
        try:
            res = entry[name](r_target=r_target, **extra, **bundle)
        except Exception as exc:
            log.warning("    equal-accuracy %s FAILED: %s", name.upper(), exc)
            continue
        out.append(res)
        log.info("    equal-accuracy %-5s in_band=%-5s calls=%-3d  %.1fs",
                 name.upper(), res.in_band, res.n_solver_calls,
                 time.perf_counter() - t0)
    return out


def run_sensitivity(bundle: dict, solvers: tuple[str, ...]) -> dict[str, list]:
    """
    One-at-a-time sensitivity sweeps over the requested solvers for one case.

    Every parameter declared in the corresponding ``*_SENSITIVITY_GRIDS`` mapping
    is swept about the module's baseline, one at a time. The grids are read from
    `benchmark/sensitivity.py` rather than restated here, so that adding a
    parameter there is picked up without an edit in this driver.

    Parameters
    ----------
    bundle : dict
        As returned by `_build`.
    solvers : tuple of str
        Subset of ('hhl', 'vqls', 'qsvt').

    Returns
    -------
    dict
        Solver name -> list of SensitivitySweepResult, one per parameter.
    """
    entry = {
        "hhl":  (sens.sensitivity_sweep_hhl,  sens.HHL_SENSITIVITY_GRIDS),
        "vqls": (sens.sensitivity_sweep_vqls, sens.VQLS_SENSITIVITY_GRIDS),
        "qsvt": (sens.sensitivity_sweep_qsvt, sens.QSVT_SENSITIVITY_GRIDS),
    }
    out: dict[str, list] = {}
    for name in solvers:
        fn, grids = entry[name]
        for param in grids:
            extra: dict = {}
            if name == "qsvt" and param == "max_degree":
                extra["param_values"] = _qsvt_degree_grid(
                    bundle["kappa"], sens.QSVT_BASELINE["epsilon"], grids[param])
            t0 = time.perf_counter()
            try:
                res = fn(param_name=param, **extra, **bundle)
            except Exception as exc:
                log.warning("    sensitivity %s/%s FAILED: %s",
                            name.upper(), param, exc)
                continue
            out.setdefault(name, []).append(res)
            log.info("    sensitivity  %-5s %-12s calls=%-3d  %.1fs",
                     name.upper(), param, res.n_solver_calls,
                     time.perf_counter() - t0)
    return out


def run_equal_accuracy_outer(bundle: dict, solvers: tuple[str, ...],
                             r_target: float) -> list:
    """
    Equal-accuracy sweep over the requested solvers for one 2-D or 3-D case.

    One entry point serves all three solvers, where 1-D needs three, because every
    solver reaches the outer layer through the same `solve()` signature and
    differs only in which `inner_options` key carries its precision knob.

    Parameters
    ----------
    bundle : dict
        As returned by `_build_outer`.
    solvers : tuple of str
        Subset of ('hhl', 'vqls', 'qsvt').
    r_target : float
        Common outer-residual target.

    Returns
    -------
    list of EqualAccuracyResult
        One entry per solver that completed.
    """
    out = []
    for name in solvers:
        t0 = time.perf_counter()
        try:
            res = ea.sweep_outer_equal_accuracy(
                solver=name, r_target=r_target, **bundle)
        except Exception as exc:
            log.warning("    equal-accuracy %s FAILED: %s", name.upper(), exc)
            continue
        out.append(res)
        log.info("    equal-accuracy %-5s in_band=%-5s calls=%-3d  %.1fs",
                 name.upper(), res.in_band, res.n_solver_calls,
                 time.perf_counter() - t0)
    return out


def run_sensitivity_outer(bundle: dict,
                          solvers: tuple[str, ...]) -> dict[str, list]:
    """
    One-at-a-time sensitivity sweeps for one 2-D or 3-D case.

    Parameters swept are read from `sensitivity.OUTER_SENSITIVITY_GRIDS` rather
    than restated here, so adding one there is picked up without an edit.

    Parameters
    ----------
    bundle : dict
        As returned by `_build_outer`.
    solvers : tuple of str
        Subset of ('hhl', 'vqls', 'qsvt').

    Returns
    -------
    dict
        Solver name -> list of SensitivitySweepResult, one per parameter.
    """
    out: dict[str, list] = {}
    for name in solvers:
        for param in sens.OUTER_SENSITIVITY_GRIDS[name]:
            t0 = time.perf_counter()
            try:
                res = sens.sensitivity_sweep_outer(
                    solver=name, param_name=param, **bundle)
            except Exception as exc:
                log.warning("    sensitivity %s/%s FAILED: %s",
                            name.upper(), param, exc)
                continue
            out.setdefault(name, []).append(res)
            log.info("    sensitivity  %-5s %-12s calls=%-3d  %.1fs",
                     name.upper(), param, res.n_solver_calls,
                     time.perf_counter() - t0)
    return out


# -- Metadata ------------------------------------------------------------------

def _git(*args: str) -> str:
    """Return git output, or "unknown" where git is unavailable."""
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def _metadata(args, case_keys, n_values, solvers) -> dict:
    """
    Environment, git state and resolved grids, recorded alongside the results.

    The parameter grids are captured explicitly because they are the independent
    variable of both studies: a sensitivity curve is uninterpretable without the
    values it was evaluated at, and those defaults live in `benchmark/` where they
    may be revised after a run.
    """
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname":  platform.node(),
        "python":    sys.version,
        "numpy":     np.__version__,
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "study":     args.study,
        "dim":       args.dim,
        "scheme":    args.scheme if args.dim > 1 else None,
        "max_wall_s": args.max_wall_s if args.dim > 1 else None,
        "order":     args.order,
        "cases":     list(case_keys),
        "n_values":  list(n_values),
        "solvers":   list(solvers),
        "r_target":  args.r_target,
        "grids": {
            "equal_accuracy": {
                "hhl_epsilon":    ea.HHL_EPSILON_GRID,
                "vqls_n_layers":  ea.VQLS_NLAYERS_GRID,
                "vqls_n_restarts": ea.VQLS_NRESTARTS_GRID,
                "qsvt_max_degree": ea.QSVT_MAXDEGREE_GRID,
                "band_factor":    ea.DEFAULT_BAND_FACTOR,
            },
            "sensitivity": {
                "hhl":  sens.HHL_SENSITIVITY_GRIDS,
                "vqls": sens.VQLS_SENSITIVITY_GRIDS,
                "qsvt": sens.QSVT_SENSITIVITY_GRIDS,
                "baselines": {"hhl": sens.HHL_BASELINE,
                              "vqls": sens.VQLS_BASELINE,
                              "qsvt": sens.QSVT_BASELINE},
            },
            # The 2-D/3-D grids are separate: the knob is an inner_options entry
            # there, and the values are smaller because each grid point is a full
            # outer solve rather than a single 1-D solve.
            "equal_accuracy_outer": ea.OUTER_EQUAL_ACCURACY_GRIDS,
            "sensitivity_outer":    sens.OUTER_SENSITIVITY_GRIDS,
        },
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty":  bool(_git("status", "--porcelain")),
    }


# -- Entry point ---------------------------------------------------------------

def main() -> int:
    global RESULTS_DIR

    parser = argparse.ArgumentParser(
        description="Equal-accuracy and sensitivity studies for the solvers.")
    parser.add_argument("--study", choices=("equal-accuracy", "sensitivity", "both"),
                        default="both")
    parser.add_argument("--dim", type=int, choices=(1, 2, 3), default=1,
                        help="Spatial dimension (default: 1). In 2-D and 3-D the "
                             "parameter drives an inner_options entry and the "
                             "measured quantity is the outer residual.")
    parser.add_argument("--order", type=int, choices=(2, 4), default=2)
    parser.add_argument("--n-values", default=None,
                        help="Resolutions. Default is per-dimension: "
                             "8 in 1-D and 2-D, 4 in 3-D.")
    parser.add_argument("--cases", default=None,
                        help="Registry keys, comma-separated. Default is "
                             "per-dimension; see DEFAULT_CASES_BY_DIM.")
    parser.add_argument("--solvers", default=",".join(SOLVERS),
                        help="Subset of hhl,vqls,qsvt.")
    parser.add_argument("--scheme", default="fmg",
                        help="Outer scheme, 2-D/3-D only (default: fmg).")
    parser.add_argument("--max-wall-s", type=float, default=None,
                        help="Per-solve outer wall-clock bound, 2-D/3-D only. "
                             "Each grid point is a full outer solve, so without "
                             "this one non-converging point can consume the job.")
    parser.add_argument("--r-target", type=float, default=ea.DEFAULT_R_TARGET,
                        help="Equal-accuracy residual target "
                             f"(default: {ea.DEFAULT_R_TARGET:g}).")
    parser.add_argument("--run-tag", default="",
                        help="Suffix distinguishing this invocation's outputs. "
                             "Also appended to the output directory name, so "
                             "two concurrent invocations cannot overwrite one "
                             "another's records.")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Output directory. Default: derived from --dim, "
                             "--order and --run-tag by results_dir_for().")
    args = parser.parse_args()

    # Output directory carries the dimension, the order and the run tag, so that
    # neither a 2-D study nor a fourth-order one can overwrite another's records;
    # see results_dir_for() for why none of the three is separable afterwards.
    RESULTS_DIR = (args.results_dir if args.results_dir is not None
                   else results_dir_for(args.dim, args.order, args.run_tag))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # An explicit --results-dir, or a directory reused across orders, can still
    # pair records of one order with metadata of another. Refuse rather than
    # merge: make_tables.py reads config.order to decide which order's tables a
    # study directory may be emitted under, so a mismatch here surfaces later as
    # a mislabelled table rather than as an error.
    prior = RESULTS_DIR / "run_metadata.json"
    if prior.exists():
        try:
            recorded = json.loads(prior.read_text(encoding="utf-8"))
            prior_order = (recorded.get("config") or {}).get("order")
        except (OSError, json.JSONDecodeError):
            prior_order = None
        if prior_order is not None and int(prior_order) != args.order:
            parser.error(
                f"{RESULTS_DIR} holds records of discretisation order "
                f"{prior_order}; this run is order {args.order}. Writing here "
                f"would displace them. Direct this run elsewhere with "
                f"--results-dir, or give it a distinct --run-tag.")

    n_default = ",".join(str(n) for n in DEFAULT_N_BY_DIM[args.dim])
    n_values = [int(v) for v in (args.n_values or n_default).split(",") if v.strip()]
    solvers = tuple(s.strip().lower() for s in args.solvers.split(",") if s.strip())
    case_default = ",".join(DEFAULT_CASES_BY_DIM[args.dim])
    case_keys = [c.strip() for c in (args.cases or case_default).split(",") if c.strip()]

    dropped = [c for c in case_keys if c in EXCLUDED_CASES]
    case_keys = [c for c in case_keys if c not in EXCLUDED_CASES]
    if dropped:
        log.warning("Excluded (quantum solves are invalid for these): %s",
                    ", ".join(dropped))

    unknown = [s for s in solvers if s not in SOLVERS]
    if unknown:
        parser.error(f"unknown solver(s): {', '.join(unknown)}")
    if not case_keys:
        parser.error("no cases left to run")

    scheme_options: dict = {}
    if args.max_wall_s is not None:
        scheme_options["max_wall_s"] = args.max_wall_s

    log.info("=" * 78)
    log.info("  %d-D PARAMETER STUDIES  -  study=%s  order=%d",
             args.dim, args.study, args.order)
    log.info("=" * 78)
    log.info("  Cases    : %s", ", ".join(case_keys))
    log.info("  N values : %s", n_values)
    log.info("  Solvers  : %s", ", ".join(solvers))
    log.info("  r_target : %g", args.r_target)
    if args.dim > 1:
        log.info("  Scheme   : %s", args.scheme)
        log.info("  Sch opts : %s", scheme_options or "(none)")
    log.info("  Output   : %s", RESULTS_DIR.resolve())

    archive = SweepArchive(RESULTS_DIR, run_tag=args.run_tag)
    archive.write_metadata(_metadata(args, case_keys, n_values, solvers))

    ea_all: list = []
    sens_all: dict[str, list] = {}
    t_start = time.perf_counter()

    for N in n_values:
        for case_key in case_keys:
            log.info("-" * 78)
            log.info("  %s   N=%d", case_key, N)
            try:
                if args.dim == 1:
                    bundle = _build(case_key, N, args.order)
                else:
                    bundle = _build_outer(case_key, N, args.order,
                                          args.scheme, scheme_options)
            except Exception as exc:
                log.error("    assembly FAILED: %s -- skipping.", exc)
                continue
            log.info("    kappa=%.4f", bundle["kappa"])

            equal_accuracy = (run_equal_accuracy if args.dim == 1
                              else run_equal_accuracy_outer)
            sensitivity = (run_sensitivity if args.dim == 1
                           else run_sensitivity_outer)

            if args.study in ("equal-accuracy", "both"):
                ea_all.extend(equal_accuracy(bundle, solvers, args.r_target))
            if args.study in ("sensitivity", "both"):
                for name, sweeps in sensitivity(bundle, solvers).items():
                    sens_all.setdefault(name, []).extend(sweeps)

            # Written after every case rather than once at the end, so that a
            # walltime kill loses only the case in flight. This mirrors the
            # incremental per-solution writes in the primary runners, and for the
            # same reason: a walltime kill should discard at most one case, not the
            # entirety of accumulated results.
            if ea_all:
                archive.append_equal_accuracy(ea_all)
            for name, sweeps in sens_all.items():
                archive.write_sensitivity(name, sweeps)

    log.info("=" * 78)
    log.info("  Complete in %.1f s  -  %d equal-accuracy record(s), "
             "%d sensitivity sweep(s)", time.perf_counter() - t_start,
             len(ea_all), sum(len(v) for v in sens_all.values()))
    log.info("  Written to %s", RESULTS_DIR.resolve())
    log.info("=" * 78)
    return 0 if (ea_all or sens_all) else 1


if __name__ == "__main__":
    sys.exit(main())
