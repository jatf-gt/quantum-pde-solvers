#!/usr/bin/env python3
"""
run_2d.py
=========
Full 2-D HPC benchmark sweep for the quantum linear solvers (HHL, VQLS, QSVT)
against the classical Thomas reference.

Design
------
The outer iteration is not implemented here.  It lives in ``solvers.outer``, so
that this script does only what a benchmark runner should: declare cases, drive
the sweep, collect metrics and write results.  Three consequences follow, each
deliberate:

*   **One scheme for the whole sweep**, so the solvers are compared on equal
    terms.  The default is full multigrid (``--scheme fmg``), which needs O(1)
    outer iterations rather than the O(N) of line-SOR or line-Jacobi.
    ``--scheme jacobi --criterion delta`` reproduces the original line-Jacobi
    behaviour exactly, for regenerating a previously published result.

*   **The work unit reported is the strip solve**, not the outer iteration.
    That is what a quantum solver actually pays for, and it is the only
    quantity that compares fairly across schemes.  Measured statevector cost
    per strip solve scales as n^α with α ~ 2.4 (HHL), ~1.3 (VQLS), ~0.6 (QSVT),
    so a scheme doing much of its work on short strips is cheaper than a raw
    solve count suggests; both the raw count and the α-weighted cost are
    recorded.

*   **Solver parameters come from the command line and are recorded in the run
    metadata**, so a sweep is reproducible from its own output.

Usage
-----
    # default sweep: five sections, three quantum solvers, FMG
    python hpc/runners/run_2d.py --max-n 32

    # tune the solvers
    python hpc/runners/run_2d.py --max-n 64 \
        -I qsvt.max_degree=300 -I hhl.epsilon=0.02 -I vqls.n_restarts=2

    # tune the outer scheme
    python hpc/runners/run_2d.py --max-n 64 -S nu1=2 -S n_coarse=8

    # reproduce the original line-Jacobi results
    python hpc/runners/run_2d.py --max-n 8 --scheme jacobi \
        --criterion delta --tol 1e-6

    # cost estimate only: no quantum, projects wall time from strip counts
    python hpc/runners/run_2d.py --max-n 64 --estimate

    # additionally record how the outer schemes compare (classical inner only)
    python hpc/runners/run_2d.py --max-n 64 --compare-schemes

    python hpc/runners/run_2d.py --list-options
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import dataclasses
import json
import logging
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core import cases
from core import het_geometry as geom
from problems.poisson_line_2d import PoissonLine2D
from solvers.outer import (InnerConfig, available_inner,available_schemes,
                           build_hierarchy, describe_inner, describe_scheme,
                           get_inner, resolve_options, solve)

# ── Output directory and logging ──────────────────────────────────────────────

RESULTS_DIR = Path("results") / "2Dhpc_run"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = RESULTS_DIR / "run.log"
_IS_MAIN_PROCESS = mp.current_process().name == "MainProcess"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  pid=%(process)-6d  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w" if _IS_MAIN_PROCESS else "a"),
    ],
)
log = logging.getLogger(__name__)

for _noisy in ("qiskit.transpiler", "qiskit_aer", "qiskit_ibm_runtime",
               "stevedore", "qiskit.passmanager", "pennylane"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)


# ── Sweep configuration ───────────────────────────────────────────────────────

# NOTE: 128 and 256 are included so --max-n 128 / --max-n 256 behave as
# expected. They are still expensive; a two-phase submission script that
# restricts the N=128,256 tier to QSVT only (via --n-values and --solvers)
# is the intended way to reach them, not a single-phase --max-n 256 run.
N_VALUES_ALL: list[int] = [4, 8, 16, 32, 64, 128, 256]

QUANTUM_SOLVERS: tuple[str, ...] = ("hhl", "vqls", "qsvt")
SOLVER_LABEL = {"thomas": "Thomas", "hhl": "HHL", "vqls": "VQLS", "qsvt": "QSVT"}

DEFAULT_SCHEME: str = "fmg"

# Default algebraic tolerance.  The meaningful target is the discretisation
# error, not machine precision: driving the algebraic residual far below the
# truncation error buys nothing and, with a quantum inner solver, costs hours.
# 1e-4 sits roughly one order below the h^2 error across this sweep range.
DEFAULT_TOL: float = 1e-4

# Per-strip-solve cost exponents t(n) ~ n^alpha, fitted from the N=4 / N=8
# statevector timings (HHL 0.267 -> 1.36 s, VQLS 0.806 -> 1.965 s,
# QSVT 0.0259 -> 0.0393 s).  Used to weight coarse-level work and to project
# wall time in --estimate mode.  Classical solves are counted linearly.
COST_ALPHA: dict[str, float] = {
    "thomas": 1.00, "hhl": 2.35, "vqls": 1.29, "qsvt": 0.60,
}
# Reference wall time for one strip solve at n = 8, in seconds (same source).
COST_T8: dict[str, float] = {
    "thomas": 2.0e-5, "hhl": 1.36, "vqls": 1.965, "qsvt": 0.0393,
}

# QSVT polynomial degree caps.  kappa(A_row) stays ~2-3 on every grid and on
# every multigrid level, so the required degree is modest; the cap is a safety
# ceiling for the large-N runs rather than an accuracy control.
QSVT_MAX_DEGREE_2D: dict[int, Optional[int]] = {
    4: None, 8: None, 16: None, 32: 500, 64: 500,
}

HHL_EPSILON_DEFAULT: float = 0.01

# Fourier reference for the two-Gaussian case.  Coefficients are computed on a
# fine grid independent of the solver N: N_FINE = 200 gives dx = 50 um against
# a Gaussian width sigma = 1 mm, so the quadrature error is below 0.01 %.
N_FOURIER_MODES: int = 50
N_FINE: int = 200

# Kept only for the run-metadata record (_save_metadata); the sections
# themselves read geometry through the case registry (core/cases.py), which
# is itself sourced from core/het_geometry.py.
HET_Lz: float = geom.L_Z
HET_Lr: float = geom.L_R
HET_phi0: float = geom.PHI_0

MAX_WORKERS_DEFAULT: int = 4


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RunResult2D:
    """
    One (case, solver, N) benchmark record.

    Field names from the previous version are kept so that
    ``plot_hpc_results.py`` and any existing analysis notebooks continue to
    work; everything new is appended after them.
    """
    # ── identity ──────────────────────────────────────────────────────────────
    case:            str
    solver:          str
    N:               int
    kappa_row:       float

    # ── accuracy (legacy field names) ─────────────────────────────────────────
    max_rel_err:     Optional[float]
    max_abs_err:     Optional[float]
    residual:        Optional[float]
    wall_time_s:     float
    converged:       bool
    n_jacobi_iters:  int                       # == n_outer, kept for compatibility
    notes:           str = ""
    rel_l2_err:      Optional[float] = None
    rms_err:         Optional[float] = None
    vqls_final_cost: Optional[float] = None
    qsvt_degree:     Optional[int]   = None
    qsvt_depth:      Optional[int]   = None
    hhl_scale_c:     Optional[float] = None
    stop_reason:     str = ""

    # ── outer scheme ──────────────────────────────────────────────────────────
    scheme:             str = ""
    n_outer:            int = 0
    convergence_factor: Optional[float] = None   # geometric mean residual ratio
    n_levels:           Optional[int] = None
    level_shapes:       str = ""                 # JSON
    level_kappas:       str = ""                 # JSON

    # ── work accounting ───────────────────────────────────────────────────────
    strip_solves:         int = 0
    strip_solves_by_size: str = ""               # JSON, {strip length: count}
    weighted_cost:        Optional[float] = None # finest-strip-solve units
    mean_strip_size:      Optional[float] = None

    # ── inner solver diagnostics ──────────────────────────────────────────────
    inner_calls:     int = 0
    inner_total_s:   Optional[float] = None
    inner_mean_s:    Optional[float] = None
    inner_max_s:     Optional[float] = None
    inner_failures:  int = 0
    inner_options:   str = ""                    # JSON, as resolved
    n_circuit_evals: Optional[float] = None

    # ── error decomposition ───────────────────────────────────────────────────
    err_vs_thomas:       Optional[float] = None  # quantum algorithmic error, %
    err_thomas_vs_exact: Optional[float] = None  # discretisation error, %
    linf_err:            Optional[float] = None  # amplitude-normalised Linf, %

    # ── derived physics ───────────────────────────────────────────────────────
    peak_E_field:   Optional[float] = None       # V/m
    peak_E_rel_err: Optional[float] = None       # % against the reference

    # ── efficiency ────────────────────────────────────────────────────────────
    s_per_strip_solve: Optional[float] = None
    solves_per_digit:  Optional[float] = None    # strip solves per decade of
                                                 # residual reduction


# ── Logging helpers ───────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    sep = "=" * 78
    log.info(sep); log.info("  %s", msg); log.info(sep)


def _section(msg: str) -> None:
    log.info("-" * 78); log.info("  %s", msg); log.info("-" * 78)


# ── Error metrics ─────────────────────────────────────────────────────────────

def _max_rel(u: np.ndarray, ref: np.ndarray, tol: float = 1e-10) -> float:
    """
    Max pointwise relative error, over nodes where the reference is non-zero.

    Masking |ref| <= tol matters for the HET manufactured solution, whose
    profile passes through zero at the anode, the cathode and the outer wall;
    including those nodes makes the metric diverge on a perfectly good field.
    """
    mask = np.abs(ref) > tol
    if not np.any(mask):
        return float("nan")
    return float(np.max(np.abs(u[mask] - ref[mask]) / np.abs(ref[mask])) * 100.0)


def _max_abs(u, ref) -> float:
    return float(np.max(np.abs(u - ref)))


def _rel_l2(u, ref) -> float:
    return float(np.linalg.norm(u - ref) / (np.linalg.norm(ref) + 1e-300))


def _rms(u, ref) -> float:
    return float(np.sqrt(np.mean((u - ref) ** 2)))


def _norm_linf(u, ref) -> float:
    """Max absolute error normalised by the reference amplitude, in per cent."""
    if u is None or ref is None:
        return float("nan")
    return float(np.max(np.abs(u - ref)) / (np.max(np.abs(ref)) + 1e-300) * 100.0)


def _accuracy(u, ref) -> dict:
    if ref is None or u is None:
        return {}
    return {"max_rel_err": _max_rel(u, ref),
            "max_abs_err": _max_abs(u, ref),
            "rel_l2_err":  _rel_l2(u, ref),
            "rms_err":     _rms(u, ref),
            "linf_err":    _norm_linf(u, ref)}


def _electric_field(phi: np.ndarray, dx: float, dy: float):
    """E = -grad(phi).  Peak magnitude is the quantity of interest for HET."""
    Ex = -np.gradient(phi, dx, axis=0)
    Ey = -np.gradient(phi, dy, axis=1)
    return Ex, Ey, float(np.max(np.sqrt(Ex**2 + Ey**2)))


# ── Problem definitions ───────────────────────────────────────────────────────
#
# All five sections' cases (source, exact/reference solution, boundary data,
# domain extents) now live in core/cases.py, read via cases.get(name).build(N)
# in each run_sectionN below - not built inline here as they used to be.


# ── Sweep settings container ──────────────────────────────────────────────────

@dataclass
class SweepConfig:
    """
    Everything controlling a sweep, in one picklable object.

    It is handed to every worker process and written verbatim into
    ``run_metadata.json``, so a run can be reproduced from its own output.
    """
    scheme:          str = DEFAULT_SCHEME
    tol:             float = DEFAULT_TOL
    max_outer:       int = 500
    solvers:         tuple[str, ...] = QUANTUM_SOLVERS
    inner_options:   dict = field(default_factory=dict)   # {solver: {key: val}}
    scheme_options:  dict = field(default_factory=dict)
    criterion:       Optional[str] = None
    save_solutions:  bool = True
    save_history:    bool = True
    estimate_only:   bool = False
    compare_schemes: bool = False

    def inner_config(self, N: int) -> InnerConfig:
        """
        Per-solver options for this N.  Sweep defaults are applied first so
        that anything given on the command line overrides them.
        """
        cfg = InnerConfig()
        cfg["thomas"] = {}
        cfg["hhl"] = {"epsilon": HHL_EPSILON_DEFAULT}
        cap = QSVT_MAX_DEGREE_2D.get(N)
        cfg["qsvt"] = {} if cap is None else {"max_degree": cap}
        cfg["vqls"] = {}
        for solver, opts in (self.inner_options or {}).items():
            cfg.setdefault(solver, {}).update(opts)
        return cfg

    def scheme_kwargs(self, scheme: Optional[str] = None) -> dict:
        """Scheme keyword arguments, using the right iteration-cap name."""
        scheme = scheme or self.scheme
        kw = dict(self.scheme_options or {})
        kw.setdefault("tol", self.tol)
        if scheme in ("multigrid", "fmg"):
            kw.pop("max_iter", None)
            kw.pop("criterion", None)
            kw.setdefault("max_cycles", min(self.max_outer, 200))
        else:
            kw.pop("max_cycles", None)
            kw.setdefault("max_iter", self.max_outer)
            if self.criterion:
                kw.setdefault("criterion", self.criterion)
        return kw


# ── Result recording ──────────────────────────────────────────────────────────

def _save_solution_2d(case, solver, N, x, y, phi, phi_ref, f_vals,
                      residual_history=None) -> None:
    fname = RESULTS_DIR / f"solutions_{case}_{solver}_N{N}.npz"
    arrays = {"x": x, "y": y, "phi_solver": phi, "f_vals": f_vals,
              # alias so the existing plotting helper, which looks for
              # "u_solver", keeps loading these files unchanged
              "u_solver": phi}
    if phi_ref is not None:
        arrays["phi_exact"] = phi_ref
        arrays["u_exact"] = phi_ref
    if residual_history is not None:
        arrays["residual_history"] = np.asarray(residual_history, dtype=float)
    np.savez_compressed(fname, **arrays)


def _record(results, case_id, solver_name, N, kappa, x, y, dx, dy,
            res, phi_ref, f_vals, phi_thomas, cfg: SweepConfig,
            notes: str = "") -> None:
    """Convert an OuterResult into a RunResult2D and archive the field."""
    label = SOLVER_LABEL.get(solver_name, solver_name.upper())

    if res is None:
        results.append(RunResult2D(
            case=case_id, solver=label, N=N, kappa_row=kappa,
            max_rel_err=None, max_abs_err=None, residual=None,
            wall_time_s=0.0, converged=False, n_jacobi_iters=0,
            notes=notes or "solver_error", scheme=cfg.scheme))
        return

    phi = res.u
    d = res.diagnostics
    acc = _accuracy(phi, phi_ref)

    # ── work accounting ───────────────────────────────────────────────────────
    by_size = res.work.solves_by_size
    total = res.work.total
    alpha = COST_ALPHA.get(solver_name, 1.0)
    mean_size = (sum(n * k for n, k in by_size.items()) / total) if total else None

    # ── efficiency ────────────────────────────────────────────────────────────
    hist = res.residual_history
    decades = None
    if len(hist) > 1 and hist[0] > 0.0 and hist[-1] > 0.0:
        decades = float(np.log10(hist[0] / hist[-1]))
    per_digit = (total / decades) if (decades and decades > 0.0) else None

    # ── physics ───────────────────────────────────────────────────────────────
    _, _, peak_E = _electric_field(phi, dx, dy)
    peak_E_err = None
    if phi_ref is not None:
        _, _, peak_E_ref = _electric_field(phi_ref, dx, dy)
        if peak_E_ref > 0.0:
            peak_E_err = float(abs(peak_E - peak_E_ref) / peak_E_ref * 100.0)

    # ── error decomposition ───────────────────────────────────────────────────
    if solver_name == "thomas":
        err_vs_thomas = 0.0
    elif phi_thomas is not None:
        err_vs_thomas = _norm_linf(phi, phi_thomas)
    else:
        err_vs_thomas = None
    err_thomas_vs_exact = (_norm_linf(phi_thomas, phi_ref)
                           if (phi_thomas is not None and phi_ref is not None)
                           else None)

    rho = res.convergence_factor
    results.append(RunResult2D(
        case=case_id, solver=label, N=N, kappa_row=kappa,
        max_rel_err=acc.get("max_rel_err"),
        max_abs_err=acc.get("max_abs_err"),
        residual=res.residual,
        wall_time_s=res.wall_time_s,
        converged=res.converged,
        n_jacobi_iters=res.n_outer,
        notes=notes,
        rel_l2_err=acc.get("rel_l2_err"),
        rms_err=acc.get("rms_err"),
        linf_err=acc.get("linf_err"),
        vqls_final_cost=d.get("final_cost_mean"),
        qsvt_degree=(int(d["polynomial_degree_mean"])
                     if d.get("polynomial_degree_mean") is not None else None),
        qsvt_depth=(int(d["circuit_depth_mean"])
                    if d.get("circuit_depth_mean") is not None else None),
        hhl_scale_c=d.get("prop_const_mean"),
        stop_reason=res.stop_reason,
        scheme=res.scheme,
        n_outer=res.n_outer,
        convergence_factor=(rho if (rho is not None and np.isfinite(rho)) else None),
        n_levels=d.get("n_levels"),
        level_shapes=json.dumps(d.get("level_shapes", []), default=str),
        level_kappas=json.dumps(d.get("level_kappas", []), default=str),
        strip_solves=total,
        strip_solves_by_size=json.dumps({str(k): v for k, v in by_size.items()}),
        weighted_cost=res.work.weighted_cost(alpha),
        mean_strip_size=mean_size,
        inner_calls=d.get("inner_calls", 0),
        inner_total_s=d.get("inner_total_s"),
        inner_mean_s=d.get("inner_mean_s"),
        inner_max_s=d.get("inner_max_s"),
        inner_failures=d.get("inner_failures", 0),
        inner_options=json.dumps(d.get("inner_options", {}), default=str),
        n_circuit_evals=d.get("n_circuit_evals_mean"),
        err_vs_thomas=err_vs_thomas,
        err_thomas_vs_exact=err_thomas_vs_exact,
        peak_E_field=peak_E,
        peak_E_rel_err=peak_E_err,
        s_per_strip_solve=(res.wall_time_s / total) if total else None,
        solves_per_digit=per_digit,
    ))

    if cfg.save_solutions:
        _save_solution_2d(case_id, label, N, x, y, phi, phi_ref, f_vals,
                          res.residual_history if cfg.save_history else None)


# ── Per-case driver ───────────────────────────────────────────────────────────

def _estimate_case(case_id, N, problem, cfg: SweepConfig) -> None:
    """
    Run the classical inner solver only and project the quantum wall time from
    the resulting strip-solve profile.

    Run this before submitting a large job: it costs seconds and tells you
    whether the sweep fits inside the walltime.  The projection uses
    t(n) = t8 * (n/8)^alpha with constants measured from earlier runs, so treat
    it as an order-of-magnitude guide rather than a guarantee.
    """
    res = solve(problem, inner="thomas", scheme=cfg.scheme,
                inner_options=cfg.inner_config(N), **cfg.scheme_kwargs())
    by_size = res.work.solves_by_size
    log.info("    %-34s N=%-3d  %d outer, %d strip solves %s",
             case_id[:34], N, res.n_outer, res.work.total,
             dict(sorted(by_size.items(), reverse=True)))
    for s in cfg.solvers:
        alpha, t8 = COST_ALPHA.get(s, 1.0), COST_T8.get(s, 1.0)
        secs = sum(k * t8 * (n / 8.0) ** alpha for n, k in by_size.items())
        log.info("        projected %-5s %10.1f s  (%6.2f h)",
                 s.upper(), secs, secs / 3600.0)


def _run_case(case_id, N, X, Y, dx, dy, f_vals, phi_ref, cfg: SweepConfig,
              results: list, problem: PoissonLine2D) -> None:
    """
    Run the Thomas reference and every requested quantum solver on one case.

    `problem` is the already-assembled PoissonLine2D from the case registry
    (core/cases.py), which has the boundary data baked in - this function no
    longer reconstructs it from bc_x0/bc_x1/bc_y0/bc_y1 arguments.
    """
    kappa = problem.kappa_row()
    inner_cfg = cfg.inner_config(N)
    kw = cfg.scheme_kwargs()

    if cfg.estimate_only:
        _estimate_case(case_id, N, problem, cfg)
        return

    levels = build_hierarchy(problem)
    log.info("    grid %dx%d  kappa=%.4f  scheme=%s  hierarchy: %s",
             N, N, kappa, cfg.scheme,
             " -> ".join(f"{lv.problem.shape[0]}x{lv.problem.shape[1]}"
                         for lv in levels))

    # A multigrid scheme needs at least two levels; PoissonLine2D.coarsen()
    # refuses below MIN_STRIP=4, so N=4 always has exactly one level.
    # solve() raises rather than silently degrading (see multigrid.py) -
    # correct for a library call, but it would abort this entire (section, N)
    # work unit, Thomas reference included, before the quantum solvers even
    # get a chance to run. Fall back to line-SOR here, at the run-once
    # benchmark level, and say so in both the log and the recorded rows.
    effective_scheme = cfg.scheme
    scheme_kw = kw
    if cfg.scheme in ("multigrid", "fmg") and len(levels) < 2:
        effective_scheme = "sor"
        scheme_kw = cfg.scheme_kwargs(effective_scheme)
        log.warning("    grid %dx%d cannot be coarsened (needs >= 2 levels); "
                   "falling back to scheme=%r for this case only.",
                   N, N, effective_scheme)

    # ── classical reference ───────────────────────────────────────────────────
    res_T = solve(problem, inner="thomas", scheme=effective_scheme,
                  inner_options=inner_cfg, **scheme_kw)
    phi_T = res_T.u
    log.info("    %-6s %5d outer  %8d solves  err=%8.4f%%  %7.2fs  %s",
             "Thomas", res_T.n_outer, res_T.work.total,
             _norm_linf(phi_T, phi_ref), res_T.wall_time_s, res_T.stop_reason)
    fallback_note = (f"scheme_fallback:{cfg.scheme}->{effective_scheme}"
                     if effective_scheme != cfg.scheme else "")
    _record(results, case_id, "thomas", N, kappa, X, Y, dx, dy,
            res_T, phi_ref, f_vals, phi_T, cfg,
            notes=(fallback_note or ("" if phi_ref is not None else "reference")))

    # Where no analytical reference exists, Thomas becomes the reference.
    reference = phi_ref if phi_ref is not None else phi_T
    ref_note = "" if phi_ref is not None else "rel_vs_thomas"

    # ── optional outer-scheme comparison, classical inner solver only ─────────
    if cfg.compare_schemes:
        for alt in available_schemes():
            if alt == cfg.scheme:
                continue
            try:
                r = solve(problem, inner="thomas", scheme=alt,
                          inner_options=inner_cfg, **cfg.scheme_kwargs(alt))
            except Exception as exc:
                log.warning("    [scheme] %-13s unavailable: %s", alt, exc)
                continue
            log.info("    [scheme] %-13s %5d outer  %8d solves  %s",
                     alt, r.n_outer, r.work.total, r.stop_reason)
            _record(results, case_id, "thomas", N, kappa, X, Y, dx, dy,
                    r, phi_ref, f_vals, phi_T,
                    dataclasses.replace(cfg, save_solutions=False),
                    notes=f"scheme_comparison:{alt}")

    # ── quantum solvers ───────────────────────────────────────────────────────
    for solver_name in cfg.solvers:
        try:
            res_q = solve(problem, inner=solver_name, scheme=effective_scheme,
                          inner_options=inner_cfg, **scheme_kw)
        except Exception as exc:
            log.error("    %-6s FAILED: %s", solver_name.upper(), exc,
                      exc_info=True)
            _record(results, case_id, solver_name, N, kappa, X, Y, dx, dy,
                    None, reference, f_vals, phi_T, cfg, notes=str(exc)[:200])
            continue

        fallbacks = res_q.diagnostics.get("inner_failures", 0)
        log.info("    %-6s %5d outer  %8d solves  err=%8.4f%%  "
                 "vs_Thomas=%7.4f%%  %8.2fs  %s%s",
                 solver_name.upper(), res_q.n_outer, res_q.work.total,
                 _norm_linf(res_q.u, reference), _norm_linf(res_q.u, phi_T),
                 res_q.wall_time_s, res_q.stop_reason,
                 f"  [{fallbacks} classical fallbacks]" if fallbacks else "")

        _record(results, case_id, solver_name, N, kappa, X, Y, dx, dy,
                res_q, reference, f_vals, phi_T, cfg,
                notes=(fallback_note or ref_note))


# ── Sections ──────────────────────────────────────────────────────────────────

def run_section1(N, cfg, results):
    _banner(f"SECTION 1 - Poisson, sinusoidal source, unit square, N={N}")
    built = cases.get("poisson_2d_sin_pi").build(N)
    X, Y = built.coords
    dx, dy = built.spacings
    _run_case("2D_Poisson_sin_hom", N, X, Y, dx, dy,
              built.f_values, built.exact, cfg, results, built.problem)


def run_section2(N, cfg, results):
    _banner(f"SECTION 2 - Two-Gaussian PlasmaNet benchmark, N={N}")
    built = cases.get("poisson_2d_two_gaussian_plasmanet").build(N)
    X, Y = built.coords
    dx, dy = built.spacings
    log.info("    Fourier reference: N_fine=%d, N_modes=%d", N_FINE, N_FOURIER_MODES)
    _run_case("2D_Poisson_TwoGaussian_PlasmaNet", N, X, Y, dx, dy,
              built.f_values, built.exact, cfg, results, built.problem)


def run_section3(N, cfg, results):
    _banner(f"SECTION 3 - Single Fourier mode (n=1, m=1), N={N}")
    built = cases.get("poisson_2d_single_mode_n1m1").build(N)
    X, Y = built.coords
    dx, dy = built.spacings
    _run_case("2D_Poisson_SingleMode_n1m1", N, X, Y, dx, dy,
              built.f_values, built.exact, cfg, results, built.problem)


def run_section4(N, cfg, results):
    _banner(f"SECTION 4 - HET MMS (SPT-100), N={N}")
    built = cases.get("het_2d_mms_spt100").build(N)
    Z, R = built.coords
    dz, dr = built.spacings
    _run_case("2D_HET_MMS_SPT100", N, Z, R, dz, dr,
              built.f_values, built.exact, cfg, results, built.problem)


def run_section5(N, cfg, results):
    _banner(f"SECTION 5 - HET sinusoidal source, N={N}")
    built = cases.get("het_2d_sin_meeting_report").build(N)
    X, Y = built.coords
    dx, dy = built.spacings
    _run_case("2D_HET_Sin_MeetingReport", N, X, Y, dx, dy,
              built.f_values, built.exact, cfg, results, built.problem)


SECTIONS = {"section1": run_section1, "section2": run_section2,
            "section3": run_section3, "section4": run_section4,
            "section5": run_section5}


# ── Serialisation ─────────────────────────────────────────────────────────────

def _load_existing_results(path: Path) -> list[RunResult2D]:
    """
    Load rows from a previous invocation for --append to build on.

    Unknown fields in the JSON (e.g. from an older schema) are dropped rather
    than raising, and a missing or unparsable file yields an empty list
    rather than aborting the run - --append should never be the reason a
    sweep fails to start.
    """
    if not path.exists():
        return []
    try:
        with open(path) as fh:
            rows = json.load(fh)
    except Exception as exc:
        log.warning("Could not parse %s (%s); starting without prior results.",
                   path, exc)
        return []
    valid = {f.name for f in dataclasses.fields(RunResult2D)}
    out = []
    for d in rows:
        try:
            out.append(RunResult2D(**{k: v for k, v in d.items() if k in valid}))
        except Exception as exc:
            log.warning("Skipping unreadable prior row: %s", exc)
    return out


def _save_results(results) -> None:
    if not results:
        return
    with open(RESULTS_DIR / "results_full.json", "w") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2, default=str)
    fieldnames = [f.name for f in dataclasses.fields(RunResult2D)]
    with open(RESULTS_DIR / "results_summary.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    log.info("Results written: %d rows -> %s", len(results), RESULTS_DIR.resolve())


def _save_metadata(N_values, cfg: SweepConfig, sections, max_workers,
                   tag: Optional[str] = None) -> None:
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "cpu_count": os.cpu_count(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "slurm_jobid": os.environ.get("SLURM_JOB_ID"),
        "N_values": N_values,
        "sections": sections,
        "max_workers": max_workers,
        "sweep_config": asdict(cfg),
        "qsvt_max_degree_default": {str(k): v for k, v in QSVT_MAX_DEGREE_2D.items()},
        "hhl_epsilon_default": HHL_EPSILON_DEFAULT,
        "cost_model_alpha": COST_ALPHA,
        "cost_model_t8_s": COST_T8,
        "het": {"Lz_m": HET_Lz, "Lr_m": HET_Lr, "phi0_V": HET_phi0},
        "fourier_reference": {"n_modes": N_FOURIER_MODES, "n_fine": N_FINE},
    }
    for mod in ("qiskit", "qiskit_aer", "pennylane", "scipy"):
        try:
            meta[mod] = __import__(mod).__version__
        except Exception:
            meta[mod] = "not installed"
    try:
        meta["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip()
        meta["git_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL).strip())
    except Exception:
        meta["git_commit"] = "unknown"
    fname = f"run_metadata_{tag}.json" if tag else "run_metadata.json"
    with open(RESULTS_DIR / fname, "w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    log.info("Metadata written -> %s", fname)


# ── Final summary ─────────────────────────────────────────────────────────────

def _print_summary(results) -> None:
    rows = [r for r in results if not r.notes.startswith("scheme_comparison")]
    if not rows:
        return

    _banner("SUMMARY")
    log.info("  %-32s %-7s %4s %6s %9s %10s %9s %9s",
             "case", "solver", "N", "outer", "solves", "w.cost", "err%", "vsThom%")
    log.info("  " + "-" * 92)
    for r in sorted(rows, key=lambda r: (r.case, r.N, r.solver)):
        err = "   FAILED" if r.linf_err is None else f"{r.linf_err:9.4f}"
        vt = "        -" if r.err_vs_thomas is None else f"{r.err_vs_thomas:9.4f}"
        wc = "         -" if r.weighted_cost is None else f"{r.weighted_cost:10.0f}"
        log.info("  %-32s %-7s %4d %6d %9d %10s %9s %9s",
                 r.case[:32], r.solver, r.N, r.n_outer, r.strip_solves,
                 wc, err, vt)

    _section("Quantum cost relative to the Thomas reference (same scheme)")
    by_key: dict = {}
    for r in rows:
        by_key.setdefault((r.case, r.N), {})[r.solver] = r
    for (case, N), d in sorted(by_key.items()):
        base = d.get("Thomas")
        if base is None or not base.wall_time_s:
            continue
        parts = [f"{s}={d[s].wall_time_s / base.wall_time_s:,.0f}x"
                 for s in ("HHL", "VQLS", "QSVT")
                 if s in d and d[s].wall_time_s]
        if parts:
            log.info("  %-32s N=%-3d  %s", case[:32], N, "   ".join(parts))

    # Any scheme-comparison rows collected with --compare-schemes.
    comp = [r for r in results if r.notes.startswith("scheme_comparison")]
    if comp:
        _section("Outer scheme comparison (classical inner solver)")
        log.info("  %-32s %4s %-13s %6s %9s", "case", "N", "scheme", "outer", "solves")
        for r in sorted(comp, key=lambda r: (r.case, r.N, r.scheme)):
            log.info("  %-32s %4d %-13s %6d %9d",
                     r.case[:32], r.N, r.scheme, r.n_outer, r.strip_solves)

    stalled = [r for r in rows if r.stop_reason == "stagnated"]
    if stalled:
        _section(f"{len(stalled)} run(s) stopped at the inner solver's error "
                 f"floor rather than at the tolerance")
        for r in stalled:
            log.info("  %-32s %-7s N=%-3d  residual floor %.2e",
                     r.case[:32], r.solver, r.N, r.residual or float("nan"))

    fell_back = [r for r in rows if r.inner_failures]
    if fell_back:
        _section("Runs where strip solves fell back to the classical solver")
        for r in fell_back:
            pct = 100.0 * r.inner_failures / max(r.inner_calls, 1)
            log.info("  %-32s %-7s N=%-3d  %d/%d calls (%.1f%%)",
                     r.case[:32], r.solver, r.N, r.inner_failures,
                     r.inner_calls, pct)


# ── CLI plumbing ──────────────────────────────────────────────────────────────

def parse_kv(items, flag: str) -> dict:
    """Parse ``key=value`` and ``solver.key=value`` pairs from the CLI."""
    out: dict = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"{flag} expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        if "." in key:
            solver, k = key.split(".", 1)
            out.setdefault(solver, {})[k] = value
        else:
            out[key] = value
    return out


def coerce_scheme_opts(d: dict) -> dict:
    """
    Coerce scheme options, which are plain function keyword arguments.

    Inner solver options are deliberately left as strings: the option registry
    in ``solvers.outer.inner`` owns their types and validates them, so this
    parser cannot drift out of step with the solvers.
    """
    out: dict = {}
    for k, v in d.items():
        if k == "omega" and v == "optimal":
            out[k] = v
        elif k == "criterion":
            out[k] = v
        elif k in ("symmetric", "fmg"):
            out[k] = str(v).lower() in ("true", "1", "yes", "on")
        elif k in ("tol", "omega"):
            out[k] = float(v)
        else:
            try:
                out[k] = int(v)
            except ValueError:
                out[k] = float(v)
    return out


# ── Work unit dispatch ────────────────────────────────────────────────────────

def _execute_work_unit(work_type: str, N: int, cfg: SweepConfig) -> list:
    """One (section, N) unit.  Must stay picklable for ProcessPoolExecutor."""
    results: list = []
    fn = SECTIONS.get(work_type)
    if fn is None:
        log.error("Unknown section %r", work_type)
        return results
    try:
        fn(N, cfg, results)
    except Exception as exc:
        # One failed unit must not take the sweep down with it.
        log.error("Section %s N=%d aborted: %s", work_type, N, exc, exc_info=True)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full 2-D HPC benchmark sweep for quantum PDE solvers.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--max-n", type=int, default=max(N_VALUES_ALL),
                    help="largest grid size to run (default %(default)s)")
    ap.add_argument("--n-values", type=str, default=None,
                    help="explicit comma-separated N list; overrides --max-n")
    ap.add_argument("--sections", type=str, default="1,2,3,4,5")
    ap.add_argument("--solvers", type=str, default=",".join(QUANTUM_SOLVERS),
                    help=f"quantum solvers to run, from {list(QUANTUM_SOLVERS)}")
    ap.add_argument("--skip-qsvt", action="store_true",
                    help="shorthand for dropping qsvt from --solvers")

    ap.add_argument("--scheme", default=DEFAULT_SCHEME, choices=available_schemes(),
                    help="outer iteration scheme, used for every solver so the "
                         "comparison is like for like (default %(default)s). "
                         "Use 'jacobi' with --criterion delta to reproduce the "
                         "original results.")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL,
                    help="algebraic tolerance (default %(default)s, roughly one "
                         "order below the discretisation error over this range)")
    ap.add_argument("--max-outer", type=int, default=500,
                    help="cap on outer iterations or cycles")
    ap.add_argument("--criterion", default=None, choices=["residual", "delta"],
                    help="stopping test for stationary schemes; 'delta' "
                         "reproduces the original convergence check")

    ap.add_argument("-I", "--inner-opt", action="append",
                    metavar="SOLVER.KEY=VAL",
                    help="inner solver option, e.g. -I qsvt.max_degree=300. "
                         "Validated: unknown keys are an error, not ignored.")
    ap.add_argument("-S", "--scheme-opt", action="append", metavar="KEY=VAL",
                    help="outer scheme option, e.g. -S nu1=2 -S n_coarse=8")
    ap.add_argument("--list-options", action="store_true",
                    help="print every tunable inner and scheme parameter, then exit")

    ap.add_argument("--estimate", action="store_true",
                    help="classical inner solver only; projects the quantum "
                         "wall time. Run this before submitting a large job.")
    ap.add_argument("--compare-schemes", action="store_true",
                    help="also record every other outer scheme with the "
                         "classical inner solver (cheap, for the scheme table)")
    ap.add_argument("--no-solutions", action="store_true",
                    help="do not archive solution fields")
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS_DEFAULT)
    ap.add_argument("--append", action="store_true",
                    help="merge with the existing results_full.json in "
                         "RESULTS_DIR instead of overwriting it. Needed for "
                         "any multi-phase job that invokes this script more "
                         "than once against the same output directory.")
    ap.add_argument("--phase-tag", default=None,
                    help="if set, also write run_metadata_<tag>.json, so a "
                         "multi-phase job keeps every phase's configuration "
                         "on disk rather than only the last one.")
    args = ap.parse_args()

    if args.list_options:
        print("\n=== INNER SOLVER OPTIONS ===\n")
        print(describe_inner())
        print("\n=== OUTER SCHEME OPTIONS ===\n")
        print(describe_scheme())
        return

    # ── resolve and validate the sweep ────────────────────────────────────────
    if args.n_values:
        N_values = [int(v) for v in args.n_values.split(",") if v.strip()]
    else:
        N_values = [n for n in N_VALUES_ALL if n <= args.max_n]
    if not N_values:
        ap.error(f"--max-n {args.max_n} excludes every N in {N_VALUES_ALL}")
    bad_n = [n for n in N_values if n < 4 or (n & (n - 1))]
    if bad_n:
        ap.error(f"N must be a power of two and at least 4 - the quantum "
                 f"b-register and the multigrid coarsening both require it. "
                 f"Offending values: {bad_n}")

    sections = [f"section{s.strip()}" for s in args.sections.split(",") if s.strip()]
    unknown = [s for s in sections if s not in SECTIONS]
    if unknown:
        ap.error(f"unknown section(s) {unknown}; valid: {sorted(SECTIONS)}")

    solvers = tuple(s.strip().lower() for s in args.solvers.split(",") if s.strip())
    if args.skip_qsvt:
        solvers = tuple(s for s in solvers if s != "qsvt")
    bad_s = [s for s in solvers if s not in available_inner()]
    if bad_s:
        ap.error(f"unknown solver(s) {bad_s}; available: {available_inner()}")

    inner_opts = parse_kv(args.inner_opt, "--inner-opt")
    flat = {k: v for k, v in inner_opts.items() if not isinstance(v, dict)}
    if flat:
        ap.error(f"--inner-opt must be namespaced by solver in a sweep, "
                 f"e.g. -I qsvt.{list(flat)[0]}=...")
    bad_o = [k for k in inner_opts if k not in available_inner()]
    if bad_o:
        ap.error(f"--inner-opt refers to unknown solver(s) {bad_o}; "
                 f"available: {available_inner()}")

    cfg = SweepConfig(
        scheme=args.scheme,
        tol=args.tol,
        max_outer=args.max_outer,
        solvers=solvers,
        inner_options=inner_opts,
        scheme_options=coerce_scheme_opts(parse_kv(args.scheme_opt, "--scheme-opt")),
        criterion=args.criterion,
        save_solutions=not args.no_solutions,
        estimate_only=args.estimate,
        compare_schemes=args.compare_schemes,
    )

    # Pre-flight, in two distinct steps so the two kinds of failure are not
    # confused with one another.
    #
    # 1. Option validation is pure and imports nothing.  A typo must fail here,
    #    in a second, rather than inside a worker several hours into a sweep.
    probe = cfg.inner_config(N_values[0])
    for name in ("thomas",) + tuple(solvers):
        try:
            resolve_options(name, probe.for_solver(name))
        except Exception as exc:
            ap.error(f"inner solver {name!r} rejected its configuration: {exc}")

    # 2. Constructing a solver imports its quantum backend.  A missing backend
    #    is an environment problem, not a configuration one, and it is not a
    #    problem at all in --estimate mode, which never calls a quantum solver.
    if not cfg.estimate_only:
        for name in ("thomas",) + tuple(solvers):
            try:
                get_inner(name, **probe.for_solver(name))
            except ImportError as exc:
                ap.error(f"inner solver {name!r} needs a backend that is not "
                         f"installed: {exc}. Install it, drop the solver from "
                         f"--solvers, or use --estimate.")
            except Exception as exc:
                ap.error(f"inner solver {name!r} could not be built: {exc}")

    _banner("QUANTUM PDE SOLVERS - 2D HPC BENCHMARK")
    log.info("  N values    : %s", N_values)
    log.info("  Sections    : %s", sections)
    log.info("  Scheme      : %s   tol=%.1e   max_outer=%d",
             cfg.scheme, cfg.tol, cfg.max_outer)
    log.info("  Solvers     : %s", list(solvers))
    log.info("  Inner opts  : %s", inner_opts or "(defaults)")
    log.info("  Scheme opts : %s", cfg.scheme_options or "(defaults)")
    log.info("  Workers     : %d", args.max_workers)
    log.info("  Output      : %s", RESULTS_DIR.resolve())
    if cfg.estimate_only:
        log.info("  ESTIMATE MODE - no quantum solver will be executed")

    _save_metadata(N_values, cfg, sections, args.max_workers)
    if args.phase_tag:
        _save_metadata(N_values, cfg, sections, args.max_workers,
                       tag=args.phase_tag)

    # ── execute ───────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    results: list = []
    if args.append:
        prior_path = RESULTS_DIR / "results_full.json"
        prior = _load_existing_results(prior_path)
        if prior:
            log.info("--append: merging with %d prior row(s) from %s",
                     len(prior), prior_path)
        results.extend(prior)
    work_units = [(s, N) for N in sorted(N_values) for s in sections]

    if args.max_workers <= 1 or cfg.estimate_only:
        log.info("Serial execution: %d units.", len(work_units))
        for work_type, N in work_units:
            results.extend(_execute_work_unit(work_type, N, cfg))
            _save_results(results)
    else:
        log.info("Parallel execution: %d units over %d workers.",
                 len(work_units), args.max_workers)
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.max_workers, max_tasks_per_child=1) as ex:
            futures = {ex.submit(_execute_work_unit, wt, N, cfg): (wt, N)
                       for wt, N in work_units}
            for fut in concurrent.futures.as_completed(futures):
                work_type, N = futures[fut]
                try:
                    results.extend(fut.result())
                    log.info("Done: %-12s N=%-3d  (%d rows so far)",
                             work_type, N, len(results))
                    # Written after every unit so a walltime kill still leaves
                    # all completed work on disk.
                    _save_results(results)
                except Exception as exc:
                    log.error("Failed: %s N=%d - %s", work_type, N, exc,
                              exc_info=True)

    _save_results(results)
    if not cfg.estimate_only:
        _print_summary(results)
    _banner(f"Complete in {time.perf_counter() - t0:.1f} s  "
            f"({len(results)} rows) -> {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()