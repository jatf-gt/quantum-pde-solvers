#!/usr/bin/env python3
"""
run_3d.py
=========
Full 3-D HPC benchmark sweep for the quantum linear solvers (HHL, VQLS, QSVT)
against the classical Thomas reference, on the line-decomposed 3-D Poisson
problem.

Relationship to the 2-D runner
------------------------------
Deliberately its near-twin: same CLI, same metrics, same result schema, same
two-phase submission pattern.  Nothing about the quantum solvers changed to
support 3-D - a 3-D problem decomposes into the same tridiagonal strips as a
2-D one, so hhl_1d / vqls_1d / qsvt_1d are used unmodified on log2(N0)
qubits with the existing TST block encoding.  What is new is the problem
class (problems/poisson_line_3d.py) and the geometry of the test cases.

Work scaling, and why 3-D is expensive
--------------------------------------
In 2-D one relaxation sweep costs N strip solves; in 3-D it costs N^2.  At
N=32 that is 1024 strip solves per sweep against 32 in 2-D.  Measured QSVT
wall time at N=16 (HET case, FMG) was ~420 s; N=32 is roughly 8x that per
cycle count.  Plan sweeps accordingly and use --estimate first.

Sections
--------
    1  cube      Triple-sin MMS on the unit cube.  Canonical 3-D Poisson
                 verification; exact solution, used for order-of-accuracy.

    2  het_mms   SPT-100 channel unwrapped to a slab, azimuthally periodic.
                 Verifies the real geometry, the periodic stencil and the
                 periodic grid-transfer operators against an exact solution.

    3  spoke     HET rotating spoke: an azimuthally modulated potential well.
                 The rotating spoke is a well-documented m = 1..6 azimuthal
                 coherent structure in Hall thrusters (Janes & Lowder 1966;
                 McDonald & Gallimore 2011; Sekerak et al. 2015), and
                 resolving it is the principal reason a HET simulation needs
                 the third (azimuthal) dimension at all.  Manufactured so
                 that an exact solution exists.

    5  laplace   Laplace equation on the unit cube: homogeneous PDE with
                 non-homogeneous Dirichlet data, phi = sin sin sinh.  The
                 only exact-solution case in which the boundary-absorption
                 path is exercised at all - every other verification case
                 has zero boundary data.

    6  gaussian  Two localised Gaussian blobs with exact non-homogeneous data
                 on all six faces.  3-D analogue of the two-Gaussian
                 PlasmaNet benchmark; steep, compact source.

    7  highmode  Single Fourier mode (2,3,4): near the grid resolution limit,
                 so it stresses the smoother and the transfer operators where
                 the smooth section-1 mode does not.

    4  discharge Realistic SPT-100 discharge: 300 V anode-to-cathode across
                 the channel with a space-charge source in the acceleration
                 region and negatively biased walls.  This is the Poisson
                 step a HET PIC/fluid code performs every timestep, so it is
                 the case whose cost actually predicts production use.  No
                 closed form; the Thomas solution is the reference.

Usage
-----
    python hpc/runners/run_3d.py --max-n 16
    python hpc/runners/run_3d.py --max-n 32 -I qsvt.max_degree=300
    python hpc/runners/run_3d.py --max-n 32 --estimate     # cost first!
    python hpc/runners/run_3d.py --n-values 32 --solvers qsvt --append
    python hpc/runners/run_3d.py --list-options
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
from problems.poisson_line_3d_4th import PoissonLine3D4th
from solvers.outer import (InnerConfig, available_inner,
                           available_schemes, build_hierarchy, describe_inner,
                           describe_scheme, get_inner, resolve_options, solve)

# ── Output directory and logging ──────────────────────────────────────────────

RESULTS_DIR = Path("results") / "3Dhpc_run"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = RESULTS_DIR / "run.log"
_IS_MAIN_PROCESS = mp.current_process().name == "MainProcess"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  pid=%(process)-6d  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    # Append, never truncate. A single PBS job invokes this runner several times
    # in sequence (submit_3d_wave1.sh runs three steps), and under mode="w" each
    # step destroyed its predecessor's log, leaving only the last. The history is
    # the sole record of how far a walltime-killed step had progressed. Sessions
    # are delimited by `_log_session_header`.
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_FILE, mode="a")],
)
log = logging.getLogger(__name__)


def _log_session_header(phase_tag: Optional[str] = None) -> None:
    """
    Delimit one invocation within an appended log.

    Called once from the parent process after the output directory and any phase
    tag are resolved. Workers do not call it: `_IS_MAIN_PROCESS` is false in a
    spawned child, and a banner per work unit would defeat the purpose.

    Parameters
    ----------
    phase_tag : str, optional
        Step label supplied by the submission script (e.g. ``wave1_qsvt_all``),
        recorded so a row can be attributed to the step that produced it.
    """
    if not _IS_MAIN_PROCESS:
        return
    log.info("=" * 78)
    log.info("SESSION START  %s  pid=%d  phase=%s  argv=%s",
             time.strftime("%Y-%m-%d %H:%M:%S"), os.getpid(),
             phase_tag or "(untagged)", " ".join(sys.argv[1:]) or "(no arguments)")
    log.info("=" * 78)


def _redirect_log_file(path: Path) -> None:
    """
    Point the root logger's file handler at `path`, preserving its formatter.

    Required because the output directory depends on `--order`, which is known
    only after argument parsing, whereas the handler is installed at import time.

    The replacement inherits the original formatter rather than accepting
    logging's bare ``"%(message)s"`` default, which stripped the timestamp and PID
    from every 4th-order log line and left the interleaved output of parallel work
    units unattributable. The new handler appends, for the reason given where the
    first one is installed.

    Parameters
    ----------
    path : Path
        Destination log file. Its parent directory must already exist.
    """
    logger = logging.getLogger()
    fmt = None
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            fmt = fmt or handler.formatter
            handler.close()
            logger.removeHandler(handler)
    new_handler = logging.FileHandler(path, mode="a")
    if fmt is not None:
        new_handler.setFormatter(fmt)
    logger.addHandler(new_handler)

for _noisy in ("qiskit.transpiler", "qiskit_aer", "qiskit_ibm_runtime",
               "stevedore", "qiskit.passmanager", "pennylane"):
    logging.getLogger(_noisy).setLevel(logging.CRITICAL)


# ── Sweep configuration ───────────────────────────────────────────────────────

# 64 is included so --max-n 64 resolves correctly, but 64^3 = 262144 unknowns
# with N^2 = 4096 strip solves per sweep is a serious job.  Always --estimate.
N_VALUES_ALL: list[int] = [4, 8, 16, 32, 64]

QUANTUM_SOLVERS: tuple[str, ...] = ("hhl", "vqls", "qsvt")
SOLVER_LABEL = {"thomas": "Thomas", "hhl": "HHL", "vqls": "VQLS", "qsvt": "QSVT"}

DEFAULT_SCHEME: str = "fmg"
DEFAULT_TOL: float = 1e-6

# Per-strip-solve cost exponents t(n) ~ n^alpha, from the 2-D N=4/N=8 timings.
COST_ALPHA: dict[str, float] = {"thomas": 1.00, "hhl": 2.35,
                                "vqls": 1.29, "qsvt": 0.60}
COST_T8: dict[str, float] = {"thomas": 2.0e-5, "hhl": 1.36,
                             "vqls": 1.965, "qsvt": 0.0393}

# kappa(A_line) -> 2 in 3-D (both transverse directions add to the diagonal),
# lower than the 2-D asymptote of 3, so the required QSVT degree is if
# anything smaller than in 2-D.  d = ceil(13 kappa ln(kappa/eps)) gives ~135
# at kappa=2, eps=0.01 - well under this cap at every N here.
QSVT_MAX_DEGREE_3D: dict[int, Optional[int]] = {
    4: None, 8: None, 16: None, 32: 500, 64: 500}

HHL_EPSILON_DEFAULT: float = 0.01

# ── SPT-100 geometry (the reference Hall thruster in the literature) ──────────
#  The discharge channel is an annulus.  For the azimuthal-axial studies that
#  motivate 3-D, it is standard practice to "unwrap" the annulus into a
#  Cartesian slab: the channel is thin (20 mm wide) compared with its mean
#  circumference (251 mm), so curvature is a second-order effect and the
#  azimuthal direction becomes a straight, periodic coordinate.  This is the
#  geometry used in axial-azimuthal PIC studies of HET instabilities.
#
#      axis 0  z  axial       [0, 40 mm]   Dirichlet: anode / cathode plane
#      axis 1  r  radial      [0, 20 mm]   Dirichlet: inner / outer wall
#      axis 2  s  azimuthal   [0, 251 mm]  PERIODIC (s = r_mean * theta)
#
#  Axis 0 is the strip direction and must be Dirichlet: a periodic strip
#  operator is cyclic-tridiagonal, not TST, which the block encoding assumes.
# Kept only for the run-metadata record (_save_metadata); the case functions
# themselves read geometry and physics constants through the case registry
# (core/cases.py), which is itself sourced from core/het_geometry.py.
HET_LZ: float = geom.L_Z
HET_R_IN, HET_R_OUT = geom.R_IN, geom.R_OUT
HET_LR: float = geom.L_R
HET_R_MEAN: float = geom.R_MEAN
HET_LS: float = geom.L_S

HET_PHI0: float = geom.PHI_0
HET_V_ANODE: float = geom.V_ANODE
HET_V_CATHODE: float = geom.V_CATHODE
HET_V_WALL: float = geom.V_WALL

SPOKE_MODE_M: int = geom.SPOKE_MODE_M
SPOKE_EPSILON: float = geom.SPOKE_EPSILON

MAX_WORKERS_DEFAULT: int = 4


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class RunResult3D:
    """One (case, solver, N) benchmark record.  Schema mirrors RunResult2D."""
    # ── identity ──────────────────────────────────────────────────────────────
    case:            str
    solver:          str
    N:               int
    shape:           str
    n_unknowns:      int
    kappa_row:       float

    # ── accuracy ──────────────────────────────────────────────────────────────
    max_rel_err:     Optional[float]
    max_abs_err:     Optional[float]
    residual:        Optional[float]
    wall_time_s:     float
    converged:       bool
    n_outer:         int
    notes:           str = ""
    rel_l2_err:      Optional[float] = None
    rms_err:         Optional[float] = None
    linf_err:        Optional[float] = None
    stop_reason:     str = ""

    # ── quantum diagnostics ───────────────────────────────────────────────────
    vqls_final_cost: Optional[float] = None
    qsvt_degree:     Optional[int]   = None
    qsvt_depth:      Optional[int]   = None
    hhl_scale_c:     Optional[float] = None

    # ── outer scheme ──────────────────────────────────────────────────────────
    scheme:             str = ""
    convergence_factor: Optional[float] = None
    n_levels:           Optional[int] = None
    level_shapes:       str = ""
    level_kappas:       str = ""
    anisotropy:         Optional[float] = None   # h_max / h_min, finest level

    # ── work accounting ───────────────────────────────────────────────────────
    strip_solves:         int = 0
    strip_solves_by_size: str = ""
    weighted_cost:        Optional[float] = None
    mean_strip_size:      Optional[float] = None

    # ── inner solver diagnostics ──────────────────────────────────────────────
    inner_calls:     int = 0
    inner_total_s:   Optional[float] = None
    inner_mean_s:    Optional[float] = None
    inner_max_s:     Optional[float] = None
    inner_failures:  int = 0
    inner_options:   str = ""
    n_circuit_evals: Optional[float] = None

    # ── error decomposition ───────────────────────────────────────────────────
    err_vs_thomas:       Optional[float] = None
    err_thomas_vs_exact: Optional[float] = None

    # ── derived physics ───────────────────────────────────────────────────────
    peak_E_field:      Optional[float] = None   # V/m, |grad phi|
    peak_E_rel_err:    Optional[float] = None   # %
    peak_E_axial:      Optional[float] = None   # V/m, the thrust-relevant one
    # Azimuthal mode content: for the spoke case, whether the solver actually
    # reproduces the mode it was given.  A solver can look accurate in Linf
    # and still smear the azimuthal structure, which is the whole point of 3-D.
    azimuthal_mode_amp:     Optional[float] = None
    azimuthal_mode_rel_err: Optional[float] = None

    # ── efficiency ────────────────────────────────────────────────────────────
    s_per_strip_solve: Optional[float] = None
    solves_per_digit:  Optional[float] = None


# ── Logging helpers ───────────────────────────────────────────────────────────

def _banner(msg: str) -> None:
    sep = "=" * 78
    log.info(sep); log.info("  %s", msg); log.info(sep)


def _section(msg: str) -> None:
    log.info("-" * 78); log.info("  %s", msg); log.info("-" * 78)


# ── Metrics ───────────────────────────────────────────────────────────────────

def _max_rel(u, ref, tol=1e-10) -> float:
    mask = np.abs(ref) > tol
    if not np.any(mask):
        return float("nan")
    return float(np.max(np.abs(u[mask] - ref[mask]) / np.abs(ref[mask])) * 100.0)


def _norm_linf(u, ref) -> float:
    if u is None or ref is None:
        return float("nan")
    return float(np.max(np.abs(u - ref)) / (np.max(np.abs(ref)) + 1e-300) * 100.0)


def _accuracy(u, ref) -> dict:
    if u is None or ref is None:
        return {}
    return {"max_rel_err": _max_rel(u, ref),
            "max_abs_err": float(np.max(np.abs(u - ref))),
            "rel_l2_err": float(np.linalg.norm(u - ref)
                                / (np.linalg.norm(ref) + 1e-300)),
            "rms_err": float(np.sqrt(np.mean((u - ref) ** 2))),
            "linf_err": _norm_linf(u, ref)}


def _electric_field(phi: np.ndarray, spacings):
    """
    E = -grad(phi) on the 3-D grid.

    Returns (Ez, Er, Es, |E|_max, |Ez|_max).  The axial component is singled
    out because it is what accelerates the ions and therefore sets thrust and
    specific impulse; an error in Ez matters more than the same error in phi.
    """
    Ez = -np.gradient(phi, spacings[0], axis=0)
    Er = -np.gradient(phi, spacings[1], axis=1)
    Es = -np.gradient(phi, spacings[2], axis=2)
    mag = np.sqrt(Ez**2 + Er**2 + Es**2)
    return Ez, Er, Es, float(np.max(mag)), float(np.max(np.abs(Ez)))


def _azimuthal_mode_amplitude(phi: np.ndarray, m: int) -> float:
    """
    Amplitude of azimuthal Fourier mode m, averaged over the (z, r) plane.

    Physically this is the quantity a spoke study reports.  It is a stricter
    test than a pointwise norm: a solver that damps or phase-shifts the
    azimuthal structure can still look acceptable in Linf while being useless
    for instability work.
    """
    if phi.shape[2] < 2 * max(m, 1):
        return float("nan")
    spectrum = np.fft.rfft(phi, axis=2)
    if m >= spectrum.shape[2]:
        return float("nan")
    return float(np.mean(np.abs(spectrum[:, :, m])) * 2.0 / phi.shape[2])


# ── Test cases ────────────────────────────────────────────────────────────────
#
# Every case's problem, exact/reference solution and source term now live in
# core/cases.py, read via cases.get(name).build(N) below - not built inline
# here as they used to be. The (case, N) -> case_id / azimuthal mode number
# mapping stays here, since that is HPC-sweep bookkeeping, not physics.

# ── Section 1: triple-sin MMS on the unit cube ────────────────────────────────

def case_cube(N: int):
    """Canonical 3-D Poisson verification case; see poisson_3d_triple_sin_cube."""
    built = cases.get("poisson_3d_triple_sin_cube").build(N)
    return (built.problem, built.exact, built.f_values,
            "3D_Poisson_TripleSin_cube", 0, built.f_faces)


# ── Section 2: HET channel MMS, azimuthally periodic ──────────────────────────

def case_het_mms(N: int, m: int = 1):
    """Manufactured solution on the unwrapped SPT-100 channel; see het_3d_mms_spt100."""
    built = cases.get("het_3d_mms_spt100").build(N)
    return (built.problem, built.exact, built.f_values,
            "3D_HET_MMS_SPT100", m, built.f_faces)


# ── Section 3: HET rotating spoke ─────────────────────────────────────────────

def case_spoke(N: int, m: int = SPOKE_MODE_M, eps: float = SPOKE_EPSILON):
    """Rotating-spoke potential structure; see het_3d_rotating_spoke."""
    built = cases.get("het_3d_rotating_spoke").build(N)
    return (built.problem, built.exact, built.f_values,
            "3D_HET_RotatingSpoke_SPT100", m, built.f_faces)


# ── Section 4: realistic SPT-100 discharge ────────────────────────────────────

def case_discharge(N: int, m: int = SPOKE_MODE_M):
    """Realistic SPT-100 discharge Poisson solve; see het_3d_discharge_spt100."""
    built = cases.get("het_3d_discharge_spt100").build(N)
    return (built.problem, built.exact, built.f_values,
            "3D_HET_Discharge_SPT100", m, built.f_faces)


# ── Section 5: Laplace equation, BC-driven ────────────────────────────────────

def case_laplace(N: int):
    """Harmonic, BC-driven case; see poisson_3d_laplace_bc_driven."""
    built = cases.get("poisson_3d_laplace_bc_driven").build(N)
    return (built.problem, built.exact, built.f_values,
            "3D_Laplace_BCdriven_cube", 0, built.f_faces)


# ── Section 6: localised Gaussian sources ─────────────────────────────────────

def case_gaussian(N: int):
    """Two opposite-sign Gaussian blobs; see poisson_3d_two_gaussian_cube."""
    built = cases.get("poisson_3d_two_gaussian_cube").build(N)
    return (built.problem, built.exact, built.f_values,
            "3D_Poisson_TwoGaussian_cube", 0, built.f_faces)


# ── Section 7: high-wavenumber Fourier mode ───────────────────────────────────

def case_highmode(N: int):
    """High-wavenumber eigenmode (n,m,l)=(2,3,4); see poisson_3d_high_mode_n2m3l4."""
    built = cases.get("poisson_3d_high_mode_n2m3l4").build(N)
    return (built.problem, built.exact, built.f_values,
            "3D_Poisson_HighMode_n2m3l4", 0, built.f_faces)


SECTIONS = {"section1": case_cube, "section2": case_het_mms,
            "section3": case_spoke, "section4": case_discharge,
            "section5": case_laplace, "section6": case_gaussian,
            "section7": case_highmode}


# ── Sweep settings ────────────────────────────────────────────────────────────

@dataclass
class SweepConfig:
    """Everything controlling a sweep, in one picklable object."""
    scheme:          str = DEFAULT_SCHEME
    tol:             float = DEFAULT_TOL
    max_outer:       int = 500
    solvers:         tuple[str, ...] = QUANTUM_SOLVERS
    inner_options:   dict = field(default_factory=dict)
    scheme_options:  dict = field(default_factory=dict)
    save_solutions:  bool = True
    save_history:    bool = True
    estimate_only:   bool = False
    order:           int = 2
    # Opt-in, OFF by default.  See the discussion in the module header of the
    # submit script: switching scheme with N makes the work-versus-N curve
    # discontinuous and conflates solver scaling with scheme scaling.
    scheme_crossover: Optional[int] = None

    def scheme_for(self, N: int) -> str:
        if self.scheme_crossover is not None and N <= self.scheme_crossover:
            if self.scheme in ("fmg", "multigrid"):
                return "sor"
        return self.scheme

    def inner_config(self, N: int) -> InnerConfig:
        cfg = InnerConfig()
        cfg["thomas"] = {}
        cfg["hhl"] = {"epsilon": HHL_EPSILON_DEFAULT}
        cap = QSVT_MAX_DEGREE_3D.get(N)
        cfg["qsvt"] = {} if cap is None else {"max_degree": cap}
        cfg["vqls"] = {}
        # The fourth-order registry entries are distinct solvers and need their
        # own sections: InnerConfig.for_solver keys on the name solve() is given,
        # so an absent section silently drops the sweep's epsilon and degree cap.
        cfg["hhl_4th"] = dict(cfg["hhl"])
        cfg["qsvt_4th"] = dict(cfg["qsvt"])
        for solver, opts in (self.inner_options or {}).items():
            cfg.setdefault(solver, {}).update(opts)
            # -I hhl.epsilon=... must reach whichever HHL this sweep runs.
            alias = f"{solver}_4th"
            if alias in cfg:
                cfg.setdefault(alias, {}).update(opts)
        return cfg

    def scheme_kwargs(self, scheme: str) -> dict:
        kw = dict(self.scheme_options or {})
        kw.setdefault("tol", self.tol)
        if scheme in ("multigrid", "fmg"):
            kw.pop("max_iter", None); kw.pop("criterion", None)
            kw.setdefault("max_cycles", min(self.max_outer, 200))
        else:
            kw.pop("max_cycles", None)
            kw.setdefault("max_iter", self.max_outer)
            kw.setdefault("criterion", "residual")
        return kw


# ── Result recording ──────────────────────────────────────────────────────────

def _save_solution_3d(case, solver, N, prob, phi, phi_ref, f_vals,
                      residual_history=None) -> None:
    """
    Archive the complete 3-D solution profile.

    Everything needed to reproduce any plot later without re-running: the
    full field, the reference, the source, the grid coordinates, the physical
    extents and the electric field components.  At N=32 a float64 field is
    262 kB, so compressed storage of all of this is cheap relative to the
    hours of quantum simulation that produced it.
    """
    fname = RESULTS_DIR / f"solution3d_{case}_{solver}_N{N}.npz"
    Xg, Yg, Zg = prob.grid()
    Ez, Er, Es, _, _ = _electric_field(phi, prob.spacings)
    arrays = {
        "phi": phi, "f": f_vals,
        "x0": Xg, "x1": Yg, "x2": Zg,
        "E0": Ez, "E1": Er, "E2": Es,
        "spacings": np.asarray(prob.spacings),
        "lengths": np.asarray(prob.lengths),
        "periodic": np.asarray(prob.periodic),
        "shape": np.asarray(prob.shape),
    }
    if phi_ref is not None:
        arrays["phi_exact"] = phi_ref
    if residual_history is not None:
        arrays["residual_history"] = np.asarray(residual_history, dtype=float)
    np.savez_compressed(fname, **arrays)


def _record(results, case_id, solver_name, N, prob, res, phi_ref, f_vals,
            phi_thomas, mode_m, cfg: SweepConfig, notes: str = "") -> None:
    label = SOLVER_LABEL.get(solver_name, solver_name.upper())
    shape_s = "x".join(str(n) for n in prob.shape)

    if res is None:
        results.append(RunResult3D(
            case=case_id, solver=label, N=N, shape=shape_s,
            n_unknowns=int(np.prod(prob.shape)), kappa_row=prob.kappa_row(),
            max_rel_err=None, max_abs_err=None, residual=None,
            wall_time_s=0.0, converged=False, n_outer=0,
            notes=notes or "solver_error", scheme=cfg.scheme))
        return

    phi = res.u
    d = res.diagnostics
    acc = _accuracy(phi, phi_ref)

    by_size = res.work.solves_by_size
    total = res.work.total
    alpha = COST_ALPHA.get(solver_name, 1.0)
    mean_size = (sum(n * k for n, k in by_size.items()) / total) if total else None

    hist = res.residual_history
    decades = (float(np.log10(hist[0] / hist[-1]))
               if len(hist) > 1 and hist[0] > 0 and hist[-1] > 0 else None)
    per_digit = (total / decades) if (decades and decades > 0) else None

    _, _, _, peak_E, peak_Ez = _electric_field(phi, prob.spacings)
    peak_E_err = None
    if phi_ref is not None:
        _, _, _, peak_ref, _ = _electric_field(phi_ref, prob.spacings)
        if peak_ref > 0:
            peak_E_err = float(abs(peak_E - peak_ref) / peak_ref * 100.0)

    amp = amp_err = None
    if prob.periodic[2] and mode_m:
        amp = _azimuthal_mode_amplitude(phi, mode_m)
        ref_field = phi_ref if phi_ref is not None else phi_thomas
        if ref_field is not None:
            amp_ref = _azimuthal_mode_amplitude(ref_field, mode_m)
            if amp_ref and np.isfinite(amp_ref) and abs(amp_ref) > 1e-30:
                amp_err = float(abs(amp - amp_ref) / abs(amp_ref) * 100.0)

    if solver_name == "thomas":
        err_vs_thomas = 0.0
    elif phi_thomas is not None:
        err_vs_thomas = _norm_linf(phi, phi_thomas)
    else:
        err_vs_thomas = None

    rho = res.convergence_factor
    results.append(RunResult3D(
        case=case_id, solver=label, N=N, shape=shape_s,
        n_unknowns=int(np.prod(prob.shape)), kappa_row=prob.kappa_row(),
        max_rel_err=acc.get("max_rel_err"), max_abs_err=acc.get("max_abs_err"),
        residual=res.residual, wall_time_s=res.wall_time_s,
        converged=res.converged, n_outer=res.n_outer, notes=notes,
        rel_l2_err=acc.get("rel_l2_err"), rms_err=acc.get("rms_err"),
        linf_err=acc.get("linf_err"), stop_reason=res.stop_reason,
        vqls_final_cost=d.get("final_cost_mean"),
        qsvt_degree=(int(d["polynomial_degree_mean"])
                     if d.get("polynomial_degree_mean") is not None else None),
        qsvt_depth=(int(d["circuit_depth_mean"])
                     if d.get("circuit_depth_mean") is not None else None),
        hhl_scale_c=d.get("prop_const_mean"),
        scheme=res.scheme,
        convergence_factor=(rho if (rho is not None and np.isfinite(rho)) else None),
        n_levels=d.get("n_levels"),
        level_shapes=json.dumps(d.get("level_shapes", []), default=str),
        level_kappas=json.dumps(d.get("level_kappas", []), default=str),
        anisotropy=float(max(prob.spacings) / min(prob.spacings)),
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
        err_thomas_vs_exact=(_norm_linf(phi_thomas, phi_ref)
                             if (phi_thomas is not None and phi_ref is not None)
                             else None),
        peak_E_field=peak_E, peak_E_rel_err=peak_E_err, peak_E_axial=peak_Ez,
        azimuthal_mode_amp=amp, azimuthal_mode_rel_err=amp_err,
        s_per_strip_solve=(res.wall_time_s / total) if total else None,
        solves_per_digit=per_digit))

    if cfg.save_solutions:
        _save_solution_3d(case_id, label, N, prob, phi, phi_ref, f_vals,
                          res.residual_history if cfg.save_history else None)


# ── Per-case driver ───────────────────────────────────────────────────────────

def _estimate_case(case_id, N, prob, cfg: SweepConfig) -> None:
    # A multigrid scheme needs two levels, and a 4x4x4 grid cannot be coarsened at
    # all, so solve() raises. _run_case already falls back to line-SOR for exactly
    # this; without the same fallback here, --estimate aborted at N=4 - on the very
    # sweep the estimate is meant to de-risk.
    scheme = cfg.scheme_for(N)
    if scheme in ("multigrid", "fmg") and len(build_hierarchy(prob)) < 2:
        scheme = "sor"
    res = solve(prob, inner="thomas", scheme=scheme,
                inner_options=cfg.inner_config(N), **cfg.scheme_kwargs(scheme))
    by_size = res.work.solves_by_size
    log.info("    %-34s N=%-3d  %d outer, %d strip solves %s",
             case_id[:34], N, res.n_outer, res.work.total,
             dict(sorted(by_size.items(), reverse=True)))
    for s in cfg.solvers:
        alpha, t8 = COST_ALPHA.get(s, 1.0), COST_T8.get(s, 1.0)
        secs = sum(k * t8 * (n / 8.0) ** alpha for n, k in by_size.items())
        log.info("        projected %-5s %11.1f s  (%7.2f h)",
                 s.upper(), secs, secs / 3600.0)


# ── Fourth order ──────────────────────────────────────────────────────────────
# Under --order 4 the runner selects a different problem object and then takes
# the ordinary solve() path. The retired 4th-order branch carried its own
# iteration loop, and in consequence the fourth-order sweeps silently lost the
# wall-clock cap, stagnation detection, level_kappas and every inner_*
# diagnostic, and counted work as w.add(N, N*iters) against 2-D's w.add(N,
# iters) - mutually inconsistent, and both wrong. All of that is inherited here.

#: Registry names of the inner solvers that must change with the order. Thomas
#: is absent because its entry already falls back to a dense solve on a
#: pentadiagonal matrix, and VQLS because it decomposes the complete matrix into
#: Paulis and was never affected by the truncation.
_INNER_BY_ORDER_4: dict[str, str] = {"hhl": "hhl_4th", "qsvt": "qsvt_4th"}


def _inner_for_order(solver: str, order: int) -> str:
    """
    The inner-solver registry name to use at this discretisation order.

    Parameters
    ----------
    solver : str
        Solver as named on the command line, e.g. ``"hhl"``.
    order : {2, 4}
        Spatial discretisation order.

    Returns
    -------
    str
        Registry name, e.g. ``"hhl_4th"`` at order 4.
    """
    return _INNER_BY_ORDER_4.get(solver, solver) if order == 4 else solver


def _to_4th_order_3d(problem, f_faces=None) -> PoissonLine3D4th:
    """
    Re-discretises an assembled 3-D case at fourth order.

    The continuous problem is unchanged - same box, same source samples, same
    boundary data, same periodicity - so only the operator and the boundary
    closure differ.

    Parameters
    ----------
    problem : PoissonLine3D
        The second-order problem built by the case registry.
    f_faces : tuple, optional
        ``(lo, hi)`` per-axis source values *on* the faces, as
        ``BuiltCase.f_faces`` carries them, with None for a periodic axis.
        Required data for the fourth-order closure; where a case does not
        supply them the problem class extrapolates from the interior, which is
        O(h⁴) and order-preserving but inaccurate for a sharply peaked source
        at coarse N.

    Returns
    -------
    PoissonLine3D4th
        The same case at fourth order.
    """
    lo, hi = f_faces if f_faces is not None else (None, None)
    return PoissonLine3D4th(
        problem.f, lengths=problem.lengths, periodic=problem.periodic,
        bc_lo=problem.bc_lo, bc_hi=problem.bc_hi,
        f_lo=lo, f_hi=hi,
    )


def _run_case(section: str, N: int, cfg: SweepConfig, results: list) -> None:
    prob, phi_ref, f_vals, case_id, mode_m, f_faces = SECTIONS[section](N)
    if cfg.order == 4:
        prob = _to_4th_order_3d(prob, f_faces)
    inner_cfg = cfg.inner_config(N)

    if cfg.estimate_only:
        _estimate_case(case_id, N, prob, cfg)
        return

    levels = build_hierarchy(prob)
    scheme = cfg.scheme_for(N)

    # Multigrid needs at least two levels; at N=4 nothing can be coarsened.
    # solve() raises rather than degrading silently, which is right for a
    # library call but would abort the whole work unit here.
    fallback_note = ""
    if scheme in ("multigrid", "fmg") and len(levels) < 2:
        fallback_note = f"scheme_fallback:{scheme}->sor"
        scheme = "sor"
        log.warning("    [%s N=%d] %s cannot be coarsened; falling back to sor.",
                   case_id, N, prob.shape)
    if scheme != cfg.scheme and not fallback_note:
        fallback_note = f"scheme_crossover:{cfg.scheme}->{scheme}"

    kw = cfg.scheme_kwargs(scheme)

    log.info("    %s  grid %s = %s unknowns  kappa=%.4f  aniso=%.1f  scheme=%s",
             case_id, "x".join(map(str, prob.shape)),
             f"{int(np.prod(prob.shape)):,}", prob.kappa_row(),
             max(prob.spacings) / min(prob.spacings), scheme)
    log.info("    hierarchy: %s", " -> ".join(
        "x".join(map(str, lv.problem.shape)) for lv in levels))

    # ── classical reference ───────────────────────────────────────────────────
    res_T = solve(prob, inner="thomas", scheme=scheme,
                  inner_options=inner_cfg, **kw)
    phi_T = res_T.u
    # Every per-solver line is tagged with [case N=..]. This is not cosmetic:
    # each worker process opens its own handle to the shared log file (see
    # the logging setup above), so with --max-workers > 1 lines from
    # different (case, N) work units interleave by wall-clock arrival, not
    # by logical grouping. A long HHL/VQLS solve (minutes) on one worker
    # will have several OTHER workers' banners and results land in between
    # it and the section banner that visually precedes it. Without an
    # explicit tag, two adjacent-looking lines can belong to different
    # cases entirely - always filter results_full.json / results_summary.csv
    # by (case, N) for an authoritative comparison, never the interleaved
    # console/log order.
    # Where no closed form exists (section 4), Thomas *is* the reference, so
    # it must be compared and recorded against itself rather than against
    # None - otherwise both the console line and the accuracy columns come
    # out NaN for a run that in fact succeeded. This must be computed BEFORE
    # the console print below, not after: printing against phi_ref directly
    # gives NaN whenever phi_ref is None, even though _record() further down
    # already used the correct fallback.
    reference = phi_ref if phi_ref is not None else phi_T
    ref_note = fallback_note or ("" if phi_ref is not None else "rel_vs_thomas")

    log.info("    [%s N=%d] %-6s %5d outer  %9d solves  err=%8.4f%%  %8.2fs  %s",
             case_id, N, "Thomas", res_T.n_outer, res_T.work.total,
             _norm_linf(phi_T, reference), res_T.wall_time_s, res_T.stop_reason)

    _record(results, case_id, "thomas", N, prob, res_T, reference, f_vals,
            phi_T, mode_m, cfg,
            notes=fallback_note or ("" if phi_ref is not None else "reference"))

    # ── quantum solvers ───────────────────────────────────────────────────────
    for solver_name in cfg.solvers:
        try:
            res_q = solve(prob, inner=_inner_for_order(solver_name, cfg.order),
                          scheme=scheme, inner_options=inner_cfg, **kw)
        except Exception as exc:
            log.error("    [%s N=%d] %-6s FAILED: %s",
                      case_id, N, solver_name.upper(), exc, exc_info=True)
            _record(results, case_id, solver_name, N, prob, None, reference,
                    f_vals, phi_T, mode_m, cfg, notes=str(exc)[:200])
            continue
        fb = res_q.diagnostics.get("inner_failures", 0)
        log.info("    [%s N=%d] %-6s %5d outer  %9d solves  err=%8.4f%%  "
                 "vs_Thomas=%7.4f%%  %9.2fs  %s%s",
                 case_id, N, solver_name.upper(), res_q.n_outer, res_q.work.total,
                 _norm_linf(res_q.u, reference), _norm_linf(res_q.u, phi_T),
                 res_q.wall_time_s, res_q.stop_reason,
                 f"  [{fb} classical fallbacks]" if fb else "")
        _record(results, case_id, solver_name, N, prob, res_q, reference,
                f_vals, phi_T, mode_m, cfg, notes=ref_note)


def run_section(section: str, N: int, cfg: SweepConfig, results: list) -> None:
    titles = {"section1": "3-D Poisson, triple-sin MMS, unit cube",
              "section2": "HET channel MMS (SPT-100, azimuthally periodic)",
              "section3": "HET rotating spoke (SPT-100)",
              "section4": "HET realistic discharge (SPT-100, 300 V)",
              "section5": "Laplace, BC-driven, unit cube",
              "section6": "Two-Gaussian source, non-homogeneous BC, unit cube",
              "section7": "High-wavenumber mode (2, 3, 4), unit cube"}
    _banner(f"{section.upper()} - {titles[section]}, N={N}")
    _run_case(section, N, cfg, results)


# ── Serialisation ─────────────────────────────────────────────────────────────

def _load_existing_results(path: Path) -> list:
    if not path.exists():
        return []
    try:
        with open(path) as fh:
            rows = json.load(fh)
    except Exception as exc:
        log.warning("Could not parse %s (%s); starting fresh.", path, exc)
        return []
    valid = {f.name for f in dataclasses.fields(RunResult3D)}
    out = []
    for d in rows:
        try:
            out.append(RunResult3D(**{k: v for k, v in d.items() if k in valid}))
        except Exception as exc:
            log.warning("Skipping unreadable prior row: %s", exc)
    return out


def _row_key(r) -> tuple:
    """
    Identity of a benchmark row for supersession purposes.

    A sweep is uniquely indexed by (case, solver, N); a second row for that triple
    is a recomputation of the first, not an additional datum.

    `scheme` is deliberately excluded. A row recorded under the N<=4 line-SOR
    fallback must be superseded by its rerun under FMG; keying on the scheme would
    retain both and leave the degraded row in the summary beside the good one.

    Parameters
    ----------
    r : RunResult3D
        The row to key.

    Returns
    -------
    tuple
        Hashable identity: (case, solver, N).
    """
    return (r.case, r.solver, r.N)


def _dedupe_results(results) -> list:
    """
    Collapse superseded rows, retaining the most recent for each identity.

    `--append` merges the previous ``results_full.json`` ahead of the rows produced
    by the current invocation, so the later occurrence of any key is by
    construction the newer measurement. Without this step, re-running a step after
    a failure - or any wave that revisits a partially-completed (section, N) -
    wrote a second row for every solver already present, and the plotting and
    gap-analysis layers had no basis on which to choose between them.

    Parameters
    ----------
    results : list of RunResult3D
        Rows in chronological order, oldest first.

    Returns
    -------
    list of RunResult3D
        Deduplicated rows, in first-seen order so the file's layout stays stable
        across appends rather than reshuffling on every save.
    """
    keep: dict = {}
    for r in results:
        keep[_row_key(r)] = r
    if len(keep) != len(results):
        log.info("Superseded %d duplicate row(s) on (case, solver, N).",
                 len(results) - len(keep))
    return list(keep.values())


def _save_results(results) -> None:
    if not results:
        return
    results = _dedupe_results(results)
    with open(RESULTS_DIR / "results_full.json", "w") as fh:
        json.dump([asdict(r) for r in results], fh, indent=2, default=str)
    names = [f.name for f in dataclasses.fields(RunResult3D)]
    with open(RESULTS_DIR / "results_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=names); w.writeheader()
        for r in results:
            w.writerow(asdict(r))
    log.info("Results written: %d rows -> %s", len(results), RESULTS_DIR.resolve())


def _save_metadata(N_values, cfg, sections, max_workers, tag=None) -> None:
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "hostname": platform.node(), "python": sys.version,
        "numpy": np.__version__, "cpu_count": os.cpu_count(),
        "pbs_jobid": os.environ.get("PBS_JOBID"),
        "slurm_jobid": os.environ.get("SLURM_JOB_ID"),
        "dimension": 3, "N_values": N_values, "sections": sections,
        "max_workers": max_workers, "sweep_config": asdict(cfg),
        "qsvt_max_degree_default": {str(k): v for k, v in QSVT_MAX_DEGREE_3D.items()},
        "hhl_epsilon_default": HHL_EPSILON_DEFAULT,
        "cost_model_alpha": COST_ALPHA, "cost_model_t8_s": COST_T8,
        "spt100": {"Lz_m": HET_LZ, "Lr_m": HET_LR, "Ls_m": HET_LS,
                   "r_in_m": HET_R_IN, "r_out_m": HET_R_OUT,
                   "V_anode": HET_V_ANODE, "V_cathode": HET_V_CATHODE,
                   "V_wall": HET_V_WALL, "spoke_m": SPOKE_MODE_M,
                   "spoke_eps": SPOKE_EPSILON},
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


# ── Summary ───────────────────────────────────────────────────────────────────

def _print_summary(results) -> None:
    if not results:
        return
    # Deduplicated for the same reason the saved summary is: under --append the
    # in-memory list still carries the superseded rows, and printing both invites
    # the reader to compare a stale measurement against its own replacement.
    results = _dedupe_results(results)
    _banner("SUMMARY")
    log.info("  %-32s %-7s %4s %6s %10s %10s %9s %9s",
             "case", "solver", "N", "outer", "solves", "w.cost", "err%", "vsThom%")
    log.info("  " + "-" * 94)
    for r in sorted(results, key=lambda r: (r.case, r.N, r.solver)):
        err = "   FAILED" if r.linf_err is None else f"{r.linf_err:9.4f}"
        vt = "        -" if r.err_vs_thomas is None else f"{r.err_vs_thomas:9.4f}"
        wc = "         -" if r.weighted_cost is None else f"{r.weighted_cost:10.0f}"
        log.info("  %-32s %-7s %4d %6d %10d %10s %9s %9s",
                 r.case[:32], r.solver, r.N, r.n_outer, r.strip_solves,
                 wc, err, vt)

    spoke = [r for r in results if r.azimuthal_mode_rel_err is not None]
    if spoke:
        _section("Azimuthal mode fidelity (does the solver reproduce the spoke?)")
        for r in sorted(spoke, key=lambda r: (r.case, r.N, r.solver)):
            log.info("  %-32s %-7s N=%-3d  amp=%.4g  err=%.4f%%",
                     r.case[:32], r.solver, r.N,
                     r.azimuthal_mode_amp, r.azimuthal_mode_rel_err)

    _section("Quantum cost relative to Thomas (same scheme)")
    by_key: dict = {}
    for r in results:
        by_key.setdefault((r.case, r.N), {})[r.solver] = r
    for (case, N), d in sorted(by_key.items()):
        base = d.get("Thomas")
        if base is None or not base.wall_time_s:
            continue
        parts = [f"{s}={d[s].wall_time_s / base.wall_time_s:,.0f}x"
                 for s in ("HHL", "VQLS", "QSVT") if s in d and d[s].wall_time_s]
        if parts:
            log.info("  %-32s N=%-3d  %s", case[:32], N, "   ".join(parts))

    stalled = [r for r in results if r.stop_reason == "stagnated"]
    if stalled:
        _section(f"{len(stalled)} run(s) stopped at the inner solver's error floor")
        for r in stalled:
            log.info("  %-32s %-7s N=%-3d  residual %.2e",
                     r.case[:32], r.solver, r.N, r.residual or float("nan"))

    fell = [r for r in results if r.inner_failures]
    if fell:
        _section("Runs with classical fallbacks")
        for r in fell:
            log.info("  %-32s %-7s N=%-3d  %d/%d calls (%.1f%%)",
                     r.case[:32], r.solver, r.N, r.inner_failures,
                     r.inner_calls, 100.0 * r.inner_failures / max(r.inner_calls, 1))


# ── CLI plumbing ──────────────────────────────────────────────────────────────

def parse_kv(items, flag: str) -> dict:
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
    out = {}
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


def _init_worker(results_dir: Path, qsvt_caps: dict) -> None:
    """
    Propagates the main process's resolved configuration into a worker.

    `ProcessPoolExecutor` is constructed with ``max_tasks_per_child=1``, and CPython
    selects the **spawn** start method whenever that argument is given without an
    explicit context. A spawned worker re-imports this module, so it sees the
    module-level defaults rather than the values `main` assigned under ``global`` —
    and `_save_solution_3d` reads `RESULTS_DIR` from inside the worker.

    Left unset, an ``--order 4`` sweep wrote its summary to ``results/3Dhpc_run_4th``
    from the parent whilst every per-solution archive went to ``results/3Dhpc_run``
    from the workers, overwriting 2nd-order fields with 4th-order ones.

    Parameters
    ----------
    results_dir : Path
        Output directory resolved by `main`, including the 4th-order variant.
    qsvt_caps : dict
        Per-N QSVT degree caps as resolved by `main`, which uncaps N=4 at 4th order.
    """
    global RESULTS_DIR
    RESULTS_DIR = results_dir
    QSVT_MAX_DEGREE_3D.clear()
    QSVT_MAX_DEGREE_3D.update(qsvt_caps)


def _execute_work_unit(section: str, N: int, cfg: SweepConfig) -> list:
    results: list = []
    if section not in SECTIONS:
        log.error("Unknown section %r", section)
        return results
    try:
        run_section(section, N, cfg, results)
    except Exception as exc:
        log.error("Section %s N=%d aborted: %s", section, N, exc, exc_info=True)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full 3-D HPC benchmark sweep for quantum PDE solvers.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--order", type=int, choices=[2, 4], default=2,
                    help="Spatial discretisation order (2 or 4).")
    ap.add_argument("--max-n", type=int, default=16,
                    help="largest grid size (default %(default)s; 3-D sweeps "
                         "cost N^2 strip solves per sweep, so raise with care)")
    ap.add_argument("--n-values", type=str, default=None)
    ap.add_argument("--sections", type=str, default="1,2,3,4,5,6,7",
                    help="1-4 as before (1 cube, 2 HET MMS, 3 spoke, "
                         "4 discharge); 5 Laplace/BC-driven, "
                         "6 two-Gaussian, 7 high-wavenumber mode")
    ap.add_argument("--solvers", type=str, default=",".join(QUANTUM_SOLVERS))
    ap.add_argument("--skip-qsvt", action="store_true")

    ap.add_argument("--scheme", default=DEFAULT_SCHEME, choices=available_schemes(),
                    help="outer scheme for the whole sweep (default %(default)s)")
    ap.add_argument("--scheme-crossover", type=int, default=None,
                    help="OPT-IN: below and including this N, use SOR instead "
                         "of the multigrid scheme. Off by default because it "
                         "makes the work-versus-N curve discontinuous and "
                         "conflates solver scaling with scheme scaling. Rows "
                         "affected are tagged scheme_crossover in notes.")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    ap.add_argument("--max-outer", type=int, default=500)

    ap.add_argument("-I", "--inner-opt", action="append", metavar="SOLVER.KEY=VAL")
    ap.add_argument("-S", "--scheme-opt", action="append", metavar="KEY=VAL")
    ap.add_argument("--list-options", action="store_true")

    ap.add_argument("--estimate", action="store_true",
                    help="classical only; project quantum wall time. Run first.")
    ap.add_argument("--no-solutions", action="store_true")
    ap.add_argument("--max-workers", type=int, default=MAX_WORKERS_DEFAULT)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--phase-tag", default=None)
    args = ap.parse_args()

    if args.list_options:
        print("\n=== INNER SOLVER OPTIONS ===\n"); print(describe_inner())
        print("\n=== OUTER SCHEME OPTIONS ===\n"); print(describe_scheme())
        return

    if args.n_values:
        N_values = [int(v) for v in args.n_values.split(",") if v.strip()]
    else:
        N_values = [n for n in N_VALUES_ALL if n <= args.max_n]
    if not N_values:
        ap.error(f"--max-n {args.max_n} excludes every N in {N_VALUES_ALL}")
    bad = [n for n in N_values if n < 4 or (n & (n - 1))]
    if bad:
        ap.error(f"N must be a power of two and >= 4 (quantum register and "
                 f"multigrid coarsening both require it); got {bad}")

    sections = [f"section{s.strip()}" for s in args.sections.split(",") if s.strip()]
    unknown = [s for s in sections if s not in SECTIONS]
    if unknown:
        ap.error(f"unknown section(s) {unknown}; valid: {sorted(SECTIONS)}")

    solvers = tuple(s.strip().lower() for s in args.solvers.split(",") if s.strip())
    if args.skip_qsvt:
        solvers = tuple(s for s in solvers if s != "qsvt")
    bad = [s for s in solvers if s not in available_inner()]
    if bad:
        ap.error(f"unknown solver(s) {bad}; available: {available_inner()}")

    inner_opts = parse_kv(args.inner_opt, "--inner-opt")
    flat = {k: v for k, v in inner_opts.items() if not isinstance(v, dict)}
    if flat:
        ap.error(f"--inner-opt must be namespaced by solver, "
                 f"e.g. -I qsvt.{list(flat)[0]}=...")
    bad = [k for k in inner_opts if k not in available_inner()]
    if bad:
        ap.error(f"--inner-opt refers to unknown solver(s) {bad}")

    cfg = SweepConfig(
        scheme=args.scheme, tol=args.tol, max_outer=args.max_outer,
        solvers=solvers, inner_options=inner_opts,
        scheme_options=coerce_scheme_opts(parse_kv(args.scheme_opt, "--scheme-opt")),
        save_solutions=not args.no_solutions, estimate_only=args.estimate,
        scheme_crossover=args.scheme_crossover, order=args.order)

    # Pre-flight: validate options (pure, no imports), then build the solvers
    # (which does import the quantum backends).  Kept separate so a missing
    # backend is reported as an environment problem, not a bad option.
    probe = cfg.inner_config(N_values[0])
    for name in ("thomas",) + tuple(solvers):
        try:
            resolve_options(name, probe.for_solver(name))
        except Exception as exc:
            ap.error(f"inner solver {name!r} rejected its configuration: {exc}")
    if not cfg.estimate_only:
        for name in ("thomas",) + tuple(solvers):
            try:
                get_inner(name, **probe.for_solver(name))
            except ImportError as exc:
                ap.error(f"inner solver {name!r} needs a backend that is not "
                         f"installed: {exc}. Check the virtualenv is active, "
                         f"install it, drop it from --solvers, or --estimate.")
            except Exception as exc:
                ap.error(f"inner solver {name!r} could not be built: {exc}")

    global RESULTS_DIR, LOG_FILE
    
    if args.order == 4:
        RESULTS_DIR = Path("results") / "3Dhpc_run_4th"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        LOG_FILE = RESULTS_DIR / "run.log"
        _redirect_log_file(LOG_FILE)
        
        # Uncap N=4 for 4th order QSVT
        if 4 in QSVT_MAX_DEGREE_3D:
            QSVT_MAX_DEGREE_3D[4] = None

    _log_session_header(args.phase_tag)
    _banner("QUANTUM PDE SOLVERS - 3D HPC BENCHMARK")
    log.info("  N values    : %s", N_values)
    log.info("  Sections    : %s", sections)
    log.info("  Scheme      : %s   tol=%.1e   max_outer=%d",
             cfg.scheme, cfg.tol, cfg.max_outer)
    if cfg.scheme_crossover:
        log.info("  Crossover   : SOR for N <= %d (opt-in)", cfg.scheme_crossover)
    log.info("  Solvers     : %s", list(solvers))
    log.info("  Inner opts  : %s", inner_opts or "(defaults)")
    log.info("  Workers     : %d", args.max_workers)
    log.info("  Output      : %s", RESULTS_DIR.resolve())
    if cfg.estimate_only:
        log.info("  ESTIMATE MODE - no quantum solver will be executed")

    _save_metadata(N_values, cfg, sections, args.max_workers)
    if args.phase_tag:
        _save_metadata(N_values, cfg, sections, args.max_workers, tag=args.phase_tag)

    t0 = time.perf_counter()
    results: list = []
    if args.append:
        prior = _load_existing_results(RESULTS_DIR / "results_full.json")
        if prior:
            log.info("--append: merging with %d prior row(s)", len(prior))
        results.extend(prior)

    work_units = [(s, N) for N in sorted(N_values) for s in sections]
    if args.max_workers <= 1 or cfg.estimate_only:
        log.info("Serial execution: %d units.", len(work_units))
        for section, N in work_units:
            results.extend(_execute_work_unit(section, N, cfg))
            _save_results(results)
    else:
        log.info("Parallel execution: %d units over %d workers.",
                 len(work_units), args.max_workers)
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.max_workers, max_tasks_per_child=1,
                initializer=_init_worker,
                initargs=(RESULTS_DIR, dict(QSVT_MAX_DEGREE_3D))) as ex:
            futures = {ex.submit(_execute_work_unit, s, N, cfg): (s, N)
                       for s, N in work_units}
            for fut in concurrent.futures.as_completed(futures):
                section, N = futures[fut]
                try:
                    results.extend(fut.result())
                    log.info("Done: %-10s N=%-3d  (%d rows so far)",
                             section, N, len(results))
                    _save_results(results)
                except Exception as exc:
                    log.error("Failed: %s N=%d - %s", section, N, exc,
                              exc_info=True)

    _save_results(results)
    if not cfg.estimate_only:
        _print_summary(results)
    _banner(f"Complete in {time.perf_counter() - t0:.1f} s  "
            f"({len(results)} rows) -> {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()