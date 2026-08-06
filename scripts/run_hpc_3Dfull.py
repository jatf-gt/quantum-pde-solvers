#!/usr/bin/env python3
"""
run_hpc_3Dfull.py
=================
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
    python scripts/run_hpc_3Dfull.py --max-n 16
    python scripts/run_hpc_3Dfull.py --max-n 32 -I qsvt.max_degree=300
    python scripts/run_hpc_3Dfull.py --max-n 32 --estimate     # cost first!
    python scripts/run_hpc_3Dfull.py --n-values 32 --solvers qsvt --append
    python scripts/run_hpc_3Dfull.py --list-options
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

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from solvers.outer import (InnerConfig, PoissonLine3D, available_inner,
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
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(LOG_FILE, mode="w" if _IS_MAIN_PROCESS else "a")],
)
log = logging.getLogger(__name__)

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
#  Cartesian slab: the channel is thin (15 mm wide) compared with its mean
#  circumference (267 mm), so curvature is a second-order effect and the
#  azimuthal direction becomes a straight, periodic coordinate.  This is the
#  geometry used in axial-azimuthal PIC studies of HET instabilities.
#
#      axis 0  z  axial       [0, 25 mm]   Dirichlet: anode / cathode plane
#      axis 1  r  radial      [0, 15 mm]   Dirichlet: inner / outer wall
#      axis 2  s  azimuthal   [0, 267 mm]  PERIODIC (s = r_mean * theta)
#
#  Axis 0 is the strip direction and must be Dirichlet: a periodic strip
#  operator is cyclic-tridiagonal, not TST, which the block encoding assumes.
HET_LZ: float = 0.025                        # channel length, m
HET_R_IN, HET_R_OUT = 0.035, 0.050           # channel radii, m
HET_LR: float = HET_R_OUT - HET_R_IN         # channel width, m
HET_R_MEAN: float = 0.5 * (HET_R_IN + HET_R_OUT)
HET_LS: float = 2.0 * np.pi * HET_R_MEAN     # mean circumference, m

HET_PHI0: float = 300.0        # discharge voltage, V (SPT-100 nominal)
HET_V_ANODE: float = 300.0     # anode potential, V
HET_V_CATHODE: float = 0.0     # cathode-plane potential, V
HET_V_WALL: float = -20.0      # wall potential, V (floating, ~ -few Te)

SPOKE_MODE_M: int = 2          # azimuthal mode number of the rotating spoke
SPOKE_EPSILON: float = 0.30    # relative amplitude of the azimuthal modulation

EPS0: float = 8.854e-12        # F/m
Q_E: float = 1.602e-19         # C

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

def _het_grid(N: int):
    """Unwrapped SPT-100 channel grid.  Azimuthal axis has no boundary node."""
    dz, dr = HET_LZ / (N + 1), HET_LR / (N + 1)
    ds = HET_LS / N
    z = np.arange(1, N + 1) * dz
    r = np.arange(1, N + 1) * dr
    s = np.arange(N) * ds
    return np.meshgrid(z, r, s, indexing="ij"), (dz, dr, ds), (z, r, s)


# ── Section 1: triple-sin MMS on the unit cube ────────────────────────────────

def case_cube(N: int):
    """
    Canonical 3-D Poisson verification case:

        phi = sin(pi x) sin(pi y) sin(pi z)
        f   = nabla^2 phi = -3 pi^2 sin(pi x) sin(pi y) sin(pi z)

    Homogeneous Dirichlet on all six faces.  Exact solution, so this is what
    the order-of-accuracy check is run on.
    """
    h = 1.0 / (N + 1)
    p = np.arange(1, N + 1) * h
    X, Y, Z = np.meshgrid(p, p, p, indexing="ij")
    phi = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sin(np.pi * Z)
    f = -3.0 * np.pi**2 * phi
    prob = PoissonLine3D(f, lengths=(1.0, 1.0, 1.0))
    return prob, phi, f, "3D_Poisson_TripleSin_cube", 0


# ── Section 2: HET channel MMS, azimuthally periodic ──────────────────────────

def case_het_mms(N: int, m: int = 1):
    """
    Manufactured solution on the unwrapped SPT-100 channel:

        phi = phi0 sin(pi z/Lz) sin(pi r/Lr) cos(2 pi m s/Ls)
        f   = -phi0 pi^2 (1/Lz^2 + 1/Lr^2 + 4 m^2/Ls^2) * (same profile)

    Zero at anode, cathode and both walls; exactly periodic in s, so the
    periodic stencil and the periodic grid-transfer operators are genuinely
    exercised rather than merely present.  Verification case with the real
    channel aspect ratio, which is severely anisotropic
    (ds/dr ~ 19 at N=16) and therefore the case that actually tests the
    anisotropic semi-coarsening in the multigrid hierarchy.
    """
    (Zg, Rg, Sg), sp, _ = _het_grid(N)
    profile = (np.sin(np.pi * Zg / HET_LZ) * np.sin(np.pi * Rg / HET_LR)
               * np.cos(2.0 * np.pi * m * Sg / HET_LS))
    phi = HET_PHI0 * profile
    lap = -HET_PHI0 * np.pi**2 * (1.0 / HET_LZ**2 + 1.0 / HET_LR**2
                                  + 4.0 * m**2 / HET_LS**2)
    f = lap * profile
    prob = PoissonLine3D(f, lengths=(HET_LZ, HET_LR, HET_LS),
                         periodic=(False, False, True))
    return prob, phi, f, "3D_HET_MMS_SPT100", m


# ── Section 3: HET rotating spoke ─────────────────────────────────────────────

def case_spoke(N: int, m: int = SPOKE_MODE_M, eps: float = SPOKE_EPSILON):
    """
    Rotating-spoke potential structure in the discharge channel.

    The "rotating spoke" is a large-scale, low-mode-number (m = 1..6)
    coherent azimuthal structure observed in essentially every Hall thruster
    since Janes & Lowder (1966), and characterised in detail by
    McDonald & Gallimore (2011) and Sekerak et al. (2015).  It rotates in the
    E x B direction at a few km/s and carries a substantial fraction of the
    discharge current, so capturing it is the single strongest physical
    argument for simulating a HET in three dimensions rather than two.

    Manufactured so an exact solution exists:

        phi = phi0 sin(pi z/Lz) sin(pi r/Lr) [1 + eps cos(2 pi m s/Ls)]

    i.e. an axial-radial potential well modulated azimuthally at relative
    amplitude eps.  Applying the Laplacian term by term:

        nabla^2 phi = phi0 sin(pi z/Lz) sin(pi r/Lr) *
                      { -(pi^2/Lz^2 + pi^2/Lr^2) [1 + eps cos(2 pi m s/Ls)]
                        - eps (2 pi m/Ls)^2 cos(2 pi m s/Ls) }

    The first group is the unmodulated well; the second is the azimuthal
    curvature of the spoke.  Because the mode amplitude is known exactly,
    ``azimuthal_mode_rel_err`` measures directly whether a quantum solver
    reproduces the spoke or merely smears it.
    """
    (Zg, Rg, Sg), sp, _ = _het_grid(N)
    base = np.sin(np.pi * Zg / HET_LZ) * np.sin(np.pi * Rg / HET_LR)
    azim = np.cos(2.0 * np.pi * m * Sg / HET_LS)
    phi = HET_PHI0 * base * (1.0 + eps * azim)
    f = HET_PHI0 * base * (
        -(np.pi**2 / HET_LZ**2 + np.pi**2 / HET_LR**2) * (1.0 + eps * azim)
        - eps * (2.0 * np.pi * m / HET_LS) ** 2 * azim)
    prob = PoissonLine3D(f, lengths=(HET_LZ, HET_LR, HET_LS),
                         periodic=(False, False, True))
    return prob, phi, f, "3D_HET_RotatingSpoke_SPT100", m


# ── Section 4: realistic SPT-100 discharge ────────────────────────────────────

def case_discharge(N: int, m: int = SPOKE_MODE_M):
    """
    Realistic SPT-100 discharge Poisson solve - the production case.

    Solves  nabla^2 phi = -rho / eps0  with the actual operating boundary
    conditions of an SPT-100 at nominal 300 V:

        anode   (z = 0)      phi = +300 V
        cathode (z = Lz)     phi =    0 V
        walls   (r = 0, Lr)  phi =  -20 V   (floating, a few Te below plasma)
        azimuthal            periodic

    The source is the net space charge in the acceleration region.  The bulk
    plasma is quasi-neutral, but charge separation develops near the exit
    plane where the ions are accelerated out faster than the electrons can
    follow; that region is modelled here as a Gaussian in z centred at the
    exit plane, tapered radially, and modulated azimuthally by the spoke:

        n_diff(z,r,s) = n0 exp(-(z - z_acc)^2 / 2 sigma_z^2)
                             sin(pi r/Lr) [1 + eps cos(2 pi m s/Ls)]
        rho = q_e n_diff,     f = -rho / eps0

    with n0 = 1e16 m^-3, about 1 % of the ~1e18 m^-3 bulk density, giving a
    space-charge potential perturbation of order 10 V on top of the 300 V
    applied - the correct order of magnitude for a real device.

    There is no closed-form solution, so the Thomas result is the reference.
    This is the case whose cost actually predicts what a quantum-in-the-loop
    HET simulation would pay per timestep.
    """
    (Zg, Rg, Sg), sp, _ = _het_grid(N)
    z_acc = 0.8 * HET_LZ          # acceleration region sits near the exit plane
    sigma_z = 0.12 * HET_LZ
    n0 = 1.0e16                   # peak net charge-carrier density, m^-3

    n_diff = (n0 * np.exp(-((Zg - z_acc) ** 2) / (2.0 * sigma_z**2))
              * np.sin(np.pi * Rg / HET_LR)
              * (1.0 + SPOKE_EPSILON * np.cos(2.0 * np.pi * m * Sg / HET_LS)))
    f = -(Q_E * n_diff) / EPS0

    Nz = N
    bc_anode = np.full((N, N), HET_V_ANODE)      # face at z = 0,  shape (r, s)
    bc_cathode = np.full((N, N), HET_V_CATHODE)  # face at z = Lz
    bc_wall_in = np.full((N, N), HET_V_WALL)     # face at r = 0,  shape (z, s)
    bc_wall_out = np.full((N, N), HET_V_WALL)    # face at r = Lr

    prob = PoissonLine3D(
        f, lengths=(HET_LZ, HET_LR, HET_LS),
        periodic=(False, False, True),
        bc_lo=(bc_anode, bc_wall_in, 0.0),
        bc_hi=(bc_cathode, bc_wall_out, 0.0))
    return prob, None, f, "3D_HET_Discharge_SPT100", m


# ── Section 5: Laplace equation, BC-driven ────────────────────────────────────

def case_laplace(N: int):
    """
    Laplace equation: homogeneous PDE, NON-homogeneous Dirichlet data.

        nabla^2 phi = 0
        phi = sin(pi x) sin(pi y) sinh(k z) / sinh(k),   k = sqrt(2) pi

    Harmonic by construction: the two negative curvatures in x and y
    (-pi^2 each) are exactly cancelled by the positive curvature of sinh in
    z (+k^2 = +2 pi^2).  Boundary data is zero on five faces and
    sin(pi x) sin(pi y) on z = Lz.

    This case exists because of a genuine gap in coverage.  Every other
    section with an exact solution carries *zero* boundary data, and the one
    section with real boundary data (section 4, the discharge) has no closed
    form.  So without this case the boundary-absorption path in
    PoissonLine3D._build_rhs is never checked against a known answer
    anywhere in 3-D - a bug there would show up only as a plausible-looking
    wrong field in the production case.

    It is also the closest generic analogue of the physics: a real HET
    discharge is dominated by the 300 V applied across the channel, not by
    the space-charge source, so a BC-driven solution is the regime that
    actually matters.
    """
    h = 1.0 / (N + 1)
    p = np.arange(1, N + 1) * h
    X, Y, Z = np.meshgrid(p, p, p, indexing="ij")
    k = np.sqrt(2.0) * np.pi
    phi = np.sin(np.pi * X) * np.sin(np.pi * Y) * np.sinh(k * Z) / np.sinh(k)
    f = np.zeros_like(phi)
    face_xy = np.sin(np.pi * p)[:, None] * np.sin(np.pi * p)[None, :]
    prob = PoissonLine3D(f, lengths=(1.0, 1.0, 1.0),
                         bc_hi=(0.0, 0.0, face_xy))
    return prob, phi, f, "3D_Laplace_BCdriven_cube", 0


# ── Section 6: localised Gaussian sources ─────────────────────────────────────

_GAUSS_SIGMA = 0.12
_GAUSS_CENTRES = ((0.3, 0.3, 0.35), (0.7, 0.65, 0.6))
_GAUSS_AMPS = (1.0, -0.8)


def _gauss_phi(X, Y, Z):
    out = np.zeros_like(X)
    for A, (cx, cy, cz) in zip(_GAUSS_AMPS, _GAUSS_CENTRES):
        out += A * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
                          / (2.0 * _GAUSS_SIGMA**2))
    return out


def _gauss_src(X, Y, Z):
    """nabla^2 of a Gaussian: exp(-r^2/2s^2) (r^2/s^4 - 3/s^2) in 3-D."""
    out = np.zeros_like(X)
    for A, (cx, cy, cz) in zip(_GAUSS_AMPS, _GAUSS_CENTRES):
        r2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
        out += (A * np.exp(-r2 / (2.0 * _GAUSS_SIGMA**2))
                * (r2 / _GAUSS_SIGMA**4 - 3.0 / _GAUSS_SIGMA**2))
    return out

def case_gaussian(N: int):
    """
    Two localised Gaussian charge blobs of opposite sign, with exact
    non-homogeneous Dirichlet data on all six faces.

    The 3-D analogue of the two-Gaussian PlasmaNet benchmark used in the 2-D
    sweep, and the standard shape of a plasma space-charge source: compact,
    steep, and poorly resolved on coarse grids.  Unlike the 2-D version,
    which needed a 200x200 Fourier reference, this one is manufactured -
    phi is the Gaussian sum and f is its analytic Laplacian - so it has an
    exact solution at no cost.  A 3-D Fourier reference would need
    O(N_modes^3) terms on a fine grid and is not worth it.

    Two things are tested that no other section covers together: a source
    with real spatial structure (sigma = 0.12 against h = 1/33 at N=32, so
    only ~4 cells per standard deviation), and non-homogeneous data on
    *every* face rather than one.  Its truncation error is correspondingly
    larger than the smooth sinusoidal cases - that is the point, not a
    defect.  Boundary values are taken at the true boundary planes
    (coordinate 0 and L), not at the first interior node; using the latter
    is an easy mistake that silently destroys second-order convergence.
    """
    h = 1.0 / (N + 1)
    p = np.arange(1, N + 1) * h
    X, Y, Z = np.meshgrid(p, p, p, indexing="ij")
    phi = _gauss_phi(X, Y, Z)
    f = _gauss_src(X, Y, Z)

    A, B = np.meshgrid(p, p, indexing="ij")
    zeros, ones = np.zeros_like(A), np.ones_like(A)
    bc_lo = (_gauss_phi(zeros, A, B), _gauss_phi(A, zeros, B),
             _gauss_phi(A, B, zeros))
    bc_hi = (_gauss_phi(ones, A, B), _gauss_phi(A, ones, B),
             _gauss_phi(A, B, ones))
    prob = PoissonLine3D(f, lengths=(1.0, 1.0, 1.0),
                         bc_lo=bc_lo, bc_hi=bc_hi)
    return prob, phi, f, "3D_Poisson_TwoGaussian_cube", 0


# ── Section 7: high-wavenumber Fourier mode ───────────────────────────────────

MODE_NML: tuple[int, int, int] = (2, 3, 4)


def case_highmode(N: int):
    """
    A single high-wavenumber Fourier eigenmode:

        phi = sin(n pi x) sin(m pi y) sin(l pi z),   (n, m, l) = (2, 3, 4)
        f   = -pi^2 (n^2 + m^2 + l^2) phi = -29 pi^2 phi

    Section 1 is the (1,1,1) mode - the smoothest solution the grid can
    carry, and the one every iterative scheme handles best.  This is the
    opposite end: at N=8 the l=4 mode has only two cells per half-wavelength,
    so it sits near the grid's resolution limit.

    Two distinct things are probed.  Discretisation: the h^2 error constant
    scales with the fourth derivative, so this case shows the true accuracy
    cost of an under-resolved solution.  And multigrid: high-frequency error
    components are precisely what the smoother must remove and what the
    coarse grid cannot represent, so a defective smoother or transfer
    operator degrades here first, while looking fine on section 1.
    """
    n, m, l = MODE_NML
    h = 1.0 / (N + 1)
    p = np.arange(1, N + 1) * h
    X, Y, Z = np.meshgrid(p, p, p, indexing="ij")
    phi = (np.sin(n * np.pi * X) * np.sin(m * np.pi * Y)
           * np.sin(l * np.pi * Z))
    f = -np.pi**2 * (n * n + m * m + l * l) * phi
    prob = PoissonLine3D(f, lengths=(1.0, 1.0, 1.0))
    return prob, phi, f, "3D_Poisson_HighMode_n2m3l4", 0


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
        for solver, opts in (self.inner_options or {}).items():
            cfg.setdefault(solver, {}).update(opts)
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
    scheme = cfg.scheme_for(N)
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


def _run_case(section: str, N: int, cfg: SweepConfig, results: list) -> None:
    prob, phi_ref, f_vals, case_id, mode_m = SECTIONS[section](N)
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
            res_q = solve(prob, inner=solver_name, scheme=scheme,
                          inner_options=inner_cfg, **kw)
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
              "section7": f"High-wavenumber mode {MODE_NML}, unit cube"}
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


def _save_results(results) -> None:
    if not results:
        return
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
        scheme_crossover=args.scheme_crossover)

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
                max_workers=args.max_workers, max_tasks_per_child=1) as ex:
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