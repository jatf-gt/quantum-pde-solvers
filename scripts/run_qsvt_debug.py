"""
run_qsvt_debug.py
-----------------
Standalone diagnostic runner for the QSVT 1-D solver.

Runs QSVT on two problems and benchmarks against Thomas:
    1. Generic 1-D Poisson with fS source (homogeneous BCs) at N=4 and N=8.
    2. HET plasma 1-D linear profile (homogeneous BCs) at N=4.

Diagnostics active:
    - Proportionality recovery diagnostics (see qsvt_1d.py::_qsvt_recovery_diagnostics).
    - Solution comparison against Thomas and analytical.

Usage
-----
    python scripts/run_qsvt_debug.py

Output
------
    Console: full diagnostic output.
    results/qsvt_debug/: figures and CSV summary.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import numpy as np

from core.config import SimConfig1D
from core.exact_solutions import EXACT_SOLUTIONS, HET_EXACT_SOLUTIONS
from core.het_config import HETConfig
from problems.het_plasma_1d import HETPoissonProblem1D
from problems.poisson_1d import PoissonProblem1D
from solvers.classical.thomas import thomas_solve, thomas_solve_system
from solvers.quantum.qsvt_1d import QSVTConfig1D, qsvt_solve, qsvt_solve_system

RESULTS_DIR = Path("results/qsvt_debug")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Utility ───────────────────────────────────────────────────────────────────

def _rel_err_pct(u: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Pointwise relative error in percent, NaN where |ref| < 1% of max."""
    scale = np.max(np.abs(ref))
    mask  = np.abs(ref) > 0.01 * scale
    return np.where(mask, np.abs(u - ref) / np.abs(ref) * 100.0, np.nan)


def _max_rel_err(u: np.ndarray, ref: np.ndarray) -> float:
    err   = _rel_err_pct(u, ref)
    valid = err[~np.isnan(err)]
    return float(np.max(valid)) if valid.size > 0 else float("nan")


def _residual(A: np.ndarray, u: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(A @ u - b) / np.linalg.norm(b))


def _section(title: str) -> None:
    print(f"\n{'═'*68}")
    print(f"  {title}")
    print(f"{'═'*68}")


def _row(label: str, rel: float, res: float, t: float, extra: str = "") -> None:
    print(
        f"  {label:<12} MaxRelErr={rel:>8.4f}%  "
        f"Residual={res:>12.4e}  Time={t:>7.3f}s  {extra}"
    )


# ── Section 1: Generic 1-D Poisson ────────────────────────────────────────────

def run_generic_poisson() -> None:
    """
    Run QSVT on the generic 1-D Poisson equation with fS source and
    homogeneous BCs at N=4, N=8, and N=16. Benchmark against Thomas.
    Analytical solution u = -sin(πx)/π² is available for exact error.
    """
    _section("QSVT Debug — Generic 1-D Poisson, fS source, homogeneous BCs")

    for N in (4, 8, 16):
        print(f"\n  {'─'*60}")
        print(f"  N={N}")
        print(f"  {'─'*60}")

        cfg     = SimConfig1D(N=N, epsilon=0.01, source_fn="fS")
        problem = PoissonProblem1D(cfg)
        u_exact = EXACT_SOLUTIONS["fS"](problem.x)

        # Thomas reference.
        t0       = time.perf_counter()
        r_thomas = thomas_solve(problem)
        t_thomas = time.perf_counter() - t0
        _row(
            "Thomas",
            _max_rel_err(r_thomas.u, u_exact),
            _residual(problem.A, r_thomas.u, problem.b),
            t_thomas,
        )

        qsvt_cfg = QSVTConfig1D(
            epsilon      = 0.01,
            angle_method = "auto",
            verbose      = False,
            max_degree   = None if N <= 16 else 5000,
            label        = f"generic-fS-N{N}",
        )

        t0     = time.perf_counter()
        r_qsvt = qsvt_solve(problem, config=qsvt_cfg)
        t_qsvt = time.perf_counter() - t0

        _row(
            "QSVT",
            _max_rel_err(r_qsvt.u, u_exact),
            _residual(problem.A, r_qsvt.u, problem.b),
            t_qsvt,
            f"deg={r_qsvt.polynomial_degree}",
        )

        print(f"\n  Pointwise relative error vs analytical:")
        print(f"    x      = {np.round(problem.x, 3).tolist()}")
        err = _rel_err_pct(r_qsvt.u, u_exact)
        print(f"    QSVT % = {np.round(err, 3).tolist()}")
        err_t = _rel_err_pct(r_thomas.u, u_exact)
        print(f"    Thom % = {np.round(err_t, 3).tolist()}")

        print(f"\n  Solution vectors:")
        print(f"    Thomas = {np.round(r_thomas.u, 6).tolist()}")
        print(f"    QSVT   = {np.round(r_qsvt.u,   6).tolist()}")
        print(f"    Exact  = {np.round(u_exact,     6).tolist()}")


# ── Section 2: HET 1-D linear profile ─────────────────────────────────────────

def run_het_1d() -> None:
    """
    Run QSVT on the HET 1-D linear profile with homogeneous BCs at N=4.
    Analytical solution is available. Benchmark against Thomas.

    This is the failing case: QSVT gives ~100% error despite the block
    encoding being correct (Max error = 0). The polynomial diagnostic
    will show whether Im(P(λ_k/α)) is proportional to 1/λ_k for all k.
    """
    _section("QSVT Debug — HET 1-D Linear Profile, homogeneous BCs")

    for N in (4, 8, 16):
        print(f"\n  {'─'*60}")
        print(f"  N={N}")
        print(f"  {'─'*60}")

        cfg     = HETConfig(N=N, epsilon=0.01, rho_profile="linear", V_discharge=0.0)
        problem = HETPoissonProblem1D(cfg)
        u_exact = HET_EXACT_SOLUTIONS["linear"](problem.x, cfg.rho_0, cfg.alpha)

        print(f"\n  Problem parameters:")
        print(f"    N={cfg.N}, alpha={cfg.alpha:.2f}, rho_0={cfg.rho_0:.4f}")

        # Thomas reference.
        t0         = time.perf_counter()
        u_thomas   = thomas_solve_system(problem.A, problem.b)
        t_thomas   = time.perf_counter() - t0
        _row(
            "Thomas",
            _max_rel_err(u_thomas, u_exact),
            _residual(problem.A, u_thomas, problem.b),
            t_thomas,
        )

        # QSVT with full diagnostics.
        qsvt_cfg = QSVTConfig1D(
            epsilon      = 0.01,
            angle_method = "auto",
            verbose      = False,
            max_degree   = None if N <= 16 else 5000,
            label        = "",   # if == "HET-3a" triggers polynomial and recovery diagnostics
        )

        t0     = time.perf_counter()
        r_qsvt = qsvt_solve_system(problem.A, problem.b, config=qsvt_cfg)
        t_qsvt = time.perf_counter() - t0

        _row(
            "QSVT",
            _max_rel_err(r_qsvt.u, u_exact),
            _residual(problem.A, r_qsvt.u, problem.b),
            t_qsvt,
            f"deg={r_qsvt.polynomial_degree}",
        )

        print(f"\n  Thomas solution:   {np.round(u_thomas, 2).tolist()}")
        print(f"  Analytical:        {np.round(u_exact,    2).tolist()}")
        print(f"  QSVT solution:     {np.round(r_qsvt.u,   2).tolist()}")

        print(f"\n  Pointwise relative error vs analytical:")
        print(f"    x      = {np.round(problem.x, 3).tolist()}")
        err_q = _rel_err_pct(r_qsvt.u, u_exact)
        print(f"    QSVT % = {np.round(err_q, 3).tolist()}")
        err_t = _rel_err_pct(u_thomas,  u_exact)
        print(f"    Thom % = {np.round(err_t, 3).tolist()}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t_start = time.perf_counter()

    print("\n" + "═"*68)
    print("  QSVT STANDALONE DIAGNOSTIC RUNNER")
    print("  Imperial College London, Department of Aeronautics")
    print("═"*68)

    run_generic_poisson()
    run_het_1d()

    print(f"\n{'─'*68}")
    print(f"  Total elapsed: {time.perf_counter() - t_start:.1f}s")
    print("═"*68)


if __name__ == "__main__":
    main()