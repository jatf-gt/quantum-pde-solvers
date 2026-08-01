"""
run_qsvt_debug.py
-----------------
Standalone diagnostic runner for the QSVT 1-D solver.

Runs QSVT on two problems and benchmarks against Thomas:
    1. Generic 1-D Poisson with fS source (homogeneous BCs) at N=4 and N=8.
    2. HET plasma 1-D linear profile (homogeneous BCs) at N=4.

All diagnostics are active:
    - Block encoding verification: checks <0|U_A|0> against A/alpha.
    - State preparation verification: checks Isometry output.
    - Full post-selected complex amplitudes.
    - QSP polynomial evaluation at each eigenvalue.
    - Proportionality recovery diagnostics.
    - Solution comparison against Thomas and analytical.

2-D cases are implemented but commented out at the bottom.

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

_debug_be_circuit = None
_debug_angles     = None
_debug_n          = None
_debug_alpha      = None


# ── Utility ------------------------------------------------------------------

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


# ── Section 1: Generic 1-D Poisson ------------------------------------------

def run_generic_poisson() -> None:
    """
    Run QSVT on the generic 1-D Poisson equation with fS source and
    homogeneous BCs at N=4 and N=8. Benchmark against Thomas.
    Analytical solution u = -sin(πx)/π² is available for exact error.
    """
    _section("QSVT Debug — Generic 1-D Poisson, fS source, homogeneous BCs")

    for N in (4, 8):
        if N == 8:
            break  # skip N=8 for now, too slow for debugging
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
            verbose      = True,
            max_degree   = None if N <= 16 else 5000,          # match QSVT_MAX_DEGREE_BY_N
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
        print(f"    x      = {np.round(problem.x, 4).tolist()}")
        err = _rel_err_pct(r_qsvt.u, u_exact)
        print(f"    QSVT % = {np.round(err, 4).tolist()}")
        err_t = _rel_err_pct(r_thomas.u, u_exact)
        print(f"    Thom % = {np.round(err_t, 4).tolist()}")

        print(f"\n  Solution vectors:")
        print(f"    Thomas = {np.round(r_thomas.u, 6).tolist()}")
        print(f"    QSVT   = {np.round(r_qsvt.u,   6).tolist()}")
        print(f"    Exact  = {np.round(u_exact,     6).tolist()}")


# ── Section 2: HET 1-D linear profile ---------------------------------------

def run_het_1d() -> None:
    """
    Run QSVT on the HET 1-D linear profile with homogeneous BCs at N=4.
    Analytical solution is available. Benchmark against Thomas.

    This is the failing case: QSVT gives ~100% error despite the block
    encoding being correct (Max error = 0). The polynomial diagnostic
    will show whether Im(P(λ_k/α)) is proportional to 1/λ_k for all k.
    """
    _section("QSVT Debug — HET 1-D Linear Profile, homogeneous BCs, N=4")

    cfg     = HETConfig(N=4, epsilon=0.01, rho_profile="linear", V_discharge=0.0)
    problem = HETPoissonProblem1D(cfg)
    u_exact = HET_EXACT_SOLUTIONS["linear"](problem.x, cfg.rho_0, cfg.alpha)

    print(f"\n  Problem parameters:")
    print(f"    N={cfg.N}, alpha={cfg.alpha:.2f}, rho_0={cfg.rho_0:.4f}")
    print(f"    ||b||   = {np.linalg.norm(problem.b):.4e}")
    print(f"    ||b||/||b_fS|| ratio ≈ "
          f"{np.linalg.norm(problem.b) / 0.003:.1f}x larger than generic Poisson")
    print(f"    b_norm_vec = {np.round(problem.b/np.linalg.norm(problem.b), 4)}")

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

    print(f"\n  Thomas solution:   {np.round(u_thomas, 2).tolist()}")
    print(f"  Analytical:        {np.round(u_exact,   2).tolist()}")

    # QSVT with full diagnostics.
    qsvt_cfg = QSVTConfig1D(
        epsilon      = 0.01,
        angle_method = "auto",
        verbose      = True,
        max_degree   = 600,
        label        = "HET-3a",   # triggers polynomial and recovery diagnostics
    )

    t0     = time.perf_counter()
    r_qsvt = qsvt_solve_system(problem.A, problem.b, config=qsvt_cfg)
    t_qsvt = time.perf_counter() - t0

    import run_qsvt_debug as _dbg
    if _dbg._debug_be_circuit is not None:
        _verify_qsvt_polynomial_directly(
            _dbg._debug_be_circuit,
            _dbg._debug_angles,
            _dbg._debug_n,
            _dbg._debug_alpha,
            problem.A * -1,  # negated A (positive definite)
            label="HET-3a direct",
        )

    _row(
        "QSVT",
        _max_rel_err(r_qsvt.u, u_exact),
        _residual(problem.A, r_qsvt.u, problem.b),
        t_qsvt,
        f"deg={r_qsvt.polynomial_degree}",
    )

    print(f"\n  QSVT solution:     {np.round(r_qsvt.u, 2).tolist()}")
    print(f"  Thomas solution:   {np.round(u_thomas,   2).tolist()}")
    print(f"  Analytical:        {np.round(u_exact,    2).tolist()}")

    print(f"\n  Pointwise relative error vs analytical:")
    print(f"    x      = {np.round(problem.x, 4).tolist()}")
    err_q = _rel_err_pct(r_qsvt.u, u_exact)
    err_t = _rel_err_pct(u_thomas,  u_exact)
    print(f"    QSVT % = {np.round(err_q, 4).tolist()}")
    print(f"    Thom % = {np.round(err_t, 4).tolist()}")


def _verify_qsvt_polynomial_directly(
    be_circuit  : object,
    angles      : np.ndarray,
    n           : int,
    alpha       : float,
    A           : np.ndarray,
    label       : str,
) -> None:
    """
    Directly verify what polynomial the QSVT circuit implements by
    applying it to each eigenvector of A separately and measuring
    the imaginary part of the output.

    For each eigenvector v_k with eigenvalue lambda_k:
        Input:  |0_anc> |v_k>
        Output: Im(<0_anc|U_QSVT|0_anc>|v_k>) = Im(P(lambda_k/alpha)) * |v_k>

    This gives the exact polynomial values Im(P(lambda_k/alpha)) without
    any approximation, directly from the circuit.
    """
    from qiskit.quantum_info import Statevector
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import Isometry

    N = 2**n
    eig_vals, eig_vecs = np.linalg.eigh(A)

    print(f"\n  Direct polynomial verification [{label}]:")
    print(
        f"  {'lambda_k':>10}  {'x=lam/alpha':>12}  "
        f"{'Im(P) circuit':>14}  {'1/(kappa*x)*0.9':>16}  "
        f"{'ratio':>8}"
    )

    kappa = float(eig_vals.max() / eig_vals.min())
    be_gate     = be_circuit.to_gate(label="U_A")
    be_inv_gate = be_circuit.inverse().to_gate(label="U_A†")
    anc_idx     = n
    n_total     = n + 1
    be_qubits   = list(range(n_total))

    for k in range(N):
        lam_k = float(eig_vals[k])
        v_k   = eig_vecs[:, k]
        x     = lam_k / alpha

        # Build the QSVT circuit with v_k as input.
        qc = QuantumCircuit(n_total, name=f"QSVT_v{k}")

        # In _verify_qsvt_polynomial_directly, after building qc:
        from qiskit.quantum_info import Statevector as QSV
        test_qc = QuantumCircuit(n)
        test_qc.append(Isometry(v_k, 0, 0), list(range(n)))
        prepared_sv = np.real(np.array(QSV(test_qc).data))
        print(f"  Eigenvector {k} preparation check:")
        print(f"    v_k prepared: {np.round(prepared_sv, 4)}")
        print(f"    v_k expected: {np.round(v_k, 4)}")
        print(f"    match: {np.allclose(prepared_sv, v_k, atol=1e-6)}")

        qc.append(Isometry(v_k, 0, 0), list(range(n)))

        degree = len(angles) - 1
        for idx, phi in enumerate(angles):
            qc.rz(2.0 * phi, anc_idx)
            if idx < degree:
                if idx % 2 == 0:
                    qc.append(be_gate,     be_qubits)
                else:
                    qc.append(be_inv_gate, be_qubits)

        sv = np.array(Statevector(qc).data)

        # Extract post-selected imaginary part.
        x_raw = np.zeros(N, dtype=complex)
        for idx in range(2**n_total):
            if ((idx >> anc_idx) & 1) == 0:
                x_raw[idx & (N - 1)] = sv[idx]

        # Im(P(lambda_k/alpha)) = Im(x_raw) / v_k (should be scalar).
        im_part = np.imag(x_raw)
        # The output should be Im(P(x)) * v_k.
        # Estimate Im(P(x)) by projecting onto v_k.
        im_P = float(np.dot(v_k, im_part))

        expected = 0.9 / (kappa * x) if x > 1e-10 else float("nan")
        ratio    = im_P / expected if abs(expected) > 1e-14 else float("nan")

        print(
            f"  {lam_k:>10.4f}  {x:>12.6f}  "
            f"{im_P:>14.6f}  {expected:>16.6f}  "
            f"{ratio:>8.4f}"
        )


# ── Section 3: 2-D cases -------------------------------------------------

def run_generic_2d() -> None:
    """
    Run QSVT-2D on the generic 2-D Poisson equation with fS source at N=4.
    Uses the line-Jacobi decomposition. Benchmark against Thomas-2D.
    """
    from problems.poisson_2d import PoissonProblem2D
    from core.config import SimConfig2D
    from solvers.classical.thomas_2d import thomas_solve_2d
    from solvers.quantum.qsvt_2d import QSVTConfig2D, qsvt_solve_2d

    _section("QSVT Debug — Generic 2-D Poisson, fS source, N=4")

    cfg_2d   = SimConfig2D(N=4, epsilon=0.01, source_fn="fS", max_iter=100)
    prob_2d  = PoissonProblem2D(cfg_2d)

    r_thomas = thomas_solve_2d(prob_2d)

    qsvt_cfg_2d = QSVTConfig2D(
        epsilon      = 0.1,
        angle_method = "auto",
        max_degree   = 200,
        verbose      = True,
    )
    r_qsvt = qsvt_solve_2d(prob_2d, config=qsvt_cfg_2d)

    print(f"  Thomas-2D: iters={r_thomas.iterations}, "
          f"converged={r_thomas.converged}")
    print(f"  QSVT-2D:   iters={r_qsvt.iterations}, "
          f"converged={r_qsvt.converged}")
    print(f"  Max |QSVT - Thomas|: "
          f"{np.max(np.abs(r_qsvt.u - r_thomas.u)):.4e}")


def run_het_2d() -> None:
    """
    Run QSVT-2D on the HET sinusoidal 2-D problem at N=4.
    Analytical solution phi = sin(πx)sin(πy) is available.
    """
    from problems.het_plasma_2d import HETConfig2D, HETSinusoidalProblem2D
    from solvers.classical.thomas_2d import thomas_solve_2d
    from solvers.quantum.qsvt_2d import QSVTConfig2D, qsvt_solve_2d

    _section("QSVT Debug — HET 2-D Sinusoidal, N=4")

    cfg_2d   = HETConfig2D(N=4, epsilon=0.01, max_iter=300)
    prob_2d  = HETSinusoidalProblem2D(cfg_2d)
    u_exact  = prob_2d.analytical_solution()

    r_thomas = thomas_solve_2d(prob_2d)

    qsvt_cfg_2d = QSVTConfig2D(
        epsilon      = 0.01,
        angle_method = "auto",
        max_degree   = 200,
        verbose      = True,
    )
    r_qsvt = qsvt_solve_2d(prob_2d, config=qsvt_cfg_2d)

    print(f"  Thomas-2D: iters={r_thomas.iterations}, "
          f"max_err={_max_rel_err(r_thomas.u.ravel(), u_exact.ravel()):.4f}%")
    print(f"  QSVT-2D:   iters={r_qsvt.iterations}, "
          f"max_err={_max_rel_err(r_qsvt.u.ravel(), u_exact.ravel()):.4f}%")


# ── Main ---------------------------------------------------------------------

def main() -> None:
    t_start = time.perf_counter()

    print("\n" + "═"*68)
    print("  QSVT STANDALONE DIAGNOSTIC RUNNER")
    print("  Imperial College London, Department of Aeronautics")
    print("═"*68)

    run_generic_poisson()
    run_het_1d()

    # Uncomment to run 2-D cases once 1-D is resolved:
    # run_generic_2d()
    # run_het_2d()

    print(f"\n{'─'*68}")
    print(f"  Total elapsed: {time.perf_counter() - t_start:.1f}s")
    print("═"*68)


if __name__ == "__main__":
    main()