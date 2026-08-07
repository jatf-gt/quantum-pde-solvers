#!/usr/bin/env python3
#!/usr/bin/env python3
"""
scripts/debug_1d_4th.py
-----------------------
Debug and validation script for the fourth-order 1D Poisson solver.

Compares the fourth-order pentadiagonal discretisation against the
second-order TST discretisation at the same N, and benchmarks the
Thomas, VQLS, and QSVT solvers on the fourth-order system.

Usage
-----
    # Thomas only (fast, no quantum backend needed):
    python scripts/debug_1d_4th.py --N 4

    # Thomas + VQLS:
    python scripts/debug_1d_4th.py --N 4 --inner vqls

    # Thomas + QSVT (proof-of-concept, epsilon=0.01):
    python scripts/debug_1d_4th.py --N 4 --inner qsvt

    # Thomas + QSVT with degree cap (faster, less accurate):
    python scripts/debug_1d_4th.py --N 4 --inner qsvt --max-degree 200

    # All solvers:
    python scripts/debug_1d_4th.py --N 4 --inner all

    # Different source function:
    python scripts/debug_1d_4th.py --N 8 --source fL --inner all

    # Accuracy/kappa comparison table across multiple N:
    python scripts/debug_1d_4th.py --compare-orders --N-values 4 8 16

    # Verify matrix assembly (print A and b for N=4):
    python scripts/debug_1d_4th.py --N 4 --dump-matrix

    # Produce solution profile plots (saved to results/debugging/):
    python scripts/debug_1d_4th.py --N 4 --inner all --plot
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ── Ensure repo root is on sys.path ──────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Terminal colour codes ─────────────────────────────────────────────────────
G = "\033[92m"   # green
Y = "\033[93m"   # yellow
R = "\033[91m"   # red
C = "\033[96m"   # cyan
B = "\033[1m"    # bold
X = "\033[0m"    # reset


# ── Error metric helpers ──────────────────────────────────────────────────────

def _max_rel_err(u: np.ndarray, ref: np.ndarray) -> float:
    """Max relative error (%), masking near-zero reference values."""
    mask = np.abs(ref) > 1e-10
    if not np.any(mask):
        return float(np.max(np.abs(u - ref))) * 100.0
    return float(np.max(np.abs((u[mask] - ref[mask]) / ref[mask]))) * 100.0


def _max_abs_err(u: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(u - ref)))


def _residual(A: np.ndarray, u: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(A @ u - b) / (np.linalg.norm(b) + 1e-300))


def _colour(err_pct: float) -> str:
    if err_pct < 5.0:
        return G
    if err_pct < 20.0:
        return Y
    return R


# ── Solver wrappers ───────────────────────────────────────────────────────────

def run_thomas(A: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, float]:
    """Solve via NumPy direct solver (exact reference for this matrix)."""
    t0 = time.perf_counter()
    u = np.linalg.solve(A, b)
    return u, time.perf_counter() - t0


def run_vqls(
    A: np.ndarray,
    b: np.ndarray,
    n_layers: int = 3,
) -> tuple[np.ndarray, float, object]:
    """Solve via VQLS using the existing vqls_solve_system infrastructure."""
    from solvers.quantum.vqls_1d import vqls_solve_system, VQLSConfig1D
    cfg = VQLSConfig1D(n_layers=n_layers)
    t0 = time.perf_counter()
    result = vqls_solve_system(A, b, config=cfg)
    wall = time.perf_counter() - t0
    return np.array(result.u), wall, result


def run_qsvt(
    A: np.ndarray,
    b: np.ndarray,
    epsilon: float = 0.01,
    max_degree: Optional[int] = None,
    angle_method: str = "auto",
) -> tuple[np.ndarray, float, object]:
    """Solve via QSVT using the existing qsvt_solve_system infrastructure."""
    from solvers.quantum.qsvt_1d import qsvt_solve_system, QSVTConfig1D
    cfg = QSVTConfig1D(
        epsilon=epsilon,
        max_degree=max_degree,
        angle_method=angle_method,
    )
    t0 = time.perf_counter()
    result = qsvt_solve_system(A, b, config=cfg)
    wall = time.perf_counter() - t0
    return np.array(result.u), wall, result


def run_hhl(
    A: np.ndarray,
    b: np.ndarray,
    epsilon: float = 0.01,
    trotter_steps: int | None = None,
) -> tuple[np.ndarray, float, object]:
    """Solve via HHL using PentadiagonalToeplitz Hamiltonian simulation."""
    from solvers.quantum.hhl_1d_4th import hhl_solve_4th
    from problems.poisson_1d_4th import PoissonProblem1D4th

    # hhl_solve_4th takes a PoissonProblem1D4th, so reconstruct one.
    # We pass A and b directly by temporarily wrapping them.
    # Since hhl_solve_4th reads problem.A, problem.b, problem.N,
    # we build a minimal duck-typed wrapper.
    class _ProbWrapper:
        pass

    prob = _ProbWrapper()
    prob.A = A
    prob.b = b
    prob.N = len(b)

    from solvers.quantum.hhl_1d_4th import hhl_solve_4th as _solve
    t0 = time.perf_counter()
    result = _solve(prob, epsilon=epsilon, trotter_steps=trotter_steps)
    wall = time.perf_counter() - t0
    return np.array(result.u), wall, result


# ── Display helpers ───────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{B}{C}{'═' * 64}{X}")
    print(f"{B}{C}  {title}{X}")
    print(f"{B}{C}{'═' * 64}{X}")


def _print_table_header() -> None:
    print(
        f"\n  {'Solver':<12}  {'vs_exact':>12}  {'vs_thomas':>11}  "
        f"{'residual':>12}  {'time':>8}  {'extra'}"
    )
    print(f"  {'─' * 72}")


def _print_row(
    label: str,
    u: np.ndarray,
    u_exact: Optional[np.ndarray],
    u_thomas: np.ndarray,
    A: np.ndarray,
    b: np.ndarray,
    wall: float,
    extra: str = "",
) -> None:
    res = _residual(A, u, b)
    rel_thomas = _max_rel_err(u, u_thomas)

    if u_exact is not None:
        rel_exact = _max_rel_err(u, u_exact)
        col = _colour(rel_exact)
        exact_str = f"{col}{rel_exact:8.3f}%{X}"
    else:
        exact_str = f"{'N/A':>9}"

    print(
        f"  {label:<12}  "
        f"vs_exact={exact_str}  "
        f"vs_thomas={rel_thomas:7.3f}%  "
        f"residual={res:.2e}  "
        f"time={wall:.3f}s  "
        f"{extra}"
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_solutions(
    x: np.ndarray,
    u_exact: Optional[np.ndarray],
    solutions: dict[str, np.ndarray],
    N: int,
    source_fn: str,
    out_dir: Path,
) -> None:
    """
    Plot solution profiles for all solvers on a single figure.

    Top panel: solution profiles (exact + all solvers).
    Bottom panel: pointwise absolute error vs Thomas reference.

    Saved to out_dir/4th_order_N{N}_{source_fn}.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  {Y}matplotlib not available — skipping plot.{X}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Colour palette: consistent across solver labels
    palette = {
        "Thomas": "black",
        "VQLS":   "royalblue",
        "QSVT":   "crimson",
    }
    # 2nd-order Thomas reference for comparison (dashed grey)
    try:
        from problems.poisson_1d import PoissonProblem1D
        from core.config import SimConfig1D
        cfg_2nd = SimConfig1D(N=N, epsilon=0.01, source_fn=source_fn)
        prob_2nd = PoissonProblem1D(cfg_2nd)
        u_thomas_2nd = np.linalg.solve(prob_2nd.A, prob_2nd.b)
        have_2nd = True
    except Exception:
        have_2nd = False

    # ── Figure layout ─────────────────────────────────────────────────────────
    n_panels = 2 if len(solutions) > 1 else 1
    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(7, 3.5 * n_panels),
        sharex=True,
    )
    if n_panels == 1:
        axes = [axes]

    ax_sol = axes[0]
    ax_err = axes[1] if n_panels == 2 else None

    # ── Solution profiles ─────────────────────────────────────────────────────
    # Exact solution (if available)
    if u_exact is not None:
        ax_sol.plot(
            x, u_exact,
            "k--", lw=1.5, label="Exact", zorder=5,
        )

    # 2nd-order Thomas (dashed grey) for comparison
    if have_2nd:
        ax_sol.plot(
            x, u_thomas_2nd,
            color="grey", lw=1.0, ls=":", label="Thomas (2nd order)",
        )

    # All 4th-order solver results
    u_thomas_4th = solutions.get("Thomas")
    for label, u in solutions.items():
        col = palette.get(label, "purple")
        ls = "-" if label != "Thomas" else "-"
        lw = 2.0 if label == "Thomas" else 1.6
        ax_sol.plot(x, u, color=col, lw=lw, ls=ls, label=f"{label} (4th order)")

    ax_sol.set_ylabel("u(x)", fontsize=11)
    ax_sol.set_title(
        f"Fourth-Order 1D Poisson — N={N}, source={source_fn}",
        fontsize=12, fontweight="bold",
    )
    ax_sol.legend(fontsize=9, loc="best")
    ax_sol.grid(True, alpha=0.3)

    # ── Error panel ───────────────────────────────────────────────────────────
    if ax_err is not None and u_thomas_4th is not None:
        for label, u in solutions.items():
            if label == "Thomas":
                continue
            col = palette.get(label, "purple")
            err = np.abs(u - u_thomas_4th)
            ax_err.semilogy(x, err + 1e-16, color=col, lw=1.6, label=label)

        if u_exact is not None:
            # Thomas vs exact
            err_thomas = np.abs(u_thomas_4th - u_exact)
            ax_err.semilogy(
                x, err_thomas + 1e-16,
                "k--", lw=1.2, label="Thomas vs exact",
            )

        ax_err.set_xlabel("x", fontsize=11)
        ax_err.set_ylabel("|error|", fontsize=11)
        ax_err.set_title("Pointwise absolute error vs Thomas (4th order)", fontsize=11)
        ax_err.legend(fontsize=9, loc="best")
        ax_err.grid(True, alpha=0.3)

    else:
        if ax_err is not None:
            ax_err.set_xlabel("x", fontsize=11)

    plt.tight_layout()
    out_path = out_dir / f"4th_order_N{N}_{source_fn}.png"
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  {G}Plot saved to: {out_path}{X}")


# ── Core study functions ──────────────────────────────────────────────────────

def compare_orders(N_values: list[int], source_fn: str = "fS") -> None:
    """
    Print a table comparing second- and fourth-order discretisations.

    For each N, reports:
      - kappa(A) for both orders and their ratio
      - Max relative error vs the analytical solution (if available)
    """
    from problems.poisson_1d_4th import PoissonProblem1D4th
    from problems.poisson_1d import PoissonProblem1D
    from core.config import SimConfig1D

    _header("2nd-Order vs 4th-Order: Accuracy and Condition Number")
    print(f"\n  Source: {source_fn}  |  Homogeneous BCs (alpha=beta=0)\n")
    print(
        f"  {'N':>4}  {'kappa_2nd':>10}  {'kappa_4th':>10}  "
        f"{'ratio':>7}  {'err_2nd%':>10}  {'err_4th%':>10}"
    )
    print(f"  {'─' * 60}")

    for N in N_values:
        # Second-order problem
        cfg = SimConfig1D(N=N, epsilon=0.01, source_fn=source_fn)
        prob_2nd = PoissonProblem1D(cfg)
        u_2nd = np.linalg.solve(prob_2nd.A, prob_2nd.b)

        # Fourth-order problem
        prob_4th = PoissonProblem1D4th(N=N, source_fn=source_fn)
        u_4th = np.linalg.solve(prob_4th.A, prob_4th.b)

        # Analytical solution (same PDE, same BCs)
        u_exact = prob_4th.exact_solution()
        if u_exact is not None:
            err_2nd = _max_rel_err(u_2nd, u_exact)
            err_4th = _max_rel_err(u_4th, u_exact)
            c2 = _colour(err_2nd)
            c4 = _colour(err_4th)
            err_2nd_str = f"{c2}{err_2nd:9.3f}%{X}"
            err_4th_str = f"{c4}{err_4th:9.3f}%{X}"
        else:
            err_2nd_str = f"{'N/A':>10}"
            err_4th_str = f"{'N/A':>10}"

        ratio = prob_4th.kappa / prob_2nd.kappa
        print(
            f"  {N:>4}  {prob_2nd.kappa:>10.2f}  {prob_4th.kappa:>10.2f}  "
            f"{ratio:>7.3f}  {err_2nd_str}  {err_4th_str}"
        )

    print()


def run_single(
    N: int,
    source_fn: str,
    inner: str,
    n_layers: int,
    epsilon: float,
    max_degree: Optional[int],
    angle_method: str,
    hhl_epsilon: float,
    trotter_steps: int | None,
    dump_matrix: bool,
    do_plot: bool,
) -> None:
    """Run the specified solver(s) on the fourth-order system at N."""
    from problems.poisson_1d_4th import PoissonProblem1D4th

    prob = PoissonProblem1D4th(N=N, source_fn=source_fn)
    u_exact = prob.exact_solution()

    _header(f"Fourth-Order 1D Poisson  —  N={N}, source={source_fn}")
    print(f"\n  {prob.summary()}")
    print(f"  ||A||_2  = {np.linalg.norm(prob.A, ord=2):.4f}")
    print(f"  ||b||_2  = {np.linalg.norm(prob.b):.4e}")
    if u_exact is not None:
        print(f"  max|u_exact| = {np.max(np.abs(u_exact)):.6f}")

    if dump_matrix:
        print(f"\n  A (pentadiagonal, N={N}):")
        print(prob.A)
        print(f"\n  b:")
        print(prob.b)

    # Collect solutions for plotting
    solutions: dict[str, np.ndarray] = {}

    # ── Thomas reference ──────────────────────────────────────────────────────
    u_thomas, t_thomas = run_thomas(prob.A, prob.b)
    solutions["Thomas"] = u_thomas

    _print_table_header()
    _print_row("Thomas", u_thomas, u_exact, u_thomas, prob.A, prob.b, t_thomas)

    # ── VQLS ─────────────────────────────────────────────────────────────────
    if inner in ("vqls", "all"):
        if n_layers == -1:
            # Adapt to N
            num_qubits = int(np.log2(N))
            n_layers_actual = max(4, num_qubits * 2)
        else:
            n_layers_actual = n_layers

        try:
            u_vqls, t_vqls, vqls_result = run_vqls(prob.A, prob.b, n_layers_actual)
            solutions["VQLS"] = u_vqls
            final_cost = (
                vqls_result.cost_history[-1]
                if vqls_result.cost_history
                else float("nan")
            )
            extra = (
                f"cost={final_cost:.2e}  "
                f"n_iters={vqls_result.n_circuit_evals}"
            )
            _print_row(
                "VQLS", u_vqls, u_exact, u_thomas,
                prob.A, prob.b, t_vqls, extra,
            )
        except Exception as exc:
            print(f"  {'VQLS':<12}  {R}FAILED: {exc}{X}")

    # ── QSVT ─────────────────────────────────────────────────────────────────
    if inner in ("qsvt", "all"):
        # print(
        #     f"\n  {Y}QSVT: epsilon={epsilon}, max_degree={max_degree}, "
        #     f"method={angle_method}{X}"
        # )
        # print(
        #     f"  {Y}Note: degree scales as O(kappa/epsilon). "
        #     f"At N={N}, kappa={prob.kappa:.1f}, "
        #     f"expect degree ~{int(prob.kappa / epsilon * 3)}.{X}"
        # )
        try:
            u_qsvt, t_qsvt, qsvt_result = run_qsvt(
                prob.A, prob.b,
                epsilon=epsilon,
                max_degree=max_degree,
                angle_method=angle_method,
            )
            solutions["QSVT"] = u_qsvt
            extra = (
                f"deg={qsvt_result.polynomial_degree}  "
                f"depth={qsvt_result.circuit_depth}"
            )
            _print_row(
                "QSVT", u_qsvt, u_exact, u_thomas,
                prob.A, prob.b, t_qsvt, extra,
            )
        except Exception as exc:
            print(f"  {'QSVT':<12}  {R}FAILED: {exc}{X}")


    # ── HHL ──────────────────────────────────────────────────────────────────
    if inner in ("hhl", "all"):
        # print(
        #     f"\n  {Y}HHL: epsilon={epsilon}, "
        #     f"trotter_steps={trotter_steps if trotter_steps else 'auto'}{X}"
        # )
        try:
            u_hhl, t_hhl, hhl_result = run_hhl(
                prob.A, prob.b,
                epsilon=epsilon,
                trotter_steps=trotter_steps,
            )
            solutions["HHL"] = u_hhl
            extra = (
                f"prop_const={hhl_result.prop_const:.3e}  "
                f"residual={hhl_result.euclidean_residual:.2e}"
            )
            _print_row(
                "HHL", u_hhl, u_exact, u_thomas,
                prob.A, prob.b, t_hhl, extra,
            )
        except Exception as exc:
            print(f"  {'HHL':<12}  {R}FAILED: {exc}{X}")

    print()

    # ── Inline comparison with 2nd order ──────────────────────────────────────
    print(f"\n  {C}Comparison with 2nd-order at the same N:{X}")
    compare_orders([N], source_fn)

    # ── Plot ──────────────────────────────────────────────────────────────────
    if do_plot and solutions:
        out_dir = REPO_ROOT / "results" / "debugging"
        plot_solutions(
            x=prob.x,
            u_exact=u_exact,
            solutions=solutions,
            N=N,
            source_fn=source_fn,
            out_dir=out_dir,
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug and validation tool for the 4th-order 1D Poisson solver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Problem specification
    parser.add_argument(
        "--N", type=int, default=4,
        help="Number of interior nodes (power of 2, >= 4). Default: 4.",
    )
    parser.add_argument(
        "--source", type=str, default="fS", choices=["fS", "fL", "fH"],
        help="Source function key. Default: fS.",
    )

    # Solver selection
    parser.add_argument(
        "--inner", type=str, default="thomas",
        choices=["thomas", "vqls", "qsvt", "hhl", "all"], 
        help="Solver(s) to run. Default: thomas.",
    )

    # VQLS options
    parser.add_argument(
        "--n-layers", type=int, default=-1,
        help="VQLS ansatz depth. Default: -1 (auto-scale with N).",
    )

    # QSVT options
    parser.add_argument(
        "--epsilon", type=float, default=0.01,
        help=(
            "QSVT polynomial approximation tolerance. "
            "Degree scales as O(kappa/epsilon). Default: 0.01."
        ),
    )
    parser.add_argument(
        "--max-degree", type=int, default=500,
        help=(
            "Hard cap on QSVT polynomial degree. "
            "None = uncapped (use pyqsp's own selection). "
            "Recommended: 200-500 for fast proof-of-concept at N=4/8."
        ),
    )
    parser.add_argument(
        "--angle-method", type=str, default="auto",
        choices=["auto", "symqsp_wrapper", "symqsp_direct"],
        help="QSP phase angle computation method. Default: auto.",
    )

    # HHL options
    parser.add_argument(
        "--hhl-epsilon", type=float, default=0.01,
        help="HHL error tolerance (also controls Trotter steps). Default: 0.01.",
    )
    parser.add_argument(
        "--trotter-steps", type=int, default=None,
        help=(
            "Trotter steps for HHL Hamiltonian simulation. "
            "None = auto-computed from epsilon. Default: None."
        ),
    )

    # Study modes
    parser.add_argument(
        "--compare-orders", action="store_true",
        help="Print 2nd vs 4th order accuracy/kappa table and exit.",
    )
    parser.add_argument(
        "--N-values", type=int, nargs="+", default=[4, 8, 16],
        help="N values for --compare-orders. Default: 4 8 16.",
    )
    parser.add_argument(
        "--dump-matrix", action="store_true",
        help="Print A and b for inspection (most useful at N=4).",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help=(
            "Save solution profile plots to results/debugging/. "
            "Requires matplotlib."
        ),
    )

    args = parser.parse_args()

    if args.compare_orders:
        compare_orders(args.N_values, args.source)
    else:
        run_single(
            N=args.N,
            source_fn=args.source,
            inner=args.inner,
            n_layers=args.n_layers,
            epsilon=args.epsilon,
            max_degree=args.max_degree,
            angle_method=args.angle_method,
            hhl_epsilon=args.hhl_epsilon,
            trotter_steps=args.trotter_steps,
            dump_matrix=args.dump_matrix,
            do_plot=args.plot,
        )


if __name__ == "__main__":
    main()