"""
Stationary line-relaxation schemes: line Jacobi, line Gauss-Seidel, line SOR.

Three roles:

1.  ``update="jacobi"`` with ``criterion="delta"`` reproduces the original
    line-Jacobi loop exactly, so the previously validated small-N results
    remain reachable and reproducible.  This is the fallback path.
2.  ``update="gauss-seidel"`` with omega > 1 is line-SOR, the current
    production scheme.
3.  omega = 1 Gauss-Seidel is the smoother used inside multigrid; it is a
    poor standalone scheme but the correct smoother.

On the choice of omega
----------------------
The classical optimal relaxation parameter

    omega* = 2 / (1 + sqrt(1 - rho_J^2)),   rho_J = cos(pi/(N+1))

minimises the spectral radius of the *exact* iteration.  It has two
properties that matter here and are easy to miss:

1.  It is still O(N).  Optimal SOR does not change the asymptotic order,
    only the constant.  Measured sweeps to discretisation accuracy on the
    unit square: 33, 66, 130, 258, 514 for N = 16 ... 256 - a clean
    doubling.  No tuning removes this.

2.  It is the *least robust* choice with respect to inner-solver error.
    The error of a converged stationary iterate is amplified by roughly
    1/(1 - rho) relative to the per-strip error, and optimal SOR is
    deliberately run with rho as close to 1 as possible.  Measured with a
    systematic 0.2 % strip error: 3.8 % solution error at N=32 and 20.7 %
    at N=64; at 2 % strip error the iteration diverges outright at N=64.

On the convergence criterion
----------------------------
``criterion="delta"`` (the original) stops on max|u_new - u_old| < tol.
It systematically *overstates* convergence: for a stationary scheme the true
error exceeds the iterate difference by a factor 1/(1 - rho) = O(N), so a run
reporting delta = 1e-6 at N=64 may still carry ~1e-4 relative error.  It is
retained for backward comparability; ``criterion="residual"`` is the honest
default and is what multigrid uses.

Author : Juan Antonio Trobajo Flecha
"""
from __future__ import annotations

import time

import numpy as np

from solvers.outer.core import (LineProblem2D, OuterResult, StagnationMonitor,
                                WorkLog, strip_sweep)


def optimal_omega(Nx: int, Ny: int) -> float:
    """Optimal SOR parameter for the 5-point Laplacian on an Nx x Ny grid."""
    rho_J = 0.5 * (np.cos(np.pi / (Nx + 1)) + np.cos(np.pi / (Ny + 1)))
    return float(np.clip(2.0 / (1.0 + np.sqrt(1.0 - rho_J**2)), 1.0, 1.99))


def solve_stationary(
    problem:    LineProblem2D,
    inner,
    omega:      float | str = "optimal",
    update:     str = "gauss-seidel",     # "gauss-seidel" | "jacobi"
    criterion:  str = "residual",         # "residual" | "delta"
    tol:        float = 1e-8,
    max_iter:   int = 5000,
    symmetric:  bool = False,
    patience:   int = 20,
    u0:         np.ndarray | None = None,
    callback=None,
) -> OuterResult:
    """
    Line relaxation until the chosen convergence measure falls below ``tol``.

    Parameters
    ----------
    omega : "optimal" for the analytic SOR parameter, or a float in (0, 2).
        Defaults to 1.0 when ``update="jacobi"``.
    update : "jacobi" updates every strip from the previous iterate (the
        original scheme); "gauss-seidel" uses already-updated strips.
    criterion : "residual" tests ||b - Au||/||b||; "delta" tests
        max|u_new - u_old| (the original test).
    u0 : optional initial guess, enabling scheme composition.
    patience : stagnation window.  A quantum inner solver has an error floor;
        once the iteration reaches it, further sweeps cost circuit simulations
        and buy nothing.  Stagnation is reported as such, not as convergence.
    callback : optional f(iteration, u, residual, delta).
    """
    if update not in ("gauss-seidel", "jacobi"):
        raise ValueError(f"update must be 'gauss-seidel' or 'jacobi', got {update!r}")
    if criterion not in ("residual", "delta"):
        raise ValueError(f"criterion must be 'residual' or 'delta', got {criterion!r}")

    Nx, Ny = problem.shape
    if omega == "optimal":
        om = 1.0 if update == "jacobi" else optimal_omega(Nx, Ny)
    else:
        om = float(omega)

    u = np.zeros((Nx, Ny)) if u0 is None else np.array(u0, dtype=float, copy=True)
    rhs = problem.rhs()
    b_norm = np.linalg.norm(rhs)
    b_norm = b_norm if b_norm > 1e-300 else 1.0

    work = WorkLog()
    history: list[float] = []      # always the true residual, for comparability
    measure: list[float] = []      # whatever `criterion` selects
    monitor = StagnationMonitor(window=patience)
    t0 = time.perf_counter()
    stop = "max_iter"
    delta = float("nan")

    for it in range(max_iter):
        u_old = u.copy()
        strip_sweep(problem, u, rhs, inner, work, omega=om,
                    reverse=symmetric and (it % 2 == 1),
                    jacobi=(update == "jacobi"))

        res = float(np.linalg.norm(rhs - problem.apply(u)) / b_norm)
        delta = float(np.max(np.abs(u - u_old)))
        history.append(res)
        measure.append(res if criterion == "residual" else delta)

        if callback is not None:
            callback(it + 1, u, res, delta)

        if not np.isfinite(res) or res > 1e6 * history[0]:
            stop = "diverged"
            break
        if measure[-1] < tol:
            stop = "tol_met"
            break
        if monitor.update(res):
            stop = "stagnated"
            break

    label = ("line-jacobi" if update == "jacobi"
             else ("line-sor" if om > 1.0 else "line-gs"))

    return OuterResult(
        u=u,
        scheme=label,
        inner=getattr(inner, "name", "?"),
        converged=(stop == "tol_met"),
        n_outer=len(history),
        residual=history[-1] if history else float("nan"),
        residual_history=history,
        work=work,
        wall_time_s=time.perf_counter() - t0,
        stop_reason=stop,
        diagnostics={
            "omega": om,
            "update": update,
            "criterion": criterion,
            "kappa_row": problem.kappa_row(),
            "final_delta": delta,
            "residual_floor": monitor.best,
        },
    )