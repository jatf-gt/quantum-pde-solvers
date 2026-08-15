"""
Does VQLS's COBYLA optimiser still converge when the cost function is
genuinely noisy (shot-based), rather than the exact classical shortcut?

This is the research question the Hadamard-test cost function
(solvers.quantum.vqls_hadamard) exists to make answerable at all -- the
classical shortcut in vqls_utils.build_cost_function has no shot count, so
this question is literally not askable through it.

Usage
-----
    python scripts/vqls_noisy_convergence_1d.py
    python scripts/vqls_noisy_convergence_1d.py --shots 500 --maxiter 60
"""
from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from scipy.optimize import minimize

from solvers.quantum.vqls_utils import (
    build_cost_function,
    n_params,
    pauli_decompose_matrix,
)
from solvers.quantum.vqls_hadamard import (
    build_hadamard_cost_function,
    circuit_count,
)


def _tst_matrix(N: int, main_diag: float, off_diag: float) -> np.ndarray:
    """Reproduces the TST matrix pauli_decompose_tst used to build internally
    before pauli_decompose_matrix generalised to take A directly -- see the
    same helper in tests/test_vqls_hadamard.py."""
    return (
        main_diag * np.eye(N)
        + off_diag * np.diag(np.ones(N - 1), k=1)
        + off_diag * np.diag(np.ones(N - 1), k=-1)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--maxiter", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    n_qubits = int(np.log2(args.N))
    pauli_terms = pauli_decompose_matrix(_tst_matrix(args.N, main_diag=-2.0, off_diag=1.0))
    n_terms = len(pauli_terms)

    rng = np.random.default_rng(args.seed)
    b = rng.normal(size=args.N)
    b_norm = b / np.linalg.norm(b)

    x0 = rng.uniform(0, 2 * np.pi, size=n_params(n_qubits, args.n_layers))

    print(f"N={args.N} ({n_qubits} qubits), n_layers={args.n_layers}, "
          f"{n_terms} Pauli terms, {n_params(n_qubits, args.n_layers)} parameters")
    print(f"Circuits per cost evaluation: {circuit_count(n_terms)} "
          f"(2L for numerator + 2L^2 for denominator)")
    print(f"Shots per circuit: {args.shots}  ->  "
          f"{args.shots * circuit_count(n_terms):,} total shots per cost call\n")

    # -- Exact-cost baseline (the classical shortcut) -----------------------
    exact_cost = build_cost_function(pauli_terms, b_norm, n_qubits, args.n_layers)

    t0 = time.time()
    result_exact = minimize(
        exact_cost, x0, method="COBYLA",
        options={"maxiter": args.maxiter, "rhobeg": 0.5},
    )
    t_exact = time.time() - t0
    print(f"Exact-cost COBYLA:  final cost={result_exact.fun:.6f}  "
          f"nfev={result_exact.nfev}  wall={t_exact:.1f}s")

    # -- Noisy (shot-based) cost, same starting point and iteration budget --
    noisy_cost = build_hadamard_cost_function(
        pauli_terms, b_norm, n_qubits, args.n_layers, shots=args.shots
    )

    t0 = time.time()
    result_noisy = minimize(
        noisy_cost, x0, method="COBYLA",
        options={"maxiter": args.maxiter, "rhobeg": 0.5},
    )
    t_noisy = time.time() - t0

    # Report the NOISY optimiser's final parameters under the EXACT cost,
    # since the noisy cost value alone is itself a noisy estimate and not a
    # fair final comparison point.
    true_cost_at_noisy_optimum = exact_cost(result_noisy.x)

    print(f"Noisy-cost COBYLA:  final noisy cost={result_noisy.fun:.6f}  "
          f"true cost at that point={true_cost_at_noisy_optimum:.6f}  "
          f"nfev={result_noisy.nfev}  wall={t_noisy:.1f}s")

    print(f"\nTotal circuits submitted (noisy run): "
          f"{result_noisy.nfev * circuit_count(n_terms):,}")
    print(f"Total shots submitted (noisy run): "
          f"{result_noisy.nfev * circuit_count(n_terms) * args.shots:,}")

    print(f"\nExact-cost optimum:                {result_exact.fun:.6f}")
    print(f"Noisy-cost optimum (true cost):     {true_cost_at_noisy_optimum:.6f}")
    print(f"Gap:                                 "
          f"{true_cost_at_noisy_optimum - result_exact.fun:+.6f}")


if __name__ == "__main__":
    main()