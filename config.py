"""
config.py
---------
Problem definitions for the 1D Poisson benchmark.

Everything that describes *what* you want to solve lives here: the source
functions, the closed-form solutions for homogeneous BCs, and the SimConfig
dataclass that bundles all run parameters into one object.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# ── Source functions (paper Section IV) ──────────────────────────────────────

def f_sin(x: np.ndarray) -> np.ndarray:
    """fS(x) = sin(πx)"""
    return np.sin(np.pi * x)


def f_linear(x: np.ndarray) -> np.ndarray:
    """fL(x) = 10x"""
    return 10.0 * x


def f_heaviside(x: np.ndarray) -> np.ndarray:
    """fH(x) = 2 H(x − 0.5) − 1  (unit step centred at 0.5)"""
    return np.where(x >= 0.5, 1.0, -1.0)


SOURCE_FUNCTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "fS": f_sin,
    "fL": f_linear,
    "fH": f_heaviside,
}


# ── Analytical solutions for homogeneous BCs: u(0) = u(1) = 0 ────────────────

def u_fS(x: np.ndarray) -> np.ndarray:
    """u'' = sin(πx)  →  u = −sin(πx) / π²"""
    return -np.sin(np.pi * x) / np.pi**2


def u_fL(x: np.ndarray) -> np.ndarray:
    """
    u'' = 10x  →  u = 5x(x² − 1) / 3.

    Derived by integrating twice and enforcing u(0) = u(1) = 0.
    """
    return 5.0 * x * (x**2 - 1.0) / 3.0


def u_fH(x: np.ndarray) -> np.ndarray:
    """
    Exact piecewise solution for fH with homogeneous BCs:
      x < 0.5 :  u = −x²/2 + x/4
      x ≥ 0.5 :  u =  x²/2 − 3x/4 + 1/4

    Obtained by integrating each piece of fH, then matching continuity of u
    and u' at x = 0.5, together with u(0) = u(1) = 0.
    """
    return np.where(
        x < 0.5,
        -x**2 / 2.0 + x / 4.0,
        x**2 / 2.0 - 3.0 * x / 4.0 + 1.0 / 4.0,
    )


EXACT_SOLUTIONS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "fS": u_fS,
    "fL": u_fL,
    "fH": u_fH,
}


# ── Run configuration ─────────────────────────────────────────────────────────

@dataclass
class SimConfig:
    """
    All parameters for a single benchmark run, following the paper's notation.

    N must be a power of 2 — amplitude encoding stores the b-vector as the
    quantum amplitudes of exactly log₂(N) qubits, so any other size would
    require zero-padding and is not supported here.

    epsilon controls the Trotter approximation inside the Hamiltonian
    simulation: smaller ε → more Trotter steps → lower discretisation error
    in the quantum circuit, but also a deeper circuit that is slower to
    simulate classically.  This corresponds directly to the ε parameter swept
    in the paper (Sections IV A–D).
    """

    N: int            # size of the N × N system matrix (= number of interior nodes)
    epsilon: float    # Trotter / QPE precision parameter
    source_fn: str    # key into SOURCE_FUNCTIONS: 'fS', 'fL', or 'fH'
    alpha: float = 0.0    # Dirichlet BC at x = 0: u(0) = alpha
    beta: float  = 0.0    # Dirichlet BC at x = 1: u(1) = beta

    def __post_init__(self) -> None:
        if self.N <= 0 or (self.N & (self.N - 1)) != 0:
            raise ValueError(f"N must be a positive power of 2, got N = {self.N}.")
        if self.source_fn not in SOURCE_FUNCTIONS:
            raise ValueError(
                f"Unknown source function '{self.source_fn}'. "
                f"Valid options: {list(SOURCE_FUNCTIONS)}"
            )
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}.")
