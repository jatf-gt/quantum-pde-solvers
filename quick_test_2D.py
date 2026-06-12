# quick_test_2d.py
import numpy as np
from core.config import SimConfig2D
from problems.poisson_2d import PoissonProblem2D
from solvers.classical.thomas_2d import thomas_solve_2d
from solvers.quantum.hhl_2d import hhl_solve_2d

cfg     = SimConfig2D(N=8, epsilon=0.01, source_fn="fS")
problem = PoissonProblem2D(cfg)
print(problem.summary())

# ── Reference solutions ───────────────────────────────────────────────────────

# Mode 3 (default): refine_factor=17 matches paper's Δh=1/153 for N=8.
print("\nComputing refined reference (refine_factor=17, matches paper)...")
u_ref_paper = problem.classical_reference_solve(refine_factor=17)
print(f"  max |u_ref_paper|: {np.max(np.abs(u_ref_paper)):.6f}")

# Mode 2: explicit target spacing, easy to adjust.
print("Computing refined reference (target_h=1/100, finer than paper)...")
u_ref_fine = problem.classical_reference_solve(target_h=1.0/100)
print(f"  max |u_ref_fine|:  {np.max(np.abs(u_ref_fine)):.6f}")

# Coarse direct solve: exact solution of the N=8 discrete system.
u_ref_coarse = problem.coarse_direct_solve()
print(f"  max |u_ref_coarse|:{np.max(np.abs(u_ref_coarse)):.6f}")

# ── Solvers ───────────────────────────────────────────────────────────────────
r_thomas = thomas_solve_2d(problem)
print(f"\nThomas-2D: {r_thomas.iterations} iters, converged={r_thomas.converged}")
print(f"  Max |Thomas - ref_paper|:  {np.max(np.abs(r_thomas.u - u_ref_paper)):.3e}")
print(f"  Max |Thomas - ref_coarse|: {np.max(np.abs(r_thomas.u - u_ref_coarse)):.3e}")

r_hhl = hhl_solve_2d(problem)
print(f"\nHHL-2D: {r_hhl.iterations} iters, converged={r_hhl.converged}")
print(f"  Max |HHL - ref_paper|:     {np.max(np.abs(r_hhl.u - u_ref_paper)):.3e}")
print(f"  Max |HHL - ref_coarse|:    {np.max(np.abs(r_hhl.u - u_ref_coarse)):.3e}")
print(f"  Max |HHL - Thomas|:        {np.max(np.abs(r_hhl.u - r_thomas.u)):.3e}")