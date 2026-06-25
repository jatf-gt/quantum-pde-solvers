# quick_test_vqls.py
import numpy as np
from core.config import SimConfig1D
from problems.poisson_1d import PoissonProblem1D
from solvers.classical.thomas import thomas_solve
from solvers.quantum.hhl_1d import hhl_solve
from solvers.quantum.vqls_1d import vqls_solve, VQLSConfig
from benchmark.metrics import compute_errors

cfg     = SimConfig1D(N=8, epsilon=0.01, source_fn="fS")
problem = PoissonProblem1D(cfg)
print(problem.summary())

# Classical reference.
thomas_sr = thomas_solve(problem)

# HHL.
hhl_sr = hhl_solve(problem)

# VQLS — start with a small config to verify it runs.
vcfg    = VQLSConfig(n_layers=4, optimiser="COBYLA", max_iter=5000,
                     tol=1e-5, verbose=True, random_seed=42)
vqls_sr = vqls_solve(problem, config=vcfg)

# Compare.
print(f"\nThomas  residual: {thomas_sr.euclidean_residual:.2e}")
print(f"HHL     residual: {hhl_sr.euclidean_residual:.2e}")
print(f"VQLS    residual: {vqls_sr.euclidean_residual:.2e}")
print(f"VQLS    cost:     {vqls_sr.final_cost:.6f}")
print(f"VQLS    evals:    {vqls_sr.n_circuit_evals}")
print(f"VQLS    success:  {vqls_sr.optimiser_success}")

print(f"\nMax |VQLS - Thomas|: {np.max(np.abs(vqls_sr.u - thomas_sr.u)):.3e}")
print(f"Max |HHL  - Thomas|: {np.max(np.abs(hhl_sr.u  - thomas_sr.u)):.3e}")