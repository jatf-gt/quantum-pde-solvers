"""
Provides console reporting utilities for the 1D Poisson benchmark evaluations.

This module formats and outputs the computed statistical metrics to standard 
output, ensuring the presentation aligns with the tabular formats utilised in 
the primary reference literature. It operates strictly on pre-computed result 
data structures.
"""
from __future__ import annotations

from benchmark.metrics import BenchmarkResult


# ── Console Reporting ─────────────────────────────────────────────────────────

def print_result_table(results: list[BenchmarkResult]) -> None:
    """
    Outputs a formatted comparative table of benchmark metrics to standard output.

    The tabular layout is structurally aligned with the figures presented in 
    the primary reference. Each row encapsulates a unique solver and configuration 
    pairing, detailing the corresponding maximum and average error metrics.
    """
    header = (
        f"{'Solver':<8} {'N':>4} {'f':>3} {'α':>5} {'β':>5} "
        f"{'ε':>8} {'MaxRel%':>9} {'AvgRel%':>9} "
        f"{'MaxAbs':>10} {'AvgAbs':>10} {'Residual':>10}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        cfg = r.config
        max_rel = f"{r.max_rel_error:9.3f}" if r.max_rel_error is not None else f"{'N/A':>9}"
        avg_rel = f"{r.avg_rel_error:9.3f}" if r.avg_rel_error is not None else f"{'N/A':>9}"
        residual = f"{r.euclidean_residual:.3e}" if r.euclidean_residual is not None else "N/A"
        
        print(
            f"{r.solver:<8} {cfg.N:>4} {cfg.source_fn:>3} "
            f"{cfg.alpha:>5.2f} {cfg.beta:>5.2f} "
            f"{cfg.epsilon:>8.4f} "
            f"{max_rel} {avg_rel} "
            f"{r.max_abs_error:>10.6f} {r.avg_abs_error:>10.6f} "
            f"{residual:>10}"
        )

    print(sep)


def print_hhl_summary(hhl_result: BenchmarkResult) -> None:
    """
    Outputs a detailed, node-by-node analytical breakdown for a singular HHL execution.

    This routine is primarily utilised for high-granularity debugging and 
    for direct nodal cross-verification against the graphical data presented 
    in the reference literature.
    """
    cfg = hhl_result.config
    print(
        f"\nHHL node-by-node: N={cfg.N}, f={cfg.source_fn}, "
        f"ε={cfg.epsilon}, α={cfg.alpha}, β={cfg.beta}"
    )
    print(f"  Proportionality constant c = {hhl_result.prop_const:.6f}")
    print(f"  Euclidean residual ||Au-b||_2 / ||b||_2 = {hhl_result.euclidean_residual:.4e}")
    print(f"\n  {'x_i':>8}  {'u_HHL':>12}  {'u_exact':>12}  {'RelErr%':>10}")
    
    for i, xi in enumerate(hhl_result.x):
        u_h = hhl_result.u_solver[i]
        u_e = hhl_result.u_exact[i] if hhl_result.u_exact is not None else float("nan")
        re  = hhl_result.rel_error[i] if hhl_result.rel_error is not None else float("nan")
        print(f"  {xi:8.4f}  {u_h:12.6f}  {u_e:12.6f}  {re:10.3f}")