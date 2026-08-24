# 2-D VQLS parameter study — PARTIAL, DO NOT QUOTE YET

Job **3894682.pbs-7**, `DIM=2 SOLVERS=vqls MAX_WALL_S=7200 RUN_TAG=vqls`,
started 2026-08-23 18:58:42, `-l walltime=48:00:00`. Snapshot copied off the
cluster 2026-08-23 22:54.

**This is one case of two.** `2D_Poisson_sin_hom` is complete — five `n_layers`
points, four `n_restarts` points, and a five-point equal-accuracy sweep, 8.7 h
in total. `het_2d_mms_spt100` has no records at any of the three.

The snapshot arrived carrying `.job_start_marker`, the sentinel
`submit_studies.sh` writes before the run and deletes in its epilogue. Its
presence means the epilogue had not executed when the copy was taken — the job
was still running, or was killed without reaching it. The marker has been
removed here; this paragraph is its record. Estimated total for both cases was
12.6 h against a 48 h wall, so a walltime kill is unlikely and the job was
probably simply still in flight.

**Nothing is merged into `results/2Dstudies` from this directory.** The
superseded VQLS records there remain in place until the job completes. Merging a
partial sweep would replace two cases' worth of records with one and leave no
trace of the difference. Re-copy when the job finishes; the merge is then the
usual solver-scoped one — VQLS records supersede, HHL and QSVT are untouched.

## Two things to check before it is quoted

**The `n_restarts = 1` record is anomalous.** It reports residual
6.285 × 10⁻⁹ and an error against Thomas of **exactly zero**, in 617 s, where
`n_restarts` ∈ {2, 3, 5} all report 1.184 × 10⁻⁵ in about 1 200 s. Fewer
restarts producing a better answer is not credible, an error of exactly zero
means the returned field equals the Thomas solution bit for bit, and
6.285 × 10⁻⁹ is precisely the converged outer residual this case reports under
QSVT. It reads as an outer solve that did not use the quantum inner solver at
all. Establish what it did before quoting any `n_restarts` conclusion.

**The VQLS diagnostic columns are null throughout** — `vqls_n_layers`,
`vqls_n_restarts`, `vqls_n_evaluations`, `vqls_converged`. The 2026-08-19 repair
of those fields reached `benchmark/sensitivity.py::run_sensitivity`, the 1-D
path, which now populates them: `results/1Dstudies_4th` records 6 733 and 10 000
evaluations and a per-point convergence flag. The outer path,
`run_sensitivity_outer`, still records nothing. Some of that is inherent — a 2-D
solve runs N independent strip optimisations per outer iteration, so there is no
single evaluation count — but `n_layers` and `n_restarts` are settings rather
than outcomes and could be recorded. Until they are, the swept value is
recoverable only from `sensitivity_value`.

## What the complete case shows

Convergence in `n_layers` is sharp and monotone: residual
1.63 × 10⁰ → 1.51 × 10⁻³ → 1.77 × 10⁻⁵ → 1.18 × 10⁻⁵ → 9.40 × 10⁻⁶ across one
to five layers, with the cost falling from 6 943 s to 1 089 s between one and
three layers — the one-layer ansatz cannot represent the solution and the
optimiser exhausts its budget failing to.
