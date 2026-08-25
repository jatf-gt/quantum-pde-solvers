# 3-D VQLS sensitivity study — PARTIAL, DO NOT QUOTE YET

Job **3894706.pbs-7**, `DIM=3 SOLVERS=vqls MAX_WALL_S=9000 RUN_TAG=vqls_sens`,
started 2026-08-23 19:17:42 at commit a50ebfb, submitted with a 72 h walltime.
Snapshot copied off the cluster 2026-08-24, at which point 33 of the sweep's
units had completed.

**This is one case of two.** `3D_Poisson_TripleSin_cube` has both sweeps —
five `n_layers` points and four `n_restarts` points. `3D_HET_MMS_SPT100` has no
records at either.

The snapshot arrived carrying `.job_start_marker`, the sentinel
`submit_studies.sh` writes before the run and deletes in its epilogue. Its
presence means the epilogue had not executed when the copy was taken; combined
with the missing second case, the job was still in flight. The marker has been
removed here; this paragraph is its record.

**Nothing is merged into `results/3Dstudies` from this directory.** That
directory has no `sensitivity_vqls.json` at all, so installing this one would
publish a single-case sweep under a name that reads as a complete one. Re-copy
when the job finishes and install it then; there is nothing to merge against, so
the installation is a straight copy of `sensitivity_vqls.json` plus a row in the
receiving `PROVENANCE.md`.

The companion jobs from the same submission — 3894705 (3-D HHL, both studies)
and 3894707 (3-D VQLS equal-accuracy) — both completed and were installed into
`results/3Dstudies/` on 2026-08-24.

## Two things that will still be true when it completes

**The `n_restarts = 1` record is invalid and must be excluded.** It reports
residual 3.98 × 10⁻⁹, an error against Thomas of 9.21 × 10⁻⁸ %,
`vqls_cost_final = null`, and it is the *only* one of the nine records that
neither converged nor was truncated on merit. This is the defect diagnosed on
2026-08-24 and since fixed in `solvers/quantum/vqls_1d.py`: at `n_restarts = 1`
the early-exit branch pads no telemetry slots and the Phase 2 refinement is
skipped, leaving a one-entry cost history that an unconditional `[:-1]` slice
reduced to an empty sequence, on which `np.argmin` raises. Every strip solve
therefore raised and every one was absorbed by the classical fallback in
`solvers.outer.inner.InnerSolverWrapper`, so the recorded field is the Thomas
solution. The identical signature is documented in
`results/2Dstudies/PROVENANCE.md`, where it appears on both cases. The fix
landed after this job was submitted, so **the completed job will still carry the
bad row** — it must be excluded there too, or the point re-measured separately.

**Eight of the nine records are wall-clock truncated** at the `max_wall_s = 9000`
budget; only `n_layers = 3` (7 229 s) ran to completion. The record schema drops
the outer `stop_reason`, so `wall_time_s ≈ max_wall_s` is the only tell. Their
wall times are the budget rather than a cost measurement, and their residuals
are upper bounds from an outer solve stopped mid-iteration. This makes the 3-D
`n_restarts` sweep uninformative as recorded — all four of its points are
truncated — and leaves the `n_layers` sweep readable only in trend:
1.26 × 10⁰ → 3.83 × 10⁻³ → 1.47 × 10⁻⁵ → 1.31 × 10⁻⁴ → 8.63 × 10⁻⁶ across one to
five layers. A 3-D VQLS strip solve is simply more expensive than the 2 .5 h
budget allows at low layer counts, where the optimiser exhausts its iteration
budget on every strip; the 2-D equivalent needed 6 943 s at one layer against a
7 200 s budget, which is the same effect one notch below the cut.
