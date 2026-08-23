# 3-D parameter studies — second-order operator

| file | provenance |
|---|---|
| `sensitivity_qsvt.json` | Job 3834215.pbs-7, `DIM=3 SOLVERS=qsvt RUN_TAG=grid_fix`, started 2026-08-20 11:13:10 at commit a6b6e03. Installed 2026-08-23. |
| `equal_accuracy.json` | Same job. |
| `run_metadata.json` | Same job, verbatim. |

Two sweeps and two equal-accuracy entries, one per case
(`3D_Poisson_TripleSin_cube`, `3D_HET_MMS_SPT100`), at N = 8. **No 3-D HHL or
VQLS parameter study exists at any order**; the equal-accuracy comparison here is
therefore QSVT against the Thomas reference alone.

## QSVT records re-measured on the corrected degree grid, 2026-08-23

The superseded records swept `max_degree ∈ {50, 100, 200, 500, None}`, entirely
past the saturation knee. The replacement sweeps
`{5, 8, 11, 15, 21, 51, 201, 501}` and resolves it, in agreement with the
fourth-order result in `results/3Dstudies_4th/`:

The quantity below is `max_rel_err_vs_thomas` [%], which is what the figures
plot and label `err_alg_pct`. The JSON field named `err_alg` is a different
measure — a relative L² error against Thomas, also in per cent. The two differ
by orders of magnitude wherever the reference field has an interior node, and
must not be conflated.

| case | κ | error at the lowest degree | error at the knee | floor by |
|---|---|---|---|---|
| `3D_Poisson_TripleSin_cube` | 1.9122 | 3.46 × 10⁻⁶ % at d/κ = 2.61 | 1.17 × 10⁻⁸ % at 5.75 | d/κ ≈ 27 |
| `3D_HET_MMS_SPT100` | 1.4606 | 1.59 × 10² % at d/κ = 3.42 | 2.31 × 10⁻² % at 10.27 | see below |

The HET case never reaches the 10⁻¹² % floor on this measure: it plateaus at
4 × 10⁻³ – 6 × 10⁻³ % from d/κ ≈ 14 upward. That plateau is an artefact of the
measure, not of the solver. `max_rel_err_vs_thomas` is a pointwise maximum of a
relative error taken against a field with an interior node, so it is unbounded
there; the L²-based `err_alg` falls to 7 × 10⁻¹³ % over the same range. Report the
knee position, which both measures agree on, rather than the plateau value.

The equal-accuracy grid was already correct, so those records were re-measured
rather than re-sited; they agree with the superseded ones to eleven significant
figures in every field except `err_alg`, repaired by the 2026-08-19 fix, and
`wall_time_s`.

The first `RUN_TAG=grid_fix` submission, of 2026-08-19, was destroyed before
retrieval by the shared-output-directory collision described in
`results/3Dstudies_4th/PROVENANCE.md`; job 3834215 is the successful re-run.
