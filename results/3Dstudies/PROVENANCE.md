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

| case | κ | err_alg at d/κ ≈ 2.6–3.4 | floor reached at |
|---|---|---|---|
| `3D_Poisson_TripleSin_cube` | 1.9122 | 1.40 × 10⁻⁷ % | d/κ ≈ 27 |
| `3D_HET_MMS_SPT100` | 1.4606 | 3.35 × 10⁻⁸ % | d/κ ≈ 14 |

`max_rel_err_vs_thomas` on the HET case is not a usable accuracy measure: it is a
pointwise maximum of a relative error taken against a reference field with an
interior node, and reads 1.59 × 10² % at d/κ = 3.42 falling only to 5.4 × 10⁻³ %
at saturation, while `err_alg` — a relative L² error — reaches the 10⁻¹² % floor.
Quote `err_alg`, which is what the figures plot.

The equal-accuracy grid was already correct, so those records were re-measured
rather than re-sited; they agree with the superseded ones to eleven significant
figures in every field except `err_alg`, repaired by the 2026-08-19 fix, and
`wall_time_s`.

The first `RUN_TAG=grid_fix` submission, of 2026-08-19, was destroyed before
retrieval by the shared-output-directory collision described in
`results/3Dstudies_4th/PROVENANCE.md`; job 3834215 is the successful re-run.
