# 2-D parameter studies — second-order operator

This directory holds records from three separate submissions. `run_metadata.json`
describes the QSVT records, which are the most recent; the table below gives the
origin of every file, and `run_metadata.json` carries a `provenance_note` field
pointing here.

| file | provenance |
|---|---|
| `sensitivity_qsvt.json` | Job 3834214.pbs-7, `DIM=2 SOLVERS=qsvt RUN_TAG=grid_fix`, started 2026-08-20 11:26:29 at commit a6b6e03. Installed 2026-08-23. |
| `sensitivity_hhl.json` | Dedicated 2-D HHL submission (`SOLVERS=hhl MAX_WALL_S=7200`), retrieved 2026-08-20. |
| `sensitivity_vqls.json` | Job 3797112.pbs-7 of 2026-08-17. **Pre-fix; do not quote.** |
| `equal_accuracy.json` | QSVT entries from job 3834214, HHL entries from the 2026-08-20 submission, VQLS entries from job 3797112. |
| `run_metadata.json` | Job 3834214, plus a `provenance_note`. |

## QSVT records re-measured on the corrected degree grid, 2026-08-23

The superseded QSVT records swept `max_degree ∈ {50, 100, 200, 500, None}`, which
lies entirely past the saturation knee: `err_alg` read 10⁻¹² or exactly zero at
every grid point, so the sweep resolved nothing. The replacement sweeps
`{5, 8, 11, 15, 21, 51, 201, 501}` and resolves the knee, in agreement with the
fourth-order result in `results/2Dstudies_4th/`:

| case | κ | err_alg at d/κ ≈ 1.8–3.4 | floor reached at |
|---|---|---|---|
| `2D_Poisson_sin_hom` | 2.7725 | 2.25 × 10⁻⁷ % | d/κ ≈ 18 |
| `2D_HET_MMS_SPT100` | 1.4629 | 1.33 × 10⁻⁷ % | d/κ ≈ 10 |

The equal-accuracy grid was already correct, so those records were re-measured
rather than re-sited; they agree with the superseded ones to eleven significant
figures in every field except `err_alg`, which the 2026-08-19 fix redefined from
a cancellation of two near-equal quantities to a measured relative L² error, and
`wall_time_s`. The replacement is taken for the repaired `err_alg`.

The first `RUN_TAG=grid_fix` submission, of 2026-08-19, was destroyed before
retrieval by the shared-output-directory collision described in
`results/3Dstudies_4th/PROVENANCE.md`; job 3834214 is the successful re-run.

## VQLS records are still the pre-fix set

No VQLS submission has been made at the corrected settings, so `sensitivity_vqls.json`
and the two VQLS entries of `equal_accuracy.json` are the only 2-D VQLS study data
that exist and they carry the defects recorded in the project memory — a null
`vqls_n_evaluations` and `vqls_converged`, and the pre-fix `err_alg`. Do not quote
them without re-running.

## HHL records replaced 2026-08-20

The superseded HHL records were the pre-fix set: they recorded
`hhl_trotter_steps = ⌈1/ε⌉` (10, 20, 100, 200) and returned an **identical
residual at every ε** — 6.8601 × 10⁻³ for the manufactured case and
1.5640 × 10⁻³ for the HET case — because ε never reached the solver.

The replacement responds as expected. Manufactured case, ε = 0.1 → 0.005:
residual 6.50 × 10⁻² → 2.41 × 10⁻² → 6.86 × 10⁻³ → 3.86 × 10⁻³. It also adds a
`trotter_steps` sweep (1, 2, 4, 8, 16) per case, which the superseded set did
not contain: residual 5.30 × 10⁻¹ → 1.02 × 10⁻³ on the manufactured case.

Neither case reaches the r_target = 10⁻³ band on the manufactured source; the
HET case does.
