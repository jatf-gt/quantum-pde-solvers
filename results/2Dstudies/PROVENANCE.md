# 2-D parameter studies — second-order operator

## HHL records replaced 2026-08-20

`sensitivity_hhl.json` and the two HHL entries of `equal_accuracy.json` were
replaced with the output of the dedicated 2-D HHL submission
(`SOLVERS=hhl MAX_WALL_S=7200`), retrieved from CX3 on 2026-08-20. The
superseded records were the pre-fix set described in the project memory: they
recorded `hhl_trotter_steps = ⌈1/ε⌉` (10, 20, 100, 200) and returned an
**identical residual at every ε** — 6.8601 × 10⁻³ for the manufactured case and
1.5640 × 10⁻³ for the HET case — because ε never reached the solver.

The replacement responds as expected. Manufactured case, ε = 0.1 → 0.005:
residual 6.50 × 10⁻² → 2.41 × 10⁻² → 6.86 × 10⁻³ → 3.86 × 10⁻³. It also adds a
`trotter_steps` sweep (1, 2, 4, 8, 16) per case, which the superseded set did
not contain: residual 5.30 × 10⁻¹ → 1.02 × 10⁻³ on the manufactured case.

Neither case reaches the r_target = 10⁻³ band on the manufactured source; the
HET case does.

## VQLS records are still the pre-fix set

`sensitivity_vqls.json` and the two VQLS entries of `equal_accuracy.json` are
unchanged from the 2026-08-17 run (job 3797112). No VQLS submission has been
made at the corrected settings, so these are the only 2-D VQLS study data that
exist and they carry the defects recorded in the project memory. Do not quote
them without re-running.

## QSVT records are on the superseded degree grid

The QSVT entries here were measured over `max_degree ∈ {50, 100, 200, 500, None}`,
which lies entirely past the saturation knee at degree/κ ≈ 11: `err_alg` reads
10⁻¹² or exactly zero at every grid point, so the sweep resolves nothing. The
`RUN_TAG=grid_fix` submission that would have re-measured them over
`{5, 8, 11, 15, 21, 51, 201, 501}` did not survive — see below. The fourth-order
equivalent, which did land, is in `results/2Dstudies_4th/` and shows the knee
clearly.

## Why the second-order re-run is missing

Every submission for a given dimension writes into the single shared directory
`results/{DIM}Dstudies` on the cluster, and `SweepArchive.write_sensitivity`
overwrites `sensitivity_<solver>.json` wholesale while
`append_equal_accuracy` merges on `(case_id, solver, N)` — a key that does not
include the discretisation order. The order-4 submission of 2026-08-19 therefore
displaced the order-2 QSVT records of the `grid_fix` submission that ran
alongside it. Any future pair of submissions differing only in `ORDER` will
collide the same way unless they are given separate output directories.
