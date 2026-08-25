# 2-D parameter studies — second-order operator

This directory holds records from three separate submissions. `run_metadata.json`
describes the QSVT records, which are the most recent; the table below gives the
origin of every file, and `run_metadata.json` carries a `provenance_note` field
pointing here.

| file | provenance |
|---|---|
| `sensitivity_qsvt.json` | Job 3834214.pbs-7, `DIM=2 SOLVERS=qsvt RUN_TAG=grid_fix`, started 2026-08-20 11:26:29 at commit a6b6e03. Installed 2026-08-23. |
| `sensitivity_hhl.json` | Dedicated 2-D HHL submission (`SOLVERS=hhl MAX_WALL_S=7200`), retrieved 2026-08-20. |
| `sensitivity_vqls.json` | Job 3894682.pbs-7, `DIM=2 SOLVERS=vqls MAX_WALL_S=7200 RUN_TAG=vqls`, started 2026-08-23 18:58:42 at commit e1a6b56. Installed 2026-08-24. |
| `equal_accuracy.json` | QSVT entries from job 3834214, HHL entries from the 2026-08-20 submission, VQLS entries from job 3894682. |
| `run_metadata.json` | Job 3834214, plus a `provenance_note`. |
| `run_metadata.vqls_3894682.json` | Job 3894682's own metadata, retained verbatim. |

A run's metadata is kept beside its records rather than reduced to prose because
`config.max_wall_s` is what identifies a wall-clock-truncated row — the record
schema drops the outer `stop_reason`, so the budget must be readable somewhere.
`config.grids` likewise pins the intended sweep against the points actually
recorded. `run_metadata.json` without a suffix always describes the QSVT records.

## QSVT records re-measured on the corrected degree grid, 2026-08-23

The superseded QSVT records swept `max_degree ∈ {50, 100, 200, 500, None}`, which
lies entirely past the saturation knee: `err_alg` read 10⁻¹² or exactly zero at
every grid point, so the sweep resolved nothing. The replacement sweeps
`{5, 8, 11, 15, 21, 51, 201, 501}` and resolves the knee, in agreement with the
fourth-order result in `results/2Dstudies_4th/`:

The quantity below is `max_rel_err_vs_thomas` [%], which is what the figures
plot and label `err_alg_pct`. The JSON field named `err_alg` is a different
measure — a relative L² error against Thomas, also in per cent. The two differ
by orders of magnitude wherever the reference field has an interior node, and
must not be conflated.

| case | κ | error at the lowest degree | error at the knee | floor (10⁻¹² %) by |
|---|---|---|---|---|
| `2D_Poisson_sin_hom` | 2.7725 | 5.51 × 10⁻⁷ % at d/κ = 1.80 | 3.50 × 10⁻⁸ % at 3.97 | d/κ ≈ 18 |
| `2D_HET_MMS_SPT100` | 1.4629 | 5.63 × 10⁻⁷ % at d/κ = 3.42 | 5.39 × 10⁻⁹ % at 7.52 | d/κ ≈ 14 |

Wall time over the same sweep rises from 4.9 s at cap 5 to 117.1 s at cap 501 on
the manufactured strip: above the knee the cap is a pure cost knob.

The equal-accuracy grid was already correct, so those records were re-measured
rather than re-sited; they agree with the superseded ones to eleven significant
figures in every field except `err_alg`, which the 2026-08-19 fix redefined from
a cancellation of two near-equal quantities to a measured relative L² error, and
`wall_time_s`. The replacement is taken for the repaired `err_alg`.

The first `RUN_TAG=grid_fix` submission, of 2026-08-19, was destroyed before
retrieval by the shared-output-directory collision described in
`results/3Dstudies_4th/PROVENANCE.md`; job 3834214 is the successful re-run.

## VQLS records replaced 2026-08-24

Job 3894682 supersedes the pre-fix set of job 3797112 in `sensitivity_vqls.json`
and in the two VQLS entries of `equal_accuracy.json`. It sweeps `n_layers`
(1…5) and `n_restarts` (1, 2, 3, 5) on both cases at N = 8, and runs the
five-point equal-accuracy sweep over `n_layers`. It arrived in two parts: a
partial snapshot on 2026-08-23 carrying only `2D_Poisson_sin_hom`, and the
completed directory on 2026-08-24. The completion is purely additive — the
`2D_Poisson_sin_hom` records are byte-identical between the two copies.

Convergence in `n_layers` is sharp and monotone on the manufactured strip:
residual 1.63 × 10⁰ → 1.51 × 10⁻³ → 1.76 × 10⁻⁵ → 1.18 × 10⁻⁵ → 9.40 × 10⁻⁶
across one to five layers, the cost falling from 6 943 s to 1 089 s between one
and three layers, because the one-layer ansatz cannot represent the solution and
the optimiser exhausts its budget failing to. The HET strip behaves identically:
3.26 × 10⁻¹ → 1.29 × 10⁻³ → 4.54 × 10⁻⁶ → 2.54 × 10⁻⁶ → 2.11 × 10⁻⁶, 6 805 s
falling to 1 074 s. Both cases reach the r_target = 10⁻³ band, at `n_layers` = 3.

### The `n_restarts = 1` records are a solver failure — exclude them

**Both `n_restarts = 1` records are invalid and must not be quoted.** They report
an error against Thomas of *exactly* zero, `vqls_cost_final = null`, and a
residual at the converged-outer floor (6.29 × 10⁻⁹ manufactured, 3.43 × 10⁻⁹
HET), in roughly half the wall time of the working configurations. The cause was
established on 2026-08-24 and is a defect in `solvers/quantum/vqls_1d.py`, since
fixed: with `n_restarts = 1` the early-exit branch pads no telemetry slots and
the Phase 2 refinement is skipped, so the cost-history list held a single entry
and an unconditional `[:-1]` slice reduced it to an empty sequence, on which
`np.argmin` raises. Every strip solve therefore raised, and every one was
absorbed by the classical fallback in
`solvers.outer.inner.InnerSolverWrapper` — so the recorded field is the Thomas
solution and the row measures nothing about VQLS. The same signature appears in
`results/3Dstudies_vqls_sens/`.

The remaining points, `n_restarts` ∈ {2, 3, 5}, are bit-identical to one another
and to the `n_layers = 4` record. That is correct rather than defective: child
seeds are drawn in sequence from one master seed, so the opening restart is the
same for any `n_restarts`, and it already meets the tolerance, after which the
early exit skips the rest. The sweep demonstrates that restarts are inert at
this conditioning; it does not measure a cost-accuracy trade.

### Two schema limitations, both still live

The VQLS diagnostic columns are null throughout — `vqls_n_layers`,
`vqls_n_restarts`, `vqls_n_evaluations`, `vqls_converged`. The 2026-08-19 repair
of those fields reached `benchmark/sensitivity.py::run_sensitivity`, the 1-D
path, which now populates them; the outer path, `sensitivity_sweep_outer`, still
records nothing. Some of that is inherent — a 2-D solve runs N independent strip
optimisations per outer iteration, so there is no single evaluation count — but
`n_layers` and `n_restarts` are settings rather than outcomes and could be
recorded. Until they are, the swept value is recoverable only from
`sensitivity_value`.

The record schema also drops the outer `stop_reason`, so a wall-clock-truncated
solve is identifiable only by `wall_time_s` reaching `max_wall_s`. No record in
this directory does: the largest is 6 943 s against the 7 200 s budget.

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
