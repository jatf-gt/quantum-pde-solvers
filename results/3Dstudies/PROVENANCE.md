# 3-D parameter studies — second-order operator

| file | provenance |
|---|---|
| `sensitivity_qsvt.json` | Job 3834215.pbs-7, `DIM=3 SOLVERS=qsvt RUN_TAG=grid_fix`, started 2026-08-20 11:13:10 at commit a6b6e03. Installed 2026-08-23. |
| `sensitivity_vqls.json` | Job 3894706.pbs-7, `DIM=3 SOLVERS=vqls MAX_WALL_S=9000 RUN_TAG=vqls_sens`, started 2026-08-23 19:17:42 at commit a50ebfb. Completed and installed 2026-08-25. |
| `sensitivity_hhl.json` | Job 3894705.pbs-7, `DIM=3 SOLVERS=hhl MAX_WALL_S=7200 RUN_TAG=hhl`, started 2026-08-23 19:15:34 at commit a50ebfb. Installed 2026-08-24. |
| `equal_accuracy.json` | QSVT entries from job 3834215; HHL entries from job 3894705; VQLS entries from job 3894707.pbs-7 (`SOLVERS=vqls MAX_WALL_S=9000 RUN_TAG=vqls_ea`, started 2026-08-23 19:15:19), installed 2026-08-24. |
| `run_metadata.json` | Job 3834215, plus a `provenance_note`. Describes the QSVT records. |
| `run_metadata.hhl_3894705.json` | Job 3894705's own metadata, retained verbatim. |
| `run_metadata.vqls_ea_3894707.json` | Job 3894707's own metadata, retained verbatim. |
| `run_metadata.vqls_sens_3894706.json` | Job 3894706's own metadata, retained verbatim. |

A run's metadata is kept beside its records rather than reduced to prose because
`config.max_wall_s` is what identifies a wall-clock-truncated row — the record
schema drops the outer `stop_reason`, so the budget must be readable somewhere.
It is 7 200 s for the HHL job and 9 000 s for the VQLS one, and both bound rows
that are installed here.

All records are at N = 8, one pair of entries per case
(`3D_Poisson_TripleSin_cube`, `3D_HET_MMS_SPT100`). As of 2026-08-25 the
directory is complete: all three quantum solvers have both an equal-accuracy
entry and a full sensitivity study. The holding directory
`results/3Dstudies_vqls_sens/`, which carried the partial snapshot of job
3894706 between 2026-08-24 and 2026-08-25, has been removed; its content is
here and the two findings it recorded are reproduced below.

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

## HHL records added 2026-08-24

Job 3894705 sweeps `epsilon` (0.1, 0.05, 0.01, 0.005) and `trotter_steps`
(1, 2, 4, 8, 16) on both cases, and runs the four-point equal-accuracy sweep over
`epsilon`. This is the first 3-D HHL parameter study at any order.

ε responds as it should — this is not the inert-ε defect repaired on 2026-08-17.
On the manufactured cube the residual falls 3.55 × 10⁻² → 8.42 × 10⁻³ →
4.97 × 10⁻³ from ε = 0.1 to 0.005, and `trotter_steps` resolves the same axis
more finely: 1.54 × 10⁻¹ → 3.55 × 10⁻² → 8.42 × 10⁻³ → 2.19 × 10⁻³ →
6.00 × 10⁻⁴ across 1 → 16 steps, a factor of ≈ 4 per doubling against a wall-time
factor of ≈ 2. There is **no turning point** over the measured range, matching
the 2-D result and opposite to the 1-D one, where HHL peaks at four Trotter steps
and then degrades.

ε = 0.1 and ε = 0.05 give bit-identical rows on both cases. That is the pinned
`trotter_steps` axis showing through: both map to the same step count, so the two
grid points are the same solve. The step count itself is not recorded —
`hhl_epsilon` and `hhl_trotter_steps` are null on every outer-path row, the
schema limitation described in `results/2Dstudies/PROVENANCE.md`.

**Neither case reaches the r_target = 10⁻³ band**, best residual 4.97 × 10⁻³
(manufactured) and 5.05 × 10⁻³ (HET) at ε = 0.005, so `in_band` is false for both
HHL equal-accuracy entries. Read them as a lower bound on the ε needed, not as a
converged comparison.

### Three HHL rows are wall-clock truncated

`max_wall_s = 7200` bound three `trotter_steps` rows: 16 on the manufactured cube
(7 200.6 s) and 8 and 16 on the HET cube (7 201.0 s, 7 200.0 s). Their residuals
come from an outer solve stopped mid-iteration, so they are upper bounds on the
error and their wall times are the budget rather than a measurement. The record
schema drops the outer `stop_reason`, so `wall_time_s ≈ max_wall_s` is the only
tell. The trend they sit on is established by the unbounded rows below them.

## VQLS equal-accuracy records added 2026-08-24

Job 3894707 runs the five-point `n_layers` sweep on both cases. Both reach the
band at `n_layers` = 3 — residual 3.00 × 10⁻⁴ (manufactured, 1 414 s) and
5.36 × 10⁻⁴ (HET, 1 963 s) — and both report two of five configurations short of
the target, those being `n_layers` = 1 and 2.

Those same two points are wall-clock truncated at the 9 000 s budget on both
cases, for the reason the 2-D study makes plain: the one- and two-layer ansätze
cannot represent the solution, so the optimiser spends its full budget on every
strip and the outer iteration never converges. Their wall times are the budget,
not a cost measurement. `n_layers` ∈ {3, 4, 5} all completed unbounded, and the
error is flat across them at ≈ 3.3 × 10⁻² % (manufactured) — the ansatz is
saturated by the third layer and further layers buy nothing but cost.

## The HET case's `max_rel_err_vs_thomas` is not usable at any solver

`3D_HET_MMS_SPT100` reports 10⁷–10¹⁰ % on this measure across the new HHL and
VQLS records, against an `err_alg` of 10⁻³–10⁻¹ % on the same rows. This is the
same artefact already documented above for QSVT: `max_rel_err_vs_thomas` is a
pointwise maximum of a relative error taken against a field with an interior
node, and is unbounded there. On this case quote `err_alg`, the relative L²
error. Note that the figures plot `max_rel_err_vs_thomas` and label it
`err_alg_pct`, so the 3-D HET curves in `figures/` are showing the unusable
measure.


## VQLS sensitivity records added 2026-08-25

Job 3894706 completes the 3-D VQLS picture with the five-point `n_layers` and
four-point `n_restarts` sweeps on both cases — eighteen records, the last of the
four jobs submitted on 2026-08-23 to return. A partial snapshot of the same job,
covering `3D_Poisson_TripleSin_cube` only, was held out of this directory from
2026-08-24; the completed file reproduces those nine records field-for-field and
adds the nine for `3D_HET_MMS_SPT100`.

`n_layers` is the axis that carries information, and it carries the same one on
both cases: the ansatz saturates at three layers.

| case | 1 layer | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| `3D_Poisson_TripleSin_cube` | 1.26 × 10⁰ | 3.83 × 10⁻³ | 1.47 × 10⁻⁵ | 1.31 × 10⁻⁴ | 8.63 × 10⁻⁶ |
| `3D_HET_MMS_SPT100` | 8.96 × 10⁻¹ | 4.59 × 10⁻³ | 1.56 × 10⁻⁵ | 9.26 × 10⁻⁶ | 7.99 × 10⁻⁶ |

The quantity is `residual`. Five orders of magnitude are bought between one and
three layers and nothing beyond, the manufactured cube's non-monotone fourth
point being optimiser scatter at a residual already below the discretisation
floor rather than a trend. This agrees with the equal-accuracy sweep of job
3894707 recorded above, which reaches the band at three layers on both cases.

### Sixteen of the eighteen records are wall-clock truncated

`max_wall_s = 9000` bound every record except `n_layers` = 3 on the manufactured
cube (7 229 s) and the invalid `n_restarts` = 1 row on the HET cube. Their wall
times are the budget rather than a measurement, and their residuals are upper
bounds from an outer solve stopped mid-iteration. The record schema drops the
outer `stop_reason`, so `wall_time_s ≈ max_wall_s` is the only tell.

The `n_restarts` sweep is therefore uninformative as recorded — every one of its
points is truncated or invalid — and the `n_layers` sweep is readable in trend
but not in cost. A 3-D VQLS strip solve is more expensive than the 2.5 h budget
allows at low layer counts, where the optimiser exhausts its iteration budget on
every strip; the 2-D equivalent needed 6 943 s at one layer against a 7 200 s
budget, the same effect one notch below the cut. Quote the residual ladder, not
the wall times.

### Both `n_restarts = 1` records are invalid and are excluded

They report residuals of 3.98 × 10⁻⁹ (manufactured) and 3.55 × 10⁻⁹ (HET),
errors against Thomas of 9.21 × 10⁻⁸ % and exactly 0, and `vqls_cost_final =
null` on both. That is the four-part signature of the defect diagnosed on
2026-08-24 and fixed in `solvers/quantum/vqls_1d.py`: at `n_restarts` = 1 the
early-exit branch pads no telemetry slots and the Phase 2 refinement is skipped,
leaving a one-entry cost history that an unconditional `[:-1]` slice reduced to
an empty sequence, on which `np.argmin` raises. Every strip solve raised, every
one was absorbed by the classical fallback in
`solvers.outer.inner.InnerSolverWrapper`, and the recorded field is the Thomas
solution. The fix post-dates this job's submission, so both rows carry it. The
identical signature appears on both cases in `results/2Dstudies/PROVENANCE.md`.

`benchmark/study_plotting.py::_is_classical_fallback` now drops such a record
before it is plotted, keyed on the signature — null `vqls_cost_final` together
with an error against Thomas at round-off — rather than on the parameter value,
so a re-measured `n_restarts` = 1 point is admitted without an edit there. The
main-body thesis figures never read these files, so they were never affected.
