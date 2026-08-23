# 3-D parameter studies — fourth-order operator

Installed 2026-08-20 from the CX3 arrival `results/updated_studies3D/`
(since deleted; a copy of the arrival is retained in the session scratchpad).

`sensitivity_qsvt.json` and `equal_accuracy.json` hold four records each — two
cases × N ∈ {8, 16} — all at `discretisation_order = 4`, over
`max_degree ∈ {5, 8, 11, 15, 21, 51, 201, 501}`. They are the output of the
`DIM=3 SOLVERS=qsvt ORDER=4 N_VALUES=8,16 RUN_TAG=o4` submission.

## The metadata does not belong to the data

The arriving `run_metadata.json` recorded job **3831760.pbs-7**, which is the
*second-order* `RUN_TAG=grid_fix` submission of 2026-08-19 20:42:04. Both jobs
write into the shared cluster directory `results/3Dstudies`, and
`run_studies.py` writes `run_metadata.json` before its first case, so the later
submission's metadata displaced the earlier one's while leaving its data in
place. The archive copied off the cluster therefore pairs one job's metadata
with another job's results.

This matters beyond bookkeeping: `hpc/runners/make_tables.py::_study_order`
reads `config.order` from that file to decide whether a study directory may be
tabulated under a given order. Left uncorrected it would have suppressed these
tables at `--order 4` and emitted them, mislabelled, at `--order 2`.

`run_metadata.json` here has `run_tag`, `config.order` and `config.n_values`
corrected from the records themselves, `config.pbs_jobid` marked unknown, and a
`provenance_note` field stating the above. The recorded parameter grids were
identical between the two submissions and match the data, so nothing else
required correction.

The displaced file was kept for a time as `run_metadata.grid_fix_3831760.json`
and was **deleted on 2026-08-23**: job 3831760 produced no retrievable data, and
the second-order sweep it was meant to record has since been re-run successfully
as job 3834215 and installed in `results/3Dstudies`. Its identifying details are
recorded in the paragraph above; nothing else referred to the file.

## Wall-clock cap on one record

`3D_HET_MMS_SPT100`, N=16, `max_degree = 501` reports `wall_time_s = 3600.1`,
i.e. it terminated on the `MAX_WALL_S = 3600` per-solve bound rather than at the
outer tolerance. Its residual (3.07 × 10⁻⁶) is that of a truncated outer solve
and is not comparable with the converged 10⁻⁹ residuals of the other seven grid
points in the same sweep. Every other record in this directory converged.
