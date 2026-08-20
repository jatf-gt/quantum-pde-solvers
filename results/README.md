# Recorded Results

Output of the benchmark sweeps and parameter studies reported in the
dissertation. Everything here is *recorded* data: it is written by the runners
in `hpc/runners/` and consumed by the post-processing layer in `benchmark/`.
Nothing under this directory is source, and nothing here should be edited by
hand.

## What is in the repository and what is not

The directory holds about 407 MB. Roughly 7 MB of that is tracked — the
summaries, tables and figure datasets, which are the provenance of every number
quoted in the dissertation. The remaining 400 MB is excluded by kind in
`.gitignore`, because git retains every version of a binary permanently and this
repository is public.

| Kind | Tracked | Size | Why |
|---|---|---|---|
| `results_full.json`, `sensitivity_*.json`, `equal_accuracy.json`, `run_metadata*.json` | yes | 5.2 MB | The recorded measurements. Every figure and table is derived from these. |
| `tables/*.tex`, `tables/*.txt` | yes | 0.2 MB | Rendered `booktabs` tables, for direct `\input{}`. |
| `*.csv` | yes | 2.3 MB | Tidy per-figure datasets, one row per plotted point. |
| `qsvt_phase_cache/*.npz` | yes | 3.6 MB | QSP phase angles. Tracked deliberately: recomputing the 1-D set at large κ takes tens of hours, and the cache key is exact, so a regenerated file is not interchangeable with the recorded one. |
| `solutions_*.npz`, `solution3d_*.npz` | **no** | 182 MB | Per-solution fields. 137 MB of it is the 3-D sweep at N = 64 alone. |
| `*.log` | **no** | 162 MB | Cluster job logs. Two files in `2Dhpc_run/` are 78 MB each. |
| `*.png`, `*.pdf` | **no** | 50 MB | Diagnostic figure sets, regenerable from the archives by `hpc/runners/plot_results.py`. |

The exclusions are written **by kind, never by directory**. Git does not descend
into an excluded directory, so a negation beneath one can never match; excluding
`results/2Dhpc_run/` would make it impossible to track anything inside it. Any
rule added to `.gitignore` that names a directory under `results/` forfeits that
ability.

## The per-solution archives

The `.npz` fields are the one thing here that cannot be regenerated without a
cluster allocation — a full re-run is measured in days of CPU time. They are not
in the repository for the size reason above.

> **Data deposit: not yet created.** The intended home is a Zenodo record with a
> DOI, cited from the dissertation. Until that exists the `.npz` archives live
> only on the author's machine and on the CX3 RDS allocation. Replace this
> paragraph with the DOI once the deposit is made.

Nothing in the dissertation depends on them: every figure and table is built
from the tracked summaries and tidy CSVs. They are needed only to re-render a
solution field or to re-derive a metric that was not recorded at the time.

## Layout

| Path | Contents |
|---|---|
| `{1,2,3}Dhpc_run/` | Primary sweep, second-order stencil, per dimension. |
| `{1,2,3}Dhpc_run_4th/` | The same at fourth order. |
| `1Dhpc_run_degcap5000/` | Uniform-degree QSVT ladder, a fixed cap of 5001 across every N. |
| `{1,2,3}Dstudies/` | Equal-accuracy and one-at-a-time sensitivity studies, second order. |
| `{2,3}Dstudies_4th/` | The same at fourth order. Each carries a `PROVENANCE.md`. |
| `thesis/` | The main-body figure datasets, `F1`–`F8`, and the `T2`/`T3` table data. One tidy CSV per figure, written by `scripts/make_thesis_figures.py`. |
| `qsvt_phase_cache/` | QSP phase angles, keyed `(κ, ε, method, max_degree)`. |
| `manifests/` | Re-run scopes: which rows of which sweep a repair job was asked to replace. |
| `investigations/` | One-off studies outside the main sweeps, including the `ibm_kingston` hardware runs and calibration. |
| `meetings/` | Supervisor meeting material. Excluded from the repository in full. |

The on-disk schema — filename convention, field-name aliases, the `missing()`
report — is declared once in `benchmark/results_io.py` and is the contract to
read these files against. `benchmark/hpc_archive.py` covers the older sweep
layout.

## A recorded run is not always the run you think

Two failure modes have produced archives that read as complete and were not.
Both are documented where they occurred, and both are worth knowing before
quoting anything from a directory you did not write yourself.

- A job that dies before its first case leaves the *previous* run's files in
  place, and the archive step then copies those under today's date. Check
  `run_metadata.json`'s `timestamp` and `pbs_jobid` against the job you
  submitted; never compare result values.
- Every studies submission for a given dimension writes into one shared
  directory, and the merge key does not include the discretisation order, so two
  jobs differing only in `ORDER` overwrite each other. `3Dstudies_4th/` carries
  the evidence: one job's metadata, preserved as
  `run_metadata.grid_fix_3831760.json`, sitting on another job's results.
