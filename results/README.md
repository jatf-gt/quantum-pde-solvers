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

They are deposited on Zenodo, under **[10.5281/zenodo.22071066](https://doi.org/10.5281/zenodo.22071066)**
— 694 fields, 179 MB, one zip per sweep, each carrying that sweep's
`results_full.json` and `run_metadata.json` so a single downloaded archive is
self-describing, plus a `MANIFEST.csv` indexing every field by case, solver and
resolution with its SHA-256. Cite that DOI for these exact bytes; the concept
DOI [10.5281/zenodo.22071065](https://doi.org/10.5281/zenodo.22071065) always
resolves to the newest version.

The deposit is rebuilt by `python scripts/utils/make_zenodo_package.py --out <dir>`
and uploaded by `python scripts/utils/zenodo_upload.py --package <dir>`, whose
metadata is `.zenodo.json` at the repository root.

Nothing in the dissertation depends on them: every figure and table is built
from the tracked summaries and tidy CSVs. They are needed only to re-render a
solution field or to re-derive a metric that was not recorded at the time.

## Layout

| Path | Contents |
|---|---|
| `{1,2,3}Dhpc_run/` | Primary sweep, second-order stencil, per dimension. |
| `{1,2,3}Dhpc_run_4th/` | The same at fourth order. |
| `1Dhpc_run_degcap5000/` | Uniform-degree QSVT ladder, a fixed cap of 5001 across every N. |
| `1Dhpc_run_degcap5000_lowN/` | The same ladder at N = 4 and 8, which the run above did not cover. Second order. |
| `1Dhpc_run_4th_degcap_lowN/` | Fourth-order ladder at N = 4 and 8, cap 14999. |
| `{2,3}Dstudies_hhldeep/` | HHL equal-accuracy re-runs on the epsilon grid extended to 1e-3. Merged into `{2,3}Dstudies/` on the record key; kept as the raw archive. |
| `{1,2,3}Dstudies/` | Equal-accuracy and one-at-a-time sensitivity studies, second order. |
| `{1,2,3}Dstudies_4th/` | The same at fourth order. Each carries a `PROVENANCE.md`. |
| `thesis/` | The main-body figure datasets, `F1`–`F9`, and the `T2`/`T3`/`T4` table data. One tidy CSV per figure, written by `scripts/make_thesis_figures.py`. |
| `qsvt_phase_cache/` | QSP phase angles, keyed `(κ, ε, method, max_degree)`. |
| `manifests/` | Re-run scopes written by `scripts/utils/gap_analysis.py`: which rows of which sweep were outstanding, and were subsequently recomputed. Kept as the record of what each sweep was missing and when. |
| `investigations/` | One-off studies outside the main sweeps, including the `ibm_kingston` hardware runs and calibration. |
| `meetings/` | Supervisor meeting material. Excluded from the repository in full. |

The two `_lowN` directories are separate archives rather than rows appended to
their parents because a capped QSP degree is not a ceiling on an uncapped one:
`qsp_angles` truncates the target Chebyshev polynomial to `max_degree` and fits
it by least squares, where the uncapped path calls `PolyOneOverX.generate`. The
two constructions differ by orders of magnitude in the residual at comparable
degree/κ, so their rows must not be mixed into one series. `poster_build`'s
accuracy figure admits a QSVT row only if it records the cap its ladder is
defined at.

A studies directory is named for the dimension, the discretisation order and the
run tag together — `1Dstudies`, `1Dstudies_4th` — because none of the three is
recoverable from the filenames within it. A tag beyond the order marks a run not
yet merged into its untagged counterpart, either because it is unfinished or
because it has not been reviewed; `2Dstudies_vqls/` and `3Dstudies_vqls_sens/`
were such directories until their jobs completed, and both are now merged into
their untagged counterparts with the tagged copies gone. No tagged directory
remains.

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
- Every studies submission for a given dimension used to write into one shared
  directory, and the merge key does not include the discretisation order, so two
  jobs differing only in `ORDER` overwrote each other. That destroyed the
  second-order `grid_fix` pair of 2026-08-19 and left `3Dstudies_4th/` holding
  one job's metadata on another job's results. **Fixed 2026-08-23**: the output
  directory now carries the dimension, the order and the run tag
  (`run_studies.py::results_dir_for`), and the runner refuses to write into a
  directory whose recorded order differs from its own. Archives predating that
  date still warrant the check above.
