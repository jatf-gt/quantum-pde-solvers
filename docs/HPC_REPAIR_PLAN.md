# HPC Benchmark Pipeline Repair Plan

**Raised:** 2026-08-09 · **Scope:** `hpc/runners/`, `hpc/jobs/`, `solvers/quantum/`,
`solvers/outer/`, `problems/`, `benchmark/results_io.py`

Repair of the 2nd- and 4th-order HPC benchmark pipeline following the failed
integration of the 4th-order pentadiagonal discretisation, and completion of the
2nd-order results left incomplete by the SPT-100 geometry correction (`861ff46`).

---

## Status

*Last revised 2026-08-10.*

| # | Phase | State |
|---|-------|-------|
| 0 | Stop, preserve, and scope | **complete** — `results_hpc_copy/` holds the re-synced backup; geometry impact settled |
| 1 | Gap manifest | **complete** — all three manifests generated; classification corrected three times against real data |
| 2 | Correctness: pentadiagonal operator | **complete** — dense block encoding in, truncation now raises, `hhl_4th`/`qsvt_4th` registered |
| 3 | Environment and deployment | complete |
| 4 | Fold order=4 into the `LineProblem` protocol | **1-D closure fixed and pinned**; 2-D/3-D protocol work outstanding (Wave 2) |
| 5 | Reporting, diagnostics and provenance | **partly done** — the items Wave 1 depends on are in; the new `RunResult` columns are not |
| 6 | QSVT phase precompute for 4th order | **1-D done** (`--order 4`, `hpc/jobs/submit_precompute_4th.sh`); 2-D/3-D refused until Phase 4 settles the operator |
| 7 | Consolidate `hpc/jobs/` and submit | **scripts ready; submission is the user's call** |

### Wave 1 is ready to submit

Outstanding work, from `results/manifests/rerun_{1,2,3}d.json`:

| Sweep | Rows | Job | Walltime |
|---|---|---|---|
| 1D | 22 of 140 | `hpc/jobs/submit_1d_wave1.sh` | 6 h |
| 2D | 55 of 100 | `hpc/jobs/submit_2d_wave1.sh` | 60 h |
| 3D | 40 of 84 | `hpc/jobs/submit_3d_wave1.sh` | 48 h |

Every job sources `_preflight.sh`, which **refuses a dirty tree** — so the working
tree must be committed and pushed before any submission.

**Regenerating the manifests requires the exact scope they were built with.**
`gap_analysis.py` takes both the results directory and the N range as arguments,
and neither default is the right one. Verified to reproduce 22 / 55 / 40 on
2026-08-10:

```bash
python scripts/gap_analysis.py --dim 1 --results-dir results/1Dhpc_run           --n-values 4,8,16,32,64
python scripts/gap_analysis.py --dim 2 --results-dir results_hpc_copy/2Dhpc_run  --n-values 4,8,16,32,64
python scripts/gap_analysis.py --dim 3 --results-dir results/3Dhpc_run           --n-values 4,8,16
```

Two traps, both of which inflate the manifest into scheduling work that already
exists or was never in scope:

- **2D reads the backup, not the live directory.** `results/2Dhpc_run/` had its
  `results_full.json` truncated to 2 rows by a job that was still running;
  `results_hpc_copy/2Dhpc_run` is the authoritative copy. Pointing 2D at the live
  directory reports **98** outstanding against the true 55.
- **3D is scoped to N = 4, 8, 16.** It has no 32 or 64 configurations at all;
  passing the 1D/2D N range invents 56 rows that were never planned and reports
  **96** against the true 40.

### What was fixed on 2026-08-10, and what it prevented

| Defect | Consequence had it shipped |
|---|---|
| `run.log` opened `mode="w"` | Each wave-1 script runs its driver 3–5 times; every step destroyed the previous step's log, so a job that ran five steps would have ended holding only the fifth. Now appended, with a `SESSION START` banner per invocation. |
| `--append` did not deduplicate | `submit_2d_wave1.sh` deliberately revisits section 3 at N=32, where three of four solvers are missing. Every already-present solver would have gained a second, contradictory row. Rows are now superseded on `(case, solver, N)`. **The 3D summary was already carrying four such duplicate sets**, since collapsed. |
| 1D driver had no scope control | Its only flag was `--max-n`, and `_save_results` overwrote. Reaching 22 outstanding rows meant re-running all 140 — including thirteen 1 h HHL solves. It now takes `--n-values`, `--sections`, `--solvers`, `--cases`, `--append` and `--phase-tag`. |
| Thirteen HHL rows misread as errors | `_run_hhl`'s hard 3600 s budget produced `u=None` and the note `"solver_error"`, indistinguishable from a crash. All thirteen record `wall_time_s = 3600.2` exactly: they are *measured* timeouts at κ ≈ 1.7e3, which is the benchmark's finding. Re-running them would have spent 13 h reproducing thirteen identical rows. The runner now records `hhl_timeout`, and `gap_analysis.py` treats a timeout as terminal. **1D outstanding fell 33 → 22.** |
| Two 3D orphan rows | `scripts/recover_orphan_rows.py` rebuilds a summary row from a stored field by calling the runner's own `_record`, so the metric definitions match the instrumented rows exactly. Recovered both ≈ 6 h HHL solves at N=16 with no simulation. Cost columns are nulled, never zeroed, and every recovered row carries `notes="recovered_from_archive"`. **3D outstanding fell 42 → 40; ≈ 12 h preserved.** |
| `gap_analysis.py` and `check_geometry_impact.py` were unrunnable as scripts | Neither put the repository root on `sys.path`, so `python3 scripts/gap_analysis.py` always failed on `import benchmark`. Both wave-1 scripts call it for their post-run analysis behind `\|\| true`, so the manifest would simply never have appeared. |
| Superseded jobs still submittable | Six one-off scripts removed (`*_gapfill*`, `*_complete`, `*_resume`, `*_geometry_fix`). `submit_hpc_1D_geometry_fix.sh` was the dangerous one: its header asserts that 1D has no `--append`, which is no longer true, and it would have overwritten 118 sound rows. |

New regression cover: `tests/test_gap_analysis.py` (16 tests pinning the
classification policy, including that the 1D wall-clock inference must *not* apply in
2D/3D, where `wall_time_s` is a whole outer iteration) and `tests/test_hpc_runners.py`
(14 tests on supersession and scope). Suite: 623 passed, 5 skipped.

---

## 0bis. Phase 8 — The professional benchmarking framework

*Added 2026-08-10, from `context/13 - Designing a professional benchmarking
framework for quantum solvers.txt`.*

A publication-grade benchmarking and sensitivity layer has been implemented in
`benchmark/`: `metrics.py` (rewritten), `results_io.py` (rewritten),
`equal_accuracy.py`, `sensitivity.py`, `hardware.py`, `tables.py`, and rewritten
`plotting.py`, `reporting.py`, `runner.py`. Part II of that document specifies the
runner adaptations still required. **Objective: submit the outstanding jobs once,
already recording every quantity the thesis and any subsequent paper will need**,
rather than resubmitting later for metrics that were not captured.

### 8.1 Collateral damage, and the repair

The rewrite replaced `benchmark/results_io.py` wholesale. That module was **not**
a generic persistence layer: it was the declared on-disk schema for the existing
`results/{1,2,3}Dhpc_run/` sweeps. Losing it broke, silently in two cases:

| Consumer | Consequence |
|---|---|
| **`scripts/gap_analysis.py`** | `TypeError` on `SweepArchive(dim=…)`. **This is the tool that generates the rerun manifests** — the only thing standing between a resubmission and the recomputation of 25–38 h rows. All three wave-1 job scripts call it behind `\|\| true`, so the manifest would simply never have appeared. |
| `benchmark/hpc_plotting.py` | `ImportError` on `SOLVER_ORDER`. Every published HPC figure, 1D/2D/3D. |
| `scripts/recover_orphan_rows.py`, `scripts/resource_feasibility_1d.py` | Same module, same failure. |
| `tests/test_integration.py` | `ImportError` on `BenchmarkResult2D`. |

**Repaired** by separating the two abstractions, which are genuinely different and
had merely collided on one name:

| | `benchmark/hpc_archive.py` *(restored)* | `benchmark/results_io.py` *(new)* |
|---|---|---|
| Describes | `results/{1,2,3}Dhpc_run/` | `results/<run_tag>/` + `tables/`, `figures/` |
| Constructor | `SweepArchive(dir, dim=…)` | `SweepArchive(root, run_tag=…)` |
| Rows | plain dicts, `FIELD_ALIASES` | typed `BenchmarkResult` |
| Mode | read-only | read/write, incremental |

Both keep the class name and are told apart by module, so the 27 existing
annotations were untouched and only import lines changed. The legacy contract is
fixed by data already on disk and cannot be renegotiated; new work targets
`results_io`. Verified: the manifests reproduce at **22 / 55 / 40**.

### 8.2 Two departures from the document, on academic grounds

The document invites improvement, and two of its proposals should not be adopted
as written.

**(a) 2-D/3-D fields must not go into `notes` as a JSON string.** Part II proposes
storing `n_outer`, `convergence_factor` and `level_kappas` in the free-text
`notes` field "until a `BenchmarkResult2D` subclass is warranted". It is
warranted. These are *primary results*: Section IV F is precisely a study of
outer-iteration residual histories, and 2-D/3-D account for the majority of the
outstanding rows. Buried in a JSON string they are untyped, absent from the
summary CSV, and invisible to `tables.py` and `plotting.py` — which is to say,
not reportable. The correct construction is a typed
`BenchmarkResult2D(BenchmarkResult)` carrying `scheme_requested`,
`scheme_effective`, `inner_solver`, `n_outer`, `converged`,
`convergence_factor`, `residual_history`, `stop_reason`, `n_strip_solves`,
`weighted_cost`, `level_kappas`, `kappa_row`, `inner_failures` and
`inner_fallback_frac`, with the N×N field itself going to the `.npz` archive as
it already does — keeping the result row flat and serialisable.

This overlaps almost exactly with the **Phase 5** columns still outstanding
(`scheme_requested`/`scheme_effective`, `capped_by`, `inner_failures`,
`inner_fallback_frac`, `degraded`), so the two should be implemented once,
together, rather than twice in different shapes.

**(b) `_build_base_result` cannot serve 2-D/3-D unmodified.** It takes `(A, b)`
and computes the residual from them. There is no assembled `A` in 2-D/3-D — the
operator exists only as `problem.apply()` — so the document's instruction to
reuse it "passing `u_thomas` as the reference" cannot be followed literally
without fabricating a dense N²×N² system, which at N=64 is a 16.8 M-entry matrix
assembled solely to compute a number `problem.apply()` already gives. It should
be generalised to accept **either** `(A, b)` **or** a precomputed residual, so
that 2-D/3-D pass the residual their own operator produces. The invariant the
document rightly insists on — that metrics are measured from the solution vector,
never inferred from parameters — is preserved either way.

### 8.3 State

- **Done:** the archive split (8.1), all new modules import, suite green except
  `tests/test_integration.py`.
- **Outstanding:** the typed `BenchmarkResult2D` and the `_build_base_result`
  generalisation (8.2); the runner adaptations of Part II for
  `run_{1,2,3}d.py` and `plot_results.py`; `--sensitivity`, `--equal-accuracy`,
  `--extract-circuits` and `--r-target` flags; post-run table and figure
  generation.

---

## 1. Summary

**No 4th-order quantum result produced to date is valid.** HHL failed on every
recorded row (the cluster virtual environment lacks the pentadiagonal matrix
module), and both HHL and QSVT were silently solving a *truncated tridiagonal*
operator rather than the pentadiagonal one. The originally reported symptom —
inconsistent log and archive formatting — is real but is the least consequential
of the defects found.

**Governing constraint on all resubmission:** results that are both correct and
expensive must never be recomputed. Individual rows in the existing archive cost
25–38 h of wall time (2D §5 N=64 HHL ≈ 38.6 h; 3D §3 N=16 VQLS ≈ 37 h). Every
resubmission is therefore driven from a generated manifest rather than from
hand-written resolution ranges.

---

## 2. Root causes

### R1 — The cluster virtual environment installs the wrong `quantum_linear_solvers`

`hpc/setup_hpc_env.sh:47`, and its GPU counterpart, install
`git+https://github.com/anedumla/quantum_linear_solvers.git` — the **upstream**
repository, which contains no `pentadiagonal_toeplitz.py`. The pentadiagonal
implementation resides in the fork `jatf-gt/quantum_linear_solvers` (commits
`b454716`, `c0a719e`), which is fully pushed. No script under `hpc/jobs/` refreshes
the package after initial environment creation.

Compounding this, the parent repository records `quantum_linear_solvers` as a
**gitlink with no `.gitmodules`**, so a fresh clone leaves the directory empty and
`pip install -e quantum_linear_solvers/` cannot succeed on the cluster.

*Consequence:* 100 % of HHL rows failed — 8 of 8 in `results/2Dhpc_run_4th/`, 7 of 7
in `results/3Dhpc_run_4th/`.

### R2 — HHL and QSVT silently solve the wrong operator under `--order 4`

- `solvers/quantum/qsvt_1d.py:270-281` reconstructs a tridiagonal matrix from two
  scalar entries (`A[0,0]`, `A[0,1]`), discarding the ±2 band entirely.
  `solvers/quantum/hhl_1d.py:131-143` does the same via `TridiagonalToeplitz`.
- `solvers/outer/inner.py` registers only these 2nd-order factories; no
  pentadiagonal entry exists. `solvers/outer/multigrid_4th.py` obtains its strip
  solver from that same registry (lines 205, 249, 435, 479).
- `solvers/quantum/hhl_1d_4th.py` and `qsvt_1d_4th.py` are **dead code**, never
  imported by any runner or by `inner.py`. Furthermore `qsvt_1d_4th.py:134-137`
  delegates back to the defective `qsvt_solve_system`, so it inherits the
  truncation even once wired in.
- `hpc/runners/run_1d.py:498,643` import the 2nd-order modules unconditionally,
  irrespective of `--order`.
- VQLS is unaffected: `vqls_1d.py:380` uses a full Pauli decomposition of the
  complete matrix.

*Tractability:* `_sznagy_dilation(M)` (`solvers/quantum/block_encoding.py:233`)
already dilates an arbitrary Hermitian `M`. Only the constructor
`build_tst_block_encoding(N, main_diag, off_diag)` is TST-specific, so a generic
`build_dense_block_encoding(A)` is a modest addition reusing the existing
dilation. `PentadiagonalToeplitz` already exists and is correct for HHL.

### R3 — No wall-clock cap exists in the 4th-order path

`max_wall_s` is implemented in `solvers/outer/multigrid.py` (checked per sweep,
`_v_cycle:228-257`) and `solvers/outer/stationary.py:153` (per iteration), and is
forwarded correctly by `solve()`. However `multigrid_4th.py` and the
`scripts/debug_*_4th.py` schemes contain **no reference to it whatsoever**, and
`_run_4th_order_solver_2d`/`_3d` never read `cfg.scheme_options["max_wall_s"]`.

A `-S max_wall_s=21600` argument is therefore parsed, accepted, and **silently
discarded**; the run proceeds to its `max_iter` bound and is terminated by PBS.
This is the origin of the reported "cap does not work / crashes things" behaviour.

The 2nd-order cap functions, but at per-sweep granularity: at N=64 with HHL one
sweep costs ≈ 3.2 h, which bounds the overshoot.

### R4 — The cluster has been executing uncommitted code

Every `run_metadata*.json` across all six result directories records
`"git_dirty": true`. The 2nd-order 2D log shows HHL tracebacks descending into
`_run_4th_order_solver_2d` under `--order 2`, at line numbers inconsistent with the
committed file. Those rows cannot be reconstructed from `git log`.

### R5 — Divergence in reporting between the two orders

| Aspect | order = 2 | order = 4 |
|---|---|---|
| Solver internals | `log.info` | `print()` with ANSI colour, then discarded by `redirect_stdout(StringIO())` (`run_2d.py:511`) |
| Strip accounting | true per-solve `WorkLog.add` within `strip_sweep` | `w.add(N, iters)` in 2D against `w.add(N, N*iters)` in 3D — mutually inconsistent, and both incorrect |
| `inner_*`, `qsvt_degree`, `level_kappas` | populated from `InnerSolverWrapper.summary()` | always `None`/`0` |
| Fallback record | `notes="scheme_fallback:…"` | encoded into the `scheme` string as `"line-sor-4th (fallback)"`; `notes` left untouched |
| `order` recorded | 2D/3D metadata only | 1D: nowhere |

Additionally: the 1D driver writes `all_solutions.npz` using flattened keys that
`SweepArchive` cannot read (and which `benchmark/results_io.py` documents as read
by nothing); `RESULTS_DIR` is a module global mutated in `main()` yet read inside
`ProcessPoolExecutor` workers (`run_1d.py:94,334,1088-1094`), which is correct only
under the `fork` start method; 1D `--order 4` skips sub-case 3c, altering row
counts; and `run.log` is opened with `mode="w"`, so **every invocation destroys the
preceding log**.

### R6 — Confirmed gaps and suspect rows

| Hole | Detail |
|---|---|
| **3D HET** (`MMS_SPT100`, `RotatingSpoke`, `Discharge`) | No rows at any N, for any solver — stripped by `scripts/cleanup_stale_geometry.py`; the rerun never executed. Approximately 36 rows. The largest confirmed gap. |
| **2D HET** (`MMS_SPT100`, `Sin_MeetingReport`) | Stripped; N=32 and N=64 show no evidence of completion. |
| 1D HET | Present at all N. κ is **unchanged** by the geometry correction — the 1D matrix is the dimensionless TST operator, and `L` scales only the source amplitude α. Whether the solution itself changed is resolved by the test in Phase 0.4 rather than by an unnecessary rerun. |
| Suspect rows | 2D `VQLS N=32 max_rel_err = 513.94 %`, `306.42 %`; `HHL N=16 = 47.32 %`; 3D `wall_time_exceeded` at N=16 for VQLS across four cases; 1D thirteen `HHL solver_error` rows at N=32/64. |

`results/2Dhpc_run/` is live-changing while a job runs. Local `.npz` sets are an
incomplete synchronisation — every file shares one mtime. **`results_full.json` is
authoritative; a directory listing is not.**

---

## 3. Plan

### Phase 0 — Stop, preserve, and scope

1. Publish this document.
2. `qdel` the three running jobs: 4th-order HHL fails on every row, its QSVT solves
   the wrong operator, and the 2D job executes uncommitted code.
3. Back up `results/` to a timestamped copy before anything can overwrite it.
4. **1D geometry-impact test — COMPLETE.** Each `het_1d_*` case was built at N=16
   under the current constants, `core.het_geometry` was patched back to the
   pre-`861ff46` values (`L_Z=0.025`, `R_IN=0.035`), `core.cases` reloaded, and the
   cases rebuilt and compared element-wise.

   Reproducible via `scripts/check_geometry_impact.py --dim {1,2,3}`. The comparison
   covers `A`, the strip operator `row_matrix()`, the spacings `h`, `b`, `f` and
   `exact`, with a 1e-12 relative threshold separating genuine change from
   floating-point round-off (`~same`).

   **Result — 7 of 15 HET cases are provably unaffected:**

   | Case | dim | row | h | f | exact | Verdict |
   |---|---|---|---|---|---|---|
   | `het_1d_3a_linear` | 1 | — | same | same | same | keep |
   | **`het_1d_3b_gaussian_Vd300`** | 1 | — | same | **3.35e-01** | — | **rerun** |
   | `het_1d_3c_neumann` | 1 | — | same | same | same | keep |
   | `het_1d_bg1998_fig5_profile` | 1 | — | same | same | same | keep |
   | `het_1d_{gaussian_hom, gaussian_Vd300_scaled, linear_scaled, step_scaled}` | 1 | — | same | same | same/— | keep |
   | `het_2d_boeuf_garrigues` | 2 | same | same | same | — | keep |
   | **`het_2d_mms_spt100`** | 2 | **9.34e-01** | 3.75e-01 | 1.17e+00 | ~same | **rerun** |
   | **`het_2d_sin_meeting_report`** | 2 | **9.34e-01** | 3.75e-01 | 9.34e-01 | ~same | **rerun** |
   | **`het_3d_discharge_spt100`** | 3 | **9.30e-01** | 6.25e-02 | ~same | — | **rerun** |
   | **`het_3d_mms_spt100`** | 3 | **9.30e-01** | 6.25e-02 | 9.13e-01 | ~same | **rerun** |
   | **`het_3d_rotating_spoke`** | 3 | **9.30e-01** | 6.25e-02 | 9.15e-01 | ~same | **rerun** |
   | `het_3d_slab_m4` | 3 | 9.30e-01 | 6.25e-02 | 6.78e-01 | ~same | rerun, but not wired into any runner section |

   Three findings worth carrying forward:

   - **1D `A` is unchanged everywhere**, confirming κ is geometry-independent in 1D:
     the operator is the dimensionless TST matrix. Only `het_1d_3b_gaussian_Vd300`
     moves, because its Gaussian is sited against the physical `L_Z`, whereas the
     `*_scaled` non-dimensional family normalises `L` out. **1D rerun scope is that
     one case** (5 resolutions × 4 solvers = 20 rows), retiring
     `hpc/jobs/submit_hpc_1D_geometry_fix.sh`, whose premise was a full sweep.
   - **`het_3d_discharge_spt100` would have been missed** by a source-only check: its
     `f` is round-off-identical (its Gaussian is placed in normalised coordinates)
     but its strip operator moved 0.93. Comparing `row_matrix()` is what catches it,
     and is also what invalidates any QSVT phases cached against the old κ.
   - `exact` is `~same` for every MMS case: the manufactured solution is defined in
     normalised coordinates, so only the operator and the source move.
5. Commit the tree. `git_dirty: true` must never appear in a run record again.

### Phase 1 — Gap manifest

New `scripts/gap_analysis.py`, reusing `benchmark/results_io.py::SweepArchive` and
`missing()`. For each sweep directory, classify every (case, solver, N) triple:

- **good** — converged, `stop_reason="tol_met"`, error within the expected band for
  the case, produced *after* the relevant cap and geometry fixes. Recorded in a
  keep-list together with its `wall_time_s`, so the cost of any accidental
  recomputation is explicit.
- **missing** — no row present in `results_full.json`.
- **degraded** — `converged=False`, or `stop_reason ∈ {wall_time_exceeded,
  stagnated, max_iter}`, or `notes` carries an exception, or the error is
  implausible for the case, or the row predates the relevant fix.

Emits `rerun_manifest.json` and a human-readable table. The runners gain
`--manifest <file>` so that a job executes exactly those triples. This is the
mechanism that prevents both recomputation of sound results and the `--append`
clobbering encountered previously.

**Implemented as `scripts/gap_analysis.py`.** Two policy corrections were forced by
the first run against real data, both of which would otherwise have destroyed sound
expensive work:

- **Stagnation is not failure.** The outer schemes detect stagnation precisely so a
  quantum solver at its inner-solver noise floor stops rather than burning futile
  strip solves. Treating `stagnated`/`not_converged` as grounds to recompute marked
  22 sound 3D rows for rerun, including 6.11 h and 5.53 h HHL solves. They are now
  recorded as *flags*, escalated only by `--strict`. 3D outstanding fell 64 → 42 and
  **24.88 h of work is preserved**.
- **Error magnitude is judged against Thomas, not the exact solution** — and never
  for Thomas itself. At N=4 the under-resolved Gaussian and high-wavenumber cases
  carry 40–50 % discretisation error in *every* solver; that is truncation error, not
  a defect. In 1D at κ≈1.7e3 a large HHL error is the benchmark's finding, so it too
  is a flag rather than a reason. 1D outstanding fell 63 → 33.

Cross-validation: the 1D manifest independently reports `stale_geometry 20`, exactly
`HET_1D_3b_gaussian_Vd300` × 5 resolutions × 4 solvers — matching the Phase 0.4
geometry test derived by a completely different route.

- **A third policy correction, 2026-08-10: a per-solve timeout is a measurement.**
  `_run_hhl` imposes a hard 3600 s budget and, on expiry, returned `u=None` — which
  `_record` wrote as the note `"solver_error"`, the same note an exception produces.
  Thirteen 1D rows carry it, every one with `wall_time_s = 3600.2`: at N=32/64 the 1D
  operator reaches κ ≈ 1.7e3 and the statevector simulation genuinely does not finish
  inside an hour. **That is the result, not a defect.** Classified as errors they were
  scheduled for recomputation, which would have spent 13 h reproducing thirteen
  identical timeouts. The runner now records `hhl_timeout` against `hhl_error`, and
  the classifier treats the former as a flag — including the `no_error_metric` that
  necessarily follows from it, which would otherwise have reinstated the same
  recomputation by the back door. **1D outstanding fell 33 → 22.** The inference used
  for rows predating the marker is guarded to `dim == 1`: in 2D/3D `wall_time_s` is a
  whole outer iteration over N or N² strip solves and routinely exceeds an hour in a
  sound run, so applying it there would suppress real reasons. Both halves are pinned
  by `tests/test_gap_analysis.py`.

  Two 1D HHL failures survive as genuine: `HET_1D_3c_gaussian_NeumannDirichlet` at
  N=8 and N=32, the latter having died after 743 s — well inside the budget.

**Orphan-archive guard.** The tool also compares archives on disk against summary
rows and refuses to be trusted when they disagree, splitting the two causes:

| Sweep | Superseded residue | Unexplained | Manifest usable? |
|---|---|---|---|
| `1Dhpc_run` | 0 | 0 | yes |
| `3Dhpc_run` | 36 (stripped HET cases) | **2** | yes, with the caveat below |
| `2Dhpc_run` | 33 | **47** | **no — do not submit from it** |

The 2D summary currently holds **2 rows** against 80 archives: the still-running job
truncates `results_full.json` on start. A manifest built from it would schedule a
full recomputation of work that already exists.

The two unexplained 3D orphans —
`solution3d_3D_Laplace_BCdriven_cube_HHL_N16.npz` and
`solution3d_3D_Poisson_TripleSin_cube_HHL_N16.npz` — are **completed fields whose
rows were lost** when a job died mid-work-unit. Both are ≈ 6 h HHL solves.

**Recovered 2026-08-10** by `scripts/recover_orphan_rows.py --dim 3`, with no quantum
simulation. The recovered metrics sit exactly where the neighbouring rows predict:
`linf_err` 0.190 % and 0.260 % against QSVT's 0.201 % and 0.285 % at the same N,
i.e. both at the mesh's discretisation floor, with `n_outer` of 10 and 9 matching the
N=8 HHL pattern. The 3D summary now holds 48 rows and reports no unexplained orphans.

The script is deliberately conservative about what it claims. It reuses the runner's
own `_record`, so the accuracy definitions match the instrumented rows rather than
being reimplemented; it recomputes the residual against a freshly assembled operator
instead of inheriting a number from the process that died; it derives `converged`
from that residual against the sweep tolerance rather than asserting a verdict; it
takes the reference from the reassembled case, not from the archive, so a superseded
definition cannot silently become the truth a solution is scored against; and it
refuses to touch the 36 geometry-superseded archives, from which a plausible record
of a solve to the *wrong problem* could otherwise be manufactured.

### Phase 2 — Correctness: present the true operator to the quantum solvers

- `solvers/quantum/block_encoding.py` — add `build_dense_block_encoding(A)`:
  α = ‖A‖₂, M = A/α, reusing `_sznagy_dilation(M)`.
- `qsvt_1d.py`, `hhl_1d.py` — retain the TST fast path, preserving bit-for-bit
  reproducibility of the published 2nd-order figures, but **raise** on a nonzero
  ±2 band rather than truncating silently.
- `qsvt_1d_4th.py` — cease delegating; use the dense encoding.
- `solvers/outer/inner.py` — register `hhl_4th` and `qsvt_4th` alongside the
  existing factories, following the registry's validated-options idiom.
- `hpc/runners/run_1d.py:498,643` — dispatch on `--order`.

### Phase 3 — Environment and deployment

- Add `.gitmodules` pointing `quantum_linear_solvers` at the **fork**, making a
  clone reproducible.
- `hpc/setup_hpc_env.sh` — install the fork rather than upstream; add the missing
  `pennylane`; reconcile the qiskit pins against `requirements.txt`.
- New `hpc/jobs/_preflight.sh`, sourced by every submission script: refuse a dirty
  tree, import-check `PentadiagonalToeplitz`, `pennylane` and `qiskit_algorithms`,
  self-heal by force-reinstalling the fork, and abort **before** consuming walltime.

### Phase 4 — Fold order = 4 into the `LineProblem` protocol

#### 4a — The 1-D boundary closure: **fixed 2026-08-10**

Applied to `problems/poisson_1d_4th.py` exactly as derived below: `b[0] -= 14α`
in place of `18α`, plus `b[0] += h²·f(0)`, symmetrically at the far end. Measured
orders, before and after, by dense direct solve against the manufactured
solutions:

| Solution | Before | After |
|---|---|---|
| u = sin(πx), α = β = 0 | 3.90 → 4.00 | 3.90 → 4.00 *(unchanged, as intended)* |
| u = x(1−x)(2−x), α = β = 0 | 2.00 | **machine-exact** |
| u = eˣ, α = 1, β = e | **−0.01, no convergence** | **3.95** |

Through the case registry, at the resolutions the sweep runs:

| Case | Order |
|---|---|
| `poisson_1d_fS_hom` | 4.00 *(unchanged)* |
| `poisson_1d_fL_hom` | machine-exact (cubic solution) |
| `poisson_1d_fS_nonhom` | **4.00** (was no convergence) |
| `het_1d_3a_linear` | machine-exact (cubic solution) |
| `poisson_1d_fH_hom` | 1.98 — correct: the source is discontinuous, so the
  O(h⁴) truncation does not apply. This is the intended stress case. |

**κ is unchanged**, confirmed to four decimals against the values Phase 6
recorded — 11.9477 / 42.1378 / 154.5126 / 586.8093 at N = 4/8/16/32 — so the
phase-angle cache and `hpc/jobs/submit_precompute_4th.sh` survive the fix, as
predicted.

**Boundary source data.** f(0) and f(1) are required data for the closure, not a
refinement. They are threaded from the registry through a new optional
`BuiltCase.f_boundary`, populated by `_dirichlet_1d` and `_build_3b` via the new
`_f_boundary_1d` helper, and forwarded by a new `run_1d._to_4th_order` — which
also replaces the three near-identical inline conversion blocks the driver
previously carried. Where a case leaves it unset the problem class falls back to
cubic extrapolation from the four nearest interior samples: O(h⁴) accurate, hence
order-preserving, and verified to be so. The fallback is *not* sufficient for
`het_1d_3b_gaussian_Vd300`, whose source is a Gaussian of magnitude ~10⁹ sited at
0.6 L with σ = 5 mm; its true f(1) ≈ −1.08×10⁷ is not recoverable by
extrapolation at N = 4–16, which is why the registry route exists.

**Regression cover:** new `tests/test_poisson_1d_4th.py`, 21 tests. Verified
load-bearing: against the defective closure 8 of them fail, and the only order-of-
convergence test that still passes is the sinusoid — the historical blind spot.
Suite: 680 passed, 5 skipped.

#### 4b — 2-D/3-D: still outstanding

> **Retained as the derivation of record.** The defect described here has been
> **fixed in 1-D** (see 4a above) but **not in 2-D/3-D**: `multigrid_4th.py` carries
> its own separate closure, `build_strip_matrix_4th` folds the ghost into
> `A[0,1] += −1` rather than `A[0,0] += 1` — an *even* reflection, wrong for
> Dirichlet data — and `_build_rhs_strip` still writes the `18α` form. Measured
> convergence orders as originally found, direct dense solves against manufactured
> solutions:
>
> | Scheme | u = sin(πx), homogeneous | u = x(1−x), homogeneous | u = eˣ, u(0)=1, u(1)=e |
> |---|---|---|---|
> | `problems/poisson_1d_4th.py` | **3.95** ✓ | 1.98 | **−0.01 (no convergence)** |
> | `solvers/outer/multigrid_4th.py` (2-D) | 0.88 | — | — |
>
> The 1-D class appeared sound only because it had been exercised on
> `poisson_1d_fS_hom`, whose solution −sin(πx)/π² is *odd about both boundaries* —
> precisely the case where the closure is exact. Any other solution loses two
> orders; any non-zero Dirichlet datum loses convergence altogether. That covers
> `1D_Poisson_fL_hom`, `fH_hom`, `1D_Poisson_fS_nonhom` (α=1, β=2) and **every HET
> case** (V_d = 300 V at the anode).
>
> **Diagnosis.** The ghost node one step outside the boundary is taken as
> `u₋₁ = 2α − u₁`, an odd reflection about the boundary. Two errors follow.
>
> 1. Substituting it into the row-0 stencil gives `16α` from the boundary node and
>    `−2α` from the ghost, totalling **14α**. The implementation sums them with the
>    same sign and uses **18α**. The residual `A·u_exact − b` is then O(1) at rows 0
>    and N−1 and at machine level everywhere else — 4.00 for u = eˣ with α = 1, i.e.
>    exactly 4α, and 10.88 = 4β at the far boundary.
> 2. The reflection itself is only O(h²) accurate:
>    `u(−h) − [2u(0) − u(h)] = h²u″(0) + O(h⁴)`. Divided by the stencil's 12h²
>    prefactor that is an O(1) consistency error at one row, capping the scheme at
>    order 2. The PDE supplies the missing term for free, since `u″(0) = f(0)`.
>
> **Fix, verified.** `b[0] -= 14α` (not 18α) and `b[0] += h²·f(0)`, symmetrically at
> the far end. Orders then measured **3.95 / machine-exact / 3.93** on the three
> solutions above. Both corrections are **right-hand-side only**: `A` is untouched,
> so its symmetry — required by `build_dense_block_encoding` and by
> `PentadiagonalToeplitz` — is preserved, and **κ, the cache keys and the phase
> precompute all remain valid**.
>
> **Consequence for the interface** *(discharged in 1-D by `BuiltCase.f_boundary`;
> the 2-D/3-D half stands)*. `f` at the two boundary nodes becomes required
> data. `PoissonProblem1D4th` previously received interior values only, so the
> correction has to be threaded from the source functions in `core/cases.py`. In
> 2-D/3-D the same closure governs the strip direction, so the same data is needed
> per strip.
>
> **Decision taken (user, 2026-08-10): true 4th order in every direction.** The
> mixed design — 4th order along the strip, 2nd order transverse — is capped at
> order 2 by construction, measured 1.95 even after the strip operator is corrected,
> because `strip_sweep` hardcodes the transverse coupling as `1/h²` at j±1 only.
> Reaching order 4 therefore also requires the transverse stencil to carry j±2,
> which means extending `strip_sweep`. Two protocol additions, both optional so that
> the 2nd-order path stays bit-identical:
>
> - `transverse_terms(axis, index, n)` — the (offset, coefficient) pairs to gather,
>   defaulting to `[(−1, 1/h²), (+1, 1/h²)]`.
> - `row_matrix_for(idx)` — defaulting to `row_matrix()`. Needed because the
>   transverse operator's diagonal differs on the two boundary-adjacent strips, so
>   there are **two** distinct strip matrices rather than one. That is affordable:
>   two block encodings and two phase sets, not N.
>
> Verification gates before any 4th-order submission: order ≈ 4 in 1-D, 2-D and 3-D
> on at least one solution that is *not* odd about the boundaries and one with
> non-zero Dirichlet data; and the 2nd-order numbers unchanged — SOR 33/66/130, FMG
> 3 cycles, legacy Jacobi 26/73.

- New `problems/poisson_line_2d_4th.py` and `problems/poisson_line_3d_4th.py`,
  mirroring `problems/poisson_line_2d.py`, providing `shape`, `dx`, `dy`,
  `row_matrix()`, `rhs()`, `apply()` and `coarsen()`. The pentadiagonal components
  already exist as `build_strip_matrix_4th`, `_build_rhs_strip` and
  `compute_residual_2d_4th` (`multigrid_4th.py:11,27,58`). Coarsening is grid-based
  and therefore stencil-agnostic.
- **Remove** `solvers/outer/multigrid_4th.py` (531 lines),
  `_run_4th_order_solver_2d` (`run_2d.py:504-594`), `_run_4th_order_solver_3d`
  (`run_3d.py:591-681`), and the `from scripts.debug_2d_4th import …` layering
  violation.
- The runners reduce to selecting the problem object, then the existing single
  `solve(...)` call. The wall-clock cap, stagnation detection, `WorkLog`,
  `level_kappas`, `inner_*` diagnostics and the N ≤ 4 SOR fallback are then all
  inherited — resolving R3 and the greater part of R5 without further work.

### Phase 5 — Reporting, diagnostics and provenance

**Partly done.** The items Wave 1 could not safely run without are in:

- The per-strip-solve wall-clock deadline (R3), enforced inside `strip_sweep` in
  `solvers/outer/core.py` and honoured by both the stationary and multigrid schemes.
  Measured overshoot 1.02–1.04× the budget, against ~1× a whole sweep before.
- `run.log` opened in **append** mode in all three drivers, with a `SESSION START`
  banner per invocation carrying the phase tag and argv, and the 4th-order log
  redirect now inheriting the formatter instead of silently dropping the timestamp
  and PID from every line.
- `--append` supersession on `(case, solver, N)` in all three drivers.
- Incremental summary writes in the 1D driver, after every completed work unit
  rather than only at the end, matching 2D/3D.
- `order`, `sections`, `solvers`, `cases_filter` and `phase_tag` recorded in the 1D
  `run_metadata.json`, plus a per-step `run_metadata_<tag>.json` so successive steps
  of one job no longer overwrite each other's provenance.
- `all_solutions.npz` **retained but merged** rather than overwritten. Retiring it
  was the original plan; under a scope-restricted run the plain overwrite would have
  replaced a complete archive with a handful of entries, so merging was the smaller
  and safer change. Retirement stays available once `results_io` owns the writes.
- Spawn-safe worker initialisation (`_init_worker`) in all three drivers, so
  `RESULTS_DIR` and the QSVT degree caps resolved in `main` actually reach the
  workers. Without it an `--order 4` sweep wrote its summary to `*_4th` from the
  parent whilst every archive went to the 2nd-order directory from the workers.

Still outstanding, all of it Wave 2: routing writes through
`benchmark/results_io.py::save_solution` / `save_summary`, the new `RunResult`
columns below, `RUN HEALTH SUMMARY`, and `run_health.json`.

New and split columns on `RunResult`, `RunResult2D` and `RunResult3D`:

| Field | Purpose |
|---|---|
| `order` | 2 or 4; currently absent from every result dataclass |
| `scheme_requested`, `scheme_effective` | Replaces the conflated `"line-sor-4th (fallback)"` string, making a fallback queryable rather than discoverable by substring match |
| `capped_by` | `null \| wall_time \| max_cycles \| max_iter \| stagnation` — answers directly which solvers were capped |
| `wall_budget_s`, `wall_used_frac` | How close a sound run came to its cap |
| `qsvt_degree_capped`, `qsvt_degree_cap` | Whether `max_degree` bound the polynomial |
| `phases_from_cache` | Cache hit, as against an expensive inline phase computation |
| `inner_failures`, `inner_fallback_frac` | Proportion of strip solves that fell back to the classical solver |
| `degraded`, `degraded_reasons[]` | A single boolean to filter on, with reasons enumerated |
| `git_commit`, `git_dirty` | Per row rather than per run; R4 makes this necessary |

End-of-run additions:

- A `RUN HEALTH SUMMARY` block in `run.log`: counts by `stop_reason` and
  `capped_by`, followed by one explicit line per degraded row (case, solver, N,
  reason).
- `run_health.json` — the same content, machine-readable, for the plotting layer.
- Open `run.log` in **append** mode with a phase tag, correcting the `mode="w"`
  truncation that has been destroying run history.
- Retire `all_solutions.npz`; pass `RESULTS_DIR` explicitly into worker processes
  rather than relying on a mutated module global.
- Tighten the deadline check from per-sweep to per-strip-solve within `strip_sweep`
  (`solvers/outer/core.py`), and expose an explicit `--max-wall-s` flag rather than
  only the generic `-S` mechanism.

### Phase 6 — QSVT phase precompute for 4th order

Extend `hpc/runners/precompute_phases.py` with `--order` and `--dim 3`, deriving κ
from the problem classes and **never from a table** — a fourth-decimal drift in κ is
a silent cache miss that relocates the expensive computation into the sweep.

Measured pentadiagonal condition numbers:

| Case | κ(N=4) | κ(N=8) | κ(N=16) |
|---|---|---|---|
| 1D pentadiagonal | 11.95 | 42.14 | 154.5 |
| 2D mixed-order strip | 2.80 | 3.36 | 3.58 |
| 3D mixed-order strip | 1.98 | 2.22 | 2.30 |

κ₄ᵗʰ/κ₂ⁿᵈ → 4/3. The "2.5×" figure in `qsvt_1d_4th.py:12-16` is the *spectral-norm*
ratio 30/12, not a condition-number ratio.

- **On the cluster:** 1D pentadiagonal at N = 4, 8, 16, staged smallest-first, with
  `max_degree=5000` as the existing 1D cache uses. **Done** —
  `hpc/jobs/submit_precompute_4th.sh`, staged via `N_VALUES`. Measured κ values,
  which match the table above and confirm the 4/3 ratio: 11.9477 / 42.1378 /
  154.5126 / 586.8093 at N = 4 / 8 / 16 / 32, against 9.4721 / 32.1634 / 116.4612 /
  440.6886 at order 2.

  Not a laptop task, contrary to the original estimate: N=4 at the *cheapest*
  epsilon did not complete in 35 minutes on one core locally, which is why this is a
  71 h batch job staged smallest-first.

  Because the Phase 4 boundary fix is right-hand-side only, `A` and therefore κ are
  unchanged by it — **this precompute stays valid and can be submitted now**, ahead
  of Phase 4.
- **2D and 3D are refused,** not merely deferred: `build_targets` raises on
  `--dim 2 --order 4`. The strip operator is unsettled (see Phase 4), so any entry
  written now would be keyed to an operator about to change, and a stale key is a
  silent miss that relocates the expensive computation into the sweep — the exact
  failure the cache exists to prevent.
- Optionally purge the four orphaned pre-geometry-fix entries (`k1p9228_*`,
  `k2p1581_*`).

### Phase 7 — Consolidate `hpc/jobs/` and submit

**Done for Wave 1.** Six superseded one-off scripts were **deleted** rather than
archived — `submit_hpc_2D_gapfill.sh`, `submit_hpc_2D_gapfill_v2.sh`,
`submit_hpc_2D_complete.sh`, `submit_hpc_3D_resume.sh`,
`submit_hpc_3D_geometry_fix.sh`, `submit_hpc_1D_geometry_fix.sh` — since git history
preserves them and a script sitting in `hpc/jobs/` is a script somebody can `qsub`.
Each is superseded by the corresponding `submit_{1,2,3}d_wave1.sh`, which takes its
scope from the manifest rather than from a hand-written gap map.

The three `*_4th.sh` scripts are **retained**: they remain the only entry points for
`--order 4`, and Phases 2/4/6 will rewrite that path before they are usable. They
carry a "do not submit" note in `hpc/README.md` until then.

Still outstanding for Wave 2: collapse the remaining scripts to five parameterised
ones — `submit_1d.sh`, `submit_2d.sh`, `submit_3d.sh`, `submit_precompute.sh`,
`submit_gpu.sh` — driven by `ORDER`, `MANIFEST`, `N_VALUES`, `SECTIONS` and
`SOLVERS`. The `--manifest` flag itself is not built; the wave-1 scripts transcribe
the manifest's scope into `--n-values`/`--sections`/`--solvers`/`--cases` by hand,
with the derivation recorded in each header.

| Wave | Contents | Gated on |
|---|---|---|
| **1** | 2nd order, driven by `rerun_manifest.json`: the 3D HET gap (≈ 36 rows), the 2D HET gaps, and only those 1D cases Phase 0.4 shows to have changed | Phases 0–3 |
| **2** | 4th order, 1D/2D/3D to N = 16, all solvers; 1D phase precompute first | Phases 4–6 |

---

## 4. Verification

**Before Wave 1**

- `pytest -m "not quantum"` (201 tests, ≈ 7 s), then the full suite (259, ≈ 26 s).
- `bash hpc/jobs/_preflight.sh` on the login node — must pass cleanly.
- `python hpc/runners/run_2d.py --n-values 4 --sections 1 --solvers hhl --max-workers 1`
  as a live smoke test.
- Inspect `rerun_manifest.json` against the keep-list by eye. This is the final
  guard against recomputing a 38 h row.
- `--estimate` per job, to set walltime from a measured strip-solve profile.

**Before Wave 2**

- **Order 2 unchanged:** SOR 33/66/130, FMG 3 cycles, legacy Jacobi 26/73 — the
  values `tests/test_outer.py` and the published figures depend upon.
- **Order 4 converges at order ≈ 4** on the manufactured solutions, with order 2
  still ≈ 2. This is what demonstrates the new `PoissonLine*4th` classes are
  correct rather than merely executing.
- **The truncation defect is caught:** assert that `qsvt_1d.py` and `hhl_1d.py`
  now raise on pentadiagonal input rather than returning a plausible wrong answer.
- Extend `tests/test_outer.py:718` (`test_wall_time_budget_is_honoured`) to a
  4th-order problem, asserting the overshoot is bounded by one strip solve.
- Figure regression requires a control run first: hash the PNGs twice with no code
  change before trusting any difference.
