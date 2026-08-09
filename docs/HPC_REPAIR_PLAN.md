# HPC Benchmark Pipeline Repair Plan

**Raised:** 2026-08-09 · **Scope:** `hpc/runners/`, `hpc/jobs/`, `solvers/quantum/`,
`solvers/outer/`, `problems/`, `benchmark/results_io.py`

Repair of the 2nd- and 4th-order HPC benchmark pipeline following the failed
integration of the 4th-order pentadiagonal discretisation, and completion of the
2nd-order results left incomplete by the SPT-100 geometry correction (`861ff46`).

---

## Status

| # | Phase | State |
|---|-------|-------|
| 0 | Stop, preserve, and scope | scoping done; `qdel` + re-sync outstanding (user) |
| 1 | Gap manifest | tool complete; 1D/3D manifests generated, 2D blocked on re-sync |
| 2 | Correctness: pentadiagonal operator | pending |
| 3 | Environment and deployment | complete |
| 4 | Fold order=4 into the `LineProblem` protocol | pending |
| 5 | Reporting, diagnostics and provenance | pending |
| 6 | QSVT phase precompute for 4th order | pending |
| 7 | Consolidate `hpc/jobs/` and submit | pending |

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
rows were lost** when a job died mid-work-unit. Both are ≈ 6 h HHL solves. Their
metrics can be recomputed from the archive with no quantum simulation at all, so they
should be recovered offline rather than resubmitted (≈ 12 h saved).

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

Route all writes through the already-declared
`benchmark/results_io.py::save_solution` and `save_summary`, which were written for
precisely this migration. Then make degradation impossible to overlook.

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

- **Locally:** 2D and 3D pentadiagonal phases (degrees ≈ 95–280) — minutes.
- **On the cluster:** 1D pentadiagonal at N = 4, 8, 16, staged smallest-first, with
  `max_degree=5000` as the existing 1D cache uses.
- Optionally purge the four orphaned pre-geometry-fix entries (`k1p9228_*`,
  `k2p1581_*`).

### Phase 7 — Consolidate `hpc/jobs/` and submit

Archive the ten superseded one-off scripts (`*_gapfill*`, `*_geometry_fix`,
`*_resume`, `*_complete`, `*_4th`) under `hpc/jobs/archive/`. Retain five
parameterised scripts — `submit_1d.sh`, `submit_2d.sh`, `submit_3d.sh`,
`submit_precompute.sh`, `submit_gpu.sh` — driven by `ORDER`, `MANIFEST`,
`N_VALUES`, `SECTIONS` and `SOLVERS`, each sourcing `_preflight.sh`.

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
