# HPC Operational Issues

Observed problems running jobs on Imperial CX3 that are not defects in the
numerics. Each entry records the symptom exactly as seen, what has been ruled
out, and the diagnostic that will settle it. Kept separate from
`HPC_REPAIR_PLAN.md`, which concerns the correctness of the benchmark itself.

---

## OI-1 — `run.log` is listed by `ls` but not readable by `tail`

**Raised:** 2026-08-11 · **Status:** open, cause not yet identified

### Symptom

On the login node, with a 2-D or 3-D job running:

```
$ ls results/2Dhpc_run/
... run.log ...                     # the file is listed

$ tail -f results/2Dhpc_run/run.log
tail: cannot open 'results/2Dhpc_run/run.log' for reading: No such file or directory
```

A second, possibly related symptom: **the 1-D job completed, but `ls` on its
results directory shows only the files that existed before the run.** Several
expected outputs are absent.

### What has been ruled out

- **CRLF line endings in the job scripts.** Checked at byte level on
  2026-08-11: `hpc/jobs/*.sh` are LF in both `HEAD` and the working tree, and
  `.gitattributes` (`*.sh text eol=lf`) is confirmed in effect via
  `git check-attr`. A CRLF script would corrupt `REPO_ROOT` and produce a
  filename with a trailing carriage return — which would present *exactly* this
  way, since `ls` renders the CR by returning the cursor and the name a human
  types then fails to match. It is the most attractive explanation and it is
  not the cause here. **Beware of `grep -c $'\r'` as a test**: under Git Bash
  the pattern can expand to empty and match every line, giving a false positive
  on every file.
- **A wrong `RESULTS_DIR` reaching the workers.** `RESULTS_DIR` is hardcoded in
  the runners (`run_2d.py:88`), not passed from the shell, and the spawn-safe
  `_init_worker` added in Phase 5 propagates it to worker processes.

### Leading hypotheses, in order

1. **The file name carries a trailing or non-printing character** from some
   other source. Same mechanism as the CRLF theory, different origin. Settled
   instantly by `ls -b`, which escapes non-printing characters.
2. **Filesystem visibility.** CX3 home and RDS are network filesystems. A file
   created and held open by a process on a compute node may be visible in the
   directory listing from the login node while its data is not yet flushed, and
   metadata caching can make `ls` and `open()` disagree. This would also explain
   the 1-D case: the outputs exist but are not yet visible from the login node.
3. **The job is not writing where it is being looked for.** The runners use a
   *relative* `Path("results")/"2Dhpc_run"`, resolved against the process
   working directory. The job scripts `cd "${REPO_ROOT}"` first, so this is
   correct only if `PBS_O_WORKDIR` resolved to the intended clone. If more than
   one clone exists on the cluster, the job may be writing into a different one.
4. **The 1-D outputs were moved, not lost.** Every job script ends with
   `cp -r "${RESULTS_SUBDIR}"/* "${HOME}/qpde-results/<sweep>_<timestamp>/"`.
   If the run completed, the results are in that timestamped directory whether
   or not they are visible in `results/`.

### Diagnostic sequence

Run on the login node, in the clone the job was submitted from. Takes seconds
and distinguishes all four hypotheses:

```bash
# (1) Hidden characters in the name — the first thing to check.
ls -b results/2Dhpc_run/ | cat -A | head

# (2) Does the kernel agree the file exists?
stat results/2Dhpc_run/run.log

# (3) Where did the job actually run, and where is it writing?
qstat -f $PBS_JOBID | grep -iE 'workdir|Output_Path|Error_Path'
grep -m3 'Work dir' results/2Dhpc_run_pbs.log       # PBS stdout, not run.log

# (4) Is the sweep progressing at all? This is the question that matters.
ls -la --time-style=full-iso results/2Dhpc_run/*.npz | tail -5

# (5) Did the 1-D results simply get copied to RDS?
ls -d ~/qpde-results/*/ | tail -5
ls ~/qpde-results/1Dhpc_run_*/ | head
```

### Why the `.npz` check is the one that matters

The runners write **per-solution `.npz` archives incrementally** and the summary
JSON/CSV only at the end. So:

- If the `.npz` mtimes are advancing, the job is doing real work and the
  inaccessible `run.log` is a logging/visibility annoyance, **not** a data
  problem. Let it run.
- If no `.npz` has appeared or the mtimes are stale, the job is not producing
  anything and should be killed.

A walltime kill loses the summary but not the per-solution data, and the
plotting layer reads the per-solution archives regardless — so a job that is
writing `.npz` files is worth letting finish even if it is later killed.

### Interim mitigation

The PBS stdout stream (`#PBS -o`) is a separate file from `run.log` and is
written by PBS itself rather than by the Python logging layer, so it is subject
to different buffering. Where `run.log` is unreadable, that file usually is not:

```bash
tail -f results/2Dhpc_run_pbs.log      # path per each script's #PBS -o
```

Note that `#PBS -o`/`-e` are resolved **at submission time relative to the
submission directory** and cannot be redirected from inside the job.
