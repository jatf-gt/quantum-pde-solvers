# HPC deployment

Running the sweeps at production scale on Imperial College London's CX3.
`hpc/jobs/` holds PBS Pro job-submission scripts — pure deployment
configuration, site-specific and not expected to port unchanged elsewhere (see
"Adapting to another cluster" below). `hpc/runners/` holds the actual driver
code the jobs execute: `run_{1,2,3}d.py` (the full sweeps), `precompute_phases.py`
(QSVT phase-angle precompute) and `plot_results.py` (post-processing,
`--dim {1,2,3}`). That code is ordinary Python with no PBS/Slurm dependency —
it is grouped here rather than left in `scripts/` because it is cluster-scale,
not laptop-scale, but it needs no changes to run on another cluster, only the
job scripts that invoke it do. The physics itself lives in `core/`, `problems/`
and `solvers/`; `scripts/` is the laptop-scale entry points that exercise the
same code at small N.

## Submit from the repository root

```bash
qsub hpc/jobs/submit_hpc_1D.sh
```

Not `cd hpc && qsub submit_hpc_1D.sh`. The job itself works either way — each
script locates the repository root by ascending from `$PBS_O_WORKDIR` until it
finds `pyproject.toml` — but the `#PBS -o` and `#PBS -e` log paths are resolved
by PBS at submission time, relative to the invoking directory, so submitting
from elsewhere scatters the PBS logs away from `results/`. The scripts detect
this and report it; they cannot correct it, as nothing inside a job may redirect
its own PBS log. Note also that `$0` cannot locate anything here: PBS copies the
job script to a spool directory before execution, so `$PBS_O_WORKDIR` is the
only reliable anchor.

## Passing list-valued variables

PBS splits `-v` arguments on commas, so a comma-separated value must be exported
in the shell and passed by name only:

```bash
export N_VALUES="4,8,16"
qsub -v N_VALUES hpc/jobs/submit_precompute_2D.sh          # correct
qsub -v N_VALUES="4,8,16" hpc/jobs/submit_precompute_2D.sh # silently truncated
```

Scalar variables are unaffected: `qsub -v MAX_DEGREE=2000 hpc/...`.

## Workflow

Setup is run once; the sweeps may then be submitted in any order.

| Stage | Script | Runs | Resources | Purpose |
| ----- | ------ | ---- | --------- | ------- |
| Setup | `setup_hpc_env.sh` | — | login node, interactive | Builds separate CPU (`qpde`) and GPU (`qpde-gpu`) virtualenvs; two are required because `qiskit-aer` and `qiskit-aer-gpu` cannot coexist. Run with `bash`, not `qsub`. |
| Precompute | `jobs/submit_precompute_hpc.sh` | `runners/precompute_phases.py --dim 1` | 1 cpu, 32 GB, 71 h | QSVT phase angles for the 1-D operator. Must be staged small-N-first: κ = O(N²) there, and large N is not guaranteed to finish within one submission. |
| Precompute | `jobs/submit_precompute_2D.sh` | `runners/precompute_phases.py --dim 2` | 1 cpu, 4 GB, 30 min | QSVT phase angles for the 2-D strip operator. Needs no staging: κ → 3⁻ gives polynomial degrees of 30–85 at every N. |
| Sweep | `jobs/submit_hpc_1D.sh` | `runners/run_1d.py` | 4 cpus, 128 GB, 24 h | Full 1-D sweep, CPU. |
| Sweep | `jobs/submit_hpc_gpu.sh` | `runners/run_1d.py` | 8 cpus, 64 GB, 1×L40S, 24 h | Full 1-D sweep, GPU via cuStateVec. |
| Sweep | `jobs/submit_hpc_2D.sh` | `runners/run_2d.py` | 4 cpus, 64 GB, 48 h | Full 2-D sweep. |
| Sweep | `jobs/submit_hpc_3D.sh` | `runners/run_3d.py` | 4 cpus, 64 GB, 72 h | Full 3-D sweep. |
| Precompute (4th order) | `jobs/submit_precompute_4th.sh` | `runners/precompute_phases.py --dim 1 --order 4` | 1 cpu, 32 GB, 71 h | QSVT phase angles for the 1-D **pentadiagonal** operator: κ = 11.95 / 42.14 / 154.5 / 586.8 at N = 4/8/16/32, distinct keys from the 2nd-order set. Staged smallest-N-first; N ≤ 16 must be computed **uncapped** and N ≥ 32 capped at degree 5000, because the cap forms part of the cache key. The job refuses any invocation whose degree tag differs from the one `run_1d.py` will request. |
| Sweep (4th order, 1-D) | `jobs/submit_hpc_1D_4th.sh` | `runners/run_1d.py --order 4` | 4 cpus, 128 GB, 24 h | Pentadiagonal 1-D sweep, N = 4, 8, 16. The operator is correct as of Phase 2 (`ace7f9b`) and Phase 4a (`8191325`). Requires the 4th-order phase cache first: the script refuses to run QSVT without it, since a miss degrades silently to a reduced-degree solve. Sub-case 3c is skipped automatically — `PoissonProblem1D4th` has no Neumann closure. |
| Sweep (4th order, 2-D/3-D) | `jobs/submit_hpc_{2D,3D}_4th.sh` | `runners/run_{2,3}d.py --order 4` | as their 2nd-order twins | **Both refuse to run**, exit 2. The 2-D/3-D closure in `solvers/outer/multigrid_4th.py` is untouched by the 1-D fix: an even reflection applied to Dirichlet data, with the same `18α` error, measured convergence order 0.88. See `docs/HPC_REPAIR_PLAN.md` Phase 4b, and each script's header for the four gates that must clear. |
| Repair | `jobs/submit_1d_wave1.sh` | `runners/run_1d.py --append --cases` | 4 cpus, 32 GB, 6 h | 22 outstanding 1-D rows: the one geometry-affected HET case at every N, plus two genuine HHL failures. |
| Repair | `jobs/submit_2d_wave1.sh` | `runners/run_2d.py --append` | 4 cpus, 64 GB, 60 h | 55 outstanding 2-D rows: the two geometry-affected HET cases at every N, the N=64 tier the original sweep never reached, and one partial section at N=32. |
| Repair | `jobs/submit_3d_wave1.sh` | `runners/run_3d.py --append` | 4 cpus, 64 GB, 48 h | 42 outstanding 3-D rows, dominated by the three geometry-affected HET cases, which have no rows at any resolution. |

The three `*_wave1.sh` scripts take their scope from
`results/manifests/rerun_{1,2,3}d.json`, generated by
`python scripts/gap_analysis.py --dim {1,2,3}`. Regenerate the manifest and read
it before submitting: it is the guard against recomputing a 38 h row. The
per-script headers record the scope actually transcribed from it, and why each
excluded row was excluded.

Post-processing requires no cluster job, reading only what the sweep wrote:
`python hpc/runners/plot_results.py --dim 2`.

## Partial results are safe

Every driver writes each solution to its own `.npz` as soon as it is produced,
and now rewrites the summary `results_full.json` after every completed work unit
rather than only at the end, so a walltime kill loses neither. The plotting layer
reads the per-solution archives directly in either case.

Resuming without repeating completed work needs `--append`, which merges the rows
already in `results_full.json` and supersedes them on `(case, solver, N)`. Any
scope-restricted run **must** pass it: without it the driver rewrites the summary
from its own rows alone and discards every row it was not asked to produce.
`jobs/submit_{1,2,3}d_wave1.sh` demonstrate the pattern.

## Adapting to another cluster

The site-specific surface is confined to `hpc/jobs/` and `setup_hpc_env.sh` —
`hpc/runners/` is ordinary Python, portable unchanged:

1. **Scheduler directives.** These are PBS Pro (`#PBS`). Under Slurm, the
   `#PBS -l select=...:ncpus=...:mem=...` and `#PBS -l walltime=...` lines become
   `#SBATCH` equivalents and `$PBS_O_WORKDIR` becomes `$SLURM_SUBMIT_DIR`; the
   repository-root resolution block works unchanged once that variable is renamed.
2. **Module names.** Every script loads `tools/prod` and
   `Python/3.12.3-GCCcore-13.3.0`. Substitute whatever provides Python ≥ 3.11.
3. **Virtualenv location.** `VENV_PATH="${HOME}/venvs/qpde"` (and `qpde-gpu` for
   the GPU script); change here and in `setup_hpc_env.sh` together.
4. **Notification address.** `#PBS -M <address>` in each script.
5. **GPU type.** `submit_hpc_gpu.sh` requests `gpu_type=L40S`. Any CUDA GPU with
   sufficient memory suffices; the binding constraint is that `qiskit-aer-gpu`
   must match the driver's CUDA version.
