# HPC deployment

PBS Pro job scripts for running the sweeps at production scale. This directory
is deployment configuration, not code: the science lives in `scripts/` and the
packages above it, and runs on a laptop at small N.

Written for Imperial College London's CX3. Adapting to another cluster requires
the changes listed at the end.

## Submit from the repository root

```bash
qsub hpc/submit_hpc_1D.sh
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
qsub -v N_VALUES hpc/submit_precompute_2D.sh          # correct
qsub -v N_VALUES="4,8,16" hpc/submit_precompute_2D.sh # silently truncated
```

Scalar variables are unaffected: `qsub -v MAX_DEGREE=2000 hpc/...`.

## Workflow

Setup is run once; the sweeps may then be submitted in any order.

| Stage | Script | Resources | Purpose |
| ----- | ------ | --------- | ------- |
| Setup | `setup_hpc_env.sh` | login node, interactive | Builds separate CPU (`qpde`) and GPU (`qpde-gpu`) virtualenvs; two are required because `qiskit-aer` and `qiskit-aer-gpu` cannot coexist. Run with `bash`, not `qsub`. |
| Precompute | `submit_precompute_hpc.sh` | 1 cpu, 32 GB, 71 h | QSVT phase angles for the 1-D operator. Must be staged small-N-first: κ = O(N²) there, and large N is not guaranteed to finish within one submission. |
| Precompute | `submit_precompute_2D.sh` | 1 cpu, 4 GB, 30 min | QSVT phase angles for the 2-D strip operator. Needs no staging: κ → 3⁻ gives polynomial degrees of 30–85 at every N. |
| Sweep | `submit_hpc_1D.sh` | 4 cpus, 128 GB, 24 h | Full 1-D sweep, CPU. |
| Sweep | `submit_hpc_gpu.sh` | 8 cpus, 64 GB, 1×L40S, 24 h | Full 1-D sweep, GPU via cuStateVec. |
| Sweep | `submit_hpc_2D.sh` | 4 cpus, 64 GB, 48 h | Full 2-D sweep. |
| Sweep | `submit_hpc_3D.sh` | 4 cpus, 64 GB, 72 h | Full 3-D sweep. |
| Repair | `submit_hpc_2D_gapfill.sh` | 4 cpus, 64 GB, 72 h | Fills exactly the (case, N, solver) combinations missing after a killed 2-D sweep, repeating nothing already completed. Its header records the gap map it was built from. |

Post-processing requires no cluster job, reading only what the sweep wrote:
`python scripts/plot_hpc_2Dfull_results.py`.

## Partial results are safe

Every driver writes each solution to its own `.npz` as soon as it is produced;
only the summary `results_full.json` is written at the end. A walltime kill
therefore loses the summary but not the per-solution data, and the plotting
layer reads the per-solution archives directly. `submit_hpc_2D_gapfill.sh`
demonstrates the pattern for resuming without repeating completed work.

## Adapting to another cluster

The site-specific surface is confined to this directory:

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
