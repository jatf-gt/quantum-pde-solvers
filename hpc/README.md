# HPC deployment

PBS Pro job scripts for running the sweeps at production scale. Everything here
is **deployment configuration, not code** — the science lives in `scripts/` and
the packages above it, and runs perfectly well on a laptop at small $N$.

**If you are not running on a cluster, you can ignore this directory entirely.**

These scripts were written for **Imperial College London's CX3**. They will not
work unmodified anywhere else; see *Adapting to another cluster* below, which is
a short and well-defined list of changes.

## Submit from the repository root

```bash
qsub hpc/submit_hpc_1D.sh
```

Not `cd hpc && qsub submit_hpc_1D.sh`. The job itself works either way — each
script locates the repository root by ascending from `$PBS_O_WORKDIR` until it
finds `pyproject.toml` — but the `#PBS -o` and `#PBS -e` log paths are resolved
by PBS *at submission time*, relative to wherever `qsub` was invoked. Submitting
from elsewhere scatters the PBS logs away from `results/`. The scripts detect
this and say so, but they cannot fix it: nothing inside a job can redirect its
own PBS log.

Note also that `$0` is useless for locating anything here — PBS copies the job
script to a spool directory before executing it, so it does not point at the
original file. `$PBS_O_WORKDIR` is the only reliable anchor.

## Passing list-valued variables

PBS splits `-v` arguments on commas, so a comma-separated value must be exported
in the shell and passed **by name only**:

```bash
export N_VALUES="4,8,16"
qsub -v N_VALUES hpc/submit_precompute_2D.sh      # correct
qsub -v N_VALUES="4,8,16" hpc/submit_precompute_hpc.sh   # WRONG: silently truncated
```

Scalar variables are fine either way: `qsub -v MAX_DEGREE=2000 hpc/...`.

## Workflow

Run once, then submit sweeps in any order.

| Stage | Script | Resources | Purpose |
| ----- | ------ | --------- | ------- |
| Setup | `setup_hpc_env.sh` | login node, interactive | Builds separate CPU (`qpde`) and GPU (`qpde-gpu`) virtualenvs. Two are needed because `qiskit-aer` and `qiskit-aer-gpu` cannot coexist in one environment. Run with `bash hpc/setup_hpc_env.sh`, not `qsub`. |
| Precompute | `submit_precompute_hpc.sh` | 1 cpu, 32 GB, 71 h | QSVT phase angles for the **1-D** operator. Must be staged small-$N$-first: $\kappa = O(N^2)$ there, and large $N$ is not guaranteed to finish in one submission. |
| Precompute | `submit_precompute_2D.sh` | 1 cpu, 4 GB, 30 min | QSVT phase angles for the **2-D** strip operator. Needs no staging: $\kappa \to 3^-$ gives polynomial degrees of 30–85 at every $N$. |
| Sweep | `submit_hpc_1D.sh` | 4 cpus, 128 GB, 24 h | Full 1-D sweep, CPU. |
| Sweep | `submit_hpc_gpu.sh` | 8 cpus, 64 GB, 1×L40S, 24 h | Full 1-D sweep, GPU via cuStateVec. |
| Sweep | `submit_hpc_2D.sh` | 4 cpus, 64 GB, 48 h | Full 2-D sweep. |
| Sweep | `submit_hpc_3D.sh` | 4 cpus, 64 GB, 72 h | Full 3-D sweep. |
| Repair | `submit_hpc_2D_gapfill.sh` | 4 cpus, 64 GB, 72 h | One-off: fills exactly the `(case, N, solver)` combinations missing after a killed 2-D sweep, repeating nothing already completed. Its header records the gap map it was built from. |

Post-processing needs no cluster job — it only reads what the sweep wrote:

```bash
python scripts/plot_hpc_2Dfull_results.py
```

## Partial results are safe

Every driver writes each solution to its own `.npz` as soon as it is produced.
Only the summary `results_full.json` is written at the end, so a walltime kill
loses the summary but **not** the per-solution data. The plotting layer reads
the per-solution archives directly, so figures can still be produced; and
`submit_hpc_2D_gapfill.sh` shows the pattern for resuming without repeating
completed work.

## Adapting to another cluster

The site-specific surface is small and confined to these files:

1. **Scheduler directives.** These are PBS Pro (`#PBS`). For Slurm, the
   `#PBS -l select=...:ncpus=...:mem=...` and `#PBS -l walltime=...` lines
   become `#SBATCH` equivalents, and `$PBS_O_WORKDIR` becomes
   `$SLURM_SUBMIT_DIR`. The repository-root resolution block works unchanged
   once that variable is renamed.

2. **Module names.** Every script loads:
   ```bash
   module load tools/prod
   module load Python/3.12.3-GCCcore-13.3.0
   ```
   Replace with whatever provides Python ≥ 3.11 on your system.

3. **Virtualenv location.** `VENV_PATH="${HOME}/venvs/qpde"` (and
   `qpde-gpu` for the GPU script). Change here and in `setup_hpc_env.sh`
   together.

4. **Notification address.** `#PBS -M <address>` in each script.

5. **GPU type.** `submit_hpc_gpu.sh` requests `gpu_type=L40S`. Any CUDA GPU with
   sufficient memory works; the constraint is that `qiskit-aer-gpu` must match
   the driver's CUDA version.

Nothing under `core/`, `problems/`, `solvers/`, `benchmark/` or `scripts/` is
site-specific. If a change is needed outside this directory to get a run going,
that is a portability bug worth reporting.

## Known inconsistency

`submit_hpc_2D.sh` sets `#PBS -M juan.trobajo-flecha25@imperial.ac.uk` whereas
every other script uses `j.trobajo-flecha24@imperial.ac.uk`. One of the two is
wrong and the corresponding job notifications are going astray. Left as-is
pending confirmation of which address is correct.
