#!/bin/bash
# ============================================================
#  submit_hpc_1D.sh
#  PBS Pro job submission script for Imperial College CX3 HPC.
#
#  Runs the full 1-D benchmark sweep, N = 4..64, all cases,
#  all solvers (Thomas, HHL, VQLS, QSVT).
#
#  Usage:
#    qsub hpc/jobs/submit_hpc_1D.sh
#
#    # Fast validation pass before committing the full walltime:
#    export MAX_N=16
#    qsub -v MAX_N hpc/jobs/submit_hpc_1D.sh
#
#    # Skip QSVT entirely:
#    export SKIP_QSVT=1
#    qsub -v SKIP_QSVT hpc/jobs/submit_hpc_1D.sh
#
#  Monitor:
#    qstat -u $USER
#    tail -f results/1Dhpc_run/run.log
#
#  Reference: https://icl-rcs-user-guide.readthedocs.io/
# ============================================================

# --- Resource requests ---
# NOTE ON ncpus vs --max-workers: the runner's MAX_WORKERS_DEFAULT is 4 and
# MUST NOT exceed the ncpus requested here. Aer simulations are already
# OpenMP-threaded internally, so running more worker processes than allocated
# cores oversubscribes the node and makes the sweep slower, not faster.
# If you raise one, raise the other.
#
# Walltime: the full N=4..64 sweep with QSVT is substantially longer than the
# previous N<=32 runs. 24h is a starting point; check the actual elapsed time
# reported at the end of run.log and adjust. If the job is killed on walltime,
# results already written to results/1Dhpc_run/ are preserved (each solution
# NPZ is written as it is produced), but results_full.json / results_summary.csv
# are written only at the END, so a killed job loses the summary table.
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=128gb

# --- Job metadata ---
#PBS -N quantum_pde_1Dfull_run
#PBS -o results/1Dhpc_run/pbs_stdout.log
#PBS -e results/1Dhpc_run/pbs_stderr.log

# --- Email notifications ---
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

# ============================================================
#  Environment setup
# ============================================================

echo "============================================================"
echo "  QUANTUM PDE SOLVER — HPC JOB START"
echo "  Job ID    : $PBS_JOBID"
echo "  Node      : $(hostname)"
echo "  Date/Time : $(date)"
echo "  Work dir  : $PBS_O_WORKDIR"
echo "  MAX_N     : ${MAX_N:-<not set: full sweep to N=64>}"
echo "  SKIP_QSVT : ${SKIP_QSVT:-0}"
echo "============================================================"

# ── Repository root resolution ───────────────────────────────
# PBS copies this script to a spool directory before executing it, so $0 and
# BASH_SOURCE do NOT point at the original file. PBS_O_WORKDIR -- the directory
# qsub was invoked from -- is the only reliable anchor. Ascending from it means
# both `qsub hpc/<script>` (from the repo root) and `cd hpc && qsub <script>`
# resolve correctly.
REPO_ROOT="${PBS_O_WORKDIR}"
while [ ! -f "${REPO_ROOT}/pyproject.toml" ] && [ "${REPO_ROOT}" != "/" ]; do
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
if [ ! -f "${REPO_ROOT}/pyproject.toml" ]; then
    echo "ERROR: no repository root (pyproject.toml) at or above ${PBS_O_WORKDIR}."
    echo "       Submit from inside a clone, e.g. qsub hpc/$(basename "$0")"
    exit 1
fi
cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}"; exit 1; }

# The #PBS -o/-e paths above are resolved by PBS at submission time, relative to
# the submission directory; no shell logic here can redirect them. Submitting
# from the repository root keeps the PBS logs alongside the results.
if [ "${PBS_O_WORKDIR}" != "${REPO_ROOT}" ]; then
    echo "NOTE: submitted from ${PBS_O_WORKDIR}, not the repository root"
    echo "      (${REPO_ROOT}). The PBS stdout/stderr logs are under the former."
fi

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

VENV_PATH="${HOME}/venvs/qpde"
if [ ! -d "${VENV_PATH}" ]; then
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    echo "       See hpc/setup_hpc_env.sh."
    exit 1
fi
source "${VENV_PATH}/bin/activate"
echo "Python: $(which python3) -- $(python3 --version)"

# pyqsp is in requirements.txt but is NOT in hpc/setup_hpc_env.sh's explicit
# install list. Guard against that gap here rather than failing on a missing
# import hours into a queued job.
python3 -c "import pyqsp" 2>/dev/null || {
    echo "pyqsp not found in venv; installing..."
    pip install pyqsp==0.2.0
}

# IMPORTANT: this must match RESULTS_DIR in hpc/runners/run_1d.py.
# The previous version created and archived results/hpc_run/ while the runner
# wrote to results/1Dhpc_run/, so the RDS copy at the end contained only the
# PBS logs and none of the actual benchmark output.
RESULTS_SUBDIR="results/1Dhpc_run"
mkdir -p "${RESULTS_SUBDIR}"

# Keep Aer's internal OpenMP threading within the allocated core count.
# Without this, each of the 4 worker processes may spawn as many OpenMP
# threads as there are physical cores on the node.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
echo "OMP_NUM_THREADS = ${OMP_NUM_THREADS}"

# ============================================================
#  Run the benchmark
# ============================================================

EXTRA_ARGS=""
if [ -n "${MAX_N}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --max-n ${MAX_N}"
    echo "INFO: sweep truncated at N=${MAX_N}."
fi
if [ "${SKIP_QSVT:-0}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --skip-qsvt"
    echo "INFO: QSVT disabled."
fi
# Worker count is pinned to the ncpus requested above.
EXTRA_ARGS="${EXTRA_ARGS} --max-workers 4"

echo "Starting benchmark at $(date)"
echo "Arguments: ${EXTRA_ARGS}"
echo "------------------------------------------------------------"

python3 hpc/runners/run_1d.py ${EXTRA_ARGS}
EXIT_CODE=$?

echo "------------------------------------------------------------"
echo "Benchmark finished at $(date) with exit code ${EXIT_CODE}"

# ============================================================
#  Copy results to permanent RDS storage
# ============================================================
RDS_RESULTS="${HOME}/qpde-results/1Dhpc_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r "${RESULTS_SUBDIR}"/* "${RDS_RESULTS}/" 2>/dev/null
echo "Results copied to: ${RDS_RESULTS}"

echo "============================================================"
echo "  JOB COMPLETE"
echo "  Exit code : ${EXIT_CODE}"
echo "  Date/Time : $(date)"
echo "============================================================"

exit ${EXIT_CODE}