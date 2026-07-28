#!/bin/bash
# ============================================================
#  submit_hpc.sh
#  PBS Pro job submission script for Imperial College CX3 HPC.
#  Uses the throughput72 queue (1-8 CPUs, up to 72h walltime).
#
#  Usage:
#    qsub submit_hpc.sh
#    qsub -v INCLUDE_N64=1 submit_hpc.sh   # to add N=64
#    qsub -v SKIP_QSVT=1 submit_hpc.sh     # to skip QSVT
#
#  Monitor:
#    qstat -u $USER
#    tail -f results/hpc_run/run.log
#
#  Reference: https://icl-rcs-user-guide.readthedocs.io/
# ============================================================

# --- Resource requests ---
# N=4,8,16,32 with QSVT for N<=8: estimated ~2-4h on HPC.
# Increase walltime to 24:00:00 if including N=64.
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=4:mem=32gb

# --- Job metadata ---
#PBS -N quantum_pde_1Dfull_run
#PBS -o results/hpc_run/pbs_stdout.log
#PBS -e results/hpc_run/pbs_stderr.log

# --- Email notifications (replace with your Imperial email) ---
#PBS -M j.trobajo-flecha24@imperial.ac.uk
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
echo "============================================================"

# Change to the submission directory (where the script was qsubmitted from).
cd "${PBS_O_WORKDIR}" || { echo "ERROR: Cannot cd to PBS_O_WORKDIR"; exit 1; }

# Load the production module environment (required on CX3 Phase 2).
# On CX3 Phase 2, tools/prod is auto-loaded on login but must be
# explicitly loaded in batch jobs.
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

# Activate the project virtual environment.
# Adjust this path to wherever you created your venv on the RDS.
VENV_PATH="${HOME}/venvs/qpde"
if [ ! -d "${VENV_PATH}" ]; then
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    echo "       Create it first with: virtualenv ${VENV_PATH}"
    echo "       Then: source ${VENV_PATH}/bin/activate && pip install -r requirements.txt"
    exit 1
fi

source "${VENV_PATH}/bin/activate"
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"

# Create output directory (in case it does not exist yet).
mkdir -p results/hpc_run

# ============================================================
#  Run the benchmark
# ============================================================

# Parse optional environment variables passed via qsub -v.
EXTRA_ARGS=""
if [ "${INCLUDE_N64:-0}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --include-n64"
    echo "INFO: N=64 included in sweep."
fi
if [ "${SKIP_QSVT:-0}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --skip-qsvt"
    echo "INFO: QSVT disabled."
fi

echo "Starting benchmark at $(date)"
echo "Arguments: ${EXTRA_ARGS}"
echo "------------------------------------------------------------"

python3 scripts/run_hpc_1Dfull.py ${EXTRA_ARGS}
EXIT_CODE=$?

echo "------------------------------------------------------------"
echo "Benchmark finished at $(date) with exit code ${EXIT_CODE}"

# ============================================================
#  Copy results to permanent RDS storage
# ============================================================
# The RDS home directory is permanent; ephemeral is for scratch.
# Copy the results there for safekeeping.
RDS_RESULTS="${HOME}/qpde-results/hpc_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r results/hpc_run/* "${RDS_RESULTS}/"
echo "Results copied to: ${RDS_RESULTS}"

echo "============================================================"
echo "  JOB COMPLETE"
echo "  Exit code : ${EXIT_CODE}"
echo "  Date/Time : $(date)"
echo "============================================================"

exit ${EXIT_CODE}