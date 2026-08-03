#!/bin/bash
# ============================================================
#  submit_hpc_2D.sh
#  PBS Pro job submission script for Imperial College CX3 HPC.
#
#  Runs the full 2-D benchmark sweep across all five sections:
#    Section 1: Generic Poisson, sinusoidal source
#    Section 2: Two-Gaussian PlasmaNet benchmark
#    Section 3: Single-mode Fourier source
#    Section 4: HET MMS manufactured solution (SPT-100)
#    Section 5: HET sinusoidal source (meeting-report case)
#
#  Solvers: Thomas-2D, HHL-2D, VQLS-2D, QSVT-2D.
#  N range: 4, 8, 16, 32, 64, 128, 256 (default: all).
#
#  PREREQUISITE: run the 2D phase precompute before this job.
#    qsub submit_precompute_2D.sh
#  Wait for it to complete, then:
#    qsub submit_hpc_2D.sh
#
#  Usage:
#    # Full sweep (default):
#    qsub submit_hpc_2D.sh
#
#    # Fast local validation pass (N=4 only, serial):
#    export MAX_N=4
#    qsub -v MAX_N submit_hpc_2D.sh
#
#    # Quick HPC validation (N=4,8 only):
#    export MAX_N=8
#    qsub -v MAX_N submit_hpc_2D.sh
#
#    # Skip QSVT entirely:
#    export SKIP_QSVT=1
#    qsub -v SKIP_QSVT submit_hpc_2D.sh
#
#    # Run specific sections only (e.g. HET cases):
#    export SECTIONS="4,5"
#    qsub -v SECTIONS submit_hpc_2D.sh
#
#    # Combine options:
#    export MAX_N=16; export SKIP_QSVT=1; export SECTIONS="1,2,3"
#    qsub -v MAX_N,SKIP_QSVT,SECTIONS submit_hpc_2D.sh
#
#  Monitor:
#    qstat -u $USER
#    tail -f results/2Dhpc_run/run.log
#
#  Reference: https://icl-rcs-user-guide.readthedocs.io/
# ============================================================

# --- Resource requests ---
#
# WALLTIME ESTIMATE (per N, all 5 sections, all solvers):
#   N=4  :   ~10 min   (dominated by HHL/VQLS inner solvers)
#   N=8  :   ~60 min
#   N=16 :   ~4 hr
#   N=32 :   ~16 hr
#   N=64 :   ~48 hr    (HHL Jacobi iterations scale with N^2)
#   N=128:   ~96 hr    (estimate; HHL may dominate)
#   N=256:   not recommended in a single job; use --sections 1,3 only
#
# The full N=4..64 sweep fits in 72h (throughput72 queue).
# N=4..128 requires the 120h queue (throughput120) if available.
# For N=256, submit as a separate job with --sections 1,3 (fast cases only).
#
# NOTE ON ncpus vs --max-workers: MAX_WORKERS is set to match ncpus below.
# Aer simulations are already OpenMP-threaded internally. Running more
# worker processes than allocated cores oversubscribes the node.
# If you change ncpus, update MAX_WORKERS accordingly.
#
# NOTE ON mem: Section 2 (Two-Gaussian) computes a 200x200 Fourier reference
# grid. Section 4 (HET MMS) uses non-square grids. At N=128/256 the Jacobi
# iteration stores multiple NxN arrays. 64gb is sufficient for N<=64;
# increase to 128gb for N=128, 256gb for N=256.
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=4:mem=64gb

# --- Job metadata ---
#PBS -N quantum_pde_2Dfull_run
#PBS -o results/2Dhpc_run/pbs_stdout.log
#PBS -e results/2Dhpc_run/pbs_stderr.log

# --- Email notifications ---
#PBS -M j.trobajo-flecha24@imperial.ac.uk
#PBS -m abe

# ============================================================
#  Environment setup
# ============================================================

echo "============================================================"
echo "  QUANTUM PDE SOLVER 2D — HPC JOB START"
echo "  Job ID    : $PBS_JOBID"
echo "  Node      : $(hostname)"
echo "  Date/Time : $(date)"
echo "  Work dir  : $PBS_O_WORKDIR"
echo "  MAX_N     : ${MAX_N:-<not set: full sweep to N=256>}"
echo "  SKIP_QSVT : ${SKIP_QSVT:-0}"
echo "  SECTIONS  : ${SECTIONS:-<not set: all sections 1-5>}"
echo "============================================================"

cd "${PBS_O_WORKDIR}" || { echo "ERROR: Cannot cd to PBS_O_WORKDIR"; exit 1; }

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

VENV_PATH="${HOME}/venvs/qpde"
if [ ! -d "${VENV_PATH}" ]; then
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    echo "       See setup_hpc_env.sh."
    exit 1
fi
source "${VENV_PATH}/bin/activate"
echo "Python: $(which python3) -- $(python3 --version)"

# Guard against missing pyqsp (same pattern as 1D script).
python3 -c "import pyqsp" 2>/dev/null || {
    echo "pyqsp not found in venv; installing..."
    pip install pyqsp==0.2.0
}

# Guard against missing scipy (needed for Section 2 interpolation).
python3 -c "import scipy" 2>/dev/null || {
    echo "scipy not found in venv; installing..."
    pip install scipy
}

# IMPORTANT: must match RESULTS_DIR in scripts/run_hpc_2Dfull.py.
RESULTS_SUBDIR="results/2Dhpc_run"
mkdir -p "${RESULTS_SUBDIR}"

# Keep Aer's internal OpenMP threading within the allocated core count.
# Without this, each of the 4 worker processes may spawn as many OpenMP
# threads as there are physical cores on the node.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
echo "OMP_NUM_THREADS = ${OMP_NUM_THREADS}"

# Verify the QSVT phase cache is populated before starting.
# A missing cache does not abort the job (phases compute on-the-fly),
# but it will add minutes of phase computation per kappa on the first run.
CACHE_DIR="results/qsvt_phase_cache"
N_CACHE_FILES=$(ls "${CACHE_DIR}"/*.npz 2>/dev/null | wc -l)
echo "QSVT phase cache: ${N_CACHE_FILES} files in ${CACHE_DIR}"
if [ "${N_CACHE_FILES}" -lt 6 ]; then
    echo "WARNING: fewer than 6 cache files found."
    echo "         Run submit_precompute_2D.sh first for best performance."
    echo "         Continuing anyway -- phases will be computed on-the-fly."
fi

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

if [ -n "${SECTIONS}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --sections ${SECTIONS}"
    echo "INFO: running sections ${SECTIONS} only."
fi

# Worker count pinned to ncpus requested above.
# Each worker handles one (section, N) work unit independently.
# With 5 sections x 7 N values = 35 work units and 4 workers,
# expected parallelism is ~4x over serial execution.
EXTRA_ARGS="${EXTRA_ARGS} --max-workers 4"

echo "Starting benchmark at $(date)"
echo "Arguments: ${EXTRA_ARGS}"
echo "------------------------------------------------------------"

python3 scripts/run_hpc_2Dfull.py ${EXTRA_ARGS}
EXIT_CODE=$?

echo "------------------------------------------------------------"
echo "Benchmark finished at $(date) with exit code ${EXIT_CODE}"

# ============================================================
#  Copy results to permanent RDS storage
# ============================================================
RDS_RESULTS="${HOME}/qpde-results/2Dhpc_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r "${RESULTS_SUBDIR}"/* "${RDS_RESULTS}/" 2>/dev/null
echo "Results copied to: ${RDS_RESULTS}"

echo "============================================================"
echo "  JOB COMPLETE"
echo "  Exit code : ${EXIT_CODE}"
echo "  Date/Time : $(date)"
echo "============================================================"

exit ${EXIT_CODE}