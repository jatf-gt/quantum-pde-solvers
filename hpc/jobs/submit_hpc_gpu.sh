#!/bin/bash
# ============================================================
#  submit_hpc_gpu.sh
#  PBS Pro GPU job submission script for Imperial College CX3
#  Phase 2, targeting the gpu72 queue.
#
#  This script requests a single L40S GPU (48 GB GDDR6, Ada
#  Lovelace, CUDA compute capability 8.9) for GPU-accelerated
#  Qiskit Aer statevector simulation via NVIDIA cuStateVec.
#
#  Expected speedup over CPU:
#    N=8  QSVT (depth 6,479):   ~10–30× → ~7–22 s (vs 222 s)
#    N=16 QSVT (depth 44,567):  ~10–50× → feasible (vs hours)
#
#  GPU mode requires serial execution (--max-workers 1) to
#  prevent CUDA context conflicts between worker processes.
#  CPU-bound cases (Thomas, HHL, VQLS) benefit from OpenMP
#  parallelism within each Aer simulation call instead.
#
#  Queue constraints (CX3 Phase 2 gpu72):
#    Nodes per job : 1
#    CPUs per node : 1–64
#    Walltime      : 0–72 h
#    GPU limit     : 12 GPUs total per user
#
#  Reference: https://icl-rcs-user-guide.readthedocs.io/
# ============================================================

# --- Resource requests ---
# 1 L40S GPU, 8 CPUs (for OpenMP within Aer), 64 GB RAM.
# The L40S has 48 GB GDDR6; statevectors for N≤32 fit comfortably
# (2^(n+1) complex128 values: N=32 → 9 qubits → 16 kB; trivial).
#PBS -l select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S
#PBS -l walltime=24:00:00
#PBS -q gpu72

# --- Job metadata ---
#PBS -N quantum_pde_gpu_run
#PBS -o results/1Dhpc_run/pbs_gpu_stdout.log
#PBS -e results/1Dhpc_run/pbs_gpu_stderr.log

# --- Email notifications ---
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

# ============================================================
#  Environment setup
# ============================================================

echo "============================================================"
echo "  QUANTUM PDE SOLVER — GPU HPC JOB START"
echo "  Job ID    : $PBS_JOBID"
echo "  Node      : $(hostname)"
echo "  Date/Time : $(date)"
echo "  Work dir  : $PBS_O_WORKDIR"
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

# Load production modules. The CUDA toolkit is bundled with the
# qiskit-aer-gpu wheel; no separate CUDA module load is required.
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

# Activate the virtual environment containing qiskit-aer-gpu.
# NOTE: qiskit-aer-gpu REPLACES qiskit-aer in the environment.
# Install with: pip install qiskit-aer-gpu  (CUDA 12)
VENV_PATH="${HOME}/venvs/qpde-gpu"
if [ ! -d "${VENV_PATH}" ]; then
    echo "ERROR: GPU virtual environment not found at ${VENV_PATH}"
    echo "       Create it with: virtualenv ${VENV_PATH}"
    echo "       Then: source ${VENV_PATH}/bin/activate"
    echo "             pip install qiskit-aer-gpu qiskit-algorithms numpy scipy"
    echo "             pip install git+https://github.com/anedumla/quantum_linear_solvers.git"
    exit 1
fi

source "${VENV_PATH}/bin/activate"

# Confirm GPU allocation and CUDA visibility.
echo "GPU allocation:"
nvidia-smi --query-gpu=name,memory.total,driver_version \
           --format=csv,noheader 2>/dev/null || echo "  nvidia-smi unavailable"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set by PBS>}"

# Set OpenMP threads for Aer CPU operations within each simulation.
# With 8 CPUs allocated and 1 worker process, all 8 threads are
# available for intra-simulation parallelism.
export OMP_NUM_THREADS=8

# Enable GPU backend in the Python code via environment variable.
export QUANTUM_PDE_USE_GPU=1

mkdir -p results/1Dhpc_run

# ============================================================
#  Run the benchmark — serial mode required for GPU
# ============================================================

EXTRA_ARGS="--max-workers 1"

if [ "${INCLUDE_N64:-0}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --include-n64"
    echo "INFO: N=64 included in sweep."
fi

echo "Starting GPU benchmark at $(date)"
echo "Arguments: ${EXTRA_ARGS}"
echo "------------------------------------------------------------"

python3 hpc/runners/run_1d.py ${EXTRA_ARGS}
EXIT_CODE=$?

echo "------------------------------------------------------------"
echo "GPU benchmark finished at $(date) with exit code ${EXIT_CODE}"

# ============================================================
#  Archive results to permanent RDS storage
# ============================================================

RDS_RESULTS="${HOME}/qpde-results/hpc_gpu_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r results/1Dhpc_run/* "${RDS_RESULTS}/"
echo "Results archived to: ${RDS_RESULTS}"

echo "============================================================"
echo "  JOB COMPLETE  |  Exit: ${EXIT_CODE}  |  $(date)"
echo "============================================================"

exit ${EXIT_CODE}