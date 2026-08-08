#!/bin/bash
# ============================================================
#  submit_hpc_1D_4th.sh
#  PBS Pro job submission script for Imperial College CX3 HPC.
#
#  Runs the full 1-D benchmark sweep for the 4th order scheme,
#  N = 4..32, all cases, all solvers (Thomas, HHL, VQLS, QSVT).
#
#  Usage:
#    qsub hpc/jobs/submit_hpc_1D_4th.sh
#
#  Monitor:
#    qstat -u $USER
#    tail -f results/1Dhpc_run_4th/run.log
# ============================================================

#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=128gb
#PBS -N quantum_pde_1D_4th
#PBS -o results/1Dhpc_run_4th/pbs_stdout.log
#PBS -e results/1Dhpc_run_4th/pbs_stderr.log
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

echo "============================================================"
echo "  QUANTUM PDE SOLVER — HPC JOB START (1D 4th Order)"
echo "  Job ID    : $PBS_JOBID"
echo "  Node      : $(hostname)"
echo "  Date/Time : $(date)"
echo "  Work dir  : $PBS_O_WORKDIR"
echo "============================================================"

REPO_ROOT="${PBS_O_WORKDIR}"
while [ ! -f "${REPO_ROOT}/pyproject.toml" ] && [ "${REPO_ROOT}" != "/" ]; do
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
if [ ! -f "${REPO_ROOT}/pyproject.toml" ]; then
    echo "ERROR: no repository root (pyproject.toml) at or above ${PBS_O_WORKDIR}."
    exit 1
fi
cd "${REPO_ROOT}" || exit 1

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

VENV_PATH="${HOME}/venvs/qpde"
if [ ! -d "${VENV_PATH}" ]; then
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    exit 1
fi
source "${VENV_PATH}/bin/activate"

python3 -c "import pyqsp" 2>/dev/null || {
    echo "pyqsp not found in venv; installing..."
    pip install pyqsp==0.2.0
}

RESULTS_SUBDIR="results/1Dhpc_run_4th"
mkdir -p "${RESULTS_SUBDIR}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

EXTRA_ARGS="--max-workers 4 --order 4 --max-n 32"
if [ -n "${MAX_N}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --max-n ${MAX_N}"
fi

echo "Starting benchmark at $(date)"
echo "Arguments: ${EXTRA_ARGS}"
echo "------------------------------------------------------------"

python3 hpc/runners/run_1d.py ${EXTRA_ARGS}
EXIT_CODE=$?

echo "------------------------------------------------------------"
echo "Benchmark finished at $(date) with exit code ${EXIT_CODE}"

RDS_RESULTS="${HOME}/qpde-results/1Dhpc_run_4th_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r "${RESULTS_SUBDIR}"/* "${RDS_RESULTS}/" 2>/dev/null
echo "Results copied to: ${RDS_RESULTS}"

exit ${EXIT_CODE}
