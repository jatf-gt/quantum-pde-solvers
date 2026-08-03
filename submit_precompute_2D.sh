#!/bin/bash
# ============================================================
#  submit_precompute_2D.sh
#  PBS Pro job submission for 2D QSVT phase precomputation.
#  Imperial College London CX3 HPC.
#
#  All 2D kappas (~2-3) give degree ~30-60.
#  Full precompute (14 kappas x 3 epsilons = 42 entries)
#  takes under 5 minutes. Walltime is generous.
#
#  Usage:
#    qsub submit_precompute_2D.sh
#    qsub -v KAPPAS="2.3586,2.7725" submit_precompute_2D.sh
#    qsub -v MAX_DEGREE=200 submit_precompute_2D.sh
#    qsub -v VERIFY=1 submit_precompute_2D.sh
# ============================================================

#PBS -N qsvt_precompute_2D
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -o results/2Dhpc_run/precompute_2D_stdout.log
#PBS -e results/2Dhpc_run/precompute_2D_stderr.log
#PBS -M j.trobajo-flecha24@imperial.ac.uk
#PBS -m abe

echo "============================================================"
echo "  QSVT 2D PHASE PRECOMPUTE — HPC JOB START"
echo "  Job ID    : $PBS_JOBID"
echo "  Node      : $(hostname)"
echo "  Date/Time : $(date)"
echo "  Work dir  : $PBS_O_WORKDIR"
echo "  KAPPAS    : ${KAPPAS:-<all 2D kappas>}"
echo "  MAX_DEGREE: ${MAX_DEGREE:-<uncapped>}"
echo "  VERIFY    : ${VERIFY:-0}"
echo "============================================================"

cd "${PBS_O_WORKDIR}" || { echo "ERROR: Cannot cd to PBS_O_WORKDIR"; exit 1; }

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

VENV_PATH="${HOME}/venvs/qpde"
if [ ! -d "${VENV_PATH}" ]; then
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    exit 1
fi
source "${VENV_PATH}/bin/activate"
echo "Python: $(which python3) -- $(python3 --version)"

python3 -c "import pyqsp" 2>/dev/null || {
    echo "pyqsp not found; installing..."
    pip install pyqsp==0.2.0
}

mkdir -p results/2Dhpc_run
mkdir -p results/qsvt_phase_cache

# ============================================================
#  Build argument list
# ============================================================
EXTRA_ARGS=""

if [ -n "${KAPPAS}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --kappas ${KAPPAS}"
fi

if [ -n "${MAX_DEGREE}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --max-degree ${MAX_DEGREE}"
fi

if [ "${VERIFY:-0}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --verify-kappas"
fi

echo "Starting precompute at $(date)"
echo "Arguments: ${EXTRA_ARGS}"
echo "------------------------------------------------------------"

python3 scripts/precompute_2D_qsvt_phases.py ${EXTRA_ARGS}
EXIT_CODE=$?

echo "------------------------------------------------------------"
echo "Precompute finished at $(date) with exit code ${EXIT_CODE}"

# ============================================================
#  Copy cache to permanent RDS storage
# ============================================================
RDS_CACHE="${HOME}/qpde-results/qsvt_phase_cache_2D_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_CACHE}"
cp -r results/qsvt_phase_cache/* "${RDS_CACHE}/" 2>/dev/null
echo "Cache copied to: ${RDS_CACHE}"

echo "============================================================"
echo "  JOB COMPLETE"
echo "  Exit code : ${EXIT_CODE}"
echo "  Date/Time : $(date)"
echo "============================================================"

exit ${EXIT_CODE}