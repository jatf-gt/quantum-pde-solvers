#!/bin/bash
# ============================================================
#  submit_precompute_2D.sh
#  PBS Pro job submission for 2D QSVT phase precomputation.
#  Imperial College London CX3 HPC.
#
#  All 2D kappas (~2-3) give degree ~30-85, so the full precompute
#  (14 kappas x 3 epsilons = 42 entries) takes under 5 minutes.
#  Walltime is generous.
#
#  Kappa is derived from PoissonLine2D -- the same class the solver
#  uses -- so specify the RESOLUTION, not the kappa. The former KAPPAS
#  variable is gone: it selected from a hand-maintained table that had
#  drifted from the solver by up to 0.28, writing every 2D HET entry
#  above N=4 under a key no solver would ever request.
#
#  List-valued variables must be passed as `-v NAME` with the value
#  exported in the shell -- `-v NAME=a,b` breaks on PBS's comma splitting.
#
#  Usage:
#    qsub hpc/jobs/submit_precompute_2D.sh
#    export N_VALUES="4,8,16"; qsub -v N_VALUES hpc/jobs/submit_precompute_2D.sh
#    qsub -v DOMAIN=het hpc/jobs/submit_precompute_2D.sh
#    qsub -v MAX_DEGREE=200 hpc/jobs/submit_precompute_2D.sh
#    qsub -v LIST_KAPPAS=1 hpc/jobs/submit_precompute_2D.sh   # print keys, compute nothing
# ============================================================

#PBS -N qsvt_precompute_2D
#PBS -l walltime=00:30:00
#PBS -l select=1:ncpus=1:mem=4gb
#PBS -o results/2Dhpc_run/precompute_2D_stdout.log
#PBS -e results/2Dhpc_run/precompute_2D_stderr.log
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

echo "============================================================"
echo "  QSVT 2D PHASE PRECOMPUTE — HPC JOB START"
echo "  Job ID    : $PBS_JOBID"
echo "  Node      : $(hostname)"
echo "  Date/Time : $(date)"
echo "  Work dir  : $PBS_O_WORKDIR"
echo "  N_VALUES  : ${N_VALUES:-<not set, script default: 4..256>}"
echo "  DOMAIN    : ${DOMAIN:-<not set, script default: all>}"
echo "  MAX_DEGREE: ${MAX_DEGREE:-<uncapped>}"
echo "  LIST_KAPPAS: ${LIST_KAPPAS:-0}"
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

# N_VALUES replaces the former KAPPAS variable. Kappa is now derived from
# PoissonLine2D -- the same class the solver uses -- rather than selected from
# a maintained table, so the resolution is the thing to specify and the cache
# key cannot drift out of step with the solver.
if [ -n "${N_VALUES}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --n-values ${N_VALUES}"
fi

# 'square', 'het' or 'all'. The strip operator depends on the grid aspect
# ratio, so the unit square and the HET channel have distinct kappa sequences.
if [ -n "${DOMAIN}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --domain ${DOMAIN}"
fi

if [ -n "${MAX_DEGREE}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --max-degree ${MAX_DEGREE}"
fi

# Replaces the former VERIFY flag. There is no longer a table to verify: this
# prints the kappa each case will use and exits without computing.
if [ "${LIST_KAPPAS:-0}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --list-kappas"
fi

echo "Starting precompute at $(date)"
echo "Arguments: ${EXTRA_ARGS}"
echo "------------------------------------------------------------"

python3 hpc/runners/precompute_phases.py --dim 2 ${EXTRA_ARGS}
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