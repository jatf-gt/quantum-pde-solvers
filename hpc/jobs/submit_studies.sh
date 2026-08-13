#!/bin/bash
# ============================================================================
#  submit_studies.sh   -   equal-accuracy and OAT sensitivity studies (1-D)
#
#  Runs `hpc/runners/run_studies.py`, which re-solves each 1-D case across a
#  parameter grid per solver. These are the two studies the primary sweep cannot
#  supply: it evaluates every solver at exactly one parameter setting, giving one
#  point per curve, so neither the equal-accuracy comparison nor the sensitivity
#  response can be recovered from it by interpolation.
#
#  Why they matter
#  ---------------
#  Comparing HHL, VQLS and QSVT at nominally equal precision parameters is
#  methodologically unsound: the VQLS cost C bounds the residual only as
#  C >= r^2/kappa^2, HHL's epsilon is coupled to the Trotter count as
#  n_T = ceil(1/epsilon), and the QSVT residual is non-monotone in polynomial
#  degree through the oscillatory Chebyshev error. The equal-accuracy protocol
#  sweeps each solver's own knob until its residual reaches a common target and
#  reports the RESOURCE COST there, which is the defensible comparison.
#
#  Scope
#  -----
#  Deliberately small. Both studies re-solve at every grid point, so the cost is
#  a multiple of a primary-sweep row -- roughly 10 solves per case for HHL, 34
#  for VQLS and 15 for QSVT. They characterise the ALGORITHMS, not the mesh, so
#  the default is N=8 over three cases spanning the accuracy range: a smooth
#  sinusoid, a discontinuous source, and the HET application case. Run at the
#  full sweep grid they would cost more than the benchmark they annotate.
#
#  Sub-case 3c is excluded unconditionally by the runner: its quantum solves
#  return ~100 % error at every N, so a parameter study over it would
#  characterise a defect rather than an algorithm.
#
#  Wall time is dominated by QSVT's epsilon sweep, whose smallest values raise
#  the polynomial degree and therefore the angle solve. Measured at N=4 on a
#  laptop: ~4 min per case for all three solvers, both studies. N=8 is several
#  times that, and the 6 h request below is a generous margin, not an estimate.
#
#  Usage
#  -----
#    qsub hpc/jobs/submit_studies.sh                       # defaults
#
#    # Both studies at a second resolution, for a resolution-dependence check:
#    export N_VALUES="16"; qsub -v N_VALUES hpc/jobs/submit_studies.sh
#
#    # One study only, or a restricted solver set:
#    export STUDY="equal-accuracy"; qsub -v STUDY hpc/jobs/submit_studies.sh
#    export SOLVERS="vqls,qsvt";    qsub -v SOLVERS hpc/jobs/submit_studies.sh
#
#    # Fourth-order operator instead of second:
#    export ORDER="4"; qsub -v ORDER hpc/jobs/submit_studies.sh
#
#  N_VALUES, CASES and SOLVERS are comma-separated and must be passed to qsub as
#  `-v NAME` (bare name, value taken from the exported shell variable). The form
#  `-v NAME=value` breaks on PBS's comma-splitting.
#
#  Monitor:
#    qstat -u $USER
#    tail -f results/1Dstudies/pbs_stdout.log     # studies.log may be unreadable
#                                                 # from the login node: OI-1
# ============================================================================

#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -N quantum_pde_studies
#PBS -o results/1Dstudies/pbs_stdout.log
#PBS -e results/1Dstudies/pbs_stderr.log
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

set -u

echo "============================================================"
echo "  QUANTUM PDE SOLVER — PARAMETER STUDIES (1-D)"
echo "  Job ID    : ${PBS_JOBID:-interactive}"
echo "  Node      : $(hostname)"
echo "  Date/Time : $(date)"
echo "  Work dir  : ${PBS_O_WORKDIR:-$(pwd)}"
echo "============================================================"

REPO_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
while [ ! -f "${REPO_ROOT}/pyproject.toml" ] && [ "${REPO_ROOT}" != "/" ]; do
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
if [ ! -f "${REPO_ROOT}/pyproject.toml" ]; then
    echo "ERROR: no repository root (pyproject.toml) at or above ${PBS_O_WORKDIR:-$(pwd)}."
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

ORDER="${ORDER:-2}"

# Refuses a dirty tree, and at ORDER=4 an environment holding the UPSTREAM
# quantum_linear_solvers, which has no pentadiagonal_toeplitz module.
ORDER="${ORDER}" bash hpc/jobs/_preflight.sh || exit 1

RESULTS_SUBDIR="results/1Dstudies"
mkdir -p "${RESULTS_SUBDIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

STUDY="${STUDY:-both}"
N_VALUES="${N_VALUES:-8}"
CASES="${CASES:-}"
SOLVERS="${SOLVERS:-hhl,vqls,qsvt}"
R_TARGET="${R_TARGET:-}"
RUN_TAG="${RUN_TAG:-}"

echo ""
echo "  STUDY     : ${STUDY}"
echo "  ORDER     : ${ORDER}"
echo "  N_VALUES  : ${N_VALUES}"
echo "  SOLVERS   : ${SOLVERS}"
echo "  CASES     : ${CASES:-<runner default: fS_hom, fH_hom, het_3a>}"
echo "  R_TARGET  : ${R_TARGET:-<runner default>}"
echo ""

OPT_ARGS=""
[ -n "${CASES}" ]    && OPT_ARGS="${OPT_ARGS} --cases ${CASES}"
[ -n "${R_TARGET}" ] && OPT_ARGS="${OPT_ARGS} --r-target ${R_TARGET}"
[ -n "${RUN_TAG}" ]  && OPT_ARGS="${OPT_ARGS} --run-tag ${RUN_TAG}"

echo "------------------------------------------------------------"
echo "STUDIES START  $(date)"
echo "------------------------------------------------------------"

python3 hpc/runners/run_studies.py \
    --study "${STUDY}" \
    --order "${ORDER}" \
    --n-values "${N_VALUES}" \
    --solvers "${SOLVERS}" \
    ${OPT_ARGS}
RC=$?

echo "------------------------------------------------------------"
echo "STUDIES FINISHED  $(date)  exit=${RC}"
echo "------------------------------------------------------------"

# Copy to RDS alongside the sweep results. The runner writes incrementally after
# every case, so a walltime kill leaves everything completed up to that point.
DEST="${HOME}/qpde-results/studies_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${DEST}"
cp -r "${RESULTS_SUBDIR}"/* "${DEST}/" 2>/dev/null || true
echo "  Results copied to ${DEST}"

exit ${RC}
