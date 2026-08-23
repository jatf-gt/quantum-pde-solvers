#!/bin/bash
# ============================================================================
#  submit_studies.sh   -   equal-accuracy and OAT sensitivity studies
#
#  Runs `hpc/runners/run_studies.py`, which re-solves each case across a
#  parameter grid per solver. These are the two studies the primary sweep cannot
#  supply: it evaluates every solver at exactly one parameter setting, giving one
#  point per curve, so neither the equal-accuracy comparison nor the sensitivity
#  response can be recovered from it by interpolation.
#
#  Why they matter
#  ---------------
#  Comparing HHL, VQLS and QSVT at nominally equal precision parameters is
#  methodologically unsound: the VQLS cost C bounds the residual only as
#  C >= r^2/kappa^2, HHL's epsilon reaches the Trotter count only through the
#  library's own derivation from eps/6 and an evolution time fixed from the
#  spectral bounds (NOT as n_T = ceil(1/epsilon), which is what this file
#  claimed until 2026-08-17 and what the runner recorded), and the QSVT residual
#  is non-monotone in polynomial degree through the oscillatory Chebyshev error.
#  The equal-accuracy protocol
#  sweeps each solver's own knob until its residual reaches a common target and
#  reports the RESOURCE COST there, which is the defensible comparison.
#
#  Scope
#  -----
#  Deliberately small. Both studies re-solve at every grid point, so the cost is
#  a multiple of a primary-sweep row -- roughly 10 solves per case for HHL, 34
#  for VQLS and 15 for QSVT. They characterise the ALGORITHMS, not the mesh, so
#  the default is N=8 over three cases in 1-D spanning the accuracy range: a
#  smooth sinusoid, a discontinuous source, and the HET application case. Run at
#  the full sweep grid they would cost more than the benchmark they annotate.
#
#  Sub-case 3c is excluded unconditionally by the runner: its quantum solves
#  return ~100 % error at every N, so a parameter study over it would
#  characterise a defect rather than an algorithm.
#
#  2-D and 3-D
#  -----------
#  Set DIM=2 or DIM=3. The parameter then drives an `inner_options` entry and the
#  measured quantity is the OUTER residual, since there is no assembled operator
#  in more than one dimension -- the domain is decomposed into strips and coupled
#  iteratively. The outer tolerance is set to r_target so that the comparison is
#  cost at matched accuracy; left at its default, every grid point converges to
#  the same residual and the study measures nothing. Two cases per dimension, one
#  manufactured and one HET.
#
#  N=8 is the SMALLEST resolution the default scheme admits in 2-D and 3-D: FMG
#  needs two grid levels and a 4-wide problem cannot be coarsened. Use
#  `--scheme sor` to go lower, at the cost of comparability with the primary
#  sweep, which uses FMG.
#
#  Cost, measured on a laptop at N=8, per outer solve:
#
#      dimension   QSVT        HHL         VQLS
#      2-D         9 - 103 s   > 200 s     > 200 s
#      3-D         34 - 41 s   ~ 40 min    ~ 1.7 h
#
#  QSVT is minutes per case in either dimension. HHL and VQLS are hours, which is
#  what the walltime below and MAX_WALL_S are sized for. For a first pass, prefer
#  SOLVERS="qsvt" -- it completes both studies in under an hour and answers the
#  question the 2-D/3-D studies exist for.
#
#  Usage
#  -----
#    qsub hpc/jobs/submit_studies.sh                       # 1-D, both studies
#
#    # 2-D and 3-D, QSVT first since it is cheap and completes:
#    export DIM="2" SOLVERS="qsvt"; qsub -v DIM,SOLVERS hpc/jobs/submit_studies.sh
#    export DIM="3" SOLVERS="qsvt"; qsub -v DIM,SOLVERS hpc/jobs/submit_studies.sh
#
#    # 2-D with the expensive solvers, once QSVT has confirmed the pipeline:
#    export DIM="2" SOLVERS="hhl,vqls" MAX_WALL_S="3600"
#    qsub -v DIM,SOLVERS,MAX_WALL_S hpc/jobs/submit_studies.sh
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
#  Output directory
#  ----------------
#  One per (DIM, ORDER, RUN_TAG): `results/2Dstudies`, `results/2Dstudies_4th`,
#  `results/3Dstudies_grid_fix`. Submissions differing in any of the three are
#  therefore independent, and two that differ in NONE are a deliberate re-run of
#  the same configuration. Retrieval installs the directory over its laptop
#  counterpart, dropping the tag: `2Dstudies_grid_fix` -> `results/2Dstudies`.
#
#  Monitor:
#    qstat -u $USER
#    tail -f results/studies_pbs_stdout.log       # studies.log may be unreadable
#                                                 # from the login node: OI-1
# ============================================================================

# 24 h covers a 2-D or 3-D run including HHL and VQLS. A 1-D run finishes in
# 1-2 h and a QSVT-only 2-D/3-D run in under one, so the request is generous
# rather than tuned -- there is no partial credit for a walltime kill in the
# summary, though the runner does write incrementally after every case.
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -N quantum_pde_studies
#PBS -o results/studies_pbs_stdout.log
#PBS -e results/studies_pbs_stderr.log
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

set -u

echo "============================================================"
echo "  QUANTUM PDE SOLVER — PARAMETER STUDIES (${DIM:-1}-D)"
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

DIM="${DIM:-1}"
RUN_TAG="${RUN_TAG:-}"

# One directory per (dimension, order, run tag). None of the three is recorded in
# a filename: run_studies.py overwrites sensitivity_<solver>.json wholesale,
# merges equal_accuracy.json on a key that omits the order, and writes
# run_metadata.json before its first case. Two submissions sharing a directory
# and differing only in ORDER therefore destroy each other's records and
# cross-stamp each other's metadata -- which is what befell the grid_fix pair of
# 2026-08-19. The rule below mirrors run_studies.py::results_dir_for, and the
# runner is told the answer with --results-dir, so the two cannot disagree.
RESULTS_SUBDIR="results/${DIM}Dstudies"
[ "${ORDER}" != "2" ] && RESULTS_SUBDIR="${RESULTS_SUBDIR}_${ORDER}th"
[ -n "${RUN_TAG}" ]   && RESULTS_SUBDIR="${RESULTS_SUBDIR}_${RUN_TAG}"
mkdir -p "${RESULTS_SUBDIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

STUDY="${STUDY:-both}"
# N_VALUES is left unset by default so the runner applies its own per-dimension
# default: 8 everywhere, which in 2-D and 3-D is also the FMG minimum.
N_VALUES="${N_VALUES:-}"
CASES="${CASES:-}"
SOLVERS="${SOLVERS:-hhl,vqls,qsvt}"
R_TARGET="${R_TARGET:-}"
SCHEME="${SCHEME:-fmg}"
# Per-solve outer wall-clock bound, 2-D/3-D only. Each grid point is a full outer
# solve, so without a bound one non-converging point can consume the whole job.
# 1 h suits QSVT comfortably and caps HHL and VQLS at a defensible measurement.
MAX_WALL_S="${MAX_WALL_S:-3600}"

echo ""
echo "  DIM       : ${DIM}"
echo "  STUDY     : ${STUDY}"
echo "  ORDER     : ${ORDER}"
echo "  N_VALUES  : ${N_VALUES:-<runner default: 8>}"
echo "  SOLVERS   : ${SOLVERS}"
echo "  CASES     : ${CASES:-<runner per-dimension default>}"
echo "  R_TARGET  : ${R_TARGET:-<runner default>}"
echo "  RUN_TAG   : ${RUN_TAG:-<none>}"
echo "  OUTPUT    : ${RESULTS_SUBDIR}"
if [ "${DIM}" != "1" ]; then
    echo "  SCHEME    : ${SCHEME}"
    echo "  MAX_WALL_S: ${MAX_WALL_S}"
fi
echo ""

OPT_ARGS=" --results-dir ${RESULTS_SUBDIR}"
[ -n "${N_VALUES}" ] && OPT_ARGS="${OPT_ARGS} --n-values ${N_VALUES}"
[ -n "${CASES}" ]    && OPT_ARGS="${OPT_ARGS} --cases ${CASES}"
[ -n "${R_TARGET}" ] && OPT_ARGS="${OPT_ARGS} --r-target ${R_TARGET}"
[ -n "${RUN_TAG}" ]  && OPT_ARGS="${OPT_ARGS} --run-tag ${RUN_TAG}"
if [ "${DIM}" != "1" ]; then
    OPT_ARGS="${OPT_ARGS} --scheme ${SCHEME} --max-wall-s ${MAX_WALL_S}"
fi

echo "------------------------------------------------------------"
echo "STUDIES START  $(date)"
echo "------------------------------------------------------------"
# The exact command, echoed verbatim. A run that quietly used parameters other
# than those intended is indistinguishable afterwards from one that never ran,
# because the runner rewrites the same filenames either way; the assembled
# argument vector is the only record of what was actually requested.
echo "COMMAND: python3 hpc/runners/run_studies.py --dim ${DIM} --study ${STUDY}" \
     "--order ${ORDER} --solvers ${SOLVERS}${OPT_ARGS}"
echo "------------------------------------------------------------"

# Sentinel for the freshness check in the epilogue. `run_studies.py` writes
# run_metadata.json before its first case, so a metadata file no newer than this
# marker proves the runner never reached that line, whatever else was printed.
STAMP="${RESULTS_SUBDIR}/.job_start_marker"
: > "${STAMP}"

python3 hpc/runners/run_studies.py \
    --dim "${DIM}" \
    --study "${STUDY}" \
    --order "${ORDER}" \
    --solvers "${SOLVERS}" \
    ${OPT_ARGS}
RC=$?

echo "------------------------------------------------------------"
echo "STUDIES FINISHED  $(date)  exit=${RC}"
echo "------------------------------------------------------------"

# Did this job produce anything at all? A job that aborts in preflight, fails an
# import, or dies before its first case leaves the PREVIOUS run's files in place
# untouched -- and the archive step below would then copy those into a directory
# stamped with today's date, which reads as a completed run and has already been
# taken for one. Establish the fact before archiving, and state it plainly.
WROTE_RESULTS=0
if [ -f "${RESULTS_SUBDIR}/run_metadata.json" ] \
   && [ "${RESULTS_SUBDIR}/run_metadata.json" -nt "${STAMP}" ]; then
    WROTE_RESULTS=1
fi
rm -f "${STAMP}"

if [ "${WROTE_RESULTS}" = "0" ]; then
    echo "=============================================================="
    echo "  WARNING: THIS JOB WROTE NO RESULTS."
    echo "  run_metadata.json in ${RESULTS_SUBDIR} predates this job, so the"
    echo "  runner never reached its first case. Whatever that directory holds"
    echo "  belongs to an EARLIER run: do not copy it off the cluster and read it"
    echo "  as the output of this submission. The cause is above in this log, and"
    echo "  in results/studies_pbs_stderr.log."
    echo "=============================================================="
fi

# Copy to RDS alongside the sweep results. The runner writes incrementally after
# every case, so a walltime kill leaves everything completed up to that point --
# which is why a non-zero exit is still archived, but under a name recording it.
SUFFIX=""
[ "${RC}" != "0" ]           && SUFFIX="${SUFFIX}_rc${RC}"
[ "${WROTE_RESULTS}" = "0" ] && SUFFIX="${SUFFIX}_STALE"
DEST="${HOME}/qpde-results/$(basename "${RESULTS_SUBDIR}")_$(date +%Y%m%d_%H%M%S)${SUFFIX}"
mkdir -p "${DEST}"
cp -r "${RESULTS_SUBDIR}"/* "${DEST}/" 2>/dev/null || true
if [ "${WROTE_RESULTS}" = "1" ]; then
    echo "  Results copied to ${DEST}"
else
    echo "  Pre-existing directory copied to ${DEST} (NOT this job's output)."
fi

exit ${RC}
