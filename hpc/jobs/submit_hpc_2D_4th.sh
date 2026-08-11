#!/bin/bash
# ============================================================================
#  submit_hpc_2D_4th.sh   -   2-D benchmark sweep, FOURTH-ORDER (pentadiagonal)
#
#  Runs `hpc/runners/run_2d.py --order 4` over sections 1-5 at N = 4, 8, 16,
#  for Thomas, HHL, VQLS and QSVT.
#
#  ####################################################################
#  #  THIS JOB REFUSES TO RUN. The 2-D 4th-order operator is wrong.   #
#  ####################################################################
#
#  Why it is gated
#  ---------------
#  `solvers/outer/multigrid_4th.py` carries its own boundary closure, separate
#  from the one corrected in 1-D by Phase 4a, and it is wrong in two ways:
#
#    * `build_strip_matrix_4th` folds the ghost node into A[0,1] += -1. That is
#      an EVEN reflection, appropriate to a Neumann condition and not to the
#      Dirichlet data these cases carry; the odd reflection belongs on the
#      diagonal, A[0,0] += 1.
#    * `_build_rhs_strip` still writes the 18*alpha form. Substituting the ghost
#      into the row-0 stencil gives 16*alpha from the boundary node and
#      -2*alpha from the ghost, i.e. 14*alpha; the implementation sums them with
#      the same sign.
#
#  Measured convergence order against a manufactured solution is 0.88, where 4
#  is intended. A sweep submitted now would spend its walltime producing rows
#  that are not 4th-order accurate and would have to be discarded in full.
#
#  A second, independent bound applies even once the closure is fixed. The mixed
#  design -- 4th order along the strip, 2nd order transverse -- is capped at
#  order 2 by construction, measured 1.95 with a correct strip operator, because
#  `strip_sweep` hardcoded the transverse coupling as 1/h^2 at j+-1. Step 1 of
#  Phase 4b (90d76f1) extended `strip_sweep` to consult the optional
#  `transverse_terms` and `row_matrix_for` hooks; step 2 -- the
#  `problems/poisson_line_2d_4th.py` class that supplies them -- is not written.
#
#  A third: `max_wall_s` is parsed, accepted and silently discarded in the
#  4th-order 2-D path (R3). `_run_4th_order_solver_2d` never reads it, so the
#  run proceeds to its max_iter bound and is terminated by PBS. Passing
#  `-S max_wall_s=...` here buys nothing until Phase 4b removes that path in
#  favour of the ordinary `solve()` call.
#
#  What has to land before the gate is lifted
#  ------------------------------------------
#    1. `problems/poisson_line_2d_4th.py` (and the 3-D twin), supplying
#       `transverse_terms` and `row_matrix_for`; removal of
#       `solvers/outer/multigrid_4th.py` and `_run_4th_order_solver_2d`.
#    2. Verified order ~4 in 2-D on a solution that is NOT odd about the
#       boundaries and one with non-zero Dirichlet data. The historical blind
#       spot is -sin(pi x)/pi^2, which is odd about both boundaries and on which
#       even the defective closure is accidentally exact.
#    3. The 2nd-order numbers unchanged: SOR 33/66/130, FMG 3 cycles, legacy
#       Jacobi 26/73, and the 15-configuration outer baseline byte-for-byte.
#    4. `hpc/jobs/submit_precompute_2D.sh` extended to order 4 -- currently
#       `precompute_phases.build_targets` raises on `--dim 2 --order 4`, because
#       a key written against an operator about to change is a silent miss.
#
#  See docs/HPC_REPAIR_PLAN.md, Phase 4b.
#
#  Override
#  --------
#  `ALLOW_BROKEN_4TH_CLOSURE=1` proceeds anyway. It exists for a deliberate
#  diagnostic run -- measuring the defect, not benchmarking against it. Rows so
#  produced must not enter the thesis archive.
#
#    qsub -v ALLOW_BROKEN_4TH_CLOSURE hpc/jobs/submit_hpc_2D_4th.sh
#
#  Monitor:
#    qstat -u $USER
#    tail -f results/2Dhpc_run_4th/pbs_stdout.log    # run.log may be unreadable
#                                                   # from the login node: OI-1
# ============================================================================

#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=128gb
#PBS -N quantum_pde_2D_4th
#PBS -o results/2Dhpc_run_4th/pbs_stdout.log
#PBS -e results/2Dhpc_run_4th/pbs_stderr.log
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

set -u

echo "============================================================"
echo "  QUANTUM PDE SOLVER — HPC JOB START (2-D, 4th order)"
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

# ── Correctness gate ─────────────────────────────────────────────────────────
# Placed before the module loads so that a mistaken submission costs seconds of
# queue time and nothing else.
ALLOW_BROKEN_4TH_CLOSURE="${ALLOW_BROKEN_4TH_CLOSURE:-0}"
if [ ! -f "problems/poisson_line_2d_4th.py" ] && [ "${ALLOW_BROKEN_4TH_CLOSURE}" != "1" ]; then
    echo ""
    echo "REFUSING TO RUN: the 2-D 4th-order boundary closure is incorrect."
    echo ""
    echo "  problems/poisson_line_2d_4th.py does not exist, so this sweep would"
    echo "  fall through to solvers/outer/multigrid_4th.py, whose closure applies"
    echo "  an even reflection (A[0,1] += -1) to Dirichlet data and writes"
    echo "  18*alpha where the row-0 stencil gives 14*alpha. Measured convergence"
    echo "  order is 0.88, where 4 is intended: every row this job produced would"
    echo "  have to be discarded."
    echo ""
    echo "  See docs/HPC_REPAIR_PLAN.md Phase 4b for the outstanding work, and"
    echo "  the header of this script for the four gates that must be cleared."
    echo ""
    echo "  ALLOW_BROKEN_4TH_CLOSURE=1 overrides this, for a deliberate"
    echo "  diagnostic run only."
    exit 2
fi
if [ "${ALLOW_BROKEN_4TH_CLOSURE}" = "1" ]; then
    echo ""
    echo "WARNING: ALLOW_BROKEN_4TH_CLOSURE=1. The 2-D 4th-order closure is"
    echo "         first-order accurate. These rows are a measurement of the"
    echo "         defect, not a benchmark, and must not enter the archive."
    echo ""
fi

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

VENV_PATH="${HOME}/venvs/qpde"
if [ ! -d "${VENV_PATH}" ]; then
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    exit 1
fi
source "${VENV_PATH}/bin/activate"

# ORDER=4 additionally checks the pentadiagonal module, absent from the upstream
# quantum_linear_solvers. Its absence cost a 21 h job in which every HHL row
# failed with ModuleNotFoundError.
ORDER=4 bash hpc/jobs/_preflight.sh || exit 1

RESULTS_SUBDIR="results/2Dhpc_run_4th"
mkdir -p "${RESULTS_SUBDIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

N_VALUES="${N_VALUES:-4,8,16}"
SECTIONS="${SECTIONS:-1,2,3,4,5}"
SOLVERS="${SOLVERS:-hhl,vqls,qsvt}"
WORKERS="${WORKERS:-4}"
SCHEME="${SCHEME:-fmg}"
TOL="${TOL:-}"
MAX_OUTER="${MAX_OUTER:-}"

echo ""
echo "  N_VALUES  : ${N_VALUES}"
echo "  SECTIONS  : ${SECTIONS}"
echo "  SOLVERS   : ${SOLVERS}"
echo "  SCHEME    : ${SCHEME}"
echo "  WORKERS   : ${WORKERS}"

OPT_ARGS=""
[ -n "${TOL}" ]       && OPT_ARGS="${OPT_ARGS} --tol ${TOL}"
[ -n "${MAX_OUTER}" ] && OPT_ARGS="${OPT_ARGS} --max-outer ${MAX_OUTER}"

# ── Sweep ────────────────────────────────────────────────────────────────────
# One step per resolution, ascending, so a walltime kill loses only the most
# expensive tail. --append is what makes that safe: without it each step would
# rewrite results_full.json from its own rows alone.
OVERALL=0

run_step () {
    local nval=$1
    echo ""
    echo "------------------------------------------------------------"
    echo "STEP N=${nval}  sections=${SECTIONS}  solvers=${SOLVERS}  $(date)"
    echo "------------------------------------------------------------"
    python3 hpc/runners/run_2d.py \
        --order 4 \
        --append --phase-tag "order4_n${nval}" \
        --n-values "${nval}" \
        --sections "${SECTIONS}" \
        --solvers "${SOLVERS}" \
        --scheme "${SCHEME}" \
        --max-workers "${WORKERS}" \
        ${OPT_ARGS}
    local rc=$?
    echo "STEP N=${nval} finished $(date) exit=${rc}"
    [ "${rc}" -ne 0 ] && [ "${OVERALL}" -eq 0 ] && OVERALL=${rc}
    return 0
}

echo ""
echo "Starting 4th-order sweep at $(date)"

for nval in $(echo "${N_VALUES}" | tr ',' ' '); do
    run_step "${nval}"
done

echo ""
echo "------------------------------------------------------------"
echo "Sweep finished at $(date) with overall exit code ${OVERALL}"

echo ""
echo "============================================================"
echo "  Gap analysis after the run"
echo "============================================================"
python3 scripts/gap_analysis.py --dim 2 \
        --results-dir "${RESULTS_SUBDIR}" --n-values "${N_VALUES}" \
        -o results/manifests/rerun_2d_order4.json || true

RDS_RESULTS="${HOME}/qpde-results/2Dhpc_run_4th_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r "${RESULTS_SUBDIR}"/* "${RDS_RESULTS}/" 2>/dev/null
echo "Results copied to: ${RDS_RESULTS}"

exit ${OVERALL}
