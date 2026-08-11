#!/bin/bash
# ============================================================================
#  submit_hpc_3D_4th.sh   -   3-D benchmark sweep, FOURTH-ORDER (pentadiagonal)
#
#  Runs `hpc/runners/run_3d.py --order 4` over sections 1-7 at N = 4, 8, for
#  Thomas, HHL, VQLS and QSVT.
#
#  ####################################################################
#  #  THIS JOB REFUSES TO RUN. The 3-D 4th-order operator is wrong.   #
#  ####################################################################
#
#  Why it is gated
#  ---------------
#  3-D shares `solvers/outer/multigrid_4th.py` with 2-D and inherits the same
#  defective boundary closure, uncorrected by the Phase 4a fix applied in 1-D:
#
#    * `build_strip_matrix_4th` folds the ghost node into A[0,1] += -1 -- an
#      EVEN reflection, appropriate to Neumann and not to the Dirichlet data
#      these cases carry. The odd reflection belongs on the diagonal.
#    * `_build_rhs_strip` writes 18*alpha where the row-0 stencil gives
#      14*alpha, summing the boundary and ghost contributions with the same sign.
#
#  Measured convergence order is 0.88, where 4 is intended.
#
#  Even with a correct strip operator the mixed design -- 4th order along the
#  strip, 2nd order transverse -- is capped at order 2 by construction (measured
#  1.95), because `strip_sweep` coupled only j+-1 at 1/h^2. Step 1 of Phase 4b
#  (90d76f1) extended `strip_sweep` to consult the optional `transverse_terms`
#  and `row_matrix_for` hooks; step 2 -- `problems/poisson_line_3d_4th.py`,
#  which supplies them -- is not written. Note that the transverse operator's
#  diagonal differs on the boundary-adjacent strips, so there are TWO distinct
#  strip matrices rather than one: two block encodings and two phase sets, not N.
#
#  `max_wall_s` is also parsed, accepted and silently discarded in the 4th-order
#  3-D path (R3): `_run_4th_order_solver_3d` never reads it, so the run proceeds
#  to its max_iter bound and is terminated by PBS. This matters more in 3-D than
#  anywhere else -- a sweep costs N^2 strip solves per outer iteration.
#
#  The 3-D work accounting is separately wrong: 2-D records `w.add(N, iters)`
#  against 3-D's `w.add(N, N*iters)`. The two are mutually inconsistent and both
#  incorrect; Phase 4b removes this path in favour of the instrumented
#  `strip_sweep`, which fixes it as a side effect.
#
#  What has to land before the gate is lifted
#  ------------------------------------------
#    1. `problems/poisson_line_3d_4th.py` supplying `transverse_terms` and
#       `row_matrix_for`; removal of `solvers/outer/multigrid_4th.py` and
#       `_run_4th_order_solver_3d`.
#    2. Verified order ~4 in 3-D on a solution NOT odd about the boundaries and
#       one with non-zero Dirichlet data.
#    3. The 2nd-order numbers unchanged: SOR and FMG at the recorded iteration
#       counts, and the 15-configuration outer baseline byte-for-byte.
#    4. A 3-D order-4 phase precompute. `precompute_phases.py` accepts
#       `--dim 1|2` only; 3-D has never had one, its strip kappa being ~2.
#
#  See docs/HPC_REPAIR_PLAN.md, Phase 4b.
#
#  Override
#  --------
#  `ALLOW_BROKEN_4TH_CLOSURE=1` proceeds anyway, for a deliberate diagnostic run
#  measuring the defect. Rows so produced must not enter the thesis archive.
#
#    qsub -v ALLOW_BROKEN_4TH_CLOSURE hpc/jobs/submit_hpc_3D_4th.sh
#
#  Monitor:
#    qstat -u $USER
#    tail -f results/3Dhpc_run_4th/pbs_stdout.log    # run.log may be unreadable
#                                                   # from the login node: OI-1
# ============================================================================

#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=128gb
#PBS -N quantum_pde_3D_4th
#PBS -o results/3Dhpc_run_4th/pbs_stdout.log
#PBS -e results/3Dhpc_run_4th/pbs_stderr.log
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

set -u

echo "============================================================"
echo "  QUANTUM PDE SOLVER — HPC JOB START (3-D, 4th order)"
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
# Before the module loads, so a mistaken submission costs seconds and nothing else.
ALLOW_BROKEN_4TH_CLOSURE="${ALLOW_BROKEN_4TH_CLOSURE:-0}"
if [ ! -f "problems/poisson_line_3d_4th.py" ] && [ "${ALLOW_BROKEN_4TH_CLOSURE}" != "1" ]; then
    echo ""
    echo "REFUSING TO RUN: the 3-D 4th-order boundary closure is incorrect."
    echo ""
    echo "  problems/poisson_line_3d_4th.py does not exist, so this sweep would"
    echo "  fall through to solvers/outer/multigrid_4th.py, whose closure applies"
    echo "  an even reflection (A[0,1] += -1) to Dirichlet data and writes"
    echo "  18*alpha where the row-0 stencil gives 14*alpha. Measured convergence"
    echo "  order is 0.88, where 4 is intended: every row this job produced would"
    echo "  have to be discarded -- at N^2 strip solves per outer iteration."
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
    echo "WARNING: ALLOW_BROKEN_4TH_CLOSURE=1. The 3-D 4th-order closure is"
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

RESULTS_SUBDIR="results/3Dhpc_run_4th"
mkdir -p "${RESULTS_SUBDIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

N_VALUES="${N_VALUES:-4,8}"
SECTIONS="${SECTIONS:-1,2,3,4,5,6,7}"
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
    python3 hpc/runners/run_3d.py \
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
python3 scripts/gap_analysis.py --dim 3 \
        --results-dir "${RESULTS_SUBDIR}" --n-values "${N_VALUES}" \
        -o results/manifests/rerun_3d_order4.json || true

RDS_RESULTS="${HOME}/qpde-results/3Dhpc_run_4th_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r "${RESULTS_SUBDIR}"/* "${RDS_RESULTS}/" 2>/dev/null
echo "Results copied to: ${RDS_RESULTS}"

exit ${OVERALL}
