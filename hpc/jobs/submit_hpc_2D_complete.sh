#!/bin/bash
# ============================================================
#  submit_hpc_2D_complete.sh
#  ONE comprehensive job covering everything still needed in 2D: the
#  ordinary N=64 gaps for the three unaffected cases, plus a full redo of
#  the two SPT-100 HET cases at every N following the geometry correction.
#  Intended to supersede the currently-stuck gapfill_v2 job - kill that one
#  before submitting this.
#
#  State this is built from (results_full.json, 78 rows, last confirmed
#  good state - i.e. BEFORE whatever the still-running job has or hasn't
#  added, which is not yet reflected on disk in any completed row):
#
#    case                          gap                          geometry?
#    sin_hom (1)                   HHL,VQLS,QSVT missing N=64    no
#    TwoGaussian (2)                HHL,VQLS,QSVT missing N=64    no
#    SingleMode (3)                 HHL done to N32; VQLS,QSVT     no
#                                   missing N=32 AND N=64
#    HET_MMS_SPT100 (4)             -- (redo everything) --        YES
#    HET_Sin_MeetingReport (5)      -- (redo everything) --        YES
#
#  Six steps, sequential (concurrent --append invocations on the same
#  results_full.json race each other - see the earlier gapfill scripts):
#
#    Step 1: HHL,  sections 1,2,3,           N=64        capped 8h
#             (fills the real gap; does NOT repeat the N=32 HHL that
#              sections 1,2,3 already have - that one is expensive)
#    Step 2: HHL,  sections 4,5,             N=4,8,16,32,64   capped 8h
#             (full redo under the corrected geometry)
#    Step 3: QSVT, sections 1,2,3,4,5,       N=4,8,16,32,64   uncapped
#             (QSVT is cheap - redoing 1,2's already-correct N<=32 costs
#              minutes, not worth a separate step to avoid; 3,4,5 need it)
#    Step 4: VQLS, sections 1,2,             N=64        capped 6h
#             (deliberately excludes N=32 - already computed, and VQLS at
#              N=32 is the single most expensive thing in this whole
#              benchmark; never repeat it without a specific reason)
#    Step 5: VQLS, section 3,                N=32,64     capped 6h
#             (genuinely missing both)
#    Step 6: VQLS, sections 4,5,             N=4,8,16,32,64   capped 6h
#             (full redo under the corrected geometry)
#
#  ON THE WALL-TIME CAP: the mechanism itself has been checked line-by-line
#  against the live repo and found to match its documented behaviour with
#  no logic bug. The still-running job is nonetheless far past any
#  theoretical worst case, sustaining ~400% CPU the entire time (confirmed
#  via `qstat -f <job> | grep resources_used` - genuinely computing, not
#  hung). The leading explanation is that real per-solve HHL cost at N=64
#  is simply much higher than the small-N extrapolation this project's cost
#  model was built from predicted - not a defect in the cap. The values
#  below (HHL_MAX_WALL_S, VQLS_MAX_WALL_S) are placeholders pending the
#  direct measurement from scripts/estimate_hhl_n64.py; override them via
#  `qsub -v HHL_MAX_WALL_S=<seconds>,VQLS_MAX_WALL_S=<seconds>` once that
#  number is in hand. This script is deliberately structured so that
#  question does not block submitting it - correctness of case coverage
#  does not depend on the cap being exactly right.
#
#  Usage:  qsub hpc/jobs/submit_hpc_2D_complete.sh
# ============================================================

#PBS -l walltime=60:00:00
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -N quantum_pde_2D_complete
#PBS -o results/2Dhpc_run/pbs_stdout_complete.log
#PBS -e results/2Dhpc_run/pbs_stderr_complete.log
#PBS -M j.trobajo-flecha24@imperial.ac.uk
#PBS -m abe

echo "============================================================"
echo "  2D COMPLETE RE-RUN   $(date)   Job ID: $PBS_JOBID"
echo "============================================================"

cd "${PBS_O_WORKDIR}" || { echo "ERROR: cannot cd to PBS_O_WORKDIR"; exit 1; }

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

VENV_PATH="${HOME}/venvs/qpde"
[ -d "${VENV_PATH}" ] || { echo "ERROR: venv not found at ${VENV_PATH}"; exit 1; }
source "${VENV_PATH}/bin/activate"
echo "Python: $(which python3) -- $(python3 --version)"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
mkdir -p results/2Dhpc_run

echo "------------------------------------------------------------"
echo "Removing stale rows for geometry-affected cases (sections 4,5)..."
python3 scripts/cleanup_stale_geometry.py results/2Dhpc_run/results_full.json
echo "------------------------------------------------------------"

HHL_MAX_WALL_S="${HHL_MAX_WALL_S:-28800}"    # 8h - PLACEHOLDER, see header
VQLS_MAX_WALL_S="${VQLS_MAX_WALL_S:-21600}"  # 6h - PLACEHOLDER, see header
STEP_EXIT=(0 0 0 0 0 0)

run_step () {
    local n=$1; shift
    echo "------------------------------------------------------------"
    echo "STEP ${n}: $* -- starting $(date)"
    echo "------------------------------------------------------------"
    python3 hpc/runners/run_2d.py --append --phase-tag "complete_${n}" "$@"
    STEP_EXIT[$((n-1))]=$?
    echo "Step ${n} finished $(date) exit=${STEP_EXIT[$((n-1))]}"
}

# ---- Step 1: HHL, sections 1,2,3, N=64 only (gap fill) ---------------------
run_step 1 --sections 1,2,3 --n-values 64 --solvers hhl \
    -S max_wall_s=${HHL_MAX_WALL_S} --max-workers 4

# ---- Step 2: HHL, sections 4,5, every N (geometry redo) --------------------
run_step 2 --sections 4,5 --n-values 4,8,16,32,64 --solvers hhl \
    -S max_wall_s=${HHL_MAX_WALL_S} --max-workers 4

# ---- Step 3: QSVT, all sections, every N ------------------------------------
run_step 3 --sections 1,2,3,4,5 --n-values 4,8,16,32,64 --solvers qsvt \
    --max-workers 4

# ---- Step 4: VQLS, sections 1,2, N=64 only (never repeat N=32) -------------
run_step 4 --sections 1,2 --n-values 64 --solvers vqls \
    -S max_wall_s=${VQLS_MAX_WALL_S} --max-workers 4

# ---- Step 5: VQLS, section 3, N=32 and N=64 (gap fill) ---------------------
run_step 5 --sections 3 --n-values 32,64 --solvers vqls \
    -S max_wall_s=${VQLS_MAX_WALL_S} --max-workers 4

# ---- Step 6: VQLS, sections 4,5, every N (geometry redo) -------------------
run_step 6 --sections 4,5 --n-values 4,8,16,32,64 --solvers vqls \
    -S max_wall_s=${VQLS_MAX_WALL_S} --max-workers 4

EXIT_CODE=0
for e in "${STEP_EXIT[@]}"; do
    [ "$e" -ne 0 ] && EXIT_CODE=$e
done

echo "------------------------------------------------------------"
echo "Step exit codes: ${STEP_EXIT[*]}"
echo "Overall exit code: ${EXIT_CODE}"

RDS_RESULTS="${HOME}/qpde-results/2Dhpc_run_complete_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r results/2Dhpc_run/* "${RDS_RESULTS}/" 2>/dev/null
echo "Results copied to: ${RDS_RESULTS}"

echo "============================================================"
echo "  JOB COMPLETE   exit=${EXIT_CODE}   $(date)"
echo "============================================================"
exit ${EXIT_CODE}