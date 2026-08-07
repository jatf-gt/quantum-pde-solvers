#!/bin/bash
# ============================================================
#  submit_hpc_2D_gapfill_v2.sh
#  Continuation of submit_hpc_2D_gapfill.sh, updated against the ACTUAL
#  state in results_full.json (76 rows) rather than the .npz file listing
#  used to plan the original script.
#
#  WHY A NEW VERSION, NOT JUST NEW ARGUMENTS TO THE OLD ONE
#  ----------------------------------------------------------
#  Two things changed since the original gapfill script was written:
#
#  1. A real bug in solvers/outer/multigrid.py: the wall-clock cap was only
#     checked once per whole V-cycle, and one V-cycle recurses through the
#     entire hierarchy. At N=64 with HHL this let ONE uninterruptible call
#     run for 139,125s against an intended 21,600s cap - a 6.4x overshoot,
#     confirmed directly in results_full.json (HHL, HET_Sin_MeetingReport,
#     N=64). Fixed: the deadline is now checked before every sweep and every
#     recursive step inside a V-cycle. Copy the corrected multigrid.py into
#     place before running this script - the check below will refuse to
#     proceed if it isn't.
#
#  2. The .npz-file-based gap analysis the original script was planned from
#     turned out to disagree with results_full.json in one place. A .npz
#     solution file is written the instant an individual solver finishes,
#     but the corresponding ROW in results_full.json is only persisted once
#     the ENTIRE work unit (Thomas + every requested solver for that case
#     and N) returns. A job killed mid-work-unit can therefore leave an
#     orphaned .npz on disk for a solver that never made it into
#     results_full.json - which is exactly what happened to
#     2D_Poisson_SingleMode_n1m1 at N=32: Thomas's .npz exists, but its row
#     does not, because HHL/VQLS/QSVT never finished before the original job
#     was killed. results_full.json, not `ls *.npz`, is the authoritative
#     source for "is this combination actually done" after any kill.
#
#  CURRENT STATE (from results_full.json, 76 rows, cross-checked against
#  run.log for the step already in flight when the upload was taken):
#
#    section                  HHL missing      VQLS missing     QSVT missing
#    1  sin_hom                64 (running)     64               64
#    2  TwoGaussian            64 (running)     64               64
#    3  SingleMode_n1m1        32, 64(running)  32, 64           32, 64
#    4  HET_MMS                64 (running)     64               64
#    5  HET_Sin_MeetingReport  -  (done)        32, 64           32, 64
#
#  Section 3's N=32 gap (ALL FOUR solvers, including Thomas) was never
#  targeted by any step of the original script - it is genuinely new work,
#  not a repeat. Section 5's HHL is fully done (both N=32 and N=64 - the
#  N=64 row is the one that hit the 6.4x overshoot; it is a valid, if very
#  slow to obtain, data point and is not redone here). The "(running)"
#  entries were mid-flight using the BUGGY code when the state was captured
#  and should be treated as not-yet-complete: kill that job, then let Step 2
#  below redo them properly.
#
#  Four steps, run SEQUENTIALLY (as before - concurrent --append invocations
#  on the same results_full.json race each other):
#
#    Step 1: HHL,  section 3 only,        N=32       capped 8h   (NEW)
#    Step 2: HHL,  sections 1,2,3,4,      N=64       capped 6h   (re-run,
#                                                     fixed code this time)
#    Step 3: QSVT, all 5 sections,        N=32,64    uncapped (cheap)
#    Step 4: VQLS, sections 1,3,5,        N=32,64    capped 6h
#    Step 5: VQLS, sections 2,4,          N=64       capped 6h
#
#  Usage:  qsub hpc/jobs/submit_hpc_2D_gapfill_v2.sh
#  Run from the repository root (or via `qsub hpc/jobs/...` from the root) -
#  PBS_O_WORKDIR is wherever the shell's CWD was at submission time, and
#  every path below (python3 hpc/runners/..., results/2Dhpc_run/...) is
#  relative to it. Nothing needs to be moved for this to work; see the
#  discussion of this in chat for why.
# ============================================================

#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -N quantum_pde_2D_gapfill_v2
#PBS -o results/2Dhpc_run/pbs_stdout_gapfill_v2.log
#PBS -e results/2Dhpc_run/pbs_stderr_gapfill_v2.log
#PBS -M j.trobajo-flecha24@imperial.ac.uk
#PBS -m abe

echo "============================================================"
echo "  2D GAP-FILL v2 JOB START   $(date)   Job ID: $PBS_JOBID"
echo "============================================================"

cd "${PBS_O_WORKDIR}" || { echo "ERROR: cannot cd to PBS_O_WORKDIR"; exit 1; }

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

VENV_PATH="${HOME}/venvs/qpde"
[ -d "${VENV_PATH}" ] || { echo "ERROR: venv not found at ${VENV_PATH}"; exit 1; }
source "${VENV_PATH}/bin/activate"
echo "Python: $(which python3) -- $(python3 --version)"

# Refuse to run against the buggy multigrid.py - this is the single most
# important guard in this file, given what it costs to get it wrong.
python3 -c "
import inspect
from solvers.outer.multigrid import _v_cycle
sig = inspect.signature(_v_cycle)
assert 'deadline' in sig.parameters, (
    'solvers/outer/multigrid.py does not have the deadline-aware _v_cycle fix. '
    'Copy the corrected file into place before running this job.')
print('multigrid.py fix confirmed present.')
" || exit 1

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
mkdir -p results/2Dhpc_run

HHL_MAX_WALL_S="${HHL_MAX_WALL_S:-28800}"    # 8h
VQLS_MAX_WALL_S="${VQLS_MAX_WALL_S:-21600}"  # 6h
STEP_EXIT=(0 0 0 0 0)

run_step () {
    local n=$1; shift
    echo "------------------------------------------------------------"
    echo "STEP ${n}: $* -- starting $(date)"
    echo "------------------------------------------------------------"
    python3 hpc/runners/run_2d.py --append --phase-tag "gapfill_v2_${n}" "$@"
    STEP_EXIT[$((n-1))]=$?
    echo "Step ${n} finished $(date) exit=${STEP_EXIT[$((n-1))]}"
}

# ---- Step 1 (NEW): HHL, section 3 only, N=32, capped -----------------------
# The gap the original script never targeted at all.
run_step 1 --sections 3 --n-values 32 --solvers hhl \
    -S max_wall_s=${HHL_MAX_WALL_S} --max-workers 4

# ---- Step 2: HHL, sections 1,2,3,4, N=64, capped ---------------------------
# Re-run of the interrupted step - nothing from that attempt is in
# results_full.json, so this is not a repeat of completed work.
run_step 2 --sections 1,2,3,4 --n-values 64 --solvers hhl \
    -S max_wall_s=${HHL_MAX_WALL_S} --max-workers 4

# ---- Step 3: QSVT, all sections, N=32 and N=64, uncapped -------------------
run_step 3 --sections 1,2,3,4,5 --n-values 32,64 --solvers qsvt --max-workers 4

# ---- Step 4: VQLS, sections 1,3,5, N=32 and N=64, capped -------------------
run_step 4 --sections 1,3,5 --n-values 32,64 --solvers vqls \
    -S max_wall_s=${VQLS_MAX_WALL_S} --max-workers 4

# ---- Step 5: VQLS, sections 2,4, N=64 only, capped -------------------------
run_step 5 --sections 2,4 --n-values 64 --solvers vqls \
    -S max_wall_s=${VQLS_MAX_WALL_S} --max-workers 4

EXIT_CODE=0
for e in "${STEP_EXIT[@]}"; do
    [ "$e" -ne 0 ] && EXIT_CODE=$e
done

echo "------------------------------------------------------------"
echo "Step exit codes: ${STEP_EXIT[*]}"
echo "Overall exit code: ${EXIT_CODE}"

RDS_RESULTS="${HOME}/qpde-results/2Dhpc_run_gapfill_v2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r results/2Dhpc_run/* "${RDS_RESULTS}/" 2>/dev/null
echo "Results copied to: ${RDS_RESULTS}"

echo "============================================================"
echo "  JOB COMPLETE   exit=${EXIT_CODE}   $(date)"
echo "============================================================"
exit ${EXIT_CODE}