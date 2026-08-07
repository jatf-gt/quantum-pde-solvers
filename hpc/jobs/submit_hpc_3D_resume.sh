#!/bin/bash
# ============================================================
#  submit_hpc_3D_resume.sh
#  Resume job: complete the remaining N=16 work from the killed 3D run,
#  with wall-clock caps on both HHL and VQLS - nothing already completed
#  is repeated.
#
#  IMPORTANT PREREQUISITE: this job requires the multigrid.py fix delivered
#  alongside this script. The wall-clock cap mechanism had a real bug: it
#  only checked the time budget once per whole V-cycle, and one V-cycle
#  recurses through every level of the hierarchy. At N=16 with HHL this let
#  a single uninterruptible V-cycle run for 139,125s against an intended
#  ~21,600s budget - a ~6x overshoot, which is exactly the kind of runaway
#  cost this cap exists to prevent. The fix checks before every sweep and
#  every recursive step inside a V-cycle instead, bounding the overshoot to
#  a small multiple of the cap rather than a multiple of the whole
#  hierarchy's cost. Copy the corrected solvers/outer/multigrid.py into
#  place before submitting this job - resubmitting with the old file would
#  reproduce the exact failure this script exists to avoid.
#
#  Exact remaining work, from the state at kill time:
#
#    section                          HHL        VQLS         QSVT
#    1  TripleSin_cube                 done       missing      missing
#    2  HET_MMS_SPT100                 done       missing      missing
#    3  HET_RotatingSpoke_SPT100       done       done         done
#    4  HET_Discharge_SPT100           done       missing      missing
#    5  Laplace_BCdriven_cube          done       missing      missing
#    6  Poisson_TwoGaussian_cube       missing    missing      missing
#    7  Poisson_HighMode_n2m3l4        missing    missing      missing
#
#  Section 3 needs nothing further - its VQLS finished (in 37.0 hours,
#  which is the single data point behind the caps chosen below) before the
#  job was killed. Sections 1,2,4,5 already have HHL; only 6 and 7 need it.
#  All of 1,2,4,5,6,7 need VQLS and QSVT.
#
#  Three steps, run SEQUENTIALLY in this one job (see the 2D gapfill script
#  for why: concurrent jobs both using --append on the same results_full.json
#  race, since each loads it once at its own startup):
#
#    Step 1: HHL,  sections 6,7,           N=16   capped 8h  (28800s)
#    Step 2: VQLS, sections 1,2,4,5,6,7,   N=16   capped 6h  (21600s)
#    Step 3: QSVT, sections 1,2,4,5,6,7,   N=16   uncapped (cheap; QSVT has
#                                                  never taken more than a
#                                                  few minutes at this N)
#
#  Cap rationale:
#    HHL:  uncapped 3D-N16 runs have been remarkably consistent at
#          ~24,900-25,500s (~7h) across five different cases so far. An 8h
#          cap is a safety margin above that observed norm, not a truncation
#          of it - sections 6/7 are expected to finish naturally within it.
#    VQLS: the only completed 3D-N16 run took 37.0h - dramatically worse
#          than HHL, and with only one data point its variance across the
#          other five cases is unknown. 6h is deliberately much tighter than
#          "let it try to finish" - the goal is a bounded, honestly-labelled
#          wall_time_exceeded data point per case, not a repeat of the
#          37-hour outcome six times over.
#
#  N=32 (Phase 2, QSVT-only) is deliberately NOT included here. Getting
#  Phase 1 (N<=16, all solvers) to a clean, fully-tagged state first, then
#  running N=32 as its own follow-up job, keeps this job's worst-case time
#  bounded and easy to reason about - see submit_hpc_3D.sh's own module
#  docstring for why N=32/64 with HHL or VQLS is not attempted at all.
#
#  Usage:  qsub hpc/jobs/submit_hpc_3D_resume.sh
# ============================================================

#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -N quantum_pde_3D_resume
#PBS -o results/3Dhpc_run/pbs_stdout_resume.log
#PBS -e results/3Dhpc_run/pbs_stderr_resume.log
#PBS -M j.trobajo-flecha24@imperial.ac.uk
#PBS -m abe

echo "============================================================"
echo "  3D RESUME JOB START   $(date)   Job ID: $PBS_JOBID"
echo "============================================================"

cd "${PBS_O_WORKDIR}" || { echo "ERROR: cannot cd to PBS_O_WORKDIR"; exit 1; }

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

VENV_PATH="${HOME}/venvs/qpde"
[ -d "${VENV_PATH}" ] || { echo "ERROR: venv not found at ${VENV_PATH}"; exit 1; }
source "${VENV_PATH}/bin/activate"
echo "Python: $(which python3) -- $(python3 --version)"

# Confirm the multigrid.py fix is actually in place before spending any
# walltime - a stale copy would silently reproduce the 6x overshoot bug.
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
mkdir -p results/3Dhpc_run

HHL_MAX_WALL_S="${HHL_MAX_WALL_S:-28800}"    # 8h
VQLS_MAX_WALL_S="${VQLS_MAX_WALL_S:-21600}"  # 6h
STEP_EXIT=(0 0 0)

run_step () {
    local n=$1; shift
    echo "------------------------------------------------------------"
    echo "STEP ${n}: $* -- starting $(date)"
    echo "------------------------------------------------------------"
    python3 hpc/runners/run_3d.py --append --phase-tag "resume_${n}" "$@"
    STEP_EXIT[$((n-1))]=$?
    echo "Step ${n} finished $(date) exit=${STEP_EXIT[$((n-1))]}"
}

# ---- Step 1: HHL, sections 6,7 only, N=16, capped 8h -----------------------
run_step 1 --sections 6,7 --n-values 16 --solvers hhl \
    -S max_wall_s=${HHL_MAX_WALL_S} --max-workers 4

# ---- Step 2: VQLS, sections 1,2,4,5,6,7, N=16, capped 6h --------------------
# (section 3 excluded - its VQLS already completed)
run_step 2 --sections 1,2,4,5,6,7 --n-values 16 --solvers vqls \
    -S max_wall_s=${VQLS_MAX_WALL_S} --max-workers 4

# ---- Step 3: QSVT, sections 1,2,4,5,6,7, N=16, uncapped ---------------------
run_step 3 --sections 1,2,4,5,6,7 --n-values 16 --solvers qsvt --max-workers 4

EXIT_CODE=0
for e in "${STEP_EXIT[@]}"; do
    [ "$e" -ne 0 ] && EXIT_CODE=$e
done

echo "------------------------------------------------------------"
echo "Step exit codes: ${STEP_EXIT[*]}"
echo "Overall exit code: ${EXIT_CODE}"

RDS_RESULTS="${HOME}/qpde-results/3Dhpc_run_resume_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r results/3Dhpc_run/* "${RDS_RESULTS}/" 2>/dev/null
echo "Results copied to: ${RDS_RESULTS}"

echo "============================================================"
echo "  JOB COMPLETE   exit=${EXIT_CODE}   $(date)"
echo "============================================================"
exit ${EXIT_CODE}