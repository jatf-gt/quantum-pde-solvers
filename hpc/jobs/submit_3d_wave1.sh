#!/bin/bash
# ============================================================================
#  submit_3d_wave1.sh   -   2nd-order 3D gap fill
#
#  Closes the largest confirmed hole in the archive: the three geometry-affected
#  HET cases have NO rows at any resolution, for any solver.
#
#      section 2  het_3d_mms_spt100          (3D_HET_MMS_SPT100)
#      section 3  het_3d_rotating_spoke      (3D_HET_RotatingSpoke_SPT100)
#      section 4  het_3d_discharge_spt100    (3D_HET_Discharge_SPT100)
#
#  Verified affected by the SPT-100 correction (commit 861ff46) using
#  scripts/check_geometry_impact.py --dim 3: the strip operator moves by 0.93 in
#  all three. Observe that het_3d_discharge_spt100's SOURCE is round-off identical -
#  its Gaussian is sited in normalised coordinates - so a source-only check would
#  have wrongly cleared it. Sections 1, 5, 6 and 7 are unaffected and are NOT
#  re-run here.
#
#  Why the previous attempt produced nothing
#  -----------------------------------------
#  submit_hpc_3D_geometry_fix.sh guarded on strip_sweep() exposing a `deadline`
#  parameter and aborted when it did not. That parameter genuinely did not exist -
#  the per-strip-solve wall-clock fix had been designed but never landed - so the
#  job exited immediately every time and the hole was never filled. The fix is now
#  in solvers/outer/core.py and the guard below passes; measured overshoot is
#  1.02-1.04x the budget, against ~1x a whole sweep before.
#
#  Ordering
#  --------
#  Resolutions ascend so the cheap, high-value rows land first: a walltime kill
#  costs only the expensive tail, and results are written incrementally per work
#  unit. Each step appends rather than overwriting.
#
#  Usage:
#      qsub hpc/jobs/submit_3d_wave1.sh
#      qsub -v HHL_MAX_WALL_S=14400 hpc/jobs/submit_3d_wave1.sh
# ============================================================================

#PBS -N qpde_3d_wave1
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -o results/3Dhpc_run/pbs_stdout_wave1.log
#PBS -e results/3Dhpc_run/pbs_stderr_wave1.log

set -u

echo "============================================================"
echo "  3D WAVE 1 - 2nd-order HET gap fill   $(date)"
echo "  Job ID: ${PBS_JOBID:-interactive}"
echo "============================================================"

cd "${PBS_O_WORKDIR:-$(pwd)}" || { echo "ERROR: cannot cd to PBS_O_WORKDIR"; exit 1; }

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
source "${HOME}/venvs/qpde/bin/activate" || { echo "ERROR: venv missing"; exit 1; }

# Refuses a dirty tree and a missing/upstream quantum_linear_solvers before any
# compute is dispatched. ORDER=2 here, so the pentadiagonal check is not required.
ORDER=2 bash hpc/jobs/_preflight.sh || exit 1

# The wall-clock cap is only meaningful if it can interrupt a sweep in progress.
python3 -c "
import inspect, sys
from solvers.outer.core import strip_sweep
if 'deadline' not in inspect.signature(strip_sweep).parameters:
    sys.exit('strip_sweep lacks the per-strip-solve deadline; cap would not bind.')
print('Per-strip-solve wall-clock cap confirmed.')
" || exit 1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p results/3Dhpc_run

SECTIONS="${SECTIONS:-2,3,4}"
WORKERS="${WORKERS:-4}"

# Budgets sized from `run_3d.py --estimate` against these exact sections, not
# guessed. Projected per case at N=16: HHL 10.5 h (MMS, spoke) and 15.0 h
# (discharge); VQLS 9.4 h and 13.4 h; QSVT 0.17-0.23 h. A uniform 6 h cap would
# therefore have truncated every N=16 HHL and VQLS row - precisely the rows this
# job exists to produce - and returned them flagged wall_time_exceeded.
SMALL_MAX_WALL_S="${SMALL_MAX_WALL_S:-3600}"    # 1 h; N<=8 needs ~0.05 h
N16_MAX_WALL_S="${N16_MAX_WALL_S:-57600}"       # 16 h; covers the 15.0 h worst case

# Superseded archives from the stripped rows would otherwise sit behind an absent
# row if a step fails, and be read by the plotting layer as though current.
echo "------------------------------------------------------------"
echo "Purging superseded archives for the three HET cases..."
rm -f results/3Dhpc_run/solution3d_3D_HET_MMS_SPT100_*.npz \
      results/3Dhpc_run/solution3d_3D_HET_RotatingSpoke_SPT100_*.npz \
      results/3Dhpc_run/solution3d_3D_HET_Discharge_SPT100_*.npz
echo "------------------------------------------------------------"

OVERALL=0

run_step () {
    local tag=$1 nvals=$2 solvers=$3 cap=$4
    echo ""
    echo "------------------------------------------------------------"
    echo "STEP ${tag}: N=${nvals} solvers=${solvers} cap=${cap}s -- $(date)"
    echo "------------------------------------------------------------"
    python3 hpc/runners/run_3d.py \
        --append --phase-tag "wave1_${tag}" \
        --n-values "${nvals}" \
        --sections "${SECTIONS}" \
        --solvers "${solvers}" \
        --max-workers "${WORKERS}" \
        -S "max_wall_s=${cap}"
    local rc=$?
    echo "STEP ${tag} finished $(date) exit=${rc}"
    [ "${rc}" -ne 0 ] && [ "${OVERALL}" -eq 0 ] && OVERALL=${rc}
    return 0
}

# QSVT across every resolution first, as insurance. It is projected at ~1.5 h for
# the whole set, so within the first couple of hours the sweep already holds a
# complete Thomas + QSVT picture of all three cases at every N. Whatever happens
# to the expensive solvers afterwards, the job cannot end with nothing.
run_step qsvt_all 4,8,16 qsvt "${N16_MAX_WALL_S}"

# Then the expensive solvers, cheapest resolutions first.
run_step hhl_vqls_small 4,8 hhl,vqls "${SMALL_MAX_WALL_S}"
run_step hhl_vqls_n16   16  hhl,vqls "${N16_MAX_WALL_S}"

echo ""
echo "============================================================"
echo "  Gap analysis after the run"
echo "============================================================"
python3 scripts/gap_analysis.py --dim 3 \
        --results-dir results/3Dhpc_run --n-values 4,8,16 \
        -o results/manifests/rerun_3d_after_wave1.json || true

echo ""
echo "3D WAVE 1 complete $(date)  overall exit=${OVERALL}"
exit ${OVERALL}
