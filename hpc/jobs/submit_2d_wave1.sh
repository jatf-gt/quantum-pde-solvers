#!/bin/bash
# ============================================================================
#  submit_2d_wave1.sh   -   2nd-order 2D gap fill
#
#  Scope taken from scripts/utils/gap_analysis.py against the current archive
#  (55 outstanding of 100 expected):
#
#      sections 4,5  N=4..64   geometry redo - no rows at any N
#                              (2D_HET_MMS_SPT100, 2D_HET_Sin_MeetingReport)
#      sections 1,2,3  N=64    never reached before the original 48 h job
#                              hit its walltime at section2/N=64
#      section 3       N=32    partial - three solvers missing
#
#  Sections 1, 2 and 3 at N<=16 are sound and are NOT re-run. Neither are the 22
#  rows that stagnated: stagnation is the designed terminal state for a quantum
#  solver sitting at its inner-solver noise floor, not a failure, and re-running
#  reproduces it at the same cost.
#
#  Cost warning
#  ------------
#  2D HHL at N=64 is the most expensive cell in the whole project - one recorded
#  row took 38.6 h under the old, ineffective cap. The cap now binds to within one
#  strip solve, so HHL_MAX_WALL_S is a real bound rather than an aspiration. Run
#  the --estimate step below before trusting the walltime request.
#
#  Ordering is by ascending cost, so a walltime kill loses only the tail.
#
#  Usage:
#      qsub hpc/jobs/submit_2d_wave1.sh
#      qsub -v SKIP_N64=1 hpc/jobs/submit_2d_wave1.sh    # cheap portion only
# ============================================================================

#PBS -N qpde_2d_wave1
#PBS -l walltime=60:00:00
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -o results/2Dhpc_run/pbs_stdout_wave1.log
#PBS -e results/2Dhpc_run/pbs_stderr_wave1.log

set -u

echo "============================================================"
echo "  2D WAVE 1 - 2nd-order gap fill   $(date)"
echo "  Job ID: ${PBS_JOBID:-interactive}"
echo "============================================================"

cd "${PBS_O_WORKDIR:-$(pwd)}" || { echo "ERROR: cannot cd to PBS_O_WORKDIR"; exit 1; }

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
source "${HOME}/venvs/qpde/bin/activate" || { echo "ERROR: venv missing"; exit 1; }

ORDER=2 bash hpc/jobs/_preflight.sh || exit 1

python3 -c "
import inspect, sys
from solvers.outer.core import strip_sweep
if 'deadline' not in inspect.signature(strip_sweep).parameters:
    sys.exit('strip_sweep lacks the per-strip-solve deadline; cap would not bind.')
print('Per-strip-solve wall-clock cap confirmed.')
" || exit 1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p results/2Dhpc_run

SOLVERS="${SOLVERS:-hhl,vqls,qsvt}"
WORKERS="${WORKERS:-4}"
MAX_WALL_S="${MAX_WALL_S:-21600}"     # 6 h per case
SKIP_N64="${SKIP_N64:-0}"

echo "------------------------------------------------------------"
echo "Purging superseded archives for the two geometry-affected cases..."
rm -f results/2Dhpc_run/solutions_2D_HET_MMS_SPT100_*.npz \
      results/2Dhpc_run/solutions_2D_HET_Sin_MeetingReport_*.npz
echo "------------------------------------------------------------"

OVERALL=0

run_step () {
    local tag=$1 sections=$2 nvals=$3
    echo ""
    echo "------------------------------------------------------------"
    echo "STEP ${tag}: sections=${sections} N=${nvals} -- $(date)"
    echo "------------------------------------------------------------"
    python3 hpc/runners/run_2d.py \
        --append --phase-tag "wave1_${tag}" \
        --n-values "${nvals}" \
        --sections "${sections}" \
        --solvers "${SOLVERS}" \
        --max-workers "${WORKERS}" \
        -S "max_wall_s=${MAX_WALL_S}"
    local rc=$?
    echo "STEP ${tag} finished $(date) exit=${rc}"
    [ "${rc}" -ne 0 ] && [ "${OVERALL}" -eq 0 ] && OVERALL=${rc}
    return 0
}

# Cheapest first: the entire HET redo below N=32 costs less than a single
# N=64 HHL cell.
run_step het_small 4,5   4,8,16
run_step s3_n32    3     32
run_step het_n32   4,5   32

if [ "${SKIP_N64}" = "1" ]; then
    echo "SKIP_N64=1 - stopping before the N=64 steps."
else
    run_step n64_generic 1,2,3 64
    run_step het_n64     4,5   64
fi

echo ""
echo "============================================================"
echo "  Gap analysis after the run"
echo "============================================================"
python3 scripts/utils/gap_analysis.py --dim 2 \
        --results-dir results/2Dhpc_run --n-values 4,8,16,32,64 \
        -o results/manifests/rerun_2d_after_wave1.json || true

echo ""
echo "2D WAVE 1 complete $(date)  overall exit=${OVERALL}"
exit ${OVERALL}
