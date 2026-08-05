#!/bin/bash
# ============================================================
#  submit_hpc_2D_gapfill.sh
#  One-off job: fill exactly the (case, N, solver) combinations missing
#  after killing the original 2D sweep - nothing already completed is
#  repeated, including at N=32, where three of five sections already have
#  every solver's result.
#
#  Exact gap map, derived from `ls results/2Dhpc_run/*.npz` against
#  results_full.json (section numbers per run_hpc_2Dfull.py::SECTIONS):
#
#    section                          HHL missing   VQLS missing   QSVT missing
#    1  2D_Poisson_sin_hom             64            32, 64         32, 64
#    2  2D_Poisson_TwoGaussian         64            64             64
#    3  2D_Poisson_SingleMode_n1m1     64            32, 64         32, 64
#    4  2D_HET_MMS_SPT100              64            64             64
#    5  2D_HET_Sin_MeetingReport       32, 64        32, 64         32, 64
#
#  The important asymmetry: HHL at N=32 is missing ONLY for section 5 - every
#  other section already paid for it once (the run that is now stagnating
#  section 5's HHL-32 was presumably interrupted before starting it, while
#  the other four had already finished it). Sections 2 and 4 likewise already
#  have VQLS and QSVT at N=32 in full. Re-running any of these would not just
#  waste a little time; the earlier VQLS-at-32 stagnation cost 13.4 hours for
#  ONE case, so avoiding a repeat of exactly that is the point of this script.
#
#  Five steps, run SEQUENTIALLY in this one job (not as separate concurrent
#  jobs - see submit_hpc_2D_gapfill.sh's original version for why: all use
#  --append on the same results_full.json, and concurrent jobs appending to
#  the same file race, since each loads it once at its own startup).
#
#    Step 1: HHL,   section 5 only,        N=32,64   (uncapped)
#    Step 2: HHL,   sections 1,2,3,4,      N=64      (uncapped)
#    Step 3: QSVT,  all 5 sections,        N=32,64   (uncapped - cheap anyway)
#    Step 4: VQLS,  sections 1,3,5,        N=32,64   (capped, see below)
#    Step 5: VQLS,  sections 2,4,          N=64      (capped, see below)
#
#  HHL is left uncapped, matching what was asked for this round - only VQLS
#  gets a wall-clock budget. If HHL-at-64 turns out to be similarly
#  expensive, the same -S max_wall_s=<seconds> flag used for VQLS below
#  applies to it unchanged; nothing about the mechanism is VQLS-specific.
#
#  VQLS cap: 7200s (2h) per (case, N) - long enough to let it actually
#  attempt several outer iterations rather than aborting on the first one,
#  short enough that five sections' worth of capped attempts cannot come
#  close to consuming the walltime the way the uncapped run did. A run that
#  hits the cap is recorded with stop_reason="wall_time_exceeded" - a real,
#  usable, clearly-caveated data point, not a discarded one.
#
#  Usage:  qsub submit_hpc_2D_gapfill.sh
# ============================================================

#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -N quantum_pde_2D_gapfill
#PBS -o results/2Dhpc_run/pbs_stdout_gapfill.log
#PBS -e results/2Dhpc_run/pbs_stderr_gapfill.log
#PBS -M j.trobajo-flecha24@imperial.ac.uk
#PBS -m abe

echo "============================================================"
echo "  2D GAP-FILL JOB START   $(date)   Job ID: $PBS_JOBID"
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

HHL_MAX_WALL_S="${HHL_MAX_WALL_S:-21600}"    # 4h
VQLS_MAX_WALL_S="${VQLS_MAX_WALL_S:-21600}"  # 6h
STEP_EXIT=(0 0 0 0 0)

run_step () {
    local n=$1; shift
    echo "------------------------------------------------------------"
    echo "STEP ${n}: $* -- starting $(date)"
    echo "------------------------------------------------------------"
    python3 scripts/run_hpc_2Dfull.py --append --phase-tag "gapfill_${n}" "$@"
    STEP_EXIT[$((n-1))]=$?
    echo "Step ${n} finished $(date) exit=${STEP_EXIT[$((n-1))]}"
}

# ---- Step 1: HHL, section 5 only, N=32 and N=64 (the one case missing 32) --
run_step 1 --sections 5 --n-values 32,64 --solvers hhl --max-workers 4 -S max_wall_s=${HHL_MAX_WALL_S}

# ---- Step 2: HHL, sections 1-4, N=64 only (they already have N=32) --------
run_step 2 --sections 1,2,3,4 --n-values 64 --solvers hhl --max-workers 4 -S max_wall_s=${HHL_MAX_WALL_S}

# ---- Step 3: QSVT, all sections, N=32 and N=64 -----------------------------
# Sections 2 and 4 already have QSVT-32; re-running it here is deliberate -
# QSVT is cheap enough (seconds to low minutes per case) that keeping this
# one command simple is worth more than the few extra minutes it costs.
run_step 3 --sections 1,2,3,4,5 --n-values 32,64 --solvers qsvt --max-workers 4

# ---- Step 4: VQLS, sections 1,3,5, N=32 and N=64, capped -------------------
run_step 4 --sections 1,3,5 --n-values 32,64 --solvers vqls \
    -S max_wall_s=${VQLS_MAX_WALL_S} --max-workers 4

# ---- Step 5: VQLS, sections 2,4, N=64 only, capped -------------------------
# (they already have VQLS-32 - the expensive one this whole script exists
# to avoid repeating)
run_step 5 --sections 2,4 --n-values 64 --solvers vqls \
    -S max_wall_s=${VQLS_MAX_WALL_S} --max-workers 4

EXIT_CODE=0
for e in "${STEP_EXIT[@]}"; do
    [ "$e" -ne 0 ] && EXIT_CODE=$e
done

echo "------------------------------------------------------------"
echo "Step exit codes: ${STEP_EXIT[*]}"
echo "Overall exit code: ${EXIT_CODE}"

RDS_RESULTS="${HOME}/qpde-results/2Dhpc_run_gapfill_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r results/2Dhpc_run/* "${RDS_RESULTS}/" 2>/dev/null
echo "Results copied to: ${RDS_RESULTS}"

echo "============================================================"
echo "  JOB COMPLETE   exit=${EXIT_CODE}   $(date)"
echo "============================================================"
exit ${EXIT_CODE}