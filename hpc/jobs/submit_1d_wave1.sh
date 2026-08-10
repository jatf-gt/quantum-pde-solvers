#!/bin/bash
# ============================================================================
#  submit_1d_wave1.sh   -   2nd-order 1D gap fill
#
#  Scope taken from results/manifests/rerun_1d.json (22 outstanding of 140):
#
#      HET_1D_3b_gaussian_Vd300   N=4..64, all solvers   20 rows, geometry redo
#      HET_1D_3c_gaussian_...     N=8,32, HHL only        2 rows, genuine failure
#
#  Why this is 22 rows and not a full sweep
#  ----------------------------------------
#  scripts/check_geometry_impact.py --dim 1 proves that the SPT-100 correction
#  (861ff46) moves exactly one 1D case. The 1D operator is the dimensionless TST
#  matrix, so kappa is geometry-independent; L_z enters only the source amplitude.
#  Of the eight het_1d_* cases, only 3b sites its Gaussian against the physical
#  L_Z, and only 3b's source moves (by 3.35e-01). Every other 1D case - including
#  the whole *_scaled family, which normalises L out - is round-off identical and
#  must NOT be re-run.
#
#  Why the thirteen HHL "solver_error" rows are NOT here
#  ----------------------------------------------------
#  They are not errors. run_1d.py imposes a hard 3600 s per-solve budget on HHL,
#  and at N=32/64 the 1D operator reaches kappa ~ 1.7e3, growing HHL's clock
#  register past what statevector simulation completes inside that budget. All
#  thirteen record wall_time_s = 3600.2 exactly. That is the benchmark's finding,
#  and re-running them would spend 13 h reproducing thirteen identical timeouts.
#  The runner now records "hhl_timeout" rather than the generic "solver_error" so
#  the two are no longer conflated, and gap_analysis.py treats a timeout as a
#  terminal measurement. The two 3c rows retained above failed differently: N=32
#  died after 743 s, well inside the budget.
#
#  Cost
#  ----
#  Measured from the rows being replaced: Thomas ~0, QSVT ~27 s total, VQLS
#  ~1740 s total, HHL 52+42+92 s at N=4/8/16 and 3600 s each at N=32/64. Total
#  ~2.7 h, of which 2 h is HHL at N=32/64 re-confirming its timeout under the
#  corrected source. A 6 h walltime therefore carries ample margin.
#
#  Usage:
#      qsub hpc/jobs/submit_1d_wave1.sh
#      qsub -v SKIP_HHL_LARGE=1 hpc/jobs/submit_1d_wave1.sh   # omit the 2 h of
#                                                             # known timeouts
# ============================================================================

#PBS -N qpde_1d_wave1
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o results/1Dhpc_run/pbs_stdout_wave1.log
#PBS -e results/1Dhpc_run/pbs_stderr_wave1.log

set -u

echo "============================================================"
echo "  1D WAVE 1 - 2nd-order gap fill   $(date)"
echo "  Job ID: ${PBS_JOBID:-interactive}"
echo "============================================================"

cd "${PBS_O_WORKDIR:-$(pwd)}" || { echo "ERROR: cannot cd to PBS_O_WORKDIR"; exit 1; }

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
source "${HOME}/venvs/qpde/bin/activate" || { echo "ERROR: venv missing"; exit 1; }

# Refuses a dirty tree and a missing/upstream quantum_linear_solvers before any
# compute is dispatched. ORDER=2 here, so the pentadiagonal check is not required.
ORDER=2 bash hpc/jobs/_preflight.sh || exit 1

# --append and --cases are what make a 22-row run possible; without them the
# driver would rewrite results_full.json from the 22 rows alone and discard the
# other 118. Fail loudly here rather than discovering it from the output.
python3 -c "
import argparse, inspect, sys
sys.argv = ['run_1d.py']
sys.path.insert(0, '.')
src = open('hpc/runners/run_1d.py', encoding='utf-8').read()
for flag in ('--append', '--cases', '--n-values', '--solvers', '--phase-tag'):
    if flag not in src:
        sys.exit(f'run_1d.py does not accept {flag}; a full sweep would overwrite '
                 f'118 sound rows.')
print('Selective-rerun flags confirmed.')
" || exit 1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p results/1Dhpc_run

WORKERS="${WORKERS:-4}"
SKIP_HHL_LARGE="${SKIP_HHL_LARGE:-0}"

OVERALL=0

run_step () {
    local tag=$1 nvals=$2 sections=$3 solvers=$4 cases=$5
    echo ""
    echo "------------------------------------------------------------"
    echo "STEP ${tag}: N=${nvals} sections=${sections} solvers=${solvers}"
    echo "            cases=${cases} -- $(date)"
    echo "------------------------------------------------------------"
    python3 hpc/runners/run_1d.py \
        --append --phase-tag "wave1_${tag}" \
        --n-values "${nvals}" \
        --sections "${sections}" \
        --solvers "${solvers}" \
        --cases "${cases}" \
        --max-workers "${WORKERS}"
    local rc=$?
    echo "STEP ${tag} finished $(date) exit=${rc}"
    [ "${rc}" -ne 0 ] && [ "${OVERALL}" -eq 0 ] && OVERALL=${rc}
    return 0
}

# Cheapest first, so a walltime kill loses only the tail. Sub-case 3b lives in
# section 2, and --cases 3b excludes 3a and 3c from that section.
run_step het3b_small  4,8,16   2  hhl,vqls,qsvt  3b
run_step het3b_n32    32       2  vqls,qsvt      3b
run_step het3b_n64    64       2  vqls,qsvt      3b

# The two genuine 3c HHL failures. N=8 is cheap; N=32 previously died after 743 s.
run_step het3c_hhl    8,32     2  hhl            3c

if [ "${SKIP_HHL_LARGE}" = "1" ]; then
    echo ""
    echo "SKIP_HHL_LARGE=1 - omitting HHL at N=32,64 for case 3b."
    echo "Those two rows will remain outstanding in the manifest."
else
    # Expected to exhaust the 3600 s budget at both resolutions and record
    # hhl_timeout. Run last, and only because the rows currently on disk were
    # measured against the superseded source term: the outcome is known, the
    # provenance is what is being corrected.
    run_step het3b_hhl_large  32,64  2  hhl  3b
fi

echo ""
echo "============================================================"
echo "  Gap analysis after the run"
echo "============================================================"
python3 scripts/gap_analysis.py --dim 1 \
        --results-dir results/1Dhpc_run --n-values 4,8,16,32,64 \
        -o results/manifests/rerun_1d_after_wave1.json || true

echo ""
echo "1D WAVE 1 complete $(date)  overall exit=${OVERALL}"
exit ${OVERALL}
