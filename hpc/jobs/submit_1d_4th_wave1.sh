#!/bin/bash
# ============================================================================
#  submit_1d_4th_wave1.sh   -   4th-order 1D gap fill
#
#  Scope taken from results/manifests/rerun_1d_4th.json (16 outstanding of 96):
#
#      HET_1D_3b_gaussian_Vd300   N=4..32, all solvers   16 rows, geometry redo
#
#  Why this is 16 rows and not a full sweep
#  ----------------------------------------
#  scripts/check_geometry_impact.py --dim 1 proves that the SPT-100 correction
#  (861ff46) moves exactly one 1D case. The 4th-order operator is the
#  dimensionless pentadiagonal TST matrix, so kappa is geometry-independent;
#  L_z enters only the source amplitude. Of the HET 1D cases, only 3b sites
#  its Gaussian against the physical L_Z, and only 3b's source moves. Every
#  other case is round-off identical and must NOT be re-run.
#
#  N=64 is absent from the 4th-order sweep (phase precompute not available).
#  3c (Neumann-Dirichlet) is excluded_unimplemented under --order 4.
#
#  Cost
#  ----
#  Measured from the rows being replaced: Thomas ~0, QSVT ~23 s total, VQLS
#  ~933 s total, HHL 23+54+194+2551 s at N=4/8/16/32. Total ~1.1 h. A 6 h
#  walltime therefore carries ample margin.
#
#  Usage:
#      qsub hpc/jobs/submit_1d_4th_wave1.sh
# ============================================================================

#PBS -N qpde_1d4th_wave1
#PBS -l walltime=06:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -o results/1Dhpc_run_4th/pbs_stdout_wave1.log
#PBS -e results/1Dhpc_run_4th/pbs_stderr_wave1.log

set -u

echo "============================================================"
echo "  1D WAVE 1 - 4th-order gap fill   $(date)"
echo "  Job ID: ${PBS_JOBID:-interactive}"
echo "============================================================"

cd "${PBS_O_WORKDIR:-$(pwd)}" || { echo "ERROR: cannot cd to PBS_O_WORKDIR"; exit 1; }

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
source "${HOME}/venvs/qpde/bin/activate" || { echo "ERROR: venv missing"; exit 1; }

# Preflight: clean tree and pentadiagonal module present.
ORDER=4 bash hpc/jobs/_preflight.sh || exit 1

# --append and --cases are what make a 16-row run possible; without them the
# driver would rewrite results_full.json from the 16 rows alone and discard the
# other 80. Fail loudly here rather than discovering it from the output.
python3 -c "
import argparse, inspect, sys
sys.argv = ['run_1d.py']
sys.path.insert(0, '.')
src = open('hpc/runners/run_1d.py', encoding='utf-8').read()
for flag in ('--append', '--cases', '--n-values', '--solvers', '--phase-tag'):
    if flag not in src:
        sys.exit(f'run_1d.py does not accept {flag}; a full sweep would overwrite '
                 f'80 sound rows.')
print('Selective-rerun flags confirmed.')
" || exit 1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p results/1Dhpc_run_4th

WORKERS="${WORKERS:-4}"

OVERALL=0

run_step () {
    local tag=$1 nvals=$2 sections=$3 solvers=$4 cases=$5
    echo ""
    echo "------------------------------------------------------------"
    echo "STEP ${tag}: N=${nvals} sections=${sections} solvers=${solvers}"
    echo "            cases=${cases} -- $(date)"
    echo "------------------------------------------------------------"
    python3 hpc/runners/run_1d.py \
        --order 4 \
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

# Cheapest first. Sub-case 3b lives in section 2.
run_step het3b_small   4,8,16   2  hhl,vqls,qsvt  3b
run_step het3b_n32     32       2  hhl,vqls,qsvt  3b

echo ""
echo "============================================================"
echo "  Gap analysis after the run"
echo "============================================================"
python3 scripts/gap_analysis.py --dim 1 --order 4 \
        --results-dir results/1Dhpc_run_4th --n-values 4,8,16,32 \
        -o results/manifests/rerun_1d_4th_after_wave1.json || true

echo ""
echo "1D 4th-order WAVE 1 complete $(date)  overall exit=${OVERALL}"
exit ${OVERALL}
