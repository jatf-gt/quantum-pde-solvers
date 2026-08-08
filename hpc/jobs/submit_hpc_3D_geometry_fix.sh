#!/bin/bash
# ============================================================
#  submit_hpc_3D_geometry_fix.sh
#  Re-runs the three 3D HET cases affected by the SPT-100 geometry
#  correction, at N=4,8,16 (N=32 was never reached for these cases even
#  before the kill, so there is nothing stale to clean up there).
#
#  Affected: het_3d_mms_spt100 (section 2), het_3d_rotating_spoke
#  (section 3), het_3d_discharge_spt100 (section 4). Unaffected (verified):
#  cube/TripleSin (1), Laplace (5), TwoGaussian_cube (6), HighMode (7).
#
#  See submit_hpc_2D_geometry_fix.sh for the full rationale behind the
#  cleanup step and the corrected (per-strip-solve) wall-time fix - identical
#  here, just for the 3D runner and case set.
#
#  Usage:  qsub hpc/jobs/submit_hpc_3D_geometry_fix.sh
# ============================================================

#PBS -l walltime=36:00:00
#PBS -l select=1:ncpus=4:mem=64gb
#PBS -N quantum_pde_3D_geometry_fix
#PBS -o results/3Dhpc_run/pbs_stdout_geomfix.log
#PBS -e results/3Dhpc_run/pbs_stderr_geomfix.log
#PBS -M j.trobajo-flecha24@imperial.ac.uk
#PBS -m abe

echo "============================================================"
echo "  3D GEOMETRY-FIX RE-RUN   $(date)   Job ID: $PBS_JOBID"
echo "============================================================"

cd "${PBS_O_WORKDIR}" || { echo "ERROR: cannot cd to PBS_O_WORKDIR"; exit 1; }

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

VENV_PATH="${HOME}/venvs/qpde"
[ -d "${VENV_PATH}" ] || { echo "ERROR: venv not found at ${VENV_PATH}"; exit 1; }
source "${VENV_PATH}/bin/activate"
echo "Python: $(which python3) -- $(python3 --version)"

python3 -c "
import inspect
from solvers.outer.core import strip_sweep
sig = inspect.signature(strip_sweep)
assert 'deadline' in sig.parameters, (
    'solvers/outer/core.py does not have the per-strip-solve wall-time fix.')
print('Fine-grained wall-time fix confirmed present in strip_sweep.')
" || exit 1

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
mkdir -p results/3Dhpc_run

echo "------------------------------------------------------------"
echo "Removing stale rows for geometry-affected cases..."
python3 scripts/cleanup_stale_geometry.py results/3Dhpc_run/results_full.json
echo "------------------------------------------------------------"

HHL_MAX_WALL_S="${HHL_MAX_WALL_S:-28800}"    # 8h
VQLS_MAX_WALL_S="${VQLS_MAX_WALL_S:-21600}"  # 6h
STEP_EXIT=(0 0)

run_step () {
    local n=$1; shift
    echo "------------------------------------------------------------"
    echo "STEP ${n}: $* -- starting $(date)"
    echo "------------------------------------------------------------"
    python3 hpc/runners/run_3d.py --append --phase-tag "geomfix_${n}" "$@"
    STEP_EXIT[$((n-1))]=$?
    echo "Step ${n} finished $(date) exit=${STEP_EXIT[$((n-1))]}"
}

# ---- Step 1: HHL, sections 2,3,4, N=4,8,16, capped 8h -----------------------
run_step 1 --sections 2,3,4 --n-values 4,8,16 --solvers hhl \
    -S max_wall_s=${HHL_MAX_WALL_S} --max-workers 4

# ---- Step 2: VQLS + QSVT, sections 2,3,4, N=4,8,16, capped 6h --------------
run_step 2 --sections 2,3,4 --n-values 4,8,16 --solvers vqls,qsvt \
    -S max_wall_s=${VQLS_MAX_WALL_S} --max-workers 4

EXIT_CODE=0
for e in "${STEP_EXIT[@]}"; do
    [ "$e" -ne 0 ] && EXIT_CODE=$e
done

echo "------------------------------------------------------------"
echo "Step exit codes: ${STEP_EXIT[*]}"
echo "Overall exit code: ${EXIT_CODE}"

RDS_RESULTS="${HOME}/qpde-results/3Dhpc_run_geomfix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r results/3Dhpc_run/* "${RDS_RESULTS}/" 2>/dev/null
echo "Results copied to: ${RDS_RESULTS}"

echo "============================================================"
echo "  JOB COMPLETE   exit=${EXIT_CODE}   $(date)"
echo "============================================================"
exit ${EXIT_CODE}