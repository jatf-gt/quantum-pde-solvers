#!/bin/bash
# ============================================================
#  submit_hpc_1D_geometry_fix.sh
#  Full re-run of the 1-D sweep after the SPT-100 geometry correction
#  (core/het_geometry.py). run_1d.py has no --append/--sections/-S flags -
#  every run is a full, fresh sweep that overwrites its results cleanly, so
#  there is no stale-row cleanup step needed here (unlike 2D/3D).
#
#  No wall-time cap: 1-D has no outer multigrid iteration at all - each case
#  is a single direct solve on the full matrix, not a strip-decomposed
#  sweep. At N=64 that is one HHL/VQLS/QSVT call, not dozens across a
#  hierarchy, so the runaway-cost mechanism behind the 2-D/3-D caps doesn't
#  apply here. If a single 1-D N=64 solve does turn out to be
#  unexpectedly expensive, that will show up directly and quickly in this
#  run's own log rather than compounding across an iteration.
#
#  Usage:  qsub hpc/jobs/submit_hpc_1D_geometry_fix.sh
# ============================================================

#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -N quantum_pde_1D_geometry_fix
#PBS -o results/1Dhpc_run/pbs_stdout_geomfix.log
#PBS -e results/1Dhpc_run/pbs_stderr_geomfix.log
#PBS -M j.trobajo-flecha24@imperial.ac.uk
#PBS -m abe

echo "============================================================"
echo "  1D GEOMETRY-FIX RE-RUN   $(date)   Job ID: $PBS_JOBID"
echo "============================================================"

cd "${PBS_O_WORKDIR}" || { echo "ERROR: cannot cd to PBS_O_WORKDIR"; exit 1; }

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

VENV_PATH="${HOME}/venvs/qpde"
[ -d "${VENV_PATH}" ] || { echo "ERROR: venv not found at ${VENV_PATH}"; exit 1; }
source "${VENV_PATH}/bin/activate"
echo "Python: $(which python3) -- $(python3 --version)"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
mkdir -p results/1Dhpc_run

echo "------------------------------------------------------------"
echo "Full 1-D sweep, all cases, starting $(date)"
echo "------------------------------------------------------------"
python3 hpc/runners/run_1d.py --max-workers 4
EXIT_CODE=$?
echo "Finished $(date) exit=${EXIT_CODE}"

RDS_RESULTS="${HOME}/qpde-results/1Dhpc_run_geomfix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r results/1Dhpc_run/* "${RDS_RESULTS}/" 2>/dev/null
echo "Results copied to: ${RDS_RESULTS}"

echo "============================================================"
echo "  JOB COMPLETE   exit=${EXIT_CODE}   $(date)"
echo "============================================================"
exit ${EXIT_CODE}