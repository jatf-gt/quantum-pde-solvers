#!/bin/bash
# ============================================================================
#  submit_precompute_4th.sh
#  QSVT phase-angle precompute for the FOURTH-ORDER (pentadiagonal) operator.
#
#  Companion to submit_precompute_hpc.sh (2nd order, 1-D) and
#  submit_precompute_2D.sh (2nd order, 2-D). Runs
#  `hpc/runners/precompute_phases.py --dim 1 --order 4`.
#
#  Why a separate job, and why 1-D only
#  ------------------------------------
#  The phase angles depend on kappa alone, and the pentadiagonal operator has a
#  different kappa from the tridiagonal one at the same N:
#
#      N     kappa (order 2)   kappa (order 4)   ratio
#      4          9.4721           11.9477       1.261
#      8         32.1634           42.1378       1.310
#      16       116.4612          154.5126       1.327
#      32       440.6886          586.8093       1.331
#
#  The ratio converges to 4/3, NOT to the "2.5x" quoted in qsvt_1d_4th.py -- that
#  figure is the ratio of spectral norms (30/12) and has nothing to do with the
#  condition number. The 4th-order set is therefore only moderately more expensive
#  than the 2nd-order set at equal N, not 2.5 times.
#
#  These kappas are distinct cache keys, so the existing 2nd-order cache does not
#  serve them: without this job every 4th-order QSVT strip solve recomputes its
#  phases inline, which is the single most expensive non-parallelisable step in the
#  pipeline.
#
#  2-D and 3-D are deliberately NOT covered here. The 2-D/3-D 4th-order strip
#  operator is unsettled: the boundary closure in solvers/outer/multigrid_4th.py is
#  first-order accurate (measured convergence 0.85-0.99 against a manufactured
#  solution, where 4 is intended), so its kappa is about to change. A cache entry
#  written against it would be keyed to a superseded operator and would silently
#  miss at runtime, relocating the expensive computation into the sweep -- exactly
#  the failure the cache exists to prevent. precompute_phases.py refuses
#  `--dim 2 --order 4` for this reason. See docs/HPC_REPAIR_PLAN.md, Phase 4.
#
#  Cost and staging
#  ----------------
#  Measured locally: N=4 (kappa 11.95) did not complete within 10 minutes on a
#  laptop core, which is why this is a batch job rather than an interactive step.
#  Cost grows steeply with kappa, so stage smallest-N-first and treat each stage as
#  independent. Whatever a killed stage completed is already on disk -- the cache is
#  written per (kappa, epsilon) pair, not at the end -- so resubmitting the same
#  stage skips what is already cached.
#
#  Usage
#  -----
#    # Stage 1 -- N=4,8. Start here.
#    export N_VALUES="4,8"
#    qsub -v N_VALUES hpc/jobs/submit_precompute_4th.sh
#
#    # Stage 2 -- N=16, only once stage 1 has completed.
#    export N_VALUES="16"
#    qsub -v N_VALUES hpc/jobs/submit_precompute_4th.sh
#
#    # Confirm the keys before committing to a long job (computes nothing):
#    python3 hpc/runners/precompute_phases.py --dim 1 --order 4 \
#            --n-values 4,8,16 --list-kappas
#
#  N_VALUES and MAX_DEGREE must be passed as `qsub -v NAME` (bare name, value from
#  the exported shell variable). `-v NAME=value` breaks on PBS's comma splitting
#  for a comma-separated list.
#
#  Monitor:
#    qstat -u $USER
#    tail -f results/qsvt_phase_precompute_4th_pbs.log
# ============================================================================

# --- Resource requests ---
# Single-threaded: qsp_angles.py does not parallelise across cores.
# The Newton solver's working array is O(degree^2), hence the generous memory.
# walltime stays just under CX3's 72 h queue cap. N=16 is not guaranteed to finish
# inside it; a killed stage loses nothing already cached.
#PBS -l walltime=71:00:00
#PBS -l select=1:ncpus=1:mem=32gb

# --- Job metadata ---
#PBS -N qsvt_precompute_4th
#PBS -o results/qsvt_phase_precompute_4th_pbs.log
#PBS -e results/qsvt_phase_precompute_4th_pbs.err

#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

echo "============================================================"
echo "  QSVT PHASE PRECOMPUTE (4th order) — HPC JOB START"
echo "  Job ID    : $PBS_JOBID"
echo "  Node      : $(hostname)"
echo "  Date/Time : $(date)"
echo "  Work dir  : $PBS_O_WORKDIR"
echo "  N_VALUES  : ${N_VALUES:-<not set, script default: 4,8>}"
echo "  MAX_DEGREE: ${MAX_DEGREE:-5000 (matching the 2nd-order 1-D cache)}"
echo "============================================================"

# ── Repository root resolution ───────────────────────────────
# PBS copies this script to a spool directory before executing it, so $0 and
# BASH_SOURCE do NOT point at the original file. PBS_O_WORKDIR -- the directory
# qsub was invoked from -- is the only reliable anchor.
REPO_ROOT="${PBS_O_WORKDIR}"
while [ ! -f "${REPO_ROOT}/pyproject.toml" ] && [ "${REPO_ROOT}" != "/" ]; do
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
if [ ! -f "${REPO_ROOT}/pyproject.toml" ]; then
    echo "ERROR: no repository root (pyproject.toml) at or above ${PBS_O_WORKDIR}."
    echo "       Submit from inside a clone, e.g. qsub hpc/jobs/$(basename "$0")"
    exit 1
fi
cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}"; exit 1; }

if [ "${PBS_O_WORKDIR}" != "${REPO_ROOT}" ]; then
    echo "NOTE: submitted from ${PBS_O_WORKDIR}, not the repository root"
    echo "      (${REPO_ROOT}). The PBS stdout/stderr logs are under the former."
fi

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

VENV_PATH="${HOME}/venvs/qpde"
if [ ! -d "${VENV_PATH}" ]; then
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    echo "       See hpc/setup_hpc_env.sh."
    exit 1
fi
source "${VENV_PATH}/bin/activate"
echo "Python: $(which python3) -- $(python3 --version)"

# ORDER=4 checks the pentadiagonal module, which this job does not itself need but
# whose absence means the phases being computed cannot be used afterwards. Better
# to discover that now than after 71 h.
ORDER=4 bash hpc/jobs/_preflight.sh || exit 1

# pyqsp supplies PolyOneOverX; without it qsp_angles falls back to a Chebyshev
# construction of markedly lower quality, which is not what should populate a cache.
python3 -c "import pyqsp" 2>/dev/null || {
    echo "pyqsp not found in venv; installing..."
    pip install pyqsp==0.2.0
}
python3 -c "import pyqsp" || {
    echo "ERROR: pyqsp still unavailable. Refusing to populate the cache with"
    echo "       Chebyshev-fallback angles, which would be silently inferior to"
    echo "       every other entry in it."
    exit 1
}

mkdir -p results/qsvt_phase_cache

# ============================================================
#  Precompute
# ============================================================
N_VALUES="${N_VALUES:-4,8}"
MAX_DEGREE="${MAX_DEGREE:-5000}"

echo ""
echo "Computing 1-D order-4 phases for N=${N_VALUES}, max_degree=${MAX_DEGREE}"
echo "Cache keys that will be written:"
python3 hpc/runners/precompute_phases.py --dim 1 --order 4 \
        --n-values "${N_VALUES}" --list-kappas

echo ""
python3 hpc/runners/precompute_phases.py \
        --dim 1 --order 4 \
        --n-values "${N_VALUES}" \
        --max-degree "${MAX_DEGREE}"
RC=$?

echo ""
echo "============================================================"
echo "  Cache contents after this stage"
echo "============================================================"
ls -1 results/qsvt_phase_cache/ | wc -l | xargs echo "  entries:"

echo ""
echo "QSVT 4th-order precompute finished $(date)  exit=${RC}"
exit ${RC}
