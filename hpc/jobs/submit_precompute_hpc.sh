#!/bin/bash
# ============================================================
#  submit_precompute_hpc.sh
#  PBS Pro job submission script for Imperial College CX3 HPC.
#  Runs hpc/runners/precompute_phases.py as a batch job instead
#  of interactively, so it survives disconnects/idle timeouts and
#  isn't limited by an interactive session's own wall-clock cap.
#
#  N_VALUES and MAX_DEGREE are read from the environment rather than
#  hardcoded, so the SAME script covers a staged rollout (small N
#  first, larger N as separate, later submissions) without editing
#  the file. IMPORTANT: pass them via `qsub -v NAME` (bare name,
#  value taken from your shell's exported variable) rather than
#  `qsub -v NAME=value` -- PBS's own -v parser splits on commas in
#  the value list, which would break a comma-separated N list like
#  "4,8,16" if embedded directly after an "=".
#
#  Usage
#  -----
#    # Stage 1 -- small N, expected safe (see precompute script docstring):
#    export N_VALUES="4,8,16"
#    qsub -v N_VALUES hpc/jobs/submit_precompute_hpc.sh
#
#    # Stage 2 -- N=32, exploratory, capped, separate job/log:
#    export N_VALUES="32"
#    export MAX_DEGREE="2000"
#    qsub -v N_VALUES,MAX_DEGREE hpc/jobs/submit_precompute_hpc.sh
#
#    # Stage 3 -- N=64, same idea, only after stage 2 is confirmed to work:
#    export N_VALUES="64"
#    export MAX_DEGREE="2000"
#    qsub -v N_VALUES,MAX_DEGREE hpc/jobs/submit_precompute_hpc.sh
#
#  Each stage writes into the SAME cache directory
#  (results/qsvt_phase_cache/), so results accumulate across stages;
#  nothing from an earlier stage is touched or re-run by a later one.
#
#  Monitor:
#    qstat -u $USER
#    tail -f results/qsvt_phase_precompute_pbs.log
#
#  Reference: https://icl-rcs-user-guide.readthedocs.io/
# ============================================================

# --- Resource requests ---
# Single-threaded (qsp_angles.py does not parallelise across cores).
# mem=32gb is generous headroom for N<=32 at modest --max-degree; if you
# push N=64 or leave --max-degree uncapped, request more (see the
# per-N memory notes from the earlier analysis -- the Newton solver's
# working array is O(degree^2)).
# walltime=71:00:00 stays just under CX3's 72h queue cap; N=32/64 are
# NOT guaranteed to finish inside it (see the precompute script's own
# docstring caveat about PolyOneOverX.generate() cost) -- if a stage
# doesn't finish, whatever it completed before being killed is already
# safe on disk (see script header), so it's fine to just resubmit the
# same stage and let it skip what's already cached.
#PBS -l walltime=71:00:00
#PBS -l select=1:ncpus=1:mem=32gb

# --- Job metadata ---
#PBS -N qsvt_precompute
#PBS -o results/qsvt_phase_precompute_pbs.log
#PBS -e results/qsvt_phase_precompute_pbs.err

# --- Email notifications (replace with your Imperial email if different) ---
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

# ============================================================
#  Environment setup
# ============================================================

echo "============================================================"
echo "  QSVT PHASE PRECOMPUTE — HPC JOB START"
echo "  Job ID    : $PBS_JOBID"
echo "  Node      : $(hostname)"
echo "  Date/Time : $(date)"
echo "  Work dir  : $PBS_O_WORKDIR"
echo "  N_VALUES  : ${N_VALUES:-<not set, script default: 4,8,16>}"
echo "  MAX_DEGREE: ${MAX_DEGREE:-<not set, uncapped>}"
echo "============================================================"

# ── Repository root resolution ───────────────────────────────
# PBS copies this script to a spool directory before executing it, so $0 and
# BASH_SOURCE do NOT point at the original file. PBS_O_WORKDIR -- the directory
# qsub was invoked from -- is the only reliable anchor. Ascending from it means
# both `qsub hpc/<script>` (from the repo root) and `cd hpc && qsub <script>`
# resolve correctly.
REPO_ROOT="${PBS_O_WORKDIR}"
while [ ! -f "${REPO_ROOT}/pyproject.toml" ] && [ "${REPO_ROOT}" != "/" ]; do
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
if [ ! -f "${REPO_ROOT}/pyproject.toml" ]; then
    echo "ERROR: no repository root (pyproject.toml) at or above ${PBS_O_WORKDIR}."
    echo "       Submit from inside a clone, e.g. qsub hpc/$(basename "$0")"
    exit 1
fi
cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}"; exit 1; }

# The #PBS -o/-e paths above are resolved by PBS at submission time, relative to
# the submission directory; no shell logic here can redirect them. Submitting
# from the repository root keeps the PBS logs alongside the results.
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

# pyqsp is in requirements.txt but is not installed by hpc/setup_hpc_env.sh's
# explicit package list -- guard against that gap rather than fail hours
# into a job on a missing import.
python3 -c "import pyqsp" 2>/dev/null || {
    echo "pyqsp not found in venv; installing..."
    pip install pyqsp==0.2.0
}

mkdir -p results/qsvt_phase_cache

# ============================================================
#  Run the precompute
# ============================================================

EXTRA_ARGS=""
if [ -n "${N_VALUES}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --n-values ${N_VALUES}"
fi
if [ -n "${MAX_DEGREE}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --max-degree ${MAX_DEGREE}"
fi

echo "Starting precompute at $(date)"
echo "Arguments: ${EXTRA_ARGS}"
echo "------------------------------------------------------------"

python3 hpc/runners/precompute_phases.py --dim 1 ${EXTRA_ARGS}
EXIT_CODE=$?

echo "------------------------------------------------------------"
echo "Precompute finished at $(date) with exit code ${EXIT_CODE}"

# ============================================================
#  Copy the cache to permanent RDS storage (belt-and-suspenders --
#  results/ should already be on RDS if PBS_O_WORKDIR is, but this
#  matches the pattern used in hpc/jobs/submit_hpc.sh)
# ============================================================
RDS_CACHE="${HOME}/qpde-results/qsvt_phase_cache_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_CACHE}"
cp -r results/qsvt_phase_cache/* "${RDS_CACHE}/" 2>/dev/null
echo "Cache copied to: ${RDS_CACHE}"

echo "============================================================"
echo "  JOB COMPLETE"
echo "  Exit code : ${EXIT_CODE}"
echo "  Date/Time : $(date)"
echo "============================================================"

exit ${EXIT_CODE}