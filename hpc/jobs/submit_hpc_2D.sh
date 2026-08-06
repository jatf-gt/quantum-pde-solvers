#!/bin/bash
# ============================================================
#  submit_hpc_2D.sh
#  PBS Pro job submission script for Imperial College CX3 HPC.
#
#  Runs the full 2-D benchmark sweep across all five sections:
#    Section 1: Generic Poisson, sinusoidal source
#    Section 2: Two-Gaussian PlasmaNet benchmark
#    Section 3: Single-mode Fourier source
#    Section 4: HET MMS manufactured solution (SPT-100)
#    Section 5: HET sinusoidal source (meeting-report case)
#
#  TWO PHASES, one job, one shared results directory:
#
#    Phase 1 (core)  N = 4, 8, 16, 32, 64      all solvers: Thomas, HHL, VQLS, QSVT
#    Phase 2 (large) N = 128, 256              QSVT only
#
#  Phase 2 uses QSVT only because it is by far the cheapest inner solver:
#  circuit depth is O(kappa) rather than the O(kappa^2) of HHL, and its
#  measured per-strip cost exponent is alpha~0.6 versus ~2.35 (HHL) and
#  ~1.29 (VQLS) - see run_hpc_2Dfull.py::COST_ALPHA. Running HHL or VQLS
#  at N=256 is not a matter of a longer walltime, it is not practical.
#
#  Both phases use the outer scheme set by OUTER_SCHEME (default: fmg, full
#  multigrid), the fastest scheme implemented: O(1) outer cycles versus the
#  O(N) sweeps of line-SOR or the original line-Jacobi. Using the same
#  scheme in both phases keeps every N in the run directly comparable.
#
#  QSVT_MAX_DEGREE (default 500) is passed explicitly to BOTH phases, rather
#  than relying on the per-N default table inside run_hpc_2Dfull.py, so the
#  run is fully self-documenting and reproducible from this script alone.
#
#  Theoretical justification for max_degree=500 up to N=256:
#  kappa(A_row) for this discretisation (dx=dy) satisfies
#      kappa = (2 + cos(theta_max)) / (2 - cos(theta_max)) < 3   for all N,
#  approaching 3 only as N -> infinity (N=256 gives kappa=2.9997, N=64 gives
#  2.9953; the HET cases, with dz != dr, have a lower asymptote, ~2.28). The
#  code's own degree estimate (solvers/quantum/qsp_angles.py,
#  polynomial_degree_estimate) is
#      d = ceil(13 * kappa * ln(kappa / epsilon)).
#  At the worst case kappa=3 and the default QSVT epsilon=0.01, that gives
#  d ~ 223 - under half the cap, with more than 2x margin, at every N this
#  script runs. The cap would only start to bind if epsilon fell below
#  ~1e-5, three orders of magnitude past the default. It is a safety
#  ceiling here, not an accuracy control, at any N reachable in this sweep.
#
#  PREREQUISITE: run the 2D phase precompute before this job.
#    qsub hpc/jobs/submit_precompute_2D.sh
#  Wait for it to complete, then:
#    qsub hpc/jobs/submit_hpc_2D.sh
#
#  Usage:
#    # Full two-phase sweep (default):
#    qsub hpc/jobs/submit_hpc_2D.sh
#
#    # Fast local validation pass (Phase 1 only, N=4, serial):
#    export MAX_N=4; export SKIP_LARGE_N=1
#    qsub -v MAX_N,SKIP_LARGE_N hpc/jobs/submit_hpc_2D.sh
#
#    # Phase 1 only, up to N=16, skip Phase 2 entirely:
#    export MAX_N=16; export SKIP_LARGE_N=1
#    qsub -v MAX_N,SKIP_LARGE_N hpc/jobs/submit_hpc_2D.sh
#
#    # Skip QSVT everywhere (also skips Phase 2, since it is QSVT-only):
#    export SKIP_QSVT=1
#    qsub -v SKIP_QSVT hpc/jobs/submit_hpc_2D.sh
#
#    # Run specific sections only (e.g. HET cases), both phases:
#    export SECTIONS="4,5"
#    qsub -v SECTIONS hpc/jobs/submit_hpc_2D.sh
#
#    # Override the large-N tier (default "128,256"):
#    export LARGE_N="128"
#    qsub -v LARGE_N hpc/jobs/submit_hpc_2D.sh
#
#    # Use a different outer scheme (default: fmg):
#    export OUTER_SCHEME=multigrid
#    qsub -v OUTER_SCHEME hpc/jobs/submit_hpc_2D.sh
#
#    # Reproduce the original line-Jacobi results (Phase 1 only makes sense
#    # here - Jacobi at N=128/256 is not the point of this script):
#    export OUTER_SCHEME=jacobi; export OUTER_CRITERION=delta
#    export OUTER_TOL=1e-6; export SKIP_LARGE_N=1
#    qsub -v OUTER_SCHEME,OUTER_CRITERION,OUTER_TOL,SKIP_LARGE_N hpc/jobs/submit_hpc_2D.sh
#
#    # Combine options:
#    export MAX_N=16; export SKIP_QSVT=1; export SECTIONS="1,2,3"
#    qsub -v MAX_N,SKIP_QSVT,SECTIONS hpc/jobs/submit_hpc_2D.sh
#
#  Before submitting a large job, estimate its cost for free:
#    python3 hpc/runners/run_2d.py --n-values 128,256 --solvers qsvt \
#            --estimate
#
#  Monitor:
#    qstat -u $USER
#    tail -f results/2Dhpc_run/run.log
#
#  Reference: https://icl-rcs-user-guide.readthedocs.io/
# ============================================================

# --- Resource requests ---
#
# WALLTIME.  The outer scheme is now full multigrid (O(1) cycles) rather
# than the original line-Jacobi (O(N) sweeps), so the old per-N hour
# estimates in this file no longer apply and are not repeated here - they
# would just be wrong in a new, non-obvious way. Get a real number instead:
#
#     python3 hpc/runners/run_2d.py --n-values <N> --solvers <solver> \
#             --estimate
#
# costs seconds (it runs only the classical reference and projects quantum
# wall time from the measured per-strip-solve cost model) and prints a
# per-solver, per-N breakdown. Run it for both phases before changing the
# walltime below. As a starting point pending that measurement:
#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=4:mem=64gb
#
# NOTE ON ncpus vs --max-workers: MAX_WORKERS is set to match ncpus below.
# Aer simulations are already OpenMP-threaded internally. Running more
# worker processes than allocated cores oversubscribes the node.
# If you change ncpus, update MAX_WORKERS accordingly.
#
# NOTE ON mem: Section 2 (Two-Gaussian) computes a 200x200 Fourier reference
# grid, independent of N. Phase 2 (N=128, 256) holds several NxN arrays per
# level of the multigrid hierarchy; 64gb is comfortable through N=256 for
# this solver set (QSVT only - no VQLS optimiser state, no HHL statevector
# buffers at those N). Increase if --solvers is widened for Phase 2.

# --- Job metadata ---
#PBS -N quantum_pde_2Dfull_run
#PBS -o results/2Dhpc_run/pbs_stdout.log
#PBS -e results/2Dhpc_run/pbs_stderr.log

# --- Email notifications ---
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

# ============================================================
#  Environment setup
# ============================================================

echo "============================================================"
echo "  QUANTUM PDE SOLVER 2D — HPC JOB START"
echo "  Job ID       : $PBS_JOBID"
echo "  Node         : $(hostname)"
echo "  Date/Time    : $(date)"
echo "  Work dir     : $PBS_O_WORKDIR"
echo "  MAX_N        : ${MAX_N:-<not set: Phase 1 runs its full range, 4..64>}"
echo "  LARGE_N      : ${LARGE_N:-128,256}"
echo "  SKIP_QSVT    : ${SKIP_QSVT:-0}"
echo "  SKIP_LARGE_N : ${SKIP_LARGE_N:-0}"
echo "  SECTIONS     : ${SECTIONS:-<not set: all sections 1-5>}"
echo "  OUTER_SCHEME : ${OUTER_SCHEME:-fmg}"
echo "  QSVT_MAX_DEG : ${QSVT_MAX_DEGREE:-500}"
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

# Guard against missing pyqsp (same pattern as 1D script).
python3 -c "import pyqsp" 2>/dev/null || {
    echo "pyqsp not found in venv; installing..."
    pip install pyqsp==0.2.0
}

# Guard against missing scipy (needed for Section 2 interpolation).
python3 -c "import scipy" 2>/dev/null || {
    echo "scipy not found in venv; installing..."
    pip install scipy
}

# IMPORTANT: must match RESULTS_DIR in hpc/runners/run_2d.py.
# Both phases below write into this same directory; Phase 2 uses --append
# so Phase 1's results are merged rather than overwritten.
RESULTS_SUBDIR="results/2Dhpc_run"
mkdir -p "${RESULTS_SUBDIR}"

# Keep Aer's internal OpenMP threading within the allocated core count.
# Without this, each of the 4 worker processes may spawn as many OpenMP
# threads as there are physical cores on the node.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
echo "OMP_NUM_THREADS = ${OMP_NUM_THREADS}"

# Verify the QSVT phase cache is populated before starting.
# A missing cache does not abort the job (phases compute on-the-fly),
# but it will add minutes of phase computation per kappa on the first run -
# and Phase 2 needs it more than Phase 1, since it is QSVT-only.
CACHE_DIR="results/qsvt_phase_cache"
N_CACHE_FILES=$(ls "${CACHE_DIR}"/*.npz 2>/dev/null | wc -l)
echo "QSVT phase cache: ${N_CACHE_FILES} files in ${CACHE_DIR}"
if [ "${N_CACHE_FILES}" -lt 6 ]; then
    echo "WARNING: fewer than 6 cache files found."
    echo "         Run submit_precompute_2D.sh first for best performance."
    echo "         Continuing anyway -- phases will be computed on-the-fly."
fi

# Worker count pinned to ncpus requested above.
# Each worker handles one (section, N) work unit independently.
MAX_WORKERS=4

# Common flags shared by both phases.
SCHEME="${OUTER_SCHEME:-fmg}"
COMMON_ARGS="--scheme ${SCHEME} --max-workers ${MAX_WORKERS}"

if [ -n "${OUTER_TOL}" ]; then
    COMMON_ARGS="${COMMON_ARGS} --tol ${OUTER_TOL}"
fi
if [ -n "${OUTER_CRITERION}" ]; then
    COMMON_ARGS="${COMMON_ARGS} --criterion ${OUTER_CRITERION}"
fi
if [ -n "${SECTIONS}" ]; then
    COMMON_ARGS="${COMMON_ARGS} --sections ${SECTIONS}"
    echo "INFO: running sections ${SECTIONS} only, both phases."
fi

QSVT_MAX_DEGREE="${QSVT_MAX_DEGREE:-500}"

PHASE1_EXIT=0
PHASE2_EXIT=0

# ============================================================
#  Phase 1 (core): N = 4..64, all solvers
# ============================================================

PHASE1_ARGS="${COMMON_ARGS} --phase-tag core"

if [ -n "${MAX_N}" ]; then
    PHASE1_ARGS="${PHASE1_ARGS} --max-n ${MAX_N}"
    echo "INFO: Phase 1 truncated at N=${MAX_N}."
fi

if [ "${SKIP_QSVT:-0}" = "1" ]; then
    PHASE1_ARGS="${PHASE1_ARGS} --skip-qsvt"
    echo "INFO: QSVT disabled for Phase 1."
fi

echo "------------------------------------------------------------"
echo "PHASE 1 (core): all solvers, N up to ${MAX_N:-64}"
echo "Starting at $(date)"
echo "Arguments: ${PHASE1_ARGS}"
echo "------------------------------------------------------------"

python3 hpc/runners/run_2d.py ${PHASE1_ARGS}
PHASE1_EXIT=$?

echo "Phase 1 finished at $(date) with exit code ${PHASE1_EXIT}"

# ============================================================
#  Phase 2 (large): N = 128, 256, QSVT only
# ============================================================
#
# Skipped if SKIP_QSVT=1 (Phase 2 would run nothing), if SKIP_LARGE_N=1
# (explicit opt-out for a Phase-1-only validation run), or if Phase 1 failed
# outright (a broken environment will fail the same way twice; do not spend
# the walltime budget confirming that).

RUN_PHASE2=1
if [ "${SKIP_QSVT:-0}" = "1" ]; then
    echo "INFO: Phase 2 skipped (SKIP_QSVT=1; Phase 2 is QSVT-only)."
    RUN_PHASE2=0
fi
if [ "${SKIP_LARGE_N:-0}" = "1" ]; then
    echo "INFO: Phase 2 skipped (SKIP_LARGE_N=1)."
    RUN_PHASE2=0
fi
if [ "${PHASE1_EXIT}" -ne 0 ]; then
    echo "WARNING: Phase 1 exited with code ${PHASE1_EXIT}; skipping Phase 2."
    RUN_PHASE2=0
fi

if [ "${RUN_PHASE2}" = "1" ]; then
    PHASE2_ARGS="${COMMON_ARGS} --phase-tag large --append"
    PHASE2_ARGS="${PHASE2_ARGS} --n-values ${LARGE_N:-128,256}"
    PHASE2_ARGS="${PHASE2_ARGS} --solvers qsvt"
    PHASE2_ARGS="${PHASE2_ARGS} -I qsvt.max_degree=${QSVT_MAX_DEGREE}"

    echo "------------------------------------------------------------"
    echo "PHASE 2 (large): QSVT only, N in {${LARGE_N:-128,256}}"
    echo "Starting at $(date)"
    echo "Arguments: ${PHASE2_ARGS}"
    echo "------------------------------------------------------------"

    python3 hpc/runners/run_2d.py ${PHASE2_ARGS}
    PHASE2_EXIT=$?

    echo "Phase 2 finished at $(date) with exit code ${PHASE2_EXIT}"
fi

# Overall exit code: non-zero if either phase that actually ran failed.
EXIT_CODE=0
[ "${PHASE1_EXIT}" -ne 0 ] && EXIT_CODE=${PHASE1_EXIT}
[ "${RUN_PHASE2}" = "1" ] && [ "${PHASE2_EXIT}" -ne 0 ] && EXIT_CODE=${PHASE2_EXIT}

echo "------------------------------------------------------------"
echo "Benchmark finished at $(date)"
echo "Phase 1 exit code : ${PHASE1_EXIT}"
if [ "${RUN_PHASE2}" = "1" ]; then
    echo "Phase 2 exit code : ${PHASE2_EXIT}"
else
    echo "Phase 2            : skipped"
fi
echo "Overall exit code : ${EXIT_CODE}"

# ============================================================
#  Copy results to permanent RDS storage
# ============================================================
RDS_RESULTS="${HOME}/qpde-results/2Dhpc_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r "${RESULTS_SUBDIR}"/* "${RDS_RESULTS}/" 2>/dev/null
echo "Results copied to: ${RDS_RESULTS}"

echo "============================================================"
echo "  JOB COMPLETE"
echo "  Exit code : ${EXIT_CODE}"
echo "  Date/Time : $(date)"
echo "============================================================"

exit ${EXIT_CODE}