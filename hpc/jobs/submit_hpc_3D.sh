#!/bin/bash
# ============================================================
#  submit_hpc_3D.sh
#  PBS Pro job submission script for Imperial College CX3 HPC.
#
#  Runs the 3-D benchmark sweep across four sections:
#    Section 1: 3-D Poisson, triple-sin MMS on the unit cube (verification)
#    Section 2: HET channel MMS, SPT-100, azimuthally periodic (verification)
#    Section 3: HET rotating spoke, SPT-100 (physics case, exact solution)
#    Section 4: HET realistic discharge, SPT-100 at 300 V (production case)
#
#  TWO PHASES, one job, one shared results directory:
#
#    Phase 1 (core)  N = 4, 8, 16       all solvers: Thomas, HHL, VQLS, QSVT
#    Phase 2 (large) N = 32             QSVT only
#
#  WHY 3-D IS MUCH MORE EXPENSIVE THAN 2-D
#  ----------------------------------------
#  One relaxation sweep costs N strip solves in 2-D but N^2 in 3-D.  At N=32
#  that is 1024 strip solves per sweep against 32 in 2-D.  Measured QSVT wall
#  time for the HET case at N=16 with FMG was ~420 s; the projected costs from
#  --estimate at N=16 are roughly:
#
#      Section 2, N=16:  QSVT  ~470 s      HHL  ~8.2 h      VQLS  ~7.4 h
#
#  so HHL and VQLS are already marginal at N=16 across four sections, and
#  impractical at N=32.  Hence Phase 2 is QSVT-only: its per-strip-solve cost
#  exponent is ~0.6 against ~2.35 (HHL) and ~1.29 (VQLS).
#
#  ALWAYS run --estimate before changing MAX_N or the walltime:
#      python3 hpc/runners/run_3d.py --max-n 32 --estimate
#  It costs seconds and prints a per-section, per-solver projection.
#
#  OUTER SCHEME
#  ------------
#  Both phases use OUTER_SCHEME (default: fmg, full multigrid) for every
#  solver and every N, ensuring rigorous comparability.
#
#  A note on the N<=16 crossover: measured 3-D HET data shows SOR and FMG are
#  a wash at N=16 (11008 vs 12544 strip solves; 411 s vs 422 s of QSVT wall
#  time), with SOR marginally ahead below that.  The runner therefore exposes
#  --scheme-crossover, but it is OFF by default and deliberately so.  Mixing
#  schemes across N makes the work-versus-N curve discontinuous and conflates
#  solver scaling with scheme scaling, which is exactly the confound this
#  benchmark exists to avoid; and the saving is at most a few per cent, at
#  the N values that are cheap anyway.  The one place a scheme change is
#  unavoidable is N=4, where a 4x4x4 grid cannot be coarsened at all; the
#  runner falls back to SOR there automatically and tags the affected rows
#  scheme_fallback in the notes column, so it is never silent.
#  To evaluate the crossover, configure SCHEME_CROSSOVER=16 below - affected
#  rows are then tagged scheme_crossover and remain identifiable in analysis.
#
#  PREREQUISITE: activate the virtualenv, and run the QSVT phase precompute.
#    qsub hpc/jobs/submit_precompute_2D.sh     # phase cache is shared with 3-D
#
#  Usage:
#    qsub hpc/jobs/submit_hpc_3D.sh                        # full two-phase sweep
#
#    export MAX_N=8; export SKIP_LARGE_N=1        # fast validation pass
#    qsub -v MAX_N,SKIP_LARGE_N hpc/jobs/submit_hpc_3D.sh
#
#    export SECTIONS="3,4"                        # HET physics cases only
#    qsub -v SECTIONS hpc/jobs/submit_hpc_3D.sh
#
#    export SKIP_QSVT=1                           # also skips Phase 2
#    qsub -v SKIP_QSVT hpc/jobs/submit_hpc_3D.sh
#
#    export LARGE_N="32"; export QSVT_MAX_DEGREE=300
#    qsub -v LARGE_N,QSVT_MAX_DEGREE hpc/jobs/submit_hpc_3D.sh
#
#  Monitor:
#    qstat -u $USER
#    tail -f results/3Dhpc_run/run.log
# ============================================================

# --- Resource requests ---
#
# WALLTIME: set from --estimate output, not from this default.  The value
# below is a placeholder sized for Phase 1 at N<=16 over four sections with
# all three quantum solvers, which the projections put in the region of
# 30-60 h of serial work; with 4 workers that is roughly 8-15 h, plus Phase 2.
# Re-measure before trusting it.
#PBS -l walltime=72:00:00
# Sized from the measured high-water marks reported by the PBS epilogue of
# every completed run of this sweep, not from a guess. 2-D and 3-D jobs peak
# at 0.7-1.9 GB and 1-D at 4.8-8.8 GB; memory is a scheduling dimension on
# CX3, so an over-request buys nothing and delays the job in the queue.
#PBS -l select=1:ncpus=4:mem=16gb
#
# ncpus must match MAX_WORKERS below.  Aer is already OpenMP-threaded
# internally, so more workers than cores oversubscribes the node.
#
# mem: a 3-D field at N=32 is 262144 float64 = 2 MB, and the multigrid
# hierarchy plus archived E-field components hold a handful of those per
# worker.  64gb is generous at N<=32; raise for N=64.

# --- Job metadata ---
#PBS -N quantum_pde_3Dfull_run
#PBS -o results/3Dhpc_run/pbs_stdout.log
#PBS -e results/3Dhpc_run/pbs_stderr.log

# --- Email notifications ---
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

# ============================================================
#  Environment setup
# ============================================================

echo "============================================================"
echo "  QUANTUM PDE SOLVER 3D — HPC JOB START"
echo "  Job ID       : $PBS_JOBID"
echo "  Node         : $(hostname)"
echo "  Date/Time    : $(date)"
echo "  MAX_N        : ${MAX_N:-<not set: Phase 1 runs 4..16>}"
echo "  LARGE_N      : ${LARGE_N:-32}"
echo "  SECTIONS     : ${SECTIONS:-<not set: all sections 1-4>}"
echo "  SKIP_QSVT    : ${SKIP_QSVT:-0}"
echo "  SKIP_LARGE_N : ${SKIP_LARGE_N:-0}"
echo "  OUTER_SCHEME : ${OUTER_SCHEME:-fmg}"
echo "  QSVT_MAX_DEG : ${QSVT_MAX_DEGREE:-500}"
echo "  CROSSOVER    : ${SCHEME_CROSSOVER:-<off>}"
echo "============================================================"

# -- Repository root resolution -------------------------------
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

# Fail fast and loudly if a backend is missing.  The runner's own pre-flight
# will also catch this, but checking here means the job dies in seconds
# rather than after the environment has been set up and logged.
python3 - <<'PYCHECK'
import sys
missing = []
for mod in ("qiskit", "qiskit_algorithms", "pennylane", "scipy"):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    print(f"ERROR: missing module(s): {', '.join(missing)}")
    print("       Is the virtualenv active? See hpc/setup_hpc_env.sh.")
    sys.exit(1)
print("Backend check: OK")
PYCHECK
if [ $? -ne 0 ]; then exit 1; fi

# Must match RESULTS_DIR in hpc/runners/run_3d.py.
RESULTS_SUBDIR="results/3Dhpc_run"
mkdir -p "${RESULTS_SUBDIR}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
echo "OMP_NUM_THREADS = ${OMP_NUM_THREADS}"

# The QSVT phase cache is keyed on kappa, and kappa(A_line) -> 2 in 3-D
# (against 3 in 2-D), so 3-D runs will populate their own entries on first
# use even if the 2-D cache is present.
CACHE_DIR="results/qsvt_phase_cache"
N_CACHE_FILES=$(ls "${CACHE_DIR}"/*.npz 2>/dev/null | wc -l)
echo "QSVT phase cache: ${N_CACHE_FILES} files in ${CACHE_DIR}"
if [ "${N_CACHE_FILES}" -lt 6 ]; then
    echo "WARNING: sparse phase cache; first QSVT solves will compute angles."
fi

MAX_WORKERS=4

# Per-case wall-clock bound on the outer iteration, forwarded to the scheme as
# -S max_wall_s. In 3D, FMG should converge quickly, so 10h is a safe generous cap.
MAX_WALL_S="${MAX_WALL_S:-36000}"
SCHEME="${OUTER_SCHEME:-fmg}"
COMMON_ARGS="--scheme ${SCHEME} --max-workers ${MAX_WORKERS} -S max_wall_s=${MAX_WALL_S}"
[ -n "${OUTER_TOL}" ]        && COMMON_ARGS="${COMMON_ARGS} --tol ${OUTER_TOL}"
[ -n "${SCHEME_CROSSOVER}" ] && COMMON_ARGS="${COMMON_ARGS} --scheme-crossover ${SCHEME_CROSSOVER}"
if [ -n "${SECTIONS}" ]; then
    COMMON_ARGS="${COMMON_ARGS} --sections ${SECTIONS}"
    echo "INFO: running sections ${SECTIONS} only, both phases."
fi
if [ -n "${N_VALUES}" ]; then
    COMMON_ARGS="${COMMON_ARGS} --n-values ${N_VALUES}"
    echo "INFO: N_VALUES override = ${N_VALUES}"
fi
if [ -n "${SOLVERS}" ]; then
    COMMON_ARGS="${COMMON_ARGS} --solvers ${SOLVERS}"
    echo "INFO: SOLVERS override = ${SOLVERS}"
fi
if [ -n "${CASES}" ]; then
    COMMON_ARGS="${COMMON_ARGS} --cases ${CASES}"
    echo "INFO: CASES override = ${CASES}"
fi

QSVT_MAX_DEGREE="${QSVT_MAX_DEGREE:-500}"
PHASE1_EXIT=0
PHASE2_EXIT=0

# ============================================================
#  Phase 1 (core): N = 4..16, all solvers
# ============================================================

# --append merges the existing results_full.json ahead of this invocation's rows
# rather than replacing it, matching Phase 2 below. Without it a scope-restricted
# Phase 1 -- or one killed by the walltime -- leaves the summary holding only the
# rows it reached, discarding every earlier row in the directory. Rows supersede
# on (case, solver, N), so a deliberate re-measurement still wins.
PHASE1_ARGS="${COMMON_ARGS} --phase-tag core --max-n ${MAX_N:-16} --append"
PHASE1_ARGS="${PHASE1_ARGS} -I qsvt.max_degree=${QSVT_MAX_DEGREE}"
if [ "${SKIP_QSVT:-0}" = "1" ]; then
    PHASE1_ARGS="${PHASE1_ARGS} --skip-qsvt"
    echo "INFO: QSVT disabled for Phase 1."
fi

if [ "${SKIP_PHASE1:-0}" = "1" ]; then
    echo "INFO: Phase 1 skipped (SKIP_PHASE1=1)."
    PHASE1_EXIT=0
else
    echo "------------------------------------------------------------"
    echo "PHASE 1 (core): all solvers, N up to ${MAX_N:-16}"
    echo "Starting at $(date)"
    echo "Arguments: ${PHASE1_ARGS}"
    echo "------------------------------------------------------------"

    python3 hpc/runners/run_3d.py ${PHASE1_ARGS}
    PHASE1_EXIT=$?
    echo "Phase 1 finished at $(date) with exit code ${PHASE1_EXIT}"
fi

# ============================================================
#  Phase 2 (large): N = 32, QSVT only
# ============================================================

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
    echo "WARNING: Phase 1 exited ${PHASE1_EXIT}; skipping Phase 2."
    RUN_PHASE2=0
fi

if [ "${RUN_PHASE2}" = "1" ]; then
    # --append merges with Phase 1's results_full.json rather than
    # overwriting it; without it the second invocation would discard Phase 1.
    PHASE2_ARGS="${COMMON_ARGS} --phase-tag large --append"
    PHASE2_ARGS="${PHASE2_ARGS} --n-values ${LARGE_N:-32} --solvers qsvt"
    PHASE2_ARGS="${PHASE2_ARGS} -I qsvt.max_degree=${QSVT_MAX_DEGREE}"

    echo "------------------------------------------------------------"
    echo "PHASE 2 (large): QSVT only, N in {${LARGE_N:-32}}"
    echo "Starting at $(date)"
    echo "Arguments: ${PHASE2_ARGS}"
    echo "------------------------------------------------------------"

    python3 hpc/runners/run_3d.py ${PHASE2_ARGS}
    PHASE2_EXIT=$?
    echo "Phase 2 finished at $(date) with exit code ${PHASE2_EXIT}"
fi

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

# ============================================================
#  Copy results to permanent RDS storage
# ============================================================
RDS_RESULTS="${HOME}/qpde-results/3Dhpc_run_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r "${RESULTS_SUBDIR}"/* "${RDS_RESULTS}/" 2>/dev/null
echo "Results copied to: ${RDS_RESULTS}"
echo "  (includes solution3d_*.npz: full phi, phi_exact, f, grid coords,"
echo "   E-field components and residual history for every run)"

echo "============================================================"
echo "  JOB COMPLETE   Exit code: ${EXIT_CODE}   $(date)"
echo "============================================================"
exit ${EXIT_CODE}