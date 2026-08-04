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
#      python3 scripts/run_hpc_3Dfull.py --max-n 32 --estimate
#  It costs seconds and prints a per-section, per-solver projection.
#
#  OUTER SCHEME
#  ------------
#  Both phases use OUTER_SCHEME (default: fmg, full multigrid) for every
#  solver and every N, so the comparison is like for like.
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
#  If you do want the crossover, set SCHEME_CROSSOVER=16 below - affected
#  rows are then tagged scheme_crossover and remain identifiable in analysis.
#
#  PREREQUISITE: activate the virtualenv, and run the QSVT phase precompute.
#    qsub submit_precompute_2D.sh     # phase cache is shared with 3-D
#
#  Usage:
#    qsub submit_hpc_3D.sh                        # full two-phase sweep
#
#    export MAX_N=8; export SKIP_LARGE_N=1        # fast validation pass
#    qsub -v MAX_N,SKIP_LARGE_N submit_hpc_3D.sh
#
#    export SECTIONS="3,4"                        # HET physics cases only
#    qsub -v SECTIONS submit_hpc_3D.sh
#
#    export SKIP_QSVT=1                           # also skips Phase 2
#    qsub -v SKIP_QSVT submit_hpc_3D.sh
#
#    export LARGE_N="32"; export QSVT_MAX_DEGREE=300
#    qsub -v LARGE_N,QSVT_MAX_DEGREE submit_hpc_3D.sh
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
#PBS -l select=1:ncpus=4:mem=64gb
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
#PBS -M j.trobajo-flecha24@imperial.ac.uk
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

cd "${PBS_O_WORKDIR}" || { echo "ERROR: Cannot cd to PBS_O_WORKDIR"; exit 1; }

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

VENV_PATH="${HOME}/venvs/qpde"
if [ ! -d "${VENV_PATH}" ]; then
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    echo "       See setup_hpc_env.sh."
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
    print("       Is the virtualenv active? See setup_hpc_env.sh.")
    sys.exit(1)
print("Backend check: OK")
PYCHECK
if [ $? -ne 0 ]; then exit 1; fi

# Must match RESULTS_DIR in scripts/run_hpc_3Dfull.py.
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

SCHEME="${OUTER_SCHEME:-fmg}"
COMMON_ARGS="--scheme ${SCHEME} --max-workers ${MAX_WORKERS}"
[ -n "${OUTER_TOL}" ]        && COMMON_ARGS="${COMMON_ARGS} --tol ${OUTER_TOL}"
[ -n "${SCHEME_CROSSOVER}" ] && COMMON_ARGS="${COMMON_ARGS} --scheme-crossover ${SCHEME_CROSSOVER}"
if [ -n "${SECTIONS}" ]; then
    COMMON_ARGS="${COMMON_ARGS} --sections ${SECTIONS}"
    echo "INFO: running sections ${SECTIONS} only, both phases."
fi

QSVT_MAX_DEGREE="${QSVT_MAX_DEGREE:-500}"
PHASE1_EXIT=0
PHASE2_EXIT=0

# ============================================================
#  Phase 1 (core): N = 4..16, all solvers
# ============================================================

PHASE1_ARGS="${COMMON_ARGS} --phase-tag core --max-n ${MAX_N:-16}"
PHASE1_ARGS="${PHASE1_ARGS} -I qsvt.max_degree=${QSVT_MAX_DEGREE}"
if [ "${SKIP_QSVT:-0}" = "1" ]; then
    PHASE1_ARGS="${PHASE1_ARGS} --skip-qsvt"
    echo "INFO: QSVT disabled for Phase 1."
fi

echo "------------------------------------------------------------"
echo "PHASE 1 (core): all solvers, N up to ${MAX_N:-16}"
echo "Starting at $(date)"
echo "Arguments: ${PHASE1_ARGS}"
echo "------------------------------------------------------------"

python3 scripts/run_hpc_3Dfull.py ${PHASE1_ARGS}
PHASE1_EXIT=$?
echo "Phase 1 finished at $(date) with exit code ${PHASE1_EXIT}"

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

    python3 scripts/run_hpc_3Dfull.py ${PHASE2_ARGS}
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