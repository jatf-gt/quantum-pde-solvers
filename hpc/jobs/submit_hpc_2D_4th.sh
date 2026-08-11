#!/bin/bash
# ============================================================================
#  submit_hpc_2D_4th.sh   -   2-D benchmark sweep, FOURTH-ORDER (pentadiagonal)
#
#  Runs `hpc/runners/run_2d.py --order 4` over sections 1-5 at N = 4, 8, 16,
#  for Thomas, HHL, VQLS and QSVT.
#
#  Readiness
#  ---------
#  This job was gated until 2026-08-12 and is now submittable. Three defects
#  invalidated every 4th-order 2-D result produced before that date, and this
#  script must not be run against a clone predating any of their fixes:
#
#    Phase 2       HHL and QSVT reconstructed their operator from A[0,0] and
#                  A[0,1] alone, discarding the +-2 band and solving a
#                  TRIDIAGONAL system. Both entry points now raise rather than
#                  truncate, and the 4th-order path block encodes A in full via
#                  the Sz.-Nagy dilation.
#
#    Phase 4b      The 2-D closure in solvers/outer/multigrid_4th.py folded the
#                  ghost node into A[0,1] -- an EVEN reflection, wrong for
#                  Dirichlet data -- and wrote 18*alpha where the row-0 stencil
#                  gives 14*alpha. Measured order 0.88. Replaced by
#                  problems/poisson_line_2d_4th.py, fourth order in BOTH
#                  directions: the mixed design (4th along the strip, 2nd
#                  transverse) is capped at order 2 by construction.
#
#    Phase 4b,     The ghost reflection needs the second derivative NORMAL to
#    second        the face. In 1-D the PDE makes that the source itself, so
#    defect        poisson_1d_4th is right to use f(0); in 2-D
#                  d2u/dn2 = f - the tangential second derivative of the
#                  Dirichlet data. Using f alone leaves each boundary row
#                  carrying a residual of exactly -f_face/12 and caps the scheme
#                  at order 2. Invisible on sin(pi x) sin(pi y).
#
#  Measured order by dense direct solve, against manufactured solutions:
#
#      solution              order 2   order 4
#      exp(x+y)                1.96      3.87
#      cos(2x) + y^2           1.95      3.90
#      sin(pi x) sin(pi y)     1.95      3.91
#      x^3 + y^3             machine-exact (both)
#
#  Phases: TWO distinct strip operators per resolution
#  --------------------------------------------------
#  The odd reflection at a transverse boundary folds the ghost node onto the
#  strip's own diagonal, so the strips adjacent to y=0 and y=Ly carry
#  A_row + c_y*I and hence a different kappa. Both are requested during a sweep
#  and both need a cache entry; precomputing only the interior one leaves 2/N of
#  the strip solves computing their phases inline. The count is two whatever N
#  is, so the cache stays small: 12 entries covers both domains at N=4,8,16 and
#  computes in under four seconds.
#
#      export DIM=2; qsub -v DIM hpc/jobs/submit_precompute_4th.sh
#
#  The guard below refuses to run QSVT unless every one of those keys is present.
#
#  Usage
#  -----
#    qsub hpc/jobs/submit_hpc_2D_4th.sh
#
#    export N_VALUES="4,8"; qsub -v N_VALUES hpc/jobs/submit_hpc_2D_4th.sh
#    export SOLVERS="vqls,qsvt"; qsub -v SOLVERS hpc/jobs/submit_hpc_2D_4th.sh
#
#  Comma-separated values must be passed as `qsub -v NAME` (bare name, value from
#  the exported shell variable); `-v NAME=value` breaks on PBS's comma splitting.
#
#  Monitor:
#    qstat -u $USER
#    tail -f results/2Dhpc_run_4th/pbs_stdout.log    # run.log may be unreadable
#                                                   # from the login node: OI-1
# ============================================================================

#PBS -l walltime=48:00:00
#PBS -l select=1:ncpus=4:mem=128gb
#PBS -N quantum_pde_2D_4th
#PBS -o results/2Dhpc_run_4th/pbs_stdout.log
#PBS -e results/2Dhpc_run_4th/pbs_stderr.log
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

set -u

echo "============================================================"
echo "  QUANTUM PDE SOLVER — HPC JOB START (2-D, 4th order)"
echo "  Job ID    : ${PBS_JOBID:-interactive}"
echo "  Node      : $(hostname)"
echo "  Date/Time : $(date)"
echo "  Work dir  : ${PBS_O_WORKDIR:-$(pwd)}"
echo "============================================================"

REPO_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
while [ ! -f "${REPO_ROOT}/pyproject.toml" ] && [ "${REPO_ROOT}" != "/" ]; do
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
if [ ! -f "${REPO_ROOT}/pyproject.toml" ]; then
    echo "ERROR: no repository root (pyproject.toml) at or above ${PBS_O_WORKDIR:-$(pwd)}."
    exit 1
fi
cd "${REPO_ROOT}" || exit 1

module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

VENV_PATH="${HOME}/venvs/qpde"
if [ ! -d "${VENV_PATH}" ]; then
    echo "ERROR: Virtual environment not found at ${VENV_PATH}"
    exit 1
fi
source "${VENV_PATH}/bin/activate"

# ORDER=4 additionally checks the pentadiagonal module, absent from the upstream
# quantum_linear_solvers. Its absence cost a 21 h job in which every HHL row
# failed with ModuleNotFoundError.
ORDER=4 bash hpc/jobs/_preflight.sh || exit 1

# The retired path is gone; a clone that still carries it predates Phase 4b and
# would produce order-0.88 rows. Cheap to check, and it is the one thing that
# distinguishes a correct 4th-order sweep from a wasted one.
if [ -f "solvers/outer/multigrid_4th.py" ]; then
    echo "ERROR: solvers/outer/multigrid_4th.py is present, so this clone predates"
    echo "       Phase 4b. Its closure is an even reflection applied to Dirichlet"
    echo "       data with the 18*alpha error, measured order 0.88."
    echo "       Pull before submitting."
    exit 1
fi

RESULTS_SUBDIR="results/2Dhpc_run_4th"
mkdir -p "${RESULTS_SUBDIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

N_VALUES="${N_VALUES:-4,8,16}"
SECTIONS="${SECTIONS:-1,2,3,4,5}"
SOLVERS="${SOLVERS:-hhl,vqls,qsvt}"
WORKERS="${WORKERS:-4}"
SCHEME="${SCHEME:-fmg}"
TOL="${TOL:-}"
MAX_OUTER="${MAX_OUTER:-}"
ALLOW_INLINE_PHASES="${ALLOW_INLINE_PHASES:-0}"

echo ""
echo "  N_VALUES  : ${N_VALUES}"
echo "  SECTIONS  : ${SECTIONS}"
echo "  SOLVERS   : ${SOLVERS}"
echo "  SCHEME    : ${SCHEME}"
echo "  WORKERS   : ${WORKERS}"

# ── QSVT phase-cache coverage ────────────────────────────────────────────────
# Every distinct strip operator, not one per resolution. A miss does not fail:
# the phases are computed inline instead, which is the expensive
# non-parallelisable step this cache exists to remove.
case ",${SOLVERS}," in
  *,qsvt,*)
    echo ""
    echo "------------------------------------------------------------"
    echo "  QSVT phase-cache coverage (2-D, order 4)"
    echo "------------------------------------------------------------"
    python3 - "${N_VALUES}" "${ALLOW_INLINE_PHASES}" <<'PY' || exit 1
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "hpc/runners")

import run_2d
import precompute_phases as pp
import solvers.quantum.qsp_angles as qa

n_values = sorted({int(t) for t in sys.argv[1].split(",") if t.strip()})
allow = sys.argv[2].strip() == "1"
epsilon = round(run_2d.HHL_EPSILON_DEFAULT, 8)

targets = pp.build_targets(2, n_values, "all", 4)

print(f"  {'strip family':>18} {'N':>5} {'kappa':>12} {'key':>8} {'cached':>8}")
print("  " + "-" * 56)

missing = []
for label, N, kappa in targets:
    cap = run_2d.QSVT_MAX_DEGREE_2D.get(N)
    tag = cap if cap is not None else -1
    cached = qa._load_disk((round(kappa, 4), epsilon, "auto", tag)) is not None
    print(f"  {label:>18} {N:>5} {kappa:>12.4f} {'d' + str(tag):>8} "
          f"{str(cached):>8}")
    if not cached:
        missing.append(f"{label} N={N}")

if not missing:
    print(f"\n  All {len(targets)} distinct strip operators are cached.")
    sys.exit(0)

print(f"\n  No precomputed phases for: {', '.join(missing)}")
if allow:
    print("  ALLOW_INLINE_PHASES=1: proceeding. Those strips will compute their")
    print("  phases inline, on the critical path of the sweep.")
    sys.exit(0)

print("  Run:  export DIM=2; qsub -v DIM hpc/jobs/submit_precompute_4th.sh")
print("  It completes in seconds. Or set ALLOW_INLINE_PHASES=1, or drop qsvt")
print("  from SOLVERS.")
sys.exit(1)
PY
    ;;
esac

OPT_ARGS=""
[ -n "${TOL}" ]       && OPT_ARGS="${OPT_ARGS} --tol ${TOL}"
[ -n "${MAX_OUTER}" ] && OPT_ARGS="${OPT_ARGS} --max-outer ${MAX_OUTER}"

# ── Sweep ────────────────────────────────────────────────────────────────────
# One step per resolution, ascending, so a walltime kill loses only the most
# expensive tail. --append is what makes that safe: without it each step would
# rewrite results_full.json from its own rows alone.
OVERALL=0

run_step () {
    local nval=$1
    echo ""
    echo "------------------------------------------------------------"
    echo "STEP N=${nval}  sections=${SECTIONS}  solvers=${SOLVERS}  $(date)"
    echo "------------------------------------------------------------"
    python3 hpc/runners/run_2d.py \
        --order 4 \
        --append --phase-tag "order4_n${nval}" \
        --n-values "${nval}" \
        --sections "${SECTIONS}" \
        --solvers "${SOLVERS}" \
        --scheme "${SCHEME}" \
        --max-workers "${WORKERS}" \
        ${OPT_ARGS}
    local rc=$?
    echo "STEP N=${nval} finished $(date) exit=${rc}"
    [ "${rc}" -ne 0 ] && [ "${OVERALL}" -eq 0 ] && OVERALL=${rc}
    return 0
}

echo ""
echo "Starting 4th-order sweep at $(date)"

for nval in $(echo "${N_VALUES}" | tr ',' ' '); do
    run_step "${nval}"
done

echo ""
echo "------------------------------------------------------------"
echo "Sweep finished at $(date) with overall exit code ${OVERALL}"

echo ""
echo "============================================================"
echo "  Gap analysis after the run"
echo "============================================================"
python3 scripts/gap_analysis.py --dim 2 \
        --results-dir "${RESULTS_SUBDIR}" --n-values "${N_VALUES}" \
        -o results/manifests/rerun_2d_order4.json || true

RDS_RESULTS="${HOME}/qpde-results/2Dhpc_run_4th_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r "${RESULTS_SUBDIR}"/* "${RDS_RESULTS}/" 2>/dev/null
echo "Results copied to: ${RDS_RESULTS}"

exit ${OVERALL}
