#!/bin/bash
# ============================================================================
#  submit_hpc_3D_4th.sh   -   3-D benchmark sweep, FOURTH-ORDER (pentadiagonal)
#
#  Runs `hpc/runners/run_3d.py --order 4` over sections 1-7 at N = 4, 8, for
#  Thomas, HHL, VQLS and QSVT.
#
#  Readiness
#  ---------
#  This job was gated until 2026-08-12 and is now submittable. The defects that
#  invalidated every earlier 4th-order 3-D result are recorded in the header of
#  submit_hpc_2D_4th.sh, which 3-D shared in full: the +-2 band truncated by the
#  quantum entry points; the even reflection and the 18*alpha error in the
#  shared closure; and the use of f on the face where the closure needs the
#  second derivative NORMAL to it, which in more than one dimension is f less
#  the tangential second derivatives of the Dirichlet data. The replacement is
#  problems/poisson_line_3d_4th.py, fourth order along every axis rather than
#  along the strip alone.
#
#  Measured order by dense direct solve, against manufactured solutions:
#
#      solution                          order 2   order 4
#      exp(x+y+z)                          1.99      3.84
#      triple sin                          1.98      3.98
#      exp(x+y) cos(2 pi z), periodic        --      4.12
#      x^3 + y^3 + z^3                   machine-exact (both)
#
#  Phases: up to FOUR distinct strip operators per resolution
#  ---------------------------------------------------------
#  The odd reflection at a transverse boundary folds the ghost node onto the
#  strip's own diagonal, so a strip carries A_row + c_d*I for each transverse
#  boundary it touches. With two Dirichlet transverse axes that is four distinct
#  operators -- none, axis 1, axis 2, both -- and with the azimuthally periodic
#  HET slab it is two, a periodic axis having no boundary. All are requested
#  during a sweep and all need a cache entry. The count is independent of N: 15
#  entries covers both domains at N=4,8,16 and computes in under three seconds.
#
#      export DIM=3; qsub -v DIM hpc/jobs/submit_precompute_4th.sh
#
#  The guard below refuses to run QSVT unless every one of those keys is present.
#
#  Cost
#  ----
#  A 3-D sweep costs N^2 strip solves per outer iteration, so it is the most
#  expensive of the three by a wide margin. N is capped at 8 by default for that
#  reason; raise it deliberately, having read an --estimate first.
#
#  Usage
#  -----
#    qsub hpc/jobs/submit_hpc_3D_4th.sh
#
#    export N_VALUES="4"; qsub -v N_VALUES hpc/jobs/submit_hpc_3D_4th.sh
#    export SOLVERS="vqls,qsvt"; qsub -v SOLVERS hpc/jobs/submit_hpc_3D_4th.sh
#
#  Comma-separated values must be passed as `qsub -v NAME` (bare name, value from
#  the exported shell variable); `-v NAME=value` breaks on PBS's comma splitting.
#
#  Monitor:
#    qstat -u $USER
#    tail -f results/3Dhpc_run_4th/pbs_stdout.log    # run.log may be unreadable
#                                                   # from the login node: OI-1
# ============================================================================

#PBS -l walltime=48:00:00
# Sized from the measured high-water marks reported by the PBS epilogue of
# every completed run of this sweep, not from a guess. 2-D and 3-D jobs peak
# at 0.7-1.9 GB and 1-D at 4.8-8.8 GB; memory is a scheduling dimension on
# CX3, so an over-request buys nothing and delays the job in the queue.
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -N quantum_pde_3D_4th
#PBS -o results/3Dhpc_run_4th/pbs_stdout.log
#PBS -e results/3Dhpc_run_4th/pbs_stderr.log
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

set -u

echo "============================================================"
echo "  QUANTUM PDE SOLVER — HPC JOB START (3-D, 4th order)"
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
export PREFLIGHT_ALLOW_DIRTY=1
ORDER=4 bash hpc/jobs/_preflight.sh || exit 1

# The retired path is gone; a clone that still carries it predates Phase 4b and
# would produce order-0.88 rows -- at N^2 strip solves per outer iteration.
if [ -f "solvers/outer/multigrid_4th.py" ]; then
    echo "ERROR: solvers/outer/multigrid_4th.py is present, so this clone predates"
    echo "       Phase 4b. Its closure is an even reflection applied to Dirichlet"
    echo "       data with the 18*alpha error, measured order 0.88."
    echo "       Pull before submitting."
    exit 1
fi

RESULTS_SUBDIR="results/3Dhpc_run_4th"
mkdir -p "${RESULTS_SUBDIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

N_VALUES="${N_VALUES:-16}"
SECTIONS="${SECTIONS:-1,2,3,4,5,6,7}"
SOLVERS="${SOLVERS:-hhl,vqls,qsvt}"
WORKERS="${WORKERS:-4}"
SCHEME="${SCHEME:-fmg}"
TOL="${TOL:-}"
MAX_OUTER="${MAX_OUTER:-}"
ALLOW_INLINE_PHASES="${ALLOW_INLINE_PHASES:-0}"
# Per-case wall-clock bound on the outer iteration, forwarded to the scheme as
# -S max_wall_s. Without it a single non-converging case consumes the whole PBS
# walltime and the remaining resolutions are never attempted -- the failure mode
# recorded as defect 3 in docs/HPC_REPAIR_PLAN.md. The runner honours the cap and
# records stop_reason="wall_time_exceeded", which the gap analysis reads as a
# terminal measurement rather than a defect. 16 h matches the 2nd-order sweep,
# whose worst sound N=16 case measured 15.0 h.
MAX_WALL_S="${MAX_WALL_S:-21600}"

echo ""
echo "  N_VALUES  : ${N_VALUES}"
echo "  SECTIONS  : ${SECTIONS}"
echo "  SOLVERS   : ${SOLVERS}"
echo "  SCHEME    : ${SCHEME}"
echo "  WORKERS   : ${WORKERS}"

# -- QSVT phase-cache coverage ------------------------------------------------
# Every distinct strip operator, not one per resolution. A miss does not fail:
# the phases are computed inline instead, on the critical path of a sweep that
# already costs N^2 strip solves per outer iteration.
case ",${SOLVERS}," in
  *,qsvt,*)
    echo ""
    echo "------------------------------------------------------------"
    echo "  QSVT phase-cache coverage (3-D, order 4)"
    echo "------------------------------------------------------------"
    python3 - "${N_VALUES}" "${ALLOW_INLINE_PHASES}" <<'PY' || exit 1
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "hpc/runners")

import run_3d
import precompute_phases as pp
import solvers.quantum.qsp_angles as qa

n_values = sorted({int(t) for t in sys.argv[1].split(",") if t.strip()})
allow = sys.argv[2].strip() == "1"
epsilon = round(run_3d.HHL_EPSILON_DEFAULT, 8)

targets = pp.build_targets(3, n_values, "all", 4)

print(f"  {'strip family':>18} {'N':>5} {'kappa':>12} {'key':>8} {'cached':>8}")
print("  " + "-" * 56)

missing = []
for label, N, kappa in targets:
    cap = run_3d.QSVT_MAX_DEGREE_3D.get(N)
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

print("  Run:  export DIM=3; qsub -v DIM hpc/jobs/submit_precompute_4th.sh")
print("  It completes in seconds. Or set ALLOW_INLINE_PHASES=1, or drop qsvt")
print("  from SOLVERS.")
sys.exit(1)
PY
    ;;
esac

OPT_ARGS=""
[ -n "${TOL}" ]       && OPT_ARGS="${OPT_ARGS} --tol ${TOL}"
[ -n "${MAX_OUTER}" ] && OPT_ARGS="${OPT_ARGS} --max-outer ${MAX_OUTER}"

# -- Sweep --------------------------------------------------------------------
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
    python3 hpc/runners/run_3d.py \
        --order 4 \
        --append --phase-tag "order4_n${nval}" \
        --n-values "${nval}" \
        --sections "${SECTIONS}" \
        --solvers "${SOLVERS}" \
        --scheme "${SCHEME}" \
        --max-workers "${WORKERS}" \
        -S "max_wall_s=${MAX_WALL_S}" \
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
python3 scripts/utils/gap_analysis.py --dim 3 --order 4 \
        --results-dir "${RESULTS_SUBDIR}" --n-values "${N_VALUES}" \
        --geometry-rerun-complete 3D_HET_Discharge_SPT100,3D_HET_MMS_SPT100,3D_HET_RotatingSpoke_SPT100 \
        -o results/manifests/rerun_3d_order4.json || true

RDS_RESULTS="${HOME}/qpde-results/3Dhpc_run_4th_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r "${RESULTS_SUBDIR}"/* "${RDS_RESULTS}/" 2>/dev/null
echo "Results copied to: ${RDS_RESULTS}"

exit ${OVERALL}
