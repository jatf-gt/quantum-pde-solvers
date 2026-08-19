#!/bin/bash
# ============================================================================
#  submit_hpc_1D_4th.sh   -   1-D benchmark sweep, FOURTH-ORDER (pentadiagonal)
#
#  Runs `hpc/runners/run_1d.py --order 4` over sections 1, 1b and 2 at
#  N = 4, 8, 16, for Thomas, HHL, VQLS and QSVT.
#
#  Readiness
#  ---------
#  Two defects that invalidated every 4th-order result produced before
#  2026-08-10 are fixed, and this script must not be submitted against a clone
#  predating either:
#
#    Phase 2 (ace7f9b)  HHL and QSVT reconstructed their operator from A[0,0]
#                       and A[0,1] alone, discarding the +-2 band and solving a
#                       TRIDIAGONAL system -- errors of 52 % / 237 % / 117 % at
#                       N = 4 / 8 / 16, reported as sound because the residual
#                       was measured against the truncated operator. Both entry
#                       points now raise rather than truncate, and the 4th-order
#                       path block encodes A in full via the Sz.-Nagy dilation.
#
#    Phase 4a (8191325) The boundary closure applied 18*alpha where the row-0
#                       stencil gives 14*alpha, and omitted the h^2*f(0) term the
#                       PDE supplies for free. Convergence was order 2 for a
#                       general solution and absent altogether for non-zero
#                       Dirichlet data. Both corrections are right-hand-side
#                       only, so A, kappa and the phase-cache keys are unchanged.
#
#  The 2-D and 3-D closures were fixed on 2026-08-12 (Phase 4b) and their
#  scripts are submittable too.
#
#  Scope
#  -----
#  Sub-case 3c (Neumann-Dirichlet) is skipped automatically under --order 4:
#  PoissonProblem1D4th does not implement the Neumann closure. The row count is
#  correspondingly lower than the 2nd-order sweep's, which is expected and not a
#  gap.
#
#  N = 32 is deliberately NOT in the default scope. It requires the last stage
#  of hpc/jobs/submit_precompute_4th.sh (kappa = 586.8, capped at degree 5000),
#  which is exploratory and not guaranteed to complete. Raise N_VALUES only once
#  that cache entry exists -- the guard below will refuse otherwise.
#
#  Observe that N=16 requires a CAPPED precompute at order 4, unlike order 2: the
#  pentadiagonal operator needs degree 19375 there against the angle solver's
#  sanity limit of 15000, so an uncapped entry cannot be produced at all.
#  run_1d.qsvt_max_degree(N, order) is the single source of truth for the cap,
#  and this guard reads it rather than restating it.
#
#  Usage
#  -----
#    qsub hpc/jobs/submit_hpc_1D_4th.sh
#
#    export N_VALUES="4,8"; qsub -v N_VALUES hpc/jobs/submit_hpc_1D_4th.sh
#    export SOLVERS="vqls,qsvt"; qsub -v SOLVERS hpc/jobs/submit_hpc_1D_4th.sh
#
#  Comma-separated values must be passed as `qsub -v NAME` (bare name, value from
#  the exported shell variable); `-v NAME=value` breaks on PBS's comma splitting.
#
#  Monitor:
#    qstat -u $USER
#    tail -f results/1Dhpc_run_4th/pbs_stdout.log     # run.log may be unreadable
#                                                    # from the login node: OI-1
# ============================================================================

#PBS -l walltime=24:00:00
# Sized from the measured high-water marks reported by the PBS epilogue of
# every completed run of this sweep, not from a guess. 2-D and 3-D jobs peak
# at 0.7-1.9 GB and 1-D at 4.8-8.8 GB; memory is a scheduling dimension on
# CX3, so an over-request buys nothing and delays the job in the queue.
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -N quantum_pde_1D_4th
#PBS -o results/1Dhpc_run_4th/pbs_stdout.log
#PBS -e results/1Dhpc_run_4th/pbs_stderr.log
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

set -u

echo "============================================================"
echo "  QUANTUM PDE SOLVER — HPC JOB START (1-D, 4th order)"
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

# Refuses a dirty tree, and -- because ORDER=4 -- an environment holding the
# UPSTREAM quantum_linear_solvers, which has no pentadiagonal_toeplitz module.
# That single omission cost a 21 h job in which every HHL row failed.
ORDER=4 bash hpc/jobs/_preflight.sh || exit 1

RESULTS_SUBDIR="results/1Dhpc_run_4th"
mkdir -p "${RESULTS_SUBDIR}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

N_VALUES="${N_VALUES:-4,8,16}"
SECTIONS="${SECTIONS:-1,1b,2}"
SOLVERS="${SOLVERS:-hhl,vqls,qsvt}"
WORKERS="${WORKERS:-4}"
HHL_TIMEOUT_S="${HHL_TIMEOUT_S:-5400}"
ALLOW_INLINE_PHASES="${ALLOW_INLINE_PHASES:-0}"

echo ""
echo "  N_VALUES        : ${N_VALUES}"
echo "  SECTIONS        : ${SECTIONS}"
echo "  SOLVERS         : ${SOLVERS}"
echo "  WORKERS         : ${WORKERS}"
echo "  HHL_TIMEOUT_S   : ${HHL_TIMEOUT_S}"

# -- QSVT phase-cache coverage ------------------------------------------------
# The phase angles are the expensive, non-parallelisable stage of QSVT, and the
# cache key includes the degree tag: uncapped entries are recorded as d-1 and
# run_1d.py requests exactly the cap that qsvt_max_degree gives. A miss does not
# fail -- _resolve_qsvt_max_degree quietly drops to min(5000, int(15*kappa)) and
# the sweep proceeds at reduced polynomial accuracy -- so it must be checked
# here rather than discovered in the results.
case ",${SOLVERS}," in
  *,qsvt,*)
    echo ""
    echo "------------------------------------------------------------"
    echo "  QSVT phase-cache coverage (order 4)"
    echo "------------------------------------------------------------"
    python3 - "${N_VALUES}" "${ALLOW_INLINE_PHASES}" <<'PY' || exit 1
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "hpc/runners")

import run_1d
import precompute_phases as pp
import solvers.quantum.qsp_angles as qa

n_values = sorted({int(t) for t in sys.argv[1].split(",") if t.strip()})
allow = sys.argv[2].strip() == "1"
epsilon = round(run_1d.HHL_EPSILON, 8)

print(f"  {'N':>5} {'kappa':>12} {'key':>10} {'cached':>8}")
print("  " + "-" * 38)

missing = []
for N in n_values:
    cap = run_1d.qsvt_max_degree(N, 4)
    tag = cap if cap is not None else -1
    kappa = pp.kappa_1d(N, 4)
    cached = qa._load_disk((round(kappa, 4), epsilon, "auto", tag)) is not None
    print(f"  {N:>5} {kappa:>12.4f} {'d' + str(tag):>10} {str(cached):>8}")
    if not cached:
        missing.append(N)

if not missing:
    print("\n  All requested resolutions are cached.")
    sys.exit(0)

print(f"\n  No precomputed phases for N={missing} at order 4.")
if allow:
    print("  ALLOW_INLINE_PHASES=1: proceeding. QSVT will run at the fallback")
    print("  degree min(5000, int(15*kappa)), NOT at the accuracy the cached")
    print("  entries would give. Rows so produced are not comparable with rows")
    print("  produced from the cache.")
    sys.exit(0)

print("  Submit hpc/jobs/submit_precompute_4th.sh first (staged smallest-N")
print("  first), or set ALLOW_INLINE_PHASES=1 to accept the reduced-degree")
print("  fallback, or drop qsvt from SOLVERS.")
sys.exit(1)
PY
    ;;
esac

# -- Sweep --------------------------------------------------------------------
# One step per resolution, ascending, so that a walltime kill loses only the
# most expensive tail rather than the whole sweep. --append is what makes that
# safe: without it each step would rewrite results_full.json from its own rows
# alone and discard the preceding steps.
OVERALL=0

run_step () {
    local nval=$1
    echo ""
    echo "------------------------------------------------------------"
    echo "STEP N=${nval}  sections=${SECTIONS}  solvers=${SOLVERS}  $(date)"
    echo "------------------------------------------------------------"
    python3 hpc/runners/run_1d.py \
        --order 4 \
        --append --phase-tag "order4_n${nval}" \
        --n-values "${nval}" \
        --sections "${SECTIONS}" \
        --solvers "${SOLVERS}" \
        --hhl-timeout-s "${HHL_TIMEOUT_S}" \
        --max-workers "${WORKERS}"
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

# Classification of what the sweep produced. Behind `|| true`: a failure here
# must not mask the sweep's own exit code, and the rows are already on disk.
echo ""
echo "============================================================"
echo "  Gap analysis after the run"
echo "============================================================"
python3 scripts/utils/gap_analysis.py --dim 1 --order 4 \
        --results-dir "${RESULTS_SUBDIR}" --n-values "${N_VALUES}" \
        --geometry-rerun-complete HET_1D_3b_gaussian_Vd300 \
        -o results/manifests/rerun_1d_order4.json || true

RDS_RESULTS="${HOME}/qpde-results/1Dhpc_run_4th_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_RESULTS}"
cp -r "${RESULTS_SUBDIR}"/* "${RDS_RESULTS}/" 2>/dev/null
echo "Results copied to: ${RDS_RESULTS}"

exit ${OVERALL}
