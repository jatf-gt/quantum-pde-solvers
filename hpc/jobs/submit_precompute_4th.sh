#!/bin/bash
# ============================================================================
#  submit_precompute_4th.sh
#  QSVT phase-angle precompute for the FOURTH-ORDER (pentadiagonal) operator.
#
#  Companion to submit_precompute_hpc.sh (2nd order, 1-D) and
#  submit_precompute_2D.sh (2nd order, 2-D). Runs
#  `hpc/runners/precompute_phases.py --dim ${DIM} --order 4`, DIM defaulting to 1.
#
#  Why a separate job from the 2nd-order ones
#  ------------------------------------------
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
#  2-D and 3-D are covered too, selected with DIM, and cost minutes rather than
#  days: their strip kappa is bounded -- 2.49-3.14 in 2-D and 1.41-2.08 in 3-D
#  across the whole sweep -- so the polynomial degree stays small at every N. They
#  were previously refused because the strip operator was unsettled; it is now
#  problems/poisson_line_{2,3}d_4th.py, verified to order 3.84-4.12.
#
#  ONE RESOLUTION CONTRIBUTES SEVERAL KEYS in 2-D and 3-D at order 4. The odd
#  reflection at a transverse boundary folds the ghost node onto the strip's own
#  diagonal, so the boundary-adjacent strips carry a different operator and hence
#  a different kappa: two distinct matrices in 2-D, up to four in 3-D, and two
#  when a transverse axis is periodic. A sweep requests all of them.
#  precompute_phases.py enumerates them; --list-kappas prints the full set.
#
#  THE DEGREE TAG IS PART OF THE CACHE KEY
#  ---------------------------------------
#  The key is (round(kappa,4), round(epsilon,8), method, max_degree), where an
#  uncapped run is recorded as the tag `d-1`. run_1d.py asks for the cap given by
#  QSVT_MAX_DEGREE_BY_N at the resolution being solved:
#
#      N = 4, 8       ->  None   ->  tag d-1     (uncapped)
#      N = 16, 32, 64 ->  5000   ->  tag d5000
#
#  At order 2 the boundary sits one resolution higher, N=16 being uncapped
#  there: the pentadiagonal operator crosses the solver's degree limit first.
#
#  run_2d.py and run_3d.py do the same through QSVT_MAX_DEGREE_{2D,3D}, whose cap
#  above N=16 is 500 rather than 5000. The guard reads whichever table applies to
#  DIM rather than restating any of them here.
#
#  Passing --max-degree 5000 at N = 4, 8 or 16 therefore writes an entry under a
#  key no solver will ever request. The miss is silent: _resolve_qsvt_max_degree
#  falls back to min(5000, int(15*kappa)) -- degree 179 / 632 / 2317 at
#  N = 4 / 8 / 16 -- and the sweep proceeds at reduced polynomial accuracy having
#  ignored 71 h of precompute. MAX_DEGREE is consequently UNSET by default here,
#  and the guard below refuses any invocation whose tag does not match the tag the
#  solver will look up. Verified 2026-08-11: none of the four order-4 keys is
#  present in results/qsvt_phase_cache/.
#
#  Epsilon ordering
#  ----------------
#  The sweep itself only ever requests epsilon = 0.01 (run_1d.HHL_EPSILON). The
#  additional 0.5 and 0.1 entries serve the Phase 8 sensitivity study, and
#  precompute_phases.py processes epsilons largest-first -- i.e. it would compute
#  the two optional entries before the one the sweep depends on, and a walltime
#  kill would take exactly the entry that matters. This job therefore runs the
#  sweep-critical epsilon in a first pass on its own, and the optional ones after.
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
#  The stage boundaries are set by the degree tag, not by taste: N <= 16 must be
#  computed uncapped and N >= 32 capped, so the two cannot share one invocation.
#  Staging matters only in 1-D. In 2-D and 3-D kappa is bounded below 3.2, so the
#  entire set completes in minutes and needs no staging at all.
#
#  Usage
#  -----
#    # Stage 1 -- N=4,8, uncapped. Start here.
#    export N_VALUES="4,8"
#    qsub -v N_VALUES hpc/jobs/submit_precompute_4th.sh
#
#    # Stage 2 -- N=16, CAPPED at 5000. Not uncapped: the pentadiagonal operator
#    # needs degree 19375 there, against the angle solver's sanity limit of 15000,
#    # so an uncapped solve is REFUSED outright and writes nothing. The
#    # tridiagonal operator needs 14177 and passes just under, which is why the
#    # 2nd-order cache has an uncapped N=16 entry and this one cannot.
#    export N_VALUES="16" MAX_DEGREE="5000"
#    qsub -v N_VALUES,MAX_DEGREE hpc/jobs/submit_precompute_4th.sh
#
#    # Stage 3 -- N=32, capped at 5000 to match QSVT_MAX_DEGREE_BY_N. Exploratory:
#    # kappa = 586.8 and completion within 71 h is not guaranteed. Required only if
#    # the 1-D order-4 sweep is to be run beyond N=16.
#    export N_VALUES="32" MAX_DEGREE="5000"
#    qsub -v N_VALUES,MAX_DEGREE hpc/jobs/submit_precompute_4th.sh
#
#    # 2-D and 3-D -- minutes, not hours. Both domains, every distinct strip
#    # operator. Defaults cover N = 4, 8, 16, uncapped.
#    export DIM="2"; qsub -v DIM hpc/jobs/submit_precompute_4th.sh
#    export DIM="3"; qsub -v DIM hpc/jobs/submit_precompute_4th.sh
#
#    # 2-D at N = 32, 64. A SEPARATE invocation, because the runner caps the
#    # degree at 500 above N=16 and the cap is part of the cache key: a run
#    # mixing capped and uncapped resolutions is refused by the guard.
#    export DIM="2" N_VALUES="32,64" MAX_DEGREE="500"
#    qsub -v DIM,N_VALUES,MAX_DEGREE hpc/jobs/submit_precompute_4th.sh
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

set -u

echo "============================================================"
echo "  QSVT PHASE PRECOMPUTE (4th order) — HPC JOB START"
echo "  Job ID    : ${PBS_JOBID:-interactive}"
echo "  Node      : $(hostname)"
echo "  Date/Time : $(date)"
echo "  Work dir  : ${PBS_O_WORKDIR:-$(pwd)}"
echo "  DIM       : ${DIM:-<not set, script default: 1>}"
echo "  N_VALUES  : ${N_VALUES:-<not set, per-dimension default>}"
echo "  MAX_DEGREE: ${MAX_DEGREE:-<not set: uncapped, tag d-1>}"
echo "============================================================"

# ── Repository root resolution ───────────────────────────────
# PBS copies this script to a spool directory before executing it, so $0 and
# BASH_SOURCE do NOT point at the original file. PBS_O_WORKDIR -- the directory
# qsub was invoked from -- is the only reliable anchor.
REPO_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
while [ ! -f "${REPO_ROOT}/pyproject.toml" ] && [ "${REPO_ROOT}" != "/" ]; do
    REPO_ROOT="$(dirname "${REPO_ROOT}")"
done
if [ ! -f "${REPO_ROOT}/pyproject.toml" ]; then
    echo "ERROR: no repository root (pyproject.toml) at or above ${PBS_O_WORKDIR:-$(pwd)}."
    echo "       Submit from inside a clone, e.g. qsub hpc/jobs/submit_precompute_4th.sh"
    exit 1
fi
cd "${REPO_ROOT}" || { echo "ERROR: cannot cd to ${REPO_ROOT}"; exit 1; }

# ── Exit verdict ─────────────────────────────────────────────────────────────
# Every early exit in this job -- a dirty tree caught by _preflight.sh, a
# degree-tag mismatch caught by the cache-key guard -- prints its reason to
# STDOUT and returns non-zero, so the PBS .err file stays empty and the job
# looks from the outside exactly like one that ran and wrote nothing. It also
# takes the same 10-30 s either way, that being the cost of importing qiskit,
# so the wall time does not distinguish them. This trap makes the difference
# unmissable, and reports how many cache entries were actually added.

_cache_count () { ls -1 results/qsvt_phase_cache 2>/dev/null | wc -l | tr -d " "; }
CACHE_BEFORE="$(_cache_count)"

final_report () {
    local rc=$1
    local after
    after="$(_cache_count)"
    echo ""
    echo "============================================================"
    if [ "${rc}" -ne 0 ]; then
        echo "  JOB ABORTED (exit ${rc}) - NOTHING WAS COMPUTED"
        echo "  Cache entries unchanged: ${CACHE_BEFORE}"
        echo ""
        echo "  The reason is printed ABOVE, in this file (stdout). The .err"
        echo "  file is empty because these checks do not write to stderr."
        echo "  Search for:  PREFLIGHT FAILED    (dirty tree, missing module)"
        echo "               ABORTING            (degree-tag mismatch)"
    else
        echo "  Cache entries: ${CACHE_BEFORE} before, ${after} after, "
        echo "                 $((after - CACHE_BEFORE)) added"
        if [ "${after}" -eq "${CACHE_BEFORE}" ]; then
            echo ""
            echo "  NOTHING NEW WAS WRITTEN, and that is a complete result, not"
            echo "  a failure: every key requested was already cached. See the"
            echo "  'already cached, skipping' lines above. The 2-D and 3-D"
            echo "  order-4 sets at N <= 16 are committed to the repository, so"
            echo "  those jobs are no-ops on a fresh clone."
        fi
    fi
    echo "  finished $(date)"
    echo "============================================================"
}
trap 'final_report $?' EXIT

if [ "${PBS_O_WORKDIR:-${REPO_ROOT}}" != "${REPO_ROOT}" ]; then
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

DIM="${DIM:-1}"
# Every default stops at N=16. The runners cap the QSVT degree above that
# (QSVT_MAX_DEGREE_* switches from None to 5000 in 1-D, 500 in 2-D/3-D), the cap
# is part of the cache key, and one invocation writes one tag -- so a default
# spanning the boundary would abort on the guard every time it was run without
# arguments. N >= 32 is a deliberate second stage; see Usage.
case "${DIM}" in
  1) N_VALUES="${N_VALUES:-4,8}" ;;
  2) N_VALUES="${N_VALUES:-4,8,16}" ;;
  3) N_VALUES="${N_VALUES:-4,8,16}" ;;
  *) echo "ERROR: DIM must be 1, 2 or 3; got ${DIM}"; exit 1 ;;
esac
MAX_DEGREE="${MAX_DEGREE:-}"
SWEEP_EPSILON="${SWEEP_EPSILON:-0.01}"
EXTRA_EPSILONS="${EXTRA_EPSILONS:-0.5,0.1}"

# ============================================================
#  Cache-key guard
# ============================================================
# Refuses the job if the degree tag it would write differs from the tag run_1d.py
# will request, which is the one way to spend 71 h and end with a cache the sweep
# cannot see. The kappa is taken from the same problem class the solver builds,
# never from the table in this header.
echo ""
echo "------------------------------------------------------------"
echo "  Cache-key check -- what this job writes vs what run_1d asks"
echo "------------------------------------------------------------"
python3 - "${DIM}" "${N_VALUES}" "${MAX_DEGREE}" "${SWEEP_EPSILON}" <<'PY' || exit 1
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "hpc/runners")

import precompute_phases as pp
import solvers.quantum.qsp_angles as qa

dim = int(sys.argv[1])
n_values = sorted({int(t) for t in sys.argv[2].split(",") if t.strip()})
raw_cap = sys.argv[3].strip()
epsilon = round(float(sys.argv[4]), 8)

tag_written = int(raw_cap) if raw_cap else -1

# The cap the SWEEP will request, taken from the runner's own table in every
# dimension rather than restated here: a table restated in a shell script is a
# table that drifts, and a drifted degree tag is a silent cache miss.
if dim == 1:
    import run_1d
    # Order-aware: the pentadiagonal operator crosses the angle solver's degree
    # sanity limit at N=16, where the tridiagonal one passes just under it.
    def cap_for(N):
        return run_1d.qsvt_max_degree(N, 4)
elif dim == 2:
    import run_2d
    def cap_for(N):
        return run_2d.QSVT_MAX_DEGREE_2D.get(N)
else:
    import run_3d
    def cap_for(N):
        return run_3d.QSVT_MAX_DEGREE_3D.get(N)

# Every distinct strip operator, not one per resolution: in 2-D and 3-D at order
# 4 the boundary-adjacent strips carry their own kappa and their own cache entry.
targets = pp.build_targets(dim, n_values, "all", 4)

print(f"  {'strip family':>18} {'N':>5} {'kappa':>12} {'degree':>8} "
      f"{'solver key':>11} {'this job':>9} {'cached':>7}")
print("  " + "-" * 82)

runner = {1: "run_1d.py", 2: "run_2d.py", 3: "run_3d.py"}[dim]

mismatched = []
impossible = []
for label, N, kappa in targets:
    cap = cap_for(N)
    tag_solver = cap if cap is not None else -1
    est = qa.polynomial_degree_estimate(kappa, epsilon)
    cached = qa._load_disk((round(kappa, 4), epsilon, "auto", tag_solver)) is not None
    print(f"  {label:>18} {N:>5} {kappa:>12.4f} {est:>8} "
          f"{'d' + str(tag_solver):>11} {'d' + str(tag_written):>9} "
          f"{str(cached):>7}")
    if tag_solver != tag_written and (N, tag_solver) not in mismatched:
        mismatched.append((N, tag_solver))
    # An uncapped entry above the solver's sanity limit cannot be produced at
    # all: compute_inversion_angles refuses it outright, in milliseconds. The
    # precompute would record a failure and write nothing while appearing to
    # run, and the sweep would then miss and fall back to a reduced degree. So
    # this is caught here rather than discovered in the log.
    if tag_written == -1 and est > qa._DEGREE_SANITY_LIMIT and N not in impossible:
        impossible.append(N)

if impossible:
    print()
    print(f"  ABORTING: an uncapped solve is impossible at N={impossible}.")
    print("  The estimated degree exceeds the angle solver's sanity limit of "
          f"{qa._DEGREE_SANITY_LIMIT}, so")
    print("  compute_inversion_angles refuses it outright and this job would")
    print("  write nothing while appearing to run.")
    print()
    print("  Set MAX_DEGREE to the cap the runner requests for those")
    print("  resolutions, and run them as a stage separate from the uncapped ones.")
    sys.exit(1)

if mismatched:
    print()
    print("  ABORTING: the degree tag written would not be the tag looked up.")
    for N, tag_solver in mismatched:
        print(f"    N={N}: {runner} requests tag d{tag_solver}, "
              f"this job would write d{tag_written}.")
    print()
    print(f"  The miss would be silent: {runner} would find no entry and compute")
    print("  the phases inline, at whatever degree its own fallback allows, having")
    print("  ignored this entire computation.")
    print()
    print("  Fix: match MAX_DEGREE to the runner's table over the N in scope --")
    print("  1-D: unset for N <= 16, 5000 for N >= 32.")
    print("  2-D/3-D: unset for N <= 16, 500 for N >= 32.")
    print("  Capped and uncapped resolutions cannot share one invocation.")
    sys.exit(1)

print()
print(f"  Cache keys confirmed: this job writes exactly what {runner} requests,")
print(f"  for all {len(targets)} distinct strip operator(s) in scope.")
PY

# ============================================================
#  Precompute
# ============================================================
CAP_ARGS=""
if [ -n "${MAX_DEGREE}" ]; then
    CAP_ARGS="--max-degree ${MAX_DEGREE}"
fi

echo ""
echo "Computing ${DIM}-D order-4 phases for N=${N_VALUES}"
echo "Cache keys that will be written:"
python3 hpc/runners/precompute_phases.py --dim "${DIM}" --order 4 \
        --n-values "${N_VALUES}" --list-kappas

# Pass 1 -- the epsilon the sweep actually requests, alone, so that a walltime
# kill cannot take it while the optional sensitivity entries survive.
echo ""
echo "------------------------------------------------------------"
echo "  PASS 1/2 -- epsilon=${SWEEP_EPSILON} (required by the sweep)  $(date)"
echo "------------------------------------------------------------"
python3 hpc/runners/precompute_phases.py \
        --dim "${DIM}" --order 4 \
        --n-values "${N_VALUES}" \
        --epsilon "${SWEEP_EPSILON}" \
        --extra-epsilons "" \
        ${CAP_ARGS}
RC=$?
echo "PASS 1 finished $(date)  exit=${RC}"

# Pass 2 -- the looser epsilons, for the Phase 8 sensitivity study only. Already
# cached pairs are skipped, so this re-runs nothing from pass 1.
if [ -n "${EXTRA_EPSILONS}" ]; then
    echo ""
    echo "------------------------------------------------------------"
    echo "  PASS 2/2 -- epsilon=${EXTRA_EPSILONS} (sensitivity study)  $(date)"
    echo "------------------------------------------------------------"
    python3 hpc/runners/precompute_phases.py \
            --dim "${DIM}" --order 4 \
            --n-values "${N_VALUES}" \
            --epsilon "${SWEEP_EPSILON}" \
            --extra-epsilons "${EXTRA_EPSILONS}" \
            ${CAP_ARGS}
    RC2=$?
    echo "PASS 2 finished $(date)  exit=${RC2}"
    [ "${RC}" -eq 0 ] && RC=${RC2}
fi

echo ""
echo "============================================================"
echo "  Cache contents after this stage"
echo "============================================================"
ls -1 results/qsvt_phase_cache/ | wc -l | xargs echo "  entries:"

# The cache is the most expensive artefact this pipeline produces -- up to 71 h of
# non-parallelisable computation per stage -- and results/ is not backed up. Copy
# it to RDS, as every sweep job does with its results.
RDS_CACHE="${HOME}/qpde-results/qsvt_phase_cache_4th_${DIM}d_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${RDS_CACHE}"
cp -r results/qsvt_phase_cache/* "${RDS_CACHE}/" 2>/dev/null
echo "  cache copied to: ${RDS_CACHE}"

echo ""
echo "QSVT 4th-order precompute finished $(date)  exit=${RC}"
exit ${RC}
