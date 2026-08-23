#!/bin/bash
# ============================================================================
#  submit_census.sh   -   transpiled circuit resource census
#
#  Runs `scripts/utils/circuit_census.py`, which builds each sweep's operator,
#  transpiles the solver circuit to the Heron r2 native basis and merges the
#  measured depth, gate count and two-qubit gate count into that sweep's
#  `results_full.json`. It is the difference between a hardware-feasibility
#  verdict that is measured and one resting on a heuristic.
#
#  Why this exists as a PBS job
#  ---------------------------
#  The census is post-processing and reads no `.npz`, so most of it is minutes of
#  work. One measurement is not: the fourth-order 1-D HHL circuit grows roughly
#  twentyfold per doubling of N -- 1 839, 40 852 and 793 553 two-qubit gates at
#  N = 4, 8, 16 -- so N = 32 is of order 1.6e7 gates and some five hours of
#  transpilation. That belongs on the cluster.
#
#  Scope of one submission
#  -----------------------
#  One measurement per (operator, solver, N). In 1-D the operator carries no case
#  dependence, so a single measurement serves every case at that (solver, N):
#  the seven estimated rows at 1-D order 4, N = 32 are ONE transpilation and
#  cannot be split across jobs. VQLS and QSVT at the same N are re-measured in
#  passing and cost little -- VQLS transpiles a shallow ansatz, and QSVT goes
#  through the composed estimator in `core.resources.qsvt_resource_estimate`
#  rather than at full degree.
#
#  Usage
#  -----
#    # The outstanding measurement: 1-D order 4 at N=32.
#    export DIM="1" ORDER="4" N_VALUES="32" TIMEOUT_S="36000"
#    qsub -v DIM,ORDER,N_VALUES,TIMEOUT_S hpc/jobs/submit_census.sh
#
#    # A dry run, which reports and writes nothing:
#    export DIM="1" ORDER="4" N_VALUES="32" DRY_RUN="1"
#    qsub -v DIM,ORDER,N_VALUES,DRY_RUN hpc/jobs/submit_census.sh
#
#  N_VALUES is comma-separated and must be passed to qsub as `-v N_VALUES`
#  (bare name, value taken from the exported shell variable); the form
#  `-v N_VALUES=4,8` breaks on PBS's comma-splitting. Leaving it unset censuses
#  EVERY resolution in the sweep, which at 1-D order 4 includes N = 64 -- days of
#  transpilation, and memory-bound. Always set it.
#
#  Retrieval
#  ---------
#  The census writes only `results/<sweep>/results_full.json` and
#  `results_summary.csv`, both tracked in git, plus a timestamped
#  `results_full.<stamp>.pre-census.json` backup beside them. So the measurement
#  comes home as a commit rather than as an scp; the epilogue prints the exact
#  two-line recipe.
#
#  Provenance caveat
#  -----------------
#  A transpiled gate count is a function of the transpiler version. The 474 rows
#  measured before this script were measured in the laptop environment, which
#  carries qiskit 1.4.5; the cluster venv records 1.4.6 in every sweep's
#  run_metadata.json. The job prints the version it used. Record it beside the
#  counts, and note the difference before setting a cluster-measured row against
#  a laptop-measured one.
#
#  Monitor:
#    qstat -u $USER
#    tail -f results/census_pbs_stdout.log
# ============================================================================

# Five hours expected for 1-D order 4 at N=32, on a measured twentyfold growth
# per doubling from ~900 s at N=16. The request is generous rather than tuned: a
# measurement that outruns --timeout-s is abandoned and its row falls back to the
# heuristic, labelled 'estimated', so the job cannot silently half-finish.
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=8:mem=64gb
#PBS -N quantum_pde_census
#PBS -o results/census_pbs_stdout.log
#PBS -e results/census_pbs_stderr.log
#PBS -M juan.trobajo-flecha25@imperial.ac.uk
#PBS -m abe

set -u

echo "============================================================"
echo "  QUANTUM PDE SOLVER - CIRCUIT CENSUS"
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

DIM="${DIM:-1}"
ORDER="${ORDER:-2}"
N_VALUES="${N_VALUES:-}"
TIMEOUT_S="${TIMEOUT_S:-36000}"
DRY_RUN="${DRY_RUN:-0}"

# Refuses a dirty tree, and at ORDER=4 an environment holding the UPSTREAM
# quantum_linear_solvers, which has no pentadiagonal_toeplitz module. A clean
# tree here is also what makes the result retrievable as a commit.
ORDER="${ORDER}" bash hpc/jobs/_preflight.sh || exit 1

SWEEP="results/${DIM}Dhpc_run"
[ "${ORDER}" != "2" ] && SWEEP="${SWEEP}_${ORDER}th"
if [ ! -f "${SWEEP}/results_full.json" ]; then
    echo "ERROR: no sweep summary at ${SWEEP}/results_full.json."
    echo "       The census merges into an existing sweep; it does not create one."
    echo "       Run 'git pull' here first: that file is tracked."
    exit 1
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

OPT_ARGS=""
[ -n "${N_VALUES}" ]    && OPT_ARGS="${OPT_ARGS} --n-values ${N_VALUES}"
[ "${DRY_RUN}" != "0" ] && OPT_ARGS="${OPT_ARGS} --dry-run"

echo ""
echo "  DIM       : ${DIM}"
echo "  ORDER     : ${ORDER}"
echo "  SWEEP     : ${SWEEP}"
echo "  N_VALUES  : ${N_VALUES:-<every N in the sweep -- see the header>}"
echo "  TIMEOUT_S : ${TIMEOUT_S}"
echo "  DRY_RUN   : ${DRY_RUN}"
python3 -c "import qiskit; print('  QISKIT    : ' + qiskit.__version__)"
echo ""

echo "------------------------------------------------------------"
echo "CENSUS START  $(date)"
echo "------------------------------------------------------------"
# The exact command, echoed verbatim: the census rewrites the same filenames
# whatever it was asked for, so the assembled argument vector is the only record
# of what was actually requested.
echo "COMMAND: python3 scripts/utils/circuit_census.py --dim ${DIM}" \
     "--order ${ORDER} --timeout-s ${TIMEOUT_S}${OPT_ARGS}"
echo "------------------------------------------------------------"

python3 scripts/utils/circuit_census.py \
    --dim "${DIM}" \
    --order "${ORDER}" \
    --timeout-s "${TIMEOUT_S}" \
    ${OPT_ARGS}
RC=$?

echo "------------------------------------------------------------"
echo "CENSUS FINISHED  $(date)  exit=${RC}"
echo "------------------------------------------------------------"

# What changed, stated plainly. A measurement that exceeded --timeout-s leaves
# its row on the heuristic and is reported as 'estimated' in the table above;
# the exit status alone does not distinguish that from a complete pass.
echo "  Tracked files now differing from HEAD:"
git status --porcelain "${SWEEP}" || true

# Copy to RDS alongside the sweep results. Small: the census touches two JSON
# files and one CSV, and reads no .npz.
DEST="${HOME}/qpde-results/census_${DIM}D_order${ORDER}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${DEST}"
cp "${SWEEP}"/results_full*.json "${SWEEP}"/results_summary.csv "${DEST}/" 2>/dev/null || true
echo "  Summaries copied to ${DEST}"
echo ""
echo "  To bring the measurement home, from this repository on the cluster:"
echo "    git add ${SWEEP}/results_full.json ${SWEEP}/results_summary.csv"
echo "    git commit -m 'Census: ${DIM}-D order ${ORDER}, N=${N_VALUES}' && git push"

exit ${RC}
