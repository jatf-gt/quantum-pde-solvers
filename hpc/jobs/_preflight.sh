#!/bin/bash
# ============================================================================
#  _preflight.sh
#
#  Environment and provenance gate, run by every submission script before any
#  compute is dispatched.
#
#  Rationale
#  ---------
#  Two failure modes have each cost a full cluster allocation:
#
#    1. A 21 h 4th-order job in which every HHL row failed with
#       ModuleNotFoundError, because the venv held the UPSTREAM
#       quantum_linear_solvers (no pentadiagonal_toeplitz.py) rather than the
#       project fork. The defect was invisible until the results were read back.
#
#    2. Runs executed against an uncommitted working tree. Every
#       run_metadata.json on disk records "git_dirty": true, so the code that
#       produced those results cannot be reconstructed from git history.
#
#  Both are detectable in under a second. This script detects them, repairs the
#  first where it can, and refuses to proceed otherwise.
#
#  Usage
#  -----
#    bash hpc/jobs/_preflight.sh            # standalone check
#    bash hpc/jobs/_preflight.sh || exit 1  # from a submission script
#
#  Environment knobs
#  -----------------
#    ORDER=4                 Require the pentadiagonal module (default: 2).
#    PREFLIGHT_ALLOW_DIRTY=1 Permit an uncommitted tree. Records the fact rather
#                            than aborting. Use only when knowingly debugging.
#    PREFLIGHT_NO_HEAL=1     Do not attempt to reinstall a missing fork.
# ============================================================================

QLS_FORK_URL="https://github.com/jatf-gt/quantum_linear_solvers.git"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QLS_LOCAL="${REPO_ROOT}/quantum_linear_solvers"
ORDER="${ORDER:-2}"

failures=0
warnings=0

pass()  { printf '  [ ok ]  %s\n' "$1"; }
warn()  { printf '  [warn]  %s\n' "$1"; warnings=$((warnings + 1)); }
fail()  { printf '  [FAIL]  %s\n' "$1"; failures=$((failures + 1)); }

# `python3 -c` returns 0 on a clean import. stderr is suppressed so the caller
# sees the one-line verdict rather than a traceback; the traceback is re-emitted
# only when a required import fails.
can_import() { python3 -c "import $1" 2>/dev/null; }

echo ""
echo "=============================================================================="
echo "  PREFLIGHT  -  repo=${REPO_ROOT}"
echo "  order=${ORDER}  python=$(python3 --version 2>&1)"
echo "=============================================================================="

# -- Provenance ---------------------------------------------------------------
if git -C "${REPO_ROOT}" rev-parse --git-dir >/dev/null 2>&1; then
    commit="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
    if [ -z "$(git -C "${REPO_ROOT}" status --porcelain)" ]; then
        pass "working tree clean at ${commit}"
    elif [ "${PREFLIGHT_ALLOW_DIRTY}" = "1" ]; then
        warn "working tree DIRTY at ${commit} (permitted by PREFLIGHT_ALLOW_DIRTY=1)"
        git -C "${REPO_ROOT}" status --porcelain | sed 's/^/            /'
    else
        fail "working tree DIRTY at ${commit}; results would not be reproducible"
        git -C "${REPO_ROOT}" status --porcelain | sed 's/^/            /'
        echo "            Commit the changes, or set PREFLIGHT_ALLOW_DIRTY=1 to override."
    fi
else
    warn "not a git repository; provenance cannot be recorded"
fi

# -- Interpreter and third-party stack ----------------------------------------
[ -n "${VIRTUAL_ENV}" ] && pass "venv active: ${VIRTUAL_ENV}" \
                        || warn "no virtualenv active; using $(command -v python3)"

for module in numpy scipy qiskit; do
    can_import "${module}" && pass "${module}" || fail "${module} not importable"
done

can_import qiskit_aer     && pass "qiskit_aer" \
                          || fail "qiskit_aer not importable (Aer backend unavailable)"
can_import qiskit_algorithms && pass "qiskit_algorithms  (HHL)" \
                          || fail "qiskit_algorithms not importable; HHL will fail on every row"
can_import pennylane      && pass "pennylane  (VQLS)" \
                          || fail "pennylane not importable; VQLS will fail on every row"

# pyqsp is optional: qsp_angles falls back to a Chebyshev construction when it is
# absent, at some cost in phase-angle quality. Absence is a warning, not a stop.
can_import pyqsp && pass "pyqsp  (QSVT phase angles)" \
                 || warn "pyqsp absent; QSVT will use the Chebyshev fallback"

# -- quantum_linear_solvers: the fork, not upstream ---------------------------
TRI="quantum_linear_solvers.linear_solvers.matrices.tridiagonal_toeplitz"
PENTA="quantum_linear_solvers.linear_solvers.matrices.pentadiagonal_toeplitz"

can_import "${TRI}" && pass "quantum_linear_solvers (tridiagonal)" \
                    || fail "quantum_linear_solvers not importable"

if [ "${ORDER}" = "4" ]; then
    if can_import "${PENTA}"; then
        pass "quantum_linear_solvers (pentadiagonal)  - fork confirmed"
    elif [ "${PREFLIGHT_NO_HEAL}" = "1" ]; then
        fail "pentadiagonal module absent and healing disabled"
    else
        warn "pentadiagonal module absent - the UPSTREAM package is installed."
        echo "            Reinstalling from the project fork..."
        if [ -f "${QLS_LOCAL}/setup.py" ] || [ -f "${QLS_LOCAL}/pyproject.toml" ]; then
            pip install --force-reinstall --no-deps -e "${QLS_LOCAL}" >/dev/null 2>&1
        else
            pip install --force-reinstall --no-deps "git+${QLS_FORK_URL}" >/dev/null 2>&1
        fi
        if can_import "${PENTA}"; then
            pass "pentadiagonal module installed from the fork"
        else
            fail "could not install the fork; 4th-order HHL cannot run"
            python3 -c "import ${PENTA}" 2>&1 | sed 's/^/            /'
        fi
    fi
fi

# -- Project packages ---------------------------------------------------------
# pytest.ini sets pythonpath=., but a PBS job invoking python3 directly does not
# read it, so the runners rely on being launched from the repository root.
cd "${REPO_ROOT}" || exit 1
for module in core.cases solvers.outer solvers.backend_factory benchmark.results_io; do
    can_import "${module}" && pass "${module}" || fail "${module} not importable"
done

# -- Backend report -----------------------------------------------------------
device="$(python3 -c "
from solvers.backend_factory import get_aer_backend
b = get_aer_backend()
print(getattr(b.options, 'device', 'unknown'))
" 2>/dev/null)"
[ -n "${device}" ] && pass "Aer backend device: ${device}" \
                   || warn "could not resolve an Aer backend"

# -- Verdict ------------------------------------------------------------------
echo "------------------------------------------------------------------------------"
if [ "${failures}" -gt 0 ]; then
    echo "  PREFLIGHT FAILED  -  ${failures} error(s), ${warnings} warning(s)"
    echo "  Aborting before any walltime is consumed."
    echo "=============================================================================="
    echo ""
    exit 1
fi
echo "  PREFLIGHT PASSED  -  ${warnings} warning(s)"
echo "=============================================================================="
echo ""
exit 0
