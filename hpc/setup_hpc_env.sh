#!/bin/bash
# ============================================================
#  setup_hpc_env.sh
#  Run ONCE on the CX3 login node to set up the Python
#  virtual environment for the qpde-solvers project.
#
#  Usage:
#    ssh username@login.cx3.hpc.ic.ac.uk
#    bash hpc/setup_hpc_env.sh
# ============================================================

set -e   # Exit on any error.

# ------------------------------------------------------------
#  quantum_linear_solvers must come from the PROJECT FORK, not
#  from upstream anedumla. The fork carries
#  linear_solvers/matrices/pentadiagonal_toeplitz.py, which the
#  4th-order (pentadiagonal) HHL solver imports; upstream does
#  not have that file at all. Installing upstream here is what
#  caused every 4th-order HHL row to fail with
#  ModuleNotFoundError while the file was present locally.
#
#  The checked-out submodule is preferred over the remote so the
#  installed package matches the commit this repository pins.
#  Falls back to the fork URL when the submodule is not
#  initialised.
# ------------------------------------------------------------
QLS_FORK_URL="https://github.com/jatf-gt/quantum_linear_solvers.git"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QLS_LOCAL="${REPO_ROOT}/quantum_linear_solvers"

install_quantum_linear_solvers() {
    if [ -f "${QLS_LOCAL}/setup.py" ] || [ -f "${QLS_LOCAL}/pyproject.toml" ]; then
        echo "Installing quantum_linear_solvers (editable) from ${QLS_LOCAL}"
        pip install -e "${QLS_LOCAL}"
    else
        echo "Submodule not initialised; installing from ${QLS_FORK_URL}"
        echo "  (run 'git submodule update --init' for a pinned, reproducible install)"
        pip install "git+${QLS_FORK_URL}"
    fi
}

echo "=== Setting up qpde-solvers environment on CX3 ==="

# Load the production module environment.
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

echo "Python: $(which python3) — $(python3 --version)"

# Create the virtual environment in the RDS home directory.
# The RDS home has 1 TB quota and is permanent storage.
VENV_PATH="${HOME}/venvs/qpde"
mkdir -p "${HOME}/venvs"
virtualenv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

echo "Virtual environment created at: ${VENV_PATH}"

# Install all required packages.
# Qiskit and qiskit-aer are the heavy dependencies.
pip install --upgrade pip

# The qiskit pins here deliberately differ from requirements.txt (which targets
# qiskit 1.4.5 for the laptop conda environment). The cluster stack is validated
# against 0.45.3 and the two are not required to match.
#
# pennylane backs VQLS (solvers/quantum/vqls_utils.py) and pyqsp supplies the QSVT
# phase angles. Both were previously absent here and were being installed ad hoc by
# individual submission scripts, so a fresh venv could not run VQLS at all.
pip install \
    qiskit==0.45.3 \
    qiskit-aer==0.13.3 \
    qiskit-algorithms==0.3.0 \
    pennylane==0.45.0 \
    pyqsp==0.2.0 \
    numpy \
    scipy \
    matplotlib \
    pandas \
    openpyxl

install_quantum_linear_solvers

echo ""
echo "=== Environment setup complete ==="
echo "Activate with: source ${VENV_PATH}/bin/activate"
echo "Verify with:   bash hpc/jobs/_preflight.sh"


# ============================================================
#  GPU virtual environment (separate from CPU venv)
#  qiskit-aer-gpu REPLACES qiskit-aer; they cannot coexist
#  in the same environment. The GPU venv is used exclusively
#  with hpc/jobs/submit_hpc_gpu.sh.
# ============================================================

VENV_GPU_PATH="${HOME}/venvs/qpde-gpu"
mkdir -p "${HOME}/venvs"
virtualenv "${VENV_GPU_PATH}"
source "${VENV_GPU_PATH}/bin/activate"

pip install --upgrade pip

# qiskit-aer-gpu provides the same API as qiskit-aer but with
# CUDA 12 GPU support via NVIDIA cuStateVec (cuQuantum).
pip install \
    qiskit-aer-gpu==0.15.1 \
    qiskit-algorithms==0.3.0 \
    pennylane==0.45.0 \
    pyqsp==0.2.0 \
    numpy \
    scipy \
    matplotlib \
    pandas \
    openpyxl

install_quantum_linear_solvers

echo ""
echo "GPU environment: ${VENV_GPU_PATH}"
echo "Verify with: python3 -c \""
echo "  from qiskit_aer import AerSimulator"
echo "  b = AerSimulator(method='statevector', device='GPU')"
echo "  print(b.available_devices())\""