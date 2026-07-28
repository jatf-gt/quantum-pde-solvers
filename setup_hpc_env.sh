#!/bin/bash
# ============================================================
#  setup_hpc_env.sh
#  Run ONCE on the CX3 login node to set up the Python
#  virtual environment for the qpde-solvers project.
#
#  Usage:
#    ssh username@login.cx3.hpc.ic.ac.uk
#    bash setup_hpc_env.sh
# ============================================================

set -e   # Exit on any error.

echo "=== Setting up qpde-solvers environment on CX3 ==="

# Load the production module environment.
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0

echo "Python: $(which python3) — $(python3 --version)"

# Create the virtual environment in the RDS home directory.
# The RDS home has 1 TB quota and is permanent storage.
VENV_PATH="${HOME}/venvs/qpde"
mkdir -p "${HOME}/venv"
virtualenv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

echo "Virtual environment created at: ${VENV_PATH}"

# Install all required packages.
# Qiskit and qiskit-aer are the heavy dependencies.
pip install --upgrade pip

pip install \
    qiskit==0.45.3 \
    qiskit-aer==0.13.3 \
    qiskit-algorithms==0.3.0 \
    numpy \
    scipy \
    matplotlib \
    pandas \
    openpyxl

# Install the quantum_linear_solvers library from GitHub.
pip install git+https://github.com/anedumla/quantum_linear_solvers.git

echo ""
echo "=== Environment setup complete ==="
echo "Activate with: source ${VENV_PATH}/bin/activate"
echo "Test with:     python3 -c 'import qiskit; print(qiskit.__version__)'"


# ============================================================
#  GPU virtual environment (separate from CPU venv)
#  qiskit-aer-gpu REPLACES qiskit-aer; they cannot coexist
#  in the same environment. The GPU venv is used exclusively
#  with submit_hpc_gpu.sh.
# ============================================================

VENV_GPU_PATH="${HOME}/venvs/qpde-gpu"
mkdir -p "${HOME}/venv"
virtualenv "${VENV_GPU_PATH}"
source "${VENV_GPU_PATH}/bin/activate"

pip install --upgrade pip

# qiskit-aer-gpu provides the same API as qiskit-aer but with
# CUDA 12 GPU support via NVIDIA cuStateVec (cuQuantum).
pip install \
    qiskit-aer-gpu==0.15.1 \
    qiskit-algorithms==0.3.0 \
    numpy \
    scipy \
    matplotlib \
    pandas \
    openpyxl

pip install git+https://github.com/anedumla/quantum_linear_solvers.git

echo ""
echo "GPU environment: ${VENV_GPU_PATH}"
echo "Verify with: python3 -c \""
echo "  from qiskit_aer import AerSimulator"
echo "  b = AerSimulator(method='statevector', device='GPU')"
echo "  print(b.available_devices())\""