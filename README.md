# Quantum Linear System Resolution of the 1D and 2D Poisson Equations

This repository contains the computational framework developed to evaluate the Harrow-Hassidim-Lloyd (HHL) quantum algorithm for solving the Poisson boundary value problem across one-dimensional and two-dimensional spatial domains. 

The codebase is engineered to systematically replicate and extend the numerical benchmarks detailed in **Ghafourpour & Laizet (2025)**. The 1D formulation employs a direct Toeplitz Symmetric Tridiagonal (TST) operator, whilst the 2D resolution is achieved via a hybrid quantum-classical line-Jacobi iterative decomposition scheme.

## 1. Project Architecture

The codebase has been refactored into a highly modular, decoupled architecture, separating problem instantiation, algorithmic execution, and statistical post-processing:

```text
poisson_hhl/
├── core/                      # Shared infrastructure and configuration
│   ├── config.py              # Strict dataclasses for 1D/2D parameter validation
│   ├── exact_solutions.py     # Analytical derivations for homogeneous constraints
│   └── source_functions.py    # Analytical forcing functions (fS, fL, fH)
├── problems/                  # Domain discretisation and operator assembly
│   ├── poisson_1d.py          # 1D global TST matrix and RHS assembly
│   └── poisson_2d.py          # 2D line-Jacobi sub-problem and reference assembly
├── solvers/                   # Algorithmic resolution implementations
│   ├── classical/             # Thomas algorithm and direct NumPy baselines
│   └── quantum/               # Qiskit-based HHL implementations and state extraction
├── benchmark/                 # Evaluation orchestration and reporting
│   ├── metrics.py             # Purely classical error computation and statistics
│   ├── plotting.py            # Matplotlib visualisation and contour mapping
│   ├── reporting.py           # Standard output tabular formatting
│   └── runner.py              # Execution drivers for Sweeps A through G
├── scripts/                   # Top-level execution entry points
│   ├── run_1d_benchmark.py    
│   └── run_2d_benchmark.py    
├── quantum_linear_solvers/    # Git Submodule: Specialised TST Hamiltonian simulation
├── requirements.txt           # Explicit Python environment dependencies
└── README.md
```

## 2. Prerequisites and Installation

This framework necessitates an isolated Python environment to prevent dependency conflicts, particularly concerning Qiskit quantum information modules and classical scientific computing libraries.

**1. Clone the repository (including the required solver submodule):**
```bash
git clone --recurse-submodules [https://github.com/YourUsername/YourRepository.git](https://github.com/YourUsername/YourRepository.git)
cd YourRepository
```
*(Note: If the repository was cloned without the `--recurse-submodules` flag, execute `git submodule update --init --recursive` to populate the `quantum_linear_solvers` directory).*

**2. Provision the Conda environment:**
```bash
conda create -n msc_qiskit python=3.11
conda activate msc_qiskit
```

**3. Install dependencies and the local quantum submodule:**
```bash
pip install -r requirements.txt
pip install -e quantum_linear_solvers/
```

## 3. Execution Protocols

Execution authority is strictly delegated to the entry-point scripts located within the `scripts/` directory. These modules dynamically resolve system paths, ensuring internal imports function correctly regardless of the active working directory.

### 1D Benchmark Evaluations (Sections IV A–D)
Evaluates direct HHL resolutions against classical Thomas baselines across varying spatial resolutions and Trotterisation precision parameters.
```bash
python scripts/run_1d_benchmark.py
```

### 2D Benchmark Evaluations (Sections IV E–F)
Evaluates the hybrid quantum-classical line-Jacobi iterative scheme.
```bash
python scripts/run_2d_benchmark.py
```

**Execution Time Note (2D):** Simulation of the HHL circuit via Qiskit's `Statevector` is computationally intensive. A standard 2D configuration (e.g., N=8, ε=0.01) typically necessitates 50 to 100 line-Jacobi iterations, equating to hundreds of distinct quantum circuit simulations. Execution times may range from 10 to 30 minutes on standard local hardware. It is recommended to utilise standard configurations prior to expanding spatial resolution (N).

## 4. Benchmark Sweep Directory

The execution runners sequentially process predefined computational sweeps mimicking the primary literature:

* **Sweep A:** 1D Homogeneous constraints; evaluates analytical source functions at N ∈ {8, 16}.
* **Sweep B:** 1D Trotterisation (ε) sensitivity analysis.
* **Sweep C:** 1D Non-homogeneous boundary constraints (asymmetrical Dirichlet conditions).
* **Sweep D:** Verification of theoretical O(N²) condition number scaling for 1D operators.
* **Sweep E:** 2D Homogeneous constraints; iterative line-Jacobi stability.
* **Sweep F:** 2D Non-homogeneous constraints; assesses convergence divergence under specific topological thresholds.
* **Sweep G:** Verification of asymptotic condition number scaling (κ → 3) for the 2D row operator.

## 5. Methodological Notes

* **Statevector Extraction:** The quantum solutions are presently evaluated via deterministic statevector simulations and rigorous post-selection (flag qubit = |1⟩, clock register = |0⟩). Shot-noise simulation via stochastic backends is bypassed to establish baseline theoretical algorithmic accuracy.
* **Physical Hardware Viability:** As acknowledged in contemporary literature, the necessity to extract the comprehensive spatial solution vector at each iterative step of the 2D line-Jacobi cycle poses severe limitations for near-term physical quantum hardware. This framework is therefore strictly utilised as a theoretical simulator to study algorithmic behaviour, Hamiltonian scaling, and pure quantum error propagation.