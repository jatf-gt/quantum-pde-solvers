# Quantum Linear System Solvers for the Poisson Equation: HHL, VQLS, and Application to Hall Effect Thruster Plasma Modelling

This repository contains the computational framework developed for the MSc thesis *"Quantum Algorithms for Coupled Fluid-Thermal Partial Differential Equations: Application to Regenerative Cooling in Rocket Nozzles"*, Department of Aeronautics, Imperial College London (2026).

The codebase implements and benchmarks two quantum linear system algorithms — the Harrow-Hassidim-Lloyd (HHL) algorithm and the Variational Quantum Linear Solver (VQLS) — applied to the Poisson boundary value problem in one and two spatial dimensions. A physically motivated application to the electrostatic Poisson equation in Hall Effect Thruster (HET) plasma modelling is included as a proof-of-concept demonstration.

The 1D formulation employs a direct Toeplitz Symmetric Tridiagonal (TST) operator. The 2D resolution is achieved via a hybrid quantum-classical line-Jacobi iterative decomposition scheme. All results are benchmarked against the Thomas algorithm (classical tridiagonal direct solver) and, where available, against closed-form analytical solutions.

The numerical benchmarks systematically replicate and extend those of **Ghafourpour & Laizet (2025)** (*Physical Review Applied* 24, 024032), with the VQLS implementation following **Bravo-Prieto et al. (2023)** (*Quantum* 7, 1188). The HET plasma application draws on the physical model of **Boeuf & Garrigues (1998)** (*Journal of Applied Physics* 84, 3541).

---

## Table of Contents

1. [Repository Architecture](#1-repository-architecture)
2. [Prerequisites and Installation](#2-prerequisites-and-installation)
3. [Quick-Start Verification](#3-quick-start-verification)
4. [Execution Protocols](#4-execution-protocols)
5. [Benchmark Sweep Directory](#5-benchmark-sweep-directory)
6. [Physical Application: HET Plasma Modelling](#6-physical-application-het-plasma-modelling)
7. [Test Suite](#7-test-suite)
8. [Methodological Notes](#8-methodological-notes)
9. [References](#9-references)

---

## 1. Repository Architecture

The codebase is organised into a strictly layered, decoupled architecture separating problem instantiation, algorithmic execution, post-processing, and physical application. Data flows in one direction only: `core` → `problems` → `solvers` → `benchmark` → `scripts`.

```text
poisson_hhl/
│
├── core/                          # Shared infrastructure — PDE-agnostic
│   ├── config.py                  # SimConfig1D, SimConfig2D, ClassicalConfig2D
│   ├── exact_solutions.py         # Analytical solutions: 1D (fS, fL, fH) and 2D sinusoidal
│   ├── het_config.py              # HETConfig, HETPhysicalConfig, physical constants
│   └── source_functions.py        # Source functions: fS, fL, fH (1D and 2D); HET profiles
│
├── problems/                      # Domain discretisation and operator assembly
│   ├── poisson_1d.py              # 1D TST matrix, RHS, PoissonProblem1D container
│   ├── poisson_2d.py              # 2D line-Jacobi sub-problems, PoissonProblem2D container
│   ├── het_plasma_1d.py           # HET 1D Poisson: HETPoissonProblem1D, HETPhysicalProblem1D
│   └── het_plasma_2d.py           # HET 2D Poisson: HETPoissonProblem2D, HETSinusoidalProblem2D
│
├── solvers/                       # Algorithmic resolution implementations
│   ├── classical/
│   │   ├── thomas.py              # Thomas algorithm: thomas_solve, thomas_solve_system
│   │   ├── thomas_2d.py           # Thomas line-Jacobi for 2D: thomas_solve_2d
│   │   └── numpy_ref.py           # NumPy direct solver (debugging reference)
│   └── quantum/
│       ├── result.py              # SolverResult, VQLSSolverResult, SolverResult2D
│       ├── hhl_1d.py              # HHL 1D: hhl_solve, hhl_solve_system
│       ├── hhl_2d.py              # HHL 2D line-Jacobi: hhl_solve_2d
│       ├── vqls_utils.py          # LCU Pauli decomposition, ansatz, cost function
│       ├── vqls_1d.py             # VQLS 1D: vqls_solve, vqls_solve_system, VQLSConfig1D
│       └── vqls_2d.py             # VQLS 2D line-Jacobi: vqls_solve_2d, VQLSConfig2D
│
├── benchmark/                     # Evaluation orchestration and reporting
│   ├── metrics.py                 # BenchmarkResult, BenchmarkResult2D, compute_errors
│   ├── plotting.py                # Matplotlib: 1D profiles, 2D contours, convergence history
│   ├── reporting.py               # Tabular console output for 1D and 2D results
│   └── runner.py                  # Sweep drivers A-G; run_pair_1d, run_pair_2d
│
├── scripts/                       # Top-level execution entry points
│   ├── run_1d_benchmark.py        # Sweeps A-D: 1D Poisson (Ghafourpour & Laizet 2025)
│   ├── run_2d_benchmark.py        # Sweeps E-G: 2D Poisson (Ghafourpour & Laizet 2025)
│   ├── run_het_benchmark.py       # HET 1D plasma benchmark (Boeuf & Garrigues 1998)
│   ├── run_het_plasma_benchmark.py# HET 1D physical operating condition benchmark
│   ├── run_het_2d_benchmark.py    # HET 2D plasma benchmark (Cases A and B)
│   └── run_verification_study.py  # Cross-dimensional V&V study with figure generation
│
├── tests/                         # Pytest test suite
│   ├── conftest.py                # Shared fixtures and tolerance constants
│   ├── test_problem_setup.py      # Matrix assembly, grid, RHS, condition number
│   ├── test_classical_solvers.py  # Thomas 1D/2D and NumPy reference solvers
│   ├── test_hhl_1d.py             # HHL 1D solver correctness
│   ├── test_hhl_2d.py             # HHL 2D line-Jacobi solver correctness
│   ├── test_vqls_1d.py            # VQLS 1D solver correctness
│   ├── test_vqls_2d.py            # VQLS 2D line-Jacobi solver correctness
│   ├── test_het_problem.py        # HET 1D problem assembly and solver compatibility
│   ├── test_het_2d.py             # HET 2D problem assembly and solver compatibility
│   └── test_integration.py        # End-to-end pipeline: problem -> solver -> metrics
│
├── results/                       # Auto-generated output artefacts (git-ignored)
│   ├── sweep_*.csv
│   ├── het/
│   ├── het_2d/
│   └── verification/
│
├── quantum_linear_solvers/        # Git submodule: TST Hamiltonian simulation (Vazquez et al.)
├── pytest.ini                     # Pytest configuration
├── requirements.txt               # Python environment dependencies
└── README.md
```

---

## 2. Prerequisites and Installation

This framework requires an isolated Python environment to prevent dependency conflicts between Qiskit quantum information modules, PennyLane variational quantum components, and classical scientific computing libraries.

### Supported Python version

Python 3.11 is recommended. Python 3.10 and 3.12 are compatible but untested.

### Step 1 — Clone the repository with the required submodule

```bash
git clone --recurse-submodules https://github.com/YourUsername/YourRepository.git
cd YourRepository
```

> **Note:** If the repository was cloned without the `--recurse-submodules` flag, populate the submodule manually:
>
> ```bash
> git submodule update --init --recursive
> ```
>
> The `quantum_linear_solvers/` directory must be non-empty before proceeding.

### Step 2 — Provision the Conda environment

```bash
conda create -n msc_qiskit python=3.11
conda activate msc_qiskit
```

### Step 3 — Install dependencies and the local quantum submodule

```bash
pip install -r requirements.txt
pip install -e quantum_linear_solvers/
```

### Key dependencies

| Package | Role |
| --- | --- |
| `qiskit` >= 1.0 | Quantum circuit construction and statevector simulation |
| `qiskit-aer` | High-performance statevector backend |
| `pennylane` | VQLS variational optimisation and automatic differentiation |
| `numpy`, `scipy` | Classical linear algebra and optimisation |
| `matplotlib` | Result visualisation |
| `pytest` | Automated test suite |

> **Qiskit 1.0 compatibility note:** The `quantum_linear_solvers` submodule has been patched to replace the deprecated `QuantumCircuit.isometry()` method with the `Isometry` gate from `qiskit.circuit.library`. This patch is applied to the submodule source directly and requires no action from the user.

---

## 3. Quick-Start Verification

Before executing the full benchmark sweeps, run the following verification scripts in order to confirm that all components are functioning correctly. Each script is designed to complete within 5–10 minutes on standard hardware.

### 3.1 — 1D Poisson HHL (generic)

```bash
python quick_test.py
```

Expected output: Thomas residual < 1e-12; HHL residual < 5e-2; `Max |HHL - Thomas|` < 1e-3 for N=8, fS, epsilon=0.01.

### 3.2 — 1D VQLS (generic)

```bash
python quick_test_vqls.py
```

Expected output: VQLS cost < 1e-5; `Max |VQLS - Thomas|` < 1e-4 for N=8, fS, 4 layers.

### 3.3 — 2D Poisson HHL and VQLS

```bash
python quick_test_2d.py
```

Expected output: Thomas-2D converges in approximately 55 iterations; HHL-2D and VQLS-2D produce finite solutions with `Max |solver - Thomas|` < 1e-3 for N=8, fS.

### 3.4 — HET plasma 1D

```bash
python quick_test_het.py
```

Expected output: All three cases (linear/homogeneous, Gaussian/homogeneous, Gaussian/physical BCs) pass with HHL relative error < 5% and VQLS relative error < 2%.

### 3.5 — HET plasma 2D

```bash
python quick_test_het_2d.py
```

Expected output: Thomas-2D and VQLS-2D produce finite solutions; `max|phi_exact|` approximately 1.0 for the sinusoidal case; sign consistency between all solvers.

### 3.6 — Automated test suite

```bash
pytest
```

Expected output: 115+ tests passing in under 4 minutes. The full suite covers problem assembly, classical solvers, HHL 1D/2D, VQLS 1D/2D, HET problem assembly, and end-to-end pipeline integration.

To run only the fast classical tests (milliseconds):

```bash
pytest tests/test_problem_setup.py tests/test_classical_solvers.py
```

---

## 4. Execution Protocols

All benchmark scripts are located in `scripts/` and are designed to be executed from the repository root.

### 4.1 — 1D Poisson benchmark (Sections IV A–D of Ghafourpour & Laizet 2025)

Evaluates HHL and VQLS against the Thomas algorithm across all source functions, mesh sizes, and epsilon values reported in the paper.

```bash
python scripts/run_1d_benchmark.py
```

**Output:** CSV files in `results/`, console tables, and matplotlib figures for each sweep.

### 4.2 — 2D Poisson benchmark (Sections IV E–F of Ghafourpour & Laizet 2025)

Evaluates the hybrid quantum-classical line-Jacobi scheme for HHL and VQLS on the 2D Poisson equation.

```bash
python scripts/run_2d_benchmark.py
```

> **Runtime note:** A single 2D configuration at N=8, epsilon=0.01 requires 50–100 line-Jacobi iterations, each containing N HHL or VQLS circuit simulations. Expected wall-clock time: 20–60 minutes per configuration on standard hardware. It is recommended to verify correctness at N=4 before running N=8.

### 4.3 — HET plasma 1D benchmark

Applies HHL and VQLS to the 1D axial Poisson equation for the electrostatic potential in a Hall Effect Thruster discharge channel, using physical parameters from Boeuf & Garrigues (1998).

```bash
python scripts/run_het_benchmark.py
python scripts/run_het_plasma_benchmark.py
```

### 4.4 — HET plasma 2D benchmark

Applies Thomas, HHL, and VQLS to the 2D HET Poisson equation. Case A uses a sinusoidal source with an exact analytical solution; Case B uses the Boeuf-Garrigues Gaussian profile with physical boundary conditions.

```bash
python scripts/run_het_2d_benchmark.py
```

**Output:** Three PDF figures (solution contours, error maps, electric field vector plots) and a CSV metrics file in `results/het_2d/`.

### 4.5 — Cross-dimensional verification study

Produces a structured verification and validation report across all implemented cases, including a summary comparison table suitable for a progress report or supervisor meeting.

```bash
python scripts/run_verification_study.py
```

**Output:** Four PDF figures and a CSV file in `results/verification/`. Estimated runtime: 15–30 minutes.

---

## 5. Benchmark Sweep Directory

The execution runners in `benchmark/runner.py` sequentially process predefined computational sweeps replicating the primary literature benchmarks.

### Generic Poisson sweeps

| Sweep | Dimension | Description |
| --- | --- | --- |
| A | 1D | Homogeneous BCs; source functions fS, fL, fH; N in {8, 16}; epsilon = 0.01 |
| B | 1D | Trotter/VQLS epsilon sensitivity; N = 16; epsilon in {0.1, 0.01, 0.001} |
| C | 1D | Non-homogeneous Dirichlet BCs; fH; N in {16, 32} |
| D | 1D | Condition number scaling kappa(A) ~ (4/pi^2)(N+1)^2; N in {4, 8, 16, 32} |
| E | 2D | Homogeneous BCs; fS, fL, fH; N in {8, 16}; epsilon in {0.01, 0.5} |
| F | 2D | Non-homogeneous BCs; convergence behaviour under asymmetric conditions |
| G | 2D | Row matrix condition number kappa(A_row) -> 3^-; N in {4, 8, 16, 32} |

### HET plasma sweeps

| Sweep | Description |
| --- | --- |
| H1 | 1D Gaussian profile, homogeneous BCs; N in {4, 8} |
| H2 | 1D Gaussian profile, physical BCs (V_d = 300 V); N in {4, 8} |
| H3 | 1D all three charge density profiles; N = 8; physical BCs |
| H4 | Condition number and alpha = L^2/lambda_D^2 scaling diagnostics |

---

## 6. Physical Application: HET Plasma Modelling

### Problem formulation

The electrostatic potential phi in the discharge channel of a Hall Effect Thruster satisfies the Poisson equation. After non-dimensionalisation using the Debye length lambda_D and the electron thermal voltage phi_0 = k_B T_e / e, the governing equation becomes:

$$
\begin{aligned}
\frac{d^2 \tilde{\phi}}{d\tilde{x}^2} &= -\alpha \cdot \delta\tilde{n}(\tilde{x}) && \text{(1D axial)} \\
\frac{\partial^2 \tilde{\phi}}{\partial\tilde{x}^2} + \frac{\partial^2 \tilde{\phi}}{\partial\tilde{y}^2} &= -\alpha \cdot \delta\tilde{n}(\tilde{x}, \tilde{y}) && \text{(2D axial-radial)}
\end{aligned}
$$

where $\alpha = L^2 / \lambda_D^2$ is the dimensionless Debye scaling parameter and $\delta\tilde{n} = (n_i - n_e) / n_0$ is the non-dimensional net charge density.

### Physical parameters (Boeuf & Garrigues 1998, Table 1)

| Parameter | Symbol | Value |
| --- | --- | --- |
| Channel length | L | 25 mm |
| Discharge voltage | V_d | 300 V |
| Electron temperature | T_e | 20 eV |
| Reference density | n_0 | 5 x 10^17 m^-3 |
| Debye length | lambda_D | ~128 micrometres |
| Scaling parameter | alpha = L^2 / lambda_D^2 | ~38 000 |

### Charge density profiles

Three physically motivated source term profiles are implemented:

| Key | Profile | Description |
| --- | --- | --- |
| `gaussian` | delta_n = delta_0 exp(-(x - x_peak)^2 / sigma^2) | Smooth ionisation zone near exit plane |
| `linear` | delta_n = delta_0 * x | Uniform space charge gradient; analytical solution available |
| `step` | delta_n = delta_0 * sign(x - x_ion) | Sharp ionisation front |

The charge separation amplitude is set to $\delta_0 = \delta_{0_\text{factor}} / \alpha$ (default $\delta_{0_\text{factor}} = 5$) to ensure the space charge contribution $\alpha \cdot \delta_0 = \mathcal{O}(1)$ remains a physically realistic small perturbation on the applied voltage $\alpha_{\text{bc}} = V_d / \phi_0 \sim 15$.

### 2D analytical solution

For the sinusoidal source term $f(\tilde{x}, \tilde{y}) = -2\pi^2 \sin(\pi \tilde{x}) \sin(\pi \tilde{y})$ with homogeneous Dirichlet boundary conditions, the exact analytical solution is:

$$
\tilde{\phi}(\tilde{x}, \tilde{y}) = \sin(\pi \tilde{x}) \cdot \sin(\pi \tilde{y})
$$

This manufactured solution enables rigorous quantitative error assessment of all three solvers (Thomas, HHL, VQLS) in two dimensions without dependence on a numerical reference.

### Key results (development benchmarks, N = 4–8)

- HHL achieves relative errors of 1–5% against the analytical solution for epsilon = 0.01, consistent with the Trotter approximation error
- VQLS achieves relative errors of 0.1–2% with COBYLA optimisation (cost < 10^-6), outperforming HHL at equivalent resolution
- The peak electric field for the physical operating condition (V_d = 300 V, Gaussian profile) is of order 10^4 V/m, in qualitative agreement with Boeuf & Garrigues (1998), Fig. 3

---

## 7. Test Suite

The automated test suite is located in `tests/` and is executed via `pytest`. Tests are designed to verify structural correctness and solver functionality rather than publication-level numerical accuracy. All quantum solver tests use N=4 (2 qubits) to bound individual test runtime.

### Test file summary

| File | Coverage | Approx. runtime |
| --- | --- | --- |
| `test_problem_setup.py` | Matrix structure, grid, RHS, config validation, exact solutions | < 1 s |
| `test_classical_solvers.py` | Thomas 1D/2D accuracy, NumPy agreement, convergence | < 5 s |
| `test_hhl_1d.py` | HHL 1D solution shape, sign, proportionality recovery | ~2 min |
| `test_hhl_2d.py` | HHL 2D structure, sign consistency, iteration history | ~3 min |
| `test_vqls_1d.py` | VQLS cost convergence, parameter shape, reproducibility | ~2 min |
| `test_vqls_2d.py` | VQLS 2D structure, warm-start, sign consistency | ~3 min |
| `test_het_problem.py` | HET config derived quantities, matrix structure, solver compatibility | ~2 min |
| `test_het_2d.py` | HET 2D assembly, boundary conditions, solver compatibility | ~3 min |
| `test_integration.py` | End-to-end pipeline, BenchmarkResult consistency | ~2 min |

### Running the tests

```bash
# Full suite
pytest

# Fast classical tests only (under 10 seconds)
pytest tests/test_problem_setup.py tests/test_classical_solvers.py

# Single test file
pytest tests/test_vqls_1d.py -v

# Single test function
pytest tests/test_hhl_1d.py::TestHHL1D::test_agrees_with_thomas_loose -v
```

### Pass/fail criteria

Tests verify that solvers:

- Return results of the correct shape and type
- Produce finite (non-NaN, non-Inf) solution values
- Agree with the Thomas reference to within a loose tolerance (20% for HHL, 15% for VQLS at N=4)
- Preserve the correct sign of the dominant solution component
- Raise appropriate exceptions for invalid inputs (non-Hermitian matrix, non-power-of-2 system size, zero RHS)

---

## 8. Methodological Notes

### Statevector simulation

All quantum circuits are evaluated via deterministic statevector simulation using Qiskit's `Statevector` class from `qiskit.quantum_info`. Shot-noise simulation via stochastic backends is bypassed to establish baseline theoretical algorithmic accuracy, consistent with the methodology of Ghafourpour & Laizet (2025).

### HHL solution extraction

The HHL output state is post-selected on the ancilla (flag) qubit being in state |1> and the clock (QPE) register being in state |0...0>. The b-register amplitudes satisfying this condition are extracted from the full statevector using explicit bit-masking in Qiskit's little-endian convention. The proportionality constant c is recovered against the normalised system `A_norm = A / norm(A)_2` to prevent amplification of Trotter approximation errors by the large factor `norm(b) / norm(A)_2` that arises in physically scaled problems (e.g., the HET case where $\|b\| \sim \alpha \cdot h^2 \sim 700$).

### VQLS implementation

The VQLS cost function is:

$$
C(\boldsymbol{\theta}) = 1 - \frac{|\langle b | A | x(\boldsymbol{\theta}) \rangle|^2}{\langle x(\boldsymbol{\theta}) | A^\dagger A | x(\boldsymbol{\theta}) \rangle}
$$

evaluated via direct statevector arithmetic on the Pauli LCU decomposition of A. The ansatz is a hardware-efficient layered circuit of RY rotations and nearest-neighbour CNOT gates with `n_qubits * (n_layers + 1)` variational parameters. Optimisation uses COBYLA with a three-stage restart strategy (`rhobeg` in {0.5, 0.1, 0.01}) to escape local minima. The proportionality constant is recovered using the same normalised-system approach as HHL.

### 2D line-Jacobi decomposition

The 2D Poisson equation is decomposed into a sequence of 1D TST sub-problems via the line-Jacobi scheme (Ghafourpour & Laizet 2025, Eq. 9):

$$
u^{n+1}_{i+1,j} - 4 u^{n+1}_{i,j} + u^{n+1}_{i-1,j} = h^2 f(x_i, y_j) - (u^n_{i,j-1} + u^n_{i,j+1})
$$

Each row sub-problem has a TST matrix with `a = -4`, `b = 1` and condition number `kappa(A_row) -> 3^-` as N -> infinity — far more favourable than the O(N^2) scaling of the 1D Poisson matrix. The VQLS 2D solver additionally employs warm-starting: the optimal variational parameters from iteration n serve as the initial guess for iteration n+1, exploiting the slow variation of the RHS between Jacobi iterates.

### Physical hardware viability

Extracting the full solution vector at each step of the 2D line-Jacobi cycle is not physically realisable on near-term quantum hardware, as it requires O(N) quantum state tomography operations per iteration. This framework is therefore strictly a theoretical simulator study of algorithmic behaviour, Hamiltonian scaling, and quantum error propagation, consistent with the stated scope of the project.

---

## 9. References

1. Ghafourpour, L. & Laizet, S. (2025). Applicability of solving the one- and two-dimensional Poisson equations with the quantum Harrow-Hassidim-Lloyd algorithm. *Physical Review Applied*, 24, 024032.

2. Harrow, A. W., Hassidim, A. & Lloyd, S. (2009). Quantum algorithm for linear systems of equations. *Physical Review Letters*, 103, 150502.

3. Bravo-Prieto, C., LaRose, R., Cerezo, M., Subasi, Y., Cincio, L. & Coles, P. J. (2023). Variational quantum linear solver. *Quantum*, 7, 1188.

4. Vazquez, A. C., Hiptmair, R. & Woerner, S. (2022). Enhancing the quantum linear systems algorithm using Richardson extrapolation. *ACM Transactions on Quantum Computing*, 3, 1.

5. Boeuf, J. P. & Garrigues, L. (1998). Low frequency oscillations in a stationary plasma thruster. *Journal of Applied Physics*, 84(7), 3541–3554.

6. Brearley, P. & Laizet, S. (2024). Quantum algorithm for solving the advection equation using Hamiltonian simulation. *Physical Review A*, 110, 012430.

7. Over, P., Bengoechea, S., Brearley, P., Laizet, S. & Rung, T. (2025). Quantum algorithm for the advection-diffusion equation by direct block encoding of the time-marching operator. *Physical Review A*, 112, L010401.

8. Tennie, F., Laizet, S., Lloyd, S. & Magri, L. (2025). Quantum computing for nonlinear differential equations and turbulence. *Nature Reviews Physics*, 7, 220–230.
