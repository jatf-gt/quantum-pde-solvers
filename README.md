# Quantum Linear System Solvers for the Poisson Equation: HHL, VQLS, and QSVT Applied to Hall Effect Thruster Plasma Modelling

This repository contains the computational framework developed for the MSc thesis *"Quantum Algorithms for Coupled Fluid-Thermal Partial Differential Equations: Application to Regenerative Cooling in Rocket Nozzles"*, Department of Aeronautics, Imperial College London (2026).

The codebase implements and benchmarks three quantum linear system algorithms — the Harrow-Hassidim-Lloyd (HHL) algorithm, the Variational Quantum Linear Solver (VQLS), and the Quantum Singular Value Transformation (QSVT) — applied to the Poisson boundary value problem in one and two spatial dimensions. A physically motivated application to the electrostatic Poisson equation in Hall Effect Thruster (HET) plasma modelling is included as a proof-of-concept demonstration.

The 1D formulation employs a direct Toeplitz Symmetric Tridiagonal (TST) operator. The 2D resolution is achieved via a hybrid quantum-classical line-Jacobi iterative decomposition scheme. All results are benchmarked against the Thomas algorithm (classical tridiagonal direct solver) and, where available, against closed-form analytical solutions.

The numerical benchmarks systematically replicate and extend those of **Ghafourpour & Laizet (2025)** (*Physical Review Applied* 24, 024032), with the VQLS implementation following **Bravo-Prieto et al. (2023)** (*Quantum* 7, 1188) and the QSVT implementation following **Gilyen et al. (2019)** (*STOC 2019*) and **Martyn et al. (2021)** (*PRX Quantum* 2, 040203). The HET plasma application draws on the physical model of **Boeuf & Garrigues (1998)** (*Journal of Applied Physics* 84, 3541).

---

## Table of Contents

1. [Repository Architecture](#1-repository-architecture)
2. [Prerequisites and Installation](#2-prerequisites-and-installation)
3. [Execution Protocols](#3-execution-protocols)
4. [Benchmark Sweep Directory](#4-benchmark-sweep-directory)
5. [Physical Application: HET Plasma Modelling](#5-physical-application-het-plasma-modelling)
6. [Algorithm Summary](#6-algorithm-summary)
7. [Test Suite](#7-test-suite)
8. [Methodological Notes](#8-methodological-notes)
9. [References](#9-references)

---

## 1. Repository Architecture

The codebase is organised into a strictly layered, decoupled architecture separating problem instantiation, algorithmic execution, post-processing, and physical application. Data flows in one direction only: `core` → `problems` → `solvers` → `benchmark` → `scripts`.

```text
poisson_hhl/
│
├── core/                            # Shared infrastructure — PDE-agnostic
│   ├── config.py                    # SimConfig1D, SimConfig2D, ClassicalConfig2D
│   ├── exact_solutions.py           # Analytical solutions: 1D (fS, fL, fH) and 2D sinusoidal
│   ├── het_config.py                # HETConfig, HETPhysicalConfig, physical constants
│   └── source_functions.py          # Source functions: fS, fL, fH (1D and 2D); HET profiles
│
├── problems/                        # Domain discretisation and operator assembly
│   ├── poisson_1d.py                # 1D TST matrix, RHS, PoissonProblem1D container
│   ├── poisson_2d.py                # 2D line-Jacobi sub-problems, PoissonProblem2D container
│   ├── het_plasma_1d.py             # HET 1D: HETPoissonProblem1D, HETPhysicalProblem1D
│   └── het_plasma_2d.py             # HET 2D: HETPoissonProblem2D, HETSinusoidalProblem2D
│
├── solvers/                         # Algorithmic resolution implementations
│   ├── classical/
│   │   ├── thomas.py                # Thomas algorithm: thomas_solve, thomas_solve_system
│   │   ├── thomas_2d.py             # Thomas line-Jacobi for 2D: thomas_solve_2d
│   │   └── numpy_ref.py             # NumPy direct solver (debugging reference)
│   └── quantum/
│       ├── result.py                # SolverResult, VQLSSolverResult, QSVTSolverResult, SolverResult2D
│       ├── hhl_1d.py                # HHL 1D: hhl_solve, hhl_solve_system
│       ├── hhl_2d.py                # HHL 2D line-Jacobi: hhl_solve_2d
│       ├── vqls_utils.py            # LCU Pauli decomposition, ansatz, cost function
│       ├── vqls_1d.py               # VQLS 1D: vqls_solve, vqls_solve_system, VQLSConfig1D
│       ├── vqls_2d.py               # VQLS 2D line-Jacobi: vqls_solve_2d, VQLSConfig2D
│       ├── block_encoding.py        # Sz.-Nagy block encoding for TST matrices
│       ├── qsp_angles.py            # QSP phase angle computation via pyqsp / Chebyshev
│       ├── qsvt_1d.py               # QSVT 1D: qsvt_solve, qsvt_solve_system, QSVTConfig
│       └── qsvt_2d.py               # QSVT 2D line-Jacobi: qsvt_solve_2d, QSVTConfig2D
│
├── benchmark/                       # Evaluation orchestration and reporting
│   ├── metrics.py                   # BenchmarkResult, BenchmarkResult2D, compute_errors
│   ├── plotting.py                  # Matplotlib: 1D profiles, 2D contours, convergence history
│   ├── reporting.py                 # Tabular console output for 1D and 2D results
│   └── runner.py                    # Sweep drivers A-G; run_pair_1d, run_pair_2d
│
├── scripts/                         # Top-level execution entry points
│   ├── run_1d_benchmark.py          # Sweeps A-D: 1D Poisson (Ghafourpour & Laizet 2025)
│   ├── run_2d_benchmark.py          # Sweeps E-G: 2D Poisson (Ghafourpour & Laizet 2025)
│   ├── run_het_benchmark.py         # HET 1D plasma benchmark (Boeuf & Garrigues 1998)
│   ├── run_het_plasma_benchmark.py  # HET 1D physical operating condition benchmark
│   ├── run_het_2d_benchmark.py      # HET 2D plasma benchmark (Cases A and B)
│   ├── run_verification_study.py    # Cross-dimensional V&V study with figure generation
│   └── run_meeting4_report.py       # Supervisor progress report: HHL vs VQLS vs QSVT
│
├── tests/                           # Pytest test suite
│   ├── conftest.py                  # Shared fixtures and tolerance constants
│   ├── test_problem_setup.py        # Matrix assembly, grid, RHS, condition number
│   ├── test_classical_solvers.py    # Thomas 1D/2D and NumPy reference solvers
│   ├── test_hhl_1d.py               # HHL 1D solver correctness
│   ├── test_hhl_2d.py               # HHL 2D line-Jacobi solver correctness
│   ├── test_vqls_1d.py              # VQLS 1D solver correctness
│   ├── test_vqls_2d.py              # VQLS 2D line-Jacobi solver correctness
│   ├── test_het_problem.py          # HET 1D problem assembly and solver compatibility
│   ├── test_het_2d.py               # HET 2D problem assembly and solver compatibility
│   ├── test_qsvt_1d.py              # QSVT 1D block encoding, phase angles, solver correctness
│   ├── test_qsvt_2d.py              # QSVT 2D line-Jacobi solver correctness
│   └── test_integration.py          # End-to-end pipeline: problem → solver → metrics
│
├── results/                         # Auto-generated output artefacts (git-ignored)
│   ├── sweep_*.csv
│   ├── het/
│   ├── het_2d/
│   ├── verification/
│   └── meeting_report/
│
├── quantum_linear_solvers/          # Git submodule: TST Hamiltonian simulation (Vázquez et al.)
├── pytest.ini                       # Pytest configuration
├── requirements.txt                 # Python environment dependencies
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
| `pyqsp` | QSP phase angle computation for QSVT matrix inversion polynomial |
| `numpy`, `scipy` | Classical linear algebra and optimisation |
| `openpyxl` | Excel workbook export for benchmark metrics |
| `matplotlib` | Result visualisation |
| `pytest` | Automated test suite |

> **Qiskit 1.0 compatibility note:** The `quantum_linear_solvers` submodule has been patched to replace the deprecated `QuantumCircuit.isometry()` method with the `Isometry` gate from `qiskit.circuit.library`. This patch is applied to the submodule source directly and requires no action from the user.

---

## 3. Execution Protocols

All benchmark scripts are located in `scripts/` and are designed to be executed from the repository root.

### 3.1 — Supervisor progress report (HHL vs VQLS vs QSVT)

Produces the full algorithm comparison report across five sections: 1D algorithm comparison, QSVT circuit complexity analysis, HET 1D plasma application, HET 2D plasma application, and 2D QSVT demonstration. Generates five PDF figures and an Excel workbook.

```bash
python scripts/run_meeting4_report.py
```

**Estimated runtime:** 60–120 minutes on a standard laptop. **Output:** `results/meeting_report/`.

### 3.2 — 1D Poisson benchmark (Sections IV A–D of Ghafourpour & Laizet 2025)

Evaluates HHL and VQLS against the Thomas algorithm across all source functions, mesh sizes, and $\varepsilon$ values reported in the paper.

```bash
python scripts/run_1d_benchmark.py
```

**Output:** CSV files in `results/`, console tables, and matplotlib figures for each sweep.

### 3.3 — 2D Poisson benchmark (Sections IV E–F of Ghafourpour & Laizet 2025)

Evaluates the hybrid quantum-classical line-Jacobi scheme for HHL and VQLS on the 2D Poisson equation.

```bash
python scripts/run_2d_benchmark.py
```

> **Runtime note:** A single 2D configuration at $N=8$, $\varepsilon=0.01$ requires 50–100 line-Jacobi iterations, each containing $N$ HHL or VQLS circuit simulations. Expected wall-clock time: 20–60 minutes per configuration on standard hardware.

### 3.4 — HET plasma 1D benchmark

Applies HHL and VQLS to the 1D axial Poisson equation for the electrostatic potential in a Hall Effect Thruster discharge channel.

```bash
python scripts/run_het_benchmark.py
python scripts/run_het_plasma_benchmark.py
```

### 3.5 — HET plasma 2D benchmark

Applies Thomas, HHL, and VQLS to the 2D HET Poisson equation. Case A uses a sinusoidal source with an exact analytical solution; Case B uses the Boeuf-Garrigues Gaussian profile with physical boundary conditions.

```bash
python scripts/run_het_2d_benchmark.py
```

**Output:** Three PDF figures and a CSV metrics file in `results/het_2d/`.

### 3.6 — Cross-dimensional verification study

Produces a structured verification and validation report across all implemented cases.

```bash
python scripts/run_verification_study.py
```

**Output:** Four PDF figures and a CSV file in `results/verification/`. Estimated runtime: 15–30 minutes.

### 3.7 — Automated test suite

```bash
pytest
```

To run only the fast classical tests (under 10 seconds):

```bash
pytest tests/test_problem_setup.py tests/test_classical_solvers.py
```

---

## 4. Benchmark Sweep Directory

The execution runners in `benchmark/runner.py` sequentially process predefined computational sweeps replicating the primary literature benchmarks.

### Generic Poisson sweeps

| Sweep | Dimension | Description |
| --- | --- | --- |
| A | 1D | Homogeneous BCs; source functions fS, fL, fH; $N \in \{8, 16\}$; $\varepsilon = 0.01$ |
| B | 1D | Trotter/VQLS $\varepsilon$ sensitivity; $N = 16$; $\varepsilon \in \{0.1, 0.01, 0.001\}$ |
| C | 1D | Non-homogeneous Dirichlet BCs; fH; $N \in \{16, 32\}$ |
| D | 1D | Condition number scaling $\kappa(A) \sim (4/\pi^2)(N+1)^2$; $N \in \{4, 8, 16, 32\}$ |
| E | 2D | Homogeneous BCs; fS, fL, fH; $N \in \{8, 16\}$; $\varepsilon \in \{0.01, 0.5\}$ |
| F | 2D | Non-homogeneous BCs; convergence behaviour under asymmetric conditions |
| G | 2D | Row matrix condition number $\kappa(A_\text{row}) \to 3^-$; $N \in \{4, 8, 16, 32\}$ |

### HET plasma sweeps

| Sweep | Description |
| --- | --- |
| H1 | 1D Gaussian profile, homogeneous BCs; $N \in \{4, 8\}$ |
| H2 | 1D Gaussian profile, physical BCs ($V_d = 300$ V); $N \in \{4, 8\}$ |
| H3 | 1D all three charge density profiles; $N = 8$; physical BCs |
| H4 | Condition number and $\alpha = L^2/\lambda_D^2$ scaling diagnostics |

---

## 5. Physical Application: HET Plasma Modelling

### Problem formulation

The electrostatic potential $\tilde{\phi}$ in the discharge channel of a Hall Effect Thruster satisfies the non-dimensionalised Poisson equation:

$$
\begin{aligned}
\frac{d^2 \tilde{\phi}}{d\tilde{x}^2} &= -\alpha \, \delta\tilde{n}(\tilde{x}) && \text{(1D axial)} \\[6pt]
\frac{\partial^2 \tilde{\phi}}{\partial \tilde{x}^2} + \frac{\partial^2 \tilde{\phi}}{\partial \tilde{y}^2} &= -\alpha \, \delta\tilde{n}(\tilde{x}, \tilde{y}) && \text{(2D axial-radial)}
\end{aligned}
$$

where $\alpha = L^2 / \lambda_D^2$ is the dimensionless Debye scaling parameter and $\delta\tilde{n} = (n_i - n_e)/n_0$ is the non-dimensional net charge density.

### Physical parameters (Boeuf & Garrigues 1998, Table 1)

| Parameter | Symbol | Value |
| --- | --- | --- |
| Channel length | $L$ | 25 mm |
| Discharge voltage | $V_d$ | 300 V |
| Electron temperature | $T_e$ | 20 eV |
| Reference density | $n_0$ | $5 \times 10^{17}$ m$^{-3}$ |
| Debye length | $\lambda_D$ | $\approx 128\,\mu$m |
| Scaling parameter | $\alpha = L^2/\lambda_D^2$ | $\approx 38\,000$ |

### Charge density profiles

Three physically motivated source term profiles are implemented:

| Key | Profile | Description |
| --- | --- | --- |
| `gaussian` | $\delta\tilde{n} = \delta_0 \exp\!\left(-(\tilde{x}-\tilde{x}_\text{peak})^2/\sigma^2\right)$ | Smooth ionisation zone near exit plane |
| `linear` | $\delta\tilde{n} = \delta_0\,\tilde{x}$ | Uniform space charge gradient; analytical solution available |
| `step` | $\delta\tilde{n} = \delta_0\,\mathrm{sign}(\tilde{x} - \tilde{x}_\text{ion})$ | Sharp ionisation front |

The charge separation amplitude is set to $\delta_0 = \delta_{0,\text{factor}}/\alpha$ (default $\delta_{0,\text{factor}} = 5$) to ensure $\alpha\,\delta_0 = \mathcal{O}(1)$, keeping the space charge contribution a physically realistic small perturbation on the applied voltage $\alpha_\text{bc} = V_d/\phi_0 \approx 15$.

### 2D analytical solution

For the sinusoidal source term $f(\tilde{x}, \tilde{y}) = -2\pi^2 \sin(\pi\tilde{x})\sin(\pi\tilde{y})$ with homogeneous Dirichlet boundary conditions, the exact analytical solution is:

$$
\tilde{\phi}(\tilde{x}, \tilde{y}) = \sin(\pi\tilde{x})\,\sin(\pi\tilde{y})
$$

This manufactured solution enables rigorous quantitative error assessment of all three quantum solvers in two dimensions without dependence on a numerical reference.

### Key results (development benchmarks, $N = 4$–$8$)

- HHL achieves relative errors of 1–5% against the analytical solution for $\varepsilon = 0.01$, consistent with the Trotter approximation error
- VQLS achieves relative errors of 0.1–2% with COBYLA optimisation (cost $< 10^{-6}$), outperforming HHL at equivalent resolution
- QSVT achieves near-machine-precision residuals ($\lesssim 10^{-12}$) at $N=4$ with the Sz.-Nagy block encoding and pyqsp phase angles
- The peak electric field for the physical operating condition ($V_d = 300$ V, Gaussian profile) is of order $10^4$ V/m, in qualitative agreement with Boeuf & Garrigues (1998), Fig. 3

---

## 6. Algorithm Summary

Three quantum linear system algorithms are implemented, each targeting the system $A|x\rangle = |b\rangle$:

### Complexity comparison

| Algorithm | Circuit depth | Qubit count | Condition number scaling |
| --- | --- | --- | --- |
| HHL | $\mathcal{O}(\kappa^2 \log N / \varepsilon)$ | $\mathcal{O}(\log N + \log\kappa)$ | Quadratic in $\kappa$ |
| VQLS | $\mathcal{O}(n_\text{layers} \cdot n_\text{qubits})$ | $\mathcal{O}(\log N)$ | Variational (no formal guarantee) |
| QSVT | $\mathcal{O}(\kappa \log(1/\varepsilon))$ | $\mathcal{O}(\log N + 2)$ | Linear in $\kappa$ |

For the 1D Poisson matrix with $\kappa \sim \mathcal{O}(N^2)$, QSVT achieves a quadratic improvement in condition number dependence over HHL. For the 2D line-Jacobi row matrix with $\kappa_\text{row} \to 3^-$, the QSVT polynomial degree is essentially constant in $N$:

$$
d_\text{2D} = \mathcal{O}\!\left(\kappa_\text{row} \log(1/\varepsilon)\right) \approx \mathcal{O}\!\left(3\log(1/\varepsilon)\right)
$$

### QSVT block encoding

The block encoding uses the Sz.-Nagy unitary dilation. Given $M = A/\alpha$ with $\|M\|_2 \leq 1$, the $2N \times 2N$ unitary:

$$
U_A = \begin{pmatrix} M & \sqrt{I - M^2} \\ \sqrt{I - M^2} & -M \end{pmatrix}
$$

satisfies $(\langle 0|_\text{anc} \otimes I_N)\, U_A\, (|0\rangle_\text{anc} \otimes I_N) = M = A/\alpha$ exactly, using a single ancilla qubit. The subnormalisation factor $\alpha = \|A\|_2$ (spectral norm) is computed via eigendecomposition.

### VQLS cost function

$$
C(\boldsymbol{\theta}) = 1 - \frac{\left|\langle b \,|\, A \,|\, x(\boldsymbol{\theta})\rangle\right|^2}{\langle x(\boldsymbol{\theta}) \,|\, A^\dagger A \,|\, x(\boldsymbol{\theta})\rangle}
$$

evaluated via direct statevector arithmetic on the Pauli LCU decomposition of $A$. Optimisation uses COBYLA with a three-stage restart strategy ($\rho_\text{beg} \in \{0.5, 0.1, 0.01\}$).

### 2D line-Jacobi decomposition

$$
u^{n+1}_{i+1,j} - 4\,u^{n+1}_{i,j} + u^{n+1}_{i-1,j} = h^2 f(x_i, y_j) - \left(u^n_{i,j-1} + u^n_{i,j+1}\right)
$$

Each row sub-problem has a TST matrix with $a = -4$, $b = 1$ and $\kappa(A_\text{row}) \to 3^-$ as $N \to \infty$, far more favourable than the $\mathcal{O}(N^2)$ scaling of the 1D Poisson matrix.

---

## 7. Test Suite

The automated test suite is located in `tests/` and is executed via `pytest`. Tests verify structural correctness and solver functionality. All quantum solver tests use $N=4$ (2 qubits) to bound individual test runtime.

### Test file summary

| File | Coverage | Approx. runtime |
| --- | --- | --- |
| `test_problem_setup.py` | Matrix structure, grid, RHS, config validation, exact solutions | $<1$ s |
| `test_classical_solvers.py` | Thomas 1D/2D accuracy, NumPy agreement, convergence | $<5$ s |
| `test_hhl_1d.py` | HHL 1D solution shape, sign, proportionality recovery | $\sim 2$ min |
| `test_hhl_2d.py` | HHL 2D structure, sign consistency, iteration history | $\sim 3$ min |
| `test_vqls_1d.py` | VQLS cost convergence, parameter shape, reproducibility | $\sim 2$ min |
| `test_vqls_2d.py` | VQLS 2D structure, warm-start, sign consistency | $\sim 3$ min |
| `test_qsvt_1d.py` | Block encoding unitarity, QSP angle shape, QSVT solver correctness | $\sim 5$ min |
| `test_qsvt_2d.py` | QSVT 2D cache pre-computation, row solve, HET compatibility | $\sim 3$ min |
| `test_het_problem.py` | HET config derived quantities, matrix structure, solver compatibility | $\sim 2$ min |
| `test_het_2d.py` | HET 2D assembly, boundary conditions, solver compatibility | $\sim 3$ min |
| `test_integration.py` | End-to-end pipeline, BenchmarkResult consistency | $\sim 2$ min |

### Running the tests

```bash
# Full suite
pytest

# Fast classical tests only (under 10 seconds)
pytest tests/test_problem_setup.py tests/test_classical_solvers.py

# Single test file
pytest tests/test_qsvt_1d.py -v

# Single test function
pytest tests/test_hhl_1d.py::TestHHL1D::test_agrees_with_thomas_loose -v
```

### Pass/fail criteria

Tests verify that solvers:

- Return results of the correct shape and type
- Produce finite (non-NaN, non-Inf) solution values
- Agree with the Thomas reference to within a loose tolerance (20% for HHL, 15% for VQLS, 20% for QSVT at $N=4$)
- Preserve the correct sign of the dominant solution component
- Raise appropriate exceptions for invalid inputs

---

## 8. Methodological Notes

### Statevector simulation

All quantum circuits are evaluated via deterministic statevector simulation using Qiskit's `Statevector` class from `qiskit.quantum_info`. Shot-noise simulation via stochastic backends is bypassed to establish baseline theoretical algorithmic accuracy, consistent with the methodology of Ghafourpour & Laizet (2025).

### HHL solution extraction

The HHL output state is post-selected on the ancilla (flag) qubit being in state $|1\rangle$ and the clock (QPE) register being in state $|0\cdots0\rangle$. The proportionality constant $c$ is recovered against the normalised system $A_\text{norm} = A/\|A\|_2$ to prevent amplification of Trotter approximation errors by the factor $\|b\|/\|A\|_2$, which is large in physically scaled problems (e.g. the HET case where $\|b\| \sim \alpha h^2 \sim 700$).

### QSVT block encoding and phase angles

The block encoding uses the Sz.-Nagy dilation with $\alpha = \|A\|_2$ (spectral norm), giving $\kappa_\text{eff} = \kappa(A)$ exactly. The QSP phase angles are computed via the `pyqsp` library (`PolyOneOverX.generate(kappa, epsilon)`), which uses an internal degree selection algorithm. For the 1D Poisson matrix at $N=4$ ($\kappa \approx 9.5$, $\varepsilon = 0.05$), pyqsp returns degree $\approx 939$; for the 2D row matrix ($\kappa \approx 2.36$, $\varepsilon = 0.1$), degree $\approx 33$. A Chebyshev-based fallback is provided when pyqsp is unavailable.

### 2D QSVT performance optimisation

The block encoding circuit and QSP phase angles are pre-computed once before the line-Jacobi loop and cached, eliminating $\mathcal{O}(N \times \text{max\_iter})$ redundant computations. Row-level parallelisation via `concurrent.futures.ProcessPoolExecutor` is supported through the `n_workers` parameter in `QSVTConfig2D`, and is recommended for HPC deployment.

### Physical hardware viability

Extracting the full solution vector at each step of the 2D line-Jacobi cycle requires $\mathcal{O}(N)$ quantum state tomography operations per iteration and is not physically realisable on near-term quantum hardware. This framework is therefore strictly a theoretical simulator study of algorithmic behaviour, Hamiltonian scaling, and quantum error propagation.

---

## 9. References

1. Ghafourpour, L. & Laizet, S. (2025). Applicability of solving the one- and two-dimensional Poisson equations with the quantum Harrow-Hassidim-Lloyd algorithm. *Physical Review Applied*, 24, 024032.

2. Harrow, A. W., Hassidim, A. & Lloyd, S. (2009). Quantum algorithm for linear systems of equations. *Physical Review Letters*, 103, 150502.

3. Bravo-Prieto, C., LaRose, R., Cerezo, M., Subasi, Y., Cincio, L. & Coles, P. J. (2023). Variational quantum linear solver. *Quantum*, 7, 1188.

4. Gilyen, A., Su, Y., Low, G. H. & Wiebe, N. (2019). Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics. *Proceedings of the 51st Annual ACM STOC*, pp. 193–204.

5. Martyn, J. M., Rossi, Z. M., Tan, A. K. & Chuang, I. L. (2021). Grand unification of quantum algorithms. *PRX Quantum*, 2, 040203.

6. Vazquez, A. C., Hiptmair, R. & Woerner, S. (2022). Enhancing the quantum linear systems algorithm using Richardson extrapolation. *ACM Transactions on Quantum Computing*, 3, 1.

7. Boeuf, J. P. & Garrigues, L. (1998). Low frequency oscillations in a stationary plasma thruster. *Journal of Applied Physics*, 84(7), 3541–3554.

8. Brearley, P. & Laizet, S. (2024). Quantum algorithm for solving the advection equation using Hamiltonian simulation. *Physical Review A*, 110, 012430.

9. Over, P., Bengoechea, S., Brearley, P., Laizet, S. & Rung, T. (2025). Quantum algorithm for the advection-diffusion equation by direct block encoding of the time-marching operator. *Physical Review A*, 112, L010401.

10. Tennie, F., Laizet, S., Lloyd, S. & Magri, L. (2025). Quantum computing for nonlinear differential equations and turbulence. *Nature Reviews Physics*, 7, 220–230.
