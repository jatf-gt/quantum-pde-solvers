# Quantum Linear System Solvers for the Poisson Equation: HHL, VQLS, and QSVT Applied to Hall Effect Thruster Plasma Modelling

This repository contains the computational framework developed for the MSc thesis *"Quantum Algorithms for Coupled Fluid-Thermal Partial Differential Equations: Application to Regenerative Cooling in Rocket Nozzles"*, Department of Aeronautics, Imperial College London (2026).

The codebase implements and benchmarks three quantum linear system algorithms — the Harrow-Hassidim-Lloyd (HHL) algorithm, the Variational Quantum Linear Solver (VQLS), and the Quantum Singular Value Transformation (QSVT) — applied to the Poisson boundary value problem in one and two spatial dimensions. A physically motivated application to the electrostatic Poisson equation in Hall Effect Thruster (HET) plasma modelling is included as a proof-of-concept demonstration.

The 1D formulation employs a direct Toeplitz Symmetric Tridiagonal (TST) operator. The 2D resolution is achieved via a hybrid quantum-classical line-Jacobi iterative decomposition scheme. All results are benchmarked against the Thomas algorithm (classical tridiagonal direct solver) and, where available, against closed-form analytical solutions.

The numerical benchmarks systematically replicate and extend those of **Ghafourpour & Laizet (2025)** (*Physical Review Applied* 24, 024032), with the VQLS implementation following **Bravo-Prieto et al. (2023)** (*Quantum* 7, 1188) and the QSVT implementation following **Gilyen et al. (2019)** (*STOC 2019*) and **Martyn et al. (2021)** (*PRX Quantum* 2, 040203). The HET plasma application draws on the physical model of **Boeuf & Garrigues (1998)** (*Journal of Applied Physics* 84, 3541).

### Project status

**New here? Run `python scripts/tutorial.py --dim 2 --N 32`.** It solves a 2D
Poisson problem and prints a comparison of the outer schemes in a couple of
seconds, with no quantum backend required. `--dim 1` and `--dim 3` work the
same way, and `--inner qsvt` swaps in a quantum solver.

The laptop-scale benchmarking pipeline covers 1D, 2D and 3D Poisson problems and
the HET plasma application, all runnable directly via `scripts/`.

An **HPC deployment on Imperial College's CX3 cluster** pushes each sweep to
larger $N$ than is practical on a laptop, with both CPU and GPU (cuStateVec)
execution paths. The 1D, 2D and 3D pipelines are all complete and described in
§4.

**Architecture note.** 2D and 3D problems have no solvers of their own. The
domain is decomposed into 1D strips and an *outer iteration* (`solvers/outer`)
sweeps over them, handing each strip to the same 1D solver used in the 1D case.
Every quantum solver therefore works in any dimension unmodified, and — because
the strip operator is far better conditioned than the 1D Poisson operator
($\kappa 	o 3$ in 2D and $	o 2$ in 3D, against $O(N^2)$ in 1D) — is cheaper
per strip in higher dimensions, not dearer.

---

## Table of Contents

1. [Repository Architecture](#1-repository-architecture)
2. [Prerequisites and Installation](#2-prerequisites-and-installation)
3. [Execution Protocols (Local)](#3-execution-protocols-local)
4. [HPC / Cluster Execution (Imperial College CX3)](#4-hpc--cluster-execution-imperial-college-cx3)
5. [Benchmark Sweep Directory](#5-benchmark-sweep-directory)
6. [Physical Application: HET Plasma Modelling](#6-physical-application-het-plasma-modelling)
7. [Algorithm Summary](#7-algorithm-summary)
8. [Test Suite](#8-test-suite)
9. [Methodological Notes](#9-methodological-notes)
10. [References](#10-references)

---

## 1. Repository Architecture

The codebase is organised into a strictly layered, decoupled architecture separating problem instantiation, algorithmic execution, post-processing, and physical application. Data flows in one direction only: `core` → `problems` → `solvers` → `benchmark` → `scripts`.

The root of the repository also carries a set of HPC job-submission scripts and standalone diagnostic/plotting utilities that sit outside this layered core (see §4).

```
poisson_hhl/
│
├── core/                            # Shared infrastructure — PDE-agnostic
│   ├── cases.py                     # Canonical case registry: 26 cases, 1D/2D/3D
│   ├── het_geometry.py              # Single SPT-100 geometry, shared by 1D/2D/3D
│   ├── config.py                    # SimConfig1D, SimConfig2D
│   ├── exact_solutions.py           # Analytical solutions: 1D (fS, fL, fH) and 2D sinusoidal
│   ├── het_config.py                # HETConfig, HETPhysicalConfig, physical constants
│   └── source_functions.py          # Source functions: fS, fL, fH (1D and 2D); HET profiles
│
├── problems/                        # Domain discretisation and operator assembly
│   ├── poisson_1d.py                # 1D TST matrix, RHS, PoissonProblem1D container
│   ├── poisson_line_2d.py           # 2D line-decomposed problem, PoissonLine2D
│   ├── poisson_line_3d.py           # 3D line-decomposed problem, PoissonLine3D
│   ├── het_plasma_1d.py             # HET 1D: HETPoissonProblem1D, HETPhysicalProblem1D
│   └── het_plasma_2d.py             # HET 2D: thin PoissonLine2D builders over core/het_config.py
│
├── solvers/                         # Algorithmic resolution implementations
│   ├── backend_factory.py           # Centralised Aer backend selection (CPU/GPU)
│   ├── classical/
│   │   ├── thomas.py                # Thomas algorithm: thomas_solve, thomas_solve_system
│   │   └── numpy_ref.py             # NumPy direct solver (debugging reference)
│   ├── quantum/
│   │   ├── result.py                # SolverResult, VQLSSolverResult, QSVTSolverResult
│   │   ├── hhl_1d.py                # HHL 1D: hhl_solve, hhl_solve_system
│   │   ├── vqls_utils.py            # LCU Pauli decomposition, ansatz, cost function
│   │   ├── vqls_1d.py               # VQLS 1D: vqls_solve, vqls_solve_system, VQLSConfig1D
│   │   ├── block_encoding.py        # Sz.-Nagy block encoding for TST matrices
│   │   ├── qsp_angles.py            # QSP phase angle computation via pyqsp / Chebyshev
│   │   └── qsvt_1d.py               # QSVT 1D: qsvt_solve, qsvt_solve_system, QSVTConfig1D
│   └── outer/                       # The single 2D/3D architecture: strip decomposition
│       ├── core.py                  # LineProblem2D/InnerSolver protocols, strip_sweep
│       ├── inner.py                 # Validated (A,b)->x registry: thomas/hhl/vqls/qsvt
│       ├── stationary.py            # jacobi / gauss-seidel / sor
│       └── multigrid.py             # V-cycle / full multigrid (fmg)
│
├── benchmark/                       # Evaluation orchestration and reporting
│   ├── results_io.py                # On-disk sweep schema contract (read + write)
│   ├── diagnostics.py               # Shared comparison-table/study primitives for debug_2d/3d.py
│   ├── metrics.py                   # BenchmarkResult, BenchmarkResult2D, compute_errors
│   ├── plotting.py                  # Matplotlib figure primitives: arrays in, Figure out
│   ├── hpc_plotting.py              # HPC sweep post-processing: load -> reshape -> draw -> save
│   ├── reporting.py                 # Tabular console output for 1D and 2D results
│   └── runner.py                    # Sweep drivers A-H4; run_pair_1d, run_pair_2d
│
├── scripts/                         # Top-level execution entry points
│   ├── tutorial.py                  # ⭐ START HERE — one entry point, --dim {1,2,3}
│   ├── debug_1d.py                  # 1D solver diagnostics: raw (A,b) cases, QSVT dump, kappa tables
│   ├── debug_2d.py                  # 2D scheme/solver diagnostics, noise and polish studies
│   ├── debug_3d.py                  # 3D equivalent
│   ├── example_report.py            # Copy-me template for a small, laptop-scale report
│   ├── run_hpc_1Dfull.py            # HPC driver: full 1D sweep N=4..64, all solvers (see §4)
│   ├── run_hpc_2Dfull.py            # HPC driver: full 2D sweep (see §4.7)
│   ├── run_hpc_3Dfull.py            # HPC driver: full 3D sweep (see §4.7)
│   ├── plot_hpc_{1,2,3}Dfull_results.py  # Thin CLIs over benchmark/hpc_plotting.py
│   ├── precompute_qsvt_phases.py    # QSVT phase-angle precompute, --dim {1,2} (see §4.4)
│   └── archive/                     # Superseded one-off scripts; see its README
│
├── tests/                           # Pytest test suite
│   ├── conftest.py                  # Shared fixtures and tolerance constants
│   ├── test_problem_setup.py        # 1D matrix assembly, grid, RHS, condition number
│   ├── test_classical_solvers.py    # Thomas 1D and NumPy reference solvers
│   ├── test_line_problems.py        # PoissonLine2D/3D operators, BCs, coarsening
│   ├── test_outer.py                # solvers/outer: schemes, registry, multigrid
│   ├── test_hhl_1d.py               # HHL 1D solver correctness
│   ├── test_vqls_1d.py              # VQLS 1D solver correctness
│   ├── test_qsvt_1d.py              # QSVT 1D block encoding, phase angles, solver correctness
│   ├── test_het_problem.py          # HET 1D problem assembly and solver compatibility
│   └── test_integration.py          # End-to-end pipelines, 1D and 2D
│
├── results/                         # Auto-generated output artefacts (git-ignored)
│   ├── sweep_*.csv
│   ├── qsvt_phase_cache/            # Cached QSP phase angles (see §4.4)
│   ├── tutorial/                    # Output of scripts/tutorial.py (1D --plot)
│   ├── debugging/                   # Output of debug_1d.py / debug_2d.py / debug_3d.py
│   ├── example_report/              # Output of scripts/example_report.py
│   ├── 1Dhpc_run/                   # Output of the HPC 1D full sweep (see §4)
│   ├── 2Dhpc_run/                   # Output of the HPC 2D full sweep (see §4.7)
│   └── 3Dhpc_run/                   # Output of the HPC 3D full sweep (see §4.7)
│
├── quantum_linear_solvers/          # Git submodule: TST Hamiltonian simulation (Vázquez et al.)
│
├── hpc/                             # PBS deployment for Imperial CX3 — see hpc/README.md
│   ├── setup_hpc_env.sh             # One-time environment setup (CPU + GPU venvs) — §4.1
│   ├── submit_hpc_1D.sh             # PBS job: full 1D sweep, CPU — §4.2
│   ├── submit_hpc_gpu.sh            # PBS job: full 1D sweep, GPU (L40S / cuStateVec) — §4.3
│   ├── submit_hpc_2D.sh             # PBS job: full 2D sweep — §4.7
│   ├── submit_hpc_2D_gapfill.sh     # PBS job: fill gaps left by a killed 2D sweep
│   ├── submit_hpc_3D.sh             # PBS job: full 3D sweep — §4.7
│   ├── submit_precompute_hpc.sh     # PBS job: 1D QSVT phase-angle precompute — §4.4
│   └── submit_precompute_2D.sh      # PBS job: 2D QSVT phase-angle precompute — §4.4
│
├── pytest.ini                       # Pytest configuration
├── requirements.txt                 # Python environment dependencies
└── README.md
```

---

## 2. Prerequisites and Installation

This framework requires an isolated Python environment to prevent dependency conflicts between Qiskit quantum information modules, PennyLane variational quantum components, and classical scientific computing libraries.

### Supported Python version

Python 3.11 is recommended. Python 3.10 and 3.12 are compatible but untested locally (the CX3 cluster environment in §4 uses 3.12.3 via the `Python/3.12.3-GCCcore-13.3.0` module).

### Step 1 — Clone the repository with the required submodule

```bash
git clone --recurse-submodules https://github.com/jatf-gt/quantum-pde-solvers.git
cd quantum-pde-solvers
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

| Package          | Role                                                             |
| ---------------- | ---------------------------------------------------------------- |
| `qiskit` >= 1.0  | Quantum circuit construction and statevector simulation          |
| `qiskit-aer`     | High-performance statevector backend                             |
| `pennylane`      | VQLS variational optimisation and automatic differentiation      |
| `pyqsp`          | QSP phase angle computation for QSVT matrix inversion polynomial |
| `numpy`, `scipy` | Classical linear algebra and optimisation                        |
| `openpyxl`       | Excel workbook export for benchmark metrics                      |
| `matplotlib`     | Result visualisation                                             |
| `pytest`         | Automated test suite                                             |

> **Qiskit 1.0 compatibility note:** The `quantum_linear_solvers` submodule has been patched to replace the deprecated `QuantumCircuit.isometry()` method with the `Isometry` gate from `qiskit.circuit.library`. This patch is applied to the submodule source directly and requires no action from the user.

---

## 3. Execution Protocols (Local)

All entry points are located in `scripts/` and are designed to be executed from the repository root. These are the laptop-scale entry points; for larger-$N$ unattended runs on the cluster, see §4. Five files cover the whole local surface — a tutorial, three per-dimension debug drivers, and a copy-me report template — plus `scripts/archive/`, superseded one-off scripts kept for provenance (see its README).

### 3.1 — `tutorial.py`: start here

Solves a Poisson problem in one, two or three dimensions with any combination of solvers and prints a comparison table. For 2D/3D it is a thin front end onto `debug_2d.py`/`debug_3d.py`.

```bash
python scripts/tutorial.py --dim 2 --N 32
python scripts/tutorial.py --dim 1 --N 8 --inner all
python scripts/tutorial.py --dim 2 --N 8 --inner qsvt
python scripts/tutorial.py --list-cases        # every registered case for --dim
python scripts/tutorial.py --list-options      # every tunable inner/scheme parameter
```

**Runtime:** seconds for the classical solver; under a minute for one quantum solver at $N \le 16$.

### 3.2 — `debug_1d.py`: 1D solver diagnostics

Runs any of the 11 registered 1D cases (`core/cases.py`, `available(dim=1)`) — including the raw-matrix sub-cases 3b and 3c, which are not `PoissonProblem1D` instances — through the same `(A, b) -> x` inner-solver registry the outer iteration uses per strip.

```bash
python scripts/debug_1d.py --case poisson_1d_fS_hom --N 8
python scripts/debug_1d.py --case het_1d_3c_neumann --N 16 --inner qsvt
python scripts/debug_1d.py --dump --case het_1d_3a_linear --N 8 --inner qsvt
python scripts/debug_1d.py --kappa-table       # kappa(N) vs the O(N^2) theoretical scaling
```

### 3.3 — `debug_2d.py` / `debug_3d.py`: 2D/3D outer-scheme diagnostics

Compares inner solvers and outer schemes on a line-decomposed problem: scheme comparison table, multigrid hierarchy inspection, inner-solver noise tolerance, and multigrid-then-polish studies. Cases come from `core/cases.py` (`available(dim=2)`/`available(dim=3)`); short aliases (`square`, `het`, `cube`, `slab`) from the original tools still work.

```bash
python scripts/debug_2d.py --case square --N 64
python scripts/debug_2d.py --case het --N 8 --inner hhl
python scripts/debug_2d.py --N 64 --scheme fmg -S nu1=2 -S n_coarse=8
python scripts/debug_2d.py --noise-study --N 32
python scripts/debug_3d.py --case cube --N 16
python scripts/debug_3d.py --convergence-study --case cube
```

> **Runtime note:** a single quantum-solver comparison at $N=8$–16 runs in seconds to a couple of minutes on a laptop, because the outer scheme (default `fmg`) needs a grid-independent number of strip solves rather than the $O(N)$ line-Jacobi sweeps of the originally published scheme. Pass `--scheme jacobi` to reproduce that literature baseline, at correspondingly higher cost.

### 3.4 — `example_report.py`: copy-me template

A small, laptop-scale report — one 1D section, one 2D section, one figure each, one CSV — heavily commented to mark exactly which lines to change. Copy it to a new file and edit the `# CHANGE ME` lines to build your own report; see `scripts/archive/run_meeting5.py` for the fuller (but unmaintained) structure this was modelled on.

```bash
python scripts/example_report.py
```

**Output:** `results/example_report/report_{1d,2d}.png`, `report_metrics.csv`. Runtime: well under a minute at the shipped defaults.

### 3.5 — Automated test suite

```bash
pytest
```

To run only the fast classical tests (under 10 seconds):

```bash
pytest tests/test_problem_setup.py tests/test_classical_solvers.py
```

---

## 4. HPC / Cluster Execution (Imperial College CX3)

Some configurations — particularly QSVT at large $N$, and any solver across the full $N=4\ldots64$ sweep — are impractical to run to completion on a laptop within a reasonable wall-clock time. This section covers the PBS Pro job infrastructure built for Imperial College's **CX3** HPC cluster to run these unattended, with both CPU and GPU execution paths.

**Current scope:** this infrastructure drives the **1D full sweep** end-to-end (environment setup → job submission → QSVT phase precompute → post-processing). The **2D equivalent is in progress** — see §4.7.

### 4.1 — One-time environment setup

Run once, from a CX3 login node:

```bash
ssh username@login.cx3.hpc.ic.ac.uk
bash hpc/setup_hpc_env.sh
```

This creates **two** separate virtual environments under the RDS home directory (`~/venvs/`, backed by 1 TB permanent quota):

| Environment          | Path                | Purpose                                                        |
| --------------------- | ------------------- | ---------------------------------------------------------------- |
| `qpde`     (CPU)      | `~/venvs/qpde`      | `qiskit==0.45.3`, `qiskit-aer==0.13.3`, `qiskit-algorithms==0.3.0`, NumPy/SciPy/matplotlib/pandas/openpyxl, plus `quantum_linear_solvers` from GitHub |
| `qpde-gpu` (GPU)      | `~/venvs/qpde-gpu`  | Same stack but with `qiskit-aer-gpu==0.15.1` (CUDA 12 / cuStateVec) in place of `qiskit-aer` — the two packages cannot coexist in one environment |

`pyqsp` is intentionally **not** in this explicit install list (it's covered separately at job-submission time — see the note in §4.2) so that a missing import doesn't surface hours into a queued job.

### 4.2 — CPU: full 1D sweep submission

```bash
qsub hpc/submit_hpc_1D.sh
```

Runs `scripts/run_hpc_1Dfull.py` across $N=4\ldots64$, all cases, all four solvers (Thomas, HHL, VQLS, QSVT).

**Resource request:** `select=1:ncpus=4:mem=128gb`, `walltime=24:00:00`.

**Useful overrides** (fast validation pass, or skipping the most expensive solver):
```bash
export MAX_N=16
qsub -v MAX_N hpc/submit_hpc_1D.sh

export SKIP_QSVT=1
qsub -v SKIP_QSVT hpc/submit_hpc_1D.sh
```

**Monitoring:**
```bash
qstat -u $USER
tail -f results/1Dhpc_run/run.log
```

**Notes:**
- `--max-workers` is pinned to 4 to match `ncpus`. Aer simulations are already OpenMP-threaded internally, so more worker processes than allocated cores oversubscribes the node rather than speeding it up — if you raise `ncpus`, raise `--max-workers` (and `OMP_NUM_THREADS`) to match.
- `pyqsp` is checked for at runtime and installed on the fly (`pip install pyqsp==0.2.0`) if missing from the venv.
- Results write incrementally: each solution `.npz` is saved as it's produced, but `results_full.json` / `results_summary.csv` are only written at the **end** of the run — a job killed on walltime keeps the per-solution files but loses the summary table.
- On completion, results are copied to permanent storage at `~/qpde-results/1Dhpc_run_<timestamp>/`.

### 4.3 — GPU-accelerated 1D sweep

```bash
qsub hpc/submit_hpc_gpu.sh
```

Targets the `gpu72` queue with a single **NVIDIA L40S** (48 GB GDDR6, Ada Lovelace, compute capability 8.9), using `qiskit-aer-gpu`'s cuStateVec backend. Also drives `scripts/run_hpc_1Dfull.py`, but forces **serial** execution (`--max-workers 1`) since concurrent worker processes would conflict over the CUDA context.

**Expected speedup over CPU** (per the script's own estimates):
- $N=8$ QSVT (circuit depth 6,479): ~10–30× → ~7–22 s (vs 222 s on CPU)
- $N=16$ QSVT (circuit depth 44,567): ~10–50× → feasible (vs hours on CPU)

**Resource request:** `select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S`, `walltime=24:00:00`, queue `gpu72`.

```bash
export INCLUDE_N64=1
qsub -v INCLUDE_N64 hpc/submit_hpc_gpu.sh
```

Requires the separate `qpde-gpu` venv from §4.1 to be present; the script exits early with setup instructions if it isn't found.

### 4.4 — QSVT phase-angle precompute

QSP phase-angle generation (`pyqsp.PolyOneOverX.generate`) is by far the most expensive part of setting up a QSVT solve at large $N$/condition number, and its cost doesn't parallelise across cores. `hpc/submit_precompute_hpc.sh` runs this as its own single-threaded batch job (`scripts/precompute_qsvt_phases.py`) so it survives disconnects and isn't capped by an interactive session's own wall-clock limit, caching results to `results/qsvt_phase_cache/`.

The intended usage is a **staged rollout** — small, safe $N$ first, then progressively larger and more exploratory:

```bash
# Stage 1 — small N, expected safe:
export N_VALUES="4,8,16"
qsub -v N_VALUES hpc/submit_precompute_hpc.sh

# Stage 2 — N=32, exploratory, degree-capped, separate job/log:
export N_VALUES="32"
export MAX_DEGREE="2000"
qsub -v N_VALUES,MAX_DEGREE hpc/submit_precompute_hpc.sh

# Stage 3 — N=64, only after Stage 2 is confirmed working:
export N_VALUES="64"
export MAX_DEGREE="2000"
qsub -v N_VALUES,MAX_DEGREE hpc/submit_precompute_hpc.sh
```

> **PBS quirk:** pass `N_VALUES` / `MAX_DEGREE` via `qsub -v NAME` (bare name, value taken from the shell's exported variable), **not** `qsub -v NAME=value` — PBS's own `-v` parser splits on commas, which breaks a comma-separated list like `"4,8,16"` if it's embedded directly after an `=`.

Each stage writes into the same cache directory, so results accumulate across stages and nothing already cached is re-run. If a stage is killed before finishing, whatever it completed is already safe on disk — just resubmit the same stage.

**Resource request:** single-threaded, `mem=32gb`, `walltime=71:00:00` (just under CX3's 72h queue cap). $N=32$/$64$ are **not** guaranteed to finish inside one submission.

### 4.5 — Post-processing

```bash
python scripts/plot_hpc_1Dfull_results.py --results-dir results/1Dhpc_run --save-pdf
```

Reads `results_full.json` and the per-case/per-solver `.npz` solution files and produces:

1. Solution profiles (Thomas vs HHL vs VQLS vs QSVT) with pointwise error, per case
2. Max relative error vs $N$ (log-log convergence plot, with an $\mathcal{O}(N^{-2})$ reference line)
3. Residual $\|Au-b\|/\|b\|$ vs $N$
4. Wall time vs $N$
5. HET 1D potential profiles and electric-field magnitude (sub-cases 3a/3b/3c)
6. A combined 2×2 summary figure across the four generic Poisson cases

Figures are saved as PNG (always) and PDF (with `--save-pdf`) directly into the results directory.

### 4.6 — Standalone QSVT diagnostics

`scripts/debug_1d.py --dump` (§3.2) is the debugging counterpart to the HPC sweep: rather than running the full benchmark pipeline, it exercises the QSVT solver directly on one case — the proportionality-recovery diagnostics from `qsvt_1d.py::_qsvt_recovery_diagnostics`, the polynomial degree actually solved, and a node-by-node solution/error dump — on a small, fast case. Useful for isolating a QSVT regression before committing a multi-hour cluster job to it. (Superseded `run_qsvt_debug.py` is kept in `scripts/archive/` for provenance.)

### 4.7 — 2D and 3D HPC runners

`scripts/run_hpc_2Dfull.py` and `scripts/run_hpc_3Dfull.py` mirror the 1D driver, submitted via `hpc/submit_hpc_2D.sh` and `hpc/submit_hpc_3D.sh`. Both share the 1D driver's incremental-write behaviour — per-configuration `.npz` output as it is produced — so partial progress survives a walltime kill; only the summary JSON/CSV is lost, and the plotting layer reads the per-solution archives regardless.

The cost profile differs substantially from the 1D driver. A 1D configuration is a single solve; a 2D or 3D configuration is an outer iteration over many strip solves, so the resource request and the choice of outer scheme matter more than the solver does. Use `--scheme fmg` unless you are specifically reproducing the originally published line-Jacobi results, for which `--scheme jacobi` exists.

Phase-angle precompute for the strip operator is cheap in both dimensions: $\kappa_	ext{row} 	o 3^-$ in 2D and $	o 2^-$ in 3D gives polynomial degrees of 30–85 irrespective of $N$, against the steeply growing 1D degrees. `hpc/submit_precompute_2D.sh` completes the whole set in minutes and needs no staging, unlike its 1D counterpart (§4.4).

Post-processing for all three dimensions lives in `benchmark/hpc_plotting.py`, with `scripts/plot_hpc_{1,2,3}Dfull_results.py` as thin command-line wrappers. The on-disk schema itself — the summary format, the archive filename convention and the field-name aliases — is declared once in `benchmark/results_io.py`, which both the readers and (in due course) the runners share.

**2D and 3D share a result schema and therefore share their scalar-metric plots** (convergence, accuracy vs $N$, cost vs $N$, quantum overhead, error decomposition). **1D does not.** Its summary rows carry no `scheme`, `stop_reason`, `linf_err`, `weighted_cost` or `err_vs_thomas` — precisely the fields those plots read — so the 1D entry point calls none of them and has its own parallel implementations in a different visual style. Field visualisation is dimension-specific in all three: profiles in 1D, fields in 2D, orthogonal slices and polar unwrapping in 3D.

`SweepArchive.missing()` reports summary rows whose solution archive is absent, which distinguishes a partial sweep from a broken filename convention; without it both present identically, as a quietly incomplete figure set.

---

## 5. Benchmark Sweep Directory

The execution runners in `benchmark/runner.py` sequentially process predefined computational sweeps replicating the primary literature benchmarks.

### Generic Poisson sweeps

| Sweep | Dimension | Description                                                                            |
| ----- | --------- | -------------------------------------------------------------------------------------- |
| A     | 1D        | Homogeneous BCs; source functions fS, fL, fH; $N \in \{8, 16\}$; $\varepsilon = 0.01$    |
| B     | 1D        | Trotter/VQLS $\varepsilon$ sensitivity; $N = 16$; $\varepsilon \in \{0.1, 0.01, 0.001\}$ |
| C     | 1D        | Non-homogeneous Dirichlet BCs; fH; $N \in \{16, 32\}$                                    |
| D     | 1D        | Condition number scaling $\kappa(A) \sim (4/\pi^2)(N+1)^2$; $N \in \{4, 8, 16, 32\}$     |
| E     | 2D        | Homogeneous BCs; fS, fL, fH; $N \in \{8, 16\}$; $\varepsilon \in \{0.01, 0.5\}$          |
| F     | 2D        | Non-homogeneous BCs; convergence behaviour under asymmetric conditions                 |
| G     | 2D        | Row matrix condition number $\kappa(A_\text{row}) \to 3^-$; $N \in \{4, 8, 16, 32\}$     |

### HET plasma sweeps

| Sweep | Description                                                          |
| ----- | -------------------------------------------------------------------- |
| H1    | 1D Gaussian profile, homogeneous BCs; $N \in \{4, 8\}$                |
| H2    | 1D Gaussian profile, physical BCs ($V_d = 300$ V); $N \in \{4, 8\}$   |
| H3    | 1D all three charge density profiles; $N = 8$; physical BCs          |
| H4    | Condition number and $\alpha = L^2/\lambda_D^2$ scaling diagnostics   |

---

## 6. Physical Application: HET Plasma Modelling

### Problem formulation

The electrostatic potential $\tilde{\phi}$ in the discharge channel of a Hall Effect Thruster satisfies the non-dimensionalised Poisson equation:

$$
\frac{d^2 \tilde{\phi}}{d\tilde{x}^2} = -\alpha \, \delta\tilde{n}(\tilde{x}) \quad \text{(1D axial)}
$$

$$
\frac{\partial^2 \tilde{\phi}}{\partial \tilde{x}^2} + \frac{\partial^2 \tilde{\phi}}{\partial \tilde{y}^2} = -\alpha \, \delta\tilde{n}(\tilde{x}, \tilde{y}) \quad \text{(2D axial-radial)}
$$

where $\alpha = L^2 / \lambda_D^2$ is the dimensionless Debye scaling parameter and $\delta\tilde{n} = (n_i - n_e)/n_0$ is the non-dimensional net charge density.

### Physical parameters (Boeuf & Garrigues 1998, Table 1)

| Parameter            | Symbol                      | Value                       |
| -------------------- | --------------------------- | ---------------------------- |
| Channel length       | $L$                         | 25 mm                       |
| Discharge voltage    | $V_d$                       | 300 V                        |
| Electron temperature | $T_e$                       | 20 eV                        |
| Reference density    | $n_0$                       | $5 \times 10^{17}$ m$^{-3}$  |
| Debye length         | $\lambda_D$                 | $\approx 128\,\mu$m           |
| Scaling parameter    | $\alpha = L^2/\lambda_D^2$  | $\approx 38{,}000$            |

### Charge density profiles

Three physically motivated source term profiles are implemented:

| Key        | Profile                                                                                        | Description                                                  |
| ---------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `gaussian` | $\delta\tilde{n} = \delta_0 \exp\left(-(\tilde{x}-\tilde{x}_\text{peak})^2/\sigma^2\right)$      | Smooth ionisation zone near exit plane                       |
| `linear`   | $\delta\tilde{n} = \delta_0\,\tilde{x}$                                                          | Uniform space charge gradient; analytical solution available |
| `step`     | $\delta\tilde{n} = \delta_0\,\mathrm{sign}(\tilde{x} - \tilde{x}_\text{ion})$                    | Sharp ionisation front                                       |

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

## 7. Algorithm Summary

Three quantum linear system algorithms are implemented, each targeting the system $A|x\rangle = |b\rangle$:

### Complexity comparison

| Algorithm | Circuit depth                                          | Qubit count                        | Condition number scaling          |
| --------- | ------------------------------------------------------ | ----------------------------------- | ---------------------------------- |
| HHL       | $\mathcal{O}(\kappa^2 \log N / \varepsilon)$            | $\mathcal{O}(\log N + \log\kappa)$  | Quadratic in $\kappa$              |
| VQLS      | $\mathcal{O}(n_\text{layers} \cdot n_\text{qubits})$    | $\mathcal{O}(\log N)$               | Variational (no formal guarantee)  |
| QSVT      | $\mathcal{O}(\kappa \log(1/\varepsilon))$               | $\mathcal{O}(\log N + 2)$           | Linear in $\kappa$                 |

For the 1D Poisson matrix with $\kappa \sim \mathcal{O}(N^2)$, QSVT achieves a quadratic improvement in condition number dependence over HHL. For the 2D line-Jacobi row matrix with $\kappa_\text{row} \to 3^-$, the QSVT polynomial degree is essentially constant in $N$:

$$
d_\text{2D} = \mathcal{O}\left(\kappa_\text{row} \log(1/\varepsilon)\right) \approx \mathcal{O}\left(3\log(1/\varepsilon)\right)
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

## 8. Test Suite

The automated test suite is located in `tests/` and is executed via `pytest`. Tests verify structural correctness and solver functionality rather than publication-grade accuracy. All quantum solver tests use $N=4$ (2 qubits) to bound individual test runtime.

The suite covers the 1D solvers directly and the 2D/3D solvers through `solvers/outer`, the single outer-iteration architecture. Because every multi-dimensional solve is an outer iteration over 1D strip solves, testing the 1D solvers and the outer layer separately covers the 2D and 3D paths without paying for a full quantum line-relaxation run in the test suite.

### Test file summary

| File                        | Coverage                                                                       | Tests | Approx. runtime |
| --------------------------- | ------------------------------------------------------------------------------ | ----- | ---------------- |
| `test_problem_setup.py`     | 1D matrix structure, grid, RHS, config validation, exact solutions             | 41    | $\sim 1$ s      |
| `test_classical_solvers.py` | Thomas 1D accuracy, NumPy agreement                                             | 9     | $\sim 1$ s      |
| `test_line_problems.py`     | `PoissonLine2D`/`PoissonLine3D`: operators, Dirichlet absorption, periodicity, conditioning, coarsening | 36 | $\sim 1$ s |
| `test_outer.py`             | `solvers/outer`: work accounting, stagnation detection, strip sweep, option registry, stationary schemes, multigrid transfer operators and cycles | 84 | $\sim 8$ s |
| `test_hhl_1d.py`            | HHL 1D solution shape, sign, proportionality recovery                          | 11    | $\sim 8$ s      |
| `test_vqls_1d.py`           | VQLS cost convergence, parameter shape, reproducibility                        | 15    | $\sim 5$ s      |
| `test_qsvt_1d.py`           | Block encoding unitarity, QSP angle shape, QSVT solver correctness             | 24    | $\sim 4$ s      |
| `test_het_problem.py`       | HET config derived quantities, matrix structure, solver compatibility          | 22    | $\sim 4$ s      |
| `test_integration.py`       | End-to-end pipelines, 1D and 2D, and `BenchmarkResult` consistency             | 17    | $\sim 5$ s      |
| **Total**                   |                                                                                | **259** | $\sim 31$ s   |

Two properties receive dedicated tests because a silent regression in either would be hard to attribute:

- **Line-Jacobi reproducibility.** `scheme="jacobi"` with `criterion="delta"` must reproduce the original line-Jacobi loop exactly. `test_outer.py` reconstructs that loop from first principles and asserts agreement in the field, the iteration count and the stopping point.
- **Option validation.** The inner-solver registry must reject unknown keys rather than absorbing them. A registry that silently ignored `max_degrees=500` would let an HPC run cost an order of magnitude more than intended whilst appearing to honour the setting.

### Markers

One marker is defined, `quantum`, applied to every test that builds and simulates a circuit:

```bash
pytest -m "not quantum"     # 193 tests, ~11 s, no quantum backend required
```

This selects the pure-classical subset: problem assembly, the classical solvers, and the whole outer-iteration layer. It is the fastest meaningful check and the one to run when iterating on 2D/3D solver structure.

There is deliberately no `slow` marker. Every test in the suite completes in under four seconds, so a fast/slow split would carry no information; the earlier marker predated the consolidation onto `solvers/outer` and had become mis-applied to tests taking barely a second.

### Running the tests

```bash
# Full suite
pytest

# Pure-classical subset — no quantum backend needed (~11 s)
pytest -m "not quantum"

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

## 9. Methodological Notes

### Statevector simulation

All quantum circuits are evaluated via deterministic statevector simulation using Qiskit's `Statevector` class from `qiskit.quantum_info`. Shot-noise simulation via stochastic backends is bypassed to establish baseline theoretical algorithmic accuracy, consistent with the methodology of Ghafourpour & Laizet (2025).

### HHL solution extraction

The HHL output state is post-selected on the ancilla (flag) qubit being in state $|1\rangle$ and the clock (QPE) register being in state $|0\cdots0\rangle$. The proportionality constant $c$ is recovered against the normalised system $A_\text{norm} = A/\|A\|_2$ to prevent amplification of Trotter approximation errors by the factor $\|b\|/\|A\|_2$, which is large in physically scaled problems (e.g. the HET case where $\|b\| \sim \alpha h^2 \sim 700$).

### QSVT block encoding and phase angles

The block encoding uses the Sz.-Nagy dilation with $\alpha = \|A\|_2$ (spectral norm), giving $\kappa_\text{eff} = \kappa(A)$ exactly. The QSP phase angles are computed via the `pyqsp` library (`PolyOneOverX.generate(kappa, epsilon)`), which uses an internal degree selection algorithm. For the 1D Poisson matrix at $N=4$ ($\kappa \approx 9.5$, $\varepsilon = 0.05$), pyqsp returns degree $\approx 939$; for the 2D row matrix ($\kappa \approx 2.36$, $\varepsilon = 0.1$), degree $\approx 33$. A Chebyshev-based fallback is provided when pyqsp is unavailable. This degree gap (939 vs 33) is exactly why the 1D phase precompute (§4.4) needs staged, cluster-scale treatment while the 2D row matrix is expected to be comparatively cheap (§4.7).

### 2D QSVT performance optimisation

The block encoding circuit and QSP phase angles are pre-computed once before the line-Jacobi loop and cached, eliminating $\mathcal{O}(N \times \text{max\_iter})$ redundant computations. Row-level parallelisation via `concurrent.futures.ProcessPoolExecutor` is supported through the `n_workers` parameter in `QSVTConfig2D`, and is recommended for HPC deployment — this is the hook the in-progress 2D HPC runner (§4.7) is expected to use.

### Physical hardware viability

Extracting the full solution vector at each step of the 2D line-Jacobi cycle requires $\mathcal{O}(N)$ quantum state tomography operations per iteration and is not physically realisable on near-term quantum hardware. This framework is therefore strictly a theoretical simulator study of algorithmic behaviour, Hamiltonian scaling, and quantum error propagation.

---

## 10. References

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
