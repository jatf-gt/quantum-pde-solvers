# Quantum Linear System Solvers for the Poisson Equation

**HHL, VQLS, and QSVT applied to Hall Effect Thruster plasma modelling**

MSc thesis, *A Comparative Study of HHL, VQLS, and QSVT Algorithms for Solving Poisson-Type PDEs with Application to Hall-Effect Thruster Plasma Modelling*, Department of Aeronautics, Imperial College London, 2026.

Three quantum linear system algorithms — Harrow-Hassidim-Lloyd (HHL), the Variational Quantum Linear Solver (VQLS), and the Quantum Singular Value Transformation (QSVT) — are implemented and benchmarked against the classical Thomas algorithm on the Poisson boundary value problem in one, two, and three spatial dimensions. A physical application models the electrostatic field in a Hall Effect Thruster (HET) discharge channel.

The numerical benchmarks replicate and extend **Ghafourpour and Laizet (2025)** (*Phys. Rev. Applied* 24, 024032). The VQLS implementation follows **Bravo-Prieto et al. (2023)** (*Quantum* 7, 1188). The QSVT implementation follows **Gilyen et al. (2019)** (*STOC*) and **Martyn et al. (2021)** (*PRX Quantum* 2, 040203). The HET model draws on **Boeuf and Garrigues (1998)** (*J. Appl. Phys.* 84, 3541).

---

### Start here

```bash
python scripts/tutorial.py --dim 2 --N 32
```

Solves a 2D Poisson problem, compares outer schemes, and prints results in seconds. No quantum backend is needed unless you pass `--inner hhl`, `--inner vqls`, or `--inner qsvt`.

---

### Architecture note

2D and 3D problems have no dedicated solvers. The domain is cut into 1D strips; `solvers/outer` sweeps over them, passing each strip to the same 1D inner solver used in the 1D case. Every quantum solver therefore works in any dimension without modification. The strip operator is far better conditioned than the full 1D Poisson matrix: $\kappa_\text{row} \to 3^-$ in 2D and $\to 2^-$ in 3D as $N \to \infty$, against $\mathcal{O}(N^2)$ in 1D. Quantum solvers are therefore *cheaper* per strip in higher dimensions, not dearer.

---

## Table of Contents

1. [Repository layout](#1-repository-layout)
2. [Prerequisites and installation](#2-prerequisites-and-installation)
3. [Local execution](#3-local-execution)
4. [HPC execution (Imperial CX3)](#4-hpc-execution-imperial-cx3)
5. [Benchmark sweep catalogue](#5-benchmark-sweep-catalogue)
6. [Physical application: HET plasma](#6-physical-application-het-plasma)
7. [Algorithm summary](#7-algorithm-summary)
8. [Test suite](#8-test-suite)
9. [Methodological notes](#9-methodological-notes)
10. [Hardware results](#10-hardware-results)
11. [References](#11-references)
12. [Use of generative AI](#12-use-of-generative-ai)
13. [Licence and citation](#13-licence-and-citation)

---

## 1. Repository layout

Data flows in one direction: `core` -> `problems` -> `solvers` -> `benchmark` -> `scripts`.

```
quantum-pde-solvers/
|
+-- core/                            # PDE-agnostic shared infrastructure
|   +-- cases.py                     # 27-case registry (1D/2D/3D); register/get/available/describe
|   +-- config.py                    # SimConfig1D, SimConfig2D; N must be a power of 2
|   +-- exact_solutions.py           # Analytical solutions: 1D (fS, fL, fH) and 2D sinusoidal
|   +-- execution.py                 # Post-selection and state recovery: statevector, shots, device
|   +-- hardware.py                  # Device-side primitives: post-selection counts, batched estimation
|   +-- het_config.py                # HET physical constants; HETConfig, HETPhysicalConfig
|   +-- het_geometry.py              # Single SPT-100 geometry shared across all dimensions
|   +-- noise.py                     # Depolarising and shot-noise models for the robustness sweeps
|   +-- resources.py                 # Gate-count and qubit-count models; device budget from calibration
|   +-- source_functions.py          # Source functions fS, fL, fH; HET charge-density profiles
|
+-- problems/                        # Operator assembly and domain discretisation
|   +-- poisson_1d.py                # 2nd-order 1D TST matrix; PoissonProblem1D
|   +-- poisson_1d_4th.py            # 4th-order 1D pentadiagonal matrix; PoissonProblem1D4th
|   +-- poisson_line_2d.py           # 2nd-order 2D line-decomposed problem; PoissonLine2D
|   +-- poisson_line_2d_4th.py       # 4th-order 2D; j+-2 transverse stencil; boundary closure
|   +-- poisson_line_3d.py           # 2nd-order 3D line-decomposed problem
|   +-- poisson_line_3d_4th.py       # 4th-order 3D; full normal-derivative boundary treatment
|   +-- het_plasma_1d.py             # HET 1D: HETPoissonProblem1D, HETPhysicalProblem1D
|   +-- het_plasma_2d.py             # HET 2D: thin PoissonLine2D builders
|
+-- solvers/
|   +-- backend_factory.py           # Central Aer backend selection (CPU / GPU cuStateVec)
|   +-- classical/
|   |   +-- thomas.py                # Thomas tridiagonal direct solver
|   |   +-- numpy_ref.py             # NumPy reference (debugging)
|   +-- quantum/
|   |   +-- result.py                # SolverResult, VQLSSolverResult, QSVTSolverResult
|   |   +-- hhl_1d.py                # HHL for 2nd-order 1D TST systems
|   |   +-- hhl_1d_4th.py            # HHL for 4th-order 1D pentadiagonal systems
|   |   +-- trotter_pinning.py       # Pins the Trotter step count against epsilon drift
|   |   +-- vqls_utils.py            # LCU Pauli decomposition, ansatz, cost function
|   |   +-- vqls_1d.py               # VQLS 2nd-order solver; VQLSConfig1D
|   |   +-- vqls_1d_4th.py           # VQLS 4th-order solver
|   |   +-- vqls_hadamard.py         # Circuit-level Hadamard-test cost evaluation (opt-in)
|   |   +-- block_encoding.py        # Sz.-Nagy unitary dilation for TST and pentadiagonal A
|   |   +-- qsp_angles.py            # QSP phase angles via pyqsp / Chebyshev fallback; disk cache
|   |   +-- qsvt_1d.py               # QSVT 2nd-order solver; QSVTConfig1D
|   |   +-- qsvt_1d_4th.py           # QSVT 4th-order solver
|   +-- outer/                       # Single 2D/3D architecture: strip decomposition
|       +-- core.py                  # LineProblem2D / InnerSolver protocols; strip_sweep
|       +-- inner.py                 # Validated (A,b)->x registry; thomas/hhl/vqls/qsvt
|       +-- stationary.py            # Jacobi / Gauss-Seidel / SOR
|       +-- multigrid.py             # V-cycle / Full Multigrid (FMG)
|
+-- benchmark/                       # Evaluation orchestration and reporting
|   +-- equal_accuracy.py            # Equal-accuracy protocol: matched-residual comparison
|   +-- sensitivity.py               # OAT sensitivity sweeps across solver parameters
|   +-- tables.py                    # LaTeX and ASCII table generation for thesis output
|   +-- hardware.py                  # IBM Quantum hardware adapter (ZNE; opt-in)
|   +-- metrics.py                   # BenchmarkResult, BenchmarkResult2D, compute_errors
|   +-- plotting.py                  # Matplotlib figure primitives
|   +-- reporting.py                 # Tabular console output for 1D and 2D results
|   +-- runner.py                    # Sweep drivers A-H4; run_pair_1d, run_pair_2d
|   +-- diagnostics.py               # Comparison-table / study primitives for debug scripts
|   +-- reference_2d.py              # Fine-mesh FMG reference solution for 2D error metrics
|   +-- hpc_plotting.py              # HPC sweep post-processing: load -> reshape -> draw -> save
|   +-- study_plotting.py            # Equal-accuracy and sensitivity study figures
|   +-- thesis_figures.py            # The dissertation's main-body figures and their tidy CSVs
|   +-- hpc_archive.py               # Legacy on-disk schema for existing HPC run directories
|   +-- results_io.py                # Publication archive schema (new runs); read + write
|
+-- scripts/                         # Laptop-scale entry points
|   +-- tutorial.py                  # START HERE -- --dim {1,2,3}, --inner, --scheme
|   +-- make_thesis_figures.py       # Builds every main-body figure and its CSV from the archives
|   +-- debug/                       # Interactive per-dimension diagnostics
|   |   +-- debug_1d.py              # 1D: raw (A,b) cases, kappa tables, QSVT dump
|   |   +-- debug_1d_4th.py          # 4th-order 1D; convergence-order verification
|   |   +-- debug_2d.py              # 2D scheme comparison, noise study, polish study
|   |   +-- debug_3d.py              # 3D equivalent of debug_2d.py
|   +-- studies/                     # The parameter studies reported in the dissertation
|   |   +-- resource_feasibility_1d.py   # Transpiled gate counts against the device budget
|   |   +-- robustness_sweep_1d.py       # Shot noise, depolarising sweep, fake backend
|   |   +-- hhl_shot_overhead.py         # Post-selection overhead, measured against 1/kappa^2
|   |   +-- vqls_noisy_convergence_1d.py # Does COBYLA converge on a shot-based cost?
|   |   +-- qsvt_2d_line_degree_sweep.py # QSVT degree against accuracy and device fidelity
|   +-- hardware/                    # Real-device submission (second environment; see 2.1)
|   |   +-- ibm_hardware_run.py          # Submission entry point
|   |   +-- block_encoding_fidelity.py   # Direct fidelity estimation of one block encoding
|   |   +-- qsvt_degree_composition_hardware.py # Fidelity against degree; the composition law
|   |   +-- delta_amplification_hardware.py     # Amplitude-amplification feasibility
|   |   +-- compare_mitigation.py        # Paired run with and without readout mitigation
|   +-- utils/                       # Archive maintenance and auditing
|   |   +-- circuit_census.py            # Transpiled depth and two-qubit counts, merged into a sweep
|   |   +-- gap_analysis.py              # What a sweep is missing; writes a rerun manifest
|   |   +-- recover_orphan_rows.py       # Rebuilds summary rows from surviving .npz fields
|   |   +-- normalise_recovered_metrics.py # Recomputes recovered rows through the runner's metrics
|   |   +-- check_geometry_impact.py     # Which cases a geometry change actually moves
|   |   +-- make_zenodo_package.py       # Builds the per-solution field deposit
|   |   +-- zenodo_upload.py             # Uploads it against .zenodo.json
|   +-- matlab/                      # Optional MATLAB re-rendering of the tidy CSVs
|
+-- hpc/                             # Cluster deployment for Imperial CX3 -- see hpc/README.md
|   +-- setup_hpc_env.sh             # One-time env setup: CPU qpde + GPU qpde-gpu venvs
|   +-- runners/                     # Python driver code (portable; no PBS dependency)
|   |   +-- run_1d.py                # Full 1D sweep: N=4..64, all solvers
|   |   +-- run_2d.py                # Full 2D sweep
|   |   +-- run_3d.py                # Full 3D sweep
|   |   +-- run_studies.py           # Equal-accuracy and sensitivity studies
|   |   +-- precompute_phases.py     # QSVT phase-angle precompute; --dim {1,2,3}, --order {2,4}
|   |   +-- plot_results.py          # Sweep post-processing; --dim {1,2,3}
|   |   +-- plot_studies.py          # Study post-processing
|   |   +-- make_tables.py           # Renders the booktabs tables into results/*/tables/
|   +-- jobs/                        # PBS Pro job scripts (site-specific)
|       +-- _preflight.sh            # Git-state and module gate; sourced by every job
|       +-- submit_precompute_hpc.sh # 1D QSVT phase-angle precompute
|       +-- submit_precompute_2D.sh  # 2D QSVT phase-angle precompute
|       +-- submit_precompute_4th.sh # 4th-order phase-angle precompute; DIM selects dimension
|       +-- submit_hpc_1D.sh         # Full 1D CPU sweep
|       +-- submit_hpc_gpu.sh        # Full 1D GPU sweep (L40S / cuStateVec)
|       +-- submit_hpc_2D.sh         # Full 2D sweep
|       +-- submit_hpc_3D.sh         # Full 3D sweep
|       +-- submit_hpc_1D_4th.sh     # 4th-order 1D sweep
|       +-- submit_hpc_2D_4th.sh     # 4th-order 2D sweep
|       +-- submit_hpc_3D_4th.sh     # 4th-order 3D sweep
|       +-- submit_studies.sh        # Equal-accuracy and sensitivity studies; DIM selects dimension
|       +-- submit_census.sh         # Transpiled gate counts merged into an existing sweep
|
+-- results/                         # Recorded measurements -- see results/README.md
|   +-- {1,2,3}Dhpc_run{,_4th}/      # The sweeps: summaries, metadata, tidy CSVs
|   +-- {1,2,3}Dstudies{,_4th}/      # Equal-accuracy and sensitivity studies
|   +-- thesis/                      # One tidy CSV per main-body figure, F1-F9
|   +-- qsvt_phase_cache/            # QSP phase angles, keyed (kappa, epsilon, method, max_degree)
|   +-- investigations/              # ibm_kingston hardware runs and calibration
|   +-- manifests/                   # Recorded gap-analysis scopes
|
+-- tests/                           # Pytest suite: 742 tests
+-- quantum_linear_solvers/          # Submodule: the patched Carrera Vazquez et al. fork
+-- .zenodo.json                     # Metadata for the per-solution field deposit
+-- CITATION.cff
+-- requirements.txt
+-- README.md
```

---

## 2. Prerequisites and installation

Python 3.11 or newer. The floor is set by the pinned stack, not by preference: numpy, scipy and PennyLane each declare `requires-python >= 3.11`. CX3 uses 3.12.3 via `Python/3.12.3-GCCcore-13.3.0`.

**Clone with submodule**

```bash
git clone --recurse-submodules https://github.com/jatf-gt/quantum-pde-solvers.git
cd quantum-pde-solvers
```

If cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

**Set up the environment**

```bash
conda create -n msc_qiskit python=3.11
conda activate msc_qiskit
pip install -r requirements.txt
pip install -e quantum_linear_solvers/
```

**Key dependencies**

| Package | Purpose |
| --- | --- |
| `qiskit >= 1.0` | Circuit construction and statevector simulation |
| `qiskit-aer` | High-performance Aer backend (CPU and GPU paths) |
| `pennylane` | VQLS variational optimisation |
| `pyqsp` | QSP phase-angle computation for QSVT |
| `numpy`, `scipy` | Classical linear algebra |
| `matplotlib` | Plotting |
| `openpyxl` | Excel export for benchmark metrics |
| `pytest` | Test suite |

> **Compatibility note.** `quantum_linear_solvers/` has been patched to replace the deprecated `QuantumCircuit.isometry()` with the `Isometry` gate from `qiskit.circuit.library`. This patch is applied to the vendored source and requires no user action. Do not revert it.

### 2.1 Two-environment setup

The pinned environment (`qiskit==1.4.5`) produces every simulator and HPC result, and it is the environment the regression baseline is locked against. 

A **second environment** with `qiskit >= 2.x` and `qiskit-ibm-runtime >= 0.40` is required *only* for real-hardware submission (everything under `scripts/hardware/`). Hardware scripts deliberately avoid importing PennyLane-dependent modules so they run cleanly in this minimal second environment.

Local pinned versions (`qiskit==1.4.5`, `qiskit-aer==0.17.2`, `pennylane==0.45.0`, `pyqsp==0.2.0`) differ from the HPC venv (`qiskit==0.45.3`). The two environments are not required to match.

---

## 3. Local execution

All scripts run from the repository root. `pyproject.toml` sets pytest's `pythonpath = ["."]`, so `core`, `problems`, `solvers` and `benchmark` resolve as top-level packages.

### 3.1 `tutorial.py` -- start here

```bash
python scripts/tutorial.py --dim 2 --N 32
python scripts/tutorial.py --dim 1 --N 8 --inner all
python scripts/tutorial.py --dim 2 --N 8 --inner qsvt
python scripts/tutorial.py --list-cases       # all registered cases for --dim
python scripts/tutorial.py --list-options     # all tunable inner/scheme parameters
```

**Runtime:** seconds for classical; under a minute for one quantum solver at $N \le 16$.

### 3.2 `debug/debug_1d.py` -- 2nd-order 1D diagnostics

Runs any of the 11 registered 1D cases (including raw-matrix sub-cases 3b and 3c) through the inner-solver registry.

```bash
python scripts/debug/debug_1d.py --case poisson_1d_fS_hom --N 8
python scripts/debug/debug_1d.py --case het_1d_3c_neumann --N 16 --inner qsvt
python scripts/debug/debug_1d.py --dump --case het_1d_3a_linear --N 8 --inner qsvt
python scripts/debug/debug_1d.py --kappa-table      # kappa(N) vs O(N^2) scaling
```

### 3.3 `debug/debug_1d_4th.py` -- 4th-order 1D diagnostics

Verifies $\mathcal{O}(h^4)$ convergence for the pentadiagonal discretisation across all three solvers. Checks boundary closure, condition number, and polynomial degree relative to 2nd-order.

```bash
python scripts/debug/debug_1d_4th.py --N 8
python scripts/debug/debug_1d_4th.py --convergence-order --solver qsvt
```

### 3.4 `debug/debug_2d.py` / `debug/debug_3d.py` -- outer-scheme diagnostics

Compares inner solvers and outer schemes on line-decomposed problems: scheme comparison table, multigrid hierarchy inspection, inner-solver noise tolerance, polish studies.

```bash
python scripts/debug/debug_2d.py --case square --N 64
python scripts/debug/debug_2d.py --case het --N 8 --inner hhl
python scripts/debug/debug_2d.py --N 64 --scheme fmg -S nu1=2 -S n_coarse=8
python scripts/debug/debug_2d.py --noise-study --N 32
python scripts/debug/debug_3d.py --case cube --N 16
python scripts/debug/debug_3d.py --convergence-study --case cube
```

Pass `--scheme jacobi` to reproduce the originally published line-Jacobi results at correspondingly higher cost.

### 3.5 `utils/gap_analysis.py` -- HPC result gap detection

Scans a sweep directory and emits a rerun manifest. It separates a row that was never computed from one whose solution `.npz` survived a walltime kill while its summary row did not, because the second is recoverable without recomputing anything. This is the only safe way to identify what needs resubmission; without it, a partial sweep and a broken filename convention are indistinguishable.

```bash
python scripts/utils/gap_analysis.py --dim 2 --results-dir results/2Dhpc_run
```

### 3.6 Further diagnostic scripts

| Script | Purpose |
| --- | --- |
| `studies/resource_feasibility_1d.py` | Transpiled two-qubit counts against the measured device budget |
| `studies/robustness_sweep_1d.py` | Shot noise, depolarising sweep and fake backend |
| `studies/hhl_shot_overhead.py` | Post-selection overhead, measured against the $1/\kappa^2$ expectation |
| `studies/vqls_noisy_convergence_1d.py` | Whether COBYLA still converges on a shot-based cost function |
| `studies/qsvt_2d_line_degree_sweep.py` | QSVT degree against algorithmic accuracy and device fidelity |
| `hardware/block_encoding_fidelity.py` | Unitarity and direct fidelity estimation for the Sz.-Nagy encoding |
| `hardware/delta_amplification_hardware.py` | Hardware-adapted amplitude amplification feasibility |
| `utils/circuit_census.py` | Transpiled depth and gate counts, merged into an existing sweep |
| `make_thesis_figures.py` | Every main-body figure and its tidy CSV, from the recorded archives |

The three scripts under `utils/` that maintain the archive rather than measure
anything — `recover_orphan_rows.py`, `normalise_recovered_metrics.py` and
`check_geometry_impact.py` — are provenance for rows already in `results/`, not
part of a fresh run. A replication does not need them.

### 3.7 Test suite

```bash
pytest                         # 742 tests
pytest -m "not quantum"        # the classical subset, no backend needed
pytest tests/test_outer.py -v  # single file
pytest tests/test_hhl_1d.py::TestHHL1D::test_agrees_with_thomas_loose -v
```

See [Section 8](#8-test-suite) for full coverage details.

---

## 4. HPC execution (Imperial CX3)

The HPC surface separates driver code from deployment config: `hpc/runners/` holds ordinary Python (portable to any cluster); `hpc/jobs/` holds PBS Pro submission scripts (CX3-specific). See `hpc/README.md` for the full operational reference.

> **Before any submission.** `_preflight.sh` blocks execution on a dirty git tree or a missing pentadiagonal module. Ensure the working tree is clean and the submodule is populated.

### 4.1 One-time environment setup

```bash
ssh username@login.cx3.hpc.ic.ac.uk
bash hpc/setup_hpc_env.sh
```

Creates two venvs under `~/venvs/`:

| Venv | Contents |
| --- | --- |
| `qpde` (CPU) | `qiskit==0.45.3`, `qiskit-aer==0.13.3`, standard scientific stack |
| `qpde-gpu` (GPU) | Same stack but `qiskit-aer-gpu==0.15.1` (CUDA 12 / cuStateVec) |

`qiskit-aer` and `qiskit-aer-gpu` cannot coexist in a single environment; the two venvs exist for this reason.

### 4.2 1D sweep (CPU)

```bash
qsub hpc/jobs/submit_hpc_1D.sh
```

Runs `hpc/runners/run_1d.py` over $N=4 \ldots 64$, all cases, all four solvers. Resource: `select=1:ncpus=4:mem=128gb`, `walltime=24:00:00`.

**Useful overrides:**

```bash
export MAX_N=16;      qsub -v MAX_N hpc/jobs/submit_hpc_1D.sh
export SKIP_QSVT=1;   qsub -v SKIP_QSVT hpc/jobs/submit_hpc_1D.sh
```

Results write incrementally: each `.npz` is saved as produced, and `results_full.json` is rewritten after every completed work unit, so a walltime kill loses neither. Restrict the scope of a resubmission and the runner merges into what is already there; see `hpc/README.md`.

### 4.3 1D sweep (GPU)

```bash
qsub hpc/jobs/submit_hpc_gpu.sh
```

Targets the `gpu72` queue with an NVIDIA L40S (48 GB, compute capability 8.9). Executes serially (`--max-workers 1`) to avoid CUDA context conflicts. Resource: `select=1:ncpus=8:mem=64gb:ngpus=1:gpu_type=L40S`, `walltime=24:00:00`.

Expected speedup over CPU: 10-50x at $N=16$ for QSVT, making previously infeasible $N$ tractable.

### 4.4 2D and 3D sweeps

```bash
qsub hpc/jobs/submit_hpc_2D.sh
qsub hpc/jobs/submit_hpc_3D.sh
```

Mirror the 1D driver's incremental-write behaviour.

The 2D and 3D cost profile differs from 1D: a 2D/3D configuration is an outer iteration over many strip solves, so the outer scheme choice matters more than the solver. Use `--scheme fmg` unless reproducing published line-Jacobi results, for which `--scheme jacobi` exists.

Phase precompute for the 2D strip operator is cheap: $\kappa_\text{row} \to 3^-$ gives polynomial degrees 30-85 at every $N$. The whole set finishes in minutes; no staging is needed.

### 4.5 QSVT phase-angle precompute

```bash
# Stage 1: safe sizes
export N_VALUES="4,8,16"
qsub -v N_VALUES hpc/jobs/submit_precompute_hpc.sh

# Stage 2: N=32, exploratory
export N_VALUES="32"; export MAX_DEGREE="2000"
qsub -v N_VALUES,MAX_DEGREE hpc/jobs/submit_precompute_hpc.sh

# Stage 3: N=64 only after Stage 2 is confirmed
export N_VALUES="64"; export MAX_DEGREE="2000"
qsub -v N_VALUES,MAX_DEGREE hpc/jobs/submit_precompute_hpc.sh
```

> **PBS quirk.** Pass `N_VALUES` / `MAX_DEGREE` as `qsub -v NAME` (bare name, value from the shell's exported variable). Using `qsub -v NAME=value` breaks PBS's comma-splitting on comma-separated lists.

Results accumulate in `results/qsvt_phase_cache/` across stages; nothing already cached is recomputed. Resource: single-threaded, `mem=32gb`, `walltime=71:00:00`. $N=32$ and $N=64$ are not guaranteed to finish in one 71 h submission.

The 2D precompute is a separate job (`submit_precompute_2D.sh`) and completes in minutes.

The cache key is `(round(kappa, 4), round(epsilon, 8), method, max_degree)`. A condition number differing in the fourth decimal place is a silent cache miss that forces the full phase computation into the sweep. A previous hardcoded 2D table had drifted by up to 0.28 for this reason; $\kappa$ is now derived live from the same problem classes the solvers use.

### 4.6 Post-processing

```bash
python hpc/runners/plot_results.py --dim 1 --results-dir results/1Dhpc_run --save-pdf
python hpc/runners/plot_results.py --dim 2 --results-dir results/2Dhpc_run --save-pdf
python hpc/runners/plot_results.py --dim 3 --results-dir results/3Dhpc_run --save-pdf
```

1D produces 25 figures (solution profiles, convergence, residual, wall time, HET profiles). 2D and 3D produce 3 scalar-metric figures each. 2D and 3D share a result schema and therefore share their plotting code; 1D does not (its rows carry no `scheme`, `linf_err`, `weighted_cost`, or `err_vs_thomas`).

The on-disk schema — filename convention, field-name aliases (`u_solver` / `phi_solver` / `phi`) — is declared in `benchmark/results_io.py` for new runs and `benchmark/hpc_archive.py` for the existing HPC run directories. Do not conflate the two: `hpc_archive.py` is read-only and describes data already on disk that cannot be rewritten.

### 4.7 4th-order HPC sweeps

```bash
qsub -v DIM=1 hpc/jobs/submit_precompute_4th.sh   # phase angles first
qsub hpc/jobs/submit_hpc_1D_4th.sh
```

The pentadiagonal sweeps produce every fourth-order number in the dissertation.
Each is gated on the fourth-order phase cache: a cache miss degrades silently to
a reduced-degree solve, so the job refuses to start QSVT without it. In 1-D the
angles must be staged smallest-$N$ first, and $N \le 16$ computed **uncapped**
while $N \ge 32$ is capped, because the cap forms part of the cache key. In 2-D
and 3-D $\kappa \le 3.14$ and the whole set computes in seconds, but one
resolution contributes several keys — two strip operators in 2-D, up to four in
3-D — and a sweep requests all of them.

### 4.8 Reproducing the study end to end

The order below is the whole campaign. Stages 2 and 3 are independent of each
other; everything else is sequential. Wall-clock totals are cluster time, not
elapsed time, and the 3-D sweep dominates.

```bash
# 1. Phase angles. Sweeps refuse to run QSVT without them.
export N_VALUES="4,8,16"; qsub -v N_VALUES hpc/jobs/submit_precompute_hpc.sh
export N_VALUES="32";     qsub -v N_VALUES hpc/jobs/submit_precompute_hpc.sh
export N_VALUES="64";     qsub -v N_VALUES hpc/jobs/submit_precompute_hpc.sh
qsub hpc/jobs/submit_precompute_2D.sh
qsub -v DIM=1 hpc/jobs/submit_precompute_4th.sh
qsub -v DIM=2 hpc/jobs/submit_precompute_4th.sh
qsub -v DIM=3 hpc/jobs/submit_precompute_4th.sh

# 2. Second-order sweeps.
qsub hpc/jobs/submit_hpc_1D.sh
qsub hpc/jobs/submit_hpc_2D.sh
qsub hpc/jobs/submit_hpc_3D.sh

# 3. Fourth-order sweeps.
qsub hpc/jobs/submit_hpc_1D_4th.sh
qsub hpc/jobs/submit_hpc_2D_4th.sh
qsub hpc/jobs/submit_hpc_3D_4th.sh

# 4. Equal-accuracy and sensitivity studies, per dimension.
for d in 1 2 3; do qsub -v DIM=$d hpc/jobs/submit_studies.sh; done

# 5. Transpiled gate counts, merged into the sweeps. Always set N_VALUES.
export N_VALUES="4,8,16,32"; qsub -v N_VALUES hpc/jobs/submit_census.sh

# 6. Post-processing and the dissertation figures. No cluster job needed.
for d in 1 2 3; do python hpc/runners/plot_results.py --dim $d; done
python hpc/runners/make_tables.py
python scripts/make_thesis_figures.py --no-titles --out-dir <figures-dir>
```

The device measurements of Section 10 are separate and need the second
environment of Section 2.1 plus an IBM Quantum account; they are not part of
the simulator campaign and nothing in the dissertation's main results depends
on them.

After every stage, `python scripts/utils/gap_analysis.py --dim {1,2,3}` reports
what is missing before you conclude a stage is done.

### 4.9 Monitoring

```bash
qstat -u $USER
tail -f results/2Dhpc_run/run.log     # may be unreadable; see note below
tail -f results/2Dhpc_run_pbs.log     # PBS stdout stream; usually readable
ls -la --time-style=full-iso results/2Dhpc_run/*.npz | tail -5   # vital check
```

> **Known issue.** On CX3's network filesystems, `run.log` is sometimes listed by `ls` but unreadable by `tail` due to metadata caching. The `.npz` modification-time check is the reliable progress indicator: if `.npz` mtimes are advancing, the job is doing real work.

---

## 5. Benchmark sweep catalogue

### Generic Poisson sweeps

| Sweep | Dim | Description |
| --- | --- | --- |
| A | 1D | Homogeneous BCs; sources fS, fL, fH; $N \in \{8, 16\}$; $\varepsilon = 0.01$ |
| B | 1D | Trotter / VQLS $\varepsilon$ sensitivity; $N = 16$; $\varepsilon \in \{0.1, 0.01, 0.001\}$ |
| C | 1D | Non-homogeneous Dirichlet BCs; fH; $N \in \{16, 32\}$ |
| D | 1D | Condition-number scaling $\kappa \sim (4/\pi^2)(N+1)^2$; $N \in \{4, 8, 16, 32\}$ |
| E | 2D | Homogeneous BCs; fS, fL, fH; $N \in \{8, 16\}$; $\varepsilon \in \{0.01, 0.5\}$ |
| F | 2D | Non-homogeneous BCs; asymmetric convergence behaviour |
| G | 2D | Row-matrix condition number $\kappa_\text{row} \to 3^-$; $N \in \{4, 8, 16, 32\}$ |

### HET plasma sweeps

| Sweep | Description |
| --- | --- |
| H1 | 1D Gaussian profile; homogeneous BCs; $N \in \{4, 8\}$ |
| H2 | 1D Gaussian profile; physical BCs ($V_d = 300$ V); $N \in \{4, 8\}$ |
| H3 | 1D all three charge-density profiles; $N = 8$; physical BCs |
| H4 | Condition number and $\alpha = L^2/\lambda_D^2$ scaling diagnostics |

---

## 6. Physical application: HET plasma

### Problem formulation

The electrostatic potential $\tilde{\phi}$ in the discharge channel satisfies:

$$\frac{d^2\tilde{\phi}}{d\tilde{x}^2} = -\alpha\,\delta\tilde{n}(\tilde{x}) \qquad \text{(1D axial)}$$

$$\frac{\partial^2\tilde{\phi}}{\partial\tilde{x}^2} + \frac{\partial^2\tilde{\phi}}{\partial\tilde{y}^2} = -\alpha\,\delta\tilde{n}(\tilde{x},\tilde{y}) \qquad \text{(2D axial-radial)}$$

where $\alpha = L^2/\lambda_D^2$ is the dimensionless Debye scaling parameter and $\delta\tilde{n} = (n_i - n_e)/n_0$ is the non-dimensional net charge density.

### Physical parameters (Boeuf and Garrigues 1998, Table 1)

| Parameter | Symbol | Value |
| --- | --- | --- |
| Channel length | $L$ | 25 mm |
| Discharge voltage | $V_d$ | 300 V |
| Electron temperature | $T_e$ | 20 eV |
| Reference density | $n_0$ | $5 \times 10^{17}$ m$^{-3}$ |
| Debye length | $\lambda_D$ | $\approx 128\,\mu$m |
| Scaling parameter | $\alpha = L^2/\lambda_D^2$ | $\approx 38{,}000$ |

### Charge-density profiles

| Key | Profile | Description |
| --- | --- | --- |
| `gaussian` | $\delta\tilde{n} = \delta_0\exp\!\bigl(-(\tilde{x}-\tilde{x}_\text{peak})^2/\sigma^2\bigr)$ | Smooth ionisation zone near exit plane |
| `linear` | $\delta\tilde{n} = \delta_0\,\tilde{x}$ | Uniform space-charge gradient; analytical solution available |
| `step` | $\delta\tilde{n} = \delta_0\,\mathrm{sign}(\tilde{x}-\tilde{x}_\text{ion})$ | Sharp ionisation front |

The charge-separation amplitude is $\delta_0 = \delta_{0,\text{factor}}/\alpha$ (default $\delta_{0,\text{factor}} = 5$), keeping $\alpha\,\delta_0 = \mathcal{O}(1)$.

### 2D analytical solution

For the sinusoidal source $f(\tilde{x},\tilde{y}) = -2\pi^2\sin(\pi\tilde{x})\sin(\pi\tilde{y})$ with homogeneous Dirichlet boundary conditions:

$$\tilde{\phi}(\tilde{x},\tilde{y}) = \sin(\pi\tilde{x})\sin(\pi\tilde{y})$$

This manufactured solution allows rigorous error quantification for all three quantum solvers in 2D without a numerical reference.

---

## 7. Algorithm summary

All three algorithms target the linear system $A|x\rangle = |b\rangle$.

### Complexity at a glance

| Algorithm | Circuit depth | Qubit count | Condition-number scaling |
| --- | --- | --- | --- |
| HHL | $\mathcal{O}(\kappa^2 \log N / \varepsilon)$ | $\mathcal{O}(\log N + \log\kappa)$ | Quadratic in $\kappa$ |
| VQLS | $\mathcal{O}(n_\text{layers} \cdot n_\text{qubits})$ | $\mathcal{O}(\log N)$ | No formal guarantee |
| QSVT | $\mathcal{O}(\kappa\log(1/\varepsilon))$ | $\mathcal{O}(\log N + 2)$ | Linear in $\kappa$ |

For the 1D Poisson matrix with $\kappa \sim \mathcal{O}(N^2)$, QSVT achieves a quadratic improvement over HHL. For the 2D row matrix with $\kappa_\text{row} \to 3^-$, the QSVT polynomial degree is essentially constant in $N$:

$$d_\text{2D} \approx \mathcal{O}\!\left(\kappa_\text{row}\log(1/\varepsilon)\right) \approx \mathcal{O}\!\left(3\log(1/\varepsilon)\right)$$

This degree gap (degree ~939 for the 1D $N=4$ matrix vs ~33 for the 2D row matrix) is why 1D phase precompute requires staged cluster treatment while the 2D precompute finishes in minutes.

### QSVT block encoding

The Sz.-Nagy unitary dilation with $M = A/\alpha$, $\|M\|_2 \le 1$:

$$U_A = \begin{pmatrix} M & \sqrt{I - M^2} \\ \sqrt{I - M^2} & -M \end{pmatrix}$$

satisfies $(\langle 0|_\text{anc} \otimes I_N)\,U_A\,(|0\rangle_\text{anc} \otimes I_N) = M = A/\alpha$ exactly using a single ancilla qubit. $\alpha = \|A\|_2$ via eigendecomposition. This encoding works for both TST (tridiagonal) and pentadiagonal matrices.

### VQLS cost function

$$C(\boldsymbol{\theta}) = 1 - \frac{\left|\langle b\,|\,A\,|\,x(\boldsymbol{\theta})\rangle\right|^2}{\langle x(\boldsymbol{\theta})\,|\,A^\dagger A\,|\,x(\boldsymbol{\theta})\rangle}$$

Evaluated by direct statevector arithmetic over the Pauli LCU decomposition of $A$ (`vqls_utils.py`). Optimisation uses COBYLA with a three-stage restart ($\rho_\text{beg} \in \{0.5, 0.1, 0.01\}$). Note: the bound $C \ge r^2/\kappa^2$ means a low cost value does not guarantee a low residual; see the equal-accuracy protocol below.

`vqls_hadamard.py` provides a circuit-level Hadamard-test implementation of the same cost function for use with shot-based hardware. It is an opt-in extension that does not touch the regression-pinned `vqls_1d.py` path.

### 2D line decomposition

$$u^{n+1}_{i+1,j} - 4\,u^{n+1}_{i,j} + u^{n+1}_{i-1,j} = h^2 f(x_i,y_j) - \left(u^n_{i,j-1} + u^n_{i,j+1}\right)$$

Each row sub-problem is a TST system with $\kappa_\text{row} \to 3^-$ as $N \to \infty$.

### 4th-order (pentadiagonal) discretisation

The five-point stencil (coefficients $-30, 16, -1$ divided by $12h^2$) gives $\mathcal{O}(h^4)$ truncation error against $\mathcal{O}(h^2)$ for the standard three-point stencil. Practical consequences:

- The same solution accuracy is reached at $N^{1/2}$ grid points, reducing the amplitude-encoded register by one qubit per halving of $N$.
- $\kappa$ increases by roughly 2.5x over the tridiagonal case at the same $N$, raising QSVT polynomial degree moderately.
- The block encoding and Pauli LCU decomposition are extended to pentadiagonal structure; the Sz.-Nagy dilation still uses a single ancilla qubit.

**1D boundary closure.** The original even-reflection closure injected an $\mathcal{O}(1)$ consistency error, capping convergence at order 2. The correct formulation adjusts the right-hand side using explicit boundary source data: $b[0] \mathrel{-}= 14\alpha$ and $b[0] \mathrel{+}= h^2 f(0)$. The matrix $A$ remains symmetric, preserving $\kappa$ and validating all cached phase angles.

**2D/3D boundary closure.** The normal second derivative at a boundary face requires tangential components: $\partial^2 u/\partial n^2|_\text{face} = f|_\text{face} - \sum_t \partial^2 u/\partial t^2|_\text{face}$. Tangential terms depend only on Dirichlet data. Using $f$ alone leaves a residual error of $-f_\text{face}/12$ at the boundary row. The updated `poisson_line_2d_4th.py` and `poisson_line_3d_4th.py` implement this rigorously, with the strip sweep extending to $j\pm 2$ transverse stencil coupling.

### Equal-accuracy protocol

A naive comparison at nominally equal precision parameters is methodologically unsound:

1. The VQLS cost $C$ is not the residual $r$. The bound $C \ge r^2/\kappa^2$ means cost $10^{-6}$ guarantees only $r \le \kappa \times 10^{-3}$ ($\approx 0.95\%$ at $\kappa \approx 9.5$, $N=4$).
2. HHL $\varepsilon$ and Trotter steps are coupled ($n_T = \lceil 1/\varepsilon \rceil$). Reducing $\varepsilon$ changes both QPE resolution and Hamiltonian simulation error simultaneously.
3. The QSVT residual is not monotone in polynomial degree due to oscillatory Chebyshev approximation error.

`benchmark/equal_accuracy.py` implements the correct protocol: for each solver and each problem $(N, \text{case})$, sweep the primary precision parameter, measure the achieved residual, and select the result whose residual falls within an acceptance band around a shared target $r_\text{target}$. All solvers are then compared at the same achieved accuracy.

`benchmark/sensitivity.py` provides one-at-a-time (OAT) sweeps, restricted to $N \in \{4, 8\}$ to keep total runtime tractable (a full OAT sweep at $N=8$ for QSVT takes $\approx 18$ minutes).

`benchmark/tables.py` generates LaTeX tables (booktabs/siunitx style, directly `\input{}`-able into the thesis) and aligned ASCII tables for HPC log inspection. Table catalogue: `primary_comparison`, `equal_accuracy`, `sensitivity`, `circuit_resources`, `het_application`, `order_comparison`.

### IBM Quantum hardware adapter

`benchmark/hardware.py` provides a thin adapter for submitting benchmark circuits to real IBM Quantum hardware via Qiskit IBM Runtime. Zero-noise extrapolation (ZNE) via `RuntimeEstimatorV2` is supported (noise scaling factors $[1, 2, 3]$, linear extrapolation). Hardware execution is gated behind `ENABLE_HARDWARE_RUN` to prevent accidental job submission. Hardware results supplement the primary statevector benchmark; they do not replace it.

---

## 8. Test suite

```bash
pytest                          # 742 tests
pytest -m "not quantum"         # the classical subset, no backend needed
pytest tests/test_outer.py -v
pytest tests/test_hhl_1d.py::TestHHL1D::test_agrees_with_thomas_loose -v
```

`quantum` is the only marker; it is applied to every test that builds and simulates a circuit. There is no `slow` marker.

### Coverage by file

Counted with `pytest --collect-only`, not from memory.

| File | Coverage | Tests |
| --- | --- | --- |
| `test_cases.py` | Every registered case: identity, geometry, source, boundary data, solvability | 230 |
| `test_outer.py` | `solvers/outer`: work accounting, stagnation, strip sweep, option registry, stationary schemes, multigrid transfer operators and cycles | 84 |
| `test_problem_setup.py` | 1D matrix structure, grid, RHS, config validation, exact solutions | 41 |
| `test_poisson_line_4th.py` | 4th-order 2D/3D line operators, transverse stencil, boundary closure | 40 |
| `test_line_problems.py` | `PoissonLine2D/3D`: operators, Dirichlet absorption, periodicity, conditioning, coarsening | 36 |
| `test_hpc_archive.py` | The legacy on-disk sweep schema and its field aliases | 30 |
| `test_qsvt_1d.py` | Block-encoding unitarity, QSP angle shape, QSVT solver correctness | 24 |
| `test_execution.py` | Post-selection and state recovery across statevector, shot and device paths | 23 |
| `test_order4_wiring.py` | That a 4th-order request reaches a 4th-order operator and solver | 23 |
| `test_resources.py` | Gate-count and qubit-count models against the device budget | 22 |
| `test_het_problem.py` | HET config derived quantities, matrix structure, solver compatibility | 22 |
| `test_poisson_1d_4th.py` | The pentadiagonal 1D operator and its closure | 21 |
| `test_delta_amplification.py` | Amplitude amplification and its hardware adaptation | 18 |
| `test_vqls_1d.py` | VQLS cost convergence, parameter shape, reproducibility | 17 |
| `test_noise.py` | Depolarising and shot-noise models | 17 |
| `test_gap_analysis.py` | Missing rows, orphan rows, manifest generation | 15 |
| `test_hpc_runners.py` | Runner argument handling and the append/merge contract | 14 |
| `test_vqls_hadamard.py` | Circuit-level Hadamard-test cost evaluation | 13 |
| `test_hardware.py` | The device adapter, gated so it never submits | 13 |
| `test_neumann_encoding.py` | The Neumann sub-case and its encoding | 12 |
| `test_hhl_1d.py` | HHL solution shape, sign, proportionality recovery | 11 |
| `test_classical_solvers.py` | Thomas 1D accuracy, NumPy agreement | 9 |
| `test_regression_baseline.py` | The locked tagged numbers in `tests/baselines/baseline_v1.json` | 7 |
| **Total** | | **742** |

**A known coverage gap.** There is no end-to-end regression test of the 2-D
chain from `PoissonLine2D` through `solvers.outer.solve` to the reporting
adapter. The module that covered it was written against the pre-Phase-8
reporting schema, could not import after that rewrite, and had been inert
behind a module-level skip ever since; it is removed rather than left to be
counted as cover it does not provide. Its components are covered
(`test_line_problems.py`, `test_outer.py`); the assembled chain is not.

On the pinned environment of Section 2 the suite runs clean: 742 collected, no
failures. Three checks skip themselves when the archive they read against is
not in the checkout, so a run may report a few skips rather than a full pass.

**`pyqsp` is a hard dependency of the QSVT tests.** It is listed in
`requirements.txt`, and without it every test that builds real QSP angles fails
on import: twelve in `test_qsvt_1d.py` and two in `test_neumann_encoding.py`.
`solvers/quantum/qsp_angles.py` falls back to a Chebyshev construction for
ordinary use, so the rest of the suite passes without it, but a green run
requires it installed. Those fourteen are also the slow ones: generating real
angles, rather than the fallback, is most of the suite's runtime.

### Two load-bearing tests

**Line-Jacobi reproducibility.** `test_outer.py` reconstructs the original line-Jacobi loop from first principles and asserts `scheme="jacobi"` reproduces it exactly (field, iteration count, stopping point). The retired implementation is gone; this test is the only guard on the published 2D figures.

**Option validation.** The inner-solver registry must reject unknown keys, not absorb them silently. A registry that ignored `max_degrees=500` would let an HPC run cost an order of magnitude more than intended while appearing to honour the setting.

### Pass / fail criteria

Tests verify that solvers return results of the correct shape, produce finite values, agree with Thomas to within a loose tolerance (20% for HHL, 15% for VQLS, 20% for QSVT at $N=4$), preserve the correct sign, and raise appropriate exceptions for invalid inputs.

---

## 9. Methodological notes

### Statevector simulation

All circuits are evaluated via `qiskit.quantum_info.Statevector` (deterministic, no shot noise), consistent with Ghafourpour and Laizet (2025). This establishes the theoretical algorithmic baseline but cannot capture NISQ noise or shot-based measurement variance; the `hardware.py` adapter addresses the latter for small $N$.

### HHL solution extraction

The output state is post-selected on the ancilla (flag) qubit being $|1\rangle$ and the QPE clock register being $|0\cdots0\rangle$. The proportionality constant is recovered against the normalised system $A/\|A\|_2$ to prevent Trotter error from being amplified by $\|b\|/\|A\|_2$, which is large in physically scaled problems (for the HET case, $\|b\| \sim \alpha h^2 \sim 700$).

### QSVT phase angles

`pyqsp.PolyOneOverX.generate(kappa, epsilon)` selects the polynomial degree internally. For the 1D Poisson matrix at $N=4$ ($\kappa \approx 9.5$, $\varepsilon = 0.05$) the degree is ~939; for the 2D row matrix ($\kappa \approx 2.36$, $\varepsilon = 0.1$) the degree is ~33. A Chebyshev-based fallback is provided when `pyqsp` is unavailable. The residual is not monotone in degree; measure it, do not estimate from degree alone.

### 2D reference solution

`benchmark/reference_2d.py::fine_mesh_reference` runs an FMG solve on a mesh refined 19x per direction and samples back to the coarse nodes. Its `REFERENCE_TOL = 1e-10` is truncation-limited; tightening it costs ~5x runtime and changes nothing. The previous implementation used line-Jacobi capped at 5000 sweeps, which cannot converge on a $152^2$-$608^2$ mesh; any 2D error metric predating this function was measured against an under-converged ground truth.

### Reproducing the published 2D figures

Three settings are each required and each silently change the numbers if omitted:

1. `scheme="jacobi"` (defaults: `update="jacobi"`, `criterion="delta"`)
2. `patience=max_iter+1` to disable stagnation detection, which otherwise truncates the non-converging residual histories that Section IV F of Ghafourpour and Laizet (2025) sets out to show
3. `inner_options={"epsilon": max(cfg.epsilon, 0.1)}` for HHL, reproducing the 10-step Trotter cap; without it an $\varepsilon=0.01$ sweep costs 10x more for no accuracy gain

### Physical hardware viability

Extracting the full solution vector at each line-Jacobi step requires $\mathcal{O}(N)$ quantum state tomography operations per iteration. This is not realisable on near-term hardware. The framework is therefore a theoretical simulator study of algorithmic behaviour, Hamiltonian scaling, and quantum error propagation; hardware experiments are supplementary.

---

## 10. Hardware results

A hardware measurement campaign on IBM `ibm_kingston` (Heron r2, 156 qubits) was conducted in August 2026. The test circuit was the 2-D line row operator at $N_x=4$ ($\kappa_\text{row} = 2.3586$), requiring 3 qubits (2 data + 1 block-encoding ancilla). Direct Fidelity Estimation (DFE) was performed over 20 Pauli terms, 2048 shots per term.

### Summary of measured values

| Metric | Result | Interpretation |
| --- | --- | --- |
| Single-application fidelity | 0.9333 | State preparation + one block-encoding application |
| Per-application fidelity ($F_{UA}$) | 0.918 $\pm$ 0.004 | Derived from shallow sweep ($d \in [0, 7]$); $R^2 = 0.9921$ |
| Error composition | Multiplicative | Hardware error scales as $1 - F_{UA}^d$, verifying the independent-error assumption |
| Depth ceiling | $d \lesssim 21$ | Fidelity saturates at the depolarisation floor ($1/2^n = 0.125$) past $\approx 780$ gates. Deep points yield noise, an instrumental limit |
| Readout mitigation | Inconclusive | Intercept shifts physical bounds ($0.963 \to 1.033$); slope difference ($1.7\sigma$) is unresolved |

These validations underpin the resource estimates in `core/resources.py` and predictions in `scripts/hardware/block_encoding_fidelity.py`. The raw hardware result archives are under `results/investigations/`.

### Known limitation: VQLS hardware viability
VQLS is not wired for real-hardware submission. Its Hadamard-test cost function evaluates $2L + 2L^2$ individual circuits per step. Routing PennyLane QNodes one-at-a-time via `qiskit.remote` is impractically slow. Viable hardware execution requires rebuilding those circuits natively in Qiskit for batched evaluation via `core.hardware.hardware_estimate_batch` (documented in `core/hardware.py`).

---

## 11. References

1. Ghafourpour, L. and Laizet, S. (2025). Applicability of solving the one- and two-dimensional Poisson equations with the quantum Harrow-Hassidim-Lloyd algorithm. *Physical Review Applied*, 24, 024032.
2. Harrow, A. W., Hassidim, A. and Lloyd, S. (2009). Quantum algorithm for linear systems of equations. *Physical Review Letters*, 103, 150502.
3. Bravo-Prieto, C., LaRose, R., Cerezo, M., Subasi, Y., Cincio, L. and Coles, P. J. (2023). Variational quantum linear solver. *Quantum*, 7, 1188.
4. Gilyen, A., Su, Y., Low, G. H. and Wiebe, N. (2019). Quantum singular value transformation and beyond: exponential improvements for quantum matrix arithmetics. *Proceedings of the 51st Annual ACM STOC*, pp. 193-204.
5. Martyn, J. M., Rossi, Z. M., Tan, A. K. and Chuang, I. L. (2021). Grand unification of quantum algorithms. *PRX Quantum*, 2, 040203.
6. Vazquez, A. C., Hiptmair, R. and Woerner, S. (2022). Enhancing the quantum linear systems algorithm using Richardson extrapolation. *ACM Transactions on Quantum Computing*, 3, 1.
7. Boeuf, J. P. and Garrigues, L. (1998). Low frequency oscillations in a stationary plasma thruster. *Journal of Applied Physics*, 84(7), 3541-3554.
8. Brearley, P. and Laizet, S. (2024). Quantum algorithm for solving the advection equation using Hamiltonian simulation. *Physical Review A*, 110, 012430.
9. Over, P., Bengoechea, S., Brearley, P., Laizet, S. and Rung, T. (2025). Quantum algorithm for the advection-diffusion equation by direct block encoding of the time-marching operator. *Physical Review A*, 112, L010401.
10. Tennie, F., Laizet, S., Lloyd, S. and Magri, L. (2025). Quantum computing for nonlinear differential equations and turbulence. *Nature Reviews Physics*, 7, 220-230.

---

## 12. Use of generative AI

Two tools were used in building this repository and the dissertation it
supports: **Claude** (Anthropic, <https://claude.ai/code>), through
Claude Code, and **Google Antigravity** (Google, <https://antigravity.google>).
Claude was used to write and refactor parts of the benchmarking, figure and
post-processing layers. Antigravity was used to comment source code and for
routine refactoring.

---

## 13. Licence and citation

**Code: MIT** (see `LICENSE`). Use it, change it, build on it, commercially or
not. The one condition is that the copyright notice travels with it.

**Data: CC-BY-4.0.** The per-solution field deposit on Zenodo
([10.5281/zenodo.22071066](https://doi.org/10.5281/zenodo.22071066)) is a
dataset, and CC-BY is the right instrument for data as MIT is for code.
Creative Commons themselves advise against CC licences for software, which is
why the two differ.

**The submodule is not covered by either.** `quantum_linear_solvers/` is a
separate repository, a fork of the Carrera Vázquez et al. implementation, under
**Apache-2.0**. It keeps its own terms.

**On citation.** No open-source licence can compel an academic citation. MIT
requires attribution only in the narrow legal sense that the copyright notice
must be retained in copies. If you use this work in research, please cite it as
well; `CITATION.cff` carries the metadata, and GitHub's "Cite this repository"
button reads it. Cite the software and the dataset separately if you use both —
they have different identifiers, and the dataset DOI names exact bytes.
