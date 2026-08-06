# Archived scripts

Scripts kept for provenance, not for use.

Each of these produced a result that was reported at some point during the
project — a meeting slide, a verification table, a debugging session that
settled a design question. They are retained so those results remain
attributable, and deliberately **not** maintained.

## These scripts do not run

Three of the four import modules that no longer exist. The legacy 2-D solver
stack (`problems/poisson_2d.py`, `solvers/classical/thomas_2d.py`,
`solvers/quantum/hhl_2d.py`, `vqls_2d.py`, `qsvt_2d.py`) was retired once
`solvers/outer` was shown to reproduce its results exactly, and these scripts
were written against it.

**This is expected and is not a regression.** Do not "fix" the imports: the
correct response to needing one of these results again is to reproduce it
through the current architecture, which is both simpler and faster. An import
error here is a signpost, not a bug.

| Script | What it did | Superseded by |
| ------ | ----------- | ------------- |
| `run_meeting4_report.py` | Meeting 4 report: HHL/VQLS/QSVT comparison across the 1-D Poisson and HET cases, with figures and tables. | Nothing directly — a one-off report generator. The underlying comparison is now `scripts/run_1d_benchmark.py` plus `benchmark/hpc_plotting.py`. |
| `run_meeting5.py` | Meeting 5 report: added error decomposition (discretisation versus quantum algorithmic error) and the first 2-D results. | As above. The error decomposition it introduced now lives in the HPC plotting layer as `plot_error_decomposition`. |
| `run_verification_study.py` | Structured verification and validation study of HHL and VQLS on the 1-D and 2-D Poisson equation for the HET application. | `tests/` for correctness, and `scripts/run_hpc_*full.py` for the systematic sweeps. |
| `debug_2d_solvers.py` | Exploratory 2-D solver debugging; introduced line-SOR in place of line-Jacobi and demonstrated the O(N) versus O(N²) iteration counts. | `scripts/debug_outer_2d.py`, per this script's own docstring. The SOR result it established is now `solvers/outer/stationary.py`. |

## If you need one of these results again

- **A solver comparison at fixed N** — `python scripts/explore.py --dim {1,2,3}`.
- **A systematic sweep** — `scripts/run_hpc_1Dfull.py` / `2Dfull` / `3Dfull`.
- **Scheme or inner-solver debugging** — `scripts/debug_outer_2d.py`,
  `scripts/debug_outer_3d.py`.
- **Publication figures from HPC output** — `scripts/plot_hpc_*_results.py`,
  which are thin wrappers over `benchmark/hpc_plotting.py`.
