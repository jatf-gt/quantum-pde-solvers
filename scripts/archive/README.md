# Archived scripts

One-off scripts retained for provenance, so that results reported during the
project remain attributable. They are not maintained and are not expected to
run: all but `debug_2d_solvers.py` import the legacy 2-D solver stack at module
level, and that stack was retired once `solvers/outer` was shown to reproduce
its results exactly.

| Script | Produced | Current equivalent |
| ------ | -------- | ------------------ |
| `run_meeting4_report.py` | Meeting 4: HHL/VQLS/QSVT comparison over the 1-D Poisson and HET cases. | `scripts/run_1d_benchmark.py` |
| `run_meeting5.py` | Meeting 5: discretisation versus quantum algorithmic error decomposition; first 2-D results. | `benchmark/hpc_plotting.py::plot_error_decomposition` |
| `run_verification_study.py` | Verification and validation study of HHL and VQLS on 1-D and 2-D Poisson for the HET application. | `tests/` for correctness; `scripts/run_hpc_*full.py` for the sweeps. |
| `debug_2d_solvers.py` | Exploratory 2-D debugging; established line-SOR over line-Jacobi and the O(N) versus O(N²) iteration counts. | `scripts/debug_outer_2d.py`; the SOR result is now `solvers/outer/stationary.py`. |
