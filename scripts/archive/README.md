# Archived scripts

One-off scripts retained for provenance, so that results reported during the
project remain attributable. They are not maintained and are not expected to
run: all but `debug_2d_solvers.py` and the six added in the Phase 3 scripts
consolidation import the legacy 2-D solver stack at module level, and that
stack was retired once `solvers/outer` was shown to reproduce its results
exactly.

| Script | Produced | Current equivalent |
| ------ | -------- | ------------------ |
| `run_meeting4_report.py` | Meeting 4: HHL/VQLS/QSVT comparison over the 1-D Poisson and HET cases. | `scripts/run_1d_benchmark.py` (this file, below) |
| `run_meeting5.py` | Meeting 5: discretisation versus quantum algorithmic error decomposition; first 2-D results. | `benchmark/hpc_plotting.py::plot_error_decomposition`; structure reused by `scripts/example_report.py` |
| `run_verification_study.py` | Verification and validation study of HHL and VQLS on 1-D and 2-D Poisson for the HET application. | `tests/` for correctness; `hpc/` sweeps for the campaigns. |
| `debug_2d_solvers.py` | Exploratory 2-D debugging; established line-SOR over line-Jacobi and the O(N) versus O(N²) iteration counts. | `scripts/debug_2d.py`; the SOR result is now `solvers/outer/stationary.py`. |
| `run_1d_benchmark.py` | Sweeps A/B/C/D of the generic 1-D Poisson benchmark (`benchmark/runner.py`). | `scripts/debug_1d.py` for interactive exploration; `benchmark/runner.py`'s sweep functions still exist for programmatic use. |
| `run_2d_benchmark.py` | Sweeps E/F/G of the generic 2-D Poisson benchmark. Never completed a full run against the legacy 2-D stack - see the "known bugs" note below. | `scripts/debug_2d.py`; `benchmark/runner.py`'s sweep functions. |
| `run_het_benchmark.py` | Sweeps H1-H4, the 1-D HET plasma HHL/VQLS benchmark. | `scripts/debug_1d.py --case het_1d_*`; case definitions now live in `core/cases.py`. |
| `run_het_plasma_benchmark.py` | 1-D HET plasma solver comparison with Boeuf-Garrigues thresholds and physical E-field reporting. | `scripts/debug_1d.py`; the case is `het_1d_gaussian_Vd300_scaled` and its relatives in `core/cases.py`. |
| `run_het_2d_benchmark.py` | 2-D HET Case B: Boeuf-Garrigues charge density on the axial-radial channel. Its docstring advertised QSVT, which its `SOLVERS` dict never actually included. | `scripts/debug_2d.py --case het_2d_boeuf_garrigues`, which does run QSVT. |
| `debug_2d_4th.py` | Interactive 2-D fourth-order diagnostics, built on the *mixed-order* scheme: fourth order along the strip, second order transverse. That design is capped at order 2 by construction, and its boundary closure was separately defective — an even reflection applied to Dirichlet data, and `18α` where the row-0 stencil gives `14α`, measured convergence order 0.88. Archived rather than deleted because `hpc/runners/run_2d.py` imported `jacobi_2d_4th` from it, so it is the provenance of every 4th-order 2-D row produced before 2026-08-12. | `scripts/debug_2d.py` against a `problems.poisson_line_2d_4th.PoissonLine2D4th`, which is fourth order in both directions and verified to 3.87. |
| `debug_3d_4th.py` | The 3-D counterpart of the above, same mixed-order design and same closure. | `scripts/debug_3d.py` against `problems.poisson_line_3d_4th.PoissonLine3D4th`. |
| `run_qsvt_debug.py` | Standalone QSVT diagnostic runner (generic Poisson + HET 1-D, N=4/8/16), used to isolate the proportionality-recovery failure documented in `solvers/quantum/qsvt_1d.py::_qsvt_recovery_diagnostics`. Its docstring claimed a `results/qsvt_debug/` figure and CSV summary that the script never wrote. | `scripts/debug_1d.py --dump`, which prints the same node-by-node diagnostics and does not claim outputs it doesn't produce. |

Not archived, but worth noting here: `debug_outer_2d.py`, `debug_outer_3d.py`
and `explore.py` were **renamed**, not retired - they are
`scripts/debug_2d.py`, `scripts/debug_3d.py` and `scripts/tutorial.py`
respectively, with their case definitions moved to `core/cases.py` and their
shared machinery moved to `benchmark/diagnostics.py`. Git history is the
record of the rename; there was no reason to keep the old names around.
