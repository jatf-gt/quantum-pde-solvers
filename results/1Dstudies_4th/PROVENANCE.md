# 1-D parameter studies — fourth-order operator

Job **3894681.pbs-7**, `DIM=1 ORDER=4`, started 2026-08-23 18:59:14. Installed
2026-08-23. All three solvers, `study=both`, N = 8, three cases
(`poisson_1d_fS_hom`, `poisson_1d_fH_hom`, `het_1d_3a_linear`), κ = 42.14. The
job wrote directly into this directory: it is the first submission under the
per-(dimension, order, run tag) naming, and `hpc/runners/make_tables.py` already
resolved `results/1Dstudies_4th` for `--order 4`.

`config.git_dirty` records **true**, and `config.git_commit` is `e1a6b56`, a
commit present on neither the laptop nor the remote. The cluster repository
carries an unpushed local commit; push it if the numbers are to be reproducible
from history.

## HHL has no accuracy knob at fourth order — and this is not the ε bug

The ε sweep is **bit-exactly flat**: one distinct residual across
ε ∈ {0.1, 0.05, 0.01, 0.005, 0.001} on every case —
7.082799476314498 × 10⁻¹⁴, 1.1979482890346973 × 10⁻³ and
2.259610736127598 × 10⁻³ respectively. The `trotter_steps` sweep alongside it
holds **no records at all**: every one of its six values was refused.

Both follow from one fact. A pentadiagonal operator is not Toeplitz
tridiagonal, so `solvers/quantum/hhl_1d.py` simulates it with `NumPyMatrix`,
which forms the matrix exponential exactly through `scipy.linalg.expm`. There is
no Trotter decomposition, so a step count is meaningless and is **rejected with
a `ValueError` rather than accepted and ignored** — which is why that sweep is
empty — and there is no Trotter error for ε to trade against, which is why the ε
sweep does not move. What remains sets the floor: QPE register resolution and
the post-selection.

**This is a different phenomenon from the pre-2026-08-17 defect** in which ε
never reached the solver, recorded in the project memory. The two are told apart
by one field: that defect recorded `hhl_trotter_steps = ⌈1/ε⌉` — 10, 20, 100,
200 — whereas every record here carries `hhl_trotter_steps = None`, the honest
report of a branch that performs no Trotterisation. Do not "repair" these rows.

The consequence is quotable in its own right: at fourth order in one dimension,
HHL's accuracy is **not tunable**. Its single available configuration is also
its best, which is why the equal-accuracy table reports it reaching the 10⁻³
band at ε = 0.1 — the loosest setting on the grid.

It is also more accurate than the second-order solver at the same N, exactly
because the Trotter error is gone: residual 1.20 × 10⁻³ against 1.80 × 10⁻³ on
`fH_hom`, and 7.08 × 10⁻¹⁴ against 1.83 × 10⁻² on `fS_hom`. The cost of that is
paid in circuit width and depth, not in this table — the exact exponential is a
far denser evolution circuit, which is what the census measures as roughly
twentyfold growth per doubling of N.

## The QSVT degree threshold holds at fourth order

Natural degree 6575 at κ = 42.14. Sweeping the cap over
{20, 50, 100, 200, 500, 1000, 2000, uncapped}:

| case | error at d/κ = 4.77 | at 11.89 | at 23.76 |
|---|---|---|---|
| `1D_Poisson_fH_hom` | 5.85 × 10⁻¹ % | 1.77 × 10⁻⁴ % | 1.64 × 10⁻⁹ % |
| `HET_1D_3a_linear_hom` | 3.10 × 10⁻¹ % | 1.59 × 10⁻³ % | 8.53 × 10⁻¹⁰ % |

The knee sits at d/κ ≈ 12, recovering on a pentadiagonal operator the threshold
near 11 that figure F3 locates on the tridiagonal one.

`1D_Poisson_fS_hom` is at machine precision at **every** degree, including
d/κ = 0.50. Its right-hand side is an eigenvector of the operator, so the
solution is reached without the polynomial having to approximate 1/x across the
spectrum. This is the same reason the thesis takes `fS_nonhom` rather than
`fS_hom` as its primary 1-D case; do not read this row as a QSVT result.

## Scope

No `trotter_steps` axis exists at this order, so the question of whether HHL's
Trotter turning point moves with κ — open between the 1-D order-2 result (a peak
at four steps) and the 2-D one (no turning point up to sixteen) — cannot be
addressed here. It needs an order-2 sweep across N, where κ runs 29.3 → 1708.7.
