# 2-D parameter studies — fourth-order operator

Installed 2026-08-20 from the CX3 arrival `results/updated_studies2D/2Dstudies/`
(since deleted; a copy of the arrival is retained in the session scratchpad).

| file | provenance |
|---|---|
| `sensitivity_qsvt.json` | Job 3831762.pbs-7, `DIM=2 SOLVERS=qsvt ORDER=4 RUN_TAG=o4`, started 2026-08-19 20:46:58 at commit a6b6e03. Two sweeps (one per case), `max_degree ∈ {5, 8, 11, 15, 21, 51, 201, 501}`, N=8. |
| `equal_accuracy.json` | Same job; the two QSVT entries only. |
| `run_metadata.json` | Same job, verbatim. |

The arriving directory also held `sensitivity_hhl.json`, `sensitivity_vqls.json`
and four further `equal_accuracy.json` entries, none of which were produced at
fourth order: every submission for a given dimension writes into the single
shared directory `results/{DIM}Dstudies` on the cluster, so unrelated jobs
accumulate there. Those records were second order and were routed to
`results/2Dstudies` (HHL) or discarded as pre-existing (VQLS) instead.

