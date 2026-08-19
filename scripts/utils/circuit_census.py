#!/usr/bin/env python3
"""
Measure transpiled circuit resources for every solver, and record them per sweep row.

Why this exists
---------------
`hpc/runners/run_1d.py` declares `circuit_depth_t`, `n_gates_total` and
`n_gates_2q` in its row schema and no code path ever populates them. The
consequence surfaces two layers away: `hpc/runners/make_tables.py` prefers a
measured two-qubit count wherever a sweep recorded one and falls back to a
heuristic otherwise, so with nothing recorded the hardware-feasibility table is
heuristic for HHL and VQLS at every resolution, and uses QSVT's *pre*-transpilation
depth as a stand-in for a gate count everywhere else.

The heuristic is not neutral. Its VQLS branch assumes a two-layer ansatz against
the six to fourteen layers the sweeps actually run, understating that circuit by
up to sevenfold — and VQLS being "within budget at every N" is precisely the
headline the table reports. A claim about hardware feasibility should rest on a
transpiled gate count.

This script supplies that count. It builds each solver's circuit at each
resolution, transpiles it to the Heron r2 native basis through
`core.resources.transpile_report`, and merges the result into the sweep summaries
as the three columns that were always meant to hold it. **No solve is performed.**
Circuit construction and transpilation are the entire cost, which is why this is
post-processing rather than another sweep.

What is measured, and what it means
-----------------------------------
Three quantities per (case, solver, N), all after transpilation to {rz, sx, x, cz}
at optimisation level 3, against an unconstrained coupling map:

  `circuit_depth_t`  Transpiled depth. Sets the execution time, and with the
                     device's T₂ decides whether the circuit finishes before the
                     state decoheres.
  `n_gates_total`    Total native gates.
  `n_gates_2q`       Two-qubit (cz) gates. **The figure that matters.** On
                     superconducting hardware two-qubit gates dominate the error
                     budget by roughly an order of magnitude, so a depth bound
                     that counts single-qubit rotations equally misstates the
                     constraint.

Routing is left unconstrained deliberately. A heavy-hex coupling map inflates the
count by inserting swaps, and that inflation is a property of the qubit
allocation rather than of the algorithm; the unconstrained count is the algorithm's
own cost and is a lower bound on any physical realisation. A circuit already over
budget unconstrained is over budget on hardware, which is the direction every
conclusion in this work runs.

Cost and scope
--------------
QSVT at a production degree does not transpile directly in reasonable time, which
is what `core.resources.qsvt_resource_estimate` exists for: it transpiles a single
block-encoding application and scales by the degree, an upper bound validated
against direct transpilation to within 0.4 % at N = 16. That composed route is
used for QSVT; HHL and VQLS are transpiled directly.

Usage
-----
    python scripts/utils/circuit_census.py --dim 1 --order 2 --dry-run
    python scripts/utils/circuit_census.py --dim 1 --order 2
    python scripts/utils/circuit_census.py --dim 1 --order 2 --n-values 4,8,16

`--dry-run` prints the table and writes nothing. Without it a timestamped backup
of `results_full.json` is written before the columns are merged in, and
`results_summary.csv` is regenerated. Rows are matched on (case, solver, N);
nothing else in the row is touched, and no `.npz` is read or written.

References
----------
  Preskill, J. (2018). Quantum Computing in the NISQ era and beyond.
      Quantum, 2, 79.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import shutil
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.resources import (HERON_R2_BASIS_GATES,                    # noqa: E402
                            qsvt_resource_estimate, transpile_report)

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("census")

# Qiskit emits one INFO record per transpiler pass. At optimisation level 3 on a
# QPE circuit that is thousands of lines per measurement, burying the table this
# script exists to print. Silenced exactly as `hpc/runners/run_1d.py` does.
for _noisy in ("qiskit.transpiler", "qiskit.transpiler.passes",
               "qiskit.transpiler.runningpassmanager", "qiskit.passmanager",
               "qiskit_ibm_runtime", "qiskit.compiler.transpiler"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

SWEEP_DIR: dict[tuple[int, int], str] = {
    (1, 2): "results/1Dhpc_run",   (1, 4): "results/1Dhpc_run_4th",
    (2, 2): "results/2Dhpc_run",   (2, 4): "results/2Dhpc_run_4th",
    (3, 2): "results/3Dhpc_run",   (3, 4): "results/3Dhpc_run_4th",
}

MEASURED_FIELDS: tuple[str, ...] = (
    "circuit_depth_t", "n_gates_total", "n_gates_2q",
)

# HPC case identifier -> `core.cases` registry name, mirroring the section
# functions of `hpc/runners/run_2d.py` and `hpc/runners/run_3d.py`. The census
# must build each case through the same registry the sweep used, since the strip
# operator depends on the domain's aspect ratio: the unit square and the SPT-100
# channel give materially different spectra at equal N.
CASE_REGISTRY_NAME: dict[str, str] = {
    # 2-D, hpc/runners/run_2d.py:run_section1..5
    "2D_Poisson_sin_hom":               "poisson_2d_sin_pi",
    "2D_Poisson_TwoGaussian_PlasmaNet": "poisson_2d_two_gaussian_plasmanet",
    "2D_Poisson_SingleMode_n1m1":       "poisson_2d_single_mode_n1m1",
    "2D_HET_MMS_SPT100":                "het_2d_mms_spt100",
    "2D_HET_Sin_MeetingReport":         "het_2d_sin_meeting_report",
    # 3-D, hpc/runners/run_3d.py:case_cube..case_high_mode
    "3D_Poisson_TripleSin_cube":        "poisson_3d_triple_sin_cube",
    "3D_HET_MMS_SPT100":                "het_3d_mms_spt100",
    "3D_HET_RotatingSpoke_SPT100":      "het_3d_rotating_spoke",
    "3D_HET_Discharge_SPT100":          "het_3d_discharge_spt100",
    "3D_Laplace_BCdriven_cube":         "poisson_3d_laplace_bc_driven",
    "3D_Poisson_TwoGaussian_cube":      "poisson_3d_two_gaussian_cube",
    "3D_Poisson_HighMode_n2m3l4":       "poisson_3d_high_mode_n2m3l4",
}


# ── Sweep Operators ──────────────────────────────────────────────────────────

def sweep_operator(dim: int, order: int, case_id: str, N: int) -> np.ndarray:
    """
    Build the exact linear operator a given sweep row's quantum solver received.

    The census measures a circuit, and a circuit is determined by the operator it
    encodes. Measuring the wrong operator reports resources for a circuit the
    sweep never built, which is why this function reconstructs each case through
    `core.cases` — the same registry `run_2d.py` and `run_3d.py` use — rather
    than approximating with a generic matrix.

    What the operator is, per sweep
    -------------------------------
    1-D    The full system matrix. Order 2 is the Toeplitz symmetric tridiagonal
           operator of `problems/poisson_1d.py`, κ = O(N²); order 4 is the
           pentadiagonal operator of `problems/poisson_1d_4th.py`, better
           conditioned at equal N by an asymptotic factor 4/3.

    2-D/3-D
           The **strip** operator, not the full N² or N³ system: `solvers/outer`
           decomposes the domain into 1-D strips and hands each to a 1-D quantum
           solver, so the circuit width is log₂(N) and not log₂(N²). The
           transverse coupling contributes a diagonal shift that bounds the
           condition number near 3 in 2-D and 2 in 3-D, which is precisely why
           the quantum solvers remain tractable there and why the 1-D operator
           must not be substituted.

    Which strip, at fourth order
    ----------------------------
    The odd reflection at a transverse boundary folds the ghost node onto the
    strip's own diagonal, so a fourth-order sweep requests two distinct strip
    operators in 2-D and up to four in 3-D. This function returns the **interior**
    operator, `row_matrix()` — the one the overwhelming majority of strips use and
    the one `kappa_row` reports, so the measurement is consistent with the κ
    recorded beside it in the row. The boundary-adjacent families differ only by a
    diagonal shift and their κ by under 2 %, so their circuits differ negligibly;
    the interior figure is representative rather than exhaustive.

    Parameters
    ----------
    dim : {1, 2, 3}
        Spatial dimension of the sweep.
    order : {2, 4}
        Spatial discretisation order.
    case_id : str
        HPC case identifier as recorded in `results_full.json`. Ignored when
        `dim == 1`, whose operator carries no case dependence.
    N : int
        Resolution; the returned operator is (N, N).

    Returns
    -------
    np.ndarray
        (N, N) operator, symmetric and real.

    Raises
    ------
    KeyError
        If `case_id` is absent from `CASE_REGISTRY_NAME`.
    """
    if dim == 1:
        if order == 4:
            from problems.poisson_1d_4th import PoissonProblem1D4th
            return PoissonProblem1D4th(N=N, f_vals=np.zeros(N),
                                       alpha=0.0, beta=0.0).A
        from problems.poisson_1d import build_tst_matrix
        return build_tst_matrix(N)

    from core import cases

    if case_id not in CASE_REGISTRY_NAME:
        raise KeyError(
            f"No registry name for case {case_id!r}. Add it to "
            f"CASE_REGISTRY_NAME, taking the mapping from the section function "
            f"of hpc/runners/run_{dim}d.py that emits this identifier.")

    built = cases.get(CASE_REGISTRY_NAME[case_id]).build(N)
    problem = built.problem

    if order == 4:
        # Re-discretised exactly as the runners do, so that the operator carries
        # the same boundary closure the sweep solved against.
        if dim == 2:
            from hpc.runners.run_2d import _to_4th_order_2d
            problem = _to_4th_order_2d(problem, built.f_faces)
        else:
            from hpc.runners.run_3d import _to_4th_order_3d
            problem = _to_4th_order_3d(problem, built.f_faces)

    return np.asarray(problem.row_matrix(), dtype=float)


# ── Circuit Construction ───────────────────────────────────────────────────────

def _hhl_circuit(A: np.ndarray, b: np.ndarray, epsilon: float):
    """
    Build the HHL circuit for one operator without simulating it.

    Mirrors `solvers.quantum.hhl_1d.hhl_solve_system` exactly through the
    construction stages — the same spectral normalisation, the same Toeplitz
    detection, the same `HHL(epsilon=…)` — and stops at `construct_circuit`
    rather than proceeding to `solve`. Any divergence here would measure a
    circuit the benchmark never ran.

    Parameters
    ----------
    A : np.ndarray, shape (N, N)
        Hermitian system matrix.
    b : np.ndarray, shape (N,)
        Right-hand side.
    epsilon : float
        Overall algorithm precision.

    Returns
    -------
    QuantumCircuit
        The unexecuted HHL circuit.
    """
    from quantum_linear_solvers.linear_solvers.hhl import HHL
    from quantum_linear_solvers.linear_solvers.matrices.numpy_matrix import (
        NumPyMatrix)
    from quantum_linear_solvers.linear_solvers.matrices.tridiagonal_toeplitz import (
        TridiagonalToeplitz)

    from solvers.quantum.block_encoding import is_toeplitz_tridiagonal

    n_qubits = int(np.log2(len(b)))
    alpha = float(np.linalg.norm(A, ord=2))
    b_norm = b / float(np.linalg.norm(b))

    if is_toeplitz_tridiagonal(A):
        matrix = TridiagonalToeplitz(
            num_state_qubits=n_qubits,
            main_diag=A[0, 0] / alpha,
            off_diag=A[0, 1] / alpha,
            trotter_steps=1,
        )
    else:
        matrix = NumPyMatrix(A / alpha, tolerance=epsilon)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return HHL(epsilon=epsilon).construct_circuit(matrix, b_norm)


def _vqls_circuit(N: int, n_layers: int):
    """
    Build the VQLS ansatz at the layer count the sweeps actually use.

    The layer count is the whole point of measuring rather than estimating: the
    heuristic in `benchmark/hardware.py` assumes two layers, while
    `hpc/runners/run_1d.py` runs ``max(6, 2·n_qubits + 2)``, which is 6 at N = 4
    and 14 at N = 64.

    Parameters
    ----------
    N : int
        Problem size.
    n_layers : int
        Entangling layers.

    Returns
    -------
    QuantumCircuit
        The ansatz with its parameters bound to arbitrary values, since gate
        counts do not depend on rotation angles.
    """
    from qiskit.circuit.library import RealAmplitudes

    n_qubits = int(np.log2(N))
    ansatz = RealAmplitudes(n_qubits, reps=n_layers, entanglement="linear")
    rng = np.random.default_rng(0)
    return ansatz.assign_parameters(
        rng.uniform(0.0, 2.0 * np.pi, ansatz.num_parameters))


def _vqls_layers(N: int) -> int:
    """
    The ansatz depth the 1-D sweep runs at this resolution.

    Restated from `hpc/runners/run_1d.py::_run_vqls` rather than imported,
    because importing that module pulls in the whole runner. Kept adjacent to the
    reason it matters so that a change there is visible here.

    Parameters
    ----------
    N : int
        Problem size.

    Returns
    -------
    int
        Entangling layer count.
    """
    return max(6, 2 * int(np.log2(N)) + 2)


# ── Measurement ────────────────────────────────────────────────────────────────

# Order in which solvers are measured at a given resolution, cheapest first.
#
# This is not cosmetic. A direct transpilation of the HHL circuit is by far the
# dearest measurement here — its QPE clock register grows with kappa, so the
# circuit roughly quadruples in depth per refinement, reaching 752 532 gates at
# N = 32 — while VQLS is an ansatz of a few dozen gates and QSVT is composed
# arithmetically rather than transpiled at full degree. Measuring alphabetically
# put HHL first, and at N = 64 that single measurement consumed the whole run and
# returned nothing at all, including for the two solvers that would have finished
# in seconds.
SOLVER_COST_RANK: dict[str, int] = {"VQLS": 0, "QSVT": 1, "HHL": 2}


def _measure_worker(solver, A, b, N, kappa, degree, epsilon, q) -> None:
    """
    Run one measurement in a child process and return it through `q`.

    Parameters
    ----------
    q : multiprocessing.Queue
        Channel for the result mapping.
    """
    try:
        q.put(_measure_inproc(solver, A, b, N, kappa, degree, epsilon))
    except Exception as exc:                       # noqa: BLE001
        q.put({"error": f"{type(exc).__name__}: {exc}"})


def measure(solver: str, A: np.ndarray, b: np.ndarray, N: int,
            kappa: float, degree: Optional[int],
            epsilon: float = 0.01,
            timeout_s: Optional[float] = None) -> Optional[dict]:
    """
    Transpile one solver's circuit and report its resource footprint.

    With `timeout_s` set the work runs in a child process that is terminated when
    the budget expires, so a circuit too large to transpile cannot consume the
    whole run. That is a real outcome rather than a hypothetical: the HHL circuit
    at N = 64 did not complete, and under the previous arrangement it took the
    QSVT and VQLS measurements at that resolution down with it.

    A measurement that does not complete returns None. The caller leaves the row's
    columns unpopulated, and `hpc/runners/make_tables.py` then falls back to its
    heuristic and labels the row `estimated` — which is the honest outcome, and
    visibly different from a measurement.

    Parameters
    ----------
    solver : str
        'HHL', 'VQLS' or 'QSVT'. Thomas is classical and has no circuit.
    A : np.ndarray, shape (N, N)
        System matrix.
    b : np.ndarray, shape (N,)
        Right-hand side.
    N : int
        Problem size.
    kappa : float
        Condition number, required by the QSVT composed estimate.
    degree : int or None
        QSVT polynomial degree as recorded by the sweep. Required for QSVT;
        ignored otherwise.
    epsilon : float
        HHL precision parameter.
    timeout_s : float or None
        Wall-clock budget for this one measurement. None runs it in process,
        without a bound.

    Returns
    -------
    dict or None
        Mapping of `MEASURED_FIELDS` plus `source`, or None where the circuit
        could not be built or the budget expired.
    """
    if timeout_s is None:
        return _measure_inproc(solver, A, b, N, kappa, degree, epsilon)

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_measure_worker,
                    args=(solver, A, b, N, kappa, degree, epsilon, q))
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        p.terminate()
        p.join()
        log.warning("      %s at N=%d exceeded the %.0f s measurement budget; "
                    "left unmeasured (the table will fall back to its heuristic "
                    "and label the row 'estimated')", solver, N, timeout_s)
        return None

    try:
        got = q.get_nowait()
    except Exception:                              # noqa: BLE001
        log.warning("      %s at N=%d returned nothing", solver, N)
        return None

    if got is None or "error" in got:
        log.warning("      %s at N=%d could not be measured: %s",
                    solver, N, (got or {}).get("error", "unknown"))
        return None
    return got


def _measure_inproc(solver: str, A: np.ndarray, b: np.ndarray, N: int,
                    kappa: float, degree: Optional[int],
                    epsilon: float = 0.01) -> Optional[dict]:
    """
    Transpile one solver's circuit and report its resource footprint.

    Parameters
    ----------
    solver : str
        'HHL', 'VQLS' or 'QSVT'. Thomas is classical and has no circuit.
    A : np.ndarray, shape (N, N)
        System matrix.
    b : np.ndarray, shape (N,)
        Right-hand side.
    N : int
        Problem size.
    kappa : float
        Condition number, required by the QSVT composed estimate.
    degree : int or None
        QSVT polynomial degree as recorded by the sweep. Required for QSVT;
        ignored otherwise.
    epsilon : float
        HHL precision parameter.

    Returns
    -------
    dict or None
        Mapping of `MEASURED_FIELDS`, plus `source` naming how it was obtained.
        None where the circuit could not be built, which is reported rather than
        silently omitted.
    """
    try:
        if solver == "HHL":
            report = transpile_report(_hhl_circuit(A, b, epsilon),
                                      basis_gates=HERON_R2_BASIS_GATES)
            return {"circuit_depth_t": report.post_depth,
                    "n_gates_total": sum(report.gate_counts.values()),
                    "n_gates_2q": report.two_qubit_count,
                    "source": "transpiled"}

        if solver == "VQLS":
            report = transpile_report(_vqls_circuit(N, _vqls_layers(N)),
                                      basis_gates=HERON_R2_BASIS_GATES)
            return {"circuit_depth_t": report.post_depth,
                    "n_gates_total": sum(report.gate_counts.values()),
                    "n_gates_2q": report.two_qubit_count,
                    "source": "transpiled"}

        if solver == "QSVT":
            if not degree:
                return None
            # Composed rather than directly transpiled: a production-degree QSVT
            # circuit does not transpile in reasonable time. The composition is a
            # validated upper bound -- see core/resources.py.
            est = qsvt_resource_estimate(N=N, kappa=kappa, degree=int(degree))
            # Depth is composed on the same basis as the gate count: `degree`
            # applications of the unit block encoding, preceded by one state
            # preparation. Reported as an upper bound for the same reason.
            depth_t = (est.unit_cost.post_depth * int(degree)
                       + est.prep_cost.post_depth)
            return {"circuit_depth_t": int(depth_t),
                    "n_gates_total": None,
                    "n_gates_2q": int(est.total_two_qubit_count),
                    "source": "composed (degree x unit block encoding)"}
    except Exception as exc:                       # noqa: BLE001
        log.warning("      %s at N=%d could not be measured: %s",
                    solver, N, exc)
        return None
    return None


# ── Driver ─────────────────────────────────────────────────────────────────────

def _write_csv(json_path: Path, csv_path: Path) -> None:
    """Regenerate the flat CSV from the JSON, preserving every column."""
    rows = json.loads(json_path.read_text(encoding="utf-8"))
    if not rows:
        return
    fieldnames = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _merge_and_persist(rows: list[dict], measured: dict[tuple[str, str, int], dict],
                       sweep: Path, json_path: Path,
                       backup_made: list[bool],
                       op_cache: dict, op_key_fn) -> int:
    """
    Merge the measurements collected so far into the summary and write it out.

    Called after every completed measurement rather than once at the end. A
    direct transpilation of the HHL circuit costs minutes at N = 32 and
    considerably more at N = 64 — the QPE clock register grows with kappa, so the
    circuit roughly quadruples in depth per refinement — and a run interrupted
    part-way through would otherwise discard every measurement that had already
    succeeded. This mirrors the sweeps' own incremental-write behaviour, and for
    the same reason.

    The backup is taken once, before the first write, so that re-entering this
    function does not overwrite the pre-census state with a partially updated
    copy.

    Parameters
    ----------
    rows : list of dict
        Summary rows, updated in place.
    measured : dict
        Measurements keyed by (solver, N).
    sweep : Path
        Sweep directory, for the regenerated CSV.
    json_path : Path
        Summary path.
    backup_made : list of bool
        Single-element mutable flag recording whether the backup has been taken.
    op_cache : dict
        {(case_id, N): operator}, so that a row is matched to the measurement of
        the operator it actually presents. In 2-D and 3-D two cases at equal
        (solver, N) may carry different strip operators, and keying the merge on
        (solver, N) alone would assign one case the other's circuit.
    op_key_fn : callable
        Maps an operator to the digest under which its measurement is filed.

    Returns
    -------
    int
        Number of rows carrying measurements after the merge.
    """
    changed = 0
    for row in rows:
        A = op_cache.get((str(row.get("case")), row.get("N")))
        if A is None:
            continue
        key = (op_key_fn(A), str(row.get("solver")), row.get("N"))
        if key not in measured:
            continue
        got = measured[key]
        for field in MEASURED_FIELDS:
            row[field] = got.get(field)
        changed += 1

    if not backup_made[0]:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        shutil.copy2(json_path,
                     json_path.with_name(
                         f"results_full.{stamp}.pre-census.json"))
        backup_made[0] = True

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_csv(json_path, sweep / "results_summary.csv")
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dim", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--order", type=int, choices=(2, 4), default=2)
    ap.add_argument("--n-values", default=None,
                    help="Comma-separated resolutions; default is every N in "
                         "the sweep.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report without writing.")
    ap.add_argument("--timeout-s", type=float, default=None,
                    help="Wall-clock budget per measurement. A circuit that "
                         "exceeds it is left unmeasured and the table falls "
                         "back to its heuristic for that row, labelled "
                         "'estimated'. Unset runs each measurement unbounded.")
    args = ap.parse_args()

    # Every (dimension, order) is supported. The operator is reconstructed per
    # sweep by `sweep_operator`, which builds each case through `core.cases` —
    # the registry the runners themselves use — so the measured circuit is the
    # circuit the sweep built. Substituting a generic matrix would report
    # resources for a circuit that never existed, silently and with the authority
    # of the word "measured"; that is why this script previously refused
    # everything but the 1-D second-order sweep rather than approximating.
    sweep = REPO_ROOT / SWEEP_DIR[(args.dim, args.order)]
    json_path = sweep / "results_full.json"
    if not json_path.exists():
        log.error("No sweep summary at %s", json_path)
        return 1

    rows = json.loads(json_path.read_text(encoding="utf-8"))
    wanted = ({int(v) for v in args.n_values.split(",")}
              if args.n_values else None)

    log.info("=" * 78)
    log.info("  CIRCUIT CENSUS  -  %d-D  order %d%s",
             args.dim, args.order, "  (dry run)" if args.dry_run else "")
    log.info("=" * 78)
    log.info("  %-34s %-7s %5s %10s %12s %12s  %s",
             "case", "solver", "N", "depth_t", "gates", "2q gates", "source")

    # One measurement per (operator, solver, N). The circuit depends on the
    # operator's size and spectrum and not on the source term, so in 1-D a single
    # measurement serves every case at that (solver, N). In 2-D and 3-D that no
    # longer holds: the unit square and the SPT-100 channel have different aspect
    # ratios and hence different strip spectra, so cases are grouped by the
    # operator they actually present. Grouping is by the operator's own bytes
    # rather than by a hand-written domain table, so two cases sharing a geometry
    # are measured once and no case is ever assigned another's circuit by
    # assumption.
    def _op_key(A: np.ndarray) -> str:
        return hashlib.sha256(np.ascontiguousarray(A, dtype=float)
                              .tobytes()).hexdigest()[:16]

    todo: list[tuple[str, str, int, np.ndarray]] = []
    seen: set[tuple[str, str, int]] = set()
    op_cache: dict[tuple[str, int], np.ndarray] = {}
    for row in rows:
        solver, N = str(row.get("solver")), row.get("N")
        case_id = str(row.get("case"))
        if solver == "Thomas" or N is None:
            continue
        if wanted is not None and N not in wanted:
            continue
        if (case_id, N) not in op_cache:
            try:
                op_cache[(case_id, N)] = sweep_operator(args.dim, args.order,
                                                        case_id, N)
            except Exception as exc:
                log.warning("  %-34s N=%-5d operator unavailable (%s); skipped",
                            case_id, N, exc)
                op_cache[(case_id, N)] = None
        A = op_cache[(case_id, N)]
        if A is None:
            continue
        key = (_op_key(A), solver, N)
        if key in seen:
            continue
        seen.add(key)
        todo.append((key[0], solver, N, A))
    # Ascending in N, and within each N cheapest solver first, so that an
    # interrupted run has banked every measurement it could afford rather than
    # having spent its whole budget on the single dearest one.
    todo.sort(key=lambda t: (t[2], SOLVER_COST_RANK.get(t[1], 99), t[1]))

    measured: dict[tuple[str, str, int], dict] = {}
    backup_made = [False]
    changed = 0
    for op_key, solver, N, A in todo:
        # The representative row for this (operator, solver, N): any case whose
        # operator hashes to `op_key` will do, since they share the circuit. The
        # None guard must precede the hash, or a case whose operator could not be
        # built would be hashed.
        row = next(r for r in rows
                   if str(r.get("solver")) == solver and r.get("N") == N
                   and op_cache.get((str(r.get("case")), N)) is not None
                   and _op_key(op_cache[(str(r.get("case")), N)]) == op_key)
        b = np.ones(N) / np.sqrt(N)
        kappa = float(row.get("kappa") or row.get("kappa_row") or 1.0)
        got = measure(solver, A, b, N, kappa, row.get("qsvt_degree"),
                      timeout_s=args.timeout_s)
        if got is None:
            continue
        measured[(op_key, solver, N)] = got
        log.info("  %-34s %-7s %5d %10s %12s %12s  %s",
                 f"op {op_key[:8]} (kappa={kappa:.4f})", solver, N,
                 got["circuit_depth_t"] if got["circuit_depth_t"] else "---",
                 got["n_gates_total"] if got["n_gates_total"] else "---",
                 f"{got['n_gates_2q']:,}", got["source"])

        # Persisted immediately. A direct transpilation costs minutes at N = 32
        # and much longer beyond it, so a run cut short must not discard the
        # measurements already in hand.
        if not args.dry_run:
            changed = _merge_and_persist(rows, measured, sweep, json_path,
                                         backup_made, op_cache, _op_key)

    if not measured:
        log.error("  Nothing measured; nothing written.")
        return 1

    log.info("-" * 78)
    if args.dry_run:
        log.info("  %d (solver, N) combination(s) measured; nothing written",
                 len(measured))
    else:
        log.info("  %d row(s) carry measured circuit metrics across %d "
                 "(solver, N) combination(s)", changed, len(measured))
    log.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
