#!/usr/bin/env python3
"""
Post-processing plots for a run_hpc_3Dfull.py sweep.

Reads results/3Dhpc_run/ (results_full.json plus the archived
solution3d_{case}_{solver}_N{N}.npz files) and produces PNGs into
results/3Dhpc_run/plots/.  Nothing here re-runs a solve; it only reads what
the runner already wrote.

How a 3-D scalar field is actually shown
-----------------------------------------
A 3-D field cannot be inspected directly the way a 2-D heatmap can.  The
standard approach in CFD/PIC post-processing (ParaView, VisIt, and the
literature this project draws on) is orthogonal cutplanes: fix one
coordinate at a representative value and plot the remaining two as an
ordinary 2-D heatmap.  Three such planes, through the midpoint of each axis,
are enough to catch almost anything a 3-D bug would produce - a sign error,
a misplaced boundary, a solver that only diverges near one face - because a
defect visible in the full field is with overwhelming probability visible in
at least one axis-aligned slice through it.  This is what ``plot_slices``
does, and it is the primary verification plot in this script.

For the HET geometry specifically, the axial-azimuthal slice (axis 0-2, at
mid-radius) is not an arbitrary choice of cutplane: it is *the* view used in
the literature to show a rotating spoke (an "unrolled" z-theta map - see
McDonald & Gallimore 2011, Sekerak et al. 2015), so it is labelled and
treated as the primary slice for the HET cases.  A bonus polar rendering
(``plot_polar_unwrap``) recasts the same slice onto an annulus for a more
immediately recognisable "plan view" of the channel; because the true inner
radius is not part of the archived data (only the channel width is), it
uses a relative radial coordinate and is explicitly labelled as schematic.

Plots produced
---------------
1. Orthogonal slices    exact/Thomas/solver fields + signed error, one PNG
                        per cutplane per (case, N).  The main plot.
2. Polar spoke view      HET cases only: axial-azimuthal slice on an annulus.
3. 3-D cutaway           single oriented view, three slice planes embedded
                        in one 3-D axes, for whichever field is available.
4. Convergence history   residual vs outer iteration, all solvers overlaid.
5. Accuracy vs N         log-log, with an O(h^2) reference slope.
6. Cost vs N             weighted strip-solve cost and wall time, log-log.
7. Quantum overhead      wall-clock ratio vs Thomas.
8. Error decomposition   algorithmic error (vs Thomas) vs discretisation
                        error (Thomas vs exact).
9. Azimuthal mode fidelity   spoke/HET-MMS cases only: does the solver
                        reproduce the manufactured azimuthal mode.

Usage
-----
    python scripts/plot_hpc_3Dfull_results.py
    python scripts/plot_hpc_3Dfull_results.py --case 3D_HET_RotatingSpoke_SPT100
    python scripts/plot_hpc_3Dfull_results.py --list

Author : Juan Antonio Trobajo Flecha
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results" / "3Dhpc_run"
PLOTS_DIR = RESULTS_DIR / "plots"

SOLVER_COLOUR = {"Thomas": "#444444", "HHL": "#d62728",
                 "VQLS": "#2ca02c", "QSVT": "#1f77b4"}
SOLVER_ORDER = ["Thomas", "HHL", "VQLS", "QSVT"]

# Axis labels per case family.  Anything not matching a HET prefix falls
# back to generic Cartesian labels.
AXIS_LABELS = {
    "het": ("axial z (mm)", "radial r (mm)", "azimuthal s (mm)"),
    "generic": ("x", "y", "z"),
}


# ============================================================================
#  Loading
# ============================================================================

def load_results() -> list[dict]:
    path = RESULTS_DIR / "results_full.json"
    if not path.exists():
        raise SystemExit(f"No results found at {path}. Run run_hpc_3Dfull.py first.")
    with open(path) as fh:
        return json.load(fh)


def load_solution(case: str, solver: str, N: int) -> dict | None:
    path = RESULTS_DIR / f"solution3d_{case}_{solver}_N{N}.npz"
    if not path.exists():
        return None
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def axis_labels_for(case: str, sol: dict) -> tuple[str, str, str]:
    if "HET" in case or bool(sol.get("periodic", (False,) * 3)[2]):
        z, r, s = AXIS_LABELS["het"]
        return z, r, s
    return AXIS_LABELS["generic"]


def _sort_key(s: str) -> tuple:
    return (SOLVER_ORDER.index(s) if s in SOLVER_ORDER else 99, s)


def group_by_case_N(rows: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["case"], r["N"]), []).append(r)
    for key in groups:
        groups[key].sort(key=lambda r: _sort_key(r["solver"]))
    return groups


def group_by_case_solver(rows: list[dict]) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["case"], r["solver"]), []).append(r)
    for key in groups:
        groups[key].sort(key=lambda r: r["N"])
    return groups


# ============================================================================
#  Plot 1 - orthogonal slices  (the primary plot)
# ============================================================================

_PLANES = [
    # (name, fixed_axis, remaining_axes, is_the_het_spoke_view)
    ("axial-radial",     2, (0, 1), False),   # fix s (or z2), show (axis0, axis1)
    ("axial-azimuthal",  1, (0, 2), True),    # fix r,          show (axis0, axis2)
    ("radial-azimuthal", 0, (1, 2), False),   # fix z,          show (axis1, axis2)
]


def _slice_at(arr: np.ndarray, fixed_axis: int, idx: int) -> np.ndarray:
    sl = [slice(None)] * 3
    sl[fixed_axis] = idx
    return arr[tuple(sl)]


def plot_slices(case: str, N: int, rows: list[dict], plt, TwoSlopeNorm) -> list[Path]:
    """One PNG per cutplane, each laid out like the 2-D fields plot."""
    sols, exact = {}, None
    for r in rows:
        sol = load_solution(case, r["solver"], N)
        if sol is None:
            continue
        sols[r["solver"]] = sol
        if exact is None and "phi_exact" in sol:
            exact = sol["phi_exact"]
    if not sols:
        return []

    any_sol = next(iter(sols.values()))
    coords = [any_sol["x0"], any_sol["x1"], any_sol["x2"]]
    labels = axis_labels_for(case, any_sol)
    shape = any_sol["phi"].shape
    ref = exact if exact is not None else sols.get("Thomas", any_sol)["phi"]
    ref_is_exact = exact is not None
    names = list(sols)

    out_paths = []
    for plane_name, fixed_ax, (a0, a1), is_spoke in _PLANES:
        idx = shape[fixed_ax] // 2
        # Coordinate grid for this plane: take the two varying axes' meshgrid
        # at the fixed slice (coords[a] is already a full 3-D meshgrid, so
        # slicing it the same way as the field gives the right 2-D grid).
        cx = _slice_at(coords[a0], fixed_ax, idx)
        cy = _slice_at(coords[a1], fixed_ax, idx)
        ref_slice = _slice_at(ref, fixed_ax, idx)

        n_cols = 1 + len(names)
        fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7), squeeze=False)
        vmin, vmax = float(ref.min()), float(ref.max())

        im = axes[0, 0].pcolormesh(cx, cy, ref_slice, cmap="RdBu_r",
                                   vmin=vmin, vmax=vmax, shading="auto")
        axes[0, 0].set_title("Exact" if ref_is_exact else "Thomas (reference)",
                             fontweight="bold")
        axes[0, 0].set_xlabel(labels[a0]); axes[0, 0].set_ylabel(labels[a1])
        plt.colorbar(im, ax=axes[0, 0], shrink=0.8)
        axes[1, 0].axis("off")

        for ci, name in enumerate(names, start=1):
            phi_slice = _slice_at(sols[name]["phi"], fixed_ax, idx)
            im = axes[0, ci].pcolormesh(cx, cy, phi_slice, cmap="RdBu_r",
                                        vmin=vmin, vmax=vmax, shading="auto")
            axes[0, ci].set_title(name, fontweight="bold",
                                  color=SOLVER_COLOUR.get(name, "black"))
            axes[0, ci].set_xlabel(labels[a0])
            plt.colorbar(im, ax=axes[0, ci], shrink=0.8)

            err = phi_slice - ref_slice
            abs_max = max(float(np.abs(err).max()), 1e-12)
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
            im2 = axes[1, ci].pcolormesh(cx, cy, err, cmap="seismic", norm=norm,
                                         shading="auto")
            pct = float(np.max(np.abs(phi_slice - ref_slice))
                       / (np.max(np.abs(ref)) + 1e-300) * 100.0)
            lab = "vs exact" if ref_is_exact else "vs Thomas"
            axes[1, ci].set_title(f"Error {lab} ({pct:.3f}%)")
            axes[1, ci].set_xlabel(labels[a0])
            plt.colorbar(im2, ax=axes[1, ci], shrink=0.8)

        tag = "  <- rotating-spoke view (z vs theta, unrolled)" if is_spoke else ""
        fig.suptitle(f"{case}  N={N}  |  {plane_name} slice{tag}", fontweight="bold")
        plt.tight_layout()
        safe_plane = plane_name.replace("-", "_")
        out = PLOTS_DIR / f"slice_{safe_plane}_{case}_N{N}.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out)
    return out_paths


# ============================================================================
#  Plot 2 - polar spoke view (HET, periodic azimuth only)
# ============================================================================

def plot_polar_unwrap(case: str, N: int, rows: list[dict], plt) -> Path | None:
    """
    Axial-azimuthal slice recast onto an annulus: the classic HET "plan
    view" used to show a rotating spoke as a coloured ring.

    The radial coordinate is RELATIVE, not the true channel radius: the
    archived data carries the channel width (Lr) but not the inner radius
    r_in, so the annulus inner/outer radii here are an arbitrary offset
    chosen only to make the ring legible.  Treat this as a schematic
    orientation aid; the rectangular unrolled view (plot_slices,
    axial-azimuthal plane) is the quantitatively meaningful one.
    """
    any_sol = None
    for r in rows:
        sol = load_solution(case, r["solver"], N)
        if sol is not None:
            any_sol = sol
            break
    if any_sol is None or not bool(any_sol.get("periodic", (False,) * 3)[2]):
        return None

    solver_name = next(r["solver"] for r in rows
                       if load_solution(case, r["solver"], N) is not None)
    sol = load_solution(case, solver_name, N)
    phi, r_coord, s_coord = sol["phi"], sol["x1"], sol["x2"]
    Lr = float(sol["lengths"][1])
    r_idx = phi.shape[0] // 2                     # mid-axial slice

    r_rel = r_coord[0, :, 0]                       # 1-D radial coordinate
    s_full = s_coord[0, 0, :]
    r_annulus = 3.0 * Lr + r_rel                    # arbitrary inner offset
    theta = 2.0 * np.pi * s_full / float(sol["lengths"][2])

    R, TH = np.meshgrid(r_annulus, theta, indexing="ij")
    X, Y = R * np.cos(TH), R * np.sin(TH)
    field = phi[r_idx, :, :]

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    im = ax.pcolormesh(X, Y, field, cmap="RdBu_r", shading="auto")
    ax.set_aspect("equal")
    ax.set_title(f"{case}  N={N}  {solver_name}\n"
                 f"plan view (schematic radius) - mid-axial slice")
    ax.set_xlabel("(schematic x)"); ax.set_ylabel("(schematic y)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="phi")
    plt.tight_layout()
    out = PLOTS_DIR / f"polar_spoke_{case}_N{N}_{solver_name}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
#  Plot 3 - 3-D cutaway orientation view
# ============================================================================

def plot_3d_cutaway(case: str, N: int, rows: list[dict], plt) -> Path | None:
    """
    Single 3-D view with the three orthogonal slice planes embedded at their
    true offsets, using matplotlib's contourf(..., zdir=..., offset=...).

    This is an orientation aid, not a quantitative tool: matplotlib's 3-D
    backend does not depth-sort filled patches reliably, so overlapping
    planes can occlude each other in a way that is not physically
    meaningful.  Use the orthogonal slice PNGs for anything that needs to be
    read precisely.
    """
    sol = None
    for r in rows:
        if r["solver"] == "Thomas":
            sol = load_solution(case, r["solver"], N)
            if sol is not None:
                break
    if sol is None:
        for r in rows:
            sol = load_solution(case, r["solver"], N)
            if sol is not None:
                break
    if sol is None:
        return None

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-D proj)

    phi = sol["phi"]
    x0, x1, x2 = sol["x0"][:, 0, 0], sol["x1"][0, :, 0], sol["x2"][0, 0, :]
    nx, ny, nz = phi.shape
    ix, iy, iz = nx // 2, ny // 2, nz // 2

    fig = plt.figure(figsize=(7, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    vmin, vmax = float(phi.min()), float(phi.max())
    cmap = plt.get_cmap("RdBu_r")

    Y, Z = np.meshgrid(x1, x2, indexing="ij")
    ax.contourf(Y, Z, phi[ix, :, :], zdir="x", offset=float(x0[ix]),
               levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    X, Z = np.meshgrid(x0, x2, indexing="ij")
    ax.contourf(X, phi[:, iy, :], Z, zdir="y", offset=float(x1[iy]),
               levels=20, cmap=cmap, vmin=vmin, vmax=vmax)
    X, Y = np.meshgrid(x0, x1, indexing="ij")
    ax.contourf(X, Y, phi[:, :, iz], zdir="z", offset=float(x2[iz]),
               levels=20, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlim(x0.min(), x0.max()); ax.set_ylim(x1.min(), x1.max())
    ax.set_zlim(x2.min(), x2.max())
    labels = axis_labels_for(case, sol)
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_zlabel(labels[2])
    ax.set_title(f"{case}  N={N}  ({sol.get('_solver_name', 'field')})\n"
                 "3-D cutaway (orientation only - see slice PNGs for values)")
    plt.tight_layout()
    out = PLOTS_DIR / f"cutaway3d_{case}_N{N}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
#  Plot 4 - convergence history
# ============================================================================

def plot_convergence(case: str, N: int, rows: list[dict], plt) -> Path | None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    any_curve = False
    for r in rows:
        sol = load_solution(case, r["solver"], N)
        if sol is None or "residual_history" not in sol:
            continue
        h = sol["residual_history"]
        if len(h) == 0:
            continue
        ax.semilogy(range(1, len(h) + 1), h, lw=1.8,
                    color=SOLVER_COLOUR.get(r["solver"]),
                    label=f"{r['solver']} ({r['scheme']})")
        any_curve = True
    if not any_curve:
        plt.close(fig)
        return None
    ax.set_xlabel("outer iteration (sweep or cycle)")
    ax.set_ylabel(r"$\|b - Au\|_2 / \|b\|_2$")
    ax.set_title(f"Convergence - {case}, N={N}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = PLOTS_DIR / f"convergence_{case}_N{N}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
#  Plots 5-8 - scalar metrics vs N  (mirror the 2-D script)
# ============================================================================

def plot_accuracy_vs_n(case: str, by_solver: dict, plt) -> Path | None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    any_line, N_all = False, []
    for (c, solver), rs in sorted(by_solver.items()):
        if c != case:
            continue
        Ns = [r["N"] for r in rs if r.get("linf_err") is not None]
        errs = [r["linf_err"] for r in rs if r.get("linf_err") is not None]
        if not Ns:
            continue
        ax.loglog(Ns, errs, "o-", lw=1.8, color=SOLVER_COLOUR.get(solver),
                  label=solver)
        N_all.extend(Ns); any_line = True
    if not any_line:
        plt.close(fig)
        return None
    N_all = sorted(set(N_all))
    if len(N_all) >= 2:
        ref = [(N_all[0] / n) ** 2 for n in N_all]
        ref_all = [r * (max(ax.get_ylim()) * 0.5 / max(ref)) for r in ref]
        ax.loglog(N_all, ref_all, "k--", lw=1, alpha=0.5, label=r"$O(h^2)$ ref.")
    ax.set_xlabel("N"); ax.set_ylabel(r"$L_\infty$ error vs. reference (%)")
    ax.set_title(f"Accuracy vs N - {case}")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    plt.tight_layout()
    out = PLOTS_DIR / f"accuracy_vs_N_{case}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_cost_vs_n(case: str, by_solver: dict, plt) -> Path | None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    any_line = False
    for (c, solver), rs in sorted(by_solver.items()):
        if c != case:
            continue
        Ns = [r["N"] for r in rs]
        wc = [r["weighted_cost"] for r in rs if r.get("weighted_cost") is not None]
        wt = [r["wall_time_s"] for r in rs]
        if wc and len(wc) == len(Ns):
            axes[0].loglog(Ns, wc, "o-", color=SOLVER_COLOUR.get(solver), label=solver)
        axes[1].loglog(Ns, wt, "o-", color=SOLVER_COLOUR.get(solver), label=solver)
        any_line = True
    if not any_line:
        plt.close(fig)
        return None
    axes[0].set_xlabel("N"); axes[0].set_ylabel("weighted strip-solve cost")
    axes[0].set_title("Cost (finest-solve units, N^2 solves/sweep in 3-D)")
    axes[1].set_xlabel("N"); axes[1].set_ylabel("wall time (s)")
    axes[1].set_title("Wall time")
    for a in axes:
        a.grid(alpha=0.3, which="both"); a.legend(fontsize=8)
    fig.suptitle(f"Cost vs N - {case}", fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / f"cost_vs_N_{case}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_overhead(case: str, by_N: dict, plt) -> Path | None:
    xs, ys, cs, labels = [], [], [], []
    for (c, N), rs in sorted(by_N.items()):
        if c != case:
            continue
        base = next((r["wall_time_s"] for r in rs if r["solver"] == "Thomas"), None)
        if not base:
            continue
        for r in rs:
            if r["solver"] == "Thomas" or not r["wall_time_s"]:
                continue
            xs.append(N); ys.append(r["wall_time_s"] / base)
            cs.append(SOLVER_COLOUR.get(r["solver"], "gray")); labels.append(r["solver"])
    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    seen = set()
    for x, y, c, l in zip(xs, ys, cs, labels):
        ax.semilogy(x, y, "o", color=c, label=l if l not in seen else None,
                   markersize=8)
        seen.add(l)
    ax.set_xlabel("N"); ax.set_ylabel("wall time / Thomas wall time")
    ax.set_title(f"Quantum overhead vs Thomas - {case}")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    plt.tight_layout()
    out = PLOTS_DIR / f"overhead_vs_thomas_{case}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_error_decomposition(case: str, by_solver: dict, plt) -> Path | None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    any_line, disc_plotted = False, False
    for (c, solver), rs in sorted(by_solver.items()):
        if c != case or solver == "Thomas":
            continue
        Ns = [r["N"] for r in rs if r.get("err_vs_thomas") is not None]
        alg = [r["err_vs_thomas"] for r in rs if r.get("err_vs_thomas") is not None]
        if Ns:
            ax.semilogy(Ns, [max(v, 1e-6) for v in alg], "o-",
                       color=SOLVER_COLOUR.get(solver), label=f"{solver} (vs Thomas)")
            any_line = True
        if not disc_plotted:
            Nd = [r["N"] for r in rs if r.get("err_thomas_vs_exact") is not None]
            disc = [r["err_thomas_vs_exact"] for r in rs
                   if r.get("err_thomas_vs_exact") is not None]
            if Nd:
                ax.semilogy(Nd, [max(v, 1e-6) for v in disc], "k--", lw=1.6,
                           label="discretisation (Thomas vs exact)")
                disc_plotted = True
    if not any_line:
        plt.close(fig)
        return None
    ax.set_xlabel("N"); ax.set_ylabel("error (%)")
    ax.set_title(f"Error decomposition - {case}")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    plt.tight_layout()
    out = PLOTS_DIR / f"error_decomposition_{case}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
#  Plot 9 - azimuthal mode fidelity (3-D-only metric)
# ============================================================================

def plot_azimuthal_fidelity(case: str, by_solver: dict, plt) -> Path | None:
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    any_line = False
    for (c, solver), rs in sorted(by_solver.items()):
        if c != case or solver == "Thomas":
            continue
        Ns = [r["N"] for r in rs if r.get("azimuthal_mode_rel_err") is not None]
        errs = [r["azimuthal_mode_rel_err"] for r in rs
               if r.get("azimuthal_mode_rel_err") is not None]
        if not Ns:
            continue
        ax.semilogy(Ns, [max(v, 1e-6) for v in errs], "o-",
                   color=SOLVER_COLOUR.get(solver), label=solver)
        any_line = True
    if not any_line:
        plt.close(fig)
        return None
    ax.set_xlabel("N"); ax.set_ylabel("azimuthal mode amplitude error (%)")
    ax.set_title(f"Spoke-mode fidelity vs N - {case}\n"
                "(does the solver reproduce the mode, not just the field norm)")
    ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    plt.tight_layout()
    out = PLOTS_DIR / f"azimuthal_fidelity_{case}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


# ============================================================================
#  Main
# ============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default=None)
    ap.add_argument("--N", type=int, default=None)
    ap.add_argument("--no-cutaway", action="store_true",
                    help="skip the 3-D cutaway orientation plot (mplot3d, "
                         "slower and least essential of the plots here)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    rows = load_results()
    if args.case:
        rows = [r for r in rows if r["case"] == args.case]
    if args.N:
        rows = [r for r in rows if r["N"] == args.N]
    if not rows:
        raise SystemExit("No matching rows.")

    if args.list:
        for r in sorted(rows, key=lambda r: (r["case"], r["N"], r["solver"])):
            print(f"  {r['case']:<34} N={r['N']:<4} {r['solver']:<8} "
                 f"scheme={r['scheme']:<10} {r['stop_reason']}")
        return

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        raise SystemExit("matplotlib is required for this script.")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    by_case_N = group_by_case_N(rows)
    by_case_solver = group_by_case_solver(rows)
    cases = sorted({c for c, _ in by_case_N})

    made = []
    for (case, N), case_rows in sorted(by_case_N.items()):
        made += plot_slices(case, N, case_rows, plt, TwoSlopeNorm)
        p = plot_polar_unwrap(case, N, case_rows, plt)
        if p: made.append(p)
        if not args.no_cutaway:
            p = plot_3d_cutaway(case, N, case_rows, plt)
            if p: made.append(p)
        p = plot_convergence(case, N, case_rows, plt)
        if p: made.append(p)

    for case in cases:
        for fn in (plot_accuracy_vs_n, plot_cost_vs_n, plot_error_decomposition,
                  plot_azimuthal_fidelity):
            p = fn(case, by_case_solver, plt)
            if p: made.append(p)
        p = plot_overhead(case, by_case_N, plt)
        if p: made.append(p)

    print(f"Wrote {len(made)} plot(s) to {PLOTS_DIR}")
    for p in made:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()