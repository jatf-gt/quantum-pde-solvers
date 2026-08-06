"""
Canonical SPT-100 Hall Effect Thruster channel geometry.

This module is the single authority for the physical dimensions of the discharge
channel. Every dimension of the HET benchmark — the 1D axial problem, the 2D
axial-radial slice and the 3D axial-radial-azimuthal slab — draws its extents
from here, so that a geometry stated once cannot drift between dimensions.

Device
------
The SPT-100 discharge chamber is an annulus of rectangular cross-section:

    axial length        L_z    = 40 mm
    inner radius        R_in   = 30 mm
    outer radius        R_out  = 50 mm
    channel width       L_r    = R_out − R_in = 20 mm
    mean radius         R_mean = ½(R_in + R_out) = 40 mm
    mean circumference  L_s    = 2π R_mean ≈ 251.3 mm

The channel width and circumference are *derived*, never declared, because a
declared width can disagree with the radii that supposedly produced it — which
is precisely the defect this module exists to remove (see `Provenance` below).

Dimensional reductions
----------------------
Each benchmark dimension is a projection of the same annulus:

    1D  axial only. Domain [0, L_z]; anode at z = 0, cathode plane at z = L_z.
    2D  axial-radial (z, r) slice. Domain [0, L_z] × [0, L_r]; the radial
        coordinate is measured from the inner wall, so r ∈ [0, L_r] maps to
        physical radius R_in + r.
    3D  axial-radial-azimuthal. The annulus is unwrapped to a Cartesian slab
        with the azimuthal coordinate s ∈ [0, L_s) periodic. The unwrapping is
        justified by the thinness of the annulus: L_r / L_s = 20/251 ≈ 0.080, so
        curvature corrections enter at O(L_r / R_mean) ≈ 0.5 in the metric and
        are neglected, as is standard for axial-azimuthal HET studies.

    Axis convention in 3D: axis 0 = axial (Dirichlet, anode/cathode),
    axis 1 = radial (Dirichlet, walls), axis 2 = azimuthal (periodic).

Periodicity in the azimuthal direction is not a numerical convenience: it is
what makes the domain a thruster channel rather than a box, and the azimuthal
modes it admits are the physical reason to compute in 3D at all.

Provenance
----------
The geometry recorded here is the SPT-100 discharge chamber exactly as reported
by Boeuf & Garrigues (1998), the primary source this benchmark's HET application
is built on — read directly from the paper (not a secondary description):
p. 3542, §II ("internal cylinder radius: R₁ = 3 cm; external cylinder radius:
R₂ = 5 cm; column length: d = 4 cm") and restated identically at p. 3546, §IV.A
for the specific simulated results ("the inner cylinder radius is 3 cm, the
inner radius of the outer cylinder is 5 cm, and the length of the column is
4 cm"). R_in = 30 mm, R_out = 50 mm, L_z = 40 mm follow directly.

An earlier version of this module used R_in = 35 mm and L_z = 25 mm, drawn from
secondary characterisations of the SPT-100 (a Wikipedia summary, a ScienceDirect
overview, the Charoy et al. 2D PIC benchmark) rather than from Boeuf & Garrigues
itself, and never cross-checked against it despite this being the paper the HET
`het_*_boeuf_garrigues`/`het_*_mms_spt100` cases are named for. Checked directly
against the primary source (2026-08-07): R_in = 35 mm does not appear anywhere
in Boeuf & Garrigues; R_in = 30 mm is stated twice, unambiguously, for exactly
the SPT-100 configuration this benchmark models. The former 2D-only literal of
20 mm for the radial extent (`L_R_LEGACY_2D`, now equal to the corrected `L_R`)
turns out to have been the physically correct value the whole time — the 3D
benchmark's "correction" to 15 mm in the prior consolidation pass was itself the
error, made without checking the primary source. That module-level distinction
is retained below purely as a historical record; new work should just use `L_R`.

References
----------
Boeuf & Garrigues, J. Appl. Phys. 84(7), 3541-3554 (1998) — the 1D axial
    discharge model, its parameter table, and the geometry above (§II, §IV.A).
Charoy et al., Plasma Sources Sci. Technol. 28, 105010 (2019) — the 2D
    axial-azimuthal particle-in-cell benchmark; note that community benchmarks
    conventionally simulate a truncated azimuthal slice (L_y ≈ 10 mm) rather than
    the full mean circumference used here.
"""
from __future__ import annotations

import math


# ── Primary Dimensions ────────────────────────────────────────────────────────
#
# These four numbers are the entire declared geometry. Everything else is
# derived from them.

L_Z: float = 0.040        # Axial channel length [m]
R_IN: float = 0.030       # Inner channel radius [m]
R_OUT: float = 0.050      # Outer channel radius [m]

PHI_0: float = 300.0      # Nominal discharge voltage scale [V]


# ── Derived Dimensions ────────────────────────────────────────────────────────

L_R: float = R_OUT - R_IN                 # Radial channel width [m] = 20 mm
R_MEAN: float = 0.5 * (R_IN + R_OUT)      # Mean channel radius [m] = 40 mm
L_S: float = 2.0 * math.pi * R_MEAN       # Mean circumference [m] ≈ 251.3 mm

ASPECT_RADIAL_AZIMUTHAL: float = L_R / L_S      # ≈ 0.080; justifies unwrapping


# ── Electrode Potentials ──────────────────────────────────────────────────────
#
# Operating point of the SPT-100 at nominal discharge conditions. The wall
# potential is the floating potential of the dielectric channel wall, which sits
# a few electron temperatures below the local plasma potential; at T_e ≈ 20 eV a
# floating wall of −20 V is representative.

V_ANODE: float = 300.0      # Anode potential [V]
V_CATHODE: float = 0.0      # Cathode-plane potential [V]
V_WALL: float = -20.0       # Floating dielectric wall potential [V]


# ── Azimuthal Mode Structure ──────────────────────────────────────────────────
#
# The rotating spoke is a coherent low-order azimuthal density perturbation
# observed in Hall thrusters, modelled here as a multiplicative modulation
# 1 + ε cos(2π m s / L_s) applied to the axial profile.

SPOKE_MODE_M: int = 2         # Azimuthal mode number of the rotating spoke
SPOKE_EPSILON: float = 0.30   # Relative amplitude of the azimuthal modulation


# ── Superseded Values ─────────────────────────────────────────────────────────

L_R_LEGACY_2D: float = 0.020
"""
Radial extent [m] used by the 2D benchmark before `core/het_geometry.py` existed.

Historical curiosity, not a live alternative: this value turns out to equal the
corrected `L_R` above (both 20 mm), now that `R_in` is read directly from
Boeuf & Garrigues rather than from secondary sources. Retained only as a record
that the pre-reconciliation 2D benchmark's bare literal was, by whatever
reasoning it was chosen, the physically correct one — and that the earlier
"correction" to 15 mm (derived from `R_in = 35 mm`, itself never checked
against the primary source) was the actual error. Use `L_R` for all new work;
this name is kept so that any 2D HET result computed under the intermediate
(15 mm) geometry remains identifiable, not reproducible-and-preferred.
"""
