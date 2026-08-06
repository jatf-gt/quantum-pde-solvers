"""
Canonical SPT-100 Hall Effect Thruster channel geometry.

This module is the single authority for the physical dimensions of the discharge
channel. Every dimension of the HET benchmark — the 1D axial problem, the 2D
axial-radial slice and the 3D axial-radial-azimuthal slab — draws its extents
from here, so that a geometry stated once cannot drift between dimensions.

Device
------
The SPT-100 discharge chamber is an annulus of rectangular cross-section:

    axial length        L_z    = 25 mm
    inner radius        R_in   = 35 mm
    outer radius        R_out  = 50 mm
    channel width       L_r    = R_out − R_in = 15 mm
    mean radius         R_mean = ½(R_in + R_out) = 42.5 mm
    mean circumference  L_s    = 2π R_mean ≈ 267 mm

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
        justified by the thinness of the annulus: L_r / L_s = 15/267 ≈ 0.056, so
        curvature corrections enter at O(L_r / R_mean) ≈ 0.35 in the metric and
        are neglected, as is standard for axial-azimuthal HET studies.

    Axis convention in 3D: axis 0 = axial (Dirichlet, anode/cathode),
    axis 1 = radial (Dirichlet, walls), axis 2 = azimuthal (periodic).

Periodicity in the azimuthal direction is not a numerical convenience: it is
what makes the domain a thruster channel rather than a box, and the azimuthal
modes it admits are the physical reason to compute in 3D at all.

Provenance
----------
The geometry recorded here is the SPT-100 discharge chamber as reported in the
Hall thruster literature: axial length 2.5 cm, inner radius 3.5 cm, outer radius
5 cm. The outer radius of 50 mm is consistent across sources; the inner radius is
variously quoted as 25, 30 or 35 mm across thruster variants and measurement
conventions, and 35 mm is the value associated with the discharge chamber
proper.

Prior to the introduction of this module the 2D benchmark declared a radial
extent of 20 mm as a bare literal, in four independent places, whilst the 3D
benchmark derived 15 mm from the radii above. Both cannot describe the same
channel. `L_R_LEGACY_2D` preserves the former value so that results predating the
reconciliation remain reproducible and identifiable; it is not the SPT-100
geometry and must not be used for new work. See `L_R_LEGACY_2D`.

References
----------
Boeuf & Garrigues, J. Appl. Phys. 84(7), 3541-3554 (1998) — the 1D axial
    discharge model and its parameter table.
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

L_Z: float = 0.025        # Axial channel length [m]
R_IN: float = 0.035       # Inner channel radius [m]
R_OUT: float = 0.050      # Outer channel radius [m]

PHI_0: float = 300.0      # Nominal discharge voltage scale [V]


# ── Derived Dimensions ────────────────────────────────────────────────────────

L_R: float = R_OUT - R_IN                 # Radial channel width [m] = 15 mm
R_MEAN: float = 0.5 * (R_IN + R_OUT)      # Mean channel radius [m] = 42.5 mm
L_S: float = 2.0 * math.pi * R_MEAN       # Mean circumference [m] ≈ 267 mm

ASPECT_RADIAL_AZIMUTHAL: float = L_R / L_S      # ≈ 0.056; justifies unwrapping


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
Radial extent [m] used by the 2D benchmark before this module existed.

Retained solely so that 2D HET results produced before the reconciliation
remain reproducible and can be identified as such. It is **not** the SPT-100
channel width: 20 mm is inconsistent with the inner and outer radii declared
above, which give 15 mm. Do not use it for new work.

Switching the 2D benchmark from this value to `L_R` changes every 2D HET number
and must therefore be carried out together with the regeneration of the 2D
results, not as an isolated edit.
"""
