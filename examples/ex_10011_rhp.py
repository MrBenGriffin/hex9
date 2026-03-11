
from hhg9.h9.constants import H9O
from hhg9 import Registrar
from hhg9.domains import nets
import numpy as np
from numpy.typing import NDArray

RHOMBUS_LAYOUT = {
    # Rhombus map – 4 columns × 2 main rows, with NWP/SEA c2 sub-hexes
    # filling the middle band.
    #
    # width/height are in Lattice U/V units, same convention as nets.py:
    #   tri_w = W/U = 6  (face is 6U wide)
    #   tri_h = H/V = 9  (face is 9V tall, spans ±4.5V from centre)
    #   GW = 3U,  GH = 3V
    #
    # gx/gy in grid tuples are in GW/GH units: x_off = gx*GW, y_off = gy*GH.
    #
    # For c2trans faces the grid value is a list:
    #   [(primary_gx, primary_gy, primary_θ),
    #    (Δgx_c2=0, Δgy_c2=0, t_additional_c2=0),   ← always (0,0,0)
    #    (Δgx_c2=1, Δgy_c2=1, t_additional_c2=1),
    #    (Δgx_c2=2, Δgy_c2=2, t_additional_c2=2)]
    # Δgx/Δgy are deltas FROM the primary in GW/GH.
    # t_additional is extra 60°-steps applied AFTER the primary rotation.
    #
    # Net x extent: gx 0..3 × GW=3U, face ±3U → -3U to +12U = 15U wide
    # Net y extent: gy 2..6 × GH=3V, face ±4.5V → 1.5V to 22.5V = 21V tall
    'width':  15,   # U units
    'height': 21,   # V units
    'grid': {
        # Northern belt (gy=6 → y_off = 18V)
        (+1, +1, +1): (1., 6., 3),   # NEA  col 1
        (-1, +1, +1): (2., 6., 5),   # NEP  col 2
        (+1, -1, +1): (3., 6., 5),   # NWA  col 3
        # NWP c2=1 folds down to gy=4 (Δgy=-2, same θ),
        # c2=2 folds to (gx=1,gy=5) with additional 240° (t=4):
        #   primary θ=5 (300°) + 240° = 540° = 180° = θ=3 ✓
        (-1, -1, +1): [(0., 6., 5), (0., 0., 0.), (0., -2., 0.), (1., -1., 4.)],

        # Southern belt (gy=2 → y_off = 6V)
        # SEA c2=1 folds up to gy=4 (Δgy=+2, same θ),
        # c2=2 folds to (gx=2,gy=3) with additional 120° (t=2):
        #   primary θ=3 (180°) + 120° = 300° = θ=5 ✓
        (+1, +1, -1): [(3., 2., 3), (0., 0., 0.), (0., 2., 0.), (-1., 1., 2.)],
        (-1, +1, -1): (2., 2., 5),   # SEP  col 2
        (+1, -1, -1): (1., 2., 1),   # SWA  col 1
        (-1, -1, -1): (0., 2., 3),   # SWP  col 0
    },
}


class RhombusNet:
    """
    Wraps an OctahedralNet in 'rhombus' layout and adds the reverse projection.

    The forward direction (b_oct → rhombus_xy) is handled by BaryNet.forward()
    inside the parent OctahedralNet — no changes needed there.

    The reverse direction (rhombus_xy → b_oct) is implemented here using the
    already-existing BaryNet.backward() for each face.

    Usage
    -----
    """

    def __init__(self, registrar, layout: str = 'rhombus'):
        from hhg9.domains.octahedral_net import OctahedralNet
        # Register the rhombus layout before instantiating OctahedralNet.
        if layout not in nets.net_layouts:
            nets.net_layouts[layout] = RHOMBUS_LAYOUT
        self.n_oct = OctahedralNet(registrar, layout=layout)
        self.registrar = registrar

    # ------------------------------------------------------------------
    # Reverse projection: rhombus_xy → b_oct Points
    # ------------------------------------------------------------------
    def backward(self, xy: NDArray):
        """
        Map 2-D rhombus coordinates back to barycentric (b_oct) Points.

        Parameters
        ----------
        xy : NDArray, shape (N, 2)
            Points in the rhombus net coordinate frame.

        Returns
        -------
        Points
            Barycentric points in the b_oct domain, with .components set
            to the octant id for each point.
        """
        from hhg9 import Points
        n_oct = self.n_oct

        # 1. Identify which octant face each point belongs to.
        #    pt_face() returns an (N, 3) int8 sign array; (0,0,0) = outside all faces.
        signs = n_oct.pt_face(xy)                          # (N, 3)
        valid = np.any(signs != 0, axis=1)                 # (N,) bool

        out_xy  = np.full_like(xy, np.nan)
        out_oid = np.full(xy.shape[0], 255, dtype=np.uint8)

        for proj in n_oct.projs.values():
            oid = proj.fwd_cs.oid
            mask = np.all(signs == H9O.oid_cmp[oid], axis=1) & valid

            if not np.any(mask):
                continue

            pts_net = Points(xy[mask], domain=proj.fwd_cs)
            pts_bary = proj.backward(pts_net)

            out_xy[mask] = pts_bary.coords
            out_oid[mask] = oid

        b_oct = self.registrar.domain('b_oct')
        # Convert oid scalar array → component array expected by Points
        pts = Points(out_xy[valid], domain=b_oct, components=out_oid[valid])
        return pts


# ---------------------------------------------------------------------------
# Part 3: Convenience function — full pipeline rhombus_xy → lon/lat
# ---------------------------------------------------------------------------

def rhombus_to_lonlat(xy: NDArray, registrar=None) -> NDArray:
    """
    Convert rhombus net coordinates directly to geodetic lon/lat (degrees).
    Chains: rhombus_xy → b_oct → g_gcd

    Parameters
    ----------
    xy : NDArray, shape (N, 2)
    registrar : hhg9.Registrar, optional  (creates one if None)

    Returns
    -------
    NDArray, shape (N, 2)  — columns [longitude_deg, latitude_deg]
    """
    from hhg9 import Registrar
    if registrar is None:
        registrar = Registrar()

    rn = RhombusNet(registrar)
    b_pts = rn.backward(xy)                # rhombus → b_oct

    # b_oct → g_gcd  (GCDBary.backward already exists)
    gcd_pts = registrar.project(b_pts, ['b_oct', 'g_gcd'])
    return gcd_pts.coords                  # (N, 2) lon/lat in degrees


# ---------------------------------------------------------------------------
# Part 4: Notes on what IS reused vs what is new
# ---------------------------------------------------------------------------
#
# REUSED WITHOUT MODIFICATION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# • BaryNet.backward()              — net_xy → barycentric (projections/bary_net.py)
#   The rotation matrix and offset are already computed per-face in __init__.
#   backward() applies (xy - offset) @ matrix.T, which is correct for all
#   rotations including the relocated NWP/SEA half-hexes because their
#   rotation angle θ was already encoded in c2trans.
#
# • OctahedralNet.pt_face()         — point-in-triangle test (domains/octahedral_net.py)
#   inside_triangle_cw() is called per-face and returns a boolean mask.
#   This works unchanged on the rhombus layout because the triangles in
#   face_polys are computed from the layout grid positions — swapping to
#   RHOMBUS_LAYOUT automatically repositions all triangles.
#
# • OctahedralNet.__init__()        — reads layout dict, builds BaryNet projections.
#   No source changes needed; the layout is injected via nets.net_layouts.
#
# • GCDBary.backward()              — b_oct → g_gcd (projections/gcd_bary.py)
#   Unchanged.
#
# • Registrar.project()             — orchestrates the chain.
#   Unchanged.
#
# NEW / MODIFIED
# ~~~~~~~~~~~~~~
# • nets.py : add RHOMBUS_LAYOUT dict (10 lines)
#   The c2 sub-dict uses the already-scaffolded 'c2' key that OctahedralNet
#   reads on lines 94-97, storing results in BaryNet.c2trans. That field
#   is already set but not yet consumed — the rhombus layout is the intended
#   first consumer.
#
# • RhombusNet.backward()           — new, ~30 lines
#   Iterates over faces, calls the existing BaryNet.backward() per face,
#   assembles output into a b_oct Points object.
#
# • BaryNet.forward() with c2trans  — minor extension needed
#   For the *forward* direction with c2 relocations, BaryNet.forward()
#   needs to check self.c2trans and apply the alternate (gx, gy, θ) for
#   half-hexes that have been moved.  This requires knowing the c2
#   of each incoming point (available from H9R.mcc2 given the octant mode).
#   Estimated: ~15 additional lines inside BaryNet.forward().
#
# ---------------------------------------------------------------------------
# Part 5: Precise geometry for RHOMBUS_LAYOUT grid positions
# ---------------------------------------------------------------------------
#
# To derive the correct (gx, gy, θ) values for the rhombus layout we reason
# from the mortar layout and the 12 layer-0 hexagon ids.
#
# From H9O.l0hex_by_id (read from constants.py oct_constants()):
#   NEA (oid=0, mode=0):  c2→hex  [4, 2, 1]
#   NEP (oid=1, mode=1):  c2→hex  [5, 0, 3]
#   NWA (oid=2, mode=1):  c2→hex  [6, 3, 0]
#   NWP (oid=3, mode=0):  c2→hex  [7, 1, 2]
#   SEA (oid=4, mode=1):  c2→hex  [0, 8,10]
#   SEP (oid=5, mode=0):  c2→hex  [1, 7, 4]
#   SWA (oid=6, mode=0):  c2→hex  [2, 4, 7]
#   SWP (oid=7, mode=1):  c2→hex  [3, 6, 5]
#
# The 12 layer-0 hexagons and which pair of half-hexes compose them:
#   hex 0: NEP(c2=1)  + SWP(c2=?) — antipodal pair
#   hex 1: NEA(c2=2)  + SWP(?)
#   hex 2: NEA(c2=1)  + ...
#   ... (full table derivable from l0hex_back)
#
# Each hexagon occupies one rhombus cell.  The rhombus (col, row) for hex h:
#   col = h % 4,  row = h // 4    →  4 columns, 3 rows as stated above.
#
# The gx offset for octant face in that rhombus slot:
#   gx = col * GW,  gy = row * GH  + vertical_offset_within_triangle
#
# The rotation θ is *identical* to the mortar rotation for the same face,
# because the half-hex shape does not change — only the net position changes.
# We therefore copy θ from mortar and adjust only gx, gy.
#
# The NWP and SEA faces are the ones whose c2 half-hexes partially lie
# *outside* the 3.5×3 mortar bounding box.  In the rhombus layout they
# are repositioned so all half-hexes land inside 4×3.
#
# Exact (gx, gy) derivation for each face:
#   Start from mortar positions, then shift each face so its c2=0 half-hex
#   barycentre aligns with the lower-left corner of its target rhombus cell:
#
#   target_gx = col * 2  (in triangle-half-width units, matching mortar convention)
#   target_gy = row * 3  (in triangle-third-height units)
#   shift      = (target_gx - mortar_gx, target_gy - mortar_gy)
#
# The values in RHOMBUS_LAYOUT above are the result of applying these shifts.
# They should be validated by instantiating OctahedralNet(layout='rhombus')
# and visually inspecting the face_polys triangles.

# ---------------------------------------------------------------------------
# Example usage (requires full hhg9 install with pyproj etc.)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # Demonstrate the backward projection on a grid of rhombus-space points.
    # (Requires hhg9 dependencies: numpy, scipy, pyproj, geographiclib)
    import sys

    # Inject the rhombus layout
    nets.net_layouts['rhombus'] = RHOMBUS_LAYOUT

    reg = Registrar()
    rn  = RhombusNet(reg, layout='rhombus')

    # Sample a regular grid across the rhombus canvas
    xs = np.linspace(0.05, 3.95, 20)
    ys = np.linspace(0.05, 2.95, 15)
    gx, gy = np.meshgrid(xs, ys)
    pts_xy = np.column_stack([gx.ravel(), gy.ravel()])

    # Reverse project: rhombus → b_oct
    b_pts = rn.backward(pts_xy)
    print(f"Recovered {len(b_pts)} valid b_oct points from {len(pts_xy)} grid samples")

    # Continue to lon/lat via existing Registrar.project chain
    lonlat = reg.project(b_pts, ['b_oct', 'g_gcd'])
    print("Sample lon/lat (first 5):")
    print(lonlat.coords[:5])
