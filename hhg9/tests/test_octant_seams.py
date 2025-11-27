"""
Unit tests for the octant seams tests.
"""
import numpy as np
import pytest

from hhg9 import Registrar, Points
from hhg9.domains import GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric
from hhg9.formats import OctahedralH9
from hhg9.h9.cells import ugc_regions, ugc_dec
from hhg9.projections import EllipsoidGCD, AKOctahedralEllipsoid


@pytest.fixture(scope="module")
def world():
    reg = Registrar()
    g_gcd = GeneralGCD(reg)
    c_ell = EllipsoidCartesian(reg)
    c_oct = OctahedralCartesian(reg)
    b_oct = OctahedralBarycentric(reg, c_oct)
    # Projections
    EllipsoidGCD(reg)
    AKOctahedralEllipsoid(reg)
    # h9f = OctahedralH9()  # formatter.
    # b_oct.register_format(h9f)

    return reg, g_gcd, c_ell, c_oct, b_oct


def test_lut_edge_consistency(world):
    reg, g_gcd, c_ell, c_oct, b_oct = world
    PRESERVE = {'EA','EP','NA','WA','WP','SA'}
    SWAP = {'NE','NW','NP','SE','SP','SW'}
    for f in range(8):
        edges_f = list(b_oct.edges_by_id[f])
        for ci, e in enumerate(edges_f):
            nb = b_oct.oid_nb[f, ci]
            edges_nb = list(b_oct.edges_by_id[nb])
            assert e in edges_nb, (
                f"Face {f} c1={ci} edge {e} not in neighbour {nb} edge list {edges_nb}"
            )
            code = int(b_oct.nb_c1map[f, ci])
            if e in PRESERVE:
                assert code == 0, f"Preserve edge {e} must map with code 0 (got {code})"
            else:
                assert e in SWAP
                assert code in (2, 3), f"Swap edge {e} must have code 2 or 3 (got {code})"


def _encroach_roundtrip_ok(b_oct, h9e, f, ci):
    mode   = int(b_oct.oid_mo[f])
    nb_face = int(b_oct.oid_nb[f, ci])
    code    = int(b_oct.nb_c1map[f, ci])     # 0,2,3
    k       = int((b_oct.rot90_idx[f] - b_oct.rot90_idx[nb_face]) % 4)
    R3 = h9e.R3

    # Build a tiny point inside the triangle near that edge; avoid exactly-on-boundary
    if mode == 1:  # up (Λ)
        if ci == 0:
            xy = np.array([[0.0, h9e.ΛF + 2e-4]])
        elif ci == 1:
            x = 1e-4; xy = np.array([[x, R3 * x + 2e-4]])
        else:
            x = 1e-4; xy = np.array([[x, -R3 * x + 2e-4]])
    else:  # down (V)
        if ci == 0:
            xy = np.array([[0.0, h9e.VC - 2e-4]])
        elif ci == 1:
            x = 1e-4; xy = np.array([[x, R3 * x + h9e.VF + 2e-4]])
        else:
            x = 1e-4; xy = np.array([[x, -R3 * x + h9e.VF + 2e-4]])

    src = Points(xy, b_oct, components=np.array([b_oct.oid_cp[f]]))
    nb_face = int(b_oct.oid_nb[f, ci])
    code    = int(b_oct.nb_c1map[f, ci])
    rturns  = int((b_oct.rot90_idx[f] - b_oct.rot90_idx[nb_face]) % 4)

    xy_adj = h9e.encroach_to_neighbour(
        src.coords, np.array([mode]),
        code=np.array([code]), rturns=np.array([rturns])
    )
    # Return leg: inverse k for PRESERVE, same k for SWAP
    k_back = (-k) % 4 if code == 0 else k

    xy_back = h9e.encroach_to_neighbour(
        xy_adj, np.array([1 - mode]),
        code=np.array([code]), rturns=np.array([k_back])
    )
    return np.allclose(xy_back, src.coords, atol=5e-4)


def test_all_seams_roundtrip(world):
    reg, g_gcd, c_ell, c_oct, b_oct = world
    h9e = b_oct.engine
    for f in range(8):
        for ci in range(3):
            assert _encroach_roundtrip_ok(b_oct, h9e, f, ci), f"roundtrip failed for face {f} c1={ci}"


def test_poles_mode_only(world):
    reg, g_gcd, c_ell, c_oct, b_oct = world
    h9e = b_oct.engine

    def check_pole(lat):
        pole = Points(np.array([[lat, 0.0]]), g_gcd)
        nbary = reg.project(pole, [g_gcd, c_ell, c_oct, b_oct])

        n1 = h9e.neighbours(nbary, 8)
        n2 = h9e.neighbours(n1, 8)

        # 1) Coordinates unchanged at vertex
        assert np.allclose(nbary.coords, n1.coords, atol=1e-12)
        assert np.allclose(nbary.coords, n2.coords, atol=1e-12)

        # 2) Octant ids unchanged
        oc0, _ = nbary.cm()
        oc1, _ = n1.cm()
        oc2, _ = n2.cm()
        assert np.array_equal(oc0, oc1)
        assert np.array_equal(oc0, oc2)

    check_pole(90.0)   # North Pole
    check_pole(-90.0)  # South Pole

import numpy as np
import pytest

# Tune these if you like
EPS_LON = 1e-13         # tiny east/west offset in degrees
DEPTH   = 30            # region depth for classification
ATOL    = 1e-40         # coord tolerance for roundtrip check
LAT_MIN = -88.0         # avoid poles/vertices
LAT_MAX =  88.0
N_SAMPLES = 100

@pytest.mark.parametrize("latitudes", [np.linspace(LAT_MIN, LAT_MAX, N_SAMPLES)])
def test_seam_greenwich_bulk(world, latitudes):
    """
    Sample ~100 points along the Greenwich meridian on both sides (±EPS_LON)
    and verify neighbour consistency:
      - oc' == oid_nb[oc, c1]
      - double neighbour returns to original coordinates
    """
    from hhg9 import Points

    reg, g_gcd, c_ell, c_oct, b_oct = world
    h9e = b_oct.engine
    dom = b_oct

    # Build west/east points on Greenwich meridian (skip a small band near equator if you want)
    lats = latitudes.astype(float)
    west = np.column_stack([lats, np.full_like(lats, -EPS_LON)])
    east = np.column_stack([lats, np.full_like(lats,  EPS_LON)])


    # Project both sides to barycentric
    P_w = reg.project(Points(west, g_gcd), [g_gcd, c_ell, c_oct, b_oct])
    P_e = reg.project(Points(east, g_gcd), [g_gcd, c_ell, c_oct, b_oct])

    # Helper to run checks vectorised for one side
    def check_side(P):
        oc, mode = P.cm()

        # Classify down the tree to get parent c1 and neighbour addresses
        regions = ugc_regions(P.coords, mode)
        c1, reg_nb = h9e.region_neighbours(regions)

        # Decode neighbour to (x,y,mode-inferred)
        xym = ugc_dec(reg_nb)
        oob = (xym[:, -1] != mode)  # octant seam crossed

        # Compute expected destination octant (per-LUT) for those crossings
        nb_expected = dom.oid_nb[oc[oob], c1[oob]]

        # Run production neighbour operator
        N1 = h9e.neighbours(P, DEPTH)

        oc_after, _ = N1.cm()
        assert np.array_equal(oc_after[oob], nb_expected), "LUT dest octant mismatch on seam"

        # Roundtrip: neighbour again should come back (up to numeric noise)
        N2 = h9e.neighbours(N1, DEPTH)

        # rows that actually crossed a seam
        src = P.coords[oob]
        dst = N2.coords[oob]

        # per-row max-abs error
        row_err = np.max(np.abs(dst - src), axis=1)
        bad = np.where(row_err > ATOL)[0]

        # If nothing failed, we're done
        if bad.size == 0:
            return

        # Otherwise, report only the failing rows
        src_bad = src[bad]
        dst_bad = dst[bad]
        err_bad = row_err[bad]

        # Optional: also show which latitudes those were (if you have lats for this side)
        # comment logs if not available in scope
        try:
            lat_bad = lats[oob][bad]
        except Exception:
            lat_bad = None

        # Build a readable diff table
        lines = ["double neighbour drifted on {} rows (ATOL={}):".format(bad.size, ATOL)]
        for i, j in enumerate(bad):
            lhs = src_bad[i]
            rhs = dst_bad[i]
            if lat_bad is None:
                lines.append(
                    f"  idx {j:>3}: src=({lhs[0]: .12f}, {lhs[1]: .12f})  "
                    f"dst=({rhs[0]: .12f}, {rhs[1]: .12f})  "
                    f"max|Δ|={err_bad[i]:.3e}"
                )
            else:
                lines.append(
                    f"  idx {j:>3} lat={lat_bad[i]: .6f}: "
                    f"src=({lhs[0]: .12f}, {lhs[1]: .12f})  "
                    f"dst=({rhs[0]: .12f}, {rhs[1]: .12f})  "
                    f"max|Δ|={err_bad[i]:.3e}"
                )

        # Finally assert only on the failing subset (so pytest shows small arrays)
        assert np.allclose(src_bad, dst_bad, atol=ATOL), "\n".join(lines)

    # Check both sides independently; between them we exercise both directions across the seam
    check_side(P_w)
    check_side(P_e)
