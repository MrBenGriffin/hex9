# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
QUICK WIN: a region-level 'always canonicalise' (packer-agnostic), plus a
diagnostic that isolates the one remaining non-symbolic step (the octant seam).

`canonicalise_regions` mirrors uuid_address._coalesce_bin's fold but returns the
canonical region chain (+ possibly-switched octant) instead of a UUID, so it can
feed ANY packer (uint64 or uuid). The fold itself is symbolic (region_neighbours
cascade); only the octant-hop re-derives the chain geometrically (y-flip).

The diagnostic answers: for seam (octant-hopping) folds, does region_neighbours'
own neighbour chain already equal the geometric y-flip chain?  If yes, the seam
hop is fully symbolic given OctConst.oid_nb (which _coalesce_bin already uses for
the octant id) — i.e. the geometry can be dropped.
"""
import numpy as np
from hhg9 import Registrar, Points
from hhg9.h9 import H9O, H9C, H9K
from hhg9.h9.addressing import hex_digits, neighbours, canonicalise
from hhg9.h9.region import region_neighbours, xy_regions
from hhg9.h9.classifier import location
from hhg9.h9.protocols import BaryLoc
from hhg9.h9.tail import TailStyle


def canonicalise_regions(coords, oc, mo, dom, layer):
    """Thin wrapper over the promoted library canonicaliser, returning the seam
    mask (octant changed = a seam hop) for this probe's diagnostics."""
    regions, oc_c = canonicalise(coords, oc, mo, dom, layer)
    seam = (oc_c != np.asarray(oc))
    return regions, oc_c, seam


def _sample(n):
    """Global grid + dense clusters near octant vertices/edges (to trigger seam folds)."""
    lat = np.linspace(-88, 88, n); lon = np.linspace(-179, 179, n)
    LA, LO = np.meshgrid(lat, lon)
    pts = [np.column_stack([LA.ravel(), LO.ravel()])]
    # Octant seams: the equator (N/S boundary) and the meridians at the face edges.
    # Sample bands a HAIR off the seam (never exactly on it) so each leaf cell is
    # unambiguous — points placed exactly on a seam have ambiguous region assignment.
    eps = 1e-7
    fine = np.linspace(-179.0, 179.0, 6000)
    for dlat in (eps, -eps, 3e-6, -3e-6):               # equatorial band, off-seam
        pts.append(np.column_stack([np.full_like(fine, dlat), fine]))
    flat = np.linspace(-89.0, 89.0, 6000)
    for clon in (0, 45, 90, 135, 180, -45, -90, -135):  # meridian bands, off-seam
        for d in (eps, -eps, 3e-6, -3e-6):
            pts.append(np.column_stack([flat, np.full_like(flat, clon + d)]))
    return np.vstack(pts)


def main(L=8, n=80):
    reg = Registrar()
    b_oct = reg.domain('b_oct'); g_gcd = reg.domain('g_gcd')
    bry = reg.project(Points(_sample(n), g_gcd), [g_gcd, b_oct])
    oc, mo = bry.cm()
    N = len(bry)

    regions, oc_c, seam = canonicalise_regions(bry.coords, oc, mo, b_oct, L)

    # (1) canonical property: leaf parent mode == 0 for every in-scope point
    x, y = bry.coords[:, 0], bry.coords[:, 1]
    in_scope = (location(H9K.R3 * x, y, mo) != BaryLoc.EXT) & \
               (location(H9K.R3 * x, y, mo) != BaryLoc.UDF)
    pen_mode = H9C.mode[regions[:, -2]]
    print(f"N={N} L={L}")
    print(f"(1) leaf-parent mode==0 after canon (in-scope): "
          f"{int((pen_mode[in_scope]==0).sum())}/{int(in_scope.sum())}")

    # (2) agreement with the library reference (neighbours coalesce)
    ref = neighbours(bry, layer=L, coalesce=True)
    ref_reg = np.asarray(xy_regions(ref.coords, ref.cm()[1], L))
    ref_oc = ref.cm()[0]
    cell_match = np.all(regions == ref_reg, axis=1) & (oc_c == ref_oc)
    print(f"(2) matches neighbours() reference: {int(cell_match.sum())}/{N}  "
          f"(non-seam {int(cell_match[~seam].sum())}/{int((~seam).sum())}, "
          f"seam {int(cell_match[seam].sum())}/{int(seam.sum())})")

    # (3) symbolic-seam diagnostic: does region_neighbours' chain already equal
    #     the geometric y-flip chain for the hopped rows?
    regions_raw = xy_regions(bry.coords, mo, L)
    act = ((location(H9K.R3*x, y, mo) != BaryLoc.EXT) &
           (location(H9K.R3*x, y, mo) != BaryLoc.UDF) & (H9C.mode[regions_raw[:, -2]] == 1))
    idx = np.flatnonzero(act)
    nbr, c2 = region_neighbours(regions_raw[idx])
    hopped = regions_raw[idx, 0] != nbr[:, 0]
    if hopped.any():
        hidx = idx[hopped]
        oc_h = H9O.oid_nb[oc[hidx], c2[hopped]]
        geom = xy_regions(np.column_stack([x[hidx], -y[hidx]]), H9O.oid_mo[oc_h], L)
        sym = nbr[hopped]
        full = np.all(sym == geom, axis=1)
        body = np.all(sym[:, :-1] == geom[:, :-1], axis=1)
        print(f"(3) seam rows={hopped.sum()}: region_neighbours chain == geometric flip chain: "
              f"full {int(full.sum())}/{len(full)}, body-only {int(body.sum())}/{len(body)}")
    else:
        print("(3) no seam (octant-hopping) folds in this sample")


if __name__ == '__main__':
    main()
