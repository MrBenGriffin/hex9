# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Direct hex mesh generation from H9 lattice LUTs.

For a given layer, supercell origins and scales are obtained analytically via
region_grid(layer-1, mode), then half-hex or full-hex vertices are placed using
H9P.hh or H9P.hx:

    verts = sc_origin + H9P.hh[sc_mode, c2] * sc_scale   # half-hex (4 verts)
    verts = sc_origin + H9P.hx[sc_mode, c2] * sc_scale   # full hex  (6 verts)

All half-hex vertices lie within their octant face.  Full-hex vertices include
2 "exterior" points in the adjacent face (noted in polygon.py as "final 2 pts
in opp. net_mode"); these project cleanly for non-c2trans faces but may show
artefacts at c2trans boundaries.

Adjacent half-hexes from neighbouring faces share their long edge; the proper
pairing step (mode-rule deduplication) is the next pass.

Last Tested
-----------
"""
import numpy as np
from hhg9 import Registrar
from hhg9.h9 import H9O
from hhg9.algorithms.geometry import points_in_triangles
from hhg9.h9.polygon import H9P, region_grid, net_poly, net_polys
from svgelements import SVG, Polygon, Matrix, Group


def _supercells(layer: int, mode: int):
    """Yield (sc_mode, sc_origin, sc_scale) for each supercell in one octant."""
    if layer == 0:
        yield mode, np.zeros(2, dtype=float), 1.0
    else:
        for _, sc_mode, sc_origin, sc_scale in region_grid(layer - 1, mode):
            yield sc_mode, sc_origin, sc_scale


def hh_verts_boct(layer: int, mode: int) -> np.ndarray:
    """Half-hex vertices in b_oct  →  (H, 4, 2).  All verts inside octant."""
    return np.array([
        sc_origin + H9P.hh[sc_mode, c2] * sc_scale
        for sc_mode, sc_origin, sc_scale in _supercells(layer, mode)
        for c2 in range(3)
    ])


def hx_verts_boct(layer: int, mo: int, sc_choice=0) -> np.ndarray:
    """Full-hex vertices in b_oct  →  (H, 6, 2).
    Last 2 verts per hex are in the adjacent octant face (opp. net_mode).
    """
    return np.array([
        sc_origin + H9P.hx[sc_mode, c2] * sc_scale
        for sc_mode, sc_origin, sc_scale in _supercells(layer, mo)
        for c2 in range(3) if sc_mode == sc_choice
    ])


def project_to_noct(verts_b, proj, n_verts):
    """Project (H, n_verts, 2) b_oct array to n_oct via BaryNet.forward().
    Returns (H, n_verts, 2); rows with any NaN are invalid (boundary artefacts).
    """
    flat_n = proj.forward(verts_b.reshape(-1, 2))
    return flat_n.reshape(-1, n_verts, 2)


if __name__ == '__main__':
    LAYER = 0
    FLAVOUR = 'butterfly:0500'

    reg = Registrar()
    n_oct = reg.domain(f'n_oct:{FLAVOUR}')
    net_p = net_polys(reg, n_oct)
    nvt = net_p.coords.reshape(-1, 3, 2)

    net_outline = net_poly(reg, n_oct)[0]

    img_w, img_h = n_oct.image_dims(pixels=800)
    full_hexes, half_hexes = [], []

    for side, sdom in n_oct.sides.items():
        oid  = sdom.oid
        mode = H9O.oid_mo[oid]
        proj = n_oct.projs[side]

        verts_b = hx_verts_boct(LAYER, mode, 0)
        verts_n = project_to_noct(verts_b, proj, 6)

        flat = verts_n.reshape(-1, 2)
        bounded = points_in_triangles(flat, nvt).reshape(-1, 6)
        in_net = bounded.all(axis=1).reshape(-1)
        full_hexes.extend(verts_n[in_net])

        if mode == 0:
            half_hexes.extend(verts_n[~in_net, :4, :])

        else:
            verts_x = hx_verts_boct(LAYER, mode, 1)
            verts_p = project_to_noct(verts_x, proj, 6)
            flat = verts_p.reshape(-1, 2)
            bounded = points_in_triangles(flat, nvt).reshape(-1, 6)
            in_net = bounded.all(axis=1).reshape(-1)
            half_hexes.extend(verts_p[~in_net, :4, :])

    print(f'n_oct: {len(full_hexes)} full hexes, {len(half_hexes)} boundary half-hexes')

    canvas = SVG(width=img_w, height=img_h, viewBox=f"0 0 {img_w} {img_h}")
    flip_matrix = Matrix(1, 0, 0, -1, 0, img_h)
    cluster_group = Group()
    scale = img_w / n_oct.wi
    svg_hexes = np.array(full_hexes) * scale
    svg_hh = np.array(half_hexes) * scale
    for pts in svg_hexes:
        points = pts.tolist()
        poly = Polygon(*points, fill="none", stroke="black", stroke_width=0.1, vector_effect="non-scaling-stroke")
        cluster_group.append(poly)

    for pts in svg_hh:
        points = pts.tolist()
        poly = Polygon(*points, fill="none", stroke="black", stroke_width=0.1, vector_effect="non-scaling-stroke")
        cluster_group.append(poly)

    # Draw a surrounding border?
    # net_poly = [(p * scale).tolist() for p in net_outline]
    # poly = Polygon(*net_poly, fill="none", stroke="black", stroke_width=1, vector_effect="non-scaling-stroke")
    # cluster_group.append(poly)

    cluster_group *= flip_matrix
    canvas.append(cluster_group)

    f_name = f'output/ex0110_{FLAVOUR}_L{LAYER}_noct_hx.svg'
    canvas.write_xml(f_name)
    print(f'Saved {f_name}')
    print('Done.')
