# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Finding acceptable authalic reference points.
11.87976511 45.
At an address that is normative,
for each hex_layer calculate the area of a hexagon.
Compare with the ak_area values.

Last Tested
26 December 2025 0.1.0a4 (passed)
17 December 2025 0.1.0a3 (passed)
"""
import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area, enu_planar_polygon_area
from hhg9.h9 import H9_RA, H9C, H9K, H9O
from hhg9.h9.region import regions_xy, xy_regions, H9R, region_neighbours
from hhg9.h9.polygon import hex_poly_layer, hh_layer, H9P, tri_grid
from hhg9.h9.addressing import hex_layer, hex_digits_reg, neighbours, hex_digits, TailStyle, hex_key, \
    tail_unpack_reversible
from hhg9.h9.classifier import location
import matplotlib as mpl


# def hexify(reg: Registrar, b_pts: Points, layers: int = 4):
#     """
#     Find hexagons for data, and display on a 'globe'.
#     """
#     pts, pops = hex_poly_layer(b_pts, layers)
#
#     # Now calculate their area as a metric. (ignore pops).
#     gm2 = 510_065_621_724_154.6  # total surface area of WGS-84 (m²)
#     bins = 12*9**layers          # number of hexes at this hex_layer
#     w_area_m2_mean = gm2/bins    # ideal equal-area per hex
#
#     c_pts = reg.project(pts, ['b_oct', 'c_oct', 'c_ell'])  # use bary.
#     g_pts = reg.project(pts, ['b_oct', 'g_gcd'])
#     w_area_m2 = wgs84_area(reg, g_pts)  # default value is 6
#     w_adj = np.abs(w_area_m2 / w_area_m2_mean) + 1e-12
#     score = np.log(w_adj)  # authalic log-density ℓ
#     # snow_globe(c_pts, 6, score, f'{layers}')

def get_tri_data(reg: Registrar, layer=3, octant_id=None):
    """Load up global sample data"""
    tg0 = tri_grid(layer, 0).reshape([-1, 2])  # triangle polygons.
    tg1 = tri_grid(layer, 1).reshape([-1, 2])  # triangle polygons.
    tgx = [tg0, tg1]
    b_oct = reg.domain('b_oct')
    if octant_id is None:
        repo = []
        for octant in b_oct.signs.keys():
            o_id = b_oct.sign_to_id[octant]
            mode = b_oct.oid_mo[o_id]
            repo.append(Points(tgx[mode].copy(), b_oct, components=octant))
        return Points.concat(repo)
    else:
        o_id = octant_id
        mode = H9O.oid_mo[o_id]
        cmp = H9O.oid_cmp[o_id]
        return Points(tgx[mode], b_oct, components=cmp)


def mid_lat(gxd: Points):
    """Return middling latitude between two points in gcd"""
    xy = gxd.coords
    mx = np.mean(xy, axis=0)
    xy3 = np.vstack([xy[0], mx, xy[1]])
    return Points(xy3, gxd.domain)


def mid_select(gxd: Points, dens, ref):
    """Return middling latitude between two points in gcd"""
    xy = gxd.coords
    ar = dens[0] - ref
    br = dens[1] - ref
    cr = dens[2] - ref
    if ar == 0 or br == 0 or cr == 0:
        if ar == 0:
            print(f'found: {xy[0]}')
        if br == 0:
            print(f'found: {xy[1]}')
        if cr == 0:
            print(f'found: {xy[2]}')
        exit(0)
    a_gtr = ar > 0
    b_gtr = br > 0
    c_gtr = cr > 0
    if a_gtr ^ b_gtr:
        mxy = np.vstack([xy[0], xy[1]])
    elif b_gtr ^ c_gtr:
        mxy = np.vstack([xy[1], xy[2]])
    else:
        raise ValueError('no crossover found')
    return Points(mxy, gxd.domain)


def log_lat_ref(gxd: Points, n=100_000):
    """Return n points in gcd"""
    xy = gxd.coords
    line = np.linspace(xy[0], xy[1], n)
    return Points(line, gxd.domain)


def min_authalic(reg, gxd, depth):
    """Given 2 points in gcd, discover a midpoint of least authalic"""
    xy = reg.project(gxd, ['g_gcd', 'b_oct'])
    b_oct = reg.domain('b_oct')
    cmp = tuple(list(xy.components[0]))
    oid = H9O.cmp_oid[cmp]  # NEA
    ref_data = get_tri_data(reg, layer=depth, octant_id=oid)
    o_ref = rg.project(ref_data, ['b_oct', 'c_oct'])
    ref_areas = enu_planar_polygon_area(rg, o_ref, 3)
    se = sum(ref_areas)  # should approximate
    ake = reg.projection('oct_ell')
    tri_0 = ake.tri_0
    ideal_hex, ideal_tri = ake.get_accuracy(depth + 1)  # !! Something odd here.
    rf_adj = tri_0 / se
    ref_areas *= rf_adj
    tr_pts = ref_data.coords.reshape([-1, 3, 2])
    tr_cts = tr_pts.mean(axis=1)
    ref = Points(tr_cts, b_oct, components=cmp, samples=ref_areas / ideal_tri)
    ref_o = rg.project(ref, ['b_oct', 'c_oct'])
    face = H9O.oid_str[oid]  # NEA
    prj = b_oct.projs[face]  # NEA matrices etc.
    q = prj.matrix.T @ prj.orient
    e1_xyz = q[:, 0]  # 3-vector
    e2_xyz = q[:, 1]  # 3-vector
    ref_j_pts = ake.jacobian(ref_o.coords)
    ref_v1 = ref_j_pts @ e1_xyz  # (hex_layer, 3)
    ref_v2 = ref_j_pts @ e2_xyz  # (hex_layer, 3)
    ref_cross = np.cross(ref_v1, ref_v2)  # (hex_layer, 3)
    ref_scale = np.linalg.norm(ref_cross, axis=1)  # (hex_layer,)
    ref_clip = np.clip(ref_scale, 1e-20, None)
    w_param = ref_areas / ref_clip  # ≈ param-space area weights
    ref_mean = np.sum(ref_clip * w_param) / np.sum(w_param)  # param-area mean of s
    log_ref = np.log(ref_mean)  # reference log-density ℓ

    for i in range(30):
        g3d = mid_lat(gxd)
        uvw = reg.project(g3d, ['g_gcd', 'c_ell', 'c_oct'])
        j_pts = ake.jacobian(uvw.coords)
        v1 = j_pts @ e1_xyz  # (hex_layer, 3)
        v2 = j_pts @ e2_xyz  # (hex_layer, 3)
        cross = np.cross(v1, v2)  # (hex_layer, 3)
        area_scale = np.linalg.norm(cross, axis=1)  # (hex_layer,)
        area_clip = np.clip(area_scale, 1e-20, None)
        density = np.log(area_clip)  # authalic log-density ℓ
        gxd = mid_select(g3d, density, log_ref)
    gpx = rg.project(ref, ['b_oct', 'g_gcd'])
    x45 = (gpx.coords[:, 1] == 45.0)
    o45 = gpx.select(x45)
    a45 = np.abs(o45.samples - 1.0)
    o10 = np.argmin(a45)
    gpa = o45.coords[o10]
    ans = Points(np.atleast_2d(gpa), g_gcd)
    print(f'ref found at: {ans.coords[0]}')
    print(f'authalic found at: {gxd.coords[0]}')
    for rxt, kind in zip([ans, gxd], ['ref', 'authalic']):
        ptx = reg.project(rxt, ['g_gcd', 'b_oct'])
        print(f'\n\n====== {kind}')
        for layer in range(30):
            hxp, _ = hex_poly_layer(ptx, layer)
            g_pts = reg.project(hxp, ['b_oct', 'g_gcd'])
            ideal_hex, ideal_tri = ake.get_accuracy(layer)
            w_area_m2 = wgs84_area(reg, g_pts)  # default value is 6
            w_adj = np.abs(w_area_m2 / ideal_hex) + 1e-12
            score = np.log(w_adj)  # authalic log-density ℓ
            print(layer, score, ideal_hex, w_area_m2)


if __name__ == '__main__':
    start = np.array([[11.0, 45.0], [12.0, 45.0]])
    rg = Registrar()  # Manage Domains & Projections
    g_gcd = rg.domain('g_gcd')
    b_oct = rg.domain('b_oct')
    b_oct.set_warp('src/l4_polished.npz')
    pts = Points(start, g_gcd)
    pt = min_authalic(rg, pts, 3)

