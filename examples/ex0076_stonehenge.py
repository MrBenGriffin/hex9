# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This reads in the GCD for Stonehenge, then converts it into a set of nested half-hexagons,
each of which represents a single Stage of the H9 Journey.
This is a much earlier proof of concept, but still gives a good idea of layers 0-12 of the H9 grid.
Last Tested 25 November 2025 √
"""
import os
import json
import numpy as np
from matplotlib import image, pyplot as plt
from scipy.spatial import KDTree
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_angular_ratio
from hhg9.algorithms.grid import qa_grid
from PIL import Image  # Pillow for clean image saving
from hhg9.h9.polygon import enmesh
from hhg9.h9.region import xy_regions


def json_load(path):
    """Load json file"""
    with (open(path) as infile):
        obj = json.load(infile)
        infile.close()
    return obj


if __name__ == '__main__':
    """
        Compose a set of zooms into stonehenge..
    """
    reg = Registrar()  # Manage Domains & Projections
    g_gcd = reg.domain('g_gcd')
    b_oct = reg.domain('b_oct')
    p_pix = reg.domain('p_pix')
    pix_gcd = reg.projection('pix_gcd')

    # metres = [100.]  # 0.01 = cm
    locs = json_load('../assets/locations.json')
    region = locs['NWA']
    spot = region['Stonehenge']
    ll0 = Points(np.array([spot]), g_gcd)
    bc0 = reg.project(ll0, [g_gcd, b_oct])  # spherical cart
    co, mo = bc0.cm()
    cmp = bc0.invert_octant_ids(co)[0]
    uri = xy_regions(bc0.coords, mo)
    polys, mesh = enmesh(bc0, 13)

    # We don't need to tile here as all share the same component in this case.
    verts = polys.reshape(-1, 2)                  # (U*4, 2)
    # co_row = np.atleast_2d(cmp)                 # (1, C)
    # co_tiled = np.repeat(co_row, verts.shape[0], axis=0)  # (U*4, C)
    bnd = Points(verts, b_oct, components=cmp)
    gel = reg.project(bnd, [b_oct, g_gcd])
    gpy = gel.coords.reshape([-1, 4, 2])

    extents = np.load('sh/extents.npy')
    for i, extent in enumerate(extents[0]):
        ratio = wgs84_angular_ratio(extent)
        b_poly = polys[i]
        g_poly = gpy[i]
        lon_min, lon_max, lat_min, lat_max = extent
        lw = lon_max - lon_min
        lh = lat_max - lat_min
        img_file = f'sh/SH_{i}.png'
        if os.path.isfile(img_file):
            img = image.imread(img_file, 'png')
        else:
            break
        pc_px = p_pix.adopt(img)
        pix_gcd.set_dim(pc_px, extent)
        pc_sp = reg.project(pc_px, [p_pix, g_gcd])

        # KDTree sample
        src = KDTree(pc_sp.coords)
        bpx, bpy = b_poly[:, 0], b_poly[:, 1]
        min_bx, max_bx = np.min(bpx), np.max(bpx)
        min_by, max_by = np.min(bpy), np.max(bpy)
        bxr = max_bx - min_bx
        byr = max_by - min_by
        grid_w = 2000 if bxr > byr else 4000
        gw, gh, pxl, msk, (grid_org, (sx, sy)) = qa_grid(b_poly, grid_w)
        in_scope = pxl[msk]
        pts = Points(in_scope, b_oct, components=cmp)
        sp1 = reg.project(pts, [b_oct, g_gcd])
        _, idx = src.query(sp1.coords, workers=-1)  # query KDTree and return indices of pc_sp
        samples = pc_sp.samples[idx]
        if samples.shape[1] == 3:
            rgba = np.hstack((samples, np.ones((samples.shape[0], 1), dtype=samples.dtype)))
        else:
            rgba = samples
        # Paint into raster
        pvx = np.round((pxl[:, 0] - grid_org[0]) * sx).astype(int)
        pvy = np.round((pxl[:, 1] - grid_org[1]) * sy).astype(int)
        pvy = gh - pvy  # flip y so top row is y=0
        out = np.zeros((gh + 1, gw + 1, 4), dtype=float)
        # Sanity check: only draw samples that fall within the raster bounds
        vx = pvx[msk]
        vy = pvy[msk]
        valid = (vx >= 0) & (vx <= gw) & (vy >= 0) & (vy <= gh)
        out[vy[valid], vx[valid]] = rgba[valid]
        image_uint8 = (out * 255).astype(np.uint8)
        Image.fromarray(image_uint8).save(f'output/sh{i:02}.png')
