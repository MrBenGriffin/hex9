# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Given a polygon in Latitude/Longitude points,
and a hex_layer, generate the hex-grid for that hex_layer
and draw it in b_oct.

Last Tested
06 Mar 2026 0.1.1a1 (better)
26 Dec 2025 0.1.0a4 (working well)
"""
import time

from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from hhg9 import Points, Registrar
from hhg9.h9 import H9K
from hhg9.h9.addressing import hex_str_encode, TailStyle
from hhg9.h9.polygon import hex_poly_layer
from geographiclib.geodesic import Geodesic
from contextlib import contextmanager
from hhg9.formats import OctahedralH9


geod = Geodesic.WGS84


@contextmanager
def time_block(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{label}: {end - start:.4f}s")


def plot_pts(pts, layer):
    """Draw points and save"""
    xmax = 2.4443197374349794
    xmin = 2.3977612662457415
    ymax = 1.9353993029397951
    ymin = 1.9101987911827665
    ratio = np.abs(xmax - xmin) / np.abs(ymax - ymin)
    fig = plt.figure(figsize=(ratio * 10, 10), dpi=200, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111)  #

    # xmin = float(np.min(pts.coords[:, 0]))
    # xmax = float(np.max(pts.coords[:, 0]))
    # ymin = float(np.min(pts.coords[:, 1]))
    # ymax = float(np.max(pts.coords[:, 1]))
    pad_x = 0.01 * (xmax - xmin) if xmax > xmin else 1.0
    pad_y = 0.01 * (ymax - ymin) if ymax > ymin else 1.0
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    ax.scatter(pts.coords[:, 0], pts.coords[:, 1], ec='none', marker='.', s=10)
    fig.savefig(f"output/ex0260_pts_{layer}.png", dpi=200)


def plot_hex(pts, name='hex', ctrs=None, labels=None, values=None, lut=None, nodata=0):
    """Plot hex polygons.
    Args:
        pts: Points containing hex vertices, shaped (H*6,2) in `pts.coords`.
        name: output filename suffix.
        values: optional per-hex categorical values (H,) (e.g. NLCD class codes).
        lut: optional uint8 LUT of shape (256,3) mapping class -> RGB.
        nodata: class code treated as nodata (alpha=0). Set to None to disable.
    """
    # xmax = 2.4443197374349794
    # xmin = 2.3977612662457415
    # ymax = 1.9353993029397951
    # ymin = 1.9101987911827665
    xmin, ymin = float(np.min(pts.coords[:, 0])), float(np.min(pts.coords[:, 1]))
    xmax, ymax = float(np.max(pts.coords[:, 0])), float(np.max(pts.coords[:, 1]))
    ratio = np.abs(xmax - xmin) / np.abs(ymax - ymin)
    fig = plt.figure(figsize=(ratio * 10, 10), dpi=400, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111)
    pad_x = 0.01 * (xmax - xmin) if xmax > xmin else 1.0
    pad_y = 0.01 * (ymax - ymin) if ymax > ymin else 1.0
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    polys = np.asarray(pts.coords, dtype=np.float64).reshape(-1, 6, 2)
    h = polys.shape[0]
    if h == 0:
        print(f"plot_hex: no polygons for {name}")
        return

    face_colors = '#33994422'
    edge_colors = 'black'
    if (values is not None) and (lut is not None):
        v = np.asarray(values)
        if v.shape != (h,):
            raise ValueError(f"plot_hex: values must have shape ({h},), got {v.shape}")
        # Map class codes -> RGBA in [0,1]
        codes = v.astype(np.int32, copy=False)
        codes = np.clip(codes, 0, lut.shape[0] - 1)
        rgb = lut[codes].astype(np.float64) / 255.0  # (H,3)
        alpha = np.ones((h, 1), dtype=np.float64)
        if nodata is not None:
            alpha[codes == int(nodata)] = 0.0
        face_colors = np.concatenate([rgb, alpha], axis=1)  # (H,4)
        edge_colors = 'none'  # cleaner for categorical fills; change to 'black' if you want outlines

    ax.add_collection(PolyCollection(polys, linewidths=0.2, facecolors=face_colors, edgecolors=edge_colors))
    if ctrs is not None:
        ctrs = ctrs.coords if isinstance(ctrs, Points) else ctrs
        if labels is None:
            ax.scatter(ctrs[:, 0], ctrs[:, 1], ec='none', marker='.', s=30)
        else:
            for pt, label in zip(ctrs, labels):
                ax.text(pt[0], pt[1], label, color='k', fontsize=4, ha='center', va='center')
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    fig.savefig(f"output/ex0261_hx_{name}.png", dpi=200)
    print(f"fig saved at output/ex0261_hx_{name}.png")


def calculate_intersections(poly_grid, v):
    """
    Finds u-coordinates where the horizontal line at v crosses polygon edges.
    poly_grid: (m, 2) array of polygon vertices in scaled UV space.
    v: The current scanline (integer row index).
    """
    # 1. Define segments (p1 to p2)
    p1 = poly_grid
    p2 = np.roll(poly_grid, -1, axis=0)

    # 2. Find edges that cross the scanline v
    # We use (p1y <= v < p2y) or (p2y <= v < p1y) to handle vertices correctly
    # and avoid double-counting horizontal edges.
    mask = ((p1[:, 1] <= v) & (p2[:, 1] > v)) | ((p2[:, 1] <= v) & (p1[:, 1] > v))

    if not np.any(mask):
        return np.array([])

    # 3. Filter segments
    e1 = p1[mask]
    e2 = p2[mask]

    # 4. Linear Interpolation to find the u-coordinate
    # u = x1 + (v - y1) * (x2 - x1) / (y2 - y1)
    u_ints = e1[:, 0] + (v - e1[:, 1]) * (e2[:, 0] - e1[:, 0]) / (e2[:, 1] - e1[:, 1])

    # 5. Return sorted intersections for pairing
    return np.sort(u_ints)


def scanline_h9_sheet(poly_n, level):
    # sn is the grid spacing factor
    # level + 1 will generate 9 points per, which helps with polygon edges.

    sn = H9K.radical.W * (3 ** -(level + 1))

    # 1. Scale polygon to integer grid space (UV Skew Space)
    x, y = poly_n.coords[:, 0], poly_n.coords[:, 1]
    # Standard Axial/Skew transform for hexagonal grids
    v_skew = (y * 2 / np.sqrt(3)) / sn
    u_skew = (x / sn) - (v_skew / 2)

    poly_grid = np.stack([u_skew, v_skew], axis=1)

    # 2. Get Row Range
    v_min = int(np.floor(poly_grid[:, 1].min()))
    v_max = int(np.ceil(poly_grid[:, 1].max()))

    u_acc, v_acc = [], []

    for v in range(v_min, v_max + 1):
        u_ints = calculate_intersections(poly_grid, v)

        # Standard even-odd fill pairing
        for i in range(0, len(u_ints), 2):
            if i + 1 >= len(u_ints): break
            u_start = int(np.ceil(u_ints[i]))
            u_end = int(np.floor(u_ints[i + 1]))

            if u_end >= u_start:
                u_range = np.arange(u_start, u_end + 1)
                u_acc.append(u_range)
                v_acc.append(np.full_like(u_range, v))

    if not u_acc:
        # Fallback to centroid if no points found
        avg = np.mean(poly_grid, axis=0)
        u_final, v_final = np.array([np.round(avg[0])]), np.array([np.round(avg[1])])
    else:
        u_final = np.concatenate(u_acc)
        v_final = np.concatenate(v_acc)

    # 3. CONVERT BACK TO N_OCT (The Critical Part)
    # Using the inverse of the skew transform:
    # y = v * sn * sqrt(3) / 2
    # x = (u + v/2) * sn

    y_n = (v_final - 0.333) * sn * (np.sqrt(3) / 2)
    x_n = (u_final + (v_final / 2)) * sn

    uv = np.stack([x_n, y_n], axis=1)
    pts = Points(uv, domain=poly_n.domain)
    result = poly_n.domain.filter(pts)
    print(f'Layer: {level}; Points: {len(result)}')
    return result


if __name__ == '__main__':
    rg = Registrar()
    h9f = OctahedralH9(rg)  # formatter.

    # Let's ensure that hhg9 and gdal/osr are speaking the same language!
    g_gcd = rg.domain('g_gcd')
    c_ell = rg.domain('c_ell')  # EPSG:4978
    b_oct = rg.domain('b_oct')
    b_oct.register_format(h9f)

    c_oct = rg.domain('c_oct')
    n_oct = rg.domain(f'n_oct:rhombus')

    # birmingham-[bristol-london]-dover motorway polygon in gcd
    b_dll = np.array([
        [51.484464524954795, -3.0462861844929585],
        [52.68223344923537, -2.189458562294672],
        [52.60858425701964, -1.432594162686184],
        [51.648685168766754, -2.0537941887799422],
        [51.52446468552721, -0.8970768988122543],
        [51.82555423461282, -0.747132064927554],
        [51.684114370958646, 0.5309691381848917],
        [51.40435883726121, 0.47384729670500597],
        [51.29190823165293, 5.4859054058927266],
        [50.95448838777691, 5.4044368907702074],
    ])
    pll = Points(b_dll, g_gcd)
    pnt = rg.project(pll, [g_gcd, b_oct])
    ntt = rg.project(pnt, [b_oct, n_oct])

    for layer in [5]:  # range(6, 8):
        with time_block(f"{layer} scanline_h9_sheet"):
            scn = scanline_h9_sheet(ntt, layer)
            plot_pts(scn, layer)
            b_pts = rg.project(scn, [n_oct, b_oct])

            # Drop points that landed outside all net faces: pt_face() returns (0,0,0) for them,
            # which _project_composites silently maps to octant 0 with wrong coordinates.
            valid = np.any(b_pts.components != 0, axis=-1)
            b_pts = Points(b_pts.coords[valid], domain=b_oct, components=b_pts.components[valid])

            hx_pts, _ = hex_poly_layer(b_pts, layers=layer)
            h_pts = rg.project(hx_pts, [b_oct, n_oct])
            # do labels.
            n_ctr = np.mean(h_pts.coords.reshape(-1, 6, 2), axis=1)
            npts = Points(n_ctr, n_oct)
            c_pt = rg.project(npts, [n_oct, b_oct])
            hd = hex_str_encode(c_pt, layer, tail_style=TailStyle.key)
            plot_hex(h_pts, f'{layer}', ctrs=n_ctr, labels=hd)
