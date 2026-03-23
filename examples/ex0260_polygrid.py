# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Given a polygon in Latitude/Longitude points,
and a hex_layer, generate the hex-grid for that hex_layer, along with ancestry.
It will extract any common prefix, and plot the hex-grid and ancestry along with hex labels.

Last Tested
13 Mar 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (working well)
"""
import time

from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from hhg9 import Points, Registrar
from hhg9.h9.addressing import tail_unpack_reversible, tail_key_from_reversible
from hhg9.h9.polygon import hex_reduce, hex_parents, ctr_from_pars
from hhg9.h9.grid import poly_net_field, hex_verts_in_noct
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

def plot_hex(pts_list, prefix='', name='hex', ctrs=None, labels=None, values=None, lut=None, nodata=0):
    """Plot hex polygons.
    Args:
        pts: Points containing hex vertices, shaped (H*6,2) in `pts.coords`.
        name: output filename suffix.
        values: optional per-hex categorical values (H,) (e.g. NLCD class codes).
        lut: optional uint8 LUT of shape (256,3) mapping class -> RGB.
        nodata: class code treated as nodata (alpha=0). Set to None to disable.
    """
    xmin, ymin, xmax, ymax = (2.401, 1.906, 2.463, 1.941)
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
    ax.text(xmax-pad_x, ymin+pad_y, f'Prefix:{prefix}', color='k', fontsize=40, ha='right', va='center')
    layers = len(pts_list) - 1
    for i, pts in enumerate(pts_list):
        lw = 0.2 * ((layers-i)+1)
        polys = np.asarray(pts.coords, dtype=np.float64).reshape(-1, 6, 2)
        h = polys.shape[0]
        if h == 0:
            continue
        face_colors = '#33994420'
        edge_colors = 'black' if i == layers else '#ffffffa0'
        ax.add_collection(PolyCollection(polys, linewidths=lw, facecolors=face_colors, edgecolors=edge_colors))
    if ctrs is not None:
        ctrs = ctrs.coords if isinstance(ctrs, Points) else ctrs
        if labels is None:
            ax.scatter(ctrs[:, 0], ctrs[:, 1], ec='none', marker='.', s=30)
        else:
            for pt, label in zip(ctrs, labels):
                ax.text(pt[0], pt[1], label, color='k', fontsize=4, ha='center', va='center')
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    fig.savefig(f"output/ex0260_hx_{name}.png", dpi=200)
    print(f"fig saved at output/ex0260_hx_{name}.png")


def plot_ctr(pts, name='hex', labels=None):
    """Plot hex centroids as dots with optional text labels."""
    ctrs = pts.coords
    xmin, ymin = float(np.min(ctrs[:, 0])), float(np.min(ctrs[:, 1]))
    xmax, ymax = float(np.max(ctrs[:, 0])), float(np.max(ctrs[:, 1]))
    ratio = np.abs(xmax - xmin) / np.abs(ymax - ymin) if ymax > ymin else 1.0
    fig = plt.figure(figsize=(ratio * 10, 10), dpi=400, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111)
    pad_x = 0.01 * (xmax - xmin) if xmax > xmin else 1.0
    pad_y = 0.01 * (ymax - ymin) if ymax > ymin else 1.0
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    if labels is None:
        ax.scatter(ctrs[:, 0], ctrs[:, 1], ec='none', marker='.', s=30)
    else:
        for pt, label in zip(ctrs, labels):
            ax.text(pt[0], pt[1], label, color='k', fontsize=4, ha='center', va='center')
    fig.savefig(f"output/ex0260_ctr_{name}.png", dpi=200)
    print(f"fig saved at output/ex0260_ctr_{name}.png")


if __name__ == '__main__':
    rg = Registrar()
    h9f = OctahedralH9(rg)  # formatter.

    # Let's ensure that hhg9 and gdal/osr are speaking the same language!
    g_gcd = rg.domain('g_gcd')
    c_ell = rg.domain('c_ell')  # EPSG:4978
    b_oct = rg.domain('b_oct')
    b_oct.register_format(h9f)

    c_oct = rg.domain('c_oct')
    n_oct = rg.domain(f'n_oct:butterfly')

    # [birmingham-bristol-london-belgium] polygon in gcd
    b_dll = np.array([
        [51.084464524954795, -3.4462861844929585],
        [52.68223344923537, -2.189458562294672],
        [52.60858425701964, -1.432594162686184],
        [51.648685168766754, -2.0537941887799422],
        [51.52446468552721, -0.8970768988122543],
        [51.82555423461282, -0.747132064927554],
        [51.684114370958646, 0.5309691381848917],
        [51.40435883726121, 0.47384729670500597],
        [51.29190823165293, 5.4859054058927266],
        [50.72448838777691, 5.4044368907702074],
    ])
    pll = Points(b_dll, g_gcd)
    # pnt = rg.project(pll, [g_gcd, b_oct])
    ntt = rg.project(pll, [g_gcd, b_oct, n_oct])

    # threshold = 2  # min L+1 subsamples inside polygon (1..3); 1=include edge hexes
    for layer in [6]:  # range(6, 8):
        scn = poly_net_field(ntt, layer)
        b_pts = rg.project(scn, [n_oct, b_oct])

        # Drop points that landed outside all net faces
        valid = b_pts.oid != 255
        b_pts = Points(b_pts.coords[valid], domain=b_oct, oid=b_pts.oid[valid])
        # Bin to L-layer parents, apply a threshold, compute centroids directly
        hex_num, hex_v, hex_inv, _ = hex_reduce(b_pts, layer)
        counts = np.bincount(hex_inv, minlength=hex_num)
        for threshold in [1, 2, 3]:
            # Oversample at L+1 to get candidates inside the polygon
            keep = counts >= threshold
            print(f'Layer {layer}: {keep.sum()} hexes (threshold={threshold})')
            hex_v_k = hex_v[keep]
            tails = tail_key_from_reversible(hex_v_k[:, -1])
            hex_par, hex_oid, scale = hex_parents(b_oct, hex_v_k, int(keep.sum()))  # owners of the points.
            ctr = ctr_from_pars(b_oct, hex_par, hex_oid, scale, tails)

            hex_adr = hex_v_k.copy()
            hex_adr[:, -1] = tails >> 4
            matches = np.all(hex_adr == hex_adr[0, :], axis=0)
            prefix_len = np.argmin(matches) if not np.all(matches) else hex_adr.shape[1]
            prefix_str = ''.join([f'{a:0x}' for a in hex_adr[0, :prefix_len]])
            sub_adr = hex_adr[:, prefix_len:]
            anc_levels = range(prefix_len, layer)  # or range(max(k, layer-2), layer) if too many
            verts = []
            for anc_level in anc_levels:
                anc_num, anc_v, _, _ = hex_reduce(ctr, anc_level)
                anc_par, anc_oid, anc_scale = hex_parents(b_oct, anc_v, anc_num, anc_level)
                anc_xpm, anc_xc2, _, _ = tail_unpack_reversible(anc_v[:, -1])
                verts.append(hex_verts_in_noct(anc_par, anc_oid, anc_xpm, anc_xc2, anc_scale, n_oct))

            hd = np.array(list(''.join([f'{a:0x}' for a in sub_adr[i]]) for i in range(sub_adr.shape[0])))

            # Build hex vertices directly in n_oct via per-face rigid transform.
            xpm, xc2, _, _ = tail_unpack_reversible(hex_v_k[:, -1])
            verts.append(hex_verts_in_noct(hex_par, hex_oid, xpm, xc2, scale, n_oct))
            ctr_n = rg.project(ctr, [b_oct, n_oct])
            plot_hex(verts, prefix_str, f'{layer}_{threshold}', ctrs=ctr_n.coords, labels=hd)
