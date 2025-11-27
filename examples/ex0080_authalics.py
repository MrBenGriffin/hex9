# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
For a given layer, compose a reference address for each
hexagon in that layer - generate the set of hexagons at that layer
and display on the globe.
Last Tested 25 November 2025 √
"""
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt, colors
from matplotlib.collections import PolyCollection
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
from hhg9.algorithms.pickers import gcd_fibonacci
# from experimental.algorithms.warp import Warper
from hhg9.h9.polygon import hex_layer as hex_layer, hh_layer
import matplotlib as mpl



def mplot_ax_vector(ax):
    """mplot3d uses azim around z and elev from xy-plane"""
    az = np.deg2rad(ax.azim)
    el = np.deg2rad(ax.elev)
    return np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])


def cull_backface(arr, axis):
    """back-face culling"""
    centroids = arr.mean(axis=1)
    sides = centroids @ axis
    return sides >= 0


def snow_globe(arr: Points, poly_len: int = 6, pop=None, layers='x'):
    """Display a 3D point cloud using matplotlib"""
    mpl.rcParams['path.simplify'] = False
    fig = plt.figure(figsize=(15, 15), dpi=200, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=40)
    axis = mplot_ax_vector(ax)
    all_polys = arr.coords.reshape(-1, poly_len, 3)
    mask = cull_backface(all_polys, axis)
    front = all_polys[mask]
    authalic_error = np.mean(np.abs(pop))
    pops = pop[mask]
    ax.set_proj_type('ortho')  # FOV = 0 deg
    if True:
        ax.set_xlim(-4e+6, 4e+6)  # fill the area with the map.
        ax.set_ylim(-4e+6, 4e+6)
        ax.set_zlim(-4e+6, 4e+6)
    polys = [p for p in front]
    col = 'none'
    if pops is not None:
        v_min = np.min(pops)
        v_max = np.max(pops)
        norm = colors.Normalize(vmin=v_min, vmax=v_max)
        cmap = plt.get_cmap('plasma')
        al = np.full((cmap.N,), 0.5)
        cmap.colors = np.column_stack([cmap.colors, al])
        col = cmap(norm(pops))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # no data needed
        # plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)

    collection = Poly3DCollection(polys, ec='black', facecolors=col, alpha=0.9, linewidth=0.05)
    ax.add_collection(collection)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    ax.title.set_text(f'Authalic Error: {authalic_error:.2f}')
    plt.tight_layout()
    fig.savefig(f"output/auth_{layers}.png", dpi=400)
    print(f'fig saved at output/auth_{layers}.png')


def show_globe(pts: Points, poly_len: int = 6, count=None):
    """Display GCD points (as hexagons) onto a cartopy global view"""
    globe = ccrs.Globe(ellipse='WGS84')
    proj = ccrs.Orthographic(central_longitude=40, central_latitude=30, globe=globe)

    """Display GCD points on the globe"""
    polys = pts.coords.reshape((-1, poly_len, 2))
    p_xy = polys[..., :, [1, 0]]  # np.swapaxes(polys, -1, -2)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150, frameon=False, subplot_kw={'projection': proj})
    ax.add_feature(cfeature.LAND, facecolor='gray')
    ax.coastlines(resolution='110m', linewidth=0.6)
    ax.set_global()
    col = 'none'
    if count is not None:
        v_min = np.min(count)
        v_max = np.max(count)
        norm = colors.Normalize(vmin=v_min, vmax=v_max)
        cmap = plt.get_cmap('plasma')
        col = cmap(norm(count))
    collection = PolyCollection(
        p_xy,
        transform=ccrs.Geodetic(),  # transform from ll to pc in Cartopy.
        facecolors=col,
        alpha=0.2,
        ec=(0, 0, 0, 0.8),
        linewidth=1,
        antialiaseds=True,
    )
    ax.add_collection(collection)
    plt.tight_layout()
    plt.show()


def get_data(reg: Registrar, force: bool = False, k_pts: int = 200, seed: int = 42):
    """Load up global sample data"""
    rng = np.random.default_rng()
    np.random.seed(seed)
    c_oct = reg.domain('c_oct')
    cache = Path(f'src/{seed}_{k_pts}k_fib.npy')
    if not cache.exists() or force:
        r_ll = gcd_fibonacci(k_pts*1000)
        g_pts = Points(r_ll, 'g_gcd')
        b_pts = reg.project(g_pts, ['g_gcd', 'b_oct'])
        o_pts = reg.project(b_pts, ['b_oct', c_oct])
        np.save(cache, o_pts.coords)
    else:
        o_data = np.load(cache)
        # rng.shuffle(o_data)
        o_pts = Points(o_data, c_oct)
        b_pts = reg.project(o_pts, [c_oct, 'b_oct'])
    return b_pts


def hexify(reg: Registrar, b_pts: Points, layers: int = 4):
    """
    Find hexagons for data, and display on a 'globe'.
    """
    gm2 = 510_065_621_724_154.6  # total surface area of WGS-84 (m²)
    bins = 12*9**layers          # number of hexes at this layer
    bin_area = gm2/bins          # ideal equal-area per hex
    b_oct = reg.domain('b_oct')

    x_hex, count, inv, oc = hex_layer(b_pts, layers)  # 6 pts
    accounted = count.sum()
    if accounted != len(b_pts):
        tested = len(b_pts)
        print(f'WARNING: {abs(tested-accounted)} samples not accounted for')
    if bins != count.shape[0]:
        print(f'WARNING: {abs(bins-count.shape[0])} hexes missing')

    f_xy, f_oc = x_hex.reshape(-1, 2), oc.reshape(-1)
    pts = Points(f_xy, domain=b_pts.domain, components=f_oc)
    g_pts = reg.project(pts, ['b_oct', 'g_gcd'])
    areas = wgs84_area(reg, g_pts)  # default value is 6
    score = (areas / bin_area) - 1
    pv = 6  # polygon vertices. hex_layer_v1
    b_hxp = x_hex.reshape(-1, 2)
    b_oc = oc.reshape(-1)
    b_hex = Points(b_hxp, domain=b_oct, components=b_oc)
    c_pts = reg.project(b_hex, ['b_oct', 'c_oct', 'c_ell'])  # use bary.
    snow_globe(c_pts, pv, score, layers)


def half_hexify(reg: Registrar, b_pts: Points, layers: int = 4):
    """
    Find hexagons for data, and display on a 'globe'.
    """
    bins = 24*9**layers
    samples = len(b_pts)
    expected = samples/bins
    x_hex, count, inv, oc = hh_layer(b_pts, layers)  # 4 pts
    b_hxp = x_hex.reshape(-1, 2)
    b_oc = oc.reshape(-1)
    pv = 4  # polygon vertices.
    b_oct = reg.domain('b_oct')
    score = np.abs(count - expected)
    b_hex = Points(b_hxp, domain=b_oct, components=b_oc)
    c_pts = reg.project(b_hex, ['b_oct', 'c_oct', 'c_ell'])  # use bary.
    snow_globe(c_pts, pv, score)


def verify_hex_by_id(rg):
    """For each octant id f (0..7), l0hex_by_id[f] must
    be a permutation of {0,1,2} mapping the first step’s c2 to the
    correct “root hex” digit on that face."""
    b_oct = rg.domain('b_oct')
    hxd = {}
    for oid in range(8):
        hx = b_oct.l0hex_by_id[oid]
        mo = int(b_oct.oid_mo[oid])
        edge = b_oct.edges_by_id[oid]
        for i, h in enumerate(hx):
            k = int(h)
            if k not in hxd:
                hxd[k] = {}
            hd = hxd[k]
            if mo not in hd:
                hd[mo] = {}
            hm = hd[mo]
            hm[oid] = [i, edge[i]]


if __name__ == '__main__':
    depth = 3
    rg = Registrar()  # Manage Domains & Projections
    data = get_data(rg, k_pts=200)  # 200, 2000, 8000
    hexify(rg, data, layers=depth)

