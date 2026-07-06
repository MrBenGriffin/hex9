# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
For a given layer, compose a reference address for each
hexagon in that layer - generate the set of hexagons at that layer
and display on the globe.

Last Tested
16 Jun 2026 0.1.3a0 (passed) 1321.7s
13 Mar 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed)
25 Nov 2025 (passed)
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from matplotlib import pyplot as plt, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area, ellipsoid_area_wgs84
from hhg9.h9 import H9_RA, H9O, H9K
from hhg9.h9.classifier import location
from hhg9.h9.protocols import BaryLoc
from hhg9.h9.region import regions_xy
from hhg9.h9.polygon import hex_poly_layer
import matplotlib as mpl


def chain_generator(initial_seed, depth, props=H9_RA.props, modes=H9_RA.modes):
    """Generator for comprehensive region chain generation"""
    def _recurse(current_chain):  # Recursive Closure
        if len(current_chain) - 2 == depth:  # Stop condition
            yield current_chain
            return
        seed = current_chain[-1]  # Get the current seed (last element)
        children = props[modes[seed]].flatten()
        for child in children:  # Iterate and dive deeper
            yield from _recurse(current_chain + [child])  # Create new list
    yield from _recurse([initial_seed])  # yield from closure.


def rgba_from(arr: np.ndarray, cmap_name: str = "plasma", norm=None, alpha: float = 1.0):
    """Return RGBA array from a 1D array of values.

    Parameters
    ----------
    arr : array-like
        Scalar values to map to colours.
    cmap_name : str
        Name of the Matplotlib colormap.
    norm : matplotlib.colors.Normalize or None
        Normalization object. If None, a simple Normalize based on arr
        is constructed.
    alpha : float
        Global alpha to apply to the colours.
    """
    arr = np.asarray(arr, dtype=float)
    if norm is None:
        norm = colors.Normalize(vmin=arr.min(), vmax=arr.max())

    base_cmap = plt.get_cmap(cmap_name)

    # If the colormap exposes a `.colors` table (ListedColormap), build a
    # new ListedColormap with an explicit alpha channel so we don't mutate
    # the global colormap in-place.
    if hasattr(base_cmap, "colors"):
        base_colors = np.asarray(base_cmap.colors)
        if base_colors.shape[1] == 3:
            # Append alpha channel
            alpha_col = np.full((base_colors.shape[0], 1), alpha, dtype=float)
            rgba_colors = np.concatenate([base_colors, alpha_col], axis=1)
        else:
            rgba_colors = base_colors.copy()
            rgba_colors[:, 3] = alpha
        cmap = colors.ListedColormap(rgba_colors, name=base_cmap.name + "_with_alpha")
    else:
        # For continuous maps, just use the base cmap and apply alpha after
        cmap = base_cmap

    rgba = cmap(norm(arr))

    # If the colormap didn't already encode alpha, enforce it here.
    if rgba.shape[1] == 4:
        rgba[:, 3] = alpha

    return rgba, norm


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


def snow_globe(arr: Points, poly_len: int = 6, scores=None, layers='x', lim_pct: float = 5.0):
    """Display a 3D point cloud using matplotlib"""
    mpl.rcParams['path.simplify'] = False
    fig = plt.figure(figsize=(15, 15), dpi=300, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=90, azim=0)
    axis = mplot_ax_vector(ax)
    all_polys = arr.coords.reshape(-1, poly_len, 3)
    mask = cull_backface(all_polys, axis)
    front = all_polys[mask]
    # front = all_polys
    # scores arrive as ℓ = log(area/ideal); show as % deviation on a FIXED
    # symmetric scale so renders are directly comparable (worst L5 hex ≈
    # +4.8%, inside ±5%). Cells beyond ±lim_pct saturate to full red/blue.
    pct = np.expm1(scores) * 100.0
    norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-lim_pct, vmax=+lim_pct)
    cmap_name = 'RdBu_r'
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    # FILL_ALPHA lightens the fill against the white ground so the stroke
    # stays differentiable from near-zero (pale) cells. Tune to taste.
    FILL_ALPHA = 0.60
    rgba, _norm = rgba_from(pct, cmap_name, norm=norm, alpha=FILL_ALPHA)
    ax.set_proj_type('ortho')  # FOV = 0 deg
    if True:
        # ±0.75e6 m (~1500 km across): close enough that the vertex bloom,
        # its rings, and the seam rays fill the frame — at ±2e6 the ±1%-capped
        # palette left a tiny cross in a vast near-white field.
        # NB all three spans must be EQUAL: set_aspect('equal') scales the view
        # to the largest axis range, so a wide zlim silently zooms back out.
        # Vertex placed ~25% from the frame edge (not centred): the structure
        # is 4-fold symmetric, so the composition gives the long run of one
        # seam ray room to grade out to the quiet interior.
        ax.set_xlim(-1.48e+6, 0.02e+6)   # moves up / down (add to go up)
        ax.set_ylim(-0.005e+6, 1.455e+6)  # moves left / right (add to go left)
        ax.set_zlim(-0.60e+6, 0.20e+6)
    polys = [p for p in front]

    # Stroke must stay legible against the fill or the tessellation vanishes:
    # at 400 dpi an L5 hex is ~15 px across, so 0.02 pt (~0.1 px) edges are
    # invisible and the near-white interior reads as a structureless field.
    # Semi-transparent grey edges at ~0.5 px keep the lattice visible without
    # blackening dense regions; fill stays opaque — the colour is the data.
    collection = Poly3DCollection(polys, ec=(0.15, 0.15, 0.15, 1.0),
                                  facecolors=rgba[mask], linewidth=0.125)
    ax.add_collection(collection)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.0,
                        fraction=0.025, aspect=40, shrink=0.5)
    cbar.set_label(f'% area deviation from ideal (scale clipped at ±{lim_pct:g}%)',
                   fontsize=10)
    lil, big = float(np.min(pct)), float(np.max(pct))
    ax.title.set_text(f'L{layers} area deviation   min {lil:+.2f}%   max {big:+.2f}%')
    plt.tight_layout()
    fig.savefig(f"output/ex0081wau_{layers}.png", dpi=400)
    print(f'fig saved at output/ex0081wau_{layers}.png')


def get_data(reg: Registrar, depth, mode=None):
    """Load up global sample data"""
    # grab generation for given depth
    b_oct = reg.domain('b_oct')
    all_rgn = [   # these are 0..11
        list(chain_generator(H9_RA.proto[0], depth)),
        list(chain_generator(H9_RA.proto[1], depth))
    ]
    rgn = H9_RA.rid2cell[np.array(all_rgn)]  # cell addresses.
    sides = []
    for oc in range(8):  # all octants
        mo = H9O.oid_mo[oc]
        if mode is not None and mo != mode:
            continue
        rgc = rgn[mo]
        xym = regions_xy(rgc)
        xy = xym[:, :-1]
        sides.append(Points(xy, b_oct, oc))
    result = Points.concat(sides)
    return result


def hexify(reg: Registrar, b_pts: Points, warp_m=None, layers: int = 4):
    """
    Find hexagons for data, and display on a 'globe'.
    """
    # Geometry and geodesic areas are view-independent and slow (~10 min at
    # L5); cache them so stroke/zoom iterations re-render in seconds.
    cache = Path(f'output/ex0081_cache_{layers}.npz')
    if cache.exists():
        z = np.load(cache)
        c_pts = SimpleNamespace(coords=z['coords'])
        score = z['score']
    else:
        pts, pops = hex_poly_layer(b_pts, layers)

        # Now calculate their area as a metric. (ignore pops).
        gm2 = ellipsoid_area_wgs84() # total surface area of WGS-84 (m²) about 510_065_621_724km
        bins = 12*(9**layers)          # number of hexes at this layer
        w_area_m2_mean = gm2/bins    # ideal equal-area per hex
        h_pts = pts.copy()
        c_pts = reg.project(h_pts, ['b_oct', 'c_oct', 'c_ell'])  # use bary.
        g_pts = reg.project(h_pts, ['b_oct', 'g_gcd'])
        w_area_m2 = wgs84_area(reg, g_pts)  # default value is 6
        w_adj = np.abs(w_area_m2 / w_area_m2_mean) + 1e-12
        score = np.log(w_adj)  # authalic log-density ℓ
        np.savez_compressed(cache, coords=c_pts.coords, score=score)
    snow_globe(c_pts, 6, score, f'{layers}', lim_pct=0.25)


if __name__ == '__main__':
    depth = 5  # 0,...5 √
    rg = Registrar()  # Manage Domains & Projections
    b_oct = rg.domain('b_oct')
    ellipsoid = 'WGS84'
    warps = ['l5_warp_data']
    for warp in warps:
        warp_path = Path(f'../hhg9/data/{ellipsoid}_{warp}.npz')
        b_oct.set_warp(warp_path)
        data = get_data(rg, depth)  # should be 8*9**depth  (eg, depth=0: 72 points, 9 points on each face, and six points in each hexagon)
        hexify(rg, data, layers=depth)


