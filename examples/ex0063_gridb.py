# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Further breakdown for warp analysis (currently experimental/and put to one side).
All renders follow the triangular sub-grid.
Another way of calculating this has been used with hexagons in ex0080_authalics
16 Jun 2026 0.1.3a0 (passed) 168.8s
13 Mar 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed)
25 Nov 2025 (passed)
"""
import numpy as np
from matplotlib import pyplot as plt, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area, enu_planar_polygon_area
from hhg9.h9 import H9O
from hhg9.h9.polygon import tri_grid
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


def snow_globe(arr: Points, poly_len: int = 6, pop=None, name='grid'):
    """Display a 3D point cloud using matplotlib"""
    mpl.rcParams['path.simplify'] = False
    fig = plt.figure(figsize=(15, 15), dpi=400, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')  # FOV = 0 deg
    ax.view_init(elev=30, azim=40)
    axis = mplot_ax_vector(ax)
    all_polys = arr.coords.reshape(-1, poly_len, 3)
    mask = cull_backface(all_polys, axis)
    front = all_polys[mask]
    rx = front.reshape(-1, 3)
    x_min, x_max = rx[:, 0].min(), rx[:, 0].max()
    y_min, y_max = rx[:, 1].min(), rx[:, 1].max()
    z_min, z_max = rx[:, 2].min(), rx[:, 2].max()
    if True:
        ax.set_xlim(x_min, x_max)  # fill the area with the map.
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
    polys = [p for p in front]
    if pop is not None:
        authalic_error = np.mean(np.abs(pop))
        pops = pop[mask]
        col_map_name = 'RdBu_r'
        max_abs = float(np.max(np.abs(pops)))
        norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=+max_abs)
        sm = plt.cm.ScalarMappable(cmap=col_map_name, norm=norm)
        sm.set_array([])

        # Map authalicity values (pops) to colours using the symmetric TwoSlopeNorm
        cmap = mpl.colormaps[col_map_name]
        facecols = cmap(norm(pops))

        # Optional colourbar (uncomment if/when needed)
        plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)

        collection = Poly3DCollection(
            polys,
            ec=(0, 0, 0, 0.3),
            facecolors=facecols,
            alpha=1.0,
            linewidth=0.05,
        )
        ax.add_collection(collection)
        ax.title.set_text(f'Authalic Error: {authalic_error:.3f}')
    else:
        collection = Poly3DCollection(polys, ec='black', alpha=1.0, linewidth=0.05)
        ax.add_collection(collection)

    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"output/ex0063b_{name}_l{depth}.png", dpi=400)
    plt.close(fig)
    print(f'file saved at output/ex0063b_{name}_l{depth}.png')


def get_data(reg: Registrar, layer=3, octant_id=None):
    """Load up global sample data"""
    tg0 = tri_grid(layer, 0).reshape([-1, 2])  # triangle polygons.
    tg1 = tri_grid(layer, 1).reshape([-1, 2])  # triangle polygons.
    tgx = [tg0, tg1]
    b_oct = reg.domain('b_oct')
    if octant_id is None:
        repo = []
        for octant in b_oct.signs.keys():
            o_id = octant_id
            mode = b_oct.oid_mo[o_id]
            repo.append(Points(tgx[mode].copy(), b_oct, components=octant))
        return Points.concat(repo)
    else:
        o_id = octant_id
        mode = H9O.oid_mo[o_id]
        return Points(tgx[mode], b_oct, oid=o_id)


def octa_layer_stats(l: int, r=6371000.0):
    """
    Return triangle/vertex/edge counts and safe spacing estimates for a global
    octahedral grid with hex_layer indexing based on emergent hexagons.

    Parameters
    ----------
    l : int
        Layer index (0-based; L=0 → 72 triangles → 12 hexes)
    r : float
        Mean Earth radius in metres (default 6,371 km)

    Returns
    -------
    dict with keys:
        s        : subdivisions per octa edge (=3**(L+1))
        triangles: total number of triangular cells
        vertices : global vertex count
        edges    : global edge count
        d_geo    : great-circle edge spacing (m)
        d_chord  : equivalent chord spacing (m)
        tol_m    : recommended dedupe tolerance (m)
    """
    s = 3 ** (l + 1)  # splits per octa edge
    tri = 8 * s * s  # 8 faces * s²
    vert = 4 * s * s + 2  # from combinatorial derivation
    edge = 12 * s * s  # 3/2 * T, or 3V - 6
    d_geo = (np.pi * r) / (2 * s)  # ~90°/s great-circle
    d_chord = 2 * r * np.sin((np.pi / (4 * s)))  # exact chord
    tol_m = 0.4 * d_geo  # conservative dedupe tolerance
    return dict(L=l, s=s, triangles=tri, vertices=vert, edges=edge,
                d_geo=d_geo, d_chord=d_chord, tol_m=tol_m)


if __name__ == '__main__':
    """
    Triangular grid will be 9 triangles per octant at hex_layer 0
    At each subsequent hex_layer, the number of triangles will increase by 9 per triangle.
    So the number of triangles will be 8*9**(hex_layer+1)
    """
    rg = Registrar()  # Manage Domains & Projections
    octant = 0  # None = entire sphere.
    for depth in [5]:  #  range(4, 5):
        data = get_data(rg, layer=depth, octant_id=octant)

        o_pts = rg.project(data, ['b_oct', 'c_oct'])
        g_pts = rg.project(data, ['b_oct', 'g_gcd'])
        g_areas = wgs84_area(rg, g_pts, 3)
        c_areas = enu_planar_polygon_area(rg, o_pts, 3)
        # e_areas = c_ell_area_estimates(rg, o_pts, 3)
        sg = sum(g_areas)
        se = sum(c_areas)
        print(
            f"Depth {depth}: "
            f"g_areas={sg:.2f} c_areas={se:.2f} "
            f"diff={se-sg:.2f} "
            f"ratio={se/sg:.2f}"
        )
        m_areas = np.abs(g_areas - c_areas)
        c_pts = rg.project(o_pts, ['c_oct', 'c_ell'])
        tr_pts = data.coords.reshape([-1, 3, 2])
        tr_cts = tr_pts.mean(axis=1)
        t_num = len(c_areas)
        t_sum = np.sum(c_areas)
        t_avg = t_sum / t_num
        c_score = (c_areas / t_avg) - 1  # fractional deviation
        snow_globe(c_pts, 3, c_score, name=f'c_score_{depth}')

        # np.save(cache, data_areas)
        t_num = len(g_areas)
        t_sum = np.sum(g_areas)
        t_avg = t_sum / t_num
        g_score = (g_areas / t_avg) - 1  # fractional deviation
        snow_globe(c_pts, 3, g_score, name=f'g_score_{depth}')

        # residual.
        score = np.abs(g_score - c_score)  # fractional deviation
        snow_globe(c_pts, 3, score, name=f'residual_{depth}')
