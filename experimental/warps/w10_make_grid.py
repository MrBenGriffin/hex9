"""
Part of the H9 project
For a given hex_layer, generate the canonical triangle grid and optionally display.
Last Tested 6 November 2025 √
"""
import numpy as np
from hhg9 import Registrar, Points
from hhg9.h9.polygon import tri_mesh
from w99_plot import plot_grid, rgba_from
from w98_density import get_density
import matplotlib.pyplot as plt

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


def run(rg: Registrar, layer: int = 3, octant_id: int = 1, plotting: bool = False, save=True):
    """
    Triangular grid will be 9 triangles per octant at hex_layer 0.
    At each subsequent hex_layer, the number of triangles will increase by 9 per triangle.
    So the number of triangles will be 8*9**(hex_layer+1).
    """
    b_oct = rg.domain('b_oct')
    cmp = b_oct.signs_by_id[octant_id]
    mode = b_oct.oid_mo[octant_id]

    # tri_mesh(levels: int = 5, mode: int = 0, h9p = H9P)
    # Base grid in barycentric XY (b_oct)
    verts, _, trx = tri_mesh(layer, mode)
    tris = np.array([verts[v] for t in trx for v in t], dtype=np.float64)

    t_grid = tris.reshape([-1, 2])  # triangle CW
    b_grid = Points(t_grid, b_oct, components=cmp)
    tri_xy = b_grid.coords.reshape([-1, 3, 2])
    xy_cent = tri_xy.mean(axis=1)

    b_vert = Points(verts, b_oct, components=cmp)
    b_cent = Points(xy_cent, b_oct, components=cmp)           # shape (N_tri, 2)
    # AK Jacobian at those centroids (using the existing projection object)
    c_ell = get_density(rg, b_cent, mode)
    v_ell = get_density(rg, b_vert, mode)

    uv_cent = rg.project(b_cent, ['b_oct', 's_oct'])
    uv_vert = rg.project(b_vert, ['b_oct', 's_oct'])

    if plotting:
        shrink = 0.85
        acol_norm = rgba_from(c_ell)
        plot_grid(b_grid, False, shrink=shrink, fc_n=acol_norm, title=f"b_oct centroids (L={layer}, octant={octant_id})")

        v_xy = b_vert.coords  # (N_vert, 2)
        vmax = np.max(np.abs(v_ell))
        ve_min, ve_max = np.min(v_ell), np.max(v_ell)
        # reuse your colour mapping, or do a symmetric cool/warm map
        # cols = rgba_from(ell_v)  # if rgba_from handles symmetric scaling
        norm_ell = v_ell / vmax if vmax > 0 else v_ell
        cols, norm = rgba_from(norm_ell)  # assuming rgba_from expects [-1,1]

        fig, ax = plt.subplots()
        ax.triplot(verts[:, 0], verts[:, 1], trx, linewidth=0.2, alpha=0.4)
        s = 4 if layer >= 5 else 10  # keep it light at high L
        ax.scatter(v_xy[:, 0], v_xy[:, 1], s=s, c=cols)
        ax.set_aspect('equal', 'box')
        ax.set_title(f"Vertex ℓ L={layer}; {ve_min:.3f}, {ve_max:.3f}")
        plt.show()

    if save:
        np.savez(
            f"grid_l{layer}_m{mode}_simplex.npz",
            components=cmp,          # octant-identity (3,)
            depth=layer,             # mesh size = 9**(hex_layer+1)
            tri_xy=tri_xy,           # triangle xy-coordinates (N_tri,3,2)
            uv_cent=uv_cent.coords,         # triangle uv-centroids (N_tri,2)
            xy_vert=b_vert.coords,
            uv_vert=uv_vert.coords,
            c_ell=c_ell,             # authalic log-density ℓ  (N_tri)
            v_ell=v_ell,             # authalic log-density ℓ  (N_verts)
        )


if __name__ == '__main__':
    reg = Registrar()
    for i in [4]:  # range(7):
        run(reg, i, 0, plotting=(0 < i < 5), save=True)
