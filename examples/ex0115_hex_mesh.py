# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0
"""
Direct hex mesh generation from H9 lattice LUTs.

Uses HexMesh.create() from hhg9.h9.grid: sc_mode=0 supercells only, boundary
hex exterior verts reflected y→-y into the adjacent face via H9O.oid_nb.
Each hex is generated exactly once; no seam artefacts, no duplication.

Last Tested
-----------
"""
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from hhg9 import Registrar, Points
from hhg9.domains import PlatePixelCarree
from hhg9.h9.grid import HexMesh, clipped_ll, valid_ll


# def snow_globe(arr: Points, poly_len: int = 6, pop=None):
#     """Display a 3D point cloud using matplotlib"""
#     ax.view_init(elev=30, azim=40)
#     axis = mplot_ax_vector(ax)
#     all_polys = arr.coords.reshape(-1, poly_len, 3)
#     mask = cull_backface(all_polys, axis)
#     front = all_polys[mask]
#     rx = front.reshape(-1, 3)
#     x_min, x_max = rx[:, 0].min(), rx[:, 0].max()
#     y_min, y_max = rx[:, 1].min(), rx[:, 1].max()
#     z_min, z_max = rx[:, 2].min(), rx[:, 2].max()
#     if True:
#         ax.set_xlim(x_min, x_max)  # fill the area with the map.
#         ax.set_ylim(y_min, y_max)
#         ax.set_zlim(z_min, z_max)
#     polys = [p for p in front]
#     if pop is not None:
#         authalic_error = np.mean(np.abs(pop))
#         pops = pop[mask]
#         # v_min = np.min(pops)
#         # v_max = np.max(pops)
#         # norm = colors.Normalize(vmin=v_min, vmax=v_max)
#         # cmap = plt.get_cmap('RdBu_r')
#         col_map_name = 'RdBu_r'
#         max_abs = float(np.max(np.abs(pops)))
#         norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=+max_abs)
#         sm = plt.cm.ScalarMappable(cmap=col_map_name, norm=norm)
#         sm.set_array([])
#
#         # Map authalicity values (pops) to colours using the symmetric TwoSlopeNorm
#         cmap = mpl.colormaps[col_map_name]
#         facecols = cmap(norm(pops))
#
#         # Optional colourbar (uncomment if/when needed)
#         plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
#
#         collection = Poly3DCollection(
#             polys,
#             ec=(0, 0, 0, 0.3),
#             facecolors=facecols,
#             alpha=1.0,
#             linewidth=0.05,
#         )
#         ax.add_collection(collection)
#         ax.title.set_text(f'Authalic Error: {authalic_error:.3f}')
#     else:
#         collection = Poly3DCollection(polys, ec='black', alpha=1.0, linewidth=0.05)
#         ax.add_collection(collection)
#
#     ax.set_aspect('equal', adjustable='box')
#     ax.set_axis_off()
#     plt.tight_layout()
#     plt.savefig(f"output/ex0063_o{octant}_l{depth}.png", dpi=400)
#     plt.close(fig)
#     print(f'file saved at output/ex0063_o{octant}_l{depth}.png')

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


def show_global(pts, poly):
    """Display GCD points on the globe using WGS84-scaled meters"""
    x, y, z = pts.coords[:, 0], pts.coords[:, 1], pts.coords[:, 2]
    cols = pts.samples.astype(np.float64) / 255

    fig = plt.figure(figsize=(12, 12), dpi=300)
    fig.patch.set_alpha(0)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')  # FOV = 0 deg
    ax.set_aspect('equal', adjustable='box')
    ax.view_init(elev=30, azim=40)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1], zoom=1.5)
    axis = mplot_ax_vector(ax)
    mask = cull_backface(poly, axis)

    x_mid, y_mid, z_mid = np.median(x), np.median(y), np.median(z)
    max_range = np.array([x.max() - x.min(), y.max() - y.min(), z.max() - z.min()]).max() / 2.0

    ax.set_xlim(x_mid - max_range, x_mid + max_range)
    ax.set_ylim(y_mid - max_range, y_mid + max_range)
    ax.set_zlim(z_mid - max_range, z_mid + max_range)

    # 4. Add the points
    # ax.scatter(x, y, z, c=cols, s=2, alpha=0.5, antialiased=True)

    # 2. Add the polygons
    poly_collection = Poly3DCollection(poly, ec='black', fc='none', linewidth=2.0)
    ax.add_collection(poly_collection)

    fig.savefig(f"output/ex0115_global.png",
                dpi=300,
                bbox_inches='tight',
                pad_inches=0,
                transparent=True)

    print(f'fig saved at output/ex0115_global.png')
    plt.close(fig)


if __name__ == '__main__':
    LAYER = 1
    DPI = 200

    reg   = Registrar()
    b_oct = reg.domain('b_oct')
    g_gcd = reg.domain('g_gcd')
    c_oct = reg.domain('c_oct')
    c_ell = reg.domain('c_ell')
    p_pix = PlatePixelCarree(reg)

    pil_img = Image.open('src/bm_3600x1800.png').convert('RGBA')
    img     = np.array(pil_img)
    ph, pw  = img.shape[:2]
    pc_px   = p_pix.adopt(img, extent=(-180.0, -90.0, 180.0, 90.0), y_up=True, center=True)
    # image   = p_pix.image(pc_px)

    mesh = HexMesh.create([0, LAYER], reg)
    layer_0 = mesh.densify(0)

    # mesh     = HexMesh.create(LAYER, reg)                    # shared-vertex mesh in b_oct
    # ll_pts   = reg.project(mesh.pts, [b_oct, g_gcd])       # project unique verts once
    # l0_hexes = ll_pts.coords[layer_0]
    # # ll_hexes = ll_pts.coords[mesh.faces]
    # ll_hexes = l0_hexes

    el_pts   = reg.project(mesh.pts, [b_oct, g_gcd, c_ell])
    el_hexes = el_pts.coords[layer_0]
    # el_px = reg.project(pc_px, [p_pix, g_gcd, c_ell])
    show_global(el_px, el_hexes)

    # ok       = valid_ll(ll_hexes)                               # drop antimeridian-crossing hexes
    # p_pts    = reg.project(ll_pts, [g_gcd, p_pix])         # project unique verts to pixel
    # full_hexes = p_pts.coords[layer_0[ok]]                      # (N_ok, 6, 2)
    # full_hexes = clipped_ll(ll_hexes)

    # print(f'b_oct native L{LAYER}: {ll_hexes.shape[0]} total, {ok.sum()} displayed')

    # fig = plt.figure(figsize=(pw / 100, ph / 100), dpi=DPI, frameon=False)
    # fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    # ax = fig.add_axes([0, 0, 1, 1])
    # ax.set_axis_off()
    # ax.set_xlim(-180.0, +180.0)
    # ax.imshow(image, alpha=1.0, extent=[-180.0, +180.0, -90.0, +90.0],
    #           origin='upper', aspect='auto')
    # ax.set_ylim(-90.0, +90.0)
    # ax.add_collection(PolyCollection(
    #     full_hexes, facecolors='none',
    #     edgecolors=[(0.95, 0.95, 0.2, 1.0)], linewidth=0.5,
    # ))
    # f_name = f'output/ex0115_flat_l{LAYER}.png'
    # fig.savefig(f_name, dpi=DPI)
    # plt.close(fig)
    # print(f'  saved {f_name}')