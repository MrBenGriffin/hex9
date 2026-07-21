# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
For a given layer, compose a reference address for each
hexagon in that layer - generate the set of hexagons at that layer
and display on the globe.

Last Tested
16 Jun 2026 0.1.3a0 (passed) 904.7s
13 Mar 2026 0.1.1a1 (passed)
26 Dec 2025 0.1.0a4 (passed)
16 Dec 2025 0.1.0a3 (passed)
25 Nov 2025 (passed)
"""
import numpy as np
from matplotlib import pyplot as plt, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area, wgs84
from hhg9.h9 import H9_RA, H9O
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
    return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])


def cull_backface(arr, axis):
    """back-face culling"""
    centroids = arr.mean(axis=1)
    sides = centroids @ axis
    return sides >= 0


def snow_globe(arr: Points, poly_len: int = 6, scores=None, layers='x'):
    """Display a 3D point cloud using matplotlib"""
    mpl.rcParams['path.simplify'] = False
    plt.rcParams['figure.constrained_layout.use'] = True
    mpl.rcParams["figure.autolayout"] = True
    fig = plt.figure(figsize=(15, 15), dpi=400, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=40)
    axis = mplot_ax_vector(ax)
    all_polys = arr.coords.reshape(-1, poly_len, 3)
    mask = cull_backface(all_polys, axis)
    front = all_polys[mask]
    # front = all_polys
    max_abs = float(np.max(np.abs(scores)))
    norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=+max_abs)
    cmap_name = 'RdBu_r'
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    rgba, _norm = rgba_from(scores, cmap_name, norm=norm)
    # pops = scores[mask]
    ax.set_proj_type('ortho')  # FOV = 0 deg
    if True:
        ax.set_xlim(-4e+6, 4e+6)  # fill the area with the map.
        ax.set_ylim(-4e+6, 4e+6)
        ax.set_zlim(-4e+6, 4e+6)
    polys = [p for p in front]

    collection = Poly3DCollection(polys, ec='black', facecolors=rgba[mask], alpha=0.9, linewidth=0.02)
    ax.add_collection(collection)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    pct = (np.exp(scores) - 1.0) * 100.0
    ax.title.set_text(
        f'Layer {layers}  |  area deviation from ideal: '
        f'min {pct.min():+.2f}%  max {pct.max():+.2f}%  '
        f'p90 {np.percentile(np.abs(pct), 90):.2f}%'
    )
    plt.tight_layout()
    fig.savefig(f"output/ex0080w_{layers}.png", dpi=400)
    print(f'fig saved at output/ex0080w_{layers}.png')


def get_data(reg: Registrar, depth, mode=None):
    """Load up global sample data"""
    # grab generation for given depth
    b_oct = reg.domain('b_oct')
    all_rgn = [  # these are 0..11
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


def hexify(reg: Registrar, b_pts: Points, layers: int = 4, name=''):
    """
    Find hexagons for data, display on a globe, and export area stats to CSV.
    """
    pts, pops = hex_poly_layer(b_pts, layers)

    # Calculate geodesic area per hexagon via geographiclib PolygonArea.
    gm2 = reg.ellipsoid_area  # 510_065_621_724_154.6   # WGS-84 total surface area (m²)
    bins = 12 * 9 ** layers  # total hexagons at this layer
    ideal_m2 = gm2 / bins  # ideal equal-area per hex
    ideal_side = np.sqrt(2 * ideal_m2 / (3 * np.sqrt(3)))

    h_pts = pts.copy()
    # c_pts = reg.project(h_pts, ['b_oct', 'c_oct', 'c_ell'])
    g_pts = reg.project(h_pts, ['b_oct', 'g_gcd'])

    w_area_m2 = wgs84_area(reg, g_pts)  # (N_hex,) actual areas
    ratio = np.abs(w_area_m2 / ideal_m2) + 1e-12
    score = np.log(ratio)  # authalic log-density ℓ

    # Geodesic centroids: average 6 vertex lat/lons per hexagon.
    # Averaging in lat/lon is valid at hex scale (hexes are small).
    latlon = g_pts.coords.reshape(-1, 6, 2)  # (N_hex, 6, [lat, lon])
    centroid_lat = latlon[:, :, 0].mean(axis=1)  # (N_hex,)
    centroid_lon = latlon[:, :, 1].mean(axis=1)
    a = latlon
    b = np.roll(latlon, -1, axis=1)
    edge_m = wgs84(a, b)  # lengths between a, b (N,6)
    e_long = edge_m.max(axis=1) / edge_m.min(axis=1)
    s_pct_dev = (np.abs(e_long) - 1.0) * 100
    edge_cv = edge_m.std(axis=1) / edge_m.mean(axis=1)
    e_log_anisotropy = np.log(edge_m.max(axis=1) / edge_m.min(axis=1))
    e_mean_log = np.mean(e_log_anisotropy)
    e_std_log = np.std(e_log_anisotropy)
    z_99 = e_mean_log + 2.326 * e_std_log

    # ── Summary statistics ──────────────────────────────────────────────────
    pct_dev = (ratio - 1.0) * 100.0  # % deviation from ideal
    print(f"\n {name}:  ({bins} hexagons) {reg.ellipsoid_name}")
    print(f"  area min/max/ ideal: {w_area_m2.min():,.0f} / {w_area_m2.max():,.0f} / {ideal_m2:,.0f} m²")
    print(f"  % dev  min/mean/max: {pct_dev.min():+.8f}% / "
          f"{pct_dev.mean():+.8f}% / {pct_dev.max():+.8f}%")
    print(f"  log-ratio std dev  : {score.std():.8f}")
    print(f"  |% dev| p50/p90/p95/p99/p99.9/p99.99: "
          f"{np.percentile(np.abs(pct_dev), 50):.8f}% / "
          f"{np.percentile(np.abs(pct_dev), 90):.8f}% / "
          f"{np.percentile(np.abs(pct_dev), 95):.8f}% / "
          f"{np.percentile(np.abs(pct_dev), 99):.8f}% / "
          f"{np.percentile(np.abs(pct_dev), 99.9):.8f}% / "
          f"{np.percentile(np.abs(pct_dev), 99.99):.8f}%"
          f"")
    print(f"  edge-anisotropy:")
    print(f"  |% dev| p50/p90/p95/p99/p99.9/p99.99: "
          f"{np.percentile(np.abs(s_pct_dev), 50):.8f}% / "
          f"{np.percentile(np.abs(s_pct_dev), 90):.8f}% / "
          f"{np.percentile(np.abs(s_pct_dev), 95):.8f}% / "
          f"{np.percentile(np.abs(s_pct_dev), 99):.8f}% / "
          f"{np.percentile(np.abs(s_pct_dev), 99.9):.8f}% / "
          f"{np.percentile(np.abs(s_pct_dev), 99.99):.8f}%"
          f"")
    print(f"  coefficient of variation:")
    print(f"  99th Percentile (Normal Approx): {np.exp(z_99)}")
    print(f"  |elongation %| p50/p90/p95/p99/p99.9/p99.99: "
          f"{np.percentile(np.abs(edge_cv) * 100, 50):.8f}% / "
          f"{np.percentile(np.abs(edge_cv) * 100, 90):.8f}% / "
          f"{np.percentile(np.abs(edge_cv) * 100, 95):.8f}% / "
          f"{np.percentile(np.abs(edge_cv) * 100, 99):.8f}% / "
          f"{np.percentile(np.abs(edge_cv) * 100, 99.9):.8f}% / "
          f"{np.percentile(np.abs(edge_cv) * 100, 99.99):.8f}%"
          f"")
    # ── CSV export ──────────────────────────────────────────────────────────
    # csv_path = f"output/ex0080w_{layers}{name}.csv"
    # with open(csv_path, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow([
    #         "hex_idx", "centroid_lat", "centroid_lon",
    #         "area_m2", "ideal_m2", "ratio", "pct_deviation", "log_ratio"
    #     ])
    #     for i in range(len(w_area_m2)):
    #         writer.writerow([
    #             i,
    #             f"{centroid_lat[i]:.8f}",
    #             f"{centroid_lon[i]:.8f}",
    #             f"{w_area_m2[i]:.8f}",
    #             f"{ideal_m2:.8f}",
    #             f"{ratio[i] - 1e-12:.6f}",   # strip the epsilon back out
    #             f"{pct_dev[i]:.8f}",
    #             f"{score[i]:.8f}",
    #         ])
    # print(f"  CSV saved: {csv_path}")

    # ── Globe plot ───────────────────────────────────────────────────────────
    # snow_globe(c_pts, 6, score, f'{layers}_{name}')


if __name__ == '__main__':
    depth = 5  # 0,...5 √
    rg = Registrar()  # Manage Domains & Projections (6378137.0, 1/298.257222101)
    # rg.set_ellipsoid(name='GRS80', a=6378137.0, inv_f=298.257222101)
    b_oct = rg.domain('b_oct')

    layer = 5
    data = get_data(rg, depth)  # should be 8*9**depth (eg, depth=0: 72 points, 9 points on each face, and six points in each hexagon)

    ellipsoid = rg.ellipsoid_name
    # warps = ['warp_data']
    # for warp in warps:
    #     warp_path = Path(f'../hhg9/data/{ellipsoid}_l{layer}_{warp}.npz')
    #     b_oct.set_warp(warp_path)
    hexify(rg, data.copy(), layers=depth, name=f'fund')
