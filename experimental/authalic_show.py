# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Uses the grids stored by authalic_deriv.
"""
from pathlib import Path
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt, colors
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
from authalic_deriv_03 import AF


def mplot_ax_vector(ax):
    """mplot3d uses azim around z and elev from xy-plane"""
    az = np.deg2rad(ax.azim)
    el = np.deg2rad(ax.elev)
    return np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])


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


def snow_globe(arr: Points, poly_len: int = 6, layer: int = 0, values=None):
    """Display a 3D point cloud using matplotlib"""
    mpl.rcParams['path.simplify'] = False
    fig = plt.figure(figsize=(15, 15), dpi=400, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')  # FOV = 0 deg
    ax.view_init(elev=30, azim=40)
    axis = mplot_ax_vector(ax)
    front = arr.coords.reshape(-1, poly_len, 3)
    rx = front.reshape(-1, 3)
    x_min, x_max = rx[:, 0].min(), rx[:, 0].max()
    y_min, y_max = rx[:, 1].min(), rx[:, 1].max()
    z_min, z_max = rx[:, 2].min(), rx[:, 2].max()
    if True:
        ax.set_xlim(x_min, x_max)  # fill the area with the map.
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
    polys = [p for p in front]

    if values is not None:
        authalic_error = np.mean(np.abs(values))
        col_map_name = 'RdBu_r'
        max_abs = float(np.max(np.abs(values)))
        norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=+max_abs)
        sm = plt.cm.ScalarMappable(cmap=col_map_name, norm=norm)
        sm.set_array([])

        # Map authalicity values (pops) to colours using the symmetric TwoSlopeNorm
        cmap = mpl.colormaps[col_map_name]
        facecols = cmap(norm(values))

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
        collection = Poly3DCollection(polys, ec='black', alpha=0.2, linewidth=3)
        ax.add_collection(collection)

    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"output/auth_tri_l{layer}.png", dpi=100)
    plt.close(fig)
    print(f'file saved at output/auth_tri_l{layer}.png')


def show_pts(arr, layer=99):
    """Display a 3D point cloud using matplotlib"""
    xx, yy, zz = arr[:, 0], arr[:, 1], arr[:, 2]
    fig = plt.figure(figsize=(10, 10), dpi=200, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=40)
    ax.set_proj_type('ortho')  # FOV = 0 deg
    ax.scatter(xx, yy, zz, marker=',', ec='none', s=20)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    fig.savefig(f"output/auth_show_{layer}.png", dpi=100)
    print(f'fig saved at output/auth_show_{layer}.png')


def show_tri(vals, layer=99):
    """Display a 3D point cloud using matplotlib"""
    # xx, yy, zz = arr[:, 0], arr[:, 1], arr[:, 2]
    polys = vals.coords.reshape((-1, 3, 3))
    fig = plt.figure(figsize=(10, 10), dpi=200, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=40)
    ax.set_proj_type('ortho')  # FOV = 0 deg
    collection = Poly3DCollection(polys, ec='black', alpha=0.4, linewidth=0.05)
    ax.add_collection(collection)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    fig.savefig(f"output/auth_show_tri_{layer}.png", dpi=400)
    print(f'fig saved at output/auth_show_tri_{layer}.png')

def show_net(i2t, af, labs, layer=99):
    sz = [1, 1, 1600, 300, 50]
    dim = 3 ** layer + 1
    tm = dim - 1
    p = i2t[:, 0].astype(np.float64)
    r = i2t[:, 2].astype(np.float64)
    i = p
    j = tm - r
    dx = 1.0 / tm  # big triangle side length ~ 1 (use 1.0 for “micro-edge = 1”)
    dy = (np.sqrt(3.0) / 2.0) * dx
    x = (j - 0.5 * i) * dx
    y = -i * dy
    cmap = mpl.colormaps["tab10"].resampled(10)  # 10 distinct colours
    norm = mpl.colors.BoundaryNorm(range(10), cmap.N)  # bins: 0..8
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    ofs = 0.25 * 1/tm

    fig = plt.figure(figsize=(8, 8), dpi=200, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111)
    # plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    ax.scatter(x, y, marker='o', c=af, cmap=cmap, norm=norm, ec='none', s=sz[layer])
    for i, txt in enumerate(labs):
        ax.annotate(txt, (x[i], y[i]), ha='center', va='center', c='white', fontsize=7)
    fig.savefig(f"output/auth_net_{layer}.png", dpi=200)
    print(f'fig saved at output/auth_net_{layer}.png')


def load_lattice(layer: int):
    """
    Load lattice data from authalic_deriv
    """
    path = Path(f"output/authalic_l{layer}.npz")
    z = np.load(path, allow_pickle=True)
    i2t = np.asarray(z['i2t'])
    t2i = np.asarray(z['t2i'])
    af = np.asarray(z['af'])
    par = np.asarray(z['par'])
    sib = np.asarray(z['sib'])
    ll = np.asarray(z['ll'])
    grid = np.asarray(z['grid'])
    tri_area = np.asarray(z['tri_area'])
    tri_rel = np.asarray(z['tri_rel'])
    return i2t, t2i, af, par, sib, ll, grid, tri_area, tri_rel

def spinal(ll, af):
    """Print diagnostic latitude sequences for the canonical half-octant.

    - "spine chain": vertices whose AF is SPINE/TWINS/QTWIN plus the fixed pole.
    - "seam chain": vertices on the left seam (lon == lon_l) across SEAMM/SEAMG/SEAMQ plus the fixed pole.

    The output is sorted north->south and also prints the final few entries with (idx, lat, lon, af)
    so it's obvious when we have duplicate/near-zero points or a non-smooth tail.
    """
    ll = np.asarray(ll, dtype=float)
    af = np.asarray(af)

    # Numerical tolerances for diagnostics.
    eps_lat = 1e-12
    eps_lon = 1e-9

    def is_solved(i: int) -> bool:
        return np.isfinite(ll[i, 0]) and ll[i, 0] != -1.0 and np.isfinite(ll[i, 1])

    def fmt_lat(x: float) -> str:
        # Keep small numbers readable; avoid scientific spam for near-zero.
        if abs(x) < 1e-10:
            return "0"
        return f"{x:.12g}"

    def fmt_lon(x: float) -> str:
        if abs(x) < 1e-10:
            return "0"
        return f"{x:.12g}"

    # Determine canonical seam longitude (left equator corner) and canonical spine longitude.
    # We infer `hex_layer` from the lattice size: n = (tm+1)(tm+2)/2 where tm = 3**hex_layer.
    n = ll.shape[0]
    tm = int(round((np.sqrt(8.0 * n + 1.0) - 3.0) / 2.0))
    # If tm is wrong for some reason, fall back to the largest lon span.
    if tm <= 0:
        tm = max(1, int(round(np.sqrt(n))))

    # Left and right equator corners exist in ll; pick the min and max lon among equator (lat ~ 0)
    # as a robust way to recover lon_l and lon_r.
    equ = np.flatnonzero(np.isfinite(ll[:, 0]) & (np.abs(ll[:, 0]) < eps_lat))
    if equ.size:
        lon_l = float(np.min(ll[equ, 1]))
        lon_r = float(np.max(ll[equ, 1]))
    else:
        lon_l = float(np.min(ll[:, 1]))
        lon_r = float(np.max(ll[:, 1]))
    spine_lon = 0.5 * (lon_l + lon_r)

    # tol_lon = 1e-9

    def af_name(x):
        try:
            return AF(int(x)).name
        except Exception:
            return str(int(x))

    # --- spine-ish chain ---
    spine_mask = np.array([is_solved(i) for i in range(ll.shape[0])], dtype=bool)
    spine_mask &= np.isin(af, [AF.FIXED, AF.SPINE, AF.TWINS, AF.QTWIN])

    spine_rows = [(i, float(ll[i, 0]), float(ll[i, 1]), int(af[i])) for i in np.flatnonzero(spine_mask)]
    spine_rows.sort(key=lambda t: (-t[1], t[2], t[0]))  # north->south

    print('\n[FIXED, SPINE, TWINS, QTWIN] (north→south)')
    for i, la, lo, d in spine_rows:
        print(fmt_lat(la))

    print('\nSpine tail (last 12):')
    for i, la, lo, d in spine_rows[-12:]:
        print(f"idx={i:5d} lat={fmt_lat(la)} lon={fmt_lon(lo)} af={af_name(d)}")

    if spine_rows:
        near0 = [(i, la, lo, d) for (i, la, lo, d) in spine_rows if abs(la) < eps_lat]
        if near0:
            print(f"\nSpine near-zero lat count={len(near0)} (eps_lat={eps_lat:g}); first few:")
            for i, la, lo, d in near0[:8]:
                print(f"  idx={i} lat={fmt_lat(la)} lon={fmt_lon(lo)} af={af_name(d)}")

    # --- seam chain (canonical left seam) ---
    seam_mask = np.array([is_solved(i) for i in range(ll.shape[0])], dtype=bool)
    seam_mask &= (np.abs(ll[:, 1] - lon_l) < eps_lon)
    seam_mask &= np.isin(af, [AF.FIXED, AF.SEAMM, AF.SEAMG, AF.SEAMQ])

    seam_rows = [(i, float(ll[i, 0]), float(ll[i, 1]), int(af[i])) for i in np.flatnonzero(seam_mask)]
    seam_rows.sort(key=lambda t: (-t[1], t[2], t[0]))

    print('\n[FIXED, SEAMM, SEAMG, SEAMQ] on canonical seam (lon=lon_l) (north→south)')
    for i, la, lo, d in seam_rows:
        print(fmt_lat(la))

    print('\nSeam tail (last 18):')
    for i, la, lo, d in seam_rows[-18:]:
        print(f"idx={i:5d} lat={fmt_lat(la)} lon={fmt_lon(lo)} af={af_name(d)}")

    if seam_rows:
        # Count exact-ish duplicates after rounding to expose pile-ups.
        key_counts: dict[tuple[float, float], int] = {}
        for _i, la, lo, _d in seam_rows:
            k = (float(np.round(la, 12)), float(np.round(lo, 12)))
            key_counts[k] = key_counts.get(k, 0) + 1

        # Report pile-ups at (0, lon_l) and any other repeated coordinate.
        pile = sorted(((k, c) for k, c in key_counts.items() if c > 1), key=lambda t: -t[1])
        near0 = [(i, la, lo, d) for (i, la, lo, d) in seam_rows if abs(la) < eps_lat]

        print(f"\nSeam near-zero lat count={len(near0)} (eps_lat={eps_lat:g})")
        if near0:
            print("Seam near-zero sample (first 12):")
            for i, la, lo, d in near0[:12]:
                print(f"  idx={i} lat={fmt_lat(la)} lon={fmt_lon(lo)} af={af_name(d)}")

        if pile:
            print("\nSeam duplicate coordinate pile-ups (top 8):")
            for (la_k, lo_k), c in pile[:8]:
                print(f"  count={c:4d} at lat={fmt_lat(la_k)} lon={fmt_lon(lo_k)}")

    # Quick sanity summary.
    if seam_rows:
        print(f"\nSeam summary: count={len(seam_rows)} lat_min={fmt_lat(seam_rows[-1][1])} lat_max={fmt_lat(seam_rows[0][1])} lon_l={fmt_lon(lon_l)}")
    if spine_rows:
        print(f"Spine summary: count={len(spine_rows)} lat_min={fmt_lat(spine_rows[-1][1])} lat_max={fmt_lat(spine_rows[0][1])} spine_lon={fmt_lon(spine_lon)}")



if __name__ == '__main__':
    layer = 4
    rg = Registrar()
    g_gcd = rg.domain('g_gcd')
    c_ell = rg.domain('c_ell')
    i2t, t2i, af, par, sib, ll, grid, tri_area, tri_rel = load_lattice(layer)
    labels = [f'{i}' for i,x in enumerate(ll)]
    spinal(ll, af)
    # labels = [f'{la:.2f}\n{lo:.2f}' for (la, lo) in ll]
    show_net(i2t, af, labels, layer)
    llg = ll[grid].reshape(-1, 2)
    pll = Points(llg, g_gcd)
    pel = rg.project(pll, [g_gcd, c_ell])
    snow_globe(pel, 3, layer, tri_rel)
    # show_pts(pel.coords, hex_layer)
    #
