import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt, colors
from matplotlib.collections import PolyCollection
from matplotlib.ticker import FuncFormatter
from hhg9.h9 import H9K


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


def set_axis(mfig, simplex: bool = False, cols=1, box=1):
    """Axis template"""
    ax = mfig.add_subplot(1, cols, box)  # (*nrows*, *ncols*, *index*)

    ax.set_aspect('equal', adjustable='box')
    if not simplex:
        ax.set_xlim(H9K.limits.TL, H9K.limits.TR)  # Use TL/TR with a 5% margin
        ax.set_ylim(H9K.limits.VF, H9K.limits.VC)  # Use VF/ΛC with a 5% margin
    else:
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
    ax.set_axis_off()
    return ax


def plot_grid(grid_pts, simplex: bool = False,
              shrink: float = 1.0,
              fc_n: tuple = None, title: str | None = None,
              cmap_name: str = "plasma"):
    """
    :param grid_pts: Points(n*3, 2) of triangle grid. Resizable to (n,3,2) triangles.
    :param simplex: if True, grid_pts is in simplex coordinates (u,v).
    :param shrink: factor in (0,1] to shrink each triangle towards its centroid.
    :param fc_n:   optional (n,4),norm RGBA array giving per-triangle colours. If None,
                       a colormap is applied based on triangle index.
    :param title: optional title for the figure.
    :param cmap_name: Matplotlib colormap name to use when facecolors is None.
    """
    tris = grid_pts.coords.reshape([-1, 3, 2])
    n_tri = tris.shape[0]
    facecolors, norm = fc_n

    if shrink != 1.0:
        # Shrink each triangle towards its centroid so edges are visually separated
        centroids = tris.mean(axis=1, keepdims=True)  # (N,1,2)
        tris = centroids + (tris - centroids) * shrink

    if facecolors is None:
        # Colour-code by triangle index so that corresponding triangles can be tracked
        cmap = mpl.colormaps[cmap_name]
        idx = np.linspace(0.0, 1.0, max(n_tri, 2), endpoint=True)
        facecolors = cmap(idx[:n_tri])

    # vals = np.asarray(values, dtype=float)
    # dev = vals - vals.mean()

    ratio = H9K.derived.H / H9K.derived.W if not simplex else 1.0
    size = 10
    fig = plt.figure(figsize=(size, ratio * size), dpi=300, frameon=False)
    fig.subplots_adjust(top=0.98, bottom=0.02, right=0.98, left=0.02)
    ax = set_axis(fig, simplex)
    collection = PolyCollection(
        tris,
        ec=(0, 0, 0, 0.25),
        facecolors=facecolors,
        alpha=1.0,
        linewidth=0.0001,
    )
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])  # no data needed
    cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb.ax.tick_params(labelsize=20)  # or 14, etc.
    cb.formatter = FuncFormatter(lambda v, pos: f"{v:+.3f}")
    cb.update_normal(sm)
    ax.add_collection(collection)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(f"{title}.png", dpi=300)
    plt.show()


def plot_bwr(idx, grid_pts, bv, wv, rv, title='Base, Warp, Residual', mask_centres=None, mask_radius=None, auto_mask_corners: bool = False):
    """Plot base, warp, residual values with a single shared colourbar.

    The colour scale is shared across all three panels and is centered on
    the mean of the base field, so the colourbar shows signed deviation
    from that mean.
    """
    tris = grid_pts.coords.reshape([-1, 3, 2])

    n_tri = tris.shape[0]
    centroids = tris.mean(axis=1)  # (N,2)

    mask = None

    # Optionally infer the three triangle-corner regions to mask based on geometry
    if auto_mask_corners and mask_radius is not None and mask_centres is None:
        # Treat the three farthest centroids from the global centroid as the face corners
        centre = centroids.mean(axis=0)
        r2_all = np.sum((centroids - centre) ** 2, axis=1)
        corner_idx = np.argsort(r2_all)[-3:]
        mask_centres = centroids[corner_idx]

    if mask_centres is not None and mask_radius is not None:
        mask_centres = np.asarray(mask_centres, dtype=float).reshape(-1, 2)
        r2 = float(mask_radius) ** 2
        mask = np.zeros(n_tri, dtype=bool)
        for c in mask_centres:
            d2 = np.sum((centroids - c) ** 2, axis=1)
            mask |= d2 <= r2

        # Helpful diagnostic: how much of the face are we discarding in stats?
        frac_masked = mask.mean()
        print(f"[plot] masking {frac_masked:.4%} of triangles ({mask.sum()}/{n_tri}) around corners")

    # 1) centre on base mean, optionally excluding masked triangles
    if mask is not None:
        valid = ~mask
        base_mean = float(bv[valid].mean())
    else:
        valid = slice(None)
        base_mean = float(bv.mean())

    dev_b = bv - base_mean
    dev_w = wv - base_mean
    dev_r = rv - base_mean

    if mask is not None:
        all_dev = np.concatenate([dev_b[valid], dev_w[valid], dev_r[valid]])
    else:
        all_dev = np.concatenate([dev_b, dev_w, dev_r])

    print(f"[plot] Δℓ stats (unmasked): min={all_dev.min():+.4f}, max={all_dev.max():+.4f}, "
          f"max|Δℓ|={np.max(np.abs(all_dev)):.4f}")

    # 2) symmetric normalisation around 0 using only unmasked triangles
    max_abs = float(np.max(np.abs(all_dev)))
    norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-max_abs, vmax=+max_abs)

    # For visual continuity, force masked triangles to sit at the colourbar centre (neutral colour)
    if mask is not None:
        dev_b = dev_b.copy()
        dev_w = dev_w.copy()
        dev_r = dev_r.copy()
        dev_b[mask] = 0.0
        dev_w[mask] = 0.0
        dev_r[mask] = 0.0

    cmap_name = 'RdBu_r'
    sm = plt.cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])

    ratio = H9K.derived.H / H9K.derived.W
    size = 10
    fig = plt.figure(figsize=(3 * size, ratio * size), dpi=300, frameon=False)
    fig.subplots_adjust(top=0.98, bottom=0.02, right=0.98, left=0.02, wspace=0.01)

    panels = [(dev_b, 'Base'), (dev_w, 'Warp'), (dev_r, 'Residual')]
    axes = []
    for i, (vals, name) in enumerate(panels):
        rgba, _norm = rgba_from(vals, cmap_name, norm=norm)
        ax = set_axis(fig, cols=3, box=i + 1)
        axes.append(ax)
        collection = PolyCollection(
            tris,
            ec=(0, 0, 0, 0.25),
            facecolors=rgba,
            alpha=1.0,
            linewidth=0.0001,
        )
        ax.add_collection(collection)
        ax.set_ylim(H9K.limits.VF, H9K.limits.VC)
        ax.set_aspect('equal', adjustable='box')
        ax.set_axis_off()
        # ax.set_title(name, fontsize=16)

    # cb = fig.colorbar(sm, ax=axes, fraction=0.04, pad=0.02)
    # cb.ax.tick_params(labelsize=16)
    # cb.formatter = FuncFormatter(lambda v, pos: f"{v:+.3f}")
    # cb.set_label('Δℓ from base mean', fontsize=16)
    # cb.update_normal(sm) ma_psi_xl5_v0305_l5_m0_n16_v0305.npz

    # if title:
    #     fig.suptitle(title, fontsize=16)

    plt.savefig(f'bwr_{idx}.jpg', dpi=300)
    # plt.show()


def plot_mesh(v, t, t_val, simplex: bool = False, title: str | None = None, cmap_name: str = "plasma"):
    """Plot a mesh with scalar data.

    Parameters
    ----------
    v : (N,2) array
        Vertex coordinates.
    t : (M,3) array
        Triangle indices (into v).
    t_val : (N,) or (M,) array
        Scalar data. If length == N, interpreted as per-vertex values.
        If length == M, interpreted as per-triangle values and averaged
        onto vertices for plotting.
    """
    v = np.asarray(v)
    t = np.asarray(t, dtype=int)
    t_val = np.asarray(t_val)

    n_vert = v.shape[0]
    n_tri = t.shape[0]

    if t_val.ndim != 1:
        t_val = t_val.ravel()

    if t_val.shape[0] == n_vert:
        # Per-vertex scalar
        z = t_val
    elif t_val.shape[0] == n_tri:
        # Per-triangle scalar: average onto vertices
        z = np.zeros(n_vert, dtype=float)
        counts = np.zeros(n_vert, dtype=float)
        for tri_idx, tri in enumerate(t):
            val = t_val[tri_idx]
            for vid in tri:
                z[vid] += val
                counts[vid] += 1.0
        counts[counts == 0.0] = 1.0
        z /= counts
    else:
        raise ValueError(
            f"t_val must have length {n_vert} (per-vertex) or {n_tri} (per-triangle), "
            f"got {t_val.shape[0]}"
        )

    ratio = H9K.derived.H / H9K.derived.W if not simplex else 1.0
    size = 10
    cmap = mpl.colormaps[cmap_name]
    fig = plt.figure(figsize=(size, ratio * size), dpi=300, frameon=False)
    fig.subplots_adjust(top=0.98, bottom=0.02, right=0.98, left=0.02)
    ax = set_axis(fig, simplex)

    tri = mpl.tri.Triangulation(v[:, 0], v[:, 1], t)
    ax.tricontourf(tri, z, levels=30, cmap=cmap)
    ax.triplot(tri, lw=0.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=40)
    plt.tight_layout()
    plt.show()
