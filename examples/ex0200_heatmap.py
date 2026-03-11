# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Part of the H9 project - Visualisation of hexagons as a heatmap (using Tokyo details).
This could be better - the sampling isn't ideal, and I have modified the approach.

Last Tested
26 December 2025 0.1.0a4 (passed)
16 December 2025 0.1.0a3 (passed - with rewrite)
23 October 2025 (passed)
"""
import os
from pathlib import Path
import numpy as np
from hhg9 import Points, Registrar
from hhg9.base.domain import Domain
from hhg9.h9.polygon import hex_poly_layer
from matplotlib import pyplot as plt, colors
from matplotlib.collections import PolyCollection


def sub_sample(pts: Points):
    """Subsample the points and samples"""
    import numpy.random as r
    xy = pts.coords
    xu = np.sort(np.unique(xy[:, 0]))
    yu = np.sort(np.unique(xy[:, 1]))
    dx = np.mean(np.diff(xu))  # typical E/W step in b_oct
    dy = np.median(np.diff(yu))  # typical hex_layer/S step in b_oct
    offsets = np.array([
        [0.0, 0.0],
        [(r.random() - 0.5) * dx, 0.0],
        [0.0, (r.random() - 0.5) * dy],
        [(r.random() - 0.5) * dx, (r.random() - 0.5) * dy],
    ])
    xy4 = np.vstack([xy + off for off in offsets])
    w4 = np.concatenate([pts.samples * 0.25] * 4)
    cpt = np.concatenate([pts.components] * 4)
    ss_pts = Points(xy4, pts.domain, cpt, samples=w4)
    return ss_pts


def load_data(src_dir: Path, base: str, dom: Domain, bbox=None, rnd=False):
    """
    Return a Points array in domain 'dom' with path to values and components
    This isn't exactly very generic yet - but it is indicative.
    """
    pop_bry_f = src_dir / f"{base}_bry.npy"
    pop_cmp_f = src_dir / f"{base}_bry_cmp.npy"
    pop_data_f = src_dir / f"{base}_lon_lat_pop.npy"  # Used only for the sample data.
    if pop_bry_f.exists():
        pop_bary = np.load(pop_bry_f)
        if pop_cmp_f.exists():
            pop_cmp = np.load(pop_cmp_f)
            if pop_data_f.exists():
                pop_data = np.load(pop_data_f)
                # Optional geographic clip: bbox = (lon_min, lon_max, lat_min, lat_max)
                if bbox is not None:
                    lon_min, lon_max, lat_min, lat_max = bbox
                    lon = pop_data[:, 0]
                    lat = pop_data[:, 1]
                    m = (lon >= lon_min) & (lon <= lon_max) & (lat >= lat_min) & (lat <= lat_max)
                    if m.any():
                        pop_bary = pop_bary[m]
                        pop_cmp = pop_cmp[m]
                        pop_data = pop_data[m]
                if rnd:
                    rng = np.random.default_rng()
                    gx_min, gx_max, gy_min, gy_max = pop_bary[:, 0].min(), pop_bary[:, 0].max(), pop_bary[:, 1].min(), pop_bary[:, 1].max()
                    # generate random values within gx/gy bounds
                    pop_bary[:, 0] = rng.random(pop_bary.shape[0]) * (gx_max - gx_min) + gx_min
                    pop_bary[:, 1] = rng.random(pop_bary.shape[0]) * (gy_max - gy_min) + gy_min
                return Points(pop_bary, dom, components=pop_cmp, samples=pop_data[:, 2])
            else:
                return Points(pop_bary, dom, components=pop_cmp)

        else:
            return Points(pop_bary, dom)


def run(layers):
    """Do the work"""
    reg = Registrar()
    # g_gcd = reg.domain('g_gcd')
    b_oct = reg.domain('b_oct')
    n_oct = reg.domain('n_oct:c_butterfly')
    ak = reg.projection('oct_ell')
    ha, ta = ak.get_accuracy(layers)
    print(f'Layer {layers}; approx per hex area: {ha:,.2f}m²')
    # Greater Tokyo Area rough bounding boxes (lon/lat, WGS84)
    tokyo_bbox_tight = (139.25, 140.21, 35.25, 36.25)
    # b = tokyo_bbox_tight
    # bc = np.array([[b[0], b[2]], [b[0], b[3]], [b[1], b[2]], [b[1], b[3]]])
    # bp = Points(bc, g_gcd)
    # bo = reg.project(bp, [g_gcd, b_oct])

    base = 'jpn__kyushu'
    dx = Path(os.getcwd()) / 'hh_heatmaps/src'
    data = load_data(dx, base, b_oct, bbox=tokyo_bbox_tight, rnd=False)
    pts = sub_sample(data)
    pts = data
    hxb, pops = hex_poly_layer(pts, layers)
    hxd = reg.project(hxb, [b_oct, n_oct])
    hexes = hxd.coords.reshape(-1, 6, 2)
    xy = hxd.coords
    nx, mx, ny, my = min(xy[:, 0]), max(xy[:, 0]), min(xy[:, 1]), max(xy[:, 1])
    ratio = (my-ny)/(mx-nx)

    norm = colors.Normalize(vmin=np.min(pops), vmax=np.max(pops))
    cmap = plt.get_cmap('plasma')
    col = cmap(norm(pops))
    collection = PolyCollection(
        hexes,
        facecolors=col,
        # ec='none',
        # linewidth=0,
        ec=(0, 0, 0, 0.2),
        linewidth=0.01,
        antialiaseds=True,
    )
    fig = plt.figure(figsize=(20, 20*ratio), dpi=150, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlim(nx, mx)
    ax.set_ylim(ny, my)
    ax.add_collection(collection)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # no data needed
    # plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.02, label='Population')
    ax.axis('off')
    out = f'output/ex0200_{base}_l{layers:02}_view.png'
    print(f'saving {out}')
    plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close()  # optional: free memory


if __name__ == '__main__':
    for layer in range(5, 11):
        run(layer)  # 3=2, 4=3, 5=3, 6=21 - 10 is over sampling.
