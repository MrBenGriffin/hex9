"""
Part of the H9 project - Visualisation of hexagons as a heatmap

Last Tested 23 October 2025 √
"""
import os
from pathlib import Path
import numpy as np
from hhg9 import Points, Registrar, Domain
from hhg9.h9.polygon import hex_layer, hh_layer
from matplotlib import pyplot as plt, colors
from matplotlib.collections import PolyCollection
from hhg9.algorithms.distance import wgs84_area


def load_data(src_dir: Path, base: str, dom: Domain, bbox=None, rnd=False):
    """
    Return a Points array in domain 'dom' with path to values and components
    This isn't exactly very generic yet - but it is indicative.
    """
    pop_bry_f = src_dir / f"{base}_bry.npy"
    pop_cmp_f = src_dir / f"{base}_bry_cmp.npy"
    pop_data_f = src_dir / f"{base}_lon_lat_pop.npy"
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
                    # pop_data[:, 2] = rng.random(pop_data.shape[0]) * 1000000.0
                    # pop_bary[:, 0] = pop_bary[:, 0] * (gx_max - gx_min) + gx_min
                    # pop_bary[:, 1] = pop_bary[:, 1] * (gy_max - gy_min) + gy_min
                    # pop_data[:, 2] = rng.random(pop_data.shape[0]) * 1000000.0
                return Points(pop_bary, dom, components=pop_cmp, samples=pop_data[:, 2])
            else:
                return Points(pop_bary, dom, components=pop_cmp)

        else:
            return Points(pop_bary, dom)


def run(layers):
    """Do the work"""
    reg = Registrar()
    b_oct = reg.domain('b_oct')
    # Greater Tokyo Area rough bounding boxes (lon/lat, WGS84)
    tokyo_bbox_tight = (139.25, 140.21, 35.25, 36.25)
    base = 'jpn__kyushu'
    dx = Path(os.getcwd()) / 'hh_heatmaps/src'
    pts = load_data(dx, base, b_oct, bbox=tokyo_bbox_tight, rnd=False)
    # hexes, count, inv, oc = hex_layer(pts, layers)
    hexes, count, inv, oc = hex_layer(pts, layers)
    xy, oz = hexes[0].reshape([-1, 2]), oc[0].reshape([-1])
    px = Points(xy, b_oct, components=oz)
    m2 = wgs84_area(reg, px)
    print(m2)

    xy = hexes.reshape([-1, 2])

    pops = np.bincount(inv, weights=pts.samples, minlength=hexes.shape[0])
    nx, mx, ny, my = min(xy[:, 0]), max(xy[:, 0]), min(xy[:, 1]), max(xy[:, 1])
    ratio = (my-ny)/(mx-nx)
    mask = pops > 10.0
    pops = pops[mask]
    hexes = hexes[mask]

    norm = colors.Normalize(vmin=np.min(pops), vmax=np.max(pops))
    cmap = plt.get_cmap('plasma')
    col = cmap(norm(pops))
    collection = PolyCollection(
        hexes,
        facecolors=col,
        ec='none',
        linewidth=0,
        # ec=(0, 0, 0, 0.2),
        # linewidth=0.01,
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
    plt.colorbar(sm, ax=ax, shrink=0.4, pad=0.02, label='Population')
    ax.axis('off')
    out = f'../output/{base}_l{layers:02}_view.png'
    plt.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.0)
    plt.close()  # optional: free memory
    # plt.show()


if __name__ == '__main__':
    run(14)  # 3=2, 4=3, 5=3, 6=21 - 10 is over sampling.
