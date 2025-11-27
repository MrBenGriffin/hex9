"""
KK Relaxation (igraph + 3D visualisation)
----------------------------------------
Runs a stress-majorization layout using igraph, seeded from AK/barycentric positions.
Outputs .npz (xyz + latlon) and shows a quick 3D diagnostic plot.
"""

from pathlib import Path
import json
import numpy as np
import time
import threading
import igraph as ig
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def mplot_ax_vector(ax):
    """Return unit view vector (world-space) from mplot3d elev/azim."""
    az = np.deg2rad(ax.azim)
    el = np.deg2rad(ax.elev)
    # Matplotlib rotates azim around Z, but Y is flipped; fix with +90°
    return np.array([
        np.cos(el) * np.sin(az - np.pi / 2),
        -np.cos(el) * np.cos(az - np.pi / 2),
        np.sin(el)
    ])

def cull_backface(xyz, view_vec):
    """Return a boolean mask for points facing the camera."""
    # Dot product of points with view vector; keep those facing camera.
    dots = xyz @ view_vec
    return dots > 0


def plot_sphere_points(xyz, elev=0, azim=10):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev, azim)
    view_axis = mplot_ax_vector(ax)
    mask = cull_backface(xyz, view_axis)
    # If we accidentally culled the front (e.g. "waxing moon"), flip the axis
    if mask.sum() < (xyz.shape[0] // 2):
        view_axis = -view_axis
        mask = cull_backface(xyz, view_axis)
    pts = xyz[mask]
    ax.scatter(*pts.T, s=0.5, alpha=0.7)
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    plt.show()


def xyz_to_ll(xyz):
    x, y, z = np.moveaxis(np.asarray(xyz), -1, 0)
    r = np.linalg.norm([x, y, z], axis=0)
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(z / r))
    return np.stack([lat, lon], axis=-1)


def run_kk(depth: int):
    print(f"\n=== KK/Stress Layout for depth={depth} ===")
    net_f = Path(f"graph_{depth}.json")
    pos_f = Path(f"graph_pos_{depth}.json")
    if not (net_f.exists() and pos_f.exists()):
        print("Missing graph or pos file — skipping.")
        return

    net = json.load(open(net_f))
    pos_n = json.load(open(pos_f))
    pos_init = np.array(list(pos_n.values()), dtype=float)
    edges = [(int(e["source"]), int(e["target"])) for e in net["links"]]
    g = ig.Graph(n=len(pos_init), edges=edges)
    # ig.set_num_threads(18)
    # drl_3d fails. fr3d=more bound at vertices
    print(f"Nodes: {g.vcount()}  Edges: {g.ecount()}")
    t0 = time.time()
    layout_obj = g.layout("fr3d", dim=3, seed=pos_init)
    dt = time.time() - t0
    print(f"\nLayout completed in {dt/60:.2f} min")
    pos = np.array(layout_obj.coords)
    pos /= np.linalg.norm(pos, axis=1, keepdims=True)  # normalise radius
    ll = xyz_to_ll(pos)

    out_npz = Path(f"kk_pos_igraph_{depth}.npz")
    np.savez_compressed(out_npz, xyz=pos, latlon=ll)
    print(f"Saved {out_npz}")

    return pos, ll


if __name__ == "__main__":
    depth = 3
    xyz, ll = run_kk(depth)
    plot_sphere_points(xyz)
