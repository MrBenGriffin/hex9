"""
KK Relaxation Re-runner
Rebuilds Kamada–Kawai layouts for the octahedral grid graphs.
Reads:  graph_{depth}.json, graph_pos_{depth}.json
Writes: kk_pos_{depth}.json  (lat/lon)  and kk_pos_{depth}.npz  (NumPy arrays)
"""

from pathlib import Path
import json
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from pyproj import CRS, Transformer
from tqdm import tqdm

crs_geodetic = CRS.from_epsg(4326)  # WGS84 lat/lon
crs_ell = CRS.from_epsg(4978)       # ECEF
to_geodetic = Transformer.from_crs(crs_ell, crs_geodetic, always_xy=True)
to_ecef = Transformer.from_crs(crs_geodetic, crs_ell, always_xy=True)

def xyz_to_ll(xyz):
    """Convert Cartesian → (lat, lon) degrees."""
    x, y, z = np.moveaxis(np.asarray(xyz), -1, 0)
    lon, lat, _ = to_geodetic.transform(x, y, z)
    return np.stack([lat, lon], axis=-1)

def ll_to_xyz(ll):
    """Convert (lat, lon) → ecef (x,y,z)."""
    lat, lon = np.moveaxis(np.asarray(ll), -1, 0)
    heights = np.zeros_like(lat)
    x, y, z = to_ecef.transform(lon, lat, heights)
    return np.stack([x, y, z], axis=-1)

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
    fig = plt.figure(figsize=(15, 15), dpi=200, frameon=False)
    fig.subplots_adjust(top=1.0, bottom=0, right=1.0, left=0, hspace=0, wspace=0)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev, azim)
    view_axis = mplot_ax_vector(ax)
    mask = cull_backface(xyz, view_axis)
    if mask.sum() < (xyz.shape[0] // 2):
        view_axis = -view_axis
        mask = cull_backface(xyz, view_axis)
    pts = xyz[mask]
    # ax.set_xlim(-4e+6, 4e+6)  # fill the area with the map.
    # ax.set_ylim(-4e+6, 4e+6)
    # ax.set_zlim(-4e+6, 4e+6)
    ax.scatter(*pts.T, s=30, alpha=0.9, marker='.', ec='none')
    ax.set_axis_off()
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()


def run_kk(depth: int):
    print(f"\n=== Running Kamada–Kawai for depth={depth} ===")
    net_f = Path(f"graph_{depth}.json")
    pos_f = Path(f"graph_pos_{depth}.json")
    if not (net_f.exists() and pos_f.exists()):
        print(f"Missing graph {net_f} or pos {pos_f} — skipping.")
        return

    with open(net_f) as f:
        net = json.load(f)
    with open(pos_f) as f:
        pos_n = json.load(f)
    pos = {int(k): np.array(v, dtype=float) for k, v in pos_n.items()}
    graph = nx.node_link_graph(net, edges="links")

    print(f"  Nodes: {len(pos)}  Edges: {len(graph.edges)}")
    print("  Running KK relaxation...")
    k_pos = nx.kamada_kawai_layout(graph, pos=pos, dim=3)
    xyz = np.array([k_pos[i] for i in sorted(k_pos.keys())])
    ll = xyz_to_ll(xyz)

    out_json = Path(f"kk_pos_{depth}.json")
    out_npz  = Path(f"kk_pos_{depth}.npz")
    with open(out_json, "w") as f:
        json.dump({i: list(ll[i]) for i in range(len(ll))}, f)
    np.savez_compressed(out_npz, xyz=xyz, latlon=ll)
    print(f"  Saved: {out_json} and {out_npz}")


if __name__ == "__main__":
    frame = np.load(f"kk_pos_3.npz", allow_pickle=True)
    xyz = frame["xyz"]
    # ll = xyz_to_ll(xyz)
    # fx = json.load(open(f"result_xyz_pos_3.json"))
    # fe = json.load(open(f"/Users/ben/Documents/Projects/kk2_boost/files/result_ll_pos_2.json"))
    # fe = json.load(open(f"ak_graph_pos_3.json"))
    # vx = np.array(list(fe.values()))
    # xyz = ll_to_xyz(ll)

    # latlon = frame["latlon"]
    plot_sphere_points(xyz)
    # np.savez_compressed(out_npz, xyz=xyz, latlon=ll)
    # for d in range(4):     # or range(4)
    #     run_kk(d)