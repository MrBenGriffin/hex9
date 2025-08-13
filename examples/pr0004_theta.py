"""
Part of the H9 project - Preparation 0004
Load the GCD quadrilateral, project onto Barycentric Coordinates.
Calculate an optimal theta to the boundary.
Display it, and store it as a single value, along with
the octahedral rectangle coordinates.
Last Tested 07 August 2025 √
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from hhg9 import Registrar, Points
from hhg9.domains import GeneralGCD, EllipsoidCartesian, OctahedralCartesian, OctahedralBarycentric
from hhg9.projections import EllipsoidGCD, AKOctahedralEllipsoid
from support import Util


def edge_based_rotation(corners):
    """
    Return rotation of a quadrilateral by
    averaging the angles of its two longest edges.
    """
    edge_vectors = np.roll(corners, -1, axis=0) - corners
    edge_lengths_sq = np.sum(edge_vectors ** 2, axis=1)
    longest_edge_indices = np.argsort(edge_lengths_sq)[-2:]
    longest_vectors = edge_vectors[longest_edge_indices]
    angle1 = np.arctan2(longest_vectors[0, 1], longest_vectors[0, 0])
    angle2 = np.arctan2(longest_vectors[1, 1], longest_vectors[1, 0])
    return (angle1 + angle2) / 2.0


if __name__ == '__main__':
    util = Util()
    reg = Registrar()                   # Manage Domains & Projections
    g_gcd = GeneralGCD(reg)             # GCD Spherical Domain (latitude/longitude)
    c_ell = EllipsoidCartesian(reg)     # Cartesian Ellipsoid (xyz)
    c_oct = OctahedralCartesian(reg)    # Cartesian Octahedron (xyz)
    b_oct = OctahedralBarycentric(reg, c_oct)  # 2d Flat for addressing.
    eg = EllipsoidGCD(reg)              # g_sph <=> c_sph
    ak = AKOctahedralEllipsoid(reg)     # c_sph <=> (c_oct <=> b_oct)
    ak.set_accuracy(0.0000000001)       # nanometre

    file = 'jpn'
    bd_file = f'src/{file}_lat_lon_bounds.npy'
    lat_lon = np.load(bd_file)  # GCD bounds: min_lat/max_lat/min_lon/max_lon
    min_lat, max_lat, min_lon, max_lon = lat_lon
    # lat/lon for projection.
    gcd_r = np.array([  # This is the GCD rectangle mapped onto Plate Carree.
        [min_lat, min_lon], [max_lat, min_lon], [max_lat, max_lon], [min_lat, max_lon]
    ])
    b_gcd = Points(gcd_r, g_gcd)
    b_data = reg.project(b_gcd, [g_gcd, c_ell, c_oct, b_oct])
    poly = b_data.coords.copy()
    centroid = np.atleast_2d(poly.mean(axis=0))
    theta = edge_based_rotation(poly)
    np.save(f'src/{file}_theta.npy', theta)
    np.save(f'src/{file}_centroid.npy', centroid)
    np.save(f'src/{file}_bry_border.npy', poly)
    print(f'Calculated and stored theta ‘{theta}’, centroid ‘{centroid}’, and pre-rotated border from barycentric projection of {bd_file} as rectangle')
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=150)
    inv_c = np.cos(-theta)
    inv_s = np.sin(-theta)
    inv_rot = np.array([[inv_c, -inv_s], [inv_s, inv_c]])
    squared = (poly-centroid) @ inv_rot.T + centroid
    np.save(f'src/{file}_rot_bry_border.npy', squared)
    quad = patches.Polygon(poly, facecolor='blue', edgecolor='darkblue', linewidth=2, alpha=0.2)
    ax.add_patch(quad)
    quad = patches.Polygon(squared, facecolor='green', edgecolor='darkgreen', linewidth=2, alpha=0.2)
    ax.add_patch(quad)
    ax.scatter(centroid[:, 0], centroid[:, 1])
    ax.set_aspect('equal', 'box')
    ax.autoscale_view()  # Automatically adjust view to fit the patch
    ax.relim()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title(f'Mapped Barycentric (Blue) and Rotated {np.rad2deg(theta):.2f}º, stored in _rot_bry_border.npy')
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    plt.show()
