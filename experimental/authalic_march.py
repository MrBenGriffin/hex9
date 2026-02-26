# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
Engine taken from ex0063_grid
"""
import math
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pygeodesy import Intersection3Tuple, PolygonArea, Ellipsoids
from hhg9 import Registrar, Points
from hhg9.algorithms.distance import wgs84_area
import matplotlib as mpl
import math
from pygeodesy.ellipsoidalKarney import LatLon, intersection3


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
        collection = Poly3DCollection(polys, ec='black', alpha=1.0, linewidth=0.05)
        ax.add_collection(collection)

    ax.set_aspect('equal', adjustable='box')
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(f"output/auth_tri_l{layer}.png", dpi=400)
    plt.close(fig)
    print(f'file saved at output/auth_tri_l{layer}.png')


# class EllipsoidalGridGeneratorBroke:
#     def __init__(self, hex_layer):
#         self.hex_layer = hex_layer
#         origin = LatLon(0, 0)
#         self.geod = origin.geodesic
#         self.points = {}
#
#         # 1. Calculate Exact Octant Area
#         self.octant_area = self.get_polygon_area([(0, 0), (0, 90), (90, 0)])
#         self.target = self.octant_area / (hex_layer ** 2)
#
#     def get_polygon_area(self, point_list):
#         if isinstance(point_list[0], LatLon):
#             point_list = [(a.lat, a.lon) for a in point_list]
#         pa = PolygonArea(earth=self.geod)
#         for (lat, lon) in point_list:
#             pa.AddPoint(lat, lon)
#         _, _, area = pa.Compute(reverse=False)
#         return abs(area)
#
#     def intersect_pair(self, start1, end1, start2, end2):
#         """Robustly intersects two lines, returning None if degenerate."""
#         # Check for degenerate lines (start roughly equals end)
#         if start1.distanceTo(end1) < 1e-6 or start2.distanceTo(end2) < 1e-6:
#             return None
#
#         res = intersection3(start1, end1, start2, end2)
#         if isinstance(res, (tuple, Intersection3Tuple)):
#             return res[0]  # Return LatLon
#         return None
#
#     def average_points(self, points):
#         """Averages a list of LatLon points (using n-vector / cartesian mean)."""
#         if not points: return None
#
#         x_sum, y_sum, z_sum = 0, 0, 0
#         valid_count = 0
#
#         for tri_points in points:
#             if tri_points is None: continue
#             c = tri_points.toCartesian()
#             x_sum += c.x
#             y_sum += c.y
#             z_sum += c.z
#             valid_count += 1
#
#         if valid_count == 0: return None
#
#         # Average Cartesian
#         # Note: We should normalize, but toLatLon handles it usually
#         # PyGeodesy Ecef/Cartesian objects have toLatLon
#         from pygeodesy import EcefCartesian
#         # Create a temp cartesian to convert back
#         # (A simple way is to use the first point's reference, but let's be generic)
#         # Simplified: Just mean the lat/lon if close (safe for this grid scale)
#         # But let's use vectors to be safe near pole.
#
#         # Simple Vector Mean
#         avg_x = x_sum / valid_count
#         avg_y = y_sum / valid_count
#         avg_z = z_sum / valid_count
#
#         # Convert back to LatLon
#         # We can construct a LatLon from cartesian using the geod's ellipsoid
#         # PyGeodesy's EcefCartesian is the robust way:
#         cart = EcefCartesian(avg_x, avg_y, avg_z)
#         return cart.toLatLon()
#
#     def get_triangles(self):
#         """
#         Connects the points (i,j) into a triangle mesh.
#         Returns a list of triangles, where each triangle is a list of 3 (x,y,z) tuples.
#         """
#         triangles = []
#
#         # Iterate through rows (stop 1 short of the top tip)
#         for i in range(self.hex_layer):
#             # Iterate through columns in this row
#             # The number of points in row i is (hex_layer - i + 1)
#             # The number of "Up" triangles is (hex_layer - i)
#             for j in range(self.hex_layer - i):
#
#                 # Retrieve coordinates for the quad vertices
#                 v_bl = self.points[(i, j)]  # Bottom-Left
#                 v_br = self.points[(i, j + 1)]  # Bottom-Right
#                 v_tl = self.points[(i + 1, j)]  # Top-Left
#                 a, b, d = v_bl, v_br, v_tl
#
#                 # --- Triangle 1: Upright ---
#                 # Connects (BL -> BR -> TL)
#                 triangles.append([[a.lat, a.lon], [b.lat, b.lon], [d.lat, d.lon]])
#
#                 # --- Triangle 2: Inverted ---
#                 # Checks if there is a "Top-Right" neighbor to form the inverted triangle
#                 # The row above (i+1) has one fewer point.
#                 # We need index j+1 to exist in row i+1.
#                 # Row i+1 has indices 0 to hex_layer-(i+1).
#                 # So we need j+1 <= hex_layer - i - 1  =>  j < hex_layer - i - 1
#                 if j < (self.hex_layer - i - 1):
#                     v_tr = self.points[(i + 1, j + 1)]  # Top-Right
#                     c = v_tr
#
#                     # Connects (BR -> TR -> TL)
#                     triangles.append([[b.lat, b.lon], [c.lat, c.lon], [d.lat, d.lon]])
#
#         return triangles
#
#     def solve_ideal_step(self):
#         min_step, max_step = 0.00001, 45.0
#         target = self.target
#
#         for _ in range(50):
#             mid = (min_step + max_step) / 2.0
#             pA = LatLon(0, 0)
#             pB = LatLon(0, mid)
#             dist_AB = pA.distanceTo(pB)
#             pC = pA.destination(dist_AB, 0)
#             area = self.get_polygon_area([pA, pB, pC])
#
#             if area < target:
#                 min_step = mid
#             else:
#                 max_step = mid
#         return mid
#
#     def generate(self):
#         print(f"--- Generating WGS84 Grid (Symmetric Centroid) hex_layer={self.hex_layer} ---")
#
#         # 1. Solve Ideal Corner
#         ideal_lon_step = self.solve_ideal_step()
#         dist_step = LatLon(0, 0).distanceTo(LatLon(0, ideal_lon_step))
#
#         # 2. Generate Boundary Edges
#         # Edge AB (Equator)
#         edge_AB = []
#         rem_dist_AB = LatLon(0, ideal_lon_step).distanceTo(LatLon(0, 90))
#         step_rem_AB = rem_dist_AB / (self.hex_layer - 1)
#
#         edge_AB.append(LatLon(0, 0))
#         edge_AB.append(LatLon(0, ideal_lon_step))
#         curr = edge_AB[-1]
#         for k in range(1, self.hex_layer - 1):
#             curr = curr.destination(step_rem_AB, 90)
#             edge_AB.append(curr)
#         edge_AB.append(LatLon(0, 90))
#
#         # Edge AC (Meridian 0)
#         edge_AC = []
#         p1 = LatLon(0, 0).destination(dist_step, 0)
#         dist_AC_full = LatLon(0, 0).distanceTo(LatLon(90, 0))
#         rem_dist_AC = dist_AC_full - dist_step
#         step_rem_AC = rem_dist_AC / (self.hex_layer - 1)
#
#         edge_AC.append(LatLon(0, 0))
#         edge_AC.append(p1)
#         curr = p1
#         for k in range(1, self.hex_layer - 1):
#             curr = curr.destination(step_rem_AC, 0)
#             edge_AC.append(curr)
#         edge_AC.append(LatLon(90, 0))
#
#         # Edge BC (Meridian 90)
#         edge_BC = [LatLon(tri_points.lat, 90) for tri_points in edge_AC]
#
#         # 3. Interior Points (3-Line Intersection Average)
#         for i in range(self.hex_layer + 1):
#             for j in range(self.hex_layer + 1 - i):
#                 k = self.hex_layer - i - j  # Barycentric remainder
#
#                 # Boundaries (Fixed)
#                 if i == 0:
#                     self.points[(i, j)] = edge_AB[j]
#                 elif j == 0:
#                     self.points[(i, j)] = edge_AC[i]
#                 elif k == 0:
#                     self.points[(i, j)] = edge_BC[i]
#                 else:
#                     # --- The 3 Symmetric Grid Lines ---
#
#                     # Line 1 (Horizontal): AC[i] -> BC[i]
#                     l1_s, l1_e = edge_AC[i], edge_BC[i]
#
#                     # Line 2 (Right-Leaning): AB[j] -> BC[hex_layer-j]
#                     # Note: On BC, index counts from bottom B.
#                     # If we follow grid line 'j', it hits BC at height hex_layer-j?
#                     # Let's trace hex_layer=3, j=1. Connects AB[1] to BC[2]. Correct.
#                     l2_s, l2_e = edge_AB[j], edge_BC[self.hex_layer - j]
#
#                     # Line 3 (Left-Leaning): AB[i+j] -> AC[i+j]
#                     # This line represents constant 'k'.
#                     # It connects the bottom edge to the left edge.
#                     # Index on AB is (i+j). Index on AC is (i+j).
#                     idx_k = i + j
#                     l3_s, l3_e = edge_AB[idx_k], edge_AC[idx_k]
#
#                     # --- Intersect All Pairs ---
#                     candidates = []
#
#                     # P1: Intersect Horizontal & Right-Lean
#                     p1 = self.intersect_pair(l1_s, l1_e, l2_s, l2_e)
#                     if p1: candidates.append(p1)
#
#                     # P2: Intersect Horizontal & Left-Lean
#                     p2 = self.intersect_pair(l1_s, l1_e, l3_s, l3_e)
#                     if p2: candidates.append(p2)
#
#                     # P3: Intersect Right-Lean & Left-Lean
#                     p3 = self.intersect_pair(l2_s, l2_e, l3_s, l3_e)
#                     if p3: candidates.append(p3)
#
#                     # --- Average the Result ---
#                     # This distributes the error into the center of the triangle
#                     # rather than along a seam.
#                     final_pt = self.average_points(candidates)
#
#                     self.points[(i, j)] = final_pt
#
#     def export_obj(self, filename="wgs84_symmetric.obj"):
#         """Standard OBJ export"""
#         with open(filename, 'w') as f:
#             f.write(f"# Symmetric WGS84 hex_layer={self.hex_layer}\n")
#             v_map = {}
#             idx = 1
#             sorted_keys = sorted(self.points.keys())
#
#             for k in sorted_keys:
#                 tri_points = self.points[k]
#                 xyz = tri_points.toCartesian()
#                 scale = 1.0 / 6371000.0
#                 f.write(f"v {xyz.x * scale:.6f} {xyz.y * scale:.6f} {xyz.z * scale:.6f}\n")
#                 v_map[k] = idx
#                 idx += 1
#
#             for i in range(self.hex_layer):
#                 for j in range(self.hex_layer - i):
#                     p1 = v_map[(i, j)]
#                     p2 = v_map[(i, j + 1)]
#                     p3 = v_map[(i + 1, j)]
#                     f.write(f"f {p1} {p2} {p3}\n")
#                     if j < (self.hex_layer - i - 1):
#                         p4 = v_map[(i + 1, j + 1)]
#                         f.write(f"f {p2} {p4} {p3}\n")


class EllipsoidalGridGenerator:
    def __init__(self, level):
        self.layer = level
        self.points = {}
        self.origin = LatLon(90, 0)  # Pole
        self.geod = self.origin.geodesic
        # Number of triangles for 1/8 the global grid will be
        # 9**level.
        self.octant_area = self.area([(90, 0), (0, 0), (0, 90)])
        self.target = self.octant_area
        for i in range(self.layer):
            self.target /= 9

    def area(self, point_list):
        if isinstance(point_list[0], LatLon):
            point_list = [(a.lat, a.lon) for a in point_list]
        pa = PolygonArea(earth=self.geod)
        for (lat, lon) in point_list:
            pa.AddPoint(lat, lon)
        _, _, area = pa.Compute(reverse=False)
        return abs(area)

    def intersect_robust(self, start1, end1, start2, end2):
        """
        Wrapper for intersection3 that returns just the LatLon point.
        Should not need to use intersection.
        """
        # intersection3 returns (LatLon, float, float) or None
        res = intersection3(start1, end1, start2, end2)
        if isinstance(res, (tuple, Intersection3Tuple)):
            return res[0]
        return res

    def lon_step(self):
        """
        Finds the Longitude step along the Equator that yields the target area
        for the corner triangle (A-B-C where A=0,0).
        """
        min_step = 0.00001
        max_step = 45.0
        target = self.target

        for _ in range(50):
            mid = (min_step + max_step) / 2.0

            pA = LatLon(0, 0)
            pB = LatLon(0, mid)

            # Constraint: Isosceles by physical distance, not degrees
            dist_AB = pA.distanceTo(pB)
            pC = pA.destination(dist_AB, 0)  # Head North

            area = self.area([pA, pB, pC])

            if area < target:
                min_step = mid
            else:
                max_step = mid
        return mid

    def lat_step(self):
        """
        Finds the Latitude that yields the target area
        for the corner triangle (P-L-R where P=90,0)
        [(90, 0), (0, 0), (0, 90)
        """
        min_step = 0.0
        max_step = 90.0
        pC = self.origin

        for _ in range(64):
            mid = (min_step + max_step) / 2.0
            pA = LatLon(mid, 0)
            pB = LatLon(mid, 90)
            area = self.area([pA, pB, pC])
            diff = abs(area - self.target)
            if area > self.target:
                min_step = mid
            else:
                max_step = mid
        return mid, diff

    def generate(self):
        # print(f"--- Generating WGS84 Grid (Fixed) hex_layer={self.hex_layer} ---")

        # 1. Solve Ideal Corner
        ideal_lon_step, diff = self.lat_step()
        return ideal_lon_step, diff, self.target
        dist_step = LatLon(0, 0).distanceTo(LatLon(0, ideal_lon_step))

        # 2. Generate Boundary Edges
        # Edge AB (Equator)
        edge_AB = []
        rem_dist_AB = LatLon(0, ideal_lon_step).distanceTo(LatLon(0, 90))
        step_rem_AB = rem_dist_AB / (self.hex_layer - 1)

        edge_AB.append(LatLon(0, 0))
        edge_AB.append(LatLon(0, ideal_lon_step))

        curr = edge_AB[-1]
        for k in range(1, self.hex_layer - 1):
            curr = curr.destination(step_rem_AB, 90)  # East
            edge_AB.append(curr)
        edge_AB.append(LatLon(0, 90))

        # Edge AC (Meridian 0)
        edge_AC = []
        p1 = LatLon(0, 0).destination(dist_step, 0)  # North

        dist_AC_full = LatLon(0, 0).distanceTo(LatLon(90, 0))
        rem_dist_AC = dist_AC_full - dist_step
        step_rem_AC = rem_dist_AC / (self.hex_layer - 1)

        edge_AC.append(LatLon(0, 0))
        edge_AC.append(p1)

        curr = p1
        for k in range(1, self.hex_layer - 1):
            curr = curr.destination(step_rem_AC, 0)  # North
            edge_AC.append(curr)
        edge_AC.append(LatLon(90, 0))

        # Edge BC (Meridian 90)
        edge_BC = [LatLon(p.lat, 90) for p in edge_AC]

        # 3. Interior Points (Hybrid Intersection Strategy)
        for i in range(self.hex_layer + 1):
            for j in range(self.hex_layer + 1 - i):
                k = self.hex_layer - i - j

                if i == 0:
                    self.points[(i, j)] = edge_AB[j]
                elif j == 0:
                    self.points[(i, j)] = edge_AC[i]
                elif k == 0:
                    self.points[(i, j)] = edge_BC[i]
                else:
                    # Define the 3 potential grid lines passing through (i,j)
                    # Line i (Horizontal): Connects Left Edge (AC) to Right Edge (BC)
                    start_i, end_i = edge_AC[i], edge_BC[i]

                    # Line j (Right-Leaning): Connects Bottom (AB) to Right Edge (BC)
                    start_j, end_j = edge_AB[j], edge_BC[self.hex_layer - j]

                    # Line k (Left-Leaning): Connects Bottom (AB) to Left Edge (AC)
                    # Note: The 'k' index on AB is (i+j)? No.
                    # Let's trace it: k-lines run parallel to the hypotenuse face?
                    # k-lines connect AB[k_idx] to AC[k_idx]?
                    # Actually, for symmetry, we just need ANY two stable lines.
                    # The k-lines connect Bottom[i+j] to Left[i+j].
                    idx_k = i + j
                    start_k, end_k = edge_AB[idx_k], edge_AC[idx_k]

                    # --- SELECTION STRATEGY ---
                    # We pick the two lines that are most orthogonal / least degenerate

                    if i > (self.hex_layer / 2):
                        # NEAR POLE:
                        # The "Horizontal" i-line is getting very short (squeezed).
                        # The j-line and k-line are vertical and strong. Intersect them.
                        self.points[(i, j)] = self.intersect_robust(start_j, end_j, start_k, end_k)

                    elif j <= (self.hex_layer / 2):
                        # NEAR CORNER A (0,0):
                        # The k-line (connecting B to C) is far away or degenerate?
                        # Actually, near A, the i-line (Horiz) and j-line (Verticalish) are 90 deg.
                        # This is the standard grid intersection.
                        self.points[(i, j)] = self.intersect_robust(start_i, end_i, start_j, end_j)

                    else:
                        # NEAR CORNER B (0, 90):
                        # The j-line connects B to... B. It collapses.
                        # Use the i-line (Horiz) and the k-line (Left-Leaning).
                        self.points[(i, j)] = self.intersect_robust(start_i, end_i, start_k, end_k)

    def get_triangles(self):
        """
        Connects the points (i,j) into a triangle mesh.
        Returns a list of triangles, where each triangle is a list of 3 (x,y,z) tuples.
        """
        triangles = []

        # Iterate through rows (stop 1 short of the top tip)
        for i in range(self.layer):
            # Iterate through columns in this row
            # The number of points in row i is (hex_layer - i + 1)
            # The number of "Up" triangles is (hex_layer - i)
            for j in range(self.layer - i):

                # Retrieve coordinates for the quad vertices
                v_bl = self.points[(i, j)]  # Bottom-Left
                v_br = self.points[(i, j + 1)]  # Bottom-Right
                v_tl = self.points[(i + 1, j)]  # Top-Left
                a, b, d = v_bl, v_br, v_tl

                # --- Triangle 1: Upright ---
                # Connects (BL -> BR -> TL)
                triangles.append([[a.lat, a.lon], [b.lat, b.lon], [d.lat, d.lon]])

                # --- Triangle 2: Inverted ---
                # Checks if there is a "Top-Right" neighbor to form the inverted triangle
                # The row above (i+1) has one fewer point.
                # We need index j+1 to exist in row i+1.
                # Row i+1 has indices 0 to hex_layer-(i+1).
                # So we need j+1 <= hex_layer - i - 1  =>  j < hex_layer - i - 1
                if j < (self.layer - i - 1):
                    v_tr = self.points[(i + 1, j + 1)]  # Top-Right
                    c = v_tr

                    # Connects (BR -> TR -> TL)
                    triangles.append([[b.lat, b.lon], [c.lat, c.lon], [d.lat, d.lon]])

        return triangles

    def export_obj(self, filename="wgs84_robust.obj"):
        """Standard OBJ export"""
        with open(filename, 'w') as f:
            f.write(f"# Robust WGS84 hex_layer={self.layer}\n")
            v_map = {}
            idx = 1
            sorted_keys = sorted(self.points.keys())

            for k in sorted_keys:
                p = self.points[k]
                xyz = p.toCartesian()
                scale = 1.0 / 6371000.0
                f.write(f"v {xyz.x * scale:.6f} {xyz.y * scale:.6f} {xyz.z * scale:.6f}\n")
                v_map[k] = idx
                idx += 1

            for i in range(self.layer):
                for j in range(self.layer - i):
                    p1 = v_map[(i, j)]
                    p2 = v_map[(i, j + 1)]
                    p3 = v_map[(i + 1, j)]
                    f.write(f"f {p1} {p2} {p3}\n")
                    if j < (self.layer - i - 1):
                        p4 = v_map[(i + 1, j + 1)]
                        f.write(f"f {p2} {p4} {p3}\n")

class SphericalGridGenerator:
    def __init__(self, N):
        self.N = N
        self.points = {}  # Key: (i, j), Value: (x, y, z)

    def to_cartesian(self, lat_deg, lon_deg):
        """Convert Lat/Lon (degrees) to Cartesian Unit Vector."""
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        x = math.cos(lat) * math.cos(lon)
        y = math.cos(lat) * math.sin(lon)
        z = math.sin(lat)
        return (x, y, z)

    def to_latlon(self, x, y, z):
        """Convert Cartesian Unit Vector to Lat/Lon (degrees)."""
        lat = math.degrees(math.asin(z))
        if abs(x) < 1e-9 and abs(y) < 1e-9:
            lon = 0.0
        else:
            lon = math.degrees(math.atan2(y, x))
        return round(lat, 5), round(lon, 5)

    def cross_product(self, a, b):
        return (a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0])

    def normalize(self, v):
        mag = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if mag == 0: return (0, 0, 0)
        return (v[0] / mag, v[1] / mag, v[2] / mag)

    def get_triangles(self):
        """
        Connects the points (i,j) into a triangle mesh.
        Returns a list of triangles, where each triangle is a list of 3 (x,y,z) tuples.
        """
        triangles = []
        for i in range(self.N):
            for j in range(self.N - i):
                v_bl = self.points[(i, j)]  # Bottom-Left
                v_br = self.points[(i, j + 1)]  # Bottom-Right
                v_tl = self.points[(i + 1, j)]  # Top-Left
                triangles.append([v_bl, v_br, v_tl])
                if j < (self.N - i - 1):
                    v_tr = self.points[(i + 1, j + 1)]  # Top-Right
                    triangles.append([v_br, v_tr, v_tl])
        return triangles

    def export_obj(self, filename="octant.obj"):
        """Helper to dump the mesh to a standard 3D .OBJ file for viewing"""
        tris = self.get_triangles()

        with open(filename, 'w') as f:
            f.write(f"# Spherical Octant hex_layer={self.N}\n")

            # 1. Write all vertices (unrolled)
            # We need a map to track vertex indices for the face definitions
            v_index_map = {}  # Key: (x,y,z), Value: OBJ index (1-based)
            current_index = 1

            # Re-iterate or just unroll existing tris
            # Efficient way: write unique points first
            sorted_keys = sorted(self.points.keys())
            for k in sorted_keys:
                x, y, z = self.points[k]
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
                v_index_map[self.points[k]] = current_index
                current_index += 1

            # 2. Write Faces
            for tri in tris:
                # tri is list of 3 (x,y,z) tuples
                idx1 = v_index_map[tri[0]]
                idx2 = v_index_map[tri[1]]
                idx3 = v_index_map[tri[2]]
                f.write(f"f {idx1} {idx2} {idx3}\n")

        print(f"Exported {len(tris)} triangles to {filename}")

    def intersect_great_circles(self, n1, n2):
        """
        Find intersection of two great circles defined by normal vectors n1 and n2.
        Result is +/- the cross product (line of intersection of planes).
        We pick the one inside the octant (x,y,z >= 0).
        """
        line = self.cross_product(n1, n2)
        # Check direction (should be positive octant)
        if line[0] < -1e-9 or line[1] < -1e-9 or line[2] < -1e-9:
            line = (-line[0], -line[1], -line[2])
        return self.normalize(line)

    def get_ideal_corner_angle(self):
        """
        Returns the edge angle (degrees) that gives the corner triangle
        the exact ideal Area = PI/(2*hex_layer^2).
        """
        # Formula: acos( cot(45 + 45/hex_layer^2) )
        term = math.radians(45.0 + (45.0 / (self.N ** 2)))
        cot_val = 1.0 / math.tan(term)
        angle_rad = math.acos(cot_val)
        return math.degrees(angle_rad)

    def generate_edge_points(self):
        """
        Generates the hex_layer+1 points along the Equator (AB).
        Returns list of Longitudes.
        """
        edge_lons = [0.0] * (self.N + 1)

        # 1. Fix endpoints
        edge_lons[0] = 0.0
        edge_lons[self.N] = 90.0

        # 2. Fix Ideal Corners (Angle from 0 and Angle from 90)
        ideal_angle = self.get_ideal_corner_angle()

        # Check for overlap (hex_layer=2 case)
        if (ideal_angle * 2) > 90.0:
            edge_lons[1] = 45.0  # Fallback for hex_layer=2
            return edge_lons

        edge_lons[1] = ideal_angle
        edge_lons[self.N - 1] = 90.0 - ideal_angle

        # 3. Interpolate the middle
        # We need to fill indices 2 to hex_layer-2
        # Arc length remaining
        start_angle = edge_lons[1]
        end_angle = edge_lons[self.N - 1]
        steps = (self.N - 1) - 1  # How many intervals between index 1 and hex_layer-1

        if steps > 0:
            step_size = (end_angle - start_angle) / steps
            for k in range(1, steps):
                edge_lons[1 + k] = start_angle + (k * step_size)

        return edge_lons

    def generate(self):
        # 1. Generate the master edge (Equator)
        # List of points E_0 to E_N (Cartesian)
        lons = self.generate_edge_points()
        edge_AB = [self.to_cartesian(0, l) for l in lons]

        # 2. Generate symmetric edges
        # Edge AC (Meridian 0): Same coords but rotated (x,y,z) -> (x,z,y)?
        # No, AC is (lat, 0). Lat corresponds to Lon on AB.
        edge_AC = [self.to_cartesian(l, 0) for l in lons]

        # Edge BC (Meridian 90): Lat corresponds to Lon on AB.
        edge_BC = [self.to_cartesian(l, 90) for l in lons]

        # 3. Generate Interior Vertices
        # We iterate Barycentric indices i, j where i + j <= hex_layer
        # i = step along AC (Lat)
        # j = step along AB (Lon)
        # k = hex_layer - i - j

        print(f"--- Generating Grid hex_layer={self.N} ---")
        print(f"Ideal Corner Angle: {self.get_ideal_corner_angle():.4f}°")

        for i in range(self.N + 1):
            for j in range(self.N + 1 - i):
                k = self.N - i - j

                # Identify Boundaries (to avoid precision jitter)
                if i == 0:
                    # On Edge AB (Equator)
                    pt = edge_AB[j]
                elif j == 0:
                    # On Edge AC (Meridian 0)
                    pt = edge_AC[i]
                elif k == 0:
                    # On Edge BC (Meridian 90)
                    # Note: indexing of BC runs from B(90,0) to C(pole)
                    # Corresponds to lat rising. index i on AC corresponds to index i on BC?
                    # Yes, i is "height".
                    pt = edge_BC[i]
                else:
                    # INTERIOR POINT
                    # Intersection of two symmetric Great Circles

                    # GC1: Connects F_i (on AC) to G_i (on BC)
                    # This is a 'horizontal' arc of constant 'height' index i
                    F_i = edge_AC[i]
                    G_i = edge_BC[i]
                    normal_horiz = self.cross_product(F_i, G_i)

                    # GC2: Connects E_j (on AB) to G_(hex_layer-j) ??
                    # Let's trace the 'vertical' lines.
                    # Lines coming from Pole C are meridians.
                    # But we are using a geodesic grid.
                    # The lines run from Edge AB to Edge BC.
                    # Point j on AB connects to Point j on BC?
                    # No, symmetry.
                    # Let's look at the corner A. Lines radiate from A.
                    # Line connects A to Point on BC.
                    # To find point (i,j), we can intersect:
                    # 1. Line from B to AC (Vertex j) -> Connects E_j (on AB) to F_j (on AC)? No
                    # Let's use the standard "Symmetric" construction:
                    # Point is intersection of Arc(AC[i], BC[i]) and Arc(AB[j], BC[hex_layer-j])

                    # Arc 1: "Horizontal" (Height i). Connects AC[i] and BC[i]
                    n1 = normal_horiz

                    # Arc 2: "Right-leaning" (Width j). Connects AB[j] and BC_rev[j]?
                    # Actually, let's use the third edge for symmetry.
                    # Connects AB[j] (Equator) to AC[hex_layer-j]? No, that cuts the corner.
                    # Connects AB[j] (Equator) to C (Pole)? Only if Meridian.
                    # Connects AB[j] to BC[hex_layer-j] (Meridian 90, top down).
                    # Let's verify:
                    # If j=0, AB[0] is A. BC[hex_layer] is C. Arc is AC. Correct.
                    # If j=hex_layer, AB[hex_layer] is B. BC[0] is B. Point is B. Correct.

                    E_j = edge_AB[j]
                    C_top = edge_BC[self.N - j]  # Point on BC corresponding to 'j' from top

                    normal_vert = self.cross_product(E_j, C_top)

                    pt = self.intersect_great_circles(n1, normal_vert)

                # Store
                self.points[(i, j)] = pt

    def print_sample(self):
        # Print a few key points
        print("\nSample Coordinates (Lat, Lon):")

        # Center Point (approx hex_layer/3, hex_layer/3 for hex_layer=3, or hex_layer/2, hex_layer/2 for even)
        # Let's find the centroid index
        ci = self.N // 3
        cj = self.N // 3
        if (ci, cj) in self.points:
            lat, lon = self.to_latlon(*self.points[(ci, cj)])
            print(f"Inner Point ({ci},{cj}): \tLat {lat}°, Lon {lon}°")

        # Print Edge Steps
        print("\nEquator Edge Steps:")
        for k in range(self.N + 1):
            p = self.points[(0, k)]
            _, lon = self.to_latlon(*p)
            print(f"v_{k}: {lon}°")

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
    fig.savefig(f"output/auth_march_{layer}.png", dpi=100)
    print(f'fig saved at output/auth_march_{layer}.png')

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
    fig.savefig(f"output/auth_tri_{layer}.png", dpi=200)
    print(f'fig saved at output/auth_tri_{layer}.png')


if __name__ == '__main__':
    rg = Registrar()
    g_gcd = rg.domain('g_gcd')
    c_ell = rg.domain('c_ell')
    print(f'lv latitude            target(m^2)      delta')
    for lev in range(1, 20):
        # points = 3**lev  # per edge?
        sgg = EllipsoidalGridGenerator(lev)
        lat, diff, target = sgg.generate()
        # :15
        print(f'{lev:02d} {lat:<19.16f} {target:<16.2f} {diff:<16.6f}')
        # tx = np.array(sgg.get_triangles())
        # px = Points(tx.reshape(-1, 2), g_gcd)
        # areas = wgs84_area(rg, px, 3)
        # t_num = len(areas)
        # t_sum = np.sum(areas)
        # t_avg = t_sum / t_num
        # values = (areas / t_avg) - 1  # fractional deviation
        # vs = rg.project(px, [g_gcd, c_ell])
        # snow_globe(vs, 3,  lev, values)
        #
