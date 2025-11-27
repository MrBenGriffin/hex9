"""
 Part of h9f by Ben Griffin.
 This file defines a generic support class
 They could all be refactored elsewhere.
"""
import numpy as np
from numpy.typing import NDArray
import matplotlib as mpl
from random import choice, sample
import json


class Util:
    """
     • Validation checks (also found in OSProjection), that check points are
      o_valid - on the surface of the unit octahedron.
      s_valid - on the surface of the unit sphere.
     • Various Random Generators.
      sph_rnd - generate n points on the unit sphere.
      ball_rnd - generate a small (0.04) n point ball at a given coordinate.
      tri_rnd - generate n points on the unit equilateral triangle (upward facing, barycentric).
      tri_eff - generate n points as tri_rnd but with an F carved from the centre.
      oct0_rnd - generate n points on octant 0.
      oct_rnd - generate n points on the unit octahedron.
      col_rnd - generate n random colours
     • Various Matrices for rotation, etc.
      rx, ry, rz: 3D rotation matrices.
      mx, my, mz: 3D mirror matrices.
      r2d: 2d rotation matrix
     • 2d/3d conversions
      d3_2 - drop z axis
      d2_3 - add z axis (and set to given value).
     • Common, unadorned GCD calculations (spherical).
      xyz_ll - convert 3D coordinates to geographic coordinates.
      ll_xyz - convert geographic coordinates to 3D coordinates.
     • JSON handlers
      json_load - load a json file
      json_save - save a json file
    """

    def __init__(self):
        pass

    # def o_valid(self, pts: NDArray) -> bool:  # Constraint: |util|+|v|+|w|=1 (surface of the unit octahedron)
    #     """
    #     Test that |util|+|v|+|w|=1 (surface of the unit octahedron)
    #     :param pts: set of 3d Euclidean points
    #     :return: that the points are on the surface of the unit octahedron.
    #     """
    #     return np.all(np.apply_along_axis((lambda a: np.abs(np.sum(np.abs(a)) - 1.) < 1e-15), -1, pts))
    #
    # def s_valid(self, pts: NDArray) -> bool:  # Constraint: √(util^2+v^2+w^2)=1 (surface of the unit sphere)
    #     """
    #     Test that √(util^2+v^2+w^2)=1 (surface of the unit sphere)
    #     :param pts: set of 3d Euclidean points
    #     :return: that the points are on the surface of the unit sphere.
    #     """
    #     return np.allclose(np.linalg.norm(pts, axis=-1, keepdims=True) - 1.0, np.zeros_like(pts))

    def wgs84_rnd(self, n_points):
        """
        Generates uniformly distributed random points on an ellipsoid surface.

        Args:
            n_points (int): The number of points to generate.
            a (float): The semi-major axis (equatorial radius).
            b (float): The semi-minor axis (polar radius).

        Returns:
            np.ndarray: An (n_points, 3) array of (x, y, z) coordinates.
        """
        # WGS 84 ellipsoid parameters in meters
        print('use algorithms/pickers!')
        a = 6378137.0
        b = 6356752.314245
        xyz = self.sph_rnd(n_points)
        x = a * xyz[:, 0]
        y = a * xyz[:, 1]
        z = b * xyz[:, 2]
        return np.stack([x, y, z], axis=1)

    def sph_rnd(self, n):
        # Generate n random spherical points
        # returned in Euclidean (x,y,z)
        # https://mathworld.wolfram.com/SpherePointPicking.html
        print('use algorithms/pickers!')
        θ = np.random.uniform(0, 2. * np.pi, n)
        u = np.random.uniform(-1., 1., n)
        x = ((1. - u ** 2.) ** 0.5) * np.cos(θ)
        y = ((1. - u ** 2.) ** 0.5) * np.sin(θ)
        z = u
        return np.stack([x, y, z], axis=1)

    # def disk_rnd(self, n):
    #     # Generate n random points within a circle.
    #     # returned in Euclidean (x,y)
    #     # https://mathworld.wolfram.com/DiskPointPicking.html
    #     θ = np.random.uniform(0, 2. * np.pi, n)
    #     r = np.sqrt(np.random.uniform(0., 1., n))
    #     x = r * np.cos(θ)
    #     y = r * np.sin(θ)
    #     return np.stack([x, y], axis=1)

    def near_to(self, lat_lon, n=100, radius_deg=1.0):
        """
        Generate `n` random lat/lon points within `radius_deg` degrees of `lat_lon`.
        """
        lat0, lon0 = lat_lon[:, 0], lat_lon[:, 1]
        # Sample radius with uniform area (sqrt of uniform r)
        r = radius_deg * np.sqrt(np.random.rand(n))
        theta = 2 * np.pi * np.random.rand(n)

        # Polar to cartesian (approximation on lat/lon)
        d_lat = r * np.sin(theta)
        d_lon = r * np.cos(theta)
        # Note: dlon scaled by cos(latitude) to approximate spherical distortion
        d_lon /= np.cos(np.radians(lat0))
        # Final points
        lat = lat0 + d_lat
        lon = lon0 + d_lon
        return np.column_stack((lat, lon))

    # def ball_rnd(self, origin=(0, 0, 0), n=1000):
    #     # create a 0.04 random ball at a given point.
    #     pts = self.sph_rnd(n)
    #     ball = pts / (25 * np.linalg.norm(pts, axis=1, keepdims=True))
    #     return ball + origin

    # def tri_rnd(self, n):
    #     w = 2 ** 0.5
    #     h = 6 ** 0.5 * 0.5
    #     w2 = w / 2.
    #     h3 = h / 3.
    #     v = np.array([(-w2, -h3), (0, 2. * h3), (w2, -h3)])
    #     x = np.sort(np.random.rand(2, n), axis=0)
    #     return np.column_stack([x[0], x[1] - x[0], 1.0 - x[1]]) @ v

    # def tri_grid(self, n):
    #     w = 2 ** 0.5
    #     h = 6 ** 0.5 * 0.5
    #     w2 = w / 2.
    #     h3 = h / 3.
    #     v = np.array([(-w2, -h3), (0, 2. * h3), (w2, -h3)])
    #     x = np.sort(np.random.rand(2, n), axis=0)
    #     return np.column_stack([x[0], x[1] - x[0], 1.0 - x[1]]) @ v

    # def _eff_bbox(self):
    #     # return three bboxes in an F shape in octant-flattened space
    #     # for the purpose of testing correct orientation and reflection
    #     # of each octant under projection and on the map.
    #     util = 2 ** 0.5 / 8.          # tri.w / 8
    #     v = 6 ** 0.5 * 0.5 / 4.5   # 2*tri.h / 9
    #     w = v / 6
    #     # x-min, y-min, x-max, y-max
    #     pt = (-util+w, v-w, util+w, v+w)    # top arm points of F
    #     pm = (-util+w, 0-w, util+w, 0+w)    # middle arm points of F
    #     pl = (-util-w, -v-w, -util+w, v+w)  # left side points of F
    #     return [pt, pm, pl]

    # def tri_eff(self, n):
    #     fpt = self._eff_bbox()
    #     pts = self.tri_rnd(n)
    #     ok = []
    #     for pt in pts:
    #         good = True
    #         for x_lo, y_lo, x_hi, y_hi in fpt:
    #             if x_lo <= pt[0] <= x_hi and y_lo <= pt[1] <= y_hi:
    #                 good = False
    #                 break
    #         if good:
    #             ok.append(pt)
    #     return np.array(ok)

    # def oct0_rnd(self, n):
    #     vx = []
    #     rxy = np.random.uniform(0, 1, [n, 2])
    #     for (a, b) in rxy:
    #         if a + b > 1:
    #             a, b = 1 - a, 1 - b
    #         vx.append([a, b, 1 - a - b])
    #     return np.array(vx)
    #
    # def oct_rnd(self, n):
    #     o0 = self.oct0_rnd(n)  # 3d triangles.
    #     signs = [
    #         (+1, -1, +1), (-1, -1, +1), (+1, +1, +1), (-1, +1, +1),
    #         (+1, -1, -1), (-1, -1, -1), (+1, +1, -1), (-1, +1, -1)
    #     ]
    #     return np.array([np.copysign(t, choice(signs)) for t in o0])


    # def oct0_ce_biased(self, n, bias=0.5):
    #     """
    #     Generate samples biased toward corners and edges of the +++ octahedral triangle.
    #     """
    #     samples = []
    #     while len(samples) < n:
    #         a, b = np.random.uniform(0, 1, 2)
    #         if a + b > 1:
    #             a, b = 1 - a, 1 - b
    #         x, y, z = a, b, 1 - a - b
    #
    #         # Determine which region (logs of the 4) this point is in
    #         # Sort coordinates to find the largest (closest to a vertex)
    #         largest = max(x, y, z)
    #
    #         # Accept if the point lies in one of the vertex-dominant regions
    #         # (where one coordinate is significantly larger than the others)
    #         if np.random.rand() * bias + largest > 0.25:  # tune threshold (e.g., 0.45–0.6) to widen/narrow the vertex region
    #             samples.append([x, y, z])
    #
    #     return np.array(samples)
    #

    # def oct0_biased(self, n, alpha=0.3):
    #     """
    #     Generate biased samples in the +++ octahedral triangle
    #     that emphasize corners and edges.
    #
    #     Parameters:
    #     - n (int): number of samples
    #     - alpha (float): bias strength (0 = uniform, closer to 1 = stronger corner/edge emphasis)
    #
    #     Returns:
    #     - np.ndarray of shape (n, 3)
    #     """
    #     # Dirichlet distribution: alpha < 1 biases toward corners and edges
    #     # lower values of alpha increase concentration on corners/edges
    #     samples = np.random.dirichlet([alpha, alpha, alpha], size=n)
    #     return samples
    #

    # def oct_edge_rnd(self, n):
    #     """
    #     Generates `n_points` points that lie randomly along the edges of the unit octahedron.
    #     Each point is a convex combination of two axes (e.g., x and y), with signs.
    #     """
    #     points = []
    #     axes = [0, 1, 2]  # x, y, z
    #     rv = np.random.uniform(0, 1, n)
    #     for v in rv:
    #         a, b = sample(axes, 2)  # choose two axes
    #         coords = [0.0, 0.0, 0.0]
    #         coords[a] = v * choice([-1, 1])
    #         coords[b] = (1.0 - v) * choice([-1, 1])
    #         points.append(coords)
    #     return np.array(points)

    # def col_rnd(self, n):
    #     col = ["#" + ''.join([choice('0123456789ABCDEF') for j in range(6)]) for i in range(n)]
    #     return mpl.colors.ListedColormap(col, 'rnd')
    #
    # def rx(self, theta):
    #     ct, st = np.cos(theta), np.sin(theta)
    #     return np.array([[1., 0, 0], [0, ct, -st], [0, st, ct]])
    #
    # def ry(self, theta):
    #     ct, st = np.cos(theta), np.sin(theta)
    #     return np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    #
    # def rz(self, theta):
    #     ct, st = np.cos(theta), np.sin(theta)
    #     return np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1.]])

    # def mx(self):
    #     return np.array([[-1., 0, 0], [0, 1., 0], [0, 0, 1.]])
    #
    # def my(self):
    #     return np.array([[1., 0, 0], [0, -1., 0], [0, 0, 1.]])
    #
    # def mz(self):
    #     return np.array([[1., 0, 0], [0, 1., 0], [0, 0, -1.]])

    # def r2d(self, theta):
    #     ct, st = np.cos(theta), np.sin(theta)
    #     return np.array([[ct, -st], [st, ct]])
    #
    # def d3_2(self, pts):
    #     return np.delete(pts, 2, -1)
    #
    # def d2_3(self, pts, val):
    #     return np.insert(pts, pts.shape[1], val, axis=1)

    # def xyz_ll(self, pts):
    #     x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    #     lat = np.degrees(np.arctan2(z, np.sqrt(x ** 2. + y ** 2.)))
    #     lon = np.degrees(np.arctan2(y, x))
    #     if isinstance(x, np.ndarray):
    #         return np.stack([lat, lon], axis=1)
    #     else:
    #         return np.array([lon, lon])
    #
    # def ll_xyz(self, ll):  # Standard GCD to Euclidean
    #     pt = np.apply_along_axis(np.radians, -1, ll)  # side_keys
    #     phi, theta = pt[..., 0], pt[..., 1]
    #     x = np.cos(phi) * np.cos(theta)
    #     y = np.cos(phi) * np.sin(theta)
    #     z = np.sin(phi)  # z is 'up'
    #     if isinstance(x, np.ndarray):
    #         return np.stack([x, y, z], axis=1)
    #     else:
    #         return np.array([x, y, z])

    def json_load(self, path):
        with (open(path, 'r') as infile):
            obj = json.load(infile)
            infile.close()
        return obj

    # def json_save(self, path, data):
    #     with open(path, 'w', encoding='utf-8') as out_file:
    #         json.dump(data, out_file, ensure_ascii=False, indent=4)
    #         out_file.close()

    def t_sub(self, abc, r1, r2=None, n=0, up=True):
        """
        given points abc return the subgrid
        This does not work on geodesic.
        Probably does not work on spherical (mean vs. slerp)
        """
        i, j, k = abc
        xy = tuple([
            np.mean([i, j, k], axis=0).tolist(),  # centre.
            i,  # pt i
            np.mean([i, i, j], axis=0).tolist(),  # iij.
            np.mean([i, j, j], axis=0).tolist(),  # ijj.
            j,  # p_j.
            np.mean([j, j, k], axis=0).tolist(),  # jjk.
            np.mean([j, k, k], axis=0).tolist(),  # jkk.
            k,  # p_k.
            np.mean([k, k, i], axis=0).tolist(),  # kki.
            np.mean([k, i, i], axis=0).tolist()  # kii.
        ])
        # TRX must be from apex, cw.
        trx = (
            [(1, 2, 9), True],  [(0, 9, 2), False], [(2, 3, 0), True],
            [(5, 0, 3), False], [(3, 4, 5), True],  [(0, 5, 6), True],
            [(6, 8, 0), False], [(8, 6, 7), True],  [(9, 0, 8), True]
        )
        hup = (  # cw from point at end of long edge.
            (6, 0, 3, 4), (9, 0, 6, 7), (3, 0, 9, 1)
        )
        hdn = (  # cw from point at end of long edge.
            (7, 8, 0, 5), (1, 2, 0, 8), (4, 5, 0, 2)
        )
        if n == 0:
            ofx = len(r1)
            r1 += [tuple(a) for a in xy]
            if r2 is not None:
                ref = hup if up else hdn
                pts = [tuple([h+ofx for h in hh]) for hh in ref]
                r2 += pts
        else:
            for t, o in trx:
                ud = o if up else not o
                self.t_sub(tuple([xy[i] for i in t]), r1, r2, n-1, ud)

    def t_grid(self, n, abc):
        """Generates a grid of points within a triangle."""
        r1, r2 = [], []
        self.t_sub(abc, r1, r2, n)
        return np.array(r1), np.array(r2)

    # def e_grid(self, n, a=1.0):
    #     """Generates a grid of points within an equilateral"""
    #     e = 1e-40
    #     hx = a / 2.0
    #     th = a * np.sqrt(3) / 6.0
    #     abc = (-hx, -th), (e, 2*th), (hx, -th)
    #     # print(abc)
    #     grid, _ = self.t_grid(n, abc)
    #     return np.unique(grid)

    # def hh_grid(self, n, a=1.0, tol=1.0e-12):
    #     """Generates a grid of points within an equilateral - with half-hex polys"""
    #     e = 1e-40
    #     hx = a / 2.0
    #     th = a * np.sqrt(3) / 6.0
    #     abc = (-hx, -th), (e, 2*th), (hx, -th)
    #     # print(abc)
    #     grid, hhx = self.t_grid(n, abc)
    #     gx, oi, ri = np.unique(np.floor(grid / tol).astype(int), return_index=True, return_inverse=True, axis=0)
    #     hp = ri[hhx]
    #     gd = grid[oi]
    #     return gd, hp


if __name__ == '__main__':
    u = Util()
    # for i in range(7):
    #     pts, hhs = util.hh_grid(i, 2.0)
    #     print(f'{i},{len(pts)},{len(hhs)}')