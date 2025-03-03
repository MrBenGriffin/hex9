import numpy as np
import matplotlib as mpl
from random import choice
import json


class Util:

    @classmethod
    def o_valid(cls, pts):  # Constraint: |u|+|v|+|w|=1 (surface of the unit octahedron)
        return np.all(np.apply_along_axis((lambda a: np.abs(np.sum(np.abs(a)) - 1.) < 1e-15), -1, pts))

    @classmethod
    def s_valid(cls, pts):  # Constraint: √(u^2+v^2+w^2)=1 (surface of the unit sphere)
        return np.allclose(np.linalg.norm(pts, axis=-1, keepdims=True) - 1.0, np.zeros_like(pts))

    @classmethod
    def sph_rnd(cls, n):
        # Generate n random spherical points
        # returned in Euclidean (x,y,z)
        # https://mathworld.wolfram.com/SpherePointPicking.html
        θ = np.random.uniform(0, 2. * np.pi, n)
        u = np.random.uniform(-1., 1., n)
        x = ((1. - u ** 2.) ** 0.5) * np.cos(θ)
        y = ((1. - u ** 2.) ** 0.5) * np.sin(θ)
        z = u
        return np.stack([x, y, z], axis=1)

    @classmethod
    def ball_rnd(cls, origin=(0, 0, 0), n=1000):
        # create a 0.2 random ball at a given point.
        pts = cls.sph_rnd(n)
        ball = pts / (25 * np.linalg.norm(pts, axis=1, keepdims=True))
        return ball + origin

    @classmethod
    def tri_rnd(cls, n):
        w = 2 ** 0.5
        h = 6 ** 0.5 * 0.5
        w2 = w / 2.
        h3 = h / 3.
        v = np.array([(-w2, -h3), (0, 2. * h3), (w2, -h3)])
        x = np.sort(np.random.rand(2, n), axis=0)
        return np.column_stack([x[0], x[1] - x[0], 1.0 - x[1]]) @ v

    @classmethod
    def eff_bbox(cls):
        # return three bboxes in an F shape in octant-flattened space
        # for the purpose of testing correct orientation and reflection
        # of each octant under projection and on the map.
        u = 2 ** 0.5 / 8.          # tri.w / 8
        v = 6 ** 0.5 * 0.5 / 4.5   # 2*tri.h / 9
        w = v / 6
        # x-min, y-min, x-max, y-max
        pt = (-u+w, v-w, u+w, v+w)    # top arm points of F
        pm = (-u+w, 0-w, u+w, 0+w)    # middle arm points of F
        pl = (-u-w, -v-w, -u+w, v+w)  # left side points of F
        return [pt, pm, pl]

    @classmethod
    def tri_eff(cls, n):
        fpt = cls.eff_bbox()
        pts = cls.tri_rnd(n)
        ok = []
        for pt in pts:
            good = True
            for x_lo, y_lo, x_hi, y_hi in fpt:
                if x_lo <= pt[0] <= x_hi and y_lo <= pt[1] <= y_hi:
                    good = False
                    break
            if good:
                ok.append(pt)
        return np.array(ok)

    @classmethod
    def oct0_rnd(cls, n):
        vx = []
        rxy = np.random.uniform(0, 1, [n, 2])
        for (a, b) in rxy:
            if a + b > 1:
                a, b = 1 - a, 1 - b
            vx.append([a, b, 1 - a - b])
        return np.array(vx)

    @classmethod
    def oct_rnd(cls, n):
        o0 = cls.oct0_rnd(n)  # 2d triangles.
        signs = [
            (+1, -1, +1), (-1, -1, +1), (+1, +1, +1), (-1, +1, +1),
            (+1, -1, -1), (-1, -1, -1), (+1, +1, -1), (-1, +1, -1)
        ]
        return np.array([np.copysign(t, choice(signs)) for t in o0])

    @classmethod
    def col_rnd(cls, n):
        col = ["#" + ''.join([choice('0123456789ABCDEF') for j in range(6)]) for i in range(n)]
        return mpl.colors.ListedColormap(col, 'rnd')

    @classmethod
    def rx(cls, theta):
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([[1., 0, 0], [0, ct, -st], [0, st, ct]])

    @classmethod
    def ry(cls, theta):
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])

    @classmethod
    def rz(cls, theta):
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1.]])

    @classmethod
    def mx(cls):
        return np.array([[-1., 0, 0], [0, 1., 0], [0, 0, 1.]])

    @classmethod
    def my(cls):
        return np.array([[1., 0, 0], [0, -1., 0], [0, 0, 1.]])

    @classmethod
    def mz(cls):
        return np.array([[1., 0, 0], [0, 1., 0], [0, 0, -1.]])

    @classmethod
    def r2d(cls, theta):
        ct, st = np.cos(theta), np.sin(theta)
        return np.array([[ct, -st], [st, ct]])

    @classmethod
    def d3_2(cls, pts):
        return np.delete(pts, 2, -1)

    @classmethod
    def d2_3(cls, pts, val):
        return np.insert(pts, pts.shape[1], val, axis=1)

    @classmethod
    def xyz_ll(cls, pts):
        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        lat = np.degrees(np.arctan2(z, np.sqrt(x ** 2. + y ** 2.)))
        lon = np.degrees(np.arctan2(y, x))
        if isinstance(x, np.ndarray):
            return np.stack([lat, lon], axis=1)
        else:
            return np.array([lon, lon])

    @classmethod
    def ll_xyz(cls, ll):  # Standard GCD to Euclidean
        pt = np.apply_along_axis(np.radians, -1, ll)  # side_keys
        phi, theta = pt[..., 0], pt[..., 1]
        x = np.cos(phi) * np.cos(theta)
        y = np.cos(phi) * np.sin(theta)
        z = np.sin(phi)  # z is 'up'
        if isinstance(x, np.ndarray):
            return np.stack([x, y, z], axis=1)
        else:
            return np.array([x, y, z])

    @classmethod
    def json_load(cls, path):
        with (open(path, 'r') as infile):
            obj = json.load(infile)
            infile.close()
        return obj

    @classmethod
    def json_save(cls, path, data):
        with open(path, 'w', encoding='utf-8') as out_file:
            json.dump(data, out_file, ensure_ascii=False, indent=4)
            out_file.close()
