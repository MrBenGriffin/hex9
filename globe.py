import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import math
import numpy as np
import svg
import json
from photo import Photo
from drawing import Drawing


def xyz_ll(xyz: tuple) -> tuple:
    x, y, z = xyz
    return math.degrees(math.atan2(z, math.sqrt(x * x + y * y))), math.degrees(math.atan2(y, x))


def ll_xyz(ll: tuple) -> tuple:
    phi, theta = math.radians(ll[0]), math.radians(ll[1])
    x = math.cos(phi) * math.cos(theta)
    y = math.cos(phi) * math.sin(theta)
    z = math.sin(phi)  # z is 'up'
    return x, y, z


class HalfHex:
    def __init__(self, t: tuple, g: tuple, r: int, n: int = 1, parent=None, hh_parent=None):
        self.tp = parent
        self.hhp = hh_parent
        self.r = r  # rotation * 60 degrees. starting with 0= /=\
        # t = 5 vertices of three spherical triangles. (though 1 is a midpoint).
        self.gxy = g  # This is the 2D grid/graph self.pts
        self.r = r  # rotation * 60 degrees. starting with 0= /=\
        self.districts = {}
        self.n = n
        if len(t) == 4:
            a, b, c, d = t
            e = np.mean(np.array([a, d]), axis=0)
        else:
            a, b, c, d, e = t
        self.pts = [a, b, c, d, e]
        a2, b2, c2, d2, e2 = self.gxy
        r_lut = {
            0: [True, False, True],
            1: [False, True, False],
            2: [True, False, True],
            3: [False, True, False],
            4: [True, False, True],
            5: [False, True, False]
        }
        # pu_i = r_lut[self.r]
        # This generates the shape of each level. On Level 1 we need the map to be coherent.
        # pu_i = [True, False, True] if n == 1 else r_lut[self.r]
        pu_i = r_lut[self.r]
        self.triangles = [
            Tri((b, e, a), pu_i[0], (b2, e2, a2), self, 0),
            Tri((e, b, c), pu_i[1], (e2, b2, c2), self, 1),
            Tri((c, d, e), pu_i[2], (c2, d2, e2), self, 2)
        ]
        col = [0., 0., 0.]
        for ch in range(3):
            for tr in self.triangles:
                col[ch] += tr.col[ch]
        self.col = [int(c/3.) for c in col]

    def add_districts(self):
        for t in range(3):
            tri = self.triangles[t]
            for i in range(3):
                self.districts[t * 3 + i] = tri.get_hh(i)


class Tri:
    photo = None
    sh = [3., 3. * math.sqrt(3.) / 2.]  # Class variable

    @classmethod
    def set_sh(cls, side: float):
        cls.sh = [side, side * math.sqrt(3.) / 2.]

    @classmethod
    def set_photo(cls, ph):
        cls.photo = ph

    @staticmethod
    def slerp(p0, p1, t):
        # works only with unit vectors.
        dot = np.dot(p0, p1)
        th = np.arccos(dot)
        s = np.sin(th)
        j = np.sin((1. - t) * th)
        k = np.sin(t * th)
        return (j * np.array(p0) + k * np.array(p1)) / s

    def __init__(self, ijk: tuple, up: bool, abc: tuple, parent=None, pos: int = None) -> None:
        # ijk is a tuple of three cartesian points on a unit sphere in clockwise order.
        # abc is a tuple of three 2D cartesian points for a grid (or a grid_position for starting).
        # Need to calculate the seven remaining points, clockwise starting at the centre, then significant point (^ or V for up/down resp)
        # up: this is 'point up' on a plane.
        # meanwhile we have the 2D grid position. (or?)
        self.hhp = parent
        self.pos = pos
        self.h_hex = {}
        self.up = up
        i, j, k = ijk
        # names = ['ctr', 'p_i', 'iij', 'ijj', 'p_j', 'jjk', 'jkk', 'p_k', 'kki', 'kii']
        _pts = [
            np.mean([i, j, k], axis=0),  # centre.
            i,  # pt i
            #
            self.slerp(i, j, 1./3.),
            self.slerp(i, j, 2./3.),
            j,  # p_j.
            self.slerp(j, k, 1. / 3.),
            self.slerp(j, k, 2. / 3.),
            k,  # p_k.
            self.slerp(k, i, 1. / 3.),
            self.slerp(k, i, 2. / 3.),
        ]
        # now normalise!!
        self.pts = np.array(_pts) / np.linalg.norm(_pts, axis=1, keepdims=True)
        self.sp = xyz_ll(self.pts[0])
        self.col = self.__class__.photo.col(*self.sp, flip=True)

        # now do 2D equivalents.
        self.gxy = abc
        i, j, k = abc
        self.xy = tuple([
            np.mean([i, j, k], axis=0),  # centre.
            i,  # pt i
            np.mean([i, i, j], axis=0),  # iij.
            np.mean([i, j, j], axis=0),  # ijj.
            j,  # p_j.
            np.mean([j, j, k], axis=0),  # jjk.
            np.mean([j, k, k], axis=0),  # jkk.
            k,  # p_k.
            np.mean([k, k, i], axis=0),  # kki.
            np.mean([k, i, i], axis=0)  # kii.
        ])

    def prep_hh(self, n: int, px: tuple):
        # return five vertices of half_hex indicated by i where i=[0,1,2] starting at pt0, going clockwise
        # if this is point up, then 0 = top hh, 1 = bottom right, 2 = bottom left.
        # if this is point down, then 0 = bottom, 1=top-left, 2=top-right
        # return them with ccw points, midpoint of bottom last.
        # this works for both 3d and 2d.
        ctr, p_i, iij, ijj, p_j, jjk, jkk, p_k, kki, kii = px
        perms = {
            True: {
                0: [[p_i, iij, ctr, kki, kii], 2],
                1: [[p_j, jjk, ctr, iij, ijj], 4],
                2: [[p_k, kki, ctr, jjk, jkk], 0]
            },
            False: {
                0: [[ijj, ctr, kii, p_i, iij], 1],
                1: [[jkk, ctr, ijj, p_j, jjk], 3],
                2: [[kii, ctr, jkk, p_k, kki], 5]
            }
        }
        return perms[self.up][n]

    def get_hh(self, idx: int) -> HalfHex:
        if idx not in self.h_hex:
            n = 1 if self.hhp is None else self.hhp.n + 1
            gxy, _ = self.prep_hh(idx, self.xy)
            pts, rot = self.prep_hh(idx, self.pts)
            self.h_hex[idx] = HalfHex(pts, tuple(gxy), rot, n, self, self.hhp)
        return self.h_hex[idx]


class IcoVertex:
    def __init__(self, idx: int, obj):
        self.idx = idx
        # self.name = obj['name']
        # self.ll = obj['ll']
        self.xyz = obj['xyz']
        # self.sp = obj['sp']


class IcoSide:
    # {"vx":[0,2,1],   "grid":[4,1], "up":false, "name":"Asia"},
    def __init__(self, obj, vertices):
        self.name = obj['name']
        self.vx = obj['vx']
        self.grid = obj['grid']
        self.up = obj['up']
        mx, my = Tri.sh
        gx, gy = self.grid
        p0x, p0y = gx * mx * 0.5, gy * my
        if self.up:
            p1x, p1y = p0x - mx * 0.5, p0y - my
            p2x, p2y = p0x + mx * 0.5, p0y - my
        else:
            p2x, p2y = p0x - mx * 0.5, p0y + my
            p1x, p1y = p0x + mx * 0.5, p0y + my

        self.xy = tuple([(p0x, p0y), (p1x, p1y), (p2x, p2y)])  # This is the 2D triangle.
        self.xyz = tuple([vertices[d].xyz for d in self.vx])  # This is the unit sphere triangle.
        self.tri = Tri(self.xyz, self.up, self.xy)

    def get_hh(self, idx: int) -> HalfHex:
        return self.tri.get_hh(idx)


class IcoSphere:
    def __init__(self, file: str, transform=None):
        self.vertices = {}
        self.mapping = {}
        self.sides = {}
        self.grid = 0, 0, 11, 4
        with (open(file, 'r') as infile):
            obj = json.load(infile)
            infile.close()
            for idx, vertex in enumerate(obj['vertices']):
                self.vertices[idx] = IcoVertex(idx, vertex)
            if transform is not None:
                for idx in self.vertices.keys():
                    xyz = self.vertices[idx].xyz
                    self.vertices[idx].xyz = transform @ np.array(xyz)
            if 'mapping' in obj and obj['mapping']:
                self.mapping = obj['mapping']
            # Each side is set to be either 'up' or 'down'
            # But each side is made up of three half-hexes 0,1,2
            if 'sides' in obj and obj['sides']:
                self.sides = {side['name']: IcoSide(side, self.vertices) for side in obj['sides']}
            if 'grid' in obj and obj['grid']:
                self.grid = tuple(obj['grid'])

def draw_hh(d: Drawing, h: HalfHex, adj: tuple):
    cp, cm = adj
    # <polygon points="100,100 150,25 150,75 200,0" fill="none" stroke="black" />
    r, g, b = [int(cm * (h+cp)) for h in h.col]
    tx = svg.Polygon(fill=f'rgb({r},{g},{b})', stroke="none")
    tx.points = [tuple(pt) for pt in h.gxy[:-1]]
    d.canvas.elements.append(tx)


def rot(x, y, z):
    xr = np.deg2rad(x)
    yr = np.deg2rad(y)
    zr = np.deg2rad(z)

    rotation_alpha = [
        [1, 0, 0],
        [0, np.cos(xr), -np.sin(xr)],
        [0, np.sin(xr), np.cos(xr)],
    ]
    rx = np.array(rotation_alpha)

    rotation_beta = [
        [np.cos(yr), 0, np.sin(yr)],
        [0, 1, 0],
        [-np.sin(yr), 0, np.cos(yr)]
    ]
    ry = np.array(rotation_beta)

    rotation_gamma = [
        [np.cos(zr), -np.sin(zr), 0,],
        [np.sin(zr), np.cos(zr), 0],
        [0, 0, 1],
    ]
    rz = np.array(rotation_gamma)

    return np.matmul(np.matmul(rx, ry), rz)


if __name__ == '__main__':
    adj = tuple([0.25, 1.1])  # brightness add 0.2 then multiply by 1.1.
    p = Photo()
    p.load('world.topo.bathy.200406.3x5400x2700')
    p.set_latlon([-90., 90.], [-180., 180.])
    Tri.set_sh(81.)  # 3^6
    Tri.set_photo(p)
    np.set_printoptions(precision=12, suppress=True)
    # This now centres and 'northifies' UK. via NA/HH2/T2
    # alt_mat = rot(14, -25, 66)
    sphere = IcoSphere('assets/maps/phi.json', None)
    a, h = Tri.sh
    lines = sphere.grid[3] - sphere.grid[1]
    columns = sphere.grid[2] - sphere.grid[0]
    dims = columns * a * 0.5, lines * h
    dx2 = Drawing('uk_h9_3', dims, False)

    # So sphere is 1xH9 divided into a bunch of 20 triangles.
    # The render of each triangle is what we need to do for that.

    for side in sphere.sides.values():
        for hh in range(3):
            d0 = side.get_hh(hh)
            d0.add_districts()
            for d1 in d0.districts.values():
                d1.add_districts()
                for d2 in d1.districts.values():
                    d2.add_districts()
                    for d3 in d2.districts.values():
                        draw_hh(dx2, d3, adj)
    dx2.save()

