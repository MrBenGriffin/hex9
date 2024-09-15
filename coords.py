import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import math
import numpy as np
import svg
import json
from photo import Photo


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
        th = math.acos(dot)
        s = math.sin(th)
        j = math.sin((1. - t) * th)
        k = math.sin(t * th)
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
        self.col = self.__class__.photo.col(*self.sp)

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
        self.name = obj['name']
        self.ll = obj['ll']
        self.xyz = obj['xyz']
        self.sp = obj['sp']


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
    def __init__(self, file: str):
        self.vertices = {}
        self.mapping = {}
        self.sides = {}
        with (open(file, 'r') as infile):
            obj = json.load(infile)
            infile.close()
            for idx, vertex in enumerate(obj['vertices']):
                self.vertices[idx] = IcoVertex(idx, vertex)
            if 'mapping' in obj and obj['mapping']:
                self.mapping = obj['mapping']
            # Each side is set to be either 'up' or 'down'
            # But each side is made up of three half-hexes 0,1,2
            if 'sides' in obj and obj['sides']:
                self.sides = {side['name']: IcoSide(side, self.vertices) for side in obj['sides']}


class Drawing:
    def __init__(self, colour_tweak: tuple = (0.0, 1.0), grid: tuple = (0, 0, 11, 4)):
        self.cp, self.cm = colour_tweak  # add color cp then multiply by cm.
        self.graph = None
        a, h = Tri.sh
        lines = grid[3] - grid[1]    # lines (y-steps) are defined by 1..3
        columns = grid[2] - grid[0]  # cols  (x-steps) are defined by 0..2 -> [x0,y0,x2,y2]
        x_off = a * 0.5 * grid[0]
        y_off = h * grid[1]
        v_height = lines * h
        v_width = columns * a * 0.5
        vb_min_x, vb_min_y, vb_max_x, vb_max_y = x_off, y_off, v_width, v_height
        self.maxy = vb_max_y
        self.canvas = svg.SVG(
            viewBox=svg.ViewBoxSpec(vb_min_x, vb_min_y, vb_max_x, vb_max_y),
            width=v_width, height=v_height, elements=[]
        )
        self.defs = svg.Defs(elements=[])
        self.canvas.elements.append(self.defs)

    def tri(self, t: Tri):
        # <polygon points="100,100 150,25 150,75 200,0" fill="none" stroke="black" />
        r, g, b = [int(self.cm * (t+self.cp)) for t in t.col]
        tx = svg.Polygon(fill=f'rgb({r},{g},{b})', stroke="none")
        tx.points = [tuple(pt) for pt in t.gxy]
        self.canvas.elements.append(tx)

    def hh(self, h: HalfHex):
        # <polygon points="100,100 150,25 150,75 200,0" fill="none" stroke="black" />
        r, g, b = [int(self.cm * (h+self.cp)) for h in h.col]
        tx = svg.Polygon(fill=f'rgb({r},{g},{b})', stroke="none")
        tx.points = [tuple(pt) for pt in h.gxy[:-1]]
        self.canvas.elements.append(tx)

    def add(self, thing):
        self.canvas.elements.append(thing)

    def define(self, thing):
        self.defs.elements.append(thing)

    def save(self, f_name: str = 'result'):
        f = open(f"{f_name}.svg", "w")
        f.write(self.canvas.as_str())
        f.close()


if __name__ == '__main__':
    # TODO:  implement these using h9
    adj = tuple([0.2, 1.1])  # brightness add 0.2 then multiply by 1.1.
    p = Photo()
    p.load('world.topo.bathy.200406.3x5400x2700')
    p.set_latlon([-90., 90.], [-180., 180.])
    Tri.set_sh(729.)  # 3^6
    Tri.set_photo(p)
    np.set_printoptions(precision=12, suppress=True)
    dym = IcoSphere('assets/maps/fuller.json')
    # side = dym.sides['North Atlantic']
    # gx, gy = side.grid  # [6,2]
    # bounds = tuple([gx-1, gy, gx+1, gy+1])
    # dx0 = Drawing(adj, bounds)
    # dx0 = Drawing(adj, )
    dx2 = Drawing(adj, )
    # dx2 = Drawing(adj, )
    # dx3 = Drawing(adj, )
    # dx4 = Drawing(adj, )
    # dx5 = Drawing(adj, )
    # dx6 = Drawing(adj, )
    for side in dym.sides.values():
        for i in range(3):
            d0 = side.get_hh(i)
            # dx0.hh(d0)
            d0.add_districts()
            for d1 in d0.districts.values():
                # dx1.hh(d1)
                d1.add_districts()
                for d2 in d1.districts.values():
                    dx2.hh(d2)
                    # d2.add_districts()
                    # for d3 in d2.districts.values():
                #         dx2.hh(d3)
                #         d3.add_districts()
                #         for d4 in d3.districts.values():
                #             # dx4.hh(d4)
                #             d4.add_districts()
                #             for d5 in d4.districts.values():
                #                 dx5.hh(d5)
                #             #     d5.add_districts()
                #             #     for d6 in d5.districts.values():
                #             #         dx6.hh(d6)
    # dx0.save('output/globe_hh0')
    dx2.save('output/globe_hh1')
    # dx2.save('output/globe_hh2')
    # dx3.save('output/globe_hh3')
    # dx4.save('output/globe_hh4')
    # dx5.save('output/globe_hh5')
    # dx6.save('output/globe_hh6')
