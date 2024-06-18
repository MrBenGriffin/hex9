import math
import numpy as np
import matplotlib as mpl
import svg
import scipy.spatial as spatial
from scipy.spatial.transform import Rotation
import json
from random import random
import cv2


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
        # self.col = self.triangles[1].col

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
            np.mean([i, i, j], axis=0),  # iij.
            np.mean([i, j, j], axis=0),  # ijj.
            j,  # p_j.
            np.mean([j, j, k], axis=0),  # jjk.
            np.mean([j, k, k], axis=0),  # jkk.
            k,  # p_k.
            np.mean([k, k, i], axis=0),  # kki.
            np.mean([k, i, i], axis=0)  # kii.
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

    # def map(self, mapping: str):
    #     if mapping in self.mapping:
    #         return np.array([self.xyz[m] for m in self.mapping[mapping]])


class Drawing:
    def __init__(self):
        a, h = Tri.sh
        lines = 4
        columns = 11  # half-width per column
        v_height = lines * h
        v_width = columns * a * 0.5
        vb_min_x, vb_min_y, vb_max_x, vb_max_y = 0, 0, v_width, v_height
        self.maxy = vb_max_y
        self.canvas = svg.SVG(
            # ViewBoxSpec are the coordinates used in the world of the drawing
            # width/height are the size of the rendered svg.
            viewBox=svg.ViewBoxSpec(vb_min_x, vb_min_y, vb_max_x, vb_max_y),
            width=v_width, height=v_height, elements=[]
        )

    def tri(self, t: Tri):
        # <polygon points="100,100 150,25 150,75 200,0" fill="none" stroke="black" />
        r, g, b = [int(t) for t in t.col]
        tx = svg.Polygon(fill=f'rgb({r},{g},{b})', stroke="none")
        tx.points = [tuple(p) for p in t.gxy]
        self.canvas.elements.append(tx)

    def hh(self, h: HalfHex):
        # <polygon points="100,100 150,25 150,75 200,0" fill="none" stroke="black" />
        r, g, b = [int(i) for i in h.col]
        tx = svg.Polygon(fill=f'rgb({r},{g},{b})', stroke="none")
        tx.points = [tuple(p) for p in h.gxy[:-1]]
        self.canvas.elements.append(tx)

    def save(self, name: str = 'output'):
        f = open(f"{name}.svg", "w")
        f.write(self.canvas.as_str())
        f.close()


class Photo:
    def __init__(self):
        # b1f = 'world.topo.bathy.200407.3x21600x21600.B1.png'
        # c1f = 'world.topo.bathy.200407.3x21600x21600.C1.png'
        # b1i = cv2.imread(b1f)
        # c1i = cv2.imread(c1f)
        # img = cv2.hconcat([b1i, c1i])
        # self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # self.height, self.width = self.img.shape[:2]
        # # 90N90W - 0N90E
        # # I got this South to North... No wonder my grid is backwards.
        # self.lat = np.linspace(0., 90., num=self.height, endpoint=True)
        # self.lon = np.linspace(-90., 90., num=self.width, endpoint=True)

        img = cv2.imread('world.topo.bathy.200406.3x5400x2700.png')
        grid_col = (255, 255, 255)
        grid_thk = 5
        gr, gc = 30, 60
        self.height, self.width = img.shape[:2]
        dy, dx = self.height / gr, self.width / gc
        for x in np.linspace(start=dx, stop=self.width - dx, num=gc - 1):
            x = int(round(x))
            cv2.line(img, (x, 0), (x, self.height), color=grid_col, thickness=grid_thk)

        # draw horizontal lines
        for y in np.linspace(start=dy, stop=self.height - dy, num=gr - 1):
            y = int(round(y))
            cv2.line(img, (0, y), (self.width, y), color=grid_col, thickness=grid_thk)

        cv2.imwrite('world.png', img)
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Latitude measures the distance north or south of the equator.
        self.lat = np.linspace(-90., 90., num=self.height, endpoint=True)
        self.lon = np.linspace(-180., 180., num=self.width, endpoint=True)

    def col(self, lat: float, lon: float):
        w = np.searchsorted(self.lon, lon) % self.width
        h = np.searchsorted(self.lat, lat) % self.height
        pixel = self.img[self.height - h - 1, w]
        return pixel


if __name__ == '__main__':
    p = Photo()
    Tri.set_sh(729.)  # 3^6
    Tri.set_photo(p)
    np.set_printoptions(precision=12, suppress=True)
    dym = IcoSphere('fuller.json')
    # et = dym.sides['North Atlantic']  # 'Europe' Up Triangle [Liberia,Norway,Arabian]
    # na = et.get_hh(0, 0)   # uk and w. europe hh 7 should be england.
    # en = na.triangles[2].get_hh(2, 1)
    # se = en.triangles[1].get_hh(0, 2)
    # lb = se.triangles[2].get_hh(1, 3)      # london, camberly, oxford, birmingham
    # ln = lb.triangles[0].get_hh(2, 4)      # st.Albans,gillingham,caterham,greenford.
    # nl = ln.triangles[2].get_hh(2, 5)
    # hy = nl.triangles[1].get_hh(1, 6)      #
    # dg = hy.triangles[0].get_hh(2, 7)
    # dg.add_districts()
    # hm = dg.districts[5]        # home district! falkland rd/duckets common/waldeck-carling+/vincent-linden+
    # for pt in hm.pts:
    #     print(xyz_ll(pt))

    dx = Drawing()
    file_name = 'd4_lines'
    for side in dym.sides.values():
        for i in range(3):
            d0 = side.get_hh(i)
            d0.add_districts()
            for d1 in d0.districts.values():
                d1.add_districts()
                for d2 in d1.districts.values():
                    d2.add_districts()
                    for d3 in d2.districts.values():
                        d3.add_districts()
                        for d4 in d3.districts.values():
                            dx.hh(d4)

    # side = dym.sides['North Atlantic']
    # d0 = side.get_hh(0)
    # file_name = 'uk_hh_6'
    # d0 = side.get_hh(0).triangles[2].get_hh(2)
    # d0.add_districts()
    # for d1 in d0.districts.values():
    #     d1.add_districts()
    #     for d2 in d1.districts.values():
    #         d2.add_districts()
    #         for d3 in d2.districts.values():
    #             d3.add_districts()
    #             for d4 in d3.districts.values():
    #                 d4.add_districts()
    #                 for d5 in d4.districts.values():
    #                     d5.add_districts()
    #                     for d6 in d5.districts.values():
    #                         dx.hh(d6)

    dx.save(file_name)
