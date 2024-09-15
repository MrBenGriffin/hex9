from abc import ABC, abstractmethod
import numpy as np
import os

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2


class Flat(ABC):
    def __init__(self, width: int, height: int):
        self.bounds = tuple([width, height])
        self.width, self.height = width, height

    @property
    def bounds(self) -> tuple[int, int]:
        return self._width_height

    @bounds.setter
    def bounds(self, value: tuple[int, int]):
        self._width_height = value

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, value: int):
        self._width = value

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, value: int):
        self._height = value


class LatLonSampler(ABC):
    @abstractmethod
    def sample(self, point: tuple[float, float]):
        raise NotImplementedError


class Sampler(ABC):
    @abstractmethod
    def sample(self, point: tuple):
        ...


class FlatValues(Flat, Sampler, ABC):
    def __init__(self, ground: np.array):
        height, width = ground.shape[:2]
        super().__init__(width, height)
        self.values = ground

    def set_pt(self, x, y, sample):
        if min(self.height - 1, max(0, y)) != y or min(self.width - 1, max(0, x)) != x:
            return
        self.values[x, y] = sample

    def sample(self, point: tuple[int, int]):
        x, y = point
        if min(self.height - 1, max(0, y)) != y or min(self.width - 1, max(0, x)) != x:
            return 0, 0, 0
        return self.values[min(self.height - 1, max(0, y)), min(self.width - 1, max(0, x))]


class EPSG32662(LatLonSampler):  # WGS84 Plate Carrée

    def __init__(self, fv: FlatValues, lat: tuple[float, float], lon: tuple[float, float]):
        self.fv = fv
        lat_from, lat_to = lat
        lon_from, lon_to = lon
        self.lat_flip = lat_from > lat_to
        self.lon_flip = lon_from > lon_to
        self.lat = np.linspace(lat_from, lat_to, num=fv.height, endpoint=True)
        self.lon = np.linspace(lon_from, lon_to, num=fv.width, endpoint=True)
        if self.lat_flip:
            self.lat = np.flip(self.lat)
        if self.lon_flip:
            self.lon = np.flip(self.lon)

    def sample(self, point: tuple[float, float]):  # given in lat/lon
        lat, lon = point
        w = np.searchsorted(self.lon, lon) % self.fv.width
        h = np.searchsorted(self.lat, lat) % self.fv.height
        h = self.fv.height - h - 1 if self.lat_flip else h
        w = self.fv.width - w - 1 if self.lon_flip else w
        return self.fv.sample((w, h))


class CVFlatValues(FlatValues):

    def __init__(self, samples: np.array):
        super().__init__(samples)

    @classmethod
    def load(cls, f_name: str) -> 'CVFlatValues':
        file = cv2.imread(f_name)
        return cls(file)

    @classmethod
    def new(cls, dim: tuple[int, int]) -> 'CVFlatValues':
        x, y = dim
        blank = np.zeros((y, x), np.uint8)
        space = cv2.cvtColor(blank, cv2.COLOR_GRAY2RGB)
        return cls(space)

    def poly(self, points, col):
        c = np.array(col, dtype='uint8').tolist()
        cv2.fillPoly(self.values, pts=points, color=c)

    def show(self, wait: bool = True):
        cv2.imshow('canvas', self.values)
        if wait:
            cv2.waitKey(0)

    def save(self, fn: str = 'output'):
        cv2.imwrite(f'output/{fn}.png', self.values)


class UnitSphere:
    # Currently this is boundless - ie, all values can be reached.
    # This can use various types of coordinate.
    # I need to consider which is best.
    # bearing in mind my primary function will be slerp,
    # it seems to make sense to use unit vectors.
    # self.xyz = tuple([vertices[d].xyz for d in self.vx])  # This is the unit sphere triangle.
    def __init__(self, source: LatLonSampler):
        self.source = source

    @classmethod
    def xyz_ll(cls, xyz: tuple) -> tuple[float, float]:  # given a vector, return the latitude/longitude
        x, y, z = xyz
        return np.degrees(np.arctan2(z, np.sqrt(x * x + y * y))), np.degrees(np.arctan2(y, x))

    def sample(self, point: tuple[float, float, float]):  # given in spherical coordinates
        ll = self.xyz_ll(point)
        return self.source.sample(ll)


class Tri:
    _tc = [
        (3, 0, 2), (2, 0, 9), (2, 9, 1),
        (6, 0, 5), (5, 0, 3), (5, 3, 4),
        (9, 0, 8), (8, 0, 6), (8, 6, 7)
    ]

    @classmethod
    def slerp(cls, p0, p1, t):
        # works only with unit vectors.
        dot = np.dot(p0, p1)
        th = np.arccos(dot)
        s = np.sin(th)
        j = np.sin((1. - t) * th)
        k = np.sin(t * th)
        return (j * np.array(p0) + k * np.array(p1)) / s

    def __init__(self, ijk, abc):
        # ijk is a tuple/list of three cartesian points on a unit sphere in clockwise order.
        # abc is a tuple/list of three 2D cartesian points
        # Need to calculate the seven remaining points, clockwise starting at the centre, then significant point (^ or V for up/down resp)
        self.ijk = ijk  # these are unit_sphere vectors.
        self.abc = abc  # These are equivalent 2D points.
        self.c = np.mean(ijk, axis=0)
        self.src = None
        self.color = None
        self.uvs = None
        self.xyp = None
        self.chn = []

    def _set_uvs(self):
        i, j, k = self.ijk
        _pts = [
            self.c,  # centre.
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
        self.uvs = np.array(_pts) / np.linalg.norm(_pts, axis=1, keepdims=True)  # each sub_triangle.

    def _set_xyp(self):
        i, j, k = self.abc
        self.xyp = tuple([
            np.mean([i, j, k], axis=0),  # centre.
            i,  # pt i
            np.mean([i, i, j], axis=0),  # iij.
            np.mean([i, j, j], axis=0),  # ijj.
            j,  # p_j.
            np.mean([j, j, k], axis=0),  # jjk.
            np.mean([j, k, k], axis=0),  # jkk.
            k,  # p_k.
            np.mean([k, k, i], axis=0),  # kki.
            np.mean([k, i, i], axis=0)   # kii.
        ])

    def create_chn(self, depth):
        if depth <= 0 or self.uvs:
            return
        self._set_uvs()
        self._set_xyp()
        for a in self._tc:
            uv = tuple([self.uvs[i] for i in a])
            xy = tuple([self.xyp[i] for i in a])
            c = Tri(uv, xy)
            if self.src:
                c.set_src(self.src)
            c.create_chn(depth - 1)
            self.chn.append(c)

    def set_src(self, us: UnitSphere):
        self.src = us
        self.color = self.src.sample(self.c)

    def poly_color(self):
        result = []
        if self.chn:
            for c in self.chn:
                result += c.poly_color()
        else:
            result.append(tuple([np.array([self.abc], dtype=np.int32), self.color]))
        return result


def do_huge_spherical_triangle():
    # This uses the central sub-triangle of the 'phi' map.
    src = CVFlatValues.load(f'assets/90WE0_90N_21600.png')
    globe = EPSG32662(src, (90., 0.), (-45., 45.))
    width = 8000
    height = int(np.round(np.sqrt(3.) / 2. * width))
    disp = CVFlatValues.new((width+1, height+1))

    # uk_uv = [[0.0, -0.5257311121191336, 0.85065080835204], [0.0, 0.5257311121191336, 0.85065080835204], [0.85065080835204, 0.0, 0.5257311121191336]]
    uk_uv = [[0.63994974, -0.21203128,  0.73858451], [0.35682209, 0., 0.93417236], [0.63994974, 0.21203128, 0.73858451]]
    uk_pt = [[0, height], [width >> 1, 0], [width, height]]
    uk = Tri(uk_uv, uk_pt)
    uk.set_src(UnitSphere(globe))
    uk.create_chn(8)
    pc_list = uk.poly_color()
    for ppts, color in pc_list:
        disp.poly(ppts, color)
    disp.save('huge_triangle')


def do_big_flat():
    src = CVFlatValues.load(f'assets/90WE0_90N_21600.png')
    globe = EPSG32662(src, (90., 0.), (-45., 45.))
    img_wid, img_hgt = 2400, 2400
    fsr = CVFlatValues.new((img_wid, img_hgt))
    la, lo = 51.58788025819696, -0.09624712666336593
    dms = 15.0
    lat_from, lat_to = la + dms, la - dms
    lon_from, lon_to = lo - dms, lo + dms
    lat = np.linspace(lat_from, lat_to, num=img_hgt, endpoint=True)
    lon = np.linspace(lon_from, lon_to, num=img_wid, endpoint=True)
    for x, lai in enumerate(lat):
        for y, loi in enumerate(lon):
            px = globe.sample((lai, loi))
            fsr.set_pt(x, y, px)
    fsr.show(True)


def do_flat_samples():
    # import cv2
    src = CVFlatValues.load(f'assets/world.topo.bathy.200406.3x5400x2700.png')
    globe = EPSG32662(src, (90., -90.), (-180., 180.))
    fsr = CVFlatValues.new((1000, 1000))
    la, lo = 51.58788025819696, -0.09624712666336593
    dms = 15.0
    lat_from, lat_to = la + dms, la - dms
    lon_from, lon_to = lo - dms, lo + dms
    lat = np.linspace(lat_from, lat_to, num=1000, endpoint=True)
    lon = np.linspace(lon_from, lon_to, num=1000, endpoint=True)
    for x, lai in enumerate(lat):
        for y, loi in enumerate(lon):
            px = globe.sample((lai, loi))
            fsr.set_pt(x, y, px)
    cv2.imshow('fsr', fsr.values)
    cv2.waitKey(0)


def do_spherical_triangle():
    src = CVFlatValues.load(f'assets/world.topo.bathy.200406.3x5400x2700.png')
    world = EPSG32662(src, (90., -90.), (-180., 180.))
    width = 1000
    height = int(np.round(np.sqrt(3.) / 2. * width))
    disp = CVFlatValues.new((width+1, height+1))

    uk_uv = [[0.0, -0.5257311121191336, 0.85065080835204], [0.0, 0.5257311121191336, 0.85065080835204], [0.85065080835204, 0.0, 0.5257311121191336]]
    uk_pt = [[0, 0], [1000, 0], [500, height]]
    uk = Tri(uk_uv, uk_pt)
    uk.set_src(UnitSphere(world))
    uk.create_chn(5)
    pc_list = uk.poly_color()
    for ppts, color in pc_list:
        disp.poly(ppts, color)
    disp.show(True)


if __name__ == '__main__':
    do_spherical_triangle()
