"""
Part of the H9 project
"""
import numpy as np
from numpy.typing import NDArray
from hhg9 import Points
from hhg9.base import CompositeDomain, ComponentDomain
from hhg9.base.point_format import PointFormat
from hhg9.projections import BaryNet


class OctantNet(ComponentDomain):
    """
    This a 2D side of an Octant that belongs to a Net.
    Validity should be easy enough since we have the 3 points that define it.
    """
    def __init__(self, registrar, dom, name: str, sign: tuple, mode: str):
        super().__init__(registrar, name)
        self.dom = dom
        self.mode = mode
        self._sign = sign

    def sig(self) -> tuple:
        return self._sign

    def valid(self, pts: NDArray) -> NDArray:
        """
        Return an array of bools according to the validity criterion
        :param pts: set of 3d Euclidean points
        """
        raise NotImplementedError


class OctahedralNet(CompositeDomain):
    """
    This is a 2d net correlate of the Octahedron.
    Triangles have edge length √2 in a unit octahedron
    """

    def __init__(self, registrar, octahedron, b_oct):
        super().__init__(registrar, 'n_oct')
        self.o = octahedron
        self.rt = np.pi / 3  # grid rotation in 60º
        self.gw = np.sqrt(2) / 2.  # grid unit width
        self.r3 = np.sqrt(3)
        self.gh = np.sqrt(6) / 6.  # grid unit height OctahedralNet
        self.width, self.height = np.sqrt(2) * 3.5, self.gh * 9
        self.glx, self.gly = (0, self.width), (0, self.height)
        self.sides = {}
        self.projs = {}
        grid = {
            (+1, +1, +1): (3., 4., 3),  # NEA
            (-1, +1, +1): (4., 5., 5),  # NEP
            (+1, -1, +1): (2., 5., 1),  # NWA
            (-1, -1, +1): (2., 7., 5),  # NWP
            (+1, +1, -1): (3., 2., 3),  # SEA
            (-1, +1, -1): (5., 4., 5),  # SEP
            (+1, -1, -1): (1., 4., 1),  # SWA
            (-1, -1, -1): (6., 5., 3)   # SWP
        }
        for sign, val in grid.items():
            side = self.o.signs[sign]
            bary = b_oct.sides[side]
            gx, gy, th = val
            n_theta = (th % 6) * np.pi/3.
            o_theta = (n_theta + bary.th) % np.pi
            mode = {'V': 'Λ', 'Λ':'V'}[bary.mode]  # TODO Fix properly.
            n_sig = f'{self.name}:{side}'
            b_sig = f'{b_oct.name}:{side}'

            self.sides[sign] = OctantNet(registrar, self, n_sig, sign, mode)
            self.projs[side] = BaryNet(registrar, side, b_sig, n_sig, n_theta, (gx * self.gw, gy * self.gh))
            self.components[sign] = self.sides[sign]
        init = True

    def ratio(self):
        """Return width/height ratio"""
        return self.glx[1]/self.gly[1]

    def valid(self, _pts: NDArray) -> NDArray:
        """
        Test that points are valid
        """
        if _pts.shape[-1] < 2:
            raise ValueError('Points must have 2 dimensions')
        gh3 = self.gh * 3
        x, y, g = _pts[..., -2] * self.r3, _pts[..., -1], _pts[..., -1] // self.gh
        return (
                ((y - gh3) <= x) & (x <= (y + 5 * gh3)) &  # We are in legal space...
                (
                    ((x <= (y + gh3)) & ((5 * gh3 - x) > y) & (g > 2)) |  # left 3 triangles.
                    (((g <= 5) & (y >= (3 * gh3 - x))) & ((x <= (y + 3 * gh3)) | (g >= 3)))
                )
        )

    def _pt_face(self, pt: NDArray) -> tuple | None:
        """
        Identify which side is being addressed by a flat coordinate.
        This depends upon the current 2d projection.
        Most of this is managed following 60º lines.
        """
        bad = 0, 0, 0
        ax, ay = pt
        gh3 = self.gh * 3
        gy = ay // self.gh
        dẋ = self.r3 * ax
        if ay - gh3 <= dẋ <= ay + 5 * gh3:  # We are in legal space...
            if dẋ <= ay + gh3:  # We are in left-3 triangles
                if 5 * gh3 - dẋ > ay and gy > 2:
                    if 3 * gh3 - dẋ < ay:
                        if gy >= 6:
                            return -1, -1, 1  # 'NWP'
                        return 1, -1, 1  # 'NWA'
                    return 1, -1, -1  # 'SWA'
                return bad
            if gy <= 5 and ay >= 3 * gh3 - dẋ:  # inside remaining 5
                if dẋ <= ay + 3 * gh3:  # We are in mid-3 triangles
                    if gy <= 2:
                        return 1, 1, -1  # 'SEA'
                    if 5 * gh3 - dẋ > ay:
                        return 1, 1, 1  # 'NEA'
                    return -1, 1, 1  # 'NEP'
                if gy >= 3:
                    if 7 * gh3 - dẋ > ay:
                        return -1, 1, -1  # 'SEP'  # final 2 triangles
                    return -1, -1, -1  # 'SWP'
        return bad

    def adopt(self, pts: NDArray):
        """
        Take an array and adopt as this domain.
        """
        good = self.where_valid(pts)
        pts = Points(good, domain=self)
        cmp = np.apply_along_axis(self._pt_face, -1, pts.coords)
        pts.components = np.array(cmp, dtype='b')
        return pts

    def register_format(self, af: PointFormat):
        """Decorator to register an AddressFormat for each component."""
        for side in self.sides:
            self.sides[side].register_format(af)

    @classmethod
    def image(cls, pts: Points, dim=None) -> NDArray:
        """
        return the image that these points represent.
        """
        xs, ys = pts.coords[:, 0], pts.coords[:, 1]
        if dim is None:
            ux, uy = np.unique(xs, axis=0), np.unique(ys, axis=0)
            w = ux.size
            h = uy.size
        else:
            w, h = dim
        x0 = np.min(xs)
        y0 = np.min(ys)
        y_adj = (h-1e-6)/(np.max(ys)-y0)
        x_adj = (w-1e-6)/(np.max(xs)-x0)
        yy = np.floor(y_adj*(ys-y0)).astype(np.uint64)
        xx = np.floor(x_adj*(xs-x0)).astype(np.uint64)
        ch = pts.samples
        y = (h - 1) - yy.astype(np.uint64)  # still in cartesian (ie, 0 is bottom left).
        x = xx.astype(np.uint64)
        channels = 1 if ch.ndim == 1 else ch.shape[1]
        ch = ch.reshape(-1, channels)
        img = np.ones((h, w, channels), dtype=ch.dtype)
        img[y, x] = ch
        return img
