import numpy as np
from modern.octahedron import Octahedron
from modern.util import Util
from numpy.typing import NDArray


class VisualGridOctant:
    """
     The 2D visual projection of Octant as a component of the VisualGrid
    """
    def __init__(self, octant, offset=None, theta=None):
        self.octant = octant
        self.name = octant.name
        self.offset = offset
        self.theta = theta
        self.util = Util()

    def place(self, pts: NDArray[np.float64]):
        """
        place points for this octant.
        3D points are first flattened.
        Points are oriented and then translated accordingly.
        """
        _pts = pts if pts.shape[-1] == 2 else self.octant.flatten(pts)
        return _pts @ self.util.r2d(self.theta) + self.offset

    def unplace(self, pts: NDArray[np.float64], unflatten=False):
        """
        Inverse of place - move points to barycentre for this octant.
        If unflatten, return 3D Octant
        """
        result = (pts - self.offset) @ self.util.r2d(-self.theta)
        if unflatten:
            return self.octant.unflatten(result)
        else:
            return result


class OctahedronNet:
    """
    This is a 2D visual representation of a flattened Octahedron.
    """
    def __init__(self, o: Octahedron):
        self.o = o
        self.rt = np.pi / 3  # grid rotation in 60º
        self.gw = Octahedron.r2 / 2.  # grid unit width
        self.gh = Octahedron.r6 / 6.  # grid unit height
        # Current projection has a maximum width of 3.5 and height of 9.
        self.glx, self.gly = (0, Octahedron.r2 * 3.5), (0, self.gh * 9)
        self.faces = {k: VisualGridOctant(v) for k, v in o.faces.items()}
        grid = {
            'NEA': (00, 3., 4.),
            'NEP': (-1, 4., 5.),
            'NWA': (+1, 2., 5.),
            'NWP': (+2, 2., 7.),
            'SEA': (+3, 3., 2.),
            'SEP': (+2, 5., 4.),
            'SWA': (-2, 1., 4.),
            'SWP': (+3, 6., 5.)
        }
        for k, (theta, x, y) in grid.items():
            self.faces[k].theta = theta * self.rt  # adjust from barycentric
            self.faces[k].offset = x * self.gw, y * self.gh

    def side(self, pt: NDArray[np.float64]):
        """
        Identify which side is being addressed by a flat coordinate.
        This depends upon the current 2d projection.
        Most of this is managed following 60º lines.
        """
        ax, ay = pt
        gh = self.gh * 3
        gy = ay // self.gh
        dẋ = self.o.r3 * ax
        if ay - gh <= dẋ <= ay + 5 * gh:  # We are in legal space...
            if dẋ <= ay + gh:  # We are in left-3 triangles
                if 5 * gh - dẋ > ay and gy > 2:
                    if 3 * gh - dẋ < ay:
                        if gy >= 6:
                            return 'NWP'
                        return 'NWA'
                    return 'SWA'
                return None
            if gy <= 5 and ay >= 3 * gh - dẋ:  # inside remaining 5
                if dẋ <= ay + 3 * gh:  # We are in mid-3 triangles
                    if gy <= 2:
                        return 'SEA'
                    if 5 * gh - dẋ > ay:
                        return 'NEA'
                    return 'NEP'
                if gy >= 3:
                    if 7 * gh - dẋ > ay:
                        return 'SEP'  # final 2 triangles
                    return 'SWP'
        return None

    def bin_points(self, xys: NDArray[np.float64]):
        """
        :param xys: NDArray of points to be binned.
        should preserve structure.
        returns NDArray of binned and placed points.
        """
        bins = {k: [] for k in self.faces.keys()}
        if xys.shape[-1] == 3:  # Seems like a reasonable way of identifying 2D/3D.
            bins3d = self.o.bin_points(np.array(xys))
            for k, v in bins3d.items():
                pts2 = self.o.faces[k].flatten(v)
                bins[k] = self.faces[k].place(pts2)
        else:
            for pt in xys:
                s = self.side(pt)
                if s is not None:
                    bins[s].append(pt)
        return bins


if __name__ == '__main__':
    from modern.display import Display
    u = Util()
    octa = Octahedron()
    onet = OctahedronNet(octa)
    ptg = u.oct_rnd(10000)

    binned = onet.bin_points(ptg)
    cols = Display.colours(10, 'tab10')
    face_col = {face: cols[i] for i, face in enumerate(octa.faces.keys())}
    cx = []
    px = []
    for f in binned:
        cp = face_col[f]
        cx += [cp for i in range(len(binned[f]))]
        px.append(binned[f])
    pts = np.vstack(px)
    Display.col_pts_2d(pts, cx, onet.glx, onet.gly)
    p3 = []
    for f, rp in binned.items():
        pf = onet.faces[f].unplace(rp, True)
        p3.append(pf)
    Display.show_pts_3d(np.vstack(p3), (-1.1, 1.1), (-1.1, 1.1), (-1.1, 1.1))
