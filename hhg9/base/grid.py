import numpy as np

"""
Grid Generation Class
"""


class Grid:
    """Generator Function"""

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
            [(1, 2, 9), True], [(0, 9, 2), False], [(2, 3, 0), True],
            [(5, 0, 3), False], [(3, 4, 5), True], [(0, 5, 6), True],
            [(6, 8, 0), False], [(8, 6, 7), True], [(9, 0, 8), True]
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
                pts = [tuple([h + ofx for h in hh]) for hh in ref]
                r2 += pts
        else:
            for t, o in trx:
                ud = o if up else not o
                self.t_sub(tuple([xy[i] for i in t]), r1, r2, n - 1, ud)

    def t_grid(self, n, abc):
        """Generates a grid of points within a triangle."""
        r1, r2 = [], []
        self.t_sub(abc, r1, r2, n)
        return np.array(r1), np.array(r2)

    def e_grid(self, n, a=1.0):
        """Generates a grid of points within an equilateral"""
        e = 1e-40
        hx = a / 2.0
        th = a * np.sqrt(3) / 6.0
        abc = (-hx, -th), (e, 2 * th), (hx, -th)
        # print(abc)
        grid, _ = self.t_grid(n, abc)
        return np.unique(grid)

    def hh_grid(self, n, a=1.0):
        """Generates a grid of points within an equilateral - with half-hex polys"""
        tol = 0.5 * (a / (3 ** (n + 1)))
        e = 1e-200
        hx = a / 2.0
        th = a * np.sqrt(3) / 6.0
        abc = (-hx, -th), (e, 2 * th), (hx, -th)
        grid, hhx = self.t_grid(n, abc)
        gx, oi, ri = np.unique(np.floor(grid / tol).astype(int), return_index=True, return_inverse=True, axis=0)
        hp = ri[hhx]
        gd = grid[oi]
        return gd, hp  # grid points, hex_points.

    @classmethod
    def sq_grid(cls, scale: float = 1000, ud: str = 'Λ'):
        """
        Return a rectilinear grid of points within an equilateral triangle centered
        at (cx, cy) in pixel space conforming to the barycentric projection of the side of a unit octahedron.
        When calling this for a net, remember to use the net's ΛV not the barycentric!
        """
        from hhg9 import H9Engine

        h9 = H9Engine()
        wid = scale
        hgt = int(scale * h9.RH)
        # generate a covering rectangle.
        fl, cl = (h9.ΛF, h9.ΛC) if ud == 'Λ' else (h9.VF, h9.VC)
        yl = np.linspace(fl, cl, num=hgt)
        xl = np.linspace(h9.TL, h9.TR, num=wid)
        xx, yy = np.meshgrid(xl, yl)
        rec = np.stack((xx.ravel(), yy.ravel()), axis=1)
        # restrict by validity.
        trx = h9.valid(rec, ud)
        return rec[trx]

    @classmethod
    def in_convex_poly(cls, points, poly):
        """
        Vectorized check if each point in `points` is inside the convex polygon.

        Parameters:
            points: (n, 2) NumPy array of n points to test.
            poly: List or array of n points (x, y) in clockwise or counter-clockwise order.

        Returns:
            A boolean NumPy array of length n indicating for each point whether it is inside.
        """
        poly = np.asarray(poly)
        points = np.atleast_2d(points)  # Ensure shape (N, 2)
        eps = 1e-200
        def cross2d(a, b):
            """Compute the 2D cross product: a_x * b_y - a_y * b_x"""
            return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

        n = points.shape[0]
        pn = poly.shape[0]
        inside = np.ones(n, dtype=bool)

        for i in range(pn):
            a = poly[i]
            b = poly[(i + 1) % pn]
            ab = b - a
            ap = points - a
            cp = cross2d(np.tile(ab, (n, 1)), ap)
            inside &= (cp >= -eps) if np.all(cp >= -eps) else (cp <= eps)
        return inside

    @classmethod
    def qa_grid(cls, quad, scale: float = 1000):
        """
        Return a rectilinear grid of points within a quadrilateral.
        Also returns the mask and scales.
        """
        quad = np.asarray(quad)
        minx, miny, maxx, maxy = quad[..., 0].min(), quad[..., 1].min(), quad[..., 0].max(), quad[..., 1].max()
        w = maxx-minx
        h = maxy-miny
        wid = scale  # np.uint32(scale / w)
        hgt = np.uint32((h/w) * wid)
        yl = np.linspace(maxy, miny, num=hgt)
        xl = np.linspace(minx, maxx, num=wid)
        xx, yy = np.meshgrid(xl, yl)
        rec = np.stack((xx.ravel(), yy.ravel()), axis=1)
        trx = cls.in_convex_poly(rec, quad)
        return wid, hgt, rec, trx, (np.array([minx, miny, maxx, maxy]), (scale-1)/(maxx-minx))
