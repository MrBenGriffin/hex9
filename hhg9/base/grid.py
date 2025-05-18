import numpy as np

"""
Triangular Hierarchic Grid Generation Class
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
    def px_grid(cls, xy=(0, 0), scale: float = 1000, ud: str = 'Λ'):
        """Return a pixel grid of points within an equilateral triangle centered at (cx, cy) in pixel space"""
        side_length = np.sqrt(2) * scale
        height = np.sqrt(3) / 2 * side_length
        cx, cy = xy
        cx *= scale
        cy *= scale
        # This seems... odd.
        dy = -2 / 3 * height if ud == 'Λ' else 2 / 3 * height - height
        oy = cy + dy
        ox = cx - side_length / 2

        # Triangle vertices in real coordinates
        if ud == 'Λ':
            v0 = np.array([ox + side_length / 2, oy])  # apex
            v1 = np.array([ox, oy + height])  # bottom-left
            v2 = np.array([ox + side_length, oy + height])  # bottom-right
        else:
            v0 = np.array([ox + side_length / 2, oy + height])  # apex (down)
            v1 = np.array([ox, oy])  # top-left
            v2 = np.array([ox + side_length, oy])  # top-right

        # Bounding box in integer pixel space
        x_min = int(np.floor(min(v0[0], v1[0], v2[0])))
        x_max = int(np.ceil(max(v0[0], v1[0], v2[0])))
        y_min = int(np.floor(min(v0[1], v1[1], v2[1])))
        y_max = int(np.ceil(max(v0[1], v1[1], v2[1])))

        # Precompute triangle area
        def edge(a, b, p):
            return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

        pixels = []
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                p = np.array([x + 0.5, y + 0.5])  # pixel center
                w0 = edge(v1, v2, p)
                w1 = edge(v2, v0, p)
                w2 = edge(v0, v1, p)
                if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                    pixels.append((x, y))

        return np.array(pixels)
