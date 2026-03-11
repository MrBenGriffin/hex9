# Part of the Hex9 (H9) Project
# Copyright ©2025, Ben Griffin
# Licensed under the Apache License, Version 2.0

"""
This is 'n_oct' flattened octahedron net xy
"""

import numpy as np
from numpy.typing import NDArray
from hhg9.base.composite import CompositeDomain, ComponentDomain
from hhg9.base.point_format import PointFormat
from hhg9.projections import BaryNet
from hhg9.domains.nets import net_layouts
from hhg9.h9 import H9K, H9P, H9O
from hhg9.algorithms.geometry import inside_convex_polygon_cw


class OctantNet(ComponentDomain):
    """
    This a 2D side of an Octant that belongs to a Net.
    Validity should be easy enough since we have the 3 points that define it.
    """
    def __init__(self, registrar, dom, name: str, oid: int, mode: int):
        super().__init__(registrar, name, dom, oid, 2)
        self.mode = mode  # need to over-ride here!

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
    from hhg9 import Points

    R3 = H9K.R3
    GW = H9K.lattice.U * 3  # grid unit width U = H9GC.U H9GC.W/6
    GH = H9K.lattice.V * 3  # grid unit height
    RT = np.pi / 3.      # grid rotation in 60º

    def __init__(self, registrar, *, layout='mortar'):
        c_oct = registrar.domain('c_oct')
        b_oct = registrar.domain('b_oct')
        if layout not in net_layouts:
            layout = 'mortar'
        super().__init__(registrar, f'n_oct:{layout}', 2)
        tp = H9P.sv  # mode vertices
        self.face_polys = {}
        self.c_oct = c_oct
        self.b_oct = b_oct
        self.tri_w = H9K.derived.W / H9K.lattice.U
        self.tri_h = H9K.derived.H / H9K.lattice.V
        self.layout = net_layouts[layout]
        vals = list(self.layout['grid'].values())

        self.wi = self.layout['width'] * H9K.lattice.U
        self.he = self.layout['height'] * H9K.lattice.V

        self.oid_mo = np.zeros((8,), dtype=np.uint8)
        for sign, val in self.layout['grid'].items():
            c2 = None
            oid = H9O.cmp_oid[sign]
            side = H9O.oid_str[oid]
            bary = b_oct.sides[side]
            if isinstance(val, list) and isinstance(val[0], tuple):
                c2 = val[1:]
                val = val[0]
            gx, gy, th = val
            x_off = gx * self.GW
            y_off = gy * self.GH
            n_theta = (th % 6) * self.RT
            mode = int(bary.mode + th) % 2
            n_sig = f'{self.name}:{side}'
            b_sig = f'{b_oct.name}:{side}'
            self.oid_mo[oid] = mode
            self.sides[side] = OctantNet(registrar, self, n_sig, oid, mode)
            face = BaryNet(registrar, side, b_sig, n_sig, n_theta, (x_off, y_off))
            self.projs[side] = face
            if c2 is None:
                tri = H9P.sv[bary.mode]  # triangle from H9P. Use bary.mo b/c will transform!
                tri_rt = tri @ face.matrix + face.offset  # bary->net
                # Map sign→triangle and sign→side for fast lookup
                self.face_polys[sign] = [tri_rt]
            else:  # if c2 is not None:
                c2x = []
                polys = []
                for (x, y, t) in c2:  # calculate adjusted mode, theta units, x offset, y offset, theta-radians.
                    m = int(mode + t) % 2
                    t = int(t % 6)
                    ox = x * self.GW
                    oy = y * self.GH
                    r = t * self.RT
                    c2x.append((m, t, ox, oy, r))
                face.set_c2trans(c2x)
                for c2 in [0, 1, 2]:  # calculate adjusted mode, theta units, x offset, y offset, theta-radians.
                    matr, off = face.c2_affine(c2)
                    hh = H9P.hh[bary.mode, c2]
                    polys.append((hh @ matr) + off + face.offset)
                self.face_polys[sign] = polys


    def ratio(self) -> float:
        """Return width/height ratio. Use NetPixel.ratio() for new code."""
        return self.wi / self.he

    def img_adj(self) -> tuple:
        """Pixel trim adjustment. Use NetPixel.img_adj() for new code."""
        l_width = self.layout['width'] / self.tri_w
        l_height = self.layout['height'] / self.tri_h
        return l_width + 0.51, l_height + 0.51

    def image_dims(self, pixels: int) -> tuple[int, int]:
        """Triangle side in pixels → image (W, H). Use NetPixel.image_dims() for new code."""
        tri_h = pixels * self.R3 * 0.5
        l_width = self.layout['width'] / self.tri_w
        l_height = self.layout['height'] / self.tri_h
        w_a, h_a = self.img_adj()
        return int(l_width * pixels - w_a), int(l_height * tri_h - h_a)

    def dim_from_image(self, pix_w: int, pix_h: int) -> float:
        """Image (W, H) → triangle side in pixels. Use NetPixel.dim_from_image() for new code."""
        tri_h = self.R3 * 0.5
        l_width = self.layout['width'] / self.tri_w
        l_height = self.layout['height'] / self.tri_h
        w_a, h_a = self.img_adj()
        img_w = (float(pix_w) + w_a) / l_width
        img_h = (float(pix_h) + h_a) / (l_height * tri_h)
        return np.rint((img_w + img_h) / 2.0)

    def filter(self, pts):
        """
        Test that points are valid
        """
        from hhg9 import Points
        if not isinstance(pts, Points):
            raise TypeError('pts must be Points')
        if pts.domain != self:
            raise ValueError('pts must be in this domain')
        signs = self.pt_face(pts.coords)
        good = np.any(np.all(signs[:, None] == H9O.oid_cmp, axis=2), axis=1)
        result = self.Points(pts.coords[good], domain=self, components=signs[good])
        if pts.samples is not None:
            result.samples = pts.samples[good]
        return result

    def valid(self, pts: NDArray, return_signs=False) -> NDArray:
        """
        Test that points are valid
        """
        if pts.shape[-1] < 2:
            raise ValueError('Points must have 2 dimensions')
        signs = self.pt_face(pts)
        return np.any(signs != 0, axis=1)

    def pt_face(self, pts: NDArray) -> NDArray:
        """Vectorised: identify octant sign for each point in net coordinates.
        Returns (hex_layer,3) int8 array of signs (±1), or (0,0,0) for invalid.
        """
        num_points = pts.shape[0]
        out = np.zeros((num_points, 3), dtype=np.int8)
        for sign, polys in self.face_polys.items():
            for poly in polys:
                mask = inside_convex_polygon_cw(pts, poly)
                if not np.any(mask):
                    continue
                out[mask] = np.array(sign, dtype=np.int8)
        return out

    def binning(self, pts: Points, sig: tuple = None):
        """Identify the components of the points"""
        cmp = self.pt_face(pts.coords)
        pts.components = np.array(cmp)

    def register_format(self, af: PointFormat):
        """Decorator to register an AddressFormat for each component."""
        for side in self.sides:
            self.sides[side].register_format(af)
